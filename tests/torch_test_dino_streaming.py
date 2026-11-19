# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CUDA ownership and read-ahead checks for the faithful SSD prefetcher."""

import contextlib
from contextlib import closing
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


def load(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    path = Path(__file__).resolve().parents[1] / "benchs/e_means" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DinoBvecs = load("dino_data").DinoBvecs
_ssd = load("ssd_streaming")
SSDRowSource, SSDPrefetch = _ssd.SSDRowSource, _ssd.SSDPrefetch


class TestStaging(unittest.TestCase):
    def test_stage_files_copies_node_rows_and_falls_back(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        blobs = [np.random.RandomState(i).bytes(4 << 20) for i in range(2)]
        paths = [root / ("%d.bin" % i) for i in range(2)]  # 1,024 rows of 4,096 B each
        for path, blob in zip(paths, blobs):
            path.write_bytes(blob)
        copies = _ssd.stage_files(paths, 4096, [(1500, 1600), (1600, 1700)], str(root / "stage"))
        self.assertEqual([Path(c).stat().st_size for c in copies], [4 << 20] * 2)
        rows = slice(1 << 20, 3 << 20)  # second file's rows 476-676, widened to whole MiB
        self.assertEqual(Path(copies[1]).read_bytes()[rows], blobs[1][rows])
        self.assertEqual(Path(copies[0]).read_bytes(), bytes(4 << 20))
        sums = root / "SHA256SUMS"
        sums.write_text("%s  0.bin\n" % ("0" * 64))
        with self.assertWarns(UserWarning):
            self.assertIsNone(_ssd.stage_files(paths, 4096, [(0, 2048)], str(root / "bad"), sums=sums))
        self.assertEqual(list((root / "bad").iterdir()), [])
        with mock.patch.object(_ssd.shutil, "disk_usage", return_value=mock.Mock(free=0)):
            with self.assertWarns(UserWarning):
                self.assertIsNone(_ssd.stage_files(paths, 4096, [(0, 10)], str(root / "full")))
        with self.assertWarns(UserWarning):
            self.assertIsNone(_ssd.stage_files(paths, 4096, [(0, 10)], str(root / "cap"), max_bytes=1))
        self.assertEqual(list((root / "cap").iterdir()), [])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDinoStreaming(unittest.TestCase):
    def source(self, data):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        chunks = root / "chunked_base_10B"
        chunks.mkdir()
        split = len(data) // 3
        for name, rows in (("00", data[:split]), ("01", data[split:])):
            packed = np.empty((len(rows), data.shape[1] + 4), dtype=np.uint8)
            packed[:, :4] = np.frombuffer(data.shape[1].to_bytes(4, "little"), dtype=np.uint8)
            packed[:, 4:] = rows
            packed.tofile(chunks / (name + ".bvecs"))
        reader = DinoBvecs(root, n=len(data))
        source = SSDRowSource(reader.paths, reader.d, 0, reader.n, header_bytes=4)
        self.addCleanup(source.close)
        return source

    def loader(self, source, batch_size):
        return SSDPrefetch(
            source, "cuda", batch_size, n_prefetch=2,
            chunk_bytes=batch_size * source.row_bytes,
            ram_budget_bytes=4 * batch_size * source.d,
            read_threads=1, extract_threads=1,
        )

    def test_nondefault_stream_order_dtype_and_logical_tails(self):
        data = np.random.RandomState(8).randint(0, 256, (37, 8), dtype=np.uint8)
        source = self.source(data)
        consumer = torch.cuda.Stream()
        mean = torch.from_numpy(np.arange(8, dtype=np.float32)).to("cuda")
        consumer.wait_stream(torch.cuda.current_stream())
        for sequential in (False, True):
            loader = self.loader(source, 10)
            order = loader.order(9)
            outputs = []
            with torch.cuda.stream(consumer):
                with closing(loader.batches(
                    order, sequential=sequential,
                    mean=mean if sequential else None,
                    scale=2.0 if sequential else None,
                )) as iterator:
                    for rows in iterator:
                        self.assertEqual(rows.dtype, torch.float32)
                        outputs.append(rows.clone())
            consumer.synchronize()
            self.assertEqual([len(x) for x in outputs], [10, 10, 10, 7])
            expected = torch.from_numpy(data if sequential else data[order]).float()
            if sequential:
                expected = (expected - mean.cpu()) / 2.0
            torch.testing.assert_close(torch.cat(outputs).cpu(), expected, rtol=0, atol=0)

    def test_shifted_grid_permutes_and_fills_requested_rows(self):
        data = np.random.RandomState(3).randint(0, 256, (37, 8), dtype=np.uint8)
        source = self.source(data)
        loader = self.loader(source, 10)
        for groups in (4, 0):
            order = _ssd.chunk_grouped_order(
                source.n, loader.rows_per_chunk, groups, loader.window_rows, 9, 3,
            ).view(_ssd.PassOrder)
            order.shift = 7  # grid offset 3 rows
            np.testing.assert_array_equal(np.sort(order), np.arange(source.n))
            with closing(loader.batches(order)) as iterator:
                outputs = [rows.clone() for rows in iterator]
            torch.cuda.synchronize()
            torch.testing.assert_close(
                torch.cat(outputs).cpu(), torch.from_numpy(data[order]).float(), rtol=0, atol=0,
            )

    def test_next_pass_prefetch_keeps_rows(self):
        data = np.random.RandomState(5).randint(0, 256, (37, 8), dtype=np.uint8)
        source = self.source(data)
        loader = self.loader(source, 10)
        orders = [loader.order(seed) for seed in (1, 2, 3)]
        outputs = []
        for p, order in enumerate(orders):
            following = orders[p + 1] if p + 1 < len(orders) else None
            with closing(loader.batches(order, next_order=following)) as iterator:
                outputs.append(torch.cat([rows.clone() for rows in iterator]))
        torch.cuda.synchronize()
        for order, rows in zip(orders, outputs):
            torch.testing.assert_close(rows.cpu(), torch.from_numpy(data[order]).float(), rtol=0, atol=0)

    def test_pipelined_stage_keeps_rows_and_falls_back(self):
        data = np.random.RandomState(6).randint(0, 256, (37, 8), dtype=np.uint8)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        packed = np.empty((len(data), 12), dtype=np.uint8)
        packed[:, :4] = np.frombuffer((8).to_bytes(4, "little"), dtype=np.uint8)
        packed[:, 4:] = data
        originals = [root / "00.bvecs", root / "01.bvecs"]
        packed[:12].tofile(originals[0])
        packed[12:].tofile(originals[1])
        for corrupt in (False, True):
            copies = _ssd.prepare_stage(originals, 12, [(0, len(data))], str(root / "stage"))
            source = SSDRowSource(copies, 8, 0, len(data), header_bytes=4)
            self.addCleanup(source.close)
            loader = self.loader(source, 10)
            orders = [loader.order(seed) for seed in (1, 2)]
            write = _ssd.os.pwrite
            with mock.patch.object(_ssd.os, "pwrite", side_effect=(
                    lambda fd, block, at: write(fd, bytes(len(block)), at)) if corrupt else write):
                stage = loader.stage = _ssd.PipelinedStage(
                    originals, copies, source, orders[0], loader.window_rows,
                    loader.rows_per_chunk, getattr(orders[0], "shift", 0), threads=2,
                )
                outputs = []
                with self.assertWarns(UserWarning) if corrupt else contextlib.nullcontext():
                    for p, order in enumerate(orders):
                        following = orders[p + 1] if p + 1 < len(orders) else None
                        with closing(loader.batches(order, next_order=following)) as iterator:
                            outputs.append(torch.cat([rows.clone() for rows in iterator]))
                    torch.cuda.synchronize()
                stage.close()
            self.assertEqual(stage.failed is not None, corrupt)
            for order, rows in zip(orders, outputs):
                torch.testing.assert_close(rows.cpu(), torch.from_numpy(data[order]).float(), rtol=0, atol=0)

    def test_early_close_preserves_owned_output(self):
        data = np.arange(37 * 8, dtype=np.uint16).astype(np.uint8).reshape(37, 8)
        source = self.source(data)
        loader = self.loader(source, 10)
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(consumer):
            iterator = loader.batches(sequential=True)
            first = next(iterator)
            copied = first.clone()
            iterator.close()
            retained = first.clone()
        consumer.synchronize()
        torch.testing.assert_close(copied.cpu(), torch.from_numpy(data[:10]).float())
        torch.testing.assert_close(retained, copied)
        np.testing.assert_array_equal(source.read(10, 14), data[10:14])

    def test_fractional_pass_stops_reads_and_keeps_batch_tail(self):
        data = np.arange(37 * 8, dtype=np.uint16).astype(np.uint8).reshape(37, 8)
        source = self.source(data)
        for count in (0, 7, 23, 37):
            with self.subTest(rows=count):
                loader = self.loader(source, 10)
                order = loader.order(9)[:count]
                requests = []
                original = loader.fill_window

                def record(request, out):
                    requests.append(request.copy())
                    original(request, out)

                loader.fill_window = record
                with closing(loader.batches(order)) as iterator:
                    outputs = [rows.clone() for rows in iterator]
                torch.cuda.synchronize()
                self.assertEqual(
                    [len(rows) for rows in outputs],
                    [min(10, count - start) for start in range(0, count, 10)],
                )
                if count:
                    np.testing.assert_array_equal(
                        np.concatenate(requests), order
                    )
                    torch.testing.assert_close(
                        torch.cat(outputs).cpu(),
                        torch.from_numpy(data[order]).float(), rtol=0, atol=0,
                    )
                else:
                    self.assertEqual(requests, [])
                    self.assertEqual(loader.stats["bytes_read"], 0)

    def test_read_ahead_exception_propagates(self):
        data = np.zeros((37, 8), dtype=np.uint8)
        source = self.source(data)
        loader = self.loader(source, 10)
        original = source.read_views

        def fail(begin, end, buffer):
            if begin >= loader.batch_size:
                raise OSError("injected read failure")
            return original(begin, end, buffer)

        source.read_views = fail
        with self.assertRaisesRegex(OSError, "injected read failure"):
            with closing(loader.batches(sequential=True)) as iterator:
                for rows in iterator:
                    rows.sum()


if __name__ == "__main__":
    unittest.main()
