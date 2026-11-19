# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SSD scans into shuffled host windows and asynchronous CUDA batches.

Adapts the window scans and chunk-group permutation from the reference
ssd_prefetch.py and data.py. Each rank reads its own contiguous row shard.
stage_files() copies the rows a node reads once to a node-local directory
and verifies them; each pass can shift the scan-chunk grid.
"""

from concurrent.futures import ThreadPoolExecutor, wait
import errno
import glob
import hashlib
import math
import mmap
import os
import queue
import shutil
import socket
import tempfile
import time
import warnings

import numpy as np
import torch


class SSDRowSource:
    """A rank's range in fixed-width uint8 files, optionally with row headers.

    ``paths`` are ordered corpus chunks. O_DIRECT uses page-aligned buffers
    and reads; filesystems that refuse it fall back to buffered preadv.
    """

    def __init__(self, paths, d, begin, end, header_bytes=0, direct=True):
        if d < 1 or header_bytes < 0 or not 0 <= begin < end:
            raise ValueError("invalid SSD row layout or range")
        self.d, self.header_bytes = int(d), int(header_bytes)
        self.row_bytes = self.d + self.header_bytes
        self.begin, self.n = int(begin), int(end - begin)
        self.fds = []
        self.direct = []
        lengths = []
        try:
            for path in paths:
                size = os.path.getsize(path)
                if size % self.row_bytes:
                    raise ValueError(f"incomplete row in {path}")
                lengths.append(size // self.row_bytes)
                try:
                    fd = os.open(
                        path, os.O_RDONLY | (os.O_DIRECT if direct else 0)
                    )
                    is_direct = bool(direct)
                except OSError as error:
                    if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                        raise
                    fd, is_direct = os.open(path, os.O_RDONLY), False
                self.fds.append(fd)
                self.direct.append(is_direct)
            self.ends = np.cumsum(lengths)
            if not lengths or end > self.ends[-1]:
                raise ValueError("SSD shard exceeds file rows")
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()

    def read(self, begin, end):
        """Read a sequential local range into an owned compact uint8 array."""
        if not 0 <= begin <= end <= self.n:
            raise IndexError("SSD row range out of bounds")
        out = np.empty((end - begin, self.d), dtype=np.uint8)
        a, b, at = self.begin + begin, self.begin + end, 0
        for fd, hi, lo in zip(self.fds, self.ends, np.r_[0, self.ends[:-1]]):
            start, stop = max(a, int(lo)), min(b, int(hi))
            if start >= stop:
                continue
            offset = (start - lo) * self.row_bytes
            length = (stop - start) * self.row_bytes
            aligned = int(offset) // mmap.PAGESIZE * mmap.PAGESIZE
            slack = int(offset) - aligned
            size = math.ceil((slack + length) / mmap.PAGESIZE) * mmap.PAGESIZE
            with mmap.mmap(-1, size) as buffer:
                got = os.preadv(fd, [buffer], aligned)
                if got < slack + length:
                    raise OSError("short SSD read")
                rows = np.ndarray(
                    (stop - start, self.row_bytes),
                    dtype=np.uint8,
                    buffer=buffer,
                    offset=slack,
                )
                out[at : at + stop - start] = rows[:, self.header_bytes :]
                del rows
            at += stop - start
        return out

    def read_views(self, begin, end, buffer):
        """Read into a reusable aligned buffer and return borrowed payload views.

        Views retain bvecs row strides and skip headers without copying. Keep
        ``buffer`` alive and unchanged until all views have been consumed.
        A range crossing files returns one view per file, with local row offsets.
        """
        if not 0 <= begin <= end <= self.n:
            raise IndexError("SSD row range out of bounds")
        a, b, cursor = self.begin + begin, self.begin + end, 0
        views = []
        for fd, hi, lo in zip(self.fds, self.ends, np.r_[0, self.ends[:-1]]):
            start, stop = max(a, int(lo)), min(b, int(hi))
            if start >= stop:
                continue
            offset = (start - int(lo)) * self.row_bytes
            length = (stop - start) * self.row_bytes
            aligned = offset // mmap.PAGESIZE * mmap.PAGESIZE
            slack = offset - aligned
            size = math.ceil((slack + length) / mmap.PAGESIZE) * mmap.PAGESIZE
            if cursor + size > len(buffer):
                raise ValueError("SSD read buffer is too small")
            target = memoryview(buffer)[cursor : cursor + size]
            try:
                got = os.preadv(fd, [target], aligned)
            finally:
                target.release()
            if got < slack + length:
                raise OSError("short SSD read")
            rows = np.ndarray(
                (stop - start, self.row_bytes), dtype=np.uint8,
                buffer=buffer, offset=cursor + slack,
            )
            views.append((start - self.begin, rows[:, self.header_bytes :]))
            cursor += size
        return views

    def gather(self, request, out, depth=256):
        """Queue small aligned reads of random rows into a host array.

        Rows sharing a starting block use one read. Payload offsets include
        bvecs headers, and reads span two blocks when a payload straddles one.
        Buffers and the ring are bounded by depth; output order is preserved.
        """
        if __package__:
            from . import ctypes_liburing as uring
        else:
            import ctypes_liburing as uring

        request = np.asarray(request, dtype=np.int64)
        if request.ndim != 1 or (
            len(request) and (request.min() < 0 or request.max() >= self.n)
        ):
            raise IndexError("random request outside SSD shard")
        if depth < 1 or depth & (depth - 1):
            raise ValueError("I/O depth must be a positive power of two")
        destination = out.numpy() if isinstance(out, torch.Tensor) else out
        if destination.shape != (len(request), self.d):
            raise ValueError("random read output has the wrong shape")
        stats = {"bytes_read": 0, "requests": 0}
        if not len(request):
            return stats
        block = 4096
        slot_bytes = math.ceil((self.d + block - 1) / block) * block
        ring = uring.io_uring()
        buffers, views = [], []
        try:
            uring.io_uring_queue_init(depth, ring, 0)
            buffers = [mmap.mmap(-1, slot_bytes) for _ in range(depth)]
            views = [
                np.frombuffer(buffer, dtype=np.uint8) for buffer in buffers
            ]
            cqe = uring.io_uring_cqe()
            columns = np.arange(self.d)
            absolute = self.begin + request
            for fd, hi, lo in zip(
                self.fds, self.ends, np.r_[0, self.ends[:-1]]
            ):
                ids = np.flatnonzero((absolute >= lo) & (absolute < hi))
                if not len(ids):
                    continue
                starts = (
                    absolute[ids] - lo
                ) * self.row_bytes + self.header_bytes
                perm = np.argsort(starts, kind="stable")
                ids, starts = ids[perm], starts[perm]
                blocks = starts // block
                bounds = np.r_[
                    0, np.flatnonzero(np.diff(blocks)) + 1, len(ids)
                ]
                for first in range(0, len(bounds) - 1, depth):
                    pending = []
                    for j in range(first, min(first + depth, len(bounds) - 1)):
                        a, b = int(bounds[j]), int(bounds[j + 1])
                        offset = int(blocks[a]) * block
                        needed = int(starts[b - 1]) + self.d - offset
                        length = math.ceil(needed / block) * block
                        slot = len(pending)
                        sqe = uring.io_uring_get_sqe(ring)
                        uring.io_uring_prep_read(
                            sqe, fd, buffers[slot], length, offset
                        )
                        sqe.user_data = slot
                        pending.append((a, b, offset, needed))
                    uring.io_uring_submit_and_wait(ring, len(pending))
                    error = None
                    for _ in pending:
                        if uring.io_uring_peek_cqe(ring, cqe):
                            raise OSError("missing random read completion")
                        a, b, offset, needed = pending[cqe.user_data]
                        if cqe.res < 0:
                            error = OSError(-cqe.res, "random SSD read failed")
                        elif cqe.res < needed:
                            error = OSError("short random SSD read")
                        else:
                            destination[ids[a:b]] = views[cqe.user_data][
                                (starts[a:b] - offset)[:, None] + columns
                            ]
                            stats["bytes_read"] += cqe.res
                            stats["requests"] += 1
                        uring.io_uring_cqe_seen(ring, cqe)
                    if error is not None:
                        raise error
        finally:
            uring.io_uring_queue_exit(ring)
            views.clear()
            for buffer in buffers:
                buffer.close()
        return stats


def stage_files(paths, row_bytes, row_ranges, directory, sums=None, max_bytes=None, threads=16, piece=1 << 30):
    """Copy the rows a node reads to a local directory; return the copies or None.

    ``row_ranges`` are [begin, end) rows of the concatenated files. Copies
    keep each file's size (parts not read stay sparse), so offsets do not
    change. A copy is checked against ``sums`` (SHA256SUMS lines) when a
    whole listed file is staged, else by its first, middle and last 16 MiB
    per range; failed ranges are copied once more. If the copy needs more
    than ``max_bytes`` or the free space, or an I/O error occurs, or a copy
    stays wrong, this warns, leaves nothing behind and returns None: read
    the originals. Copies left on this host by exited
    processes are removed first; the caller removes the returned directory.
    """
    merged = []
    for lo, hi in sorted(row_ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    spans, first = [], 0
    for path in paths:
        size = os.path.getsize(path)
        last = first + size // row_bytes
        for lo, hi in merged:  # whole MiB around the rows, for aligned reads at the edges
            if max(lo, first) < min(hi, last):
                begin = (max(lo, first) - first) * row_bytes >> 20 << 20
                spans.append((path, begin, min(size, -(-(min(hi, last) - first) * row_bytes >> 20) << 20)))
        first = last
    expected = {}
    if sums is not None and os.path.exists(sums):
        with open(sums) as f:
            expected = {os.path.basename(n): h for h, n in (line.split() for line in f if line.strip())}

    def copy(path, begin, end):
        with open(path, "rb", buffering=0) as fin, open(staged[path], "r+b") as fout:
            fin.seek(begin)
            fout.seek(begin)
            while begin < end:
                block = fin.read(min(end - begin, 16 << 20))
                if not block:
                    raise OSError(errno.EIO, "short read while staging", path)
                fout.write(block)
                begin += len(block)

    def verify(path, begin, end):
        name = os.path.basename(path)
        if name in expected and (begin, end) == (0, os.path.getsize(path)):
            digest = hashlib.sha256()
            with open(staged[path], "rb") as f:
                for block in iter(lambda: f.read(16 << 20), b""):
                    digest.update(block)
            return digest.hexdigest() == expected[name]
        with open(path, "rb") as a, open(staged[path], "rb") as b:
            for at in (begin, (begin + end) // 2, max(begin, end - (16 << 20))):
                a.seek(at)
                b.seek(at)
                if a.read(min(16 << 20, end - at)) != b.read(min(16 << 20, end - at)):
                    return False
        return True

    # Copies are named by a short hash of the host name, so the sweep can tell
    # this host's copies apart without writing the host name to disk.
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:12]
    prefix, own = "ssd_stage.%s." % host, None
    try:
        os.makedirs(directory, exist_ok=True)
        for old in glob.glob(os.path.join(glob.escape(directory), glob.escape(prefix) + "*")):
            try:  # remove copies whose process has exited
                os.kill(int(os.path.basename(old)[len(prefix):].split(".")[0]), 0)
            except ProcessLookupError:
                shutil.rmtree(old, ignore_errors=True)
            except (ValueError, PermissionError):
                pass
        need = sum(end - begin for _, begin, end in spans)
        if need + (1 << 30) > shutil.disk_usage(directory).free or need > (max_bytes or need):
            raise OSError(errno.ENOSPC, "%.1f GB needed" % (need / 1e9), directory)
        own = tempfile.mkdtemp(prefix=prefix + "%d." % os.getpid(), dir=directory)
        staged = {p: os.path.join(own, os.path.basename(p)) for p in paths}
        for path in paths:
            with open(staged[path], "wb") as f:
                f.truncate(os.path.getsize(path))
        with ThreadPoolExecutor(threads) as pool:
            todo = spans
            for _ in range(2):
                pieces = [(p, at, min(end, at + piece)) for p, begin, end in todo for at in range(begin, end, piece)]
                list(pool.map(lambda job: copy(*job), pieces))
                todo = [span for span, ok in zip(todo, pool.map(lambda span: verify(*span), todo)) if not ok]
                if not todo:
                    return [staged[p] for p in paths]
        warnings.warn("staged copies in %s stayed wrong: reading the original files" % own)
    except (OSError, MemoryError) as error:
        warnings.warn("staging in %s failed (%s): reading the original files" % (directory, error))
    if own:
        shutil.rmtree(own, ignore_errors=True)
    return None


def chunk_grouped_order(n, rows_per_chunk, groups, window_rows, seed, offset=0):
    """Permute all rows once, grouping full windows by random scan chunks.

    Each full window draws from one of ``groups`` random subsets of scan
    chunks. Full windows are shuffled across groups; short group tails are
    placed last. Groups=1 gives a uniform row permutation. Groups=0 cuts the
    randomly ordered chunks into consecutive windows instead (window-aligned):
    each window reads its own chunks plus at most two shared with its
    neighbours, so a pass reads about 1x its rows instead of 2x, but rows of
    one scan chunk always share a window. ``offset`` shifts the scan-chunk
    grid: chunk c holds rows [c * C - s, (c + 1) * C - s) with
    s = (C - offset) % C, clipped to [0, n); offset 0 keeps the plain grid.
    """
    if min(n, rows_per_chunk, window_rows) < 1 or groups < 0:
        raise ValueError("permutation dimensions must be positive")
    if not 0 <= offset < rows_per_chunk:
        raise ValueError("grid offset must lie in [0, rows_per_chunk)")
    rng = np.random.RandomState(seed)
    if groups == 1:
        return rng.permutation(n)
    shift = (rows_per_chunk - offset) % rows_per_chunk
    count = math.ceil((n + shift) / rows_per_chunk)
    begin = np.maximum(0, np.arange(count) * rows_per_chunk - shift)
    end = np.minimum(n, np.arange(1, count + 1) * rows_per_chunk - shift)
    if groups == 0:
        chunks = rng.permutation(count)
        lengths = end[chunks] - begin[chunks]
        ends = np.cumsum(lengths)
        starts = begin[chunks] - ends + lengths  # row = q + starts[q's chunk]
        output = np.empty(n, dtype=np.int64)
        for at in range(0, n, window_rows):
            q = at + rng.permutation(min(window_rows, n - at))
            output[at:at + len(q)] = q + starts[np.searchsorted(ends, q, side="right")]
        return output
    pieces = np.array_split(rng.permutation(count), min(groups, count))
    layout, full, partial = [], [], []
    for j, chunks in enumerate(pieces):
        last = chunks == count - 1
        chunks = np.r_[chunks[~last], chunks[last]]
        size = int((end[chunks] - begin[chunks]).sum())
        layout.append((chunks, size))
        full.extend((j, k) for k in range(size // window_rows))
        if size % window_rows:
            partial.append((j, size // window_rows))
    positions, at = {}, 0
    for items in (full, partial):
        for index in rng.permutation(len(items)):
            j, k = items[index]
            length = min(window_rows, layout[j][1] - k * window_rows)
            positions[j, k] = (at, length)
            at += length
    output = np.empty(n, dtype=np.int64)
    for j, (chunks, size) in enumerate(layout):
        perm = rng.permutation(size)
        lengths = end[chunks] - begin[chunks]
        ends = np.cumsum(lengths)
        starts = begin[chunks] - ends + lengths  # row = q + starts[q's chunk]
        for k in range(math.ceil(size / window_rows)):
            dest, length = positions[j, k]
            for start in range(0, length, 1 << 20):
                q = perm[
                    k * window_rows
                    + start : k * window_rows
                    + min(length, start + (1 << 20))
                ]
                output[dest + start : dest + start + len(q)] = (
                    q + starts[np.searchsorted(ends, q, side="right")]
                )
    return output


def _plan_window(request, n_rows, rows_per_chunk, shift=0):
    """Group requested rows by chunk, retaining their output destinations.

    Adapted from the reference ssd_prefetch._plan_file. Permutation windows
    have distinct rows: a mask and inverse map replace the expensive sort.
    Duplicate requests retain the stable-sort path.
    """
    request = np.asarray(request, dtype=np.int64)
    if request.ndim != 1 or (
        len(request) and (request.min() < 0 or request.max() >= n_rows)
    ):
        raise IndexError("prefetch request outside SSD shard")
    count = len(request)
    rows = None
    if count:
        mask = np.zeros(n_rows, dtype=bool)
        mask[request] = True
        rows = np.flatnonzero(mask)
        del mask
        if len(rows) == count:
            inverse = np.empty(
                n_rows, dtype=np.int32 if count < 2**31 else np.int64
            )
            inverse[request] = np.arange(count, dtype=inverse.dtype)
            order = inverse[rows].astype(np.int64)
            del inverse
        else:
            rows = None
    if rows is None:
        order = np.argsort(request, kind="stable")
        rows = request[order]
    chunk_ids = (rows + shift) // rows_per_chunk
    first = (
        np.r_[0, np.flatnonzero(np.diff(chunk_ids)) + 1]
        if count else np.empty(0, dtype=np.int64)
    )
    chunks = chunk_ids[first]
    bounds = np.r_[first, count]
    return rows, order, chunks, bounds


class PassOrder(np.ndarray):
    """One pass's row order; slices keep the pass's scan-chunk grid shift."""

    shift = 0

    def __array_finalize__(self, obj):
        self.shift = getattr(obj, "shift", 0)


class SSDPrefetch:
    """Two host windows, chunk scans, and one CUDA batch copied ahead.

    Random requests retain their exact order, including duplicates. Reads
    within each window are issued in increasing chunk order; chunks with no
    requested row are skipped. A window is reused only after its asynchronous
    copies complete. Close the iterator before closing its source.
    """

    def __init__(
        self,
        source,
        device,
        batch_size,
        *,
        n_prefetch="auto",
        chunk_groups=4,
        chunk_bytes=64 << 20,
        ram_budget_bytes=16 << 30,
        read_threads=1,
        extract_threads=4,
        grid_offset=False,
    ):
        if (
            min(
                batch_size,
                chunk_bytes,
                ram_budget_bytes,
                read_threads,
                extract_threads,
            )
            < 1
            or chunk_groups < 0
        ):
            raise ValueError(
                "SSD prefetch sizes and thread counts must be positive"
            )
        self.source, self.device = source, torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SSD prefetch requires CUDA")
        self.batch_size, self.chunk_groups = batch_size, chunk_groups
        self.grid_offset = grid_offset
        self._slots = self._ahead = None
        self.rows_per_chunk = max(1, chunk_bytes // source.row_bytes)
        steps = math.ceil(source.n / batch_size)
        capacity = ram_budget_bytes // (2 * batch_size * source.d)
        if capacity < 1:
            raise ValueError("RAM budget must fit two host batches")
        self.n_prefetch = (
            min(steps, capacity) if n_prefetch == "auto" else int(n_prefetch)
        )
        if not 1 <= self.n_prefetch <= capacity:
            raise ValueError("prefetch windows exceed RAM budget")
        self.window_rows = self.n_prefetch * batch_size
        self.read_threads, self.extract_threads = read_threads, extract_threads
        self.copy_stream = torch.cuda.Stream(self.device)
        self.stats = {"bytes_read": 0, "wait_s": 0.0, "windows": 0}

    def order(self, seed):
        """Permutation for one pass; with grid_offset, the scan-chunk grid
        moves by a random number of rows drawn from its own stream, so rows
        that shared a scan chunk in one pass rarely share one in the next."""
        offset = 0
        if self.grid_offset:
            offset = int(np.random.RandomState(
                (int(seed) * 2654435761 + 0x5EED) % 2**32).randint(self.rows_per_chunk))
        result = chunk_grouped_order(
            self.source.n,
            self.rows_per_chunk,
            self.chunk_groups,
            self.window_rows,
            seed,
            offset,
        ).view(PassOrder)
        result.shift = (self.rows_per_chunk - offset) % self.rows_per_chunk
        return result

    def fill_window(self, request, out):
        planned = time.perf_counter()
        shift = getattr(request, "shift", 0)
        rows, order, chunks, bounds = _plan_window(
            request, self.source.n, self.rows_per_chunk, shift
        )
        self.stats["plan_s"] = (
            self.stats.get("plan_s", 0.0) + time.perf_counter() - planned
        )
        if not len(rows):
            return
        destination = out.numpy()

        def tasks():
            for chunk, lo, hi in zip(chunks, bounds[:-1], bounds[1:]):
                begin = max(0, int(chunk) * self.rows_per_chunk - shift)
                end = min(self.source.n, (int(chunk) + 1) * self.rows_per_chunk - shift)
                yield begin, end, int(lo), int(hi)

        def extract(task, views):
            _, _, lo, hi = task
            for begin, data in views:
                a, b = np.searchsorted(rows[lo:hi], [begin, begin + len(data)])
                a, b = lo + int(a), lo + int(b)
                destination[order[a:b]] = data[rows[a:b] - begin]

        scanned = time.perf_counter()
        self._scan_chunks(tasks(), extract)
        self.stats["scan_s"] = (
            self.stats.get("scan_s", 0.0) + time.perf_counter() - scanned
        )
        self.stats["windows"] += 1

    def _scan_chunks(self, tasks, extract):
        """Pipeline reads and extraction with buffers leased until copying ends."""
        slots = 2 * (self.read_threads + self.extract_threads)
        # Each file fragment needs up to two extra pages for aligned I/O.
        size = math.ceil(
            (self.rows_per_chunk * self.source.row_bytes
             + 2 * mmap.PAGESIZE * len(self.source.fds)) / mmap.PAGESIZE
        ) * mmap.PAGESIZE
        buffers = []
        free = queue.Queue()
        pending_reads, pending_extracts = [], []

        def read_chunk(task):
            slot = free.get()
            try:
                views = self.source.read_views(task[0], task[1], buffers[slot])
                return task, slot, views
            except BaseException:
                free.put(slot)
                raise

        def consume(item):
            task, slot, views = item
            try:
                extract(task, views)
            finally:
                views.clear()
                free.put(slot)

        def dispatch(future, extractors):
            item = future.result()
            self.stats["bytes_read"] += (
                (item[0][1] - item[0][0]) * self.source.row_bytes
            )
            pending_extracts.append(extractors.submit(consume, item))

        try:
            for slot in range(slots):
                buffers.append(mmap.mmap(-1, size))
                free.put(slot)
            with ThreadPoolExecutor(self.read_threads) as readers, ThreadPoolExecutor(
                self.extract_threads
            ) as extractors:
                for task in tasks:
                    pending_reads.append(readers.submit(read_chunk, task))
                    if len(pending_reads) >= 2 * self.read_threads:
                        dispatch(pending_reads.pop(0), extractors)
                    if len(pending_extracts) >= 2 * self.extract_threads:
                        pending_extracts.pop(0).result()
                for future in pending_reads:
                    dispatch(future, extractors)
                for future in pending_extracts:
                    future.result()
        finally:
            # Unconsumed completed reads can retain borrowed views on errors.
            for future in pending_reads:
                if future.done() and not future.cancelled() and future.exception() is None:
                    future.result()[2].clear()
            pending_reads.clear()
            pending_extracts.clear()
            for buffer in buffers:
                buffer.close()

    def fill_sequential(self, start, end, out):
        def tasks():
            for begin in range(start, end, self.rows_per_chunk):
                yield begin, min(end, begin + self.rows_per_chunk)

        def extract(task, views):
            for begin, data in views:
                out[begin - start : begin - start + len(data)].copy_(
                    torch.from_numpy(data)
                )

        self._scan_chunks(tasks(), extract)
        self.stats["windows"] += 1

    def batches(self, order=None, *, sequential=False, mean=None, scale=None, next_order=None):
        """Yield rank-local FP32 CUDA batches, optionally centered.

        Sequential mode performs one corpus scan without permutation or
        repeated sparse window scans, for centering and frozen closing means.
        One batch per host window keeps sequential reads from delaying compute.
        Random mode accepts a permutation prefix for a fractional pass. With
        ``next_order`` (an order or a future of one), the last window's turn
        also fills the next pass's first window into the free buffer; the
        next call with that same order object starts without waiting.
        """
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        if not sequential and (order is None or len(order) > self.source.n):
            raise ValueError(
                "random mode needs at most one request per source row"
            )
        n_rows = self.source.n if sequential else len(order)
        if n_rows == 0:
            return
        if sequential:
            window_rows = min(self.batch_size, n_rows)
            slots = [
                torch.empty((window_rows, self.source.d), dtype=torch.uint8, pin_memory=True)
                for _ in range(2)
            ]
            events, pool, ahead = [None, None], ThreadPoolExecutor(1), None
        else:  # random mode keeps its two windows and fill thread across passes
            window_rows = self.window_rows
            if self._slots is None:
                self._slots = [
                    torch.empty((window_rows, self.source.d), dtype=torch.uint8, pin_memory=True)
                    for _ in range(2)
                ]
                self._events, self._pool = [None, None], ThreadPoolExecutor(1)
            slots, events, pool = self._slots, self._events, self._pool
            ahead, self._ahead = self._ahead, None
            if ahead is not None and ahead[0] is not order:
                wait([ahead[2]])
                ahead = None
        base = 0 if ahead is None else ahead[1]
        state = {"future": None}

        def fill(rows_order, start, slot):
            if events[slot] is not None:
                events[slot].synchronize()
            end = min(n_rows if sequential else len(rows_order), start + window_rows)
            if sequential:
                self.fill_sequential(start, end, slots[slot][: end - start])
            else:
                self.fill_window(rows_order[start:end], slots[slot][: end - start])
            return end - start

        def copies():
            future = ahead[2] if ahead is not None else pool.submit(fill, order, 0, base)
            for window, start in enumerate(
                range(0, n_rows, window_rows)
            ):
                slot = (base + window) % 2
                state["future"] = future
                waited = time.perf_counter()
                length = future.result()
                self.stats["wait_s"] += time.perf_counter() - waited
                future = None
                if start + window_rows < n_rows:
                    future = pool.submit(fill, order, start + window_rows, 1 - slot)
                elif next_order is not None:
                    following = getattr(next_order, "result", lambda: next_order)()
                    future = pool.submit(fill, following, 0, 1 - slot)
                    self._ahead = (following, 1 - slot, future)
                state["future"] = future
                for a in range(0, length, self.batch_size):
                    with torch.cuda.stream(self.copy_stream):
                        batch = (
                            slots[slot][a : min(length, a + self.batch_size)]
                            .to(self.device, non_blocking=True)
                            .float()
                        )
                        if mean is not None:
                            batch.sub_(mean)
                        if scale is not None:
                            batch.div_(scale)
                        done = torch.cuda.Event()
                        done.record(self.copy_stream)
                    events[slot] = done
                    yield batch, done

        iterator = copies()
        pending = None
        try:
            for batch, ready in iterator:
                if pending is not None:
                    previous, event = pending
                    torch.cuda.current_stream(self.device).wait_event(event)
                    previous.record_stream(
                        torch.cuda.current_stream(self.device)
                    )
                    yield previous
                pending = batch, ready
            if pending is not None:
                batch, ready = pending
                torch.cuda.current_stream(self.device).wait_event(ready)
                batch.record_stream(torch.cuda.current_stream(self.device))
                yield batch
        finally:
            iterator.close()
            if sequential:
                pool.shutdown(wait=True, cancel_futures=True)
            elif state["future"] is not None and (
                self._ahead is None or state["future"] is not self._ahead[2]
            ):
                wait([state["future"]])  # a pass closed early: settle its fill
            self.copy_stream.synchronize()
