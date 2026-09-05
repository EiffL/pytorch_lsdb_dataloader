"""Worker-local iteration and optional TorchData checkpoint hooks."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from lsdb_torch._compute import compute_pixel
from lsdb_torch._rows import ColumnarPartition, copy_row
from lsdb_torch._sharding import partition_indices

if TYPE_CHECKING:
    from lsdb_torch.dataset import LSDBDataset


class CatalogIterator:
    """A cursor over a fixed worker assignment; I/O is separate from progress.

    Checkpoints contain row references, not Arrow partitions or decoded samples.
    Only the current partition and partitions represented in the shuffle buffer
    need to be read again. TorchData handles batches prefetched by DataLoader.
    """

    def __init__(self, dataset: LSDBDataset, epoch: int, worker_id: int, num_workers: int):
        self.dataset = dataset
        self.epoch = epoch
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.cycle = 0
        self.partition_position = 0
        self.row_position = 0
        self.cycle_has_rows = False
        self.source_exhausted = False
        self._buffer = []
        self._restore_refs = None
        self._partition = None
        self._row_order = None
        self._future = None
        self._executor = None
        self._closed = False
        self._reset_order()

    def _reset_order(self):
        ds = self.dataset
        self._owned = partition_indices(
            len(ds._pixels),
            ds.seed,
            self.epoch,
            ds.shuffle,
            ds.rank,
            ds.world_size,
            self.worker_id,
            self.num_workers,
            ds.partition_weights,
        )
        self._rng = np.random.default_rng([ds.seed, self.epoch, ds.rank, self.worker_id])
        self._set_cycle_order()

    def _set_cycle_order(self):
        ds = self.dataset
        self._order = self._owned
        if ds.shuffle and self.cycle:
            rng = np.random.default_rng([ds.seed, self.epoch, self.cycle, ds.rank, self.worker_id])
            self._order = rng.permutation(self._owned)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            if self._restore_refs is not None:
                self._restore_buffer()
            size = self.dataset.shuffle_buffer_size
            if size:
                while len(self._buffer) < size and not self.source_exhausted:
                    item = self._next_source()
                    if item is not None:
                        ref, row = item
                        self._buffer.append((ref, copy_row(row)))
                        del item, row
                item = self._next_source()
                if not self._buffer:
                    raise StopIteration
                j = int(self._rng.integers(len(self._buffer)))
                _, row = self._buffer[j]
                if item is None:
                    self._buffer[j] = self._buffer[-1]
                    self._buffer.pop()
                else:
                    ref, incoming = item
                    self._buffer[j] = (ref, copy_row(incoming))
            else:
                item = self._next_source()
                if item is None:
                    raise StopIteration
                _, row = item
            if self.dataset.transform is None:
                return row
            try:
                return self.dataset.transform(row)
            except StopIteration as exc:
                raise RuntimeError("LSDBDataset transform raised StopIteration") from exc
        except BaseException:
            self.close()
            raise

    def _next_source(self):
        ds = self.dataset
        while not self.source_exhausted:
            if self.partition_position == len(self._order):
                if not ds.loop:
                    self.source_exhausted = True
                    return None
                if not self.cycle_has_rows:
                    raise RuntimeError(
                        f"Cannot loop over an empty shard on rank {ds.rank}, worker {self.worker_id}; "
                        "use fewer ranks/workers or a catalog with rows in every assigned shard"
                    )
                self.cycle += 1
                self.partition_position = 0
                self.cycle_has_rows = False
                self._set_cycle_order()
            i = int(self._order[self.partition_position])
            if self._partition is None:
                if self._future is None:
                    self._partition = self._load(i)
                else:
                    self._partition = self._future.result()
                    self._future = None
                if ds.prefetch and self.partition_position + 1 < len(self._order):
                    if self._executor is None:
                        # fsspec caches instances per thread: reuse this thread
                        # across partitions and local passes.
                        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lsdb_torch-prefetch")
                    self._future = self._executor.submit(self._load, int(self._order[self.partition_position + 1]))
                self.cycle_has_rows |= bool(len(self._partition))
                self._row_order = range(len(self._partition))
                if ds.shuffle:
                    rng = np.random.default_rng([ds.seed, self.epoch, self.cycle, i])
                    self._row_order = rng.permutation(len(self._partition))
            if self.row_position < len(self._partition):
                row_index = int(self._row_order[self.row_position])
                self.row_position += 1
                return (i, row_index), self._partition.row(row_index)
            self._partition = None
            self._row_order = None
            self.partition_position += 1
            self.row_position = 0
        return None

    def _load(self, i):
        pixel = self.dataset._pixels[i]
        try:
            return ColumnarPartition(compute_pixel(self.dataset.catalog, pixel))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LSDB pixel {pixel} on rank {self.dataset.rank}, worker {self.worker_id}"
            ) from exc

    def _configuration(self):
        ds = self.dataset
        return (
            ds._layout_digest,
            ds.seed,
            ds.shuffle,
            ds.loop,
            ds.shuffle_buffer_size,
            ds.rank,
            ds.world_size,
            self.worker_id,
            self.num_workers,
        )

    def state_dict(self):
        """Snapshot progress for StatefulDataLoader (or direct iterator use)."""
        if self._closed and not self.source_exhausted:
            raise RuntimeError("Cannot checkpoint an iterator closed before exhaustion")
        transform = self.dataset.transform
        return {
            "version": 1,
            "configuration": self._configuration(),
            "epoch": self.epoch,
            "cycle": self.cycle,
            "partition_position": self.partition_position,
            "row_position": self.row_position,
            "cycle_has_rows": self.cycle_has_rows,
            "source_exhausted": self.source_exhausted,
            "buffer_refs": (
                list(self._restore_refs) if self._restore_refs is not None else [ref for ref, _ in self._buffer]
            ),
            "rng": deepcopy(self._rng.bit_generator.state),
            "transform": (deepcopy(transform.state_dict()) if hasattr(transform, "state_dict") else None),
        }

    def load_state_dict(self, state):
        """Restore against the same immutable catalog, transform and topology."""
        if state["version"] != 1 or state["configuration"] != self._configuration():
            raise ValueError(
                "Checkpoint does not match the catalog pixel layout, dataset options, or worker/rank topology"
            )
        if state["transform"] is not None and not hasattr(self.dataset.transform, "load_state_dict"):
            raise ValueError("Checkpoint requires a transform with load_state_dict()")
        self.close()
        self.epoch = state["epoch"]
        self.cycle = state["cycle"]
        self._reset_order()
        self.partition_position = state["partition_position"]
        self.row_position = state["row_position"]
        self.cycle_has_rows = state["cycle_has_rows"]
        self.source_exhausted = state["source_exhausted"]
        self._restore_refs = list(state["buffer_refs"])
        self._rng.bit_generator.state = deepcopy(state["rng"])
        if state["transform"] is not None:
            self.dataset.transform.load_state_dict(deepcopy(state["transform"]))
        self._closed = False

    def _restore_buffer(self):
        grouped = defaultdict(list)
        for position, (i, row_index) in enumerate(self._restore_refs):
            grouped[i].append((position, row_index))
        self._buffer = [None] * len(self._restore_refs)
        for i, rows in grouped.items():
            partition = self._load(i)
            for position, row_index in rows:
                self._buffer[position] = (
                    (i, row_index),
                    copy_row(partition.row(row_index)),
                )
            del partition
        self._restore_refs = None

    def close(self):
        """Release rows and queued work. Already running storage reads may finish."""
        self._closed = True
        self._partition = None
        self._row_order = None
        if self._future is not None:
            self._future.cancel()
            self._future = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._buffer.clear()

    def __del__(self):
        self.close()
