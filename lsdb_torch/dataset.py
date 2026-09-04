"""An IterableDataset streaming rows of an lsdb catalog to PyTorch."""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch.distributed as dist
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from torch.utils.data import IterableDataset, get_worker_info

from lsdb_torch._compute import compute_pixel, limit_worker_threads
from lsdb_torch._rows import ColumnarPartition
from lsdb_torch._sharding import partition_order, shard


def _process_group_active() -> bool:
    return dist.is_available() and dist.is_initialized()


class LSDBDataset(IterableDataset):
    """Stream the rows of a lazy lsdb catalog into a PyTorch ``DataLoader``.

    Each DataLoader worker process (on each distributed rank) owns a disjoint
    shard of the catalog's HEALPix partitions, computes them one at a time in
    process with the synchronous Dask scheduler, and yields one dict per row.
    No Dask cluster is involved.

    Parameters
    ----------
    catalog : lsdb.Catalog (or any lsdb HealpixDataset)
        Any lazy catalog: opened, filtered, column-selected, crossmatched,
        ``map_partitions``-ed. Partition-level preprocessing belongs in lsdb.
    transform : callable, optional
        ``transform(row: dict) -> sample`` applied in the worker. Must be
        picklable (a module-level function), since worker processes may be
        spawned rather than forked.
    shuffle : bool, default True
        Shuffle the partition order every epoch and the row order within each
        partition.
    seed : int, default 0
        Base seed. Must be identical on every rank.
    loop : bool, default False
        Never stop: after exhausting an epoch, continue with the next one. This
        is the recommended mode for distributed training with a fixed number
        of steps per epoch.
    shuffle_buffer_size : int, default 0
        Size in rows of a random-eviction buffer mixing rows across partitions.
        Without it every batch comes from a single HEALPix pixel.
    rank, world_size : int, optional
        Override distributed detection (``torch.distributed`` if initialized,
        else the ``RANK``/``WORLD_SIZE`` environment variables, else 0/1).
    """

    def __init__(
        self,
        catalog: HealpixDataset,
        *,
        transform: Callable[[dict[str, Any]], Any] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        loop: bool = False,
        shuffle_buffer_size: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        if not isinstance(catalog, HealpixDataset):
            raise TypeError(f"catalog must be an lsdb catalog, got {type(catalog)}")
        self.catalog = catalog
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.loop = loop
        self.shuffle_buffer_size = shuffle_buffer_size
        self._pixels = catalog.get_healpix_pixels()
        self._next_epoch = 0
        self.rank, self.world_size = self._resolve_rank_world(rank, world_size)

    def _resolve_rank_world(self, rank: int | None, world_size: int | None) -> tuple[int, int]:
        """Explicit arguments, else the process group, else torchrun's environment, else (0, 1).

        Runs only in the main process: DataLoader workers must not touch ``torch.distributed``.
        When the process group is the source, also check that every rank built the same
        partitions, otherwise the shards would overlap or miss data silently.
        """
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be given together")
        if rank is not None:
            return int(rank), int(world_size)
        if _process_group_active():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, self._pixels)
            if any(other != self._pixels for other in gathered):
                raise RuntimeError("LSDBDataset: catalog partitions differ across ranks")
            return dist.get_rank(), dist.get_world_size()
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to derive the shuffle order (same contract as DistributedSampler).

        Not needed with ``persistent_workers=True`` or ``num_workers=0``: the
        long-lived dataset copy advances the epoch itself on every iteration.
        """
        self._next_epoch = int(epoch)

    def _partition_indices(self, epoch: int, shard_id: int, n_shards: int) -> Iterator[int]:
        """Partition indices owned by this shard, forever if ``loop``."""
        for e in itertools.count(epoch) if self.loop else [epoch]:
            order = partition_order(len(self._pixels), self.seed, e, self.shuffle)
            yield from shard(order, shard_id, n_shards, e).tolist()

    def _load(self, i: int) -> ColumnarPartition:
        return ColumnarPartition(compute_pixel(self.catalog, self._pixels[i]))

    def _partitions(self, epoch: int, shard_id: int, n_shards: int) -> Iterator[ColumnarPartition]:
        """Computed partitions, with the next one loaded on a background thread."""
        # One long-lived thread per iterator: fsspec filesystems cache an
        # instance per thread, so a thread per partition would leak.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lsdb_torch-prefetch")
        try:
            future = None
            for i in self._partition_indices(epoch, shard_id, n_shards):
                current, future = future, executor.submit(self._load, i)
                if current is not None:
                    yield current.result()
            if future is not None:
                yield future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def __iter__(self) -> Iterator[Any]:
        worker = get_worker_info()
        if worker is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker.id, worker.num_workers
            limit_worker_threads()
        shard_id = self.rank * num_workers + worker_id
        n_shards = self.world_size * num_workers

        epoch = self._next_epoch
        self._next_epoch += 1
        rng = np.random.default_rng([self.seed, epoch, shard_id])

        rows = self._rows(epoch, shard_id, n_shards, rng)
        if self.shuffle_buffer_size > 0:
            rows = _shuffle_buffer(rows, self.shuffle_buffer_size, rng)
        if self.transform is not None:
            rows = map(self.transform, rows)
        yield from rows

    def _rows(self, epoch: int, shard_id: int, n_shards: int, rng: np.random.Generator) -> Iterator[dict[str, Any]]:
        for partition in self._partitions(epoch, shard_id, n_shards):
            indices = rng.permutation(len(partition)).tolist() if self.shuffle else range(len(partition))
            for i in indices:
                yield partition.row(i)


def _shuffle_buffer(rows: Iterator[Any], size: int, rng: np.random.Generator) -> Iterator[Any]:
    """Fixed-size random-eviction buffer (tf.data ``shuffle`` semantics)."""
    buffer: list[Any] = []
    for row in rows:
        if len(buffer) < size:
            buffer.append(row)
            continue
        j = int(rng.integers(size))
        out, buffer[j] = buffer[j], row
        yield out
    rng.shuffle(buffer)
    yield from buffer
