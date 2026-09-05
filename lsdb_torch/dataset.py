"""An IterableDataset streaming rows of an LSDB catalog to PyTorch."""

from __future__ import annotations

import hashlib
import operator
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset
from torch.utils.data import IterableDataset, get_worker_info

from lsdb_torch._compute import limit_worker_threads
from lsdb_torch._iterator import CatalogIterator


def _nonnegative_int(name: str, value: int) -> int:
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class LSDBDataset(IterableDataset):
    """Stream a lazy LSDB catalog using ordinary PyTorch workers and batching.

    Parameters
    ----------
    catalog : lsdb.Catalog (or any lsdb HealpixDataset)
        Opened, filtered, crossmatched, or otherwise transformed catalog.
        Partition-level preprocessing belongs in LSDB. Project columns when
        opening a catalog to avoid reading unused data.
    transform : callable, optional
        ``transform(row: dict) -> sample`` inside the worker. Must be picklable
        when using multiprocessing. For checkpointing, use a deterministic
        transform or one managing its own state through ``state_dict`` and
        ``load_state_dict`` (including any random generators it uses).
    shuffle : bool, default True
        Shuffle partitions and rows deterministically using ``seed`` and the
        epoch supplied to :meth:`set_epoch`.
    seed : int, default 0
        Nonnegative seed, identical across data-parallel ranks.
    loop : bool, default False
        Repeat each worker's assigned partitions, reshuffling each local pass.
        Ownership stays fixed for the lifetime of the iterator. All workers
        must have some rows; an empty shard raises instead of spinning forever.
        Use one long-lived loader iterator for fixed-step distributed training.
    shuffle_buffer_size : int, default 0
        Number of rows in a random-eviction shuffle buffer. Buffered NumPy
        arrays are copied so they do not keep whole partitions alive. This is
        a row limit, not a byte limit; account for your sample sizes.
    partition_weights : sequence of float, optional
        Relative cost of each partition, in ``catalog.get_healpix_pixels()``
        order, identical on every rank. Greedily balance ranks, then workers.
        Supply known row counts, bytes, or measured costs for a fixed snapshot;
        the adapter does not infer counts for lazy operations. Without weights,
        balance partition counts. Neither guarantees equal batch counts.
    prefetch : bool, default True
        Compute one partition ahead on one background thread per iterator.
        Disable to reduce partition memory to one at a time.
    worker_cpu_threads, worker_io_threads : int or None, default 1
        Arrow CPU and I/O pool sizes in DataLoader workers. ``None`` preserves
        that pool's existing setting. Single-process loading leaves both alone.
    rank, world_size : int, optional
        Data-parallel rank and size. Pass both to override detection from
        ``torch.distributed`` or ``RANK``/``WORLD_SIZE``. Construction performs
        no collectives. Hybrid parallel jobs should pass their data-parallel
        coordinates explicitly, not their global GPU rank.
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
        partition_weights: Sequence[float] | None = None,
        prefetch: bool = True,
        worker_cpu_threads: int | None = 1,
        worker_io_threads: int | None = 1,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        if not isinstance(catalog, HealpixDataset):
            raise TypeError(f"catalog must be an lsdb catalog, got {type(catalog)}")
        self.catalog = catalog
        self.transform = transform
        self.shuffle = shuffle
        self.seed = _nonnegative_int("seed", seed)
        self.loop = loop
        self.shuffle_buffer_size = _nonnegative_int("shuffle_buffer_size", shuffle_buffer_size)
        self.prefetch = prefetch
        for name, value in (
            ("worker_cpu_threads", worker_cpu_threads),
            ("worker_io_threads", worker_io_threads),
        ):
            if value is not None and _nonnegative_int(name, value) == 0:
                raise ValueError(f"{name} must be positive or None")
        self.worker_cpu_threads = worker_cpu_threads
        self.worker_io_threads = worker_io_threads
        self._pixels = catalog.get_healpix_pixels()
        self.rank, self.world_size = self._resolve_rank_world(rank, world_size)
        self.partition_weights = None
        if partition_weights is not None:
            weights = np.array(partition_weights, dtype=np.float64, copy=True)
            if weights.shape != (len(self._pixels),) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("partition_weights must contain one finite, nonnegative value per catalog pixel")
            weights.flags.writeable = False
            self.partition_weights = weights
        # Shared tensor storage survives spawn/forkserver and reaches persistent
        # workers. Only the parent writes it, between loader iterations.
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        pixels = np.array([(p.order, p.pixel) for p in self._pixels], dtype="<i8")
        digest = hashlib.sha256(pixels.tobytes())
        if self.partition_weights is not None:
            digest.update(self.partition_weights.astype("<f8").tobytes())
        self._layout_digest = digest.hexdigest()

    @staticmethod
    def _resolve_rank_world(rank: int | None, world_size: int | None) -> tuple[int, int]:
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be given together")
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank, world_size = dist.get_rank(), dist.get_world_size()
            else:
                rank, world_size = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        rank = _nonnegative_int("rank", rank)
        world_size = _nonnegative_int("world_size", world_size)
        if not 0 <= rank < world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        return rank, world_size

    def set_epoch(self, epoch: int) -> None:
        """Set the next iterator's epoch, including in persistent workers.

        Call on every rank before ``iter(loader)``. The default epoch is zero;
        repeated iterations use the same order until this method is called.
        An existing iterator keeps its original epoch.
        """
        epoch = _nonnegative_int("epoch", epoch)
        if epoch > torch.iinfo(torch.int64).max:
            raise ValueError("epoch must fit in a signed 64-bit integer")
        self._epoch.fill_(epoch)

    def __iter__(self) -> Iterator[Any]:
        worker = get_worker_info()
        worker_id, num_workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        if worker is not None:
            limit_worker_threads(self.worker_cpu_threads, self.worker_io_threads)
        return CatalogIterator(self, int(self._epoch.item()), worker_id, num_workers)
