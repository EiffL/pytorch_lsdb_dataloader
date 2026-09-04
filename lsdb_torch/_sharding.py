"""Pure helpers deciding which partitions a given (rank, worker) consumes."""

from __future__ import annotations

import numpy as np


def partition_order(n: int, seed: int, epoch: int, shuffle: bool) -> np.ndarray:
    """Order in which the ``n`` partitions are visited during ``epoch``.

    Identical on every rank and worker for a given (seed, epoch), which is what
    makes the strided shards below disjoint and complete.
    """
    if not shuffle:
        return np.arange(n)
    return np.random.default_rng([seed, epoch]).permutation(n)


def shard(order: np.ndarray, shard_id: int, n_shards: int, epoch: int) -> np.ndarray:
    """Strided slice of ``order`` owned by ``shard_id``.

    The starting offset is rotated by ``epoch`` so that, when ``len(order)`` is
    not a multiple of ``n_shards``, the surplus partitions do not always land on
    the same shards. Shards are disjoint and cover ``order`` exactly.
    """
    if not 0 <= shard_id < n_shards:
        raise ValueError(f"shard_id {shard_id} out of range for {n_shards} shards")
    return order[(shard_id + epoch) % n_shards :: n_shards]
