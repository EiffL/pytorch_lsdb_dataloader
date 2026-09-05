"""Pure helpers deciding which partitions a given (rank, worker) consumes."""

from __future__ import annotations

import heapq

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


def _weighted_shard(order: np.ndarray, weights: np.ndarray, shard_id: int, n_shards: int, epoch: int) -> np.ndarray:
    """Balance estimated costs while retaining the supplied traversal order."""
    if not 0 <= shard_id < n_shards:
        raise ValueError(f"shard_id {shard_id} out of range for {n_shards} shards")
    if n_shards == 1 or len(order) == 0:
        return order
    # Stable sorting lets the epoch's shuffled order break equal-cost ties.
    priorities = np.argsort(-weights[order], kind="stable")
    owners = np.empty(len(order), dtype=np.int64)
    # Partition counts break equal-load ties, including an entirely zero-cost catalog.
    loads = [(0.0, 0, (i + epoch) % n_shards, i) for i in range(n_shards)]
    heapq.heapify(loads)
    for position in priorities:
        load, count, tie_break, owner = loads[0]
        owners[position] = owner
        heapq.heapreplace(loads, (load + float(weights[order[position]]), count + 1, tie_break, owner))
    return order[owners == shard_id]


def partition_indices(
    n_partitions: int,
    seed: int,
    epoch: int,
    shuffle: bool,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Assign partitions to a rank, then divide that rank's work among its workers.

    Rank ownership is independent of local worker counts. When supplied, weights
    are finite nonnegative cost estimates in catalog pixel order, validated by
    the caller. Greedy assignment balances costs at both levels; it does not
    guarantee equal row counts or equal numbers of training batches.
    """
    order = partition_order(n_partitions, seed, epoch, shuffle)
    if weights is None:
        owned = shard(order, rank, world_size, epoch)
        return shard(owned, worker_id, num_workers, epoch)

    weights = np.asarray(weights, dtype=np.float64)
    # Scaling preserves relative costs and prevents overflow when summing large estimates.
    maximum = weights.max(initial=0.0)
    if maximum > 0:
        weights = weights / maximum
    owned = _weighted_shard(order, weights, rank, world_size, epoch)
    return _weighted_shard(owned, weights, worker_id, num_workers, epoch)
