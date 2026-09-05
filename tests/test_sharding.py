import numpy as np
import pytest

from lsdb_torch._sharding import partition_indices, partition_order, shard


@pytest.mark.parametrize("n", [1, 7, 12, 100])
@pytest.mark.parametrize("n_shards", [1, 2, 5])
@pytest.mark.parametrize("epoch", [0, 1, 3])
def test_shards_are_disjoint_and_complete(n, n_shards, epoch):
    order = partition_order(n, seed=0, epoch=epoch, shuffle=True)
    pieces = [shard(order, s, n_shards, epoch) for s in range(n_shards)]
    assert sorted(np.concatenate(pieces).tolist()) == list(range(n))


def test_surplus_rotates_between_shards():
    order = np.arange(5)
    sizes = [tuple(len(shard(order, s, 2, e)) for s in range(2)) for e in range(2)]
    assert sizes == [(3, 2), (2, 3)]


def test_order_is_seeded_and_epoch_dependent():
    a = partition_order(50, 1, 0, True)
    assert np.array_equal(a, partition_order(50, 1, 0, True))
    assert not np.array_equal(a, partition_order(50, 1, 1, True))
    assert not np.array_equal(a, partition_order(50, 2, 0, True))
    assert np.array_equal(partition_order(5, 1, 3, False), np.arange(5))


@pytest.mark.parametrize("n", [0, 3, 31])
@pytest.mark.parametrize("epoch", [0, 1, 4])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_rank_ownership_is_independent_of_heterogeneous_worker_counts(n, epoch, shuffle, weighted):
    weights = np.arange(n, dtype=float) if weighted else None
    worker_counts = [1, 2, 5]
    order = partition_order(n, seed=3, epoch=epoch, shuffle=shuffle)
    positions = {int(pixel): i for i, pixel in enumerate(order)}
    all_pieces = []
    for rank, num_workers in enumerate(worker_counts):
        expected = partition_indices(n, 3, epoch, shuffle, rank, len(worker_counts), 0, 1, weights)
        pieces = [
            partition_indices(
                n,
                3,
                epoch,
                shuffle,
                rank,
                len(worker_counts),
                worker,
                num_workers,
                weights,
            )
            for worker in range(num_workers)
        ]
        assert sorted(np.concatenate(pieces).tolist()) == sorted(expected.tolist())
        for piece in pieces:
            assert [positions[int(pixel)] for pixel in piece] == sorted(positions[int(pixel)] for pixel in piece)
        all_pieces.extend(pieces)
    assert sorted(np.concatenate(all_pieces).tolist()) == list(range(n))


def test_weighted_assignment_balances_rank_and_worker_costs():
    weights = np.array([100, 1, 90, 1, 80, 1, 70, 1], dtype=float)
    rank_loads = []
    for rank in range(2):
        pieces = [partition_indices(8, 0, 0, False, rank, 2, worker, 2, weights) for worker in range(2)]
        worker_loads = [weights[piece].sum() for piece in pieces]
        rank_loads.append(sum(worker_loads))
        # A giant partition cannot be split, but small partitions should fill the lighter worker.
        assert max(worker_loads) - min(worker_loads) <= weights[np.concatenate(pieces)].max()
    assert rank_loads == [172, 172]


@pytest.mark.parametrize("weight", [0.0, 1.0, 1e308])
def test_equal_weights_balance_partition_counts_including_zero_and_large_costs(weight):
    weights = np.full(17, weight)
    counts = []
    for rank in range(3):
        pieces = [partition_indices(17, 3, 2, True, rank, 3, worker, 2, weights) for worker in range(2)]
        worker_counts = [len(piece) for piece in pieces]
        assert max(worker_counts) - min(worker_counts) <= 1
        counts.append(sum(worker_counts))
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == 17


@pytest.mark.parametrize("weighted", [False, True])
def test_partition_assignment_is_deterministic_and_seeded(weighted):
    weights = np.ones(50) if weighted else None

    def assignment(seed, epoch):
        return partition_indices(50, seed, epoch, True, 0, 2, 0, 2, weights)

    first = assignment(1, 0)
    np.testing.assert_array_equal(first, assignment(1, 0))
    assert not np.array_equal(first, assignment(2, 0))
    assert not np.array_equal(first, assignment(1, 1))
