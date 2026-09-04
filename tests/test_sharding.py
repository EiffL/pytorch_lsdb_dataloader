import numpy as np
import pytest

from lsdb_torch._sharding import partition_order, shard


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
