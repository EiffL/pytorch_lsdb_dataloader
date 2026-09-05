"""Resume consumed samples, including worker prefetch and shuffle-buffer tails."""

import itertools
import pickle

import pytest

from lsdb_torch import LSDBDataset
from tests.helpers import RandomTransform, keep_index_only


@pytest.mark.parametrize("buffer_size", [0, 17, 3000])
@pytest.mark.parametrize("consumed", [0, 73, 1995, 2000])
def test_iterator_checkpoint(catalog, buffer_size, consumed):
    ds = LSDBDataset(
        catalog,
        transform=keep_index_only,
        shuffle_buffer_size=buffer_size,
        prefetch=False,
    )
    ds.set_epoch(4)
    original = iter(ds)
    list(itertools.islice(original, consumed))
    state = original.state_dict()
    saved = pickle.dumps(state)
    expected = list(original)
    # A snapshot is independent of subsequent iteration.
    assert pickle.dumps(state) == saved
    restored = iter(
        LSDBDataset(
            catalog,
            transform=keep_index_only,
            shuffle_buffer_size=buffer_size,
            prefetch=False,
        )
    )
    restored.load_state_dict(pickle.loads(saved))
    assert list(restored) == expected


def test_restore_reads_only_needed_partitions(catalog, monkeypatch):
    import lsdb_torch._iterator as module

    reads = []
    compute = module.compute_pixel

    def record(cat, pixel):
        reads.append(pixel)
        return compute(cat, pixel)

    monkeypatch.setattr(module, "compute_pixel", record)
    ds = LSDBDataset(catalog, transform=keep_index_only, shuffle_buffer_size=30, prefetch=False)
    original = iter(ds)
    list(itertools.islice(original, 1000))
    state = original.state_dict()
    expected = next(original)
    original.close()
    reads.clear()
    restored = iter(ds)
    restored.load_state_dict(state)
    try:
        assert next(restored) == expected
        needed = {ds._pixels[i] for i, _ in state["buffer_refs"]}
        # The source may be at the last row of a partition, so allow the next
        # partition as well as the current one when resuming.
        start = state["partition_position"]
        needed.update(ds._pixels[i] for i in restored._owned[start : start + 2])
        assert set(reads) <= needed
        assert len(reads) <= len(needed) + 1  # current pixel may also be in buffer
        assert len(reads) < len(catalog.get_healpix_pixels())
    finally:
        restored.close()


@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 4},
        {"shuffle": False},
        {"world_size": 2, "rank": 0},
        {"shuffle_buffer_size": 2},
    ],
)
def test_reject_incompatible_checkpoint(catalog, changed):
    original = iter(LSDBDataset(catalog))
    state = original.state_dict()
    original.close()
    restored = iter(LSDBDataset(catalog, **changed))
    with pytest.raises(ValueError, match="does not match"):
        restored.load_state_dict(state)
    restored.close()


def test_transform_random_state_is_restored(catalog):
    def make():
        return iter(LSDBDataset(catalog, transform=RandomTransform(), shuffle_buffer_size=17))

    original = make()
    list(itertools.islice(original, 51))
    state = original.state_dict()
    expected = list(itertools.islice(original, 40))
    original.close()
    restored = make()
    restored.load_state_dict(state)
    assert list(itertools.islice(restored, 40)) == expected
    restored.close()


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("loop", [False, True])
def test_stateful_loader_restores_consumed_batches(catalog, num_workers, loop):
    torchdata = pytest.importorskip("torchdata.stateful_dataloader")

    def make():
        ds = LSDBDataset(catalog, transform=keep_index_only, shuffle_buffer_size=31, loop=loop)
        return torchdata.StatefulDataLoader(
            ds,
            batch_size=13,
            num_workers=num_workers,
            persistent_workers=bool(num_workers),
            prefetch_factor=2 if num_workers else None,
            snapshot_every_n_steps=3,
        )

    loader = make()
    loader.dataset.set_epoch(7)
    original = iter(loader)
    # A loop checkpoint after each worker has completed at least one pass.
    list(itertools.islice(original, 190 if loop else 19))
    state = pickle.loads(pickle.dumps(loader.state_dict()))
    expected = [batch.tolist() for batch in itertools.islice(original, 30)]
    del original, loader
    restored = make()
    restored.load_state_dict(state)
    it = iter(restored)
    assert [batch.tolist() for batch in itertools.islice(it, 30)] == expected
    del it, restored
