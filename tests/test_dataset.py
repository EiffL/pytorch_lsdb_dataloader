import gc
import itertools
import threading
import time
from collections import Counter

import dask
import numpy as np
import pytest
import torch
from hats.pixel_math.spatial_index import spatial_index_to_healpix
from torch.utils.data import DataLoader

from lsdb_torch import LSDBDataset
from tests.helpers import drop_odd_pixels, id_transform, keep_index_only


def collect(catalog, world_size, num_workers, **kwargs):
    out = []
    for rank in range(world_size):
        ds = LSDBDataset(catalog, rank=rank, world_size=world_size, transform=keep_index_only, **kwargs)
        out.extend(DataLoader(ds, batch_size=None, num_workers=num_workers))
    return out


@pytest.mark.parametrize("which", ["catalog", "crossmatch"])
@pytest.mark.parametrize("world_size", [1, 2, 3])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_every_row_exactly_once(request, which, world_size, num_workers, expected_index):
    cat = request.getfixturevalue(which)
    assert sorted(collect(cat, world_size, num_workers)) == expected_index


def test_rejects_non_catalog():
    with pytest.raises(TypeError):
        LSDBDataset("not a catalog")


def test_resolve_rank_world(catalog, monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert (LSDBDataset(catalog).rank, LSDBDataset(catalog).world_size) == (0, 1)
    ds = LSDBDataset(catalog, rank=2, world_size=4)
    assert (ds.rank, ds.world_size) == (2, 4)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "3")
    assert (LSDBDataset(catalog).rank, LSDBDataset(catalog).world_size) == (1, 3)
    with pytest.raises(ValueError):
        LSDBDataset(catalog, rank=1)


def test_accepts_margin_catalog(catalog):
    margin = catalog.margin
    got = list(LSDBDataset(margin, transform=keep_index_only))
    assert sorted(got) == sorted(margin.compute(progress_bar=False).index.to_list())


def test_row_contents(catalog, frame):
    ds = LSDBDataset(catalog, shuffle=False)
    row = next(iter(ds))
    assert row["_healpix_29"] == frame.index[0]
    assert isinstance(row["ra"], np.floating)
    nested = row["nested"]
    assert isinstance(nested, dict) and isinstance(nested["t"], np.ndarray)
    expected = frame["nested"].nest.to_flat().loc[frame.index[0]]
    np.testing.assert_array_equal(nested["t"], expected["t"].to_numpy())
    assert set(row) == {"_healpix_29", *frame.columns}


def test_default_collate_stacks_fixed_length_nested(catalog):
    numeric = catalog[["ra", "dec", "nested.t", "nested.flux"]]
    loader = DataLoader(LSDBDataset(numeric), batch_size=8, num_workers=0)
    batch = next(iter(loader))
    assert batch["ra"].shape == (8,)
    assert isinstance(batch["nested"]["t"], torch.Tensor)
    assert batch["nested"]["t"].shape[0] == 8


def test_shuffle_is_deterministic_and_epoch_dependent(catalog):
    def run(epoch):
        ds = LSDBDataset(catalog, seed=3, transform=keep_index_only)
        ds.set_epoch(epoch)
        return list(ds)

    a, b, c = run(0), run(0), run(1)
    assert a == b
    assert c != a and sorted(c) == sorted(a)


def test_no_shuffle_follows_catalog_order(catalog, frame):
    got = list(LSDBDataset(catalog, shuffle=False, transform=keep_index_only))
    assert got == frame.index.to_list()


def test_persistent_workers_advance_epoch(catalog):
    ds = LSDBDataset(catalog, transform=keep_index_only)
    loader = DataLoader(ds, batch_size=None, num_workers=2, persistent_workers=True)
    first, second = list(loader), list(loader)
    assert sorted(first) == sorted(second)
    assert first != second


def test_num_workers_zero_advances_epoch_without_set_epoch(catalog):
    ds = LSDBDataset(catalog, transform=keep_index_only)
    assert list(ds) != list(ds)


@pytest.mark.parametrize("context", ["fork", "spawn", "forkserver"])
def test_start_methods(catalog, context, expected_index):
    ds = LSDBDataset(catalog, transform=keep_index_only)
    loader = DataLoader(ds, batch_size=None, num_workers=2, multiprocessing_context=context)
    assert sorted(loader) == expected_index


def test_does_not_use_global_dask_scheduler(catalog, expected_index):
    def boom(*args, **kwargs):
        raise AssertionError("global scheduler must not be used")

    with dask.config.set(scheduler=boom):
        assert sorted(LSDBDataset(catalog, transform=keep_index_only)) == expected_index


def test_shuffle_buffer_mixes_pixels(catalog):
    def head(**kwargs):
        return list(itertools.islice(LSDBDataset(catalog, transform=keep_index_only, seed=0, **kwargs), 100))

    def pixels(ids):
        return set(spatial_index_to_healpix(ids, target_order=2).tolist())

    plain, mixed = head(), head(shuffle_buffer_size=500)
    assert len(pixels(mixed)) > len(pixels(plain))
    assert mixed == head(shuffle_buffer_size=500)
    full = list(LSDBDataset(catalog, transform=keep_index_only, shuffle_buffer_size=64))
    assert len(full) == len(set(full))


def test_loop_covers_everything_repeatedly(catalog, expected_index):
    n = len(expected_index)
    ds = LSDBDataset(catalog, loop=True, transform=keep_index_only)
    counts = Counter(itertools.islice(ds, 2 * n))
    assert set(counts) == set(expected_index)
    assert all(v == 2 for v in counts.values())


def test_iterator_close_releases_thread(catalog):
    def prefetch_threads():
        return sum(t.name.startswith("lsdb_torch-prefetch") for t in threading.enumerate())

    def wait_for(count):
        deadline = time.monotonic() + 30
        while prefetch_threads() != count and time.monotonic() < deadline:
            time.sleep(0.05)
        return prefetch_threads()

    gc.collect()  # iterators abandoned by earlier tests release their threads
    assert wait_for(0) == 0
    it = iter(LSDBDataset(catalog, loop=True))
    next(it)
    assert prefetch_threads() == 1
    it.close()
    assert wait_for(0) == 0


def test_empty_partitions_are_skipped(catalog, expected_index):
    sparse = catalog.map_partitions(drop_odd_pixels, include_pixel=True)
    expected = sorted(sparse.compute(progress_bar=False).index.to_list())
    assert 0 < len(expected) < len(expected_index)
    assert sorted(collect(sparse, 2, 2)) == expected


def test_transform_receives_dict(catalog):
    ds = LSDBDataset(catalog, transform=id_transform)
    assert isinstance(next(iter(ds)), dict)
