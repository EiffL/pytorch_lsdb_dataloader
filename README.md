# lsdb-torch

Stream an [LSDB](https://github.com/astronomy-commons/lsdb) catalog into a PyTorch `DataLoader`,
on one machine or across a multi-node distributed training job.

```python
import lsdb
from torch.utils.data import DataLoader
from lsdb_torch import LSDBDataset

gz10 = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gz10")
sdss = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_sdss_sdss")
xmatch = gz10.crossmatch(sdss, n_neighbors=1)

ds = LSDBDataset(xmatch, transform=decode)      # decode(row: dict) -> sample, a module-level function
loader = DataLoader(ds, batch_size=32, num_workers=10)
for batch in loader:
    ...
```

Any lazy catalog works: opened, region-filtered, column-selected (`catalog[["ra", "spectrum.flux"]]`),
crossmatched, or transformed with `catalog.map_partitions(...)`. Do partition-level preprocessing in lsdb;
do per-sample work (image decoding, tensor conversion) in `transform`, which runs inside the workers.

## How it works

An lsdb catalog is a lazy graph of HEALPix-partitioned operations. `LSDBDataset` gives each DataLoader worker
process, on each rank, a disjoint shard of the partitions. The worker asks lsdb for the task graph of one
partition at a time, culled to that single pixel (for a crossmatch: three parquet reads and the match), and
runs it in process with Dask's synchronous scheduler while the next partition is prefetched on a background
thread. Only public lsdb API is used: `catalog.to_delayed(pixels=[pixel])` and `compute(scheduler="synchronous")`.

There is no Dask cluster anywhere. The DataLoader workers are the executors, so scaling data loading is
`num_workers`, and the only thing shipped to a worker is the pickled catalog (kilobytes to a few megabytes).
Opening remote catalogs reads metadata once, in the main process.

Rows are yielded as dicts: numpy scalars for numeric columns, `{field: np.ndarray}` for nested columns
(zero-copy views into the partition), and python objects (`str`, `bytes`, struct dicts) otherwise. The HATS
spatial index `_healpix_29` is included as the sample id.

## Shuffling and epochs

- `shuffle=True` permutes the partition order every epoch and the row order within each partition.
- Each batch is collated inside one worker from consecutive rows, so without further mixing a batch comes
  from a single HEALPix pixel. Set `shuffle_buffer_size` (in rows, e.g. 1000 for tiny partitions, 10k for
  large ones) to mix rows across partitions at the cost of that much memory per worker. The buffer must fill
  before the first row is yielded, so on slow remote catalogs it delays the first batch accordingly.
- Call `ds.set_epoch(epoch)` before each epoch, exactly like `DistributedSampler`. With
  `persistent_workers=True` (recommended, worker startup imports lsdb and torch) this is not needed: the
  long-lived workers advance the epoch themselves.

## Distributed training

Rank and world size are detected from `torch.distributed` if initialized, else from the `RANK` and
`WORLD_SIZE` environment variables set by `torchrun`, else default to a single process. Pass `rank=` and
`world_size=` to override. Every rank must construct the same catalog with the same `seed` and use the same
`num_workers`. When the rank comes from the process group, construction gathers every rank's partition list and
fails if they differ.

Finite epochs are ragged: partitions have different row counts, and for crossmatches the counts are not even
known in advance, so ranks yield different numbers of batches. Two supported recipes:

```python
# 1. step-based training: never run out of data
ds = LSDBDataset(xmatch, loop=True)
for step, batch in zip(range(steps_per_epoch), loader): ...

# 2. finite epochs: let DDP handle ranks that finish early
from torch.distributed.algorithms.join import Join
with Join([ddp_model]):
    for batch in loader: ...
```

## Practical notes

- `transform` must be picklable (a module-level function): Python 3.14 starts workers with `forkserver`.
- Nested columns with variable-length lists (light curves) need a custom `collate_fn`; fixed-length ones
  (spectra) stack with the default collate.
- Nested arrays are read-only views; convert them in `transform` if you mutate in place.
- Use `DataLoader(timeout=...)` against remote catalogs so a stalled HTTP read cannot hang a distributed job.
- Memory per worker is about two partitions plus `prefetch_factor` collated batches.
- A crossmatch partition reads the whole right-hand partition (plus its margin) to match the left rows, every
  epoch. If that is expensive or bandwidth-bound (for example matching a few thousand galaxies against a
  catalog of spectra over HTTP), materialize it once with `xmatch.write_catalog(path)`, on a Dask cluster if you
  like, and train from the written catalog. The written catalog also carries row counts, so its partitions are
  known in advance.

## Development

The per-pixel `to_delayed(pixels=...)` API is not in lsdb 0.10.4 yet; until it is released, install lsdb from
a checkout that includes it (for example `pip install -e ../lsdb`), or point `PYTHONPATH` at its `src`.

```
pip install -e .[dev]
python -m pytest -q
python examples/mmu_crossmatch.py                # user story, over the network
torchrun --nproc_per_node=2 examples/ddp_smoke.py  # sharding check with a gloo process group
```
