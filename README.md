# lsdb-torch

Stream an [LSDB](https://github.com/astronomy-commons/lsdb) catalog through PyTorch `IterableDataset`
and `DataLoader`, locally or across data-parallel ranks. Lazy filters, crossmatches, and
partition preprocessing stay in LSDB; workers compute partitions and decode training samples.

## Install

```bash
pip install -e .
pip install -e '.[examples]'    # MMU example: Pillow
pip install -e '.[checkpoint]' # Optional TorchData StatefulDataLoader
```

LSDB is pinned to immutable commit `a478de002ad7ee073c826ded92f501462bd739a3`
providing efficient `to_delayed(pixels=...)` and local scheduler support. See [pyproject.toml](pyproject.toml).
Replace that pin with a version requirement when the necessary API is released.

## Basic use

```python
import lsdb
from torch.utils.data import DataLoader
from lsdb_torch import LSDBDataset

if __name__ == "__main__":
    catalog = lsdb.open_catalog("/data/catalog", columns=["ra", "dec"])
    ds = LSDBDataset(catalog, seed=42)
    loader = DataLoader(
        ds, batch_size=32, num_workers=4,
        persistent_workers=True, multiprocessing_context="spawn",
        pin_memory=True, prefetch_factor=1,
    )
    for epoch in range(3):
        ds.set_epoch(epoch)
        for batch in loader:
            # Numeric columns are CPU tensors, e.g. batch["ra"].
            ...
```

This finite-epoch example is suitable for one training rank. See the distributed
section before using a finite loader in a synchronized training loop.

Pass `transform=decode` to apply a picklable callable to each row inside the worker.
Use a module-level function or a picklable callable object. Custom `collate_fn`
functions also run in workers and should be defined at module level. The main guard
above is required for spawn-based multiprocessing.

Project columns at open, e.g. `lsdb.open_catalog(path, columns=["ra", "dec", "spectrum.flux"])`,
and apply partition preprocessing with `catalog.map_partitions(...)`; the adapter executes that lazy graph.

Rows are dictionaries containing numeric NumPy values, nested dictionaries of arrays,
and Python objects such as strings, bytes, or image structs. A missing nested record
is `None`; a valid empty record contains empty arrays. Arrays may be read-only views or copies;
copy arrays before mutating them. There is no end-to-end zero-copy guarantee.
`_healpix_29` is a spatial index and can repeat; use a source identifier when uniqueness matters.

The default collator handles fixed-size numeric arrays. Variable lengths and nulls need
a custom collator. Return tensors or containers of tensors for automatic memory pinning;
a list of NumPy arrays from a custom collator is not automatically pinned. See
[PyTorch's data loading documentation](https://docs.pytorch.org/docs/stable/data.html).

## Execution and partition ownership

The adapter enumerates pixels with `catalog.get_healpix_pixels()`, requests each owned
partition with `catalog.to_delayed(pixels=[pixel])`, and computes it with the synchronous
Dask scheduler. These are public APIs. No Dask cluster is required by the loader.
Training processes and their worker replicas hold catalog metadata and the lazy graph.

Partitions are assigned to data-parallel ranks first, then to each rank's local workers.
Ranks may use different worker counts without changing their rank-level ownership.
Changing worker count can still change batching, row order, and worker-local repetition.

By default, assignment balances the number of partitions. Optional `partition_weights`
greedily balances estimated cost at both levels:

```python
pixels = catalog.get_healpix_pixels()
# costs_by_pixel comes from a manifest or measurements for this catalog snapshot.
weights = [costs_by_pixel[pixel] for pixel in pixels]
ds = LSDBDataset(catalog, partition_weights=weights, seed=42)
```

Supply one finite, nonnegative weight per pixel, in exactly that order, on every rank.
Known row counts, bytes, or measured processing costs are reasonable inputs. Counts
from source catalogs do not describe filtered or crossmatched output automatically.
The adapter never computes row counts to infer weights. Neither assignment method
guarantees equal rows, batches, or processing time.

## Epochs and distributed training

Call `ds.set_epoch(epoch)` on every rank **before creating the next loader iterator**,
including with persistent workers. The epoch is shared with those workers. It never
advances automatically: repeated iterations use epoch zero until changed, and setting
the same epoch repeats the adapter's ordering. An existing iterator retains its epoch.
Random transforms have their own state and reproducibility requirements.

Rank/size detection uses an initialized default `torch.distributed` process group,
otherwise `RANK`/`WORLD_SIZE`, otherwise `0`/`1`. Override with both `rank=` and
`world_size=`. Construction performs no collective communication and does not verify
catalog agreement across ranks. The trainer must supply the same immutable catalog,
pixel order, seed, epoch, and weights everywhere.

In hybrid parallel training, pass **data-parallel** coordinates explicitly. Global GPU
rank is not the right coordinate for tensor/pipeline/context parallel peers that must
consume coordinated samples. Coordinate their loading in the training application.

Finite shards generally produce different batch counts. `drop_last=True` drops each
worker's partial batch; it does not equalize ranks. There is no dataset `__len__`.
For synchronous distributed training, use an explicitly balanced finite schedule or
a repeating loader with the same number of training steps on every data-parallel rank:

```python
ds = LSDBDataset(catalog, loop=True, seed=42, rank=dp_rank, world_size=dp_size)
loader = DataLoader(ds, batch_size=32, num_workers=4, persistent_workers=True,
                    multiprocessing_context="spawn", pin_memory=True, prefetch_factor=1)
ds.set_epoch(0)
batches = iter(loader)  # Retain this iterator across trainer epoch boundaries.
for training_epoch in range(num_training_epochs):
    for step in range(steps_per_epoch):  # Same schedule on every data-parallel rank.
        batch = next(batches)
        # Training application: transfer tensors, forward/backward, optimizer step.
        ...
```

With `loop=True`, each worker repeats its fixed partition assignment, reshuffling each
local pass when enabled. These cycles are independent; they are not global catalog
epochs. Smaller shards can repeat their rows more often. Every worker must have at
least one row in its assignment: an empty shard raises an error instead of spinning.
Use fewer ranks/workers or a more finely partitioned catalog when necessary.

Choose repetition and loss weighting deliberately. Equal step counts do not imply an
exactly-once catalog pass, equal sample contributions, or equal token contributions.
Variable batch/token counts also require the trainer's intended loss normalization.
[DDP Join](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.join)
alone does not reconcile ordinary optimizer/scheduler state after uneven
training, and additional collectives need their own handling; it is not a general
replacement for a consistent distributed step schedule.

## MMU images and spectra

[examples/mmu_crossmatch.py](examples/mmu_crossmatch.py) projects the GZ10 image,
redshift, and label columns and the SDSS flux/wavelength columns before crossmatching.
For repeated training, materialize the crossmatch once:

```bash
python examples/mmu_crossmatch.py --materialize /data/mmu_crossmatch
python examples/mmu_crossmatch.py --catalog /data/mmu_crossmatch
python examples/mmu_crossmatch.py  # Optional live remote crossmatch preview
```

Materialization writes a new HATS catalog without overwriting an existing one. A live
crossmatch recomputes matching and reads its source partitions and margins on every
pass; source work may exceed the size of the matched output substantially.

The example yields uint8 NCHW image tensors when image sizes agree, otherwise a list
of CHW tensors. Spectra are float32 padded tensors with lengths and a boolean padding
mask; that mask does not describe survey measurement quality. Labels, redshifts, and
spatial indices are tensors too. It checks malformed spectra, preserves missing
redshifts as NaN, and leaves image resizing, normalization, and GPU transfer to training.

## Shuffling, concurrency, and memory

| Dataset option | Default | Behavior |
| --- | --- | --- |
| `shuffle` | `True` | Permute partitions and rows using seed/epoch/local cycle. |
| `shuffle_buffer_size` | `0` | Mix rows across partitions in a worker-local random-eviction buffer. |
| `prefetch` | `True` | Compute one next partition on a background thread per iterator. |
| `worker_cpu_threads` | `1` | Arrow CPU pool size in each DataLoader worker; `None` preserves it. |
| `worker_io_threads` | `1` | Arrow I/O pool size in each DataLoader worker; `None` preserves it. |

Batches consume consecutive rows within a worker, so large pixels can dominate a
batch without buffering. The shuffle buffer fills before emitting samples. Its arrays
are copied to avoid retaining entire source partitions. Its capacity is measured in
**rows, not bytes**; large samples still require substantial memory.

Budget worker metadata, source frames and conversion intermediates, current/next
partitions, owned shuffle-buffer rows, decoded samples, and collated batches together.
DataLoader also prefetches `prefetch_factor` batches per worker; pinning introduces
additional host buffers. `prefetch=False` removes the extra computed partition, not
these other allocations. Single-process loading leaves Arrow thread settings unchanged.

Tune workers and memory per node. Four workers per GPU on 8-GPU nodes means 32 workers
per node, or 512 workers across 16 nodes. More workers cannot remove storage bandwidth
limits. Pinning and `non_blocking=True` transfers can help; actual overlap with GPU
compute requires appropriate stream handling in the trainer. See the
[PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

`DataLoader(timeout=...)` limits waiting for a worker batch. Configure storage backend
request timeouts/retries separately. Iterator cleanup releases queued work and owned
buffers, but an already-running storage read cannot be forcibly cancelled.

## Checkpointing

Install `.[checkpoint]` and replace `DataLoader` with TorchData's `StatefulDataLoader`.
The adapter's worker-local iterators implement state hooks; TorchData accounts for
worker prefetch when capturing progress corresponding to consumed batches:

```python
from torchdata.stateful_dataloader import StatefulDataLoader

loader = StatefulDataLoader(ds, batch_size=32, num_workers=4,
                           persistent_workers=True, multiprocessing_context="spawn",
                           in_order=True, snapshot_every_n_steps=100)
batches = iter(loader)
batch = next(batches)
# Complete the corresponding training step before saving the training checkpoint.
data_state = loader.state_dict()  # Save this rank's state in the training checkpoint.

# After recreating ds and loader with the same configuration:
loader.load_state_dict(data_state)
batches = iter(loader)  # Resume; do not reset the epoch or recreate this each step.
```

Save/load loader state **per data-parallel rank**, along with the trainer's model,
optimizer, scheduler, and random state at a consistent step. TorchData aggregates
workers, not ranks. Keep rank/worker counts, batch size, `in_order=True`, transforms,
collation, and dataset configuration unchanged across restoration.

The catalog contents and lazy output row order must be immutable and reproducible.
Checkpoint validation checks pixels, weights, ordering options, and topology; its
fingerprint is not a content hash and cannot detect changed source rows or transforms.
Checkpoints store shuffle-buffer row references: restore rereads each distinct buffered pixel,
then the current partition as needed, which may read that pixel again.

A stochastic transform must own its RNG and expose both `state_dict()` and
`load_state_dict()`. Arbitrary global RNG state and random collation are not captured
by the adapter. `snapshot_every_n_steps` controls worker-state transfer overhead and
possible replay during restoration; it does not schedule durable training checkpoints.
See [TorchData's StatefulDataLoader documentation](https://meta-pytorch.org/data/beta/torchdata.stateful_dataloader.html).

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
torchrun --standalone --nproc_per_node=2 examples/ddp_smoke.py
```

Tests use synthetic/local catalogs; the MMU commands explicitly access remote data.
