"""Compute a single catalog partition in the current process."""

from __future__ import annotations

import nested_pandas as npd
import pyarrow as pa
from hats.pixel_math import HealpixPixel
from lsdb.catalog.dataset.healpix_dataset import HealpixDataset


def compute_pixel(catalog: HealpixDataset, pixel: HealpixPixel) -> npd.NestedFrame:
    """Build the task graph of one partition and run it on the synchronous scheduler.

    The explicit scheduler matters: the global Dask configuration may point at a
    ``distributed.Client`` inherited from the parent process, and that client is
    dead inside a DataLoader worker.
    """
    (partition,) = catalog.to_delayed(pixels=[pixel])
    return partition.compute(scheduler="synchronous")


def limit_worker_threads(cpu_count: int | None = 1, io_count: int | None = 1) -> None:
    """Cap pyarrow's separate CPU and I/O pools inside a DataLoader worker.

    Every worker would otherwise create an all-core pool; N workers times all
    cores thrashes the node while decoding parquet.
    """
    if cpu_count is not None:
        pa.set_cpu_count(cpu_count)
    if io_count is not None:
        pa.set_io_thread_count(io_count)
