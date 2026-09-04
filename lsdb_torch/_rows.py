"""Once-per-partition column extraction and per-row dict construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from nested_pandas import NestedDtype

Getter = Callable[[int], Any]


def _nested_getter(series: pd.Series) -> Getter:
    """Zero-copy access to the lists of a nested column, one dict of arrays per row."""
    lists = series.array.list_array.combine_chunks()
    offsets = np.asarray(lists.offsets)
    structs = lists.values
    flat = {name: structs.field(name).to_numpy(zero_copy_only=False) for name in structs.type.names}

    def get(i: int) -> dict[str, np.ndarray]:
        start, stop = offsets[i], offsets[i + 1]
        return {name: values[start:stop] for name, values in flat.items()}

    return get


def _object_getter(series: pd.Series) -> Getter:
    """Python-object access (str, bytes, struct dicts, ...) through pyarrow."""
    array = pa.array(series)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    return lambda i: array[i].as_py()


def _is_plain_numeric(series: pd.Series) -> bool:
    """True when ``to_numpy()`` yields a numeric array without boxing values."""
    dtype = series.dtype
    if isinstance(dtype, pd.ArrowDtype):
        pa_type = dtype.pyarrow_dtype
        numeric = pa.types.is_floating(pa_type) or pa.types.is_integer(pa_type) or pa.types.is_boolean(pa_type)
        return numeric and not series.hasnans
    return dtype != object and pd.api.types.is_numeric_dtype(dtype)


def column_getter(series: pd.Series) -> Getter:
    """Pick the cheapest per-row accessor for a column, from its dtype alone."""
    if isinstance(series.dtype, NestedDtype):
        return _nested_getter(series)
    if _is_plain_numeric(series):
        values = series.to_numpy()
        return lambda i: values[i]
    return _object_getter(series)


class ColumnarPartition:
    """A computed partition, decomposed into per-column accessors.

    Rows are plain dicts: numpy scalars for numeric columns, a dict of numpy
    array views for nested columns, python objects (str, bytes, dict) for
    everything else. The frame's index (the HATS spatial index) is included
    under its own name.
    """

    def __init__(self, df: pd.DataFrame):
        self._getters: dict[str, Getter] = {}
        if df.index.name is not None:
            index = df.index.to_numpy()
            self._getters[df.index.name] = lambda i: index[i]
        for name in df.columns:
            self._getters[name] = column_getter(df[name])
        self._n_rows = len(df)

    def __len__(self) -> int:
        return self._n_rows

    def row(self, i: int) -> dict[str, Any]:
        return {name: get(i) for name, get in self._getters.items()}
