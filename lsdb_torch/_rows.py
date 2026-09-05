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
    """Access nested rows as array views, preserving null lists as None."""
    lists = series.array.list_array.combine_chunks()
    offsets = np.asarray(lists.offsets)
    nulls = lists.is_null().to_numpy(zero_copy_only=False) if lists.null_count else None
    structs = lists.values
    flat = {name: structs.field(name).to_numpy(zero_copy_only=False) for name in structs.type.names}

    def get(i: int) -> dict[str, np.ndarray] | None:
        if nulls is not None and nulls[i]:
            return None
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


def copy_row(row: Any) -> Any:
    """Copy NumPy arrays through dicts, lists and tuples; retain other values."""
    if isinstance(row, np.ndarray):
        copied = row.copy()
        if row.dtype == object:
            for index in np.ndindex(row.shape):
                copied[index] = copy_row(row[index])
        return copied
    if isinstance(row, dict):
        return {key: copy_row(value) for key, value in row.items()}
    if isinstance(row, list):
        return [copy_row(value) for value in row]
    if isinstance(row, tuple):
        return tuple(copy_row(value) for value in row)
    return row


class ColumnarPartition:
    """A computed partition, decomposed into per-column accessors.

    Rows are plain dicts: numpy scalars for numeric columns, a dict of numpy
    array views for nested columns (None for null rows), python objects (str,
    bytes, dict) for everything else. Empty nested rows contain empty arrays.
    The frame's index (the HATS spatial index) is included under its own name.
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
