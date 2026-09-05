import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from nested_pandas.series.ext_array import NestedExtensionArray

from lsdb_torch._rows import ColumnarPartition, copy_row


def nested_partition(chunked=False):
    values = pa.StructArray.from_arrays(
        [
            pa.array([99.0, 1.0, 2.0, 90.0, 91.0, 3.0, 100.0]),
            pa.array([9, 1, 2, 9, 9, 3, 9]),
        ],
        names=["flux", "band"],
    )
    # The null list deliberately spans nonempty values, which must not escape its mask.
    array = pa.LargeListArray.from_arrays(
        pa.array([0, 1, 3, 5, 5, 6, 7]),
        values,
        mask=pa.array([False, False, True, False, False, False]),
    )
    selected = pa.chunked_array([array.slice(1, 2), array.slice(3, 2)]) if chunked else array.slice(1, 4)
    frame = pd.DataFrame({"nested": pd.Series(NestedExtensionArray(selected))})
    frame.index = pd.Index([17, 18, 19, 20], name="source_id")
    return ColumnarPartition(frame)


@pytest.mark.parametrize("chunked", [False, True])
def test_nested_rows_preserve_null_empty_and_sliced_values(chunked):
    partition = nested_partition(chunked)

    assert len(partition) == 4
    assert partition.row(0)["source_id"] == 17
    np.testing.assert_array_equal(partition.row(0)["nested"]["flux"], [1.0, 2.0])
    np.testing.assert_array_equal(partition.row(0)["nested"]["band"], [1, 2])
    assert partition.row(1)["nested"] is None
    empty = partition.row(2)["nested"]
    assert set(empty) == {"flux", "band"}
    assert all(isinstance(value, np.ndarray) and value.size == 0 for value in empty.values())
    np.testing.assert_array_equal(partition.row(3)["nested"]["flux"], [3.0])
    np.testing.assert_array_equal(partition.row(3)["nested"]["band"], [3])


@pytest.mark.parametrize("rows", [[], [None, None]])
def test_empty_or_all_null_nested_partition(rows):
    array = pa.array(rows, type=pa.large_list(pa.struct([("flux", pa.float64())])))
    partition = ColumnarPartition(pd.DataFrame({"nested": pd.Series(NestedExtensionArray(array))}))

    assert len(partition) == len(rows)
    assert [partition.row(i)["nested"] for i in range(len(partition))] == rows


def test_copy_row_detaches_nested_partition_views():
    row = nested_partition().row(0)
    owned = copy_row(row)

    for field, original in row["nested"].items():
        result = owned["nested"][field]
        np.testing.assert_array_equal(result, original)
        assert not np.shares_memory(result, original)
        assert result.flags.owndata and result.flags.writeable
    assert owned["source_id"] is row["source_id"]


def test_copy_row_recurses_through_containers_and_preserves_other_values():
    source = np.arange(20).reshape(4, 5)
    source.setflags(write=False)
    marker = object()
    row = {
        "values": [source[1], {"pair": (source[2, ::2], None)}],
        "marker": marker,
        "bytes": b"image",
    }

    owned = copy_row(row)

    assert owned is not row and owned["values"] is not row["values"]
    assert isinstance(owned["values"][1]["pair"], tuple)
    assert owned["marker"] is marker and owned["bytes"] is row["bytes"]
    assert owned["values"][1]["pair"][1] is None
    for original, result in [
        (row["values"][0], owned["values"][0]),
        (row["values"][1]["pair"][0], owned["values"][1]["pair"][0]),
    ]:
        np.testing.assert_array_equal(result, original)
        assert not np.shares_memory(result, original)
        result[0] = -1
        assert original[0] != -1


def test_copy_row_detaches_arrays_inside_object_arrays():
    source = np.arange(100)
    objects = np.empty(2, dtype=object)
    objects[0] = source[2:4]
    objects[1] = {"values": source[8:10]}
    owned = copy_row({"objects": objects})["objects"]
    assert not np.shares_memory(owned, objects)
    for original, copied in [(objects[0], owned[0]), (objects[1]["values"], owned[1]["values"])]:
        np.testing.assert_array_equal(original, copied)
        assert not np.shares_memory(original, copied)
