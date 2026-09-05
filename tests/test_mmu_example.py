"""Exercise the MMU batch contract with synthetic rows, without remote data."""

import io

import lsdb
import nested_pandas as npd
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from examples.mmu_crossmatch import collate, decode, open_training_catalog
from lsdb_torch import LSDBDataset


def make_row(length=3, image_shape=(4, 5)):
    pixels = np.empty((*image_shape, 3), dtype=np.uint8)
    pixels[:] = [30, 80, 140]
    with io.BytesIO() as encoded:
        Image.fromarray(pixels).save(encoded, format="PNG")
        image_bytes = encoded.getvalue()
    flux = np.arange(length, dtype=np.float32)
    wavelength = np.arange(length, dtype=np.float32) + 4000
    # The adapter exposes read-only views into Arrow-backed partitions.
    flux.flags.writeable = wavelength.flags.writeable = False
    return {
        "_healpix_29": np.int64(123),
        "rgb_image_mmu_gz10": {"bytes": image_bytes},
        "spectrum_mmu_sdss_sdss": {"flux": flux, "lambda": wavelength},
        "redshift_mmu_gz10": np.float32(0.125),
        "gz10_label_mmu_gz10": np.int32(7),
    }


def test_decode_preserves_values_and_owns_spectrum_memory():
    row = make_row()
    sample = decode(row)
    assert sample["spatial_index"] == 123
    assert "id" not in sample
    assert sample["image"].shape == (3, 4, 5)
    assert sample["image"].dtype == torch.uint8
    torch.testing.assert_close(sample["image"][:, 0, 0], torch.tensor([30, 80, 140], dtype=torch.uint8))
    torch.testing.assert_close(sample["flux"], torch.tensor([0, 1, 2], dtype=torch.float32))
    torch.testing.assert_close(sample["wavelength"], torch.tensor([4000, 4001, 4002], dtype=torch.float32))
    sample["flux"][0] = -100
    sample["wavelength"][0] = 1
    assert row["spectrum_mmu_sdss_sdss"]["flux"][0] == 0
    assert row["spectrum_mmu_sdss_sdss"]["lambda"][0] == 4000


def test_dataloader_stacks_images_and_pads_spectra_with_tensor_metadata():
    samples = [decode(make_row(length=2)), decode(make_row(length=4))]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate)
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 4, 5)
    assert batch["image"].dtype == torch.uint8
    assert batch["image"].is_contiguous()
    torch.testing.assert_close(batch["flux"], torch.tensor([[0, 1, 0, 0], [0, 1, 2, 3]], dtype=torch.float32))
    torch.testing.assert_close(
        batch["wavelength"],
        torch.tensor([[4000, 4001, 0, 0], [4000, 4001, 4002, 4003]], dtype=torch.float32),
    )
    torch.testing.assert_close(batch["spectrum_valid"], torch.tensor([[True, True, False, False], [True] * 4]))
    torch.testing.assert_close(batch["spectrum_length"], torch.tensor([2, 4], dtype=torch.int64))
    torch.testing.assert_close(batch["label"], torch.tensor([7, 7], dtype=torch.int64))
    torch.testing.assert_close(batch["redshift"], torch.tensor([0.125, 0.125], dtype=torch.float32))
    torch.testing.assert_close(batch["spatial_index"], torch.tensor([123, 123], dtype=torch.int64))
    assert all(isinstance(value, torch.Tensor) and value.device.type == "cpu" for value in batch.values())


def test_collate_preserves_differently_sized_images():
    batch = collate([decode(make_row(image_shape=(4, 5))), decode(make_row(image_shape=(6, 7)))])
    assert isinstance(batch["image"], list)
    assert [image.shape for image in batch["image"]] == [(3, 4, 5), (3, 6, 7)]
    assert all(image.dtype == torch.uint8 for image in batch["image"])


def test_missing_redshift_remains_nan():
    row = make_row()
    row["redshift_mmu_gz10"] = None
    assert torch.isnan(collate([decode(row)])["redshift"]).all()


@pytest.mark.parametrize(
    "spectrum",
    [
        None,
        {},
        {"flux": [1]},
        {"flux": [1], "lambda": None},
        {"flux": [], "lambda": []},
        {"flux": [1, 2], "lambda": [4000]},
        {"flux": [[1]], "lambda": [[4000]]},
        {"flux": ["bad"], "lambda": [4000]},
    ],
)
def test_decode_rejects_malformed_spectrum_with_row_context(spectrum):
    row = make_row()
    row["spectrum_mmu_sdss_sdss"] = spectrum
    with pytest.raises(ValueError, match="Invalid spectrum at spatial index 123:"):
        decode(row)


def test_decode_rejects_missing_spectrum():
    row = make_row()
    del row["spectrum_mmu_sdss_sdss"]
    with pytest.raises(ValueError, match="expected flux and lambda arrays"):
        decode(row)


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate([])


def test_materialized_catalog_reopens_with_projected_nested_fields(tmp_path):
    rows = [make_row(length=2), make_row(length=4)]
    frame = npd.NestedFrame(
        {
            "ra": [10.0, 10.1],
            "dec": [20.0, 20.1],
            "rgb_image_mmu_gz10": pd.Series(
                [row["rgb_image_mmu_gz10"] for row in rows],
                dtype=pd.ArrowDtype(pa.struct([("bytes", pa.binary())])),
            ),
            "redshift_mmu_gz10": [0.125, 0.125],
            "gz10_label_mmu_gz10": [7, 7],
            "unused_payload": ["skip", "skip"],
        }
    )
    frame["spectrum_mmu_sdss_sdss"] = pd.Series(
        [pd.DataFrame({**row["spectrum_mmu_sdss_sdss"], "unused": 1.0}) for row in rows],
        dtype=npd.NestedDtype.from_columns({"flux": pa.float32(), "lambda": pa.float32(), "unused": pa.float32()}),
    )
    path = tmp_path / "crossmatch"
    lsdb.from_dataframe(frame, margin_threshold=None).write_catalog(path, catalog_name="mmu_crossmatch")

    catalog = open_training_catalog(path)
    assert "unused_payload" not in catalog.columns
    sample = next(iter(LSDBDataset(catalog, shuffle=False)))
    assert set(sample["spectrum_mmu_sdss_sdss"]) == {"flux", "lambda"}
    loader = DataLoader(
        LSDBDataset(catalog, transform=decode, shuffle=False),
        batch_size=2,
        collate_fn=collate,
    )
    batch = next(iter(loader))
    assert batch["image"].shape == (2, 3, 4, 5)
    assert sorted(batch["spectrum_length"].tolist()) == [2, 4]
