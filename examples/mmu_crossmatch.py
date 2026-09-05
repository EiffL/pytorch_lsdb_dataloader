"""Load MMU images and spectra as CPU tensor batches.

Run the lazy remote example with ``python examples/mmu_crossmatch.py``. For
repeated training, prepare the crossmatch once and reuse its HATS catalog::

    python examples/mmu_crossmatch.py --materialize /data/mmu_crossmatch
    python examples/mmu_crossmatch.py --catalog /data/mmu_crossmatch

Image resizing, normalization, and device transfers belong to the training
application. Install the example dependencies with ``pip install -e '.[examples]'``.
"""

import argparse
import io
import time
from collections.abc import Mapping

import lsdb
import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from lsdb_torch import LSDBDataset


def decode(row):
    """Decode a matched row in a worker, taking ownership of read-only arrays."""
    spectrum = row.get("spectrum_mmu_sdss_sdss")
    error = f"Invalid spectrum at spatial index {row['_healpix_29']}: "
    if not isinstance(spectrum, Mapping) or any(spectrum.get(key) is None for key in ("flux", "lambda")):
        raise ValueError(error + "expected flux and lambda arrays")
    try:
        flux = np.array(spectrum["flux"], dtype=np.float32, copy=True)
        wavelength = np.array(spectrum["lambda"], dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(error + "flux and lambda must be numeric arrays") from exc
    if flux.ndim != 1 or wavelength.ndim != 1 or len(flux) == 0 or flux.shape != wavelength.shape:
        raise ValueError(error + "flux and lambda must be nonempty 1-D arrays of equal length")

    with Image.open(io.BytesIO(row["rgb_image_mmu_gz10"]["bytes"])) as image:
        pixels = np.array(image.convert("RGB"), copy=True)
    return {
        # HEALPix is a spatial index, not a unique object identifier.
        "spatial_index": int(row["_healpix_29"]),
        "image": torch.from_numpy(pixels).permute(2, 0, 1),
        "flux": torch.from_numpy(flux),
        "wavelength": torch.from_numpy(wavelength),
        "redshift": np.float32(row["redshift_mmu_gz10"]),
        "label": int(row["gz10_label_mmu_gz10"]),
    }


def collate(samples):
    """Stack equal-size images; pad spectra with a mask marking unpadded values.

    Differently sized images remain a list of CHW tensors for the application
    to resize or bucket. ``spectrum_valid`` describes padding, not survey data
    quality: source values, including any NaNs, are preserved.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    images = [sample["image"] for sample in samples]
    if all(image.shape == images[0].shape for image in images):
        images = torch.stack(images)
    lengths = torch.tensor([len(sample["flux"]) for sample in samples], dtype=torch.int64)
    flux = pad_sequence([sample["flux"] for sample in samples], batch_first=True)
    wavelength = pad_sequence([sample["wavelength"] for sample in samples], batch_first=True)
    return {
        "spatial_index": torch.tensor([sample["spatial_index"] for sample in samples], dtype=torch.int64),
        "image": images,
        "flux": flux,
        "wavelength": wavelength,
        "spectrum_length": lengths,
        "spectrum_valid": torch.arange(flux.shape[1]).unsqueeze(0) < lengths.unsqueeze(1),
        "redshift": torch.tensor([sample["redshift"] for sample in samples], dtype=torch.float32),
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.int64),
    }


def open_training_catalog(path=None):
    """Open an existing crossmatch or construct a projected lazy crossmatch."""
    if path is not None:
        return lsdb.open_catalog(
            path,
            columns=[
                "rgb_image_mmu_gz10",
                "redshift_mmu_gz10",
                "gz10_label_mmu_gz10",
                "spectrum_mmu_sdss_sdss.flux",
                "spectrum_mmu_sdss_sdss.lambda",
            ],
        )
    # Retain coordinates for matching, and read only the payload fields used above.
    gz10 = lsdb.open_catalog(
        "hf://datasets/UniverseTBD/mmu_gz10",
        columns=["ra", "dec", "rgb_image", "redshift", "gz10_label"],
    )
    sdss = lsdb.open_catalog(
        "hf://datasets/UniverseTBD/mmu_sdss_sdss",
        columns=["ra", "dec", "spectrum.flux", "spectrum.lambda"],
    )
    return gz10.crossmatch(
        sdss,
        n_neighbors=1,
        suffix_method="all_columns",
        suffixes=("_mmu_gz10", "_mmu_sdss_sdss"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--catalog", help="Read a crossmatch previously written by --materialize")
    source.add_argument("--materialize", help="Write the crossmatch to a new HATS catalog and exit")
    args = parser.parse_args()

    t0 = time.time()
    catalog = open_training_catalog(args.catalog)
    if args.materialize:
        catalog.write_catalog(args.materialize, catalog_name="mmu_crossmatch")
        print(f"Wrote crossmatch to {args.materialize}")
        raise SystemExit(0)
    print(f"opened lazily in {time.time() - t0:.0f}s, {catalog.npartitions} partitions")

    # A live crossmatch rereads spectra partitions on every pass. Prefer --catalog
    # for repeated training. This example previews batches without a GPU model.
    ds = LSDBDataset(catalog, transform=decode)
    loader = DataLoader(
        ds,
        batch_size=16,
        num_workers=4,
        collate_fn=collate,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
        multiprocessing_context="spawn",
        timeout=1800,
    )

    t0, n = time.time(), 0
    for i, batch in enumerate(loader):
        n += len(batch["spatial_index"])
        if i == 0:
            print(
                "first batch:",
                batch["image"][0].shape,
                batch["flux"][0].shape,
                batch["label"][:4],
            )
        print(f"batch {i}: {n} rows, {n / (time.time() - t0):.2f} rows/s", flush=True)
        if n >= 160:
            break
