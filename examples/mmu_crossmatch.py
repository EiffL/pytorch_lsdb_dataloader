"""The user story: crossmatch two Hugging Face HATS catalogs and stream them into a DataLoader."""

import io
import time

import lsdb
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from lsdb_torch import LSDBDataset


def decode(row):
    """Runs inside the DataLoader worker."""
    image = np.asarray(Image.open(io.BytesIO(row["rgb_image_mmu_gz10"]["bytes"])).convert("RGB"))
    spectrum = row["spectrum_mmu_sdss_sdss"]
    return {
        "id": row["_healpix_29"],
        "image": image,
        "flux": spectrum["flux"].astype(np.float32),
        "wavelength": spectrum["lambda"].astype(np.float32),
        "redshift": np.float32(row["redshift_mmu_gz10"]),
        "label": row["gz10_label_mmu_gz10"],
    }


def collate(samples):
    """Spectra and images have varying lengths/sizes, so keep them as lists."""
    return {key: [s[key] for s in samples] for key in samples[0]}


if __name__ == "__main__":
    t0 = time.time()
    gz10 = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_gz10")
    sdss = lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_sdss_sdss")
    xmatch = gz10.crossmatch(sdss, n_neighbors=1, suffix_method="all_columns")
    print(f"opened and crossmatched lazily in {time.time() - t0:.0f}s, {xmatch.npartitions} partitions")

    # Each crossmatch partition downloads a whole SDSS spectra partition (several MB) to match a handful
    # of gz10 rows, so throughput is bound by bandwidth. A shuffle buffer would delay the first batch by the
    # time it takes to fill it; leave it off here (or materialize the crossmatch with write_catalog first).
    ds = LSDBDataset(xmatch, transform=decode)
    loader = DataLoader(ds, batch_size=16, num_workers=4, collate_fn=collate, persistent_workers=True, timeout=1800)

    t0, n = time.time(), 0
    for i, batch in enumerate(loader):
        n += len(batch["id"])
        if i == 0:
            print("first batch:", batch["image"][0].shape, batch["flux"][0].shape, batch["label"][:4])
        print(f"batch {i}: {n} rows, {n / (time.time() - t0):.2f} rows/s", flush=True)
        if n >= 160:
            break
