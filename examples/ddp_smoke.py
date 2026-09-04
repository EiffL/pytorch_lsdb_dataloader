"""Check sharding under a real process group: torchrun --nproc_per_node=2 examples/ddp_smoke.py"""

import tempfile

import lsdb
import torch.distributed as dist
from torch.utils.data import DataLoader

from lsdb_torch import LSDBDataset


def index_only(row):
    return int(row["_healpix_29"])


if __name__ == "__main__":
    dist.init_process_group("gloo")
    path = tempfile.gettempdir() + "/lsdb_torch_ddp_smoke"
    if dist.get_rank() == 0:
        lsdb.generate_catalog(2000, 10, seed=1).write_catalog(path, catalog_name="smoke", overwrite=True)
    dist.barrier()
    catalog = lsdb.open_catalog(path)

    ds = LSDBDataset(catalog, transform=index_only)
    mine = list(DataLoader(ds, batch_size=None, num_workers=2))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    if dist.get_rank() == 0:
        seen = sum(gathered, [])
        expected = catalog.compute(progress_bar=False).index.to_list()
        assert sorted(seen) == sorted(expected), "shards do not cover the catalog exactly once"
        print(f"ok: {len(seen)} rows over {dist.get_world_size()} ranks, per rank {[len(g) for g in gathered]}")
    dist.destroy_process_group()
