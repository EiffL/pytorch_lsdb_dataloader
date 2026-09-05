"""Check coverage and CPU DDP training with two local ranks.

Run ``torchrun --standalone --nproc_per_node=2 examples/ddp_smoke.py``.
For multi-node torchrun, pass ``--catalog-dir /existing/shared/directory``;
all ranks must see this directory at the same path. Test data uses a unique
temporary subdirectory, removed when the check finishes.
"""

import argparse
import tempfile
from pathlib import Path

import lsdb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from lsdb_torch import LSDBDataset


def index_only(row):
    return int(row["_healpix_29"])


def features(row):
    return torch.tensor([row["ra"] / 360.0, row["dec"] / 90.0], dtype=torch.float32)


def check_coverage(catalog):
    ds = LSDBDataset(catalog, transform=index_only)
    ds.set_epoch(0)
    mine = list(DataLoader(ds, batch_size=None, num_workers=1, multiprocessing_context="spawn"))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, mine)
    if dist.get_rank() == 0:
        seen = sum(gathered, [])
        expected = catalog.compute(progress_bar=False).index.to_list()
        assert sorted(seen) == sorted(expected), "shards do not cover the catalog exactly once"
        print(f"ok: {len(seen)} rows over {dist.get_world_size()} ranks, per rank {[len(g) for g in gathered]}")


def check_training(catalog):
    torch.manual_seed(1)
    model = DistributedDataParallel(torch.nn.Linear(2, 1))
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()]).clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ds = LSDBDataset(catalog[["ra", "dec"]], transform=features, loop=True)
    ds.set_epoch(0)
    loader = DataLoader(ds, batch_size=16, num_workers=1, multiprocessing_context="spawn")
    iterator = iter(loader)
    steps = 5
    try:
        for _ in range(steps):
            batch = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            target = batch.sum(dim=1, keepdim=True)
            torch.nn.functional.mse_loss(prediction, target).backward()
            optimizer.step()
    finally:
        # Releasing the nonpersistent iterator shuts its worker down before catalog cleanup.
        del iterator

    parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    assert torch.isfinite(parameters).all(), "training produced nonfinite parameters"
    assert not torch.equal(parameters, initial), "optimizer did not update the model"
    gathered = [torch.empty_like(parameters) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, parameters)
    for other in gathered:
        torch.testing.assert_close(parameters, other, rtol=0, atol=0)
    if dist.get_rank() == 0:
        print(f"ok: {steps} DDP forward/backward/Adam steps; model parameters agree across ranks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        help="Existing shared parent directory for temporary test data",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    temporary = None
    try:
        paths = [None]
        if dist.get_rank() == 0:
            temporary = tempfile.TemporaryDirectory(prefix="lsdb-torch-ddp-", dir=args.catalog_dir)
            paths[0] = str(Path(temporary.name) / "catalog")
            world_size = dist.get_world_size()
            order = 0
            while 12 * 4**order < 2 * world_size:
                order += 1
            lsdb.generate_catalog(max(2000, 32 * world_size), 10, seed=1, lowest_order=order).write_catalog(
                paths[0], catalog_name="smoke"
            )
        dist.broadcast_object_list(paths, src=0)
        catalog = lsdb.open_catalog(paths[0])
        check_coverage(catalog)
        check_training(catalog)
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
