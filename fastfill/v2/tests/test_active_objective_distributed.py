"""Real Accelerator/Gloo entry-point tests with empty ranks and windows."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from fastfill.v2.tests.test_active_objective_training import _row, _yaw_config


def _worker(rank, rendezvous, directory, mode):
    from torch import distributed as dist
    from fastfill.v2 import train
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29513")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            # torch 2.13 barrier(device_ids=[...]) chooses the available MPS
            # accelerator even for a CPU/Gloo group. This CPU test explicitly
            # selects CPU discovery; collectives/backward remain real Gloo.
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        root = Path(directory)
        rows_count = {"peer_empty": 2, "last_empty": 4, "empty_then_valid": 8, "all_empty": 4}[mode]
        data = root / "train.jsonl"
        if rank == 0:
            data.write_text("".join(json.dumps(_row(str(i))) + "\n" for i in range(rows_count)))
        dist.barrier()
        original_collate, original_preflight, original_loader = train.collate_samples, train._preflight, train.DataLoader
        state = {"after_preflight": False}
        def collate(rows, *args, **kwargs):
            batch = original_collate(rows, *args, **kwargs)
            if state["after_preflight"]:
                yaw = batch["validity"]["yaw"].clone()
                for i, row in enumerate(rows):
                    identity = int(row["provenance"]["scene_id"])
                    empty = (mode == "all_empty" or (mode == "peer_empty" and rank == 0)
                             or (mode == "last_empty" and identity >= 2)
                             or (mode == "empty_then_valid" and identity < 4))
                    if empty:
                        yaw[i] = False
                batch = {**batch, "validity": {**batch["validity"], "yaw": yaw}}
            return batch
        def preflight(*args, **kwargs):
            state["after_preflight"] = False
            result = original_preflight(*args, **kwargs)
            state["after_preflight"] = True
            return result
        def loader(rows, **kwargs):
            return original_loader(rows, **{**kwargs, "shuffle": False})
        models, before = [], []
        original_build = train.build_model
        def build(config):
            model = original_build(config)
            models.append(model)
            before.append(deepcopy(model.state_dict()))
            return model
        patches.setattr(train, "collate_samples", collate)
        patches.setattr(train, "_preflight", preflight)
        patches.setattr(train, "DataLoader", loader)
        patches.setattr(train, "build_model", build)
        config = _yaw_config()
        config["training"]["gradient_accumulation_steps"] = 1 if mode == "peer_empty" else 2
        if mode == "all_empty":
            with pytest.raises(RuntimeError, match="no active objective"):
                train.run_training(config, data, root / "run")
            changed = sum(not torch.equal(before[0][name], value) for name, value in models[0].state_dict().items())
            assert changed == 0
            observation = {"optimizer_updates": 0, "changed_parameter_tensors": changed}
        else:
            logs = train.run_training(config, data, root / "run")
            assert len(logs) == 1 and logs[0]["step"] == 1
            assert logs[0]["active_objective_count"] == {"peer_empty": 1, "last_empty": 2, "empty_then_valid": 4}[mode]
            assert logs[0]["gradient_norms"]["yaw_logits_head"] > 0
            changed = sum(not torch.equal(before[0][name], value) for name, value in models[0].state_dict().items())
            assert changed > 0
            observation = {"optimizer_updates": 1, "active_objective_count": logs[0]["active_objective_count"],
                           "changed_parameter_tensors": changed}
            if rank == 0:
                manifest = json.loads((root / "run/run_manifest.json").read_text())
                assert manifest["world_size"] == 2
                assert manifest["skipped_no_objective_windows"] == int(mode == "empty_then_valid")
        (root / f"rank-{rank}.json").write_text(json.dumps(observation, indent=2) + "\n")
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("mode", ["peer_empty", "last_empty", "empty_then_valid", "all_empty"])
def test_actual_two_rank_objective_windows(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CPU Gloo ranks: with one visible GPU, rank 1 would ask for cuda:1
    torch.multiprocessing.spawn(_worker, args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path), mode),
                               nprocs=2, join=True)
    assert all((tmp_path / f"rank-{rank}.json").exists() for rank in range(2))
