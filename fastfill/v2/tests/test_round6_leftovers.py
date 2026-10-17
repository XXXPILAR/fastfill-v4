"""Round 6 audit leftovers: per-rank diagnostics (global_counts=False) reach the scene regularizers too."""
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.tests.test_active_objective_training import _data
from fastfill.v2.tests.test_round5_train import _validation_file
from fastfill.v2.tests.test_supervision import batch, prediction
from fastfill.v2.tests.test_train_loop_fix20261006 import _config, _rows


def test_local_counts_reach_the_scene_regularizers_and_the_default_stays_global(monkeypatch):
    """Validation ranks run different numbers of batches: a per-call all_reduce in collision/boundary would pair
    different batches across ranks. Training (the default) still normalises every term by the global count."""
    calls = []

    def all_reduce(count):  # another rank holds 3 more objects
        calls.append(int(count))
        count += 3
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    b = batch()
    b["conditions"] = [{"objects": b["objects"][0], "constraints": [],
                        "room": {"floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]]}}]
    criterion = GeometryCriterion(LossConfig(collision=1., boundary=1.))
    local = criterion(prediction(b), b, global_counts=False)
    assert calls == []
    for key in ("collision", "boundary"):
        assert local[key].item() == pytest.approx(local["term_sums"][key].item() / local["term_counts"][key])
    assert local["term_counts"]["collision"] and local["term_counts"]["boundary"]
    result = criterion(prediction(b), b)  # 6d494a9's values: 5 matched terms + collision + boundary all_reduce
    assert len(calls) == 7
    assert result["loss"].item() == pytest.approx(2.453782639155785, rel=1e-6)
    assert result["collision"].item() == pytest.approx(0.2500000062088169, rel=1e-6)
    assert result["boundary"].item() == pytest.approx(0.07000000774860382, rel=1e-6)


def _regularizer_worker(rank, rendezvous, directory):
    from torch import distributed as dist
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29561")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        root = Path(directory)
        config = {**_config(steps=1, validate_every=1), "loss": {"hungarian": False, "collision": 1., "boundary": 1.}}
        train.run_training(config, root / "train.jsonl", root / "run", validation=root / "validation.jsonl")
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires Gloo")
def test_two_rank_validation_with_scene_regularizers_survives_a_shorter_shard(tmp_path, monkeypatch):
    """3 validation rows: rank 1 runs one batch fewer. Before, collision/boundary all_reduced per batch and gloo aborted."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    _data(tmp_path, _rows(2))
    _validation_file(tmp_path / "validation.jsonl", 3)
    torch.multiprocessing.spawn(_regularizer_worker, args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
                                nprocs=2, join=True)
    validation = json.loads((tmp_path / "run" / "training_log.json").read_text())[-1]["validation"]
    assert validation["batches"] == 3 and validation["counts"]["boundary"] == 3
