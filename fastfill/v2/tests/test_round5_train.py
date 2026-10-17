"""Round 5: diagnostics count every row once across ranks (T1), never draw from the global RNG (T2), and the grid
matching cost and float64 reductions run on Metal (T3)."""
from datetime import timedelta
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from fastfill.v2 import train
from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.evaluate import COLLAPSE_KEYS, project_minimal
from fastfill.v2.io import to_device
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.model import load_model
from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data
from fastfill.v2.tests.test_round4_matching import COUNTEREXAMPLE, flip_slots, grid_prediction, group_batch
from fastfill.v2.tests.test_train_loop_fix20261006 import _Killed, _config, _killing_accelerator, _rows


def _validation_file(path, count):
    rows = _rows(count)
    for row in rows:
        row["provenance"].update(split="validation", house_id="val-house")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def _close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _close(actual[key], expected[key])
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)
    else:
        assert actual == expected


def _diagnostics_worker(rank, rendezvous, directory):
    from torch import distributed as dist
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29541")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        root = Path(directory)
        train.run_training(_config(steps=1, validate_every=1), root / "train.jsonl", root / "run",
                           validation=root / "validation.jsonl")
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("count", [1, 3])  # rank 1's shard is empty / one batch shorter than rank 0's
def test_two_rank_diagnostics_count_every_row_once(tmp_path, monkeypatch, count):
    """Accelerate's even batches repeated rank 0's leading rows on rank 1 (3 rows counted 4 times); per-batch
    collectives (DDP buffer broadcast, the criterion's count all_reduce) would pair different batches or hang."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    train_rows = _rows(2)
    _data(tmp_path, train_rows)
    rows = _validation_file(tmp_path / "validation.jsonl", count)
    torch.multiprocessing.spawn(_diagnostics_worker, args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
                                nprocs=2, join=True)
    run = tmp_path / "run"
    logged = json.loads((run / "training_log.json").read_text())[-1]["validation"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    # One process scores the same rows with the validated weights (steps=1: the final export).
    from accelerate import Accelerator
    accelerator, model = Accelerator(cpu=True), load_model(run / "model")
    loader = lambda samples: DataLoader(samples, batch_size=1, collate_fn=partial(collate_samples, tokenizer=TinyTokenizer()))
    criterion, minimal = GeometryCriterion(LossConfig(hungarian=False)), [project_minimal(row) for row in rows]
    full = train._validation(model, loader(rows), criterion, accelerator)
    assert full["counts"]["position"] == full["batches"] == full["collapse"]["objects"] == count
    _close({key: logged[key] for key in full}, full)
    _close(logged["minimal"], train._validation(model, loader(minimal), criterion, accelerator))
    baseline = train._baseline(loader(rows), criterion, accelerator, train._category_median_sizes(train_rows),
                               model.config.yaw_bins)
    _close(manifest["baselines"], {**baseline, "rows": count})
    _close(manifest["selection_metric"]["ground_truth"], train._label_collapse(loader(minimal), accelerator))


def _old_state_worker(rank, rendezvous, directory):
    from torch import distributed as dist
    import accelerate
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29542")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        patches.setattr(accelerate.Accelerator, "end_training", lambda self: None)  # three runs share one Gloo group
        root = Path(directory)
        config = _config(steps=2, checkpoint_every=1)
        run = lambda name, state=None: train.run_training(
            {**config, "training": {**config["training"], "resume": state and str(state)}},
            root / "train.jsonl", root / name, validation=root / "validation.jsonl")
        original = accelerate.Accelerator
        patches.setattr(accelerate, "Accelerator", _killing_accelerator(1))
        with pytest.raises(_Killed):
            run("killed")
        patches.setattr(accelerate, "Accelerator", original)
        state = root / "killed/state-step-1"
        if rank == 0:  # the same state as saved before round 5 (the running c750c31 training's)
            shutil.copytree(state, root / "old-state")
            path = root / "old-state/custom_checkpoint_0.pkl"
            torch.save({k: v for k, v in torch.load(path, weights_only=False).items() if k != "diagnostics"}, path)
        dist.barrier()
        run("old", root / "old-state")  # plain resume, as autorun's crash restart
        run("new", state)
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires Gloo")
def test_two_rank_resume_of_a_state_before_round5_restarts_selection(tmp_path, monkeypatch):
    """A crashed c750c31 run resumed by this code mixes padded (old) and unpadded validation scores in one log:
    recorded and kept out of selection, without --allow-resume-change; this code's own states resume as before."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    _data(tmp_path, _rows(4))
    _validation_file(tmp_path / "validation.jsonl", 3)
    torch.multiprocessing.spawn(_old_state_worker, args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path)),
                                nprocs=2, join=True)
    manifests = lambda run: [json.loads((tmp_path / run / name).read_text()) for name in ("run_manifest_start.json", "run_manifest.json")]
    change = {"diagnostics": {"saved": None, "current": train.DIAGNOSTICS}}
    assert [m["resumed_with_changes"] for m in manifests("old")] == [change, change]
    assert manifests("old")[1]["selection_metric"]["best_after_step"] == 1 and manifests("old")[1]["selection_metric"]["best"]["step"] == 2
    saved = torch.load(tmp_path / "old/state-step-2/custom_checkpoint_0.pkl", weights_only=False)
    assert saved["diagnostics"] == train.DIAGNOSTICS and saved["selection_after_step"] == 1  # later plain resumes keep it
    assert [m["resumed_with_changes"] for m in manifests("new")] == [{}, {}]
    assert manifests("new")[1]["selection_metric"]["best_after_step"] == 0


def test_resume_with_dropout_and_validation_matches_the_uninterrupted_run(tmp_path, monkeypatch):
    """The baseline loader drew its base seed from the restored global RNG, shifting the resumed dropout stream."""
    import accelerate
    data = _data(tmp_path, _rows(4))
    validation = tmp_path / "validation.jsonl"
    _validation_file(validation, 2)
    config = _config(steps=4, checkpoint_every=2, validate_every=2)
    config["model"]["dropout"], config["augmentation"] = .2, NO_AUGMENTATION
    full = train.run_training(config, data, tmp_path / "full", validation=validation)
    original = accelerate.Accelerator
    monkeypatch.setattr(accelerate, "Accelerator", _killing_accelerator(2))
    with pytest.raises(_Killed):
        train.run_training(config, data, tmp_path / "killed", validation=validation)
    monkeypatch.setattr(accelerate, "Accelerator", original)
    resumed = train.run_training({**config, "training": {**config["training"], "resume": str(tmp_path / "killed/state-step-2")}},
                                 data, tmp_path / "resumed", validation=validation)
    assert [r["loss"] for r in full] == [r["loss"] for r in resumed]
    a, b = (torch.load(tmp_path / name / "model/geometry_model.pt", weights_only=True) for name in ("full", "resumed"))
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_declared_window_refuses_a_term_an_empty_shard_would_not_reduce():
    from accelerate import Accelerator
    result = {"loss": torch.tensor(1.), "term_sums": {"extra": torch.tensor(1.)}, "term_counts": {"extra": 1}}
    with pytest.raises(RuntimeError, match="outgrow"):
        train._Window(train.DIAGNOSTIC_TERMS).add(result).summary(Accelerator(cpu=True), shards=True)


class _Metal:
    """The Accelerate surface the reductions use on Metal: one process, so reduce is the identity."""
    device, num_processes = torch.device("mps"), 1

    def reduce(self, tensor, reduction):
        return tensor.clone()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal")
def test_metal_grid_matching_and_float64_reductions_match_cpu():
    b, p, criterion = group_batch([[.2, .3, 0.], [.8, .3, 0.]]), grid_prediction(COUNTEREXAMPLE), GeometryCriterion()
    for prediction in (p, flip_slots(p)):  # grid pair cost plus the size cost of the exchangeable pair
        cpu, metal = criterion(prediction, b), criterion(to_device(prediction, "mps"), to_device(b, "mps"))
        assert metal["assignment"].device.type == "mps" and torch.equal(metal["assignment"].cpu(), cpu["assignment"])
        assert metal["loss"].item() == pytest.approx(cpu["loss"].item(), rel=1e-6)
    summary = train._Window(train.DIAGNOSTIC_TERMS).add(cpu).summary(_Metal(), shards=True)
    assert summary["loss"] == pytest.approx(cpu["loss"].item()) and summary["microbatches"] == 1
    assert train._reduce_collapse(dict.fromkeys(COLLAPSE_KEYS, 1.), _Metal())["objects"] == 1
