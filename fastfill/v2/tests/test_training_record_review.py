"""Run provenance and checkpoint/validation association correctness regressions."""
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.io import fingerprint
from fastfill.v2.tests.test_execution import sample
from fastfill.v2.tests.test_supervision import batch as geometry_batch, prediction


def _config():
    return {"model": {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2,
        "decoder_layers": 1, "tiny_hidden_size": 16, "lora_rank": 0},
        "training": {"cpu": True, "steps": 1, "batch_size": 1, "learning_rate": .01,
        "max_length": 4096, "checkpoint_every": 0, "validate_every": 1, "cpu_threads": 1}}


def _data(tmp_path):
    row = sample()
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(row) + "\n")
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps({**row, "provenance": {
        **row["provenance"], "split": "validation", "house_id": "h2"}}) + "\n")
    return data, validation


def _hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _replace_targets(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    replacement = [{**row, "target": {**row["target"], "objects": [{
        **row["target"]["objects"][0], "target_size_local_m": [2., 2., 2.]}]}} for row in rows]
    path.write_text("".join(json.dumps(row) + "\n" for row in replacement))


def test_logged_validation_matches_saved_step_parameters(tmp_path, monkeypatch):
    data, validation = _data(tmp_path)
    models, validation_hashes = [], []
    original_build, original_validation = train.build_model, train._validation
    def build(config):
        model = original_build(config)
        models.append(model)
        return model
    def validate(model, loader, criterion, accelerator):
        validation_hashes.append(_hash(accelerator.unwrap_model(model)))
        return original_validation(model, loader, criterion, accelerator)
    monkeypatch.setattr(train, "build_model", build)
    monkeypatch.setattr(train, "_validation", validate)
    train.run_training(_config(), data, tmp_path / "run", validation=validation)
    assert validation_hashes == [_hash(models[0])] * 2  # full condition and its minimal projection


@pytest.mark.parametrize("which", ["train", "validation"])
def test_changed_input_during_read_rejected_before_run_publication(tmp_path, monkeypatch, which):
    data, validation = _data(tmp_path)
    changed = data if which == "train" else validation
    original_read = train.read_samples
    def read(path, **kwargs):
        rows = original_read(path, **kwargs)
        if Path(path) == changed:
            _replace_targets(path)
        return rows
    monkeypatch.setattr(train, "read_samples", read)
    with pytest.raises((ValueError, RuntimeError), match="changed|fingerprint|hash"):
        train.run_training(_config(), data, tmp_path / "run", validation=validation)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("which", ["train", "validation"])
def test_changed_input_during_training_cannot_publish_final_model(tmp_path, monkeypatch, which):
    data, validation = _data(tmp_path)
    changed = data if which == "train" else validation
    original_validation = train._validation
    def validate(*args):
        result = original_validation(*args)
        _replace_targets(changed)
        return result
    monkeypatch.setattr(train, "_validation", validate)
    with pytest.raises((ValueError, RuntimeError), match="changed|fingerprint|hash"):
        train.run_training(_config(), data, tmp_path / "run", validation=validation)
    assert not (tmp_path / "run/model").exists()
    assert not (tmp_path / "run/run_manifest.json").exists()


@pytest.mark.parametrize("include_validation", [False, True])
def test_manifest_preserves_input_and_optional_validation_fingerprints(tmp_path, include_validation):
    data, validation = _data(tmp_path)
    original_training_hash, original_validation_hash = fingerprint(data), fingerprint(validation)
    run = tmp_path / "run"
    train.run_training(_config(), data, run, validation=validation if include_validation else None)
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["data_sha256"] == original_training_hash
    assert manifest["validation_data_path"] == (str(validation.resolve()) if include_validation else None)
    assert manifest["validation_data_sha256"] == (original_validation_hash if include_validation else None)


class _PredictionModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.predictions = predictions

    def forward(self, **kwargs):
        return self.predictions


def _logging_worker(rank, rendezvous, result_directory, mode):
    from accelerate import Accelerator
    from torch import distributed as dist
    from fastfill.v2.losses import GeometryCriterion, LossConfig
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        accelerator = Accelerator(cpu=True)
        assert accelerator.num_processes == 2
        observations = {}
        for case in ("rank_without_box", "uneven_box_count", "global_zero_box", "rank_without_collision"):
            b = geometry_batch()
            valid = torch.ones((1, 2), dtype=torch.bool)
            if case == "rank_without_box" and rank == 0:
                valid = torch.zeros_like(valid)
            elif case == "uneven_box_count" and rank == 0:
                valid = torch.tensor([[True, False]])
            elif case in {"global_zero_box", "rank_without_collision"}:
                valid = torch.zeros_like(valid)
            b = {**b, "validity": {**b["validity"], "yaw": valid}}
            if case == "rank_without_collision":
                objects = b["objects"][0] if rank else b["objects"][0][:1]
                b = {**b, "objects": [objects], "slot_mask": torch.tensor([[True, bool(rank)]]),
                     "conditions": [{"objects": objects, "constraints": [], "room": {}}]}
            if mode == "validation":  # _validation also scores collapse, which reads the room polygon
                b = {**b, "conditions": [{**c, "room": {"floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]]}}
                                         for c in b["conditions"]]}
            p = prediction(b)
            p = {**p, "position_normalized": b["targets"]["position_normalized"].clone().requires_grad_()}
            criterion = GeometryCriterion(LossConfig(hungarian=False,
                box=0. if case == "rank_without_collision" else 1.,
                collision=1. if case == "rank_without_collision" else 0.))
            result = criterion(p, b)
            if mode == "validation":  # validation shards differ in length: each batch's objective under its own counts
                local = criterion(p, b, global_counts=False)
            gathered = [None, None]
            dist.all_gather_object(gathered, float((result if mode == "record" else local)["loss"].detach()))
            expected = sum(gathered) / 2
            model = _PredictionModel(p)
            if mode == "record":
                result["loss"].backward()
                optimizer = torch.optim.SGD([{"params": [model.anchor], "lr": 0., "name": "decoder"}])
                actual = train._record(train._Window().add(result), model, optimizer, 1, 0., 1, accelerator)["loss"]
            else:
                actual = train._validation(model, [b], criterion, accelerator)["geometry_objective_mean_of_batches"]
                assert model.training
            assert actual == pytest.approx(expected, rel=2e-6, abs=1e-6)
            observations[case] = {"local_dtype": str(result["loss"].dtype),
                                  "expected_global_objective": expected, "actual": actual,
                                  "counts": result["counts"]}
        (Path(result_directory) / f"{mode}-rank-{rank}.json").write_text(json.dumps(observations, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="requires Gloo distributed backend")
@pytest.mark.parametrize("mode", ["record", "validation"])
def test_actual_two_process_mixed_geometry_logging_and_validation(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if __import__("sys").platform == "darwin" else "lo")
    rendezvous = "file://" + str(tmp_path / f"{mode}-rendezvous")
    torch.multiprocessing.spawn(_logging_worker,
        args=(rendezvous, str(tmp_path), mode), nprocs=2, join=True)
    assert all((tmp_path / f"{mode}-rank-{rank}.json").exists() for rank in range(2))
