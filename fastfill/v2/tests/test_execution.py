import json

import pytest
import torch

from fastfill.v2.io import ensure_disjoint, read_samples, safe_output
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.tests.test_supervision import batch, prediction


def sample():
    return {"schema_version": "fastfill.v2", "condition": {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "floor_z_m": 0., "floor_known": True, "height_m": 2.8},
        "objects": [{"id": "a", "category": "chair", "description": "a chair", "support_parent": "floor"}], "constraints": []},
        "target": {"schema_version": "fastfill.v2", "objects": [{"id": "a", "target_size_local_m": [1., 1., 1.],
          "bottom_center_m": [2., 1., 0.], "yaw_rad": 0.}]},
        "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [True], "yaw_symmetry_order": [1]},
        "provenance": {"source": "fixture", "house_id": "h1", "scene_id": "s1", "split": "train"}}


def test_io_source_protection_split_guard(tmp_path):
    with pytest.raises(ValueError):
        safe_output("/Volumes/harddisk/3D_Room_Collections/test-output")
    with pytest.raises(FileExistsError):
        safe_output(tmp_path)
    row = sample()
    with pytest.raises(ValueError):
        ensure_disjoint([row], [row])
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({**row, "provenance": {**row["provenance"], "split": "test"}}) + "\n")
    with pytest.raises(ValueError):
        read_samples(path, training=True)


def test_optional_regularizers_are_separate_and_differentiable():
    b = batch()
    b["conditions"] = [{"objects": b["objects"][0], "constraints": [], "room": {
        "floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]], "boundary_known": True}}]
    p = prediction(b)
    result = GeometryCriterion(LossConfig(collision=1., boundary=1., box=1.))(p, b)
    assert result["collision"] > 0
    assert result["boundary"] > 0
    assert result["box"] > 0
    result["loss"].backward()
    assert torch.isfinite(p["position_normalized"].grad).all()
    b["conditions"][0]["room"]["boundary_known"] = False
    with pytest.raises(ValueError, match="unknown"):
        GeometryCriterion(LossConfig(boundary=1.))(p, b)


def test_train_checkpoint_inference_evaluation_end_to_end(tmp_path):
    from fastfill.v2.train import run_training
    from fastfill.v2.evaluate import run_evaluation
    row = sample()
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    config = {"model": {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2,
               "decoder_layers": 1, "tiny_hidden_size": 16, "lora_rank": 0},
              "training": {"cpu": True, "steps": 20, "batch_size": 1, "learning_rate": .01,
               "max_length": 4096, "checkpoint_every": 0, "validate_every": 0, "cpu_threads": 1}}
    run = tmp_path / "trained"
    logs = run_training(config, path, run)
    assert logs[-1]["loss"] < logs[0]["loss"]
    assert all(logs[0]["gradient_norms"][name] > 0 for name in ("decoder", "backbone", "size_head", "position_head", "yaw_logits_head", "yaw_residual_head"))
    test_path = tmp_path / "test.jsonl"
    test_path.write_text(json.dumps({**row, "provenance": {**row["provenance"], "split": "test", "house_id": "h2"}}) + "\n")
    report = run_evaluation(test_path, tmp_path / "evaluation", checkpoint=run / "model")
    assert report["requests"] == 1
    assert report["model"]["schema_success"] == 1
    assert report["model"]["reference"]["log_size_error"]["mean"] is not None
    assert report["model"]["reference"]["bev_iou"]["valid_objects"] == 1
    with pytest.raises(FileExistsError):
        run_training(config, path, run, dry_run=True)


def test_evaluation_keeps_failed_requests_and_atomic_asset_stages(tmp_path):
    from fastfill.v2.evaluate import run_evaluation
    row = sample()
    data = tmp_path / "evaluation.jsonl"
    data.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    layouts = tmp_path / "predictions.jsonl"
    layouts.write_text(json.dumps(row["target"]) + "\n" + json.dumps({"schema_version": "fastfill.v2", "objects": []}) + "\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"ref": "chair-asset", "category": "chair", "actual_size_local_m": [1, 1, 1],
                                    "semantic_front_local": [1, 0, 0], "capabilities": []}]))
    report = run_evaluation(data, tmp_path / "results", predictions=layouts, catalog=catalog, commit_in_memory=True)
    assert report["requests"] == 2
    assert report["model"]["schema_success"] == .5
    assert report["system"]["final_commit"] == .5
    assert report["failed_requests"] == 1
    assert report["model"]["reference"]["bottom_center_error_m"]["mean"] == 0
    records = [json.loads(line) for line in (tmp_path / "results/outcomes.jsonl").read_text().splitlines()]
    assert records[0]["runtime"]["raw_model_output"] == row["target"]
    assert records[0]["runtime"]["metrics"]["asset"]["first_pass_actual_geometry_validation"]


def test_global_count_compensates_ddp_gradient_average(monkeypatch):
    from fastfill.v2.losses import _mean
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    # Local rank has 1 label; peer has 3: global denominator is 4.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t: t.add_(3))
    local_sum = torch.tensor(6., requires_grad=True)
    value, count = _mean(local_sum, 1)
    assert count == 4
    value.backward()
    assert local_sum.grad.item() == .5  # DDP later averages by 2 -> 1/4 per-label gradient.


def test_invalid_target_geometry_is_failure_with_schema_success():
    from fastfill.v2.evaluate import summarize
    report = summarize([{"model": {"schema_success": True, "requested_ids_exactly_once": True,
                         "positive_valid_size": True, "target_geometry_valid": False},
                         "evaluation_wall_time_ms": 1.}])
    assert report["failed_requests"] == 1
    assert report["inference_failed_requests"] == 0
