"""Actual objective eligibility and optimizer-window regressions."""
from copy import deepcopy
import json

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.batch import TinyTokenizer
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.tests.test_execution import sample


def _yaw_config():
    return {"model": {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2,
        "decoder_layers": 1, "tiny_hidden_size": 16, "lora_rank": 0},
        "loss": {"position": 0., "size": 0., "yaw_cls": 1., "yaw_reg": 1., "hungarian": False},
        "training": {"cpu": True, "steps": 1, "batch_size": 1, "learning_rate": .01,
            "max_length": 4096, "checkpoint_every": 0, "validate_every": 0, "cpu_threads": 1},
        "augmentation": NO_AUGMENTATION}


NO_AUGMENTATION = {"rotate90": False, "mirror": False, "shuffle_objects": False,
                   "drop_constraints_p": 0., "drop_support_p": 0., "category_only_description_p": 0., "minimal_form_p": 0.,
                   "given_pose_p": 0., "given_pose_corrupt_p": 0., "given_pose_noise_m": 0., "given_pose_yaw_noise_deg": 0.,
                   "given_pose_size_noise": 0.}


def _row(identity, yaw=True):
    row = sample()
    return {**row, "validity": {**row["validity"], "yaw": [yaw]},
            "provenance": {**row["provenance"], "scene_id": identity}}


def _data(tmp_path, rows):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_actual_yaw_only_no_labels_fails_before_model_or_run_publication(tmp_path, monkeypatch):
    data = _data(tmp_path, [_row("empty", False)])
    builds = []
    monkeypatch.setattr(train, "build_model", lambda cfg: builds.append(cfg))
    with pytest.raises(ValueError, match="supervised|objective"):
        train.run_training(_yaw_config(), data, tmp_path / "run")
    assert not builds and not (tmp_path / "run").exists()


def test_actual_mixed_density_preflight_records_objective_exclusion(tmp_path):
    data = _data(tmp_path, [_row("empty", False), _row("valid")])
    train.run_training(_yaw_config(), data, tmp_path / "run")
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["supervised_samples"] == 1
    assert manifest["rejected"][0]["reason"] == "no_active_objective"
    assert manifest["rejected"][0]["provenance"]["scene_id"] == "empty"


def test_default_four_terms_still_accept_position_size_without_yaw(tmp_path):
    cfg = {**_yaw_config(), "loss": {"hungarian": False}}
    train.run_training(cfg, _data(tmp_path, [_row("without-yaw", False)]), tmp_path / "run")
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["supervised_samples"] == 1 and manifest["steps_completed"] == 1
    log = json.loads((tmp_path / "run/training_log.json").read_text())[0]
    assert log["active_objective_count"] == 2


@pytest.mark.parametrize("kind", ["fixed_size", "partial_position", "box_without_yaw", "collision_singleton"])
def test_preflight_uses_enabled_terms_and_complete_learning_coordinates(kind):
    row = _row(kind, False)
    weights = {key: 0. for key in ("position", "size", "yaw_cls", "yaw_reg")}
    if kind == "fixed_size":
        weights["size"] = 1.
        row["condition"]["objects"][0]["fixed_size_local_m"] = [1., 1., 1.]
    elif kind == "partial_position":
        weights["position"] = 1.
        row["validity"]["position"] = [[True, True, False]]
    elif kind == "box_without_yaw":
        weights["box"] = 1.
    else:
        weights["collision"] = 1.
    with pytest.raises(ValueError, match="supervised|objective"):
        train._preflight([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0),
                         {"max_length": 4096}, LossConfig(**weights))


def _dynamic_batches(monkeypatch, invalid_ids):
    """Inject missing labels after preflight, exercising the defensive loop guard."""
    original_collate, original_preflight, original_loader = train.collate_samples, train._preflight, train.DataLoader
    state = {"after_preflight": False}
    def collate(rows, *args, **kwargs):
        batch = original_collate(rows, *args, **kwargs)
        if state["after_preflight"]:
            batch = {**batch, "validity": {**batch["validity"], "yaw": batch["validity"]["yaw"].clone()}}
            for index, row in enumerate(rows):
                if row["provenance"]["scene_id"] in invalid_ids:
                    batch["validity"]["yaw"][index] = False
        return batch
    def preflight(*args, **kwargs):
        state["after_preflight"] = False
        result = original_preflight(*args, **kwargs)
        state["after_preflight"] = True
        return result
    def loader(rows, **kwargs):
        return original_loader(rows, **{**kwargs, "shuffle": False})
    monkeypatch.setattr(train, "collate_samples", collate)
    monkeypatch.setattr(train, "_preflight", preflight)
    monkeypatch.setattr(train, "DataLoader", loader)


def test_accumulation_preserves_prior_valid_gradient_when_last_microbatch_empty(tmp_path, monkeypatch):
    _dynamic_batches(monkeypatch, {"last-empty"})
    cfg = _yaw_config()
    cfg["training"]["gradient_accumulation_steps"] = 2
    logs = train.run_training(cfg, _data(tmp_path, [_row("first-valid"), _row("last-empty")]), tmp_path / "run")
    assert logs[0]["active_objective_count"] == 1
    assert logs[0]["gradient_norms"]["yaw_logits_head"] > 0
    assert logs[0]["step"] == 1


def test_globally_empty_window_does_not_decay_or_increment_update(tmp_path, monkeypatch):
    _dynamic_batches(monkeypatch, {"empty-a", "empty-b"})
    cfg = _yaw_config()
    cfg["training"]["gradient_accumulation_steps"] = 2
    models, before = [], []
    original_build = train.build_model
    def build(config):
        model = original_build(config)
        models.append(model)
        before.append(deepcopy(model.state_dict()))
        return model
    monkeypatch.setattr(train, "build_model", build)
    data = _data(tmp_path, [_row("empty-a"), _row("empty-b")])
    with pytest.raises(RuntimeError, match="no active objective|no supervised optimizer"):
        train.run_training(cfg, data, tmp_path / "run")
    assert all(torch.equal(before[0][name], value) for name, value in models[0].state_dict().items())
    assert not (tmp_path / "run/model").exists()


def test_empty_window_is_skipped_before_later_valid_window(tmp_path, monkeypatch):
    _dynamic_batches(monkeypatch, {"empty-a", "empty-b"})
    cfg = _yaw_config()
    cfg["training"]["gradient_accumulation_steps"] = 2
    rows = [_row(name) for name in ("empty-a", "empty-b", "valid-a", "valid-b")]
    logs = train.run_training(cfg, _data(tmp_path, rows), tmp_path / "run")
    assert len(logs) == 1 and logs[0]["step"] == 1
    assert logs[0]["active_objective_count"] == 2
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert manifest["skipped_no_objective_windows"] == 1


def test_zero_numeric_loss_with_eligible_supervision_still_updates(tmp_path, monkeypatch):
    # With K=1, CE is exactly zero. Zero residual weights predict the real GT
    # residual exactly, so the unmodified criterion returns zero numeric loss.
    original_build = train.build_model
    models, before = [], []
    def build(config):
        model = original_build(config)
        with torch.no_grad():
            model.yaw_residual_head.weight.zero_()
            model.yaw_residual_head.bias.zero_()
        models.append(model)
        before.append(deepcopy(model.state_dict()))
        return model
    monkeypatch.setattr(train, "build_model", build)
    config = _yaw_config()
    config["model"]["yaw_bins"] = 1
    logs = train.run_training(config, _data(tmp_path, [_row("valid")]), tmp_path / "run")
    assert logs[0]["loss"] == 0 and logs[0]["active_objective_count"] == 1
    assert logs[0]["step"] == 1
    assert any(not torch.equal(before[0][name], value) for name, value in models[0].state_dict().items())


def test_short_epoch_flushes_eligible_partial_accumulation_window(tmp_path):
    config = _yaw_config()
    config["training"]["gradient_accumulation_steps"] = 3
    logs = train.run_training(config, _data(tmp_path, [_row("valid")]), tmp_path / "run")
    assert len(logs) == 1 and logs[0]["step"] == 1
    assert logs[0]["active_objective_count"] == 1
    assert logs[0]["gradient_norms"]["yaw_logits_head"] > 0


@pytest.mark.parametrize("term", ["boundary", "collision"])
def test_regularizer_only_preflight_keeps_label_free_eligible_geometry(term):
    row = _row("label-free", False)
    row["validity"] = {"position": [[False] * 3], "size": [[False] * 3], "yaw": [False]}
    if term == "collision":
        row["condition"]["objects"].append({"id": "b", "category": "chair", "description": "a chair"})
    config = LossConfig(position=0., size=0., yaw_cls=0., yaw_reg=0., **{term: 1.})
    kept, rejected = train._preflight([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0),
                                     {"max_length": 4096}, config)
    assert kept == [row] and not rejected


def test_collision_only_support_exempt_pair_has_no_eligible_term():
    row = _row("supported", False)
    row["condition"]["objects"].append({"id": "b", "category": "cup", "description": "a cup", "support_parent": "a"})
    config = LossConfig(position=0., size=0., yaw_cls=0., yaw_reg=0., collision=1.)
    with pytest.raises(ValueError, match="supervised|objective"):
        train._preflight([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0), {"max_length": 4096}, config)
