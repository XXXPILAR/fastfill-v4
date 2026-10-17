"""Trainer fixes (2026-10-06): config sections, warmup/cosine, parameter groups,
resume, bf16 nonfinite guard, window logging, per-checkpoint exports, flag-filtered
validation, baselines and group certification in preflight."""
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.batch import TinyTokenizer
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig, build_model, load_model
from fastfill.v2.tests.test_accumulation_tail import _patch_scalar_training
from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data, _row, _yaw_config
from fastfill.v2.tests.test_execution import sample


def _config(**training):
    return {"model": {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2, "decoder_layers": 1,
                      "tiny_hidden_size": 16, "lora_rank": 0},
            "loss": {"hungarian": False},
            "training": {"cpu": True, "steps": 6, "batch_size": 1, "learning_rate": .01, "max_length": 4096,
                         "checkpoint_every": 0, "validate_every": 0, "cpu_threads": 1, **training}}


def _rows(count):
    rows = []
    for index in range(count):
        row = sample()
        row["target"]["objects"][0].update(bottom_center_m=[1. + .5 * index, 1. + .25 * index, 0.], yaw_rad=.3 * index)
        row["provenance"] = {**row["provenance"], "scene_id": f"s{index}"}
        rows.append(row)
    return rows


def _digest(scene_ids):
    return hashlib.sha256(json.dumps(scene_ids).encode()).hexdigest()[:16]


@pytest.mark.parametrize("section,field", [("optimizer", "lr"), ("augmentation", "flip"),
                                           ("validation", "flags"), ("training", "warmup_steps")])
def test_unknown_config_keys_raise(tmp_path, section, field):
    config = _config()
    config.setdefault(section, {})[field] = 1
    with pytest.raises(ValueError, match="unknown"):
        train.run_training(config, _data(tmp_path, _rows(1)), tmp_path / "run")
    with pytest.raises(ValueError, match="config only accepts"):
        train.run_training({**_config(), "scheduler": {}}, _data(tmp_path, _rows(1)), tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_section_defaults_and_value_checks():
    training = train._training_config({"steps": 100, "learning_rate": 1e-4})
    assert training["resume"] is None and training["export_model_every_checkpoint"] is True
    assert train._optimizer_config({}, training) == {"warmup_steps": 3, "schedule": "cosine", "decoder_lr": 1e-4}
    assert train._augmentation_config({}) == train.DEFAULT_AUGMENTATION
    assert train._validation_config({})["exclude_flags"] == ["oob_objects", "fixed_collision", "overlapping_furniture"]
    for bad in ({"warmup_steps": 100}, {"warmup_steps": -1}, {"warmup_steps": True}, {"schedule": "linear"}, {"decoder_lr": 0}):
        with pytest.raises(ValueError):
            train._optimizer_config({"optimizer": bad}, training)
    for bad in ({"rotate90": 1}, {"drop_support_p": 1.5}, {"mirror": "no"}):
        with pytest.raises(ValueError):
            train._augmentation_config({"augmentation": bad})
    for bad in ({"exclude_flags": "oob_objects"}, {"exclude_flags": [""]}):
        with pytest.raises(ValueError):
            train._validation_config({"validation": bad})
    for bad in ({"resume": 3}, {"export_model_every_checkpoint": "yes"}):
        with pytest.raises(ValueError):
            train._training_config(bad)


def test_lr_factor_warms_up_then_decays_to_ten_percent():
    factors = [train._lr_factor(k, 2, 10, "cosine") for k in range(10)]
    assert factors[:3] == [.5, 1., 1.]
    assert all(a > b for a, b in zip(factors[2:], factors[3:]))
    assert train._lr_factor(10, 2, 10, "cosine") == pytest.approx(.1)
    assert train._lr_factor(7, 0, 10, "constant") == 1.


def test_parameter_groups_and_logged_schedule(tmp_path):
    config = _config(steps=4)
    config["optimizer"] = {"warmup_steps": 1, "decoder_lr": .001}
    config["augmentation"] = NO_AUGMENTATION
    model = build_model(ModelConfig(**config["model"]))
    groups = train._parameter_groups(model, .01, .001)
    assert [(g["name"], g["lr"]) for g in groups] == [("backbone", .01), ("decoder", .001)]
    by_group = {name: {id(p) for p in g["params"]} for name, g in zip(("backbone", "decoder"), groups)}
    for name, parameter in model.named_parameters():
        assert id(parameter) in by_group["backbone" if name.startswith("backbone.") else "decoder"]
    logs = train.run_training(config, _data(tmp_path, _rows(2)), tmp_path / "run")
    observed = [value for r in logs for value in (r["learning_rates"]["backbone"], r["learning_rates"]["decoder"])]
    expected = [base * train._lr_factor(k, 1, 4, "cosine") for k in range(4) for base in (.01, .001)]
    assert observed == pytest.approx(expected)


def _assert_same_trajectory(full, resumed):
    assert [r["step"] for r in full] == [r["step"] for r in resumed]
    for a, b in zip(full, resumed):
        assert a["rows_sha256"] == b["rows_sha256"] and a["counts_window"] == b["counts_window"]
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-6, abs=1e-7)
        assert a["learning_rates"] == pytest.approx(b["learning_rates"])
        assert a["gradient_norms"] == pytest.approx(b["gradient_norms"], rel=1e-5, abs=1e-6)
        for key, value in a["unweighted"].items():
            assert (value is None) == (b["unweighted"][key] is None)
            if value is not None:
                assert value == pytest.approx(b["unweighted"][key], rel=1e-6, abs=1e-7)


class _Killed(Exception):
    pass


def _killing_accelerator(kill_step):
    import accelerate

    class Killing(accelerate.Accelerator):
        def save_state(self, output_dir, **kwargs):
            result = super().save_state(output_dir, **kwargs)
            if str(output_dir).endswith(f"state-step-{kill_step}"):
                raise _Killed()
            return result
    return Killing


def _run_full_killed_resumed(root, patches, config, data, kill_step):
    import accelerate
    original = accelerate.Accelerator
    full = train.run_training(config, data, root / "full")
    patches.setattr(accelerate, "Accelerator", _killing_accelerator(kill_step))
    with pytest.raises(_Killed):
        train.run_training(config, data, root / "killed")
    patches.setattr(accelerate, "Accelerator", original)
    resumed_config = {**config, "training": {**config["training"], "resume": str(root / f"killed/state-step-{kill_step}")}}
    resumed = train.run_training(resumed_config, data, root / "resumed")
    return full, resumed


@pytest.mark.parametrize("kill_step", [3, 4])  # four rows per epoch: mid-epoch and end-of-epoch checkpoints
def test_resume_replays_trajectory_data_order_and_weights(tmp_path, monkeypatch, kill_step):
    data = _data(tmp_path, _rows(4))
    config = _config(steps=6, checkpoint_every=kill_step)
    config["optimizer"] = {"warmup_steps": 2}
    full, resumed = _run_full_killed_resumed(tmp_path, monkeypatch, config, data, kill_step)
    assert not (tmp_path / "killed/model").exists() and (tmp_path / f"killed/state-step-{kill_step}").is_dir()
    assert len(full) == len(resumed) == 6
    assert len({r["rows_sha256"] for r in full}) > 1  # shuffled distinct rows, so order equality is informative
    _assert_same_trajectory(full, resumed)
    for name in ("full", "resumed"):
        assert json.loads((tmp_path / name / "training_log.json").read_text()) == (full if name == "full" else resumed)
    a = torch.load(tmp_path / "full/model/geometry_model.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/model/geometry_model.pt", weights_only=True)
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    manifest = json.loads((tmp_path / "resumed/run_manifest.json").read_text())
    assert manifest["resumed_at_step"] == kill_step and manifest["steps_completed"] == 6
    assert manifest["resumed_from"].endswith(f"state-step-{kill_step}")
    assert '"resumed_from"' in (tmp_path / "resumed/train.log").read_text()


def test_resume_rejects_changed_config_or_data(tmp_path, monkeypatch):
    import accelerate
    data = _data(tmp_path, _rows(2))
    config = _config(steps=2, checkpoint_every=1)
    original = accelerate.Accelerator
    monkeypatch.setattr(accelerate, "Accelerator", _killing_accelerator(1))
    with pytest.raises(_Killed):
        train.run_training(config, data, tmp_path / "killed")
    monkeypatch.setattr(accelerate, "Accelerator", original)
    state = str(tmp_path / "killed/state-step-1")
    changed = {**config, "training": {**config["training"], "resume": state, "learning_rate": .02}}
    with pytest.raises(ValueError, match="resume requires the same config"):
        train.run_training(changed, data, tmp_path / "changed-config")
    (tmp_path / "other").mkdir()
    other = _data(tmp_path / "other", _rows(3))
    with pytest.raises(ValueError, match="resume requires the same config"):
        train.run_training({**config, "training": {**config["training"], "resume": state}}, other, tmp_path / "changed-data")


class _NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


class _OffsetCriterion:
    """Scalar objective c*p + c: value c and derivative c; optional NaN backward for one scene."""

    def __init__(self, coefficients, poison=()):
        self.coefficients, self.poison = coefficients, set(poison)

    def __call__(self, prediction, batch):
        scene = batch["provenance"][0]["scene_id"]
        coefficient = self.coefficients[int(scene)]
        loss = prediction * coefficient + coefficient
        if scene in self.poison:
            loss = _NaNGradient.apply(loss)
        zero = prediction * 0.
        return {"loss": loss, "yaw_cls": loss, "counts": {"yaw": 1}, "active_objective_count_local": 1,
                "term_sums": {**dict.fromkeys(train.TERMS, zero.detach()), "yaw_cls": loss.detach()},
                "term_counts": {**dict.fromkeys(train.TERMS, 0), "yaw_cls": 1},
                **{key: zero for key in ("position", "size", "yaw_reg", "box", "collision", "boundary")}}


def test_nonfinite_gradient_window_is_skipped_without_counting_a_step(tmp_path, monkeypatch):
    observed = []
    _patch_scalar_training(monkeypatch, [1., 1., 1.], observed)
    monkeypatch.setattr(train, "GeometryCriterion", lambda config: _OffsetCriterion([1., 1., 1.], poison={"1"}))
    config = _yaw_config()
    config["training"].update(steps=2, clip_grad_norm=100.)
    logs = train.run_training(config, _data(tmp_path, [_row(str(i)) for i in range(3)]), tmp_path / "run")
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    assert observed == pytest.approx([1., 1.])  # the poisoned window never reaches the optimizer
    assert [r["step"] for r in logs] == [1, 2]
    assert [r["rows_sha256"] for r in logs] == [_digest(["0"]), _digest(["2"])]
    assert manifest["skipped_nonfinite_windows"] == 1 and manifest["steps_completed"] == 2
    assert manifest["skipped_gradient_overflow_windows"] == 0


def test_all_windows_nonfinite_fails_the_epoch(tmp_path, monkeypatch):
    _patch_scalar_training(monkeypatch, [1.], [])
    monkeypatch.setattr(train, "GeometryCriterion", lambda config: _OffsetCriterion([1.], poison={"0"}))
    with pytest.raises(RuntimeError, match="nonfinite gradients"):
        train.run_training(_yaw_config(), _data(tmp_path, [_row("0")]), tmp_path / "run")


def test_window_logging_averages_every_microbatch(tmp_path, monkeypatch):
    _patch_scalar_training(monkeypatch, [1., 3.], [])
    monkeypatch.setattr(train, "GeometryCriterion", lambda config: _OffsetCriterion([1., 3.]))
    config = _yaw_config()
    config["training"].update(gradient_accumulation_steps=2, steps=1, clip_grad_norm=100.)
    logs = train.run_training(config, _data(tmp_path, [_row("0"), _row("1")]), tmp_path / "run")
    record = logs[0]
    assert record["accumulation_microbatches"] == 2 and record["counts_window"]["yaw_cls"] == 2
    assert record["unweighted"]["yaw_cls"] == pytest.approx(2.) and record["loss"] == pytest.approx(2.)
    assert record["unweighted"]["box"] is None and record["rows_sha256"] == _digest(["0", "1"])


def test_window_summary_weights_terms_by_valid_count():
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True)
    def result(value, count):
        return {"loss": torch.tensor(value), "term_sums": {**dict.fromkeys(train.TERMS, torch.tensor(0.)), "position": torch.tensor(value * count)},
                "term_counts": {**dict.fromkeys(train.TERMS, 0), "position": count}}
    summary = train._Window().add(result(1., 1)).add(result(3., 3)).summary(accelerator)
    assert summary["unweighted"]["position"] == pytest.approx(2.5) and summary["counts"]["position"] == 4
    assert summary["loss"] == pytest.approx(2.) and summary["microbatches"] == 2


def test_checkpoint_exports_validate_filter_flags_and_baselines(tmp_path):
    rows = _rows(2)
    data = _data(tmp_path, rows)
    clean = {**rows[0], "provenance": {**rows[0]["provenance"], "split": "validation", "house_id": "h2", "scene_id": "v0"}}
    flagged = {**clean, "provenance": {**clean["provenance"], "scene_id": "v1", "legacy_flags": {"oob_objects": True}}}
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(clean) + "\n" + json.dumps(flagged) + "\n")
    config = _config(steps=2, checkpoint_every=2, validate_every=0)
    config["augmentation"] = NO_AUGMENTATION
    run = tmp_path / "run"
    logs = train.run_training(config, data, run, validation=validation)
    assert "validation" not in logs[0] and "validation" in logs[1]  # every checkpoint step validates
    assert logs[1]["validation"]["counts"]["position"] == 1 and (run / "tokenizer").is_dir()
    manifest = json.loads((run / "run_manifest.json").read_text())
    assert manifest["validation_excluded_flagged"] == 1 and manifest["validation_samples"] == 1
    assert manifest["augmentation"] == NO_AUGMENTATION and manifest["config"] == config
    assert manifest["optimizer"] == {"warmup_steps": 0, "schedule": "cosine", "decoder_lr": .01}
    baseline = manifest["baselines"]
    x, y, _ = clean["target"]["objects"][0]["bottom_center_m"]
    assert baseline["rows"] == 1 and baseline["counts"]["position"] == 1
    assert baseline["unweighted"]["position"] == pytest.approx((abs(.5 - x / 5) + abs(.5 - y / 4)) / 3)  # z is floor-fixed
    assert baseline["unweighted"]["size"] == pytest.approx(0.)  # per-category median equals every fixture size
    assert baseline["unweighted"]["yaw_cls"] == pytest.approx(math.log(12))  # uniform bins
    log_lines = (run / "train.log").read_text().splitlines()
    assert json.loads(log_lines[0])["baseline"] == baseline
    assert [json.loads(line) for line in log_lines[1:]] == logs
    # The intermediate export is a deployable checkpoint: predict.py loads it beside tokenizer/.
    from fastfill.v2.predict import main as predict
    from fastfill.v2.schema import validate_layout
    assert (run / "model-step-2/geometry_model.pt").exists()
    loaded = load_model(run / "model-step-2")
    final = load_model(run / "model")
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), final.state_dict().values()))
    request = tmp_path / "request.json"
    request.write_text(json.dumps(rows[0]["condition"]))
    predict(["--checkpoint", str(run / "model-step-2"), "--condition", str(request), "--output", str(tmp_path / "prediction.json")])
    validate_layout(json.loads((tmp_path / "prediction.json").read_text()), rows[0]["condition"])


def test_export_can_be_disabled_and_every_validation_row_flagged_fails(tmp_path):
    rows = _rows(1)
    data = _data(tmp_path, rows)
    config = _config(steps=1, checkpoint_every=1, export_model_every_checkpoint=False)
    config["augmentation"] = NO_AUGMENTATION
    train.run_training(config, data, tmp_path / "run")
    assert (tmp_path / "run/state-step-1").is_dir() and not (tmp_path / "run/model-step-1").exists()
    flagged = {**rows[0], "provenance": {**rows[0]["provenance"], "split": "validation", "house_id": "h2",
                                         "legacy_flags": {"fixed_collision": True}}}
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(flagged) + "\n")
    with pytest.raises(ValueError, match="excluded flag"):
        train.run_training(config, data, tmp_path / "run2", validation=validation)
    config["validation"] = {"exclude_flags": []}
    logs = train.run_training(config, data, tmp_path / "run3", validation=validation)
    assert logs[0]["validation"]["batches"] == 1


def test_preflight_certifies_exchangeable_groups_before_training():
    row = sample()
    row["condition"]["objects"] = [{"id": "a", "category": "chair", "description": "a chair"},
                                   {"id": "b", "category": "chair", "description": "a chair"}]
    row["condition"]["constraints"] = [{"type": "faces_direction", "object_id": "a", "direction_xy": [1, 0]}]
    row["target"]["objects"] = [{"id": "a", "target_size_local_m": [1., 1., 1.], "bottom_center_m": [2., 1., 0.], "yaw_rad": 0.},
                                {"id": "b", "target_size_local_m": [1., 1., 1.], "bottom_center_m": [3., 1., 0.], "yaw_rad": 0.}]
    row["validity"] = {"position": [[True] * 3] * 2, "size": [[True] * 3] * 2, "yaw": [True, True],
                       "yaw_symmetry_order": [1, 1], "exchangeable_group": ["g", "g"]}
    arguments = ([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0), {"max_length": 4096})
    with pytest.raises(ValueError, match="cannot be matched before training.*constraint"):
        train._preflight(*arguments, LossConfig(hungarian=True))
    kept, rejected = train._preflight(*arguments, LossConfig(hungarian=False))
    assert kept == [row] and not rejected
    row["condition"]["constraints"] = []
    kept, _ = train._preflight(*arguments, LossConfig(hungarian=True))
    assert kept == [row]


def test_augmentation_is_seeded_per_epoch_and_row_and_never_applied_to_validation(tmp_path, monkeypatch):
    seen = []
    from fastfill.v2 import batch as batch_module
    original = batch_module.augment_sample
    def recording(sample_row, augmentation, generator):
        seen.append((sample_row["provenance"]["scene_id"], int(torch.randint(1 << 30, (1,), generator=generator))))
        return original(sample_row, augmentation, generator)
    monkeypatch.setattr(train, "augment_sample", recording)
    rows = train._Rows(_rows(3), train.DEFAULT_AUGMENTATION, seed=7)
    first = [rows[i] for i in range(3)]
    again = [rows[i] for i in range(3)]
    rows.epoch = 1
    later = [rows[i] for i in range(3)]
    assert first == again and first != later and seen[:3] == seen[3:6] and seen[:3] != seen[6:]
    assert train._Rows(_rows(1), NO_AUGMENTATION, seed=7)[0] == _rows(1)[0]
    data = _data(tmp_path, _rows(1))
    validation_row = {**_rows(1)[0], "provenance": {**_rows(1)[0]["provenance"], "split": "validation", "house_id": "h2"}}
    validation = tmp_path / "validation.jsonl"
    validation.write_text(json.dumps(validation_row) + "\n")
    seen.clear()
    train.run_training(_config(steps=2, validate_every=1), data, tmp_path / "run", validation=validation)
    assert [scene for scene, _ in seen] == ["s0", "s0"]  # training rows only; validation and baselines untouched


def _resume_worker(rank, rendezvous, directory, kill_step):
    from torch import distributed as dist
    import accelerate
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29515")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        # Three runs share one Gloo group: end_training() would destroy it after the first.
        patches.setattr(accelerate.Accelerator, "end_training", lambda self: None)
        root = Path(directory)
        data = root / "train.jsonl"
        if rank == 0:
            _data(root, _rows(8))
        dist.barrier()
        config = _config(steps=6, checkpoint_every=kill_step)
        full, resumed = _run_full_killed_resumed(root, patches, config, data, kill_step)
        assert len(full) == len(resumed) == 6 and [r["step"] for r in resumed] == list(range(1, 7))
        # Restored records are the main rank's, so only rank 0 compares its own rank-local rows digests there.
        _assert_same_trajectory(full if rank == 0 else full[kill_step:], resumed if rank == 0 else resumed[kill_step:])
        if rank == 0:
            a = torch.load(root / "full/model/geometry_model.pt", weights_only=True)
            b = torch.load(root / "resumed/model/geometry_model.pt", weights_only=True)
            assert all(torch.equal(a[key], b[key]) for key in a)
            assert json.loads((root / "resumed/run_manifest.json").read_text())["world_size"] == 2
        (root / f"rank-{rank}.json").write_text(json.dumps({"rows": [r["rows_sha256"] for r in resumed]}) + "\n")
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("kill_step", [3, 4])  # eight rows over two ranks: four microbatches per rank per epoch
def test_actual_two_rank_resume_replays_each_rank(tmp_path, monkeypatch, kill_step):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.multiprocessing.spawn(_resume_worker, args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path), kill_step),
                                nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text())["rows"] for rank in range(2)]
    assert ranks[0] != ranks[1]
