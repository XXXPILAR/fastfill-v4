"""Round 2 (D2 train/eval): checkpoint/run binding (K8), minimal projection and selection metric (K5),
box-equivalent errors (K1), three-way validator counts (K10) and generic term logging."""
from copy import deepcopy
import json
import math

import pytest
import torch

from fastfill.v2 import evaluate, train
from fastfill.v2.batch import TinyTokenizer, collate_samples, render_minimal_condition
from fastfill.v2.io import backbone_provenance, fingerprint, fingerprint_tree, load_checkpoint_config
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows, row
from fastfill.v2.tests.test_train_loop_fix20261006 import _Killed, _config, _killing_accelerator, _rows
from fastfill.v2.validation import CHECK_STATUSES, validate_scene

TINY = {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2, "decoder_layers": 1, "tiny_hidden_size": 16, "lora_rank": 0}
L_ROOM = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]


def _validation_file(tmp_path, rows):
    path = tmp_path / "validation.jsonl"
    path.write_text("".join(json.dumps({**r, "provenance": {**r["provenance"], "split": "validation", "house_id": "hv",
                                                             "scene_id": f"v{i}"}}) + "\n" for i, r in enumerate(rows)))
    return path


def _swap_row(prediction_size, prediction_yaw, *, swap=True, yaw_valid=True):
    sample = row([("chair", (1., 1., 0.), (2., 1., .8), 0.)], orders=[4], yaw_valid=yaw_valid)
    sample["validity"]["size_axis_swap_allowed"] = [swap]
    layout = deepcopy(sample["target"])
    layout["objects"][0].update(target_size_local_m=list(prediction_size), yaw_rad=prediction_yaw)
    return sample, layout


# K8: run start manifest, checkpoint binding, expected hashes -------------------------------------------------------

def test_start_manifest_is_written_before_the_first_update_and_the_final_manifest_only_at_the_end(tmp_path, monkeypatch):
    import accelerate
    data = _data(tmp_path, _rows(2))
    monkeypatch.setattr(accelerate, "Accelerator", _killing_accelerator(1))
    with pytest.raises(_Killed):
        train.run_training(_config(steps=2, checkpoint_every=1), data, tmp_path / "killed")
    start = json.loads((tmp_path / "killed/run_manifest_start.json").read_text())
    assert not (tmp_path / "killed/run_manifest.json").exists()
    assert start["config"] == _config(steps=2, checkpoint_every=1) and start["data_sha256"] == fingerprint(data)
    assert start["implementation_sha256"] and start["world_size"] == 1 and start["augmentation"] == train.DEFAULT_AUGMENTATION
    assert start["backbone"] == {"path": "tiny", "revision": None, "config_sha256": None}
    assert start["tokenizer_sha256"] == fingerprint_tree(tmp_path / "killed/tokenizer")
    assert start["selection_metric"]["name"] == "validation.minimal.geometry_objective"
    assert (start["supervised_samples"], start["rejected_samples"]) == (2, 0)
    assert (start["validation_samples"], start["validation_minimal_samples"]) == (0, 0)


def test_every_export_is_bound_and_validation_reports_both_projections(tmp_path):
    data = _data(tmp_path, _rows(2))
    validation = _validation_file(tmp_path, _rows(2))
    config = {**_config(steps=2, checkpoint_every=1, max_length=3000), "augmentation": NO_AUGMENTATION}
    run = tmp_path / "run"
    logs = train.run_training(config, data, run, validation=validation)
    manifest = json.loads((run / "run_manifest.json").read_text())
    for step, directory in ((1, "model-step-1"), (2, "model-step-2"), (2, "model")):
        bound = load_checkpoint_config(run / directory)
        assert bound["step"] == step and bound["max_length"] == 3000 and bound["source"].endswith("checkpoint_manifest.json")
        assert bound["data_sha256"] == fingerprint(data) and bound["validation_data_sha256"] == fingerprint(validation)
        assert bound["config_sha256"] == manifest["config_sha256"] and bound["implementation_sha256"] == manifest["implementation_sha256"]
        assert bound["model"]["position_head"] == "regression"
    for record in logs:
        result = record["validation"]
        assert result["minimal"]["batches"] == result["batches"] == 2 and result["minimal"]["counts"]["position"] == 2
        assert result["collapse"]["objects"] == result["minimal"]["collapse"]["objects"] == 2
        # Every label is complete here, so the matched column counts the same objects.
        assert result["collapse_matched"] == result["collapse"] and result["minimal"]["collapse_matched"] == result["minimal"]["collapse"]
        assert result["selection_metric"] == result["minimal"]["geometry_objective_mean_of_batches"]
        assert result["minimal"]["collapse"]["score"] == evaluate.collapse_score(result["minimal"]["collapse"])
    best = min((r["validation"]["selection_metric"], r["step"]) for r in logs)
    assert manifest["selection_metric"]["best"] == {"value": best[0], "step": best[1]}
    assert manifest["selection_metric"]["lower_is_better"] and manifest["validation_minimal_samples"] == 2
    assert manifest["tokenizer_sha256"] == fingerprint_tree(run / "tokenizer") and manifest["backbone"]["path"] == "tiny"


def test_non_rectangular_validation_rooms_have_no_minimal_projection(tmp_path):
    rows = _rows(1)
    rows[0]["condition"]["room"]["floor_polygon_xy_m"] = L_ROOM
    config = {**_config(steps=1, validate_every=1), "augmentation": NO_AUGMENTATION}
    logs = train.run_training(config, _data(tmp_path, _rows(1)), tmp_path / "run", validation=_validation_file(tmp_path, rows))
    assert logs[0]["validation"]["minimal"] is None and logs[0]["validation"]["selection_metric"] is None
    assert json.loads((tmp_path / "run/run_manifest.json").read_text())["selection_metric"]["best"] is None


def test_expected_input_hashes_abort_before_any_output(tmp_path):
    data = _data(tmp_path, _rows(1))
    validation = _validation_file(tmp_path, _rows(1))
    config = {**_config(steps=1), "augmentation": NO_AUGMENTATION}
    for kwargs in ({"expect_data_sha256": "0" * 64}, {"validation": validation, "expect_validation_sha256": "0" * 64},
                   {"expect_validation_sha256": fingerprint(validation)}):
        with pytest.raises(ValueError, match="does not match the expected"):
            train.run_training(config, data, tmp_path / "run", **kwargs)
        assert not (tmp_path / "run").exists()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="training data sha256"):
        train.main(["--config", str(config_path), "--data", str(data), "--output", str(tmp_path / "run"),
                    "--expect-data-sha256", "f" * 64])
    train.run_training(config, data, tmp_path / "run", validation=validation,
                       expect_data_sha256=fingerprint(data), expect_validation_sha256=fingerprint(validation))
    assert (tmp_path / "run/model/checkpoint_manifest.json").exists()


def test_backbone_provenance_reads_the_hf_snapshot_revision_or_only_the_config_digest(tmp_path):
    snapshot = tmp_path / "hub/models--Qwen--Qwen3-8B/snapshots/0123abcd"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3"}\n')
    assert backbone_provenance(str(snapshot)) == {"path": str(snapshot), "revision": "0123abcd",
                                                  "config_sha256": fingerprint(snapshot / "config.json")}
    local = tmp_path / "models/Qwen3-8B"
    local.mkdir(parents=True)
    (local / "config.json").write_text('{"model_type": "qwen3"}\n')
    assert backbone_provenance(str(local)) == {"path": str(local), "revision": None,
                                               "config_sha256": fingerprint(local / "config.json")}


def test_checkpoint_config_falls_back_to_the_run_manifest_and_reads_text_exports(tmp_path):
    model = tmp_path / "old-run/model-step-7"
    build_model(ModelConfig(**TINY)).save_pretrained(model)
    assert load_checkpoint_config(model)["max_length"] is None and load_checkpoint_config(model)["source"] is None
    (tmp_path / "old-run/run_manifest.json").write_text(json.dumps(
        {"training": {"max_length": 2048}, "data_sha256": "d", "implementation_sha256": "i", "steps_completed": 9}))
    bound = load_checkpoint_config(model)
    assert (bound["max_length"], bound["data_sha256"], bound["implementation_sha256"], bound["step"]) == (2048, "d", "i", None)
    assert bound["model"]["backbone"] == "tiny"
    text = tmp_path / "text"
    text.mkdir()
    (text / "text_config.json").write_text(json.dumps({"max_length": 1024, "steps": 3, "run_metadata": {"data_sha256": "t"}}))
    bound = load_checkpoint_config(text)
    assert (bound["model"], bound["max_length"], bound["step"], bound["data_sha256"]) == (None, 1024, 3, "t")


def _tiny_checkpoint(directory, max_length=None):
    build_model(ModelConfig(**TINY)).save_pretrained(directory)
    if max_length is not None:
        (directory / "checkpoint_manifest.json").write_text(json.dumps({"step": 1, "max_length": max_length}))
    return directory


def test_evaluation_binds_max_length_to_the_checkpoint_unless_given_and_records_provenance(tmp_path):
    rows = real_rows()[:1]
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(rows[0]) + "\n")
    checkpoint = _tiny_checkpoint(tmp_path / "model", max_length=64)  # shorter than any rendered condition
    bound = evaluate.run_evaluation(data, tmp_path / "bound", checkpoint=checkpoint)
    assert bound["inference_failed_requests"] == 1 and bound["max_length"] == 64
    assert bound["max_length_source"].endswith("checkpoint_manifest.json")
    explicit = evaluate.run_evaluation(data, tmp_path / "explicit", checkpoint=checkpoint, max_length=1 << 15)
    assert explicit["inference_failed_requests"] == 0 and explicit["max_length_source"] == "argument"
    assert explicit["checkpoint"] == str(checkpoint.resolve()) and explicit["checkpoint_binding"]["step"] == 1
    assert explicit["data_sha256"] == fingerprint(data) and explicit["implementation_sha256"]
    unbound = evaluate.run_evaluation(data, tmp_path / "unbound", checkpoint=_tiny_checkpoint(tmp_path / "bare"))
    assert unbound["max_length"] == evaluate.DEFAULT_MAX_LENGTH and unbound["max_length_source"] == "default"


# K5: minimal projection and the shared collapse score ---------------------------------------------------------------

def test_minimal_projection_uses_the_shared_renderer_and_skips_non_rectangular_rooms(tmp_path, monkeypatch):
    rectangular = real_rows()[0]
    lshaped = row([("chair", (1., 1., 0.), (.5, .5, .8), 0.), ("lamp", (1., 3., 0.), (.3, .3, 1.5), 0.)], scene="L")
    lshaped["condition"]["room"]["floor_polygon_xy_m"] = L_ROOM
    assert evaluate.project_minimal(lshaped) is None
    projected = evaluate.project_minimal(rectangular)
    assert projected["condition"] == render_minimal_condition(rectangular["condition"])
    assert projected["target"] == rectangular["target"]
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps(rectangular) + "\n" + json.dumps(lshaped) + "\n")
    seen, original = [], evaluate.predict_layout
    monkeypatch.setattr(evaluate, "predict_layout", lambda model, tokenizer, condition, **kw:
                        seen.append(condition) or original(model, tokenizer, condition, **kw))
    report = evaluate.run_evaluation(data, tmp_path / "out", checkpoint=_tiny_checkpoint(tmp_path / "model"),
                                     max_length=1 << 15, projections=("full", "minimal"))
    assert report["projection"] == "full" and report["requests"] == 2 and report["skipped_non_rectangular_rooms"] == 0
    minimal = report["projections"]["minimal"]
    assert minimal["projection"] == "minimal" and minimal["requests"] == 1 and minimal["skipped_non_rectangular_rooms"] == 1
    assert seen[2] == render_minimal_condition(rectangular["condition"]) and seen[:2] == [rectangular["condition"], lshaped["condition"]]
    outcomes = [json.loads(line) for line in (tmp_path / "out/outcomes-minimal.jsonl").read_text().splitlines()]
    assert [o["row"] for o in outcomes] == [0] and outcomes[0]["raw_prediction"]["objects"]
    collapse = minimal["collapse"]["predicted"]
    assert collapse["collapse_score"] == pytest.approx(collapse["bev_overlap_rate_iou_gt_0.3"] + collapse["central_quarter_fraction"])
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(rectangular["target"]) + "\n" + json.dumps(lshaped["target"]) + "\n")
    with pytest.raises(ValueError, match="supplied predictions"):
        evaluate.run_evaluation(data, tmp_path / "supplied", predictions=predictions, projections=("full", "minimal"))
    with pytest.raises(ValueError, match="projections"):
        evaluate.run_evaluation(data, tmp_path / "bad", predictions=predictions, projections=("full", "full"))


def test_batch_collapse_counts_match_the_report_collapse_of_the_same_predictions():
    sample = real_rows()[0]
    model = build_model(ModelConfig(**TINY)).eval()
    batch = collate_samples([sample], TinyTokenizer(), max_length=1 << 15)
    with torch.no_grad():
        predictions = model(**model_inputs(batch))
    layout = evaluate.serialize_predictions(predictions, batch)[0]
    expected = evaluate.collapse_metrics(layout, sample)["predicted"]
    observed = evaluate.batch_collapse_counts(predictions, batch)
    assert observed["objects"] == expected["objects"] > 1
    for key in evaluate.COLLAPSE_KEYS:
        assert observed[key] == pytest.approx(expected[key], abs=1e-4)
    assert evaluate.collapse_score({"objects": 4, "pairs": 6, "bev_overlap_pairs": 3, "central_quarter_objects": 1}) == .75
    assert evaluate.collapse_score({"objects": 1, "pairs": 0, "bev_overlap_pairs": 0, "central_quarter_objects": 1}) == 1.
    assert evaluate.collapse_score({"objects": 0, "pairs": 0, "bev_overlap_pairs": 0, "central_quarter_objects": 0}) is None


# K1: box-equivalent errors ------------------------------------------------------------------------------------------

def test_swap_allowed_objects_score_box_equivalently_with_the_plain_convention_kept():
    plain_size = 2 * math.log(2) / 3
    sample, layout = _swap_row((1., 2., .8), math.pi / 2)  # the same box written as (sy, sx, yaw + pi/2)
    metrics = evaluate.reference_metrics(layout, sample, hungarian=False)
    assert metrics["log_size_error"]["mean"] == pytest.approx(0., abs=1e-12)
    assert metrics["yaw_error_rad"]["mean"] == pytest.approx(0., abs=1e-12)
    assert metrics["log_size_error_plain_convention"]["mean"] == pytest.approx(plain_size)
    assert metrics["box_equivalent_objects"] == 1 and metrics["bev_iou"]["mean"] == pytest.approx(1.)
    sample, layout = _swap_row((1., 2., .8), math.pi / 2, swap=False)
    metrics = evaluate.reference_metrics(layout, sample, hungarian=False)
    assert metrics["log_size_error"]["mean"] == metrics["log_size_error_plain_convention"]["mean"] == pytest.approx(plain_size)
    assert metrics["box_equivalent_objects"] == 0
    # The candidate is chosen jointly: swapping the sizes would cost a quarter turn of yaw.
    sample, layout = _swap_row((1., 2., .8), 0.)
    metrics = evaluate.reference_metrics(layout, sample, hungarian=False)
    assert metrics["log_size_error"]["mean"] == pytest.approx(plain_size) and metrics["yaw_error_rad"]["mean"] == pytest.approx(0.)
    # Without a yaw label only the two size orders compete.
    sample, layout = _swap_row((1., 2., .8), 0., yaw_valid=False)
    metrics = evaluate.reference_metrics(layout, sample, hungarian=False)
    assert metrics["log_size_error"]["mean"] == pytest.approx(0., abs=1e-12) and metrics["yaw_error_rad"]["valid_objects"] == 0


def test_report_pools_plain_and_box_equivalent_errors_and_the_size_baseline_is_box_equivalent(tmp_path):
    a = row([("chair", (1., 1., 0.), (2., 1., .8), 0.)], scene="a", orders=[4])
    b = row([("chair", (3., 3., 0.), (1., 2., .8), 0.)], scene="b", orders=[4])
    for sample in (a, b):
        sample["validity"]["size_axis_swap_allowed"] = [True]
    layouts = [deepcopy(b["target"]), deepcopy(a["target"])]
    layouts[0]["objects"][0]["bottom_center_m"], layouts[1]["objects"][0]["bottom_center_m"] = [1., 1., 0.], [3., 3., 0.]
    layouts[0]["objects"][0]["yaw_rad"] = layouts[1]["objects"][0]["yaw_rad"] = math.pi / 2
    data, predictions = tmp_path / "data.jsonl", tmp_path / "predictions.jsonl"
    data.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    predictions.write_text("".join(json.dumps(layout) + "\n" for layout in layouts))
    report = evaluate.run_evaluation(data, tmp_path / "out", predictions=predictions, hungarian=False)
    reference = report["model"]["reference"]
    assert reference["log_size_error"] == {**reference["log_size_error"], "mean": pytest.approx(0., abs=1e-12), "valid_objects": 2}
    assert reference["log_size_error_plain_convention"]["mean"] == pytest.approx(2 * math.log(2) / 3)
    assert reference["box_equivalent_objects"] == 2 and reference["yaw_error_rad"]["mean"] == pytest.approx(0., abs=1e-12)
    assert report["baselines"]["category_median_size"]["log_size_error"]["mean"] == pytest.approx(0., abs=1e-12)


# K10: three-way validator statuses and per-check counts ------------------------------------------------------------

def test_ground_truth_with_undeclared_support_is_unknown_not_violation():
    sample = row([("chair", (1., 1., 0.), (.5, .5, .8), 0.), ("lamp", (3., 3., 0.), (.3, .3, 1.5), 0.)])
    report = validate_scene(sample["condition"], sample["target"]["objects"])
    support = [c for c in report["checks"] if c["object_ids"] and c["code"].startswith("support")]
    assert [(c["code"], c["status"]) for c in support] == [("support_unknown", "unknown")] * 2
    assert not report["ok"] and report["counts"]["violation"] == 0
    assert report["counts"] == {status: sum(c["status"] == status for c in report["checks"]) for status in CHECK_STATUSES}
    assert {c["status"] for c in report["checks"]} <= set(CHECK_STATUSES)
    outside = deepcopy(sample["target"]["objects"])
    outside[0]["bottom_center_m"] = [-2., 1., 0.]
    violated = validate_scene(sample["condition"], outside)
    assert [c["status"] for c in violated["checks"] if c["code"] == "boundary"] == ["violation", "pass"]
    assert violated["counts"]["violation"] == 1


def test_evaluation_counts_pass_violation_and_unknown_per_check(tmp_path):
    sample = row([("chair", (1., 1., 0.), (.5, .5, .8), 0.), ("lamp", (3., 3., 0.), (.3, .3, 1.5), 0.)])
    outside = deepcopy(sample["target"])
    outside["objects"][0]["bottom_center_m"] = [-2., 1., 0.]
    data, predictions = tmp_path / "data.jsonl", tmp_path / "predictions.jsonl"
    data.write_text("".join(json.dumps({**sample, "provenance": {**sample["provenance"], "scene_id": s}}) + "\n" for s in "abc"))
    predictions.write_text(json.dumps(sample["target"]) + "\n" + json.dumps(outside) + "\n" + '{"bad": true}\n')
    report = evaluate.run_evaluation(data, tmp_path / "out", predictions=predictions, hungarian=False)
    checks = report["target_validation_checks"]
    assert checks["boundary"] == {"pass": 3, "violation": 1, "unknown": 0}
    assert checks["support_unknown"] == {"pass": 0, "violation": 0, "unknown": 4}
    assert checks["mesh_unchecked"] == {"pass": 0, "violation": 0, "unknown": 2}
    assert report["target_validation_not_run"] == 1 and report["failed_requests"] == 3
    # A row whose own labels exceed the declared height keeps its ceiling checks apart from model violations.
    tall = deepcopy(sample)
    tall["target"]["objects"][1]["target_size_local_m"][2] = 3.2
    tall["provenance"]["height_conflict"] = {"objects": ["obj_0001"], "max_excess_m": .2}
    data.write_text(json.dumps(tall) + "\n")
    predictions.write_text(json.dumps(tall["target"]) + "\n")
    report = evaluate.run_evaluation(data, tmp_path / "conflict", predictions=predictions, hungarian=False)
    assert report["target_validation_checks"]["ceiling_on_height_conflict_rows"] == {"pass": 1, "violation": 1, "unknown": 0}
    assert "ceiling" not in report["target_validation_checks"]


# Trainer: generic term logging and the minimal-form augmentation option ---------------------------------------------

def test_window_logs_every_term_the_criterion_reports():
    from accelerate import Accelerator
    def result(value):
        return {"loss": torch.tensor(value), "term_sums": {"position": torch.tensor(value), "position_cell": torch.tensor(2 * value)},
                "term_counts": {"position": 1, "position_cell": 2}}
    summary = train._Window().add(result(1.)).add(result(3.)).summary(Accelerator(cpu=True))
    assert summary["unweighted"] == {"position": pytest.approx(2.), "position_cell": pytest.approx(2.)}
    assert summary["counts"] == {"position": 2, "position_cell": 4}


def test_minimal_form_probability_is_validated_and_alone_activates_augmentation():
    assert train._augmentation_config({})["minimal_form_p"] == .5
    with pytest.raises(ValueError, match="minimal_form_p"):
        train._augmentation_config({"augmentation": {"minimal_form_p": 1.5}})
    assert train._Rows([], {**NO_AUGMENTATION, "minimal_form_p": .1}, seed=0).active
    assert not train._Rows([], NO_AUGMENTATION, seed=0).active
