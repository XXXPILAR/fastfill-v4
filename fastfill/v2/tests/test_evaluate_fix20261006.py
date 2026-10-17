"""Audit 2026-10-06: baselines, collapse metrics, symmetry-aware yaw, per-source table, direct-request parity."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from fastfill.v2.batch import condition_segments
from fastfill.v2.direct_layout import request_to_condition
from fastfill.v2.evaluate import baseline_metrics, collapse_metrics, fit_baselines, reference_metrics, run_evaluation
from fastfill.v2.text_sft import main as text_main

# Three rows of selected-v3.2/validation.jsonl (HSSD200 x2, MultiScan x1) in the C1 validity format.
FIXTURE = Path(__file__).parent / "fixtures" / "validation_rows_fix20261006.jsonl"


def real_rows():
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def row(objects, *, source="fixture", scene="s", orders=None, yaw_valid=True):
    """One 4 x 4 x 3 room; objects are (category, bottom_center_m, size, yaw)."""
    return {"schema_version": "fastfill.v2",
            "condition": {"schema_version": "fastfill.v2", "room": {
                "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "floor_z_m": 0., "floor_known": True, "boundary_known": True, "height_m": 3.},
                "objects": [{"id": f"obj_{i:04d}", "category": c, "description": c} for i, (c, *_) in enumerate(objects)],
                "constraints": []},
            "target": {"schema_version": "fastfill.v2", "objects": [
                {"id": f"obj_{i:04d}", "bottom_center_m": list(p), "target_size_local_m": list(s), "yaw_rad": y}
                for i, (_, p, s, y) in enumerate(objects)]},
            "validity": {"position": [[True] * 3] * len(objects), "size": [[True] * 3] * len(objects),
                         "yaw": [yaw_valid] * len(objects), "yaw_symmetry_order": orders or [1] * len(objects),
                         "exchangeable_group": [None] * len(objects)},
            "provenance": {"source": source, "house_id": "h", "scene_id": scene, "split": "test"}}


def evaluate(tmp_path, rows, layouts, **kwargs):
    data, predictions = tmp_path / "data.jsonl", tmp_path / "predictions.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    predictions.write_text("".join(json.dumps(l) + "\n" for l in layouts))
    report = run_evaluation(data, tmp_path / "out", predictions=predictions, **kwargs)
    outcomes = [json.loads(line) for line in (tmp_path / "out" / "outcomes.jsonl").read_text().splitlines()]
    return report, outcomes


def test_perfect_imitator_on_real_rows_scores_zero_and_reproduces_ground_truth_collapse(tmp_path):
    rows = real_rows()
    report, outcomes = evaluate(tmp_path, rows, [r["target"] for r in rows])
    assert report["requests"] == 3 and report["inference_failed_requests"] == 0
    assert report["model"]["schema_success"] == 1. and report["model"]["requested_ids_exactly_once"] == 1.
    reference = report["model"]["reference"]
    assert reference["bottom_center_error_m"] == {**reference["bottom_center_error_m"], "mean": 0., "valid_objects": 18}
    assert reference["log_size_error"]["mean"] == 0. and reference["log_size_error"]["valid_objects"] == 18
    assert reference["yaw_error_rad"]["mean"] == 0. and reference["yaw_error_rad"]["valid_objects"] == 11
    assert reference["bev_iou"]["mean"] == pytest.approx(1.)
    assert reference["yaw_error_rad_by_symmetry_order"] == {"1": {"mean": 0., "valid_objects": 11}}
    assert report["collapse"]["predicted"] == report["collapse"]["ground_truth"]
    assert report["collapse"]["predicted"]["objects"] == 18
    assert all(o["model"]["reference"]["matching_scope"] == "exchangeable_groups" for o in outcomes)
    assert all("reference_error" not in o for o in outcomes)
    assert report["baselines"]["fit"] == "evaluation_set_leave_one_out"
    assert report["baselines"]["room_center_position"]["bottom_center_error_m"]["mean"] > 0.
    assert report["baselines"]["uniform_yaw"]["yaw_error_rad"] == {"mean": pytest.approx(math.pi / 2), "valid_objects": 11}
    assert set(report["by_source"]) == {"HSSD200", "MultiScan"}
    assert report["by_source"]["HSSD200"]["requests"] == 2 and report["by_source"]["MultiScan"]["requests"] == 1
    assert report["by_source"]["MultiScan"]["model"]["reference"]["yaw_error_rad"]["valid_objects"] == 11
    assert report["by_source"]["HSSD200"]["model"]["reference"]["yaw_error_rad"]["valid_objects"] == 0


@pytest.mark.parametrize("hungarian", [False, True])
def test_shifting_every_prediction_by_ten_centimetres_moves_position_error_by_the_shift(tmp_path, hungarian):
    rows = real_rows()
    layouts = deepcopy([r["target"] for r in rows])
    for layout in layouts:
        for obj in layout["objects"]:
            obj["bottom_center_m"][0] += .10
    report, _ = evaluate(tmp_path, rows, layouts, hungarian=hungarian)
    error = report["model"]["reference"]["bottom_center_error_m"]
    assert error["valid_objects"] == 18
    if hungarian:
        assert error["mean"] <= .10 + 1e-9  # legal re-correspondence may only lower the error
    else:
        assert error["mean"] == pytest.approx(.10, abs=1e-9)
    assert report["model"]["reference"]["log_size_error"]["mean"] == 0.
    assert report["collapse"]["predicted"] != report["collapse"]["ground_truth"]


def test_yaw_error_uses_symmetry_order_and_reports_counts_per_order():
    sample = row([("desk", (1, 1, 0), (1, .5, .7), 0.), ("table", (3, 3, 0), (1, 1, .7), .5)], orders=[1, 2])
    layout = deepcopy(sample["target"])
    for obj in layout["objects"]:
        obj["yaw_rad"] = (obj["yaw_rad"] + math.pi + math.pi) % (2 * math.pi) - math.pi
    metrics = reference_metrics(layout, sample, hungarian=False)
    assert metrics["yaw_error_rad_by_symmetry_order"]["1"] == {"mean": pytest.approx(math.pi), "valid_objects": 1}
    assert metrics["yaw_error_rad_by_symmetry_order"]["2"] == {"mean": pytest.approx(0., abs=1e-12), "valid_objects": 1}
    assert metrics["yaw_error_rad"] == {"mean": pytest.approx(math.pi / 2), "valid_objects": 2}
    assert metrics["matching_scope"] == "fixed"


def test_collapse_metrics_separate_a_mean_predictor_from_the_labels():
    sample = row([("chair", (.5, .5, 0), (.5, .5, 1), 0.), ("chair", (3.5, .5, 0), (.5, .5, 1), 0.),
                  ("chair", (3.5, 3.5, 0), (.5, .5, 1), 0.), ("lamp", (.5, 3.5, 0), (.3, .3, 1.5), 0.)])
    collapsed = deepcopy(sample["target"])
    for obj in collapsed["objects"]:
        obj["bottom_center_m"] = [2., 2., 0.]
    result = collapse_metrics(collapsed, sample)
    predicted, truth = result["predicted"], result["ground_truth"]
    assert predicted["objects"] == truth["objects"] == 4 and predicted["pairs"] == truth["pairs"] == 6
    assert predicted["same_category_pairs"] == truth["same_category_pairs"] == 3
    assert (predicted["bev_overlap_pairs"], predicted["stacked_same_category_pairs"], predicted["central_quarter_objects"]) == (6, 3, 4)
    assert (truth["bev_overlap_pairs"], truth["stacked_same_category_pairs"], truth["central_quarter_objects"]) == (0, 0, 0)
    assert predicted["nearest_wall_distance_sum_m"] == pytest.approx(8.)
    assert truth["nearest_wall_distance_sum_m"] == pytest.approx(2.)
    # Only slots with trustworthy labels enter the ground truth; the prediction keeps every requested object.
    sample["validity"]["size"][3] = [False] * 3
    partial = collapse_metrics(collapsed, sample)
    assert (partial["predicted"]["objects"], partial["ground_truth"]["objects"]) == (4, 3)


def test_baselines_fit_leave_one_out_by_default_and_from_a_fit_file_when_given(tmp_path):
    rows = [row([("chair", (1., 1., 0.), (.4, .4, .8), 0.)], scene="a"),
            row([("chair", (3., 3., 0.), (.6, .6, 1.), 0.)], scene="b", orders=[2]),
            row([("sofa", (2., 1., 0.), (2., .9, .8), 0.)], scene="c")]
    loo = fit_baselines(rows)
    first = baseline_metrics(rows[0], loo, exclude_row=0)
    # Room centre (2, 2, 0) versus (1, 1, 0); the other chair sits at (3, 3, 0).
    assert first["room_center_position"]["bottom_center_error_m"]["mean"] == pytest.approx(math.sqrt(2))
    assert first["category_mean_position"]["bottom_center_error_m"]["mean"] == pytest.approx(math.sqrt(8))
    assert first["category_median_size"]["log_size_error"]["mean"] == pytest.approx(abs(math.log(.6 / .4)) * 2 / 3 + abs(math.log(1. / .8)) / 3)
    assert first["uniform_yaw"]["yaw_error_rad"]["mean"] == pytest.approx(math.pi / 2)
    assert first["category_fallback_objects"] == 0
    second = baseline_metrics(rows[1], loo, exclude_row=1)
    assert second["uniform_yaw"]["yaw_error_rad"]["mean"] == pytest.approx(math.pi / 4)
    # The sofa has no other sofa: both statistics fall back to the all-category pool.
    third = baseline_metrics(rows[2], loo, exclude_row=2)
    assert third["category_fallback_objects"] == 2
    assert third["category_mean_position"]["bottom_center_error_m"]["mean"] == pytest.approx(
        math.hypot(2. - 2., 1. - 2.))
    fit_file = tmp_path / "fit.jsonl"
    fit_file.write_text(json.dumps(row([("chair", (0., 0., 0.), (1., 1., 1.), 0.)])) + "\n")
    report, outcomes = evaluate(tmp_path, rows[:1], [rows[0]["target"]], baseline_fit=fit_file, hungarian=False)
    assert report["baselines"]["fit"] == f"baseline_fit_file:{fit_file}"
    assert report["baselines"]["category_mean_position"]["bottom_center_error_m"]["mean"] == pytest.approx(math.sqrt(2))
    assert report["baselines"]["category_median_size"]["log_size_error"]["mean"] == pytest.approx(abs(math.log(.4)) * 2 / 3 + abs(math.log(.8)) / 3)
    assert outcomes[0]["baselines"]["category_fallback_objects"] == 0


@pytest.mark.parametrize("lr", ["nan", "inf", "-1e-4", "0"])
def test_text_sft_rejects_nonfinite_or_nonpositive_learning_rate(tmp_path, lr):
    with pytest.raises(SystemExit):
        text_main(["--data", str(tmp_path / "missing.jsonl"), "--output", str(tmp_path / "out"),
                   "--backbone", "tiny", "--dry-run", "--lr", lr])


def rendered(condition):
    text = "".join(segment for segment, _ in condition_segments(condition))
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


def test_direct_request_renders_exactly_the_training_rectangle_fields():
    training = next(r for r in real_rows() if "fixed_objects" not in r["condition"]["room"])["condition"]
    assert len(training["room"]["floor_polygon_xy_m"]) == 4 and training["room"]["boundary_known"] and training["room"]["floor_known"]
    direct = request_to_condition({"room_type": "bedroom", "room_size_m": [4., 3., 2.6], "furniture_list": ["bed"]})
    room = rendered(direct)["room"]
    assert set(room) == set(rendered(training)["room"]) and "boundary_quality" not in room
    assert len(room["floor_polygon_xy_m"]) == 4 and room["boundary_known"] is True and room["floor_known"] is True
    assert room["floor_z_m"] == training["room"]["floor_z_m"] == 0. and room["height_m"] == 2.6
    assert set(rendered(direct)) == set(rendered(training)) and direct["constraints"] == []
    assert set(rendered(direct)["objects"][0]) == {"id", "category", "description"}
