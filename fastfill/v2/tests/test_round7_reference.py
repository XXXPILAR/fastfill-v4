"""Round 7: the ground-truth FastFill layout of a benchmark scene inverts layout_to_roomgenbench, and
reference_check scores a hand-off against it under the same validate_scene rules."""
import json
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest

from fastfill.v2.direct_layout import export_handoff, layout_to_roomgenbench, request_to_condition
from fastfill.v2.geometry import wrap_yaw
from fastfill.v2.roomgenbench import benchmark_request, ground_truth_layout, main, reference_check

REFERENCE = Path(os.environ.get("FASTFILL_ROOMGENBENCH_ROOT") or Path(__file__).resolve().parents[3] / "RoomGenBench")
BATHROOM = REFERENCE / "bench" / "inputs" / "scenes" / "bathroom.json"


def test_ground_truth_layout_inverts_layout_to_roomgenbench_including_yaws_near_pi():
    yaws = [-math.pi, -math.pi + 1e-12, math.pi - 1e-12, math.nextafter(math.pi, 0), 0., .3, -2., math.pi / 2, -math.pi / 2]
    categories = ["desk", "chair", "lamp", "book", "cup", "clock", "rug", "shelf", "bin"]
    request = {"room_type": "study", "room_size_m": [4., 3., 2.7], "furniture_list": [
        {"id": f"obj_{i:04d}", "category": c, "support_parent": "floor"} for i, c in enumerate(categories)]}
    condition = request_to_condition(request)
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": f"obj_{i:04d}", "target_size_local_m": [.3 + .1 * i, 1.7 - .1 * i, .2 + .05 * i],
         "bottom_center_m": [.4 * i, 2.9 - .3 * i, .01 * i], "yaw_rad": yaw} for i, yaw in enumerate(yaws)]}
    inverse = ground_truth_layout(condition, layout_to_roomgenbench(condition, layout))
    for a, b in zip(layout["objects"], inverse["objects"]):
        assert a["id"] == b["id"]
        assert max(abs(x - y) for x, y in zip(a["target_size_local_m"] + a["bottom_center_m"],
                                              b["target_size_local_m"] + b["bottom_center_m"])) <= 1e-9
        assert abs(wrap_yaw(a["yaw_rad"] - b["yaw_rad"])) <= 1e-9, (a["yaw_rad"], b["yaw_rad"])


@pytest.mark.skipif(not BATHROOM.is_file(), reason="RoomGenBench scenes are not checked out")
def test_a_handoff_of_the_ground_truth_scores_zero_with_identical_checks(tmp_path):
    scene = json.loads(BATHROOM.read_text())
    condition = request_to_condition(benchmark_request(scene))
    export_handoff(tmp_path / "handoff", condition, ground_truth_layout(condition, scene))
    report = reference_check(tmp_path / "handoff", BATHROOM)
    assert report["objects"] == len(scene["objects"]) == report["errors"]["all"]["objects"]
    assert sum(report["errors"][p]["objects"] for p in ("floor", "wall", "on_object")) == report["objects"]
    for row in report["per_object"]:
        assert all(row[k] == 0 for k in ("position_error_m", "log_size_error", "yaw_error_rad",
                                         "box_equivalent_log_size_error", "box_equivalent_yaw_error_rad")), row
    assert report["errors"]["all"]["room_center_position_error_m"] > 0
    assert report["checks"]["fastfill"] == report["checks"]["ground_truth"]
    assert main(["--reference-check", str(tmp_path / "handoff"), "--scene", str(BATHROOM),
                 "--output", str(tmp_path / "out" / "check.json")]) == 0
    assert json.loads((tmp_path / "out" / "check.json").read_text())["checks"] == report["checks"]


@pytest.mark.skipif(not BATHROOM.is_file(), reason="RoomGenBench scenes are not checked out")
@pytest.mark.parametrize("mutate,message", [
    (lambda s: s["objects"].pop(), "request has 55 objects, benchmark scene 54"),
    (lambda s: s["objects"].insert(0, s["objects"].pop(1)), "is not benchmark object 0"),
    (lambda s: s["objects"][3].update(type="sofa"), "is not benchmark object 3"),
])
def test_misaligned_scenes_are_refused(mutate, message):
    scene = json.loads(BATHROOM.read_text())
    condition = request_to_condition(benchmark_request(scene))
    other = deepcopy(scene)
    mutate(other)
    with pytest.raises(ValueError, match=message):
        ground_truth_layout(condition, other)


def test_renamed_request_ids_are_refused():
    request = {"room_type": "study", "room_size_m": [4., 3.], "furniture_list": [{"id": "desk", "category": "desk"}]}
    scene = {"objects": [{"type": "desk", "position": {"x": 1., "y": 1., "z": 0.}, "rotation": {"z": 0.},
                          "dimensions": {"width": 1., "length": 1., "height": 1.}}]}
    with pytest.raises(ValueError, match=r"\(desk, desk\) is not benchmark object 0 \(obj_0000, desk\)"):
        ground_truth_layout(request_to_condition(request), scene)
