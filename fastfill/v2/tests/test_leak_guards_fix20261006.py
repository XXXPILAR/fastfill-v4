"""Review repairs 2026-10-06 (H): benchmark-room guards, non-convex shell walls, decode precision,
row-level leave-one-out baselines, height-tolerance epsilon, Scan2CAD rotational symmetry."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon
import torch
import trimesh

from fastfill.v2 import cohort, io, multisource_data, multisource_verify
from fastfill.v2.direct_layout import export_handoff
from fastfill.v2.evaluate import baseline_metrics, fit_baselines
from fastfill.v2.geometry import decode_yaw, wrap_yaw
from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.qualified_data import qualify_sample
from fastfill.v2.tests.test_complete_cohort import dataset, sample as cohort_sample
from fastfill.v2.tests.test_evaluate_fix20261006 import row as evaluation_row
from fastfill.v2.tests.test_legacy_bridge import fixture, prepared, saved
from fastfill.v2.tests.test_qualified_data import sample as qualified_sample
from fastfill.v2.tests.test_roomgenbench import REFERENCE
from fastfill.v2.tests.test_roomgenbench_fix20261006 import gltf_to_sage, real

BENCHMARK = io.ROOMGENBENCH_HOLDOUT_GROUPS[0]


# --- R-leak-3: every training-data reader refuses benchmark rooms ------------------

def test_single_holdout_constant_shared_by_producer_and_mirrored_by_the_independent_verifier():
    assert multisource_data.ROOMGENBENCH_HOLDOUT_GROUPS is io.ROOMGENBENCH_HOLDOUT_GROUPS
    assert multisource_verify.ROOMGENBENCH_HOLDOUT_GROUPS == io.ROOMGENBENCH_HOLDOUT_GROUPS
    assert multisource_verify.HOLDOUT_REASON == multisource_data.HOLDOUT_REASON == io.HOLDOUT_REASON


@pytest.mark.parametrize("provenance", [{"group": BENCHMARK}, {"holdout_reason": io.HOLDOUT_REASON}])
def test_read_samples_refuses_benchmark_rooms_in_training_but_not_in_evaluation(tmp_path, provenance):
    row = cohort_sample("u", "train")
    row["provenance"].update(provenance)
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="RoomGenBench benchmark room"):
        io.read_samples(path, training=True)
    assert len(io.read_samples(path)) == 1  # evaluation readers still see the row
    clean = cohort_sample("u", "train")
    clean["provenance"]["group"] = "sage:layout_other"
    path.write_text(json.dumps(clean) + "\n")
    assert len(io.read_samples(path, training=True)) == 1


def test_cohort_refuses_benchmark_rooms_outside_the_test_split(tmp_path):
    rows = {split: [cohort_sample(split, split)] for split in cohort.SPLITS}
    rows["train"][0]["provenance"]["group"] = BENCHMARK
    with pytest.raises(ValueError, match="RoomGenBench benchmark room in train.jsonl:1"):
        cohort.build_complete_cohort(dataset(tmp_path / "parent", rows), tmp_path / "complete")
    assert not (tmp_path / "complete").exists()
    rows["train"][0]["provenance"]["group"] = "sage:layout_other"
    rows["test"][0]["provenance"]["group"] = BENCHMARK  # already held out: allowed
    result = cohort.build_complete_cohort(dataset(tmp_path / "parent2", rows), tmp_path / "complete2")
    assert result["split_samples"] == dict.fromkeys(cohort.SPLITS, 1)


# --- R3: non-convex rooms get outward walls -------------------------------------------

def test_non_convex_room_walls_are_extruded_outwards(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = real()
    # Deep notch: the bbox centre lies on the wrong side of the notch wall (3, 2) -> (3, 5).
    polygon = [[0., 0.], [6., 0.], [6., 2.], [3., 2.], [3., 5.], [0., 5.]]
    condition["room"]["floor_polygon_xy_m"] = polygon
    condition["room"]["fixed_objects"] = [{"id": "fixed_0000", "category": "door", "size_local_m": [.9, .1, 2.],
                                           "bottom_center_m": [3., 3.5, 0.], "yaw_rad": math.pi / 2}]
    for obj in layout["objects"]:
        obj["bottom_center_m"][:2] = [1.5, 1.5]
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    assert receipt["room"]["wall_normal_source"] == "floor_polygon_winding"
    room = Polygon(polygon)
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    assert len(receipt["walls"]) == len(edges)
    for wall, (a, b) in zip(receipt["walls"], edges):
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        outward = (mid[0] + .05 * wall["normal"][0], mid[1] + .05 * wall["normal"][1])
        assert not room.contains(Point(outward)), wall
    scene = trimesh.load(tmp_path / "assembled" / f'{receipt["scene_key"]}.glb', force="scene", process=False)
    names = [name for name in scene.graph.nodes_geometry if name.startswith("shell_wall_")]
    assert "shell_wall_3_door" in names and len([n for n in names if n.endswith("_s0")]) == len(edges)
    for name in names:
        transform, geometry = scene.graph[name]
        points = gltf_to_sage(trimesh.transform_points(scene.geometry[geometry].vertices, transform))
        centre = points.mean(0)
        assert not room.buffer(-1e-6).contains(Point(centre[0], centre[1])), name


# --- R3-decode-yaw-bf16 -------------------------------------------------------------------

def test_decode_yaw_keeps_bin_centres_in_float32_under_bf16_residuals():
    bins, k = 12, 10
    logits = torch.zeros(1, bins)
    logits[0, k] = 10.
    residuals = torch.zeros(1, bins, dtype=torch.bfloat16)
    residuals[0, k] = .3
    exact = wrap_yaw(torch.tensor(k * 2 * math.pi / bins + float(residuals[0, k]) * math.pi / bins, dtype=torch.float64))
    decoded = decode_yaw(logits, residuals)
    assert decoded.dtype == torch.float32 and abs(float(decoded) - float(exact)) < 1e-3
    assert decode_yaw(logits.double(), residuals.double()).dtype == torch.float64


# --- R4-evaluate-loo: row-level leave-one-out ----------------------------------------------

def test_leave_one_out_excludes_every_label_of_the_row_not_only_the_objects_own():
    twins = evaluation_row([("sofa", (1., 1., 0.), (2., .9, .8), 0.), ("sofa", (3., 1., 0.), (2., .9, .8), 0.)], scene="a")
    other = evaluation_row([("sofa", (2., 3., 0.), (1., 1., 1.), 0.)], scene="b")
    fit = fit_baselines([twins, other])
    metrics = baseline_metrics(twins, fit, exclude_row=0)
    # Both sofas are predicted from the other row only: position error to (2, 3), size error log(2 / 1) etc.
    assert metrics["category_mean_position"]["bottom_center_error_m"]["mean"] == pytest.approx((math.hypot(1., 2.) + math.hypot(1., 2.)) / 2)
    assert metrics["category_median_size"]["log_size_error"]["mean"] == pytest.approx((abs(math.log(2.)) + abs(math.log(.9)) + abs(math.log(.8))) / 3)
    assert metrics["category_fallback_objects"] == 0
    # Alone in its category, the row must fall back to the all-category pool for every object.
    lonely = fit_baselines([twins, evaluation_row([("lamp", (2., 3., 0.), (1., 1., 1.), 0.)], scene="c")])
    assert baseline_metrics(twins, lonely, exclude_row=0)["category_fallback_objects"] == 4


# --- R4: height conflict needs more than float32 rounding noise -------------------------------

def test_declared_height_survives_a_float32_rounded_size_at_the_tolerance_edge():
    parent = qualified_sample()
    parent["condition"]["room"]["height_m"] = 2.6
    parent["target"]["objects"][0]["target_size_local_m"][2] = 2.6500000953674316  # float32(2.65) in a 2.6 m room
    row = qualify_sample(parent)
    assert row["condition"]["room"]["height_m"] == 2.6
    assert "height_conflict" not in row["provenance"]
    multisource_verify.verify_full_pair(parent, row, "train")
    parent["target"]["objects"][0]["target_size_local_m"][2] = 2.66
    row = qualify_sample(parent)
    assert row["condition"]["room"]["height_m"] == 2.6 and "height_conflict" in row["provenance"]  # K2: flag only
    multisource_verify.verify_full_pair(parent, row, "train")


# --- R5: Scan2CAD rotational symmetry ---------------------------------------------------------

@pytest.mark.parametrize("sym,order,yaw_valid", [("__SYM_NONE", 2, True), ("__SYM_ROTATE_UP_2", 2, True),
                                                 ("__SYM_ROTATE_UP_4", 4, True), ("__SYM_ROTATE_UP_INF", 2, False)])
def test_scan2cad_symmetry_annotation_sets_yaw_order_or_unsupervises_yaw(sym, order, yaw_valid):
    raw = fixture()
    raw["source"], raw["uid"] = "Scan2CAD", "Scan2CAD:fixture"
    raw["objects"][0]["sym"] = sym
    p = prepared(raw)
    out = convert_selected_room(raw, p, saved(raw, p), "train")
    index = out["provenance"]["target_source_ids"].index("source-b")
    assert out["validity"]["yaw_symmetry_order"][index] == order
    assert out["validity"]["yaw"][index] is yaw_valid
    assert out["validity"]["yaw_symmetry_order"][1 - index] == 2 and out["validity"]["yaw"][1 - index] is True
