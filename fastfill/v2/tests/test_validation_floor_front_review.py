"""Independent room-floor and canonical support evidence must agree."""
from copy import deepcopy

import pytest

from fastfill.v2.runtime import Asset, AtomicMemoryHost, CatalogResolver, SupportSurface, run_pipeline
from fastfill.v2.tests.test_runtime import condition, desk, layout
from fastfill.v2.validation import validate_scene


def surface(height=.7, half_width=.4):
    return SupportSurface("top", ((-half_width, -.4), (half_width, -.4),
                                  (half_width, .4), (-half_width, .4)), height)


@pytest.mark.parametrize("candidate", [surface(-.5), surface(.8), surface(.7, .6)])
def test_asset_rejects_verified_surface_outside_its_canonical_bbox(candidate):
    with pytest.raises(ValueError, match="support.*bbox"):
        desk(size=(1., 1., .75), support_surfaces=(candidate,))


def test_surface_metadata_tolerance_is_small_and_does_not_rewrite_evidence():
    candidate = surface(.75 + 5e-7, .5 + 5e-7)
    asset = desk(size=(1., 1., .75), support_surfaces=(candidate,))
    assert asset.support_surfaces[0] == candidate
    with pytest.raises(ValueError, match="support.*bbox"):
        desk(size=(1., 1., .75), support_surfaces=(surface(.75 + 5e-6),))


@pytest.mark.parametrize("stage", ["target", "actual"])
def test_floor_lower_bound_applies_to_new_objects_without_rejecting_existing_structure(stage):
    request = condition([{"id": "lamp", "category": "lamp", "support_parent": "shelf"}])
    fixed = {"id": "shelf", "size_local_m": [1., 1., .1], "bottom_center_m": [2, 2, -.5],
             "yaw_rad": 0., "support_surfaces": [surface(.1).as_dict()]}
    request = {**request, "room": {**request["room"], "fixed_objects": [fixed]}}
    child = {"id": "lamp", "target_size_local_m": [.2, .2, .2], "actual_size_local_m": [.2, .2, .2],
             "bottom_center_m": [2, 2, -.4], "yaw_rad": 0.}
    before = deepcopy((request, child))
    report = validate_scene(request, [child], stage=stage)
    assert not report["ok"]
    failed = {tuple(check["object_ids"]) for check in report["checks"]
              if check["code"] == "floor_lower_bound" and check["status"] == "violation"}
    assert failed == {("lamp",)}
    assert (request, child) == before


def test_existing_wall_can_span_floor_while_generated_object_must_stay_above_it():
    request = condition()
    fixed = {"id": "existing_wall", "category": "wall", "size_local_m": [.1, 3., 3.],
             "bottom_center_m": [3.9, 2., -.2], "yaw_rad": 0.}
    request = {**request, "room": {**request["room"], "fixed_objects": [fixed]}}
    before = deepcopy(request)
    report = validate_scene(request, layout()["objects"])
    assert report["ok"]
    assert all(check["object_ids"] != ["existing_wall"] for check in report["checks"]
               if check["code"] == "floor_lower_bound")
    assert request == before


def test_unknown_floor_does_not_certify_a_lower_bound_from_coordinate_reference():
    request = condition()
    request = {**request, "room": {**request["room"], "floor_known": False}}
    report = validate_scene(request, layout()["objects"])
    assert not report["ok"]
    assert "floor_boundary_unknown" in report["unknown_checks"]
    assert not any(check["code"] == "floor_lower_bound" for check in report["checks"])


def test_bad_fixed_support_metadata_blocks_pipeline_before_any_commit():
    request = condition([{"id": "lamp", "category": "lamp", "support_parent": "shelf"}])
    request = {**request, "room": {**request["room"], "fixed_objects": [{
        "id": "shelf", "size_local_m": [1., 1., .75], "bottom_center_m": [2, 2, 0.],
        "yaw_rad": 0., "support_surfaces": [surface(-.5).as_dict()]}]}}
    prediction = layout([{"id": "lamp", "target_size_local_m": [.2, .2, .2],
                          "bottom_center_m": [2, 2, .75], "yaw_rad": 0.}])
    host = AtomicMemoryHost()
    report = run_pipeline(request, prediction, CatalogResolver((Asset("lamp", "lamp", (.2, .2, .2)),)),
                          host=host, expected_world_version=0, idempotency_key="invalid-fixed-surface")
    assert not report["ok"] and not report["committed"]
    assert host.snapshot()["commits"] == 0
    assert report["validation"]["final"]["checks"][0]["code"] == "invalid_geometry"


@pytest.mark.parametrize("front", [None, [0., 0., 0.], [0., 0., 1.], [1., 0., .1]])
def test_actual_required_front_is_hard_unknown_without_horizontal_evidence(front):
    request = condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                          "semantic_front_required": True}])
    actual = {**layout()["objects"][0], "actual_size_local_m": [1., 1., .75], "semantic_front_local": front}
    before = deepcopy((request, actual))
    report = validate_scene(request, [actual], stage="actual")
    assert not report["ok"]
    check = next(check for check in report["checks"] if check["code"] == "semantic_front_required")
    assert check["status"] == "unknown" and check["hard"]
    assert (request, actual) == before


def test_horizontal_front_and_ground_contact_preserve_success():
    request = condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                          "semantic_front_required": True}])
    actual = {**layout()["objects"][0], "actual_size_local_m": [1., 1., .75], "semantic_front_local": [1., 0., 0.]}
    report = validate_scene(request, [actual], stage="actual")
    assert report["ok"]
    assert any(check["code"] == "semantic_front_required" and check["status"] == "pass"
               for check in report["checks"])
