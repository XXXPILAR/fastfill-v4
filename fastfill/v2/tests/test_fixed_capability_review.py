"""Existing fixed assets keep their typed capability requirements at both stages."""
from copy import deepcopy

import pytest

from fastfill.v2.runtime import AtomicMemoryHost, CatalogResolver, RuntimeBudget, run_pipeline
from fastfill.v2.schema import validate_condition
from fastfill.v2.tests.test_runtime import condition, desk, layout
from fastfill.v2.validation import validate_scene


def with_fixed_screen(capabilities="missing", *, require=True):
    fixed = {"id": "screen", "category": "screen", "size_local_m": [.2, 1., 1.],
             "bottom_center_m": [3.5, 3., .5], "yaw_rad": 0.}
    if require:
        fixed = {**fixed, "required_capabilities": ["display_video"]}
    if capabilities != "missing":
        fixed = {**fixed, "capabilities": capabilities}
    request = condition()
    return {**request, "room": {**request["room"], "fixed_objects": [fixed]}}


@pytest.mark.parametrize("stage", ["target", "actual"])
@pytest.mark.parametrize("capabilities,status", [([], "violation"), (["audio"], "violation"),
                                                ("missing", "unknown"), (None, "unknown"),
                                                (["display_video"], "pass")])
def test_fixed_capability_requirement_is_checked_in_each_geometry_stage(stage, capabilities, status):
    request = with_fixed_screen(capabilities)
    validate_condition(request)
    actual = {**layout()["objects"][0], "actual_size_local_m": [1., 1., .75]}
    before = deepcopy((request, actual))
    report = validate_scene(request, [actual], stage=stage)
    check = next(check for check in report["checks"]
                 if check["code"] == "capabilities" and check["object_ids"] == ["screen"])
    assert check["hard"] and check["status"] == status
    assert report["ok"] is (status == "pass")
    assert check["geometry_role"] == "fixed"
    assert (request, actual) == before


def test_no_fixed_requirement_does_not_require_capability_metadata():
    request = with_fixed_screen(require=False)
    report = validate_scene(request, layout()["objects"])
    assert report["ok"]
    assert not any(check["code"] == "capabilities" for check in report["checks"])


def test_requested_target_does_not_claim_unresolved_asset_capabilities():
    request = condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                          "required_capabilities": ["work_surface"]}])
    report = validate_scene(request, layout()["objects"], stage="target")
    assert report["ok"]
    assert not any(check["code"] == "capabilities" for check in report["checks"])


@pytest.mark.parametrize("capabilities", [[], "missing", None])
def test_unsatisfied_or_unknown_fixed_capability_prevents_every_host_write(capabilities):
    request = with_fixed_screen(capabilities)
    host = AtomicMemoryHost()
    report = run_pipeline(request, layout(), CatalogResolver((desk(),)), host=host,
        expected_world_version=0, idempotency_key="fixed-capability-request", budget=RuntimeBudget(max_asset_retries=0))
    assert not report["ok"] and not report["committed"]
    assert report["metrics"]["asset"]["retrieval_coverage"] == 1.
    assert host.snapshot() == {"version": 0, "objects": [], "commits": 0}


def test_verified_fixed_capability_allows_normal_atomic_commit():
    host = AtomicMemoryHost()
    report = run_pipeline(with_fixed_screen(["display_video"]), layout(), CatalogResolver((desk(),)),
        host=host, expected_world_version=0, idempotency_key="verified-fixed-capability")
    assert report["ok"] and report["committed"]
    assert host.snapshot()["commits"] == 1
