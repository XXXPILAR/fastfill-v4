"""Unverified typed attributes cannot be satisfied by a reference catalog."""
from copy import deepcopy

import pytest

from fastfill.v2.runtime import Asset, AtomicMemoryHost, CatalogResolver, run_pipeline
from fastfill.v2.schema import validate_condition
from fastfill.v2.tests.test_runtime import condition, desk, layout
from fastfill.v2.validation import validate_scene


def attribute_condition(attributes):
    return condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                       "attributes": attributes, "description": "a red wooden work desk"}])


@pytest.mark.parametrize("attributes", [
    {"color": "red"}, {"material": "wood"}, {"has_drawers": False},
    {"finish": {"coating": "matte"}}, {"color": None},
])
def test_reference_catalog_requires_evidence_for_every_nonempty_typed_attribute_request(attributes):
    request = attribute_condition(attributes)
    validate_condition(request)
    asset = desk(description="a red wooden work desk", capabilities=("work_surface",))
    before = deepcopy((request, layout()))
    resolver = CatalogResolver((asset,))
    assert resolver.resolve(request["objects"][0], layout()["objects"][0]) is None
    assert (request, layout()) == before


@pytest.mark.parametrize("claimed_metadata", [{}, {"attributes": {"color": "red", "material": "wood"}},
                                             {"verified_attributes": {"color": "red", "material": "wood"}}])
def test_direct_actual_validator_has_hard_unknown_without_an_attribute_evidence_contract(claimed_metadata):
    request = attribute_condition({"color": "red", "material": "wood"})
    actual = {**layout()["objects"][0], "actual_size_local_m": [.9, 1., .7],
              "capabilities": ["work_surface"], **claimed_metadata}
    before = deepcopy((request, actual))
    report = validate_scene(request, [actual], stage="actual")
    assert not report["ok"]
    check = next(check for check in report["checks"] if check["code"] == "attributes_unverified")
    assert check["hard"] is True and check["status"] == "unknown"
    assert check["object_ids"] == ["desk"]
    assert "attributes_unverified" in report["unknown_checks"]
    assert (request, actual) == before


def test_external_resolver_cannot_bypass_required_attributes_or_partially_commit():
    request = condition([
        {"id": "plain", "category": "desk", "support_parent": "floor"},
        {"id": "required", "category": "desk", "support_parent": "floor",
         "attributes": {"color": "red", "material": "wood"}},
    ])
    prediction = layout([{**layout()["objects"][0], "id": "plain", "bottom_center_m": [1, 2, 0]},
                         {**layout()["objects"][0], "id": "required", "bottom_center_m": [3, 2, 0]}])
    before = deepcopy((request, prediction))

    class IgnoringResolver:
        def resolve(self, request_object, _prediction, *, excluded_refs=()):
            return Asset(request_object["id"], "desk", (.9, 1., .7), description="blue plastic desk")

    host = AtomicMemoryHost()
    report = run_pipeline(request, prediction, IgnoringResolver(), host=host,
                          expected_world_version=0, idempotency_key="attribute-request")
    assert not report["ok"] and not report["committed"]
    assert report["metrics"]["model"]["schema_success"] is True
    assert report["diagnostics"][0]["code"] == "asset_unavailable"
    assert report["diagnostics"][0]["object_id"] == "required"
    assert host.snapshot() == {"version": 0, "objects": [], "commits": 0}
    assert report["raw_model_output"] == prediction
    assert (request, prediction) == before


@pytest.mark.parametrize("attributes", [None, {}])
def test_plain_or_empty_attributes_keep_existing_size_and_capability_contract(attributes):
    request_object = {"id": "desk", "category": "desk", "support_parent": "floor",
                      "required_capabilities": ["work_surface"],
                      "fixed_size_local_m": [None, 1., None]}
    if attributes is not None:
        request_object = {**request_object, "attributes": attributes}
    request, prediction = condition([request_object]), layout()
    resolver = CatalogResolver((desk("no_capability"),
                                desk("eligible", (.9, 1., .7), capabilities=("work_surface",))))
    host = AtomicMemoryHost()
    report = run_pipeline(request, prediction, resolver, host=host,
                          expected_world_version=0, idempotency_key="plain-request")
    assert report["ok"] and report["committed"]
    assert report["final_objects"][0]["asset_ref"] == "eligible"
    assert report["final_objects"][0]["target_size_local_m"] == [1., 1., .75]
    assert report["final_objects"][0]["actual_size_local_m"] == [.9, 1., .7]
    assert "attributes_unverified" not in report["validation"]["final"]["unknown_checks"]
    assert host.snapshot()["commits"] == 1


def test_target_geometry_stage_does_not_claim_asset_attribute_verification():
    report = validate_scene(attribute_condition({"color": "red"}), layout()["objects"])
    assert report["ok"]
    assert report["stage"] == "target" and report["geometry_level"] == "bbox"
