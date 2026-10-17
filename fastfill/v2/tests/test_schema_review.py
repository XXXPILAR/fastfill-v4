"""Adversarial request-boundary tests, independent of the model implementation."""

import copy

import pytest

from fastfill.v2.schema import validate_condition
from fastfill.v2.tests.test_geometry_core import condition


def test_support_cycle_is_rejected_before_tokenization():
    c = condition()
    c["objects"] = [{"id": "a", "category": "lamp", "description": "lamp", "support_parent": "b"},
                    {"id": "b", "category": "table", "description": "table", "support_parent": "a"}]
    with pytest.raises(ValueError, match="cycle"):
        validate_condition(c)


@pytest.mark.parametrize("field,value", [("semantic_front_required", "false"),
    ("required_capabilities", [None]), ("attributes", {"weight": float("nan")}),
    ("constraint_role", None), ("capability_requirements", {"load": 4})])
def test_untyped_or_nonfinite_request_metadata_rejected(field, value):
    c = condition()
    c["objects"][0][field] = value
    with pytest.raises(ValueError):
        validate_condition(c)


@pytest.mark.parametrize("constraint", [
    {"type": "faces_direction", "direction_xy": [1, 0]},
    {"type": "faces", "object_id": "a"},
    {"type": "near", "object_id": "a", "target_id": "a", "max_distance_m": 1},
    {"type": "clearance", "object_id": "a", "target_id": "floor", "min_distance_m": 1},
    {"type": "on", "object_id": "a"},
    {"type": "new_requirement", "hard": "false"},
    {"type": "new_requirement", "parameter": float("inf")},
])
def test_incomplete_or_ill_typed_constraints_fail_at_boundary(constraint):
    c = condition()
    c["constraints"] = [constraint]
    with pytest.raises(ValueError):
        validate_condition(c)


def test_valid_fixed_support_and_unknown_capabilities_are_immutable():
    c = condition()
    c["room"]["fixed_objects"] = [{"id": "existing_table", "size_local_m": [1, 1, .7],
        "bottom_center_m": [2, 2, 0], "yaw_rad": 0,
        "support_surfaces": [{"surface_id": "top", "local_polygon_xy_m": [[-.5, -.5], [.5, -.5],
            [.5, .5], [-.5, .5]], "local_z_m": .7}], "capabilities": None}]
    c["objects"][0]["support_parent"] = "existing_table"
    c["objects"][0]["support_surface_id"] = "top"
    before = copy.deepcopy(c)
    validate_condition(c)
    assert c == before
    c["room"]["fixed_objects"][0]["support_surfaces"][0]["local_z_m"] = float("nan")
    with pytest.raises(ValueError):
        validate_condition(c)


def test_known_spatial_constraints_retain_valid_roles_and_geometry():
    c = condition()
    c["objects"] += [{"id": "b", "category": "screen", "description": "screen", "support_parent": "floor"}]
    c["constraints"] = [
        {"type": "faces_direction", "object_id": "a", "direction_xy": [1, 0], "hard": True},
        {"type": "faces", "object_id": "a", "target_id": "b"},
        {"type": "near", "object_id": "a", "target_id": "b", "max_distance_m": 2.},
        {"type": "clearance", "object_id": "a", "target_id": "b", "min_distance_m": .3},
        {"type": "on", "object_id": "b", "parent_id": "floor"},
    ]
    validate_condition(c)


def test_implementation_metadata_hashes_current_uncommitted_sources(tmp_path):
    from pathlib import Path
    from fastfill.v2 import io
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{}\n')
    metadata = io.run_metadata(manifest)
    relative = "fastfill/v2/io.py"
    assert metadata["implementation_files_sha256"][relative] == io.fingerprint(Path(io.__file__))
    assert len(metadata["implementation_sha256"]) == 64
    assert len(metadata["data_sha256"]) == 64
