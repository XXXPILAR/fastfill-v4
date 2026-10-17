"""ID renaming must follow reference fields, never ordinary text or JSON keys."""
from copy import deepcopy

import pytest
import torch

from fastfill.v2.matching import certify_group, match_batch
from fastfill.v2.schema import validate_condition


def condition(first_id="a", *, category="chair", description="chair", **fields):
    return {
        "schema_version": "fastfill.v2",
        "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m":
                 [[0, 0], [5, 0], [5, 4], [0, 4]], "floor_z_m": 0., "height_m": 3.},
        "objects": [{"id": ident, "category": category, "description": description,
                     **deepcopy(fields)} for ident in [first_id, "b"]],
        "constraints": [],
    }


@pytest.mark.parametrize("literal_field", ["category", "description", "attributes", "key", "surface", "group"])
def test_plain_literals_matching_ids_do_not_change_exchangeability(literal_field):
    kwargs = {"category": "chair", "description": "anonymous chair"}
    first_id = "literal"
    if literal_field == "key":
        first_id = "category"
    elif literal_field == "surface":
        kwargs["support_surface_id"] = first_id
    elif literal_field == "group":
        first_id = "g"
    elif literal_field == "attributes":
        kwargs["attributes"] = {first_id: {"id": first_id, "object_id": first_id,
                                          "texts": [first_id, "b"]}}
    else:
        kwargs[literal_field] = first_id
    req = condition(first_id, **kwargs)
    validate_condition(req)
    original = deepcopy(req)
    certify_group(req["objects"], req["constraints"], [0, 1])
    assert req == original


@pytest.mark.parametrize("kind", ["faces_direction", "near", "on", "custom_role"])
def test_constraint_type_and_metadata_literals_are_not_id_references(kind):
    req = condition(kind)
    if kind in {"near", "custom_role"}:
        req["objects"] += [{"id": "table", "category": "table", "description": "table"}]
    for ident in [kind, "b"]:
        row = {"type": kind, "object_id": ident,
               "metadata": {kind: kind, "object_id": kind}, "note": kind}
        if kind == "faces_direction":
            row["direction_xy"] = [1, 0]
        elif kind in {"near", "custom_role"}:
            row["target_id"] = "table"
            if kind == "near":
                row["max_distance_m"] = 2.
        else:
            row["parent_id"] = "floor"
        req["constraints"] += [row]
    validate_condition(req)
    original = deepcopy(req)
    certify_group(req["objects"], req["constraints"], [0, 1])
    assert req == original


@pytest.mark.parametrize("parent", ["floor", "wall", "table"])
def test_explicit_shared_support_references_remain_exchangeable(parent):
    req = condition("chair", support_parent=parent)
    if parent == "table":
        req["objects"] += [{"id": "table", "category": "table", "description": "table"}]
    validate_condition(req)
    certify_group(req["objects"], req["constraints"], [0, 1])


def test_child_support_reference_prevents_uncoupled_exchange():
    req = condition()
    req["objects"] += [{"id": "cup", "category": "cup", "description": "cup", "support_parent": "a"}]
    validate_condition(req)
    with pytest.raises(ValueError, match="constraint or support role"):
        certify_group(req["objects"], req["constraints"], [0, 1])


@pytest.mark.parametrize("field", ["target_id", "parent_id", "target_ids", "object_ids"])
def test_constraint_reference_fields_prevent_role_changing_exchange(field):
    req = condition()
    req["objects"] += [{"id": "table", "category": "table", "description": "table"},
                       {"id": "desk", "category": "desk", "description": "desk"}]
    row = {"type": "custom_role", "object_id": "table", field: "a"}
    if field == "target_ids":
        row = {"type": "between", "object_id": "table", "target_ids": ["a", "desk"]}
    elif field == "object_ids":
        row[field] = ["a", "desk"]
    req["constraints"] = [row]
    validate_condition(req)
    with pytest.raises(ValueError, match="constraint or support role"):
        certify_group(req["objects"], req["constraints"], [0, 1])


def test_symmetric_between_reference_lists_preserve_group():
    req = condition()
    req["objects"] += [{"id": "table", "category": "table", "description": "table"},
                       {"id": "desk", "category": "desk", "description": "desk"}]
    req["constraints"] = [{"type": "between", "object_id": ident, "target_ids": ["table", "desk"]}
                          for ident in ["a", "b"]]
    validate_condition(req)
    certify_group(req["objects"], req["constraints"], [0, 1])


def test_category_equal_to_id_receives_detached_hungarian_swap():
    req = condition("chair")
    validate_condition(req)
    pos = torch.tensor([[[.2, .3, 0.], [.8, .3, 0.]]])
    predictions = {"position_normalized": pos.flip(1).clone().requires_grad_(),
                   "size": torch.ones(1, 2, 3, requires_grad=True)}
    batch = {"objects": [req["objects"]], "conditions": [req], "exchangeable_group": [["g", "g"]],
             "slot_mask": torch.tensor([[True, True]]),
             "targets": {"position_normalized": pos, "size": torch.ones(1, 2, 3)},
             "validity": {"position": torch.ones(1, 2, 3, dtype=torch.bool),
                          "size": torch.ones(1, 2, 3, dtype=torch.bool)}}
    assignment = match_batch(predictions, batch)
    assert assignment.tolist() == [[1, 0]]
    assert not assignment.requires_grad
    assert predictions["position_normalized"].requires_grad
