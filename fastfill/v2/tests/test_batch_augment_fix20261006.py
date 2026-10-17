"""Audit fixes 2026-10-06 (owner B): group bookkeeping, rendering order, augmentation, flag filter."""
import json
import math
from pathlib import Path

import pytest
from shapely.geometry import Polygon
import torch

from fastfill.v2.batch import TinyTokenizer, augment_sample, collate_samples, condition_segments
from fastfill.v2.data import filter_rows_by_flags
from fastfill.v2.direct_layout import request_to_condition
from fastfill.v2.geometry import wrap_yaw
from fastfill.v2.matching import certify_group
from fastfill.v2.validation import _geometry, footprint

REAL_ROWS = Path("/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local/validation.jsonl")
NO_AUGMENT = {"rotate90": False, "mirror": False, "shuffle_objects": False,
              "drop_constraints_p": 0., "drop_support_p": 0., "category_only_description_p": 0., "minimal_form_p": 0.}
L_SHAPE = [[0, 0], [6, 0], [6, 3], [3, 3], [3, 5], [0, 5]]


def sample():
    """L-shaped room, support chain, every constraint reference kind, one exchangeable pair."""
    objects = [{"id": "chair_a", "category": "chair", "description": "chair"},
               {"id": "chair_b", "category": "chair", "description": "chair"},
               {"id": "table", "category": "table", "description": "oak table", "support_parent": "floor"},
               {"id": "lamp", "category": "lamp", "description": "lamp", "support_parent": "table"},
               {"id": "rug", "category": "rug", "description": "rug", "support_parent": "floor"}]
    targets = {"chair_a": ([1., 1., 0.], .3), "chair_b": ([5., 1., 0.], 2.), "table": ([2., 2., 0.], 1.),
               "lamp": ([2., 2., .7], -2.5), "rug": ([1.5, 4., 0.], 0.)}
    return {
        "schema_version": "fastfill.v2",
        "condition": {"schema_version": "fastfill.v2", "room": {
            "frame": "right_handed_z_up", "floor_polygon_xy_m": L_SHAPE, "floor_z_m": 0., "height_m": 2.8,
            "floor_known": True, "boundary_known": True, "boundary_quality": "polygon", "room_type": "office",
            "fixed_objects": [{"id": "fixed_0000", "category": "door", "size_local_m": [.1, .9, 2.1],
                               "bottom_center_m": [4.5, 0.05, 0.], "yaw_rad": 1.2}]},
            "objects": objects, "constraints": [
                {"type": "near", "object_id": "lamp", "target_id": "table", "max_distance_m": .5},
                {"type": "between", "object_id": "rug", "target_ids": ["table", "fixed_0000"]},
                {"type": "on", "object_id": "lamp", "parent_id": "table"},
                {"type": "faces_direction", "object_id": "table", "direction_xy": [1, 0]},
                {"type": "keepout", "polygon_xy_m": [[.1, .1], [.4, .1], [.4, .4], [.1, .4]]}]},
        "target": {"objects": [{"id": o["id"], "target_size_local_m": [.6, .5, .7], "bottom_center_m": targets[o["id"]][0],
                                "yaw_rad": targets[o["id"]][1]} for o in objects]},
        "validity": {"position": [[True] * 3 for _ in objects], "size": [[True] * 3 for _ in objects],
                     "yaw": [True for _ in objects], "yaw_symmetry_order": [2, 2, 2, 1, 2],
                     "size_axis_swap_allowed": [False] * len(objects),
                     "exchangeable_group": ["anonymous_0", "anonymous_0", None, None, None]},
        "provenance": {"source": "offline_test", "house_id": "h", "split": "train", "legacy_flags": {}}}


def rendered(condition):
    text = "".join(segment for segment, _ in condition_segments(condition))
    body = text[text.index("{"):text.rindex("}") + 1]
    return text, json.loads(body)


def boxes(s):
    return [_geometry(obj, "target") for obj in s["target"]["objects"]]


def real_row():
    if not REAL_ROWS.is_file():
        pytest.skip("external data disk is not mounted")
    with REAL_ROWS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if sum(obj.get("exchangeable_group") is not None for obj in row["condition"]["objects"]) >= 2:
                return row
    pytest.skip("no legacy-group row in the first rows")


# --- C1/C2: group bookkeeping -------------------------------------------------

def test_collate_exposes_groups_per_request_slot_from_validity():
    s = sample()
    s["target"]["objects"].reverse()
    s["validity"]["exchangeable_group"] = [None, None, None, "anonymous_0", "anonymous_0"]
    b = collate_samples([s], TinyTokenizer())
    assert b["exchangeable_group"] == [["anonymous_0", "anonymous_0", None, None, None]]
    assert all("exchangeable_group" not in obj for obj in b["objects"][0])


def test_legacy_condition_groups_migrate_into_validity_and_leave_the_text():
    s = sample()
    del s["validity"]["exchangeable_group"]
    for obj in s["condition"]["objects"][:2]:
        obj["exchangeable_group"] = "anonymous_7"
    s["target"]["objects"].reverse()
    b = collate_samples([s], TinyTokenizer())
    assert b["exchangeable_group"] == [["anonymous_7", "anonymous_7", None, None, None]]
    assert all("exchangeable_group" not in obj for obj in b["objects"][0] + b["conditions"][0]["objects"])
    assert "exchangeable_group" not in TinyTokenizer().decode(b["input_ids"][0])
    s["validity"]["exchangeable_group"] = [None] * 5
    with pytest.raises(ValueError, match="both condition objects and validity"):
        collate_samples([s], TinyTokenizer())
    # Inference conditions have no target rows, hence no labels (never request slot 0's label for everyone).
    inference = collate_samples([{"condition": s["condition"]}], TinyTokenizer())
    assert inference["exchangeable_group"] == [[None] * 5] and not inference["validity"]["position"].any()


def test_group_labels_must_be_nonempty_strings_or_null():
    s = sample()
    s["validity"]["exchangeable_group"][0] = ""
    with pytest.raises(ValueError, match="exchangeable_group labels"):
        collate_samples([s], TinyTokenizer())


# --- C3: rendering --------------------------------------------------------------

def test_rendered_condition_order_and_stripped_keys():
    s = sample()
    s["condition"]["objects"][0]["exchangeable_group"] = "anonymous_0"
    s["condition"]["room"]["_debug"] = {"x": 1}
    s["condition"]["objects"][1]["attributes"] = {"_hidden": 1, "colour": "red"}
    text, body = rendered(s["condition"])
    assert list(body) == ["schema_version", "room", "constraints", "objects"]
    assert "boundary_quality" not in text and "exchangeable_group" not in text and "_debug" not in text and "_hidden" not in text
    assert body["room"]["boundary_known"] is True and body["room"]["floor_known"] is True
    assert body["objects"][1]["attributes"] == {"colour": "red"} and body["room"]["fixed_objects"]
    assert body["objects"][0] == {"id": "chair_a", "category": "chair", "description": "chair"}
    with pytest.raises(ValueError, match="unknown fields"):
        condition_segments({**s["condition"], "target": {}})


def test_direct_request_renders_the_same_room_fields_as_a_training_rectangle():
    direct = request_to_condition({"room_type": "office", "room_size_m": [4., 3., 2.7], "furniture_list": ["chair"]})
    training = {**sample()["condition"]}
    training["room"] = {k: v for k, v in training["room"].items() if k != "fixed_objects"}  # legacy rows omit an empty list
    _, direct_body = rendered(direct)
    _, training_body = rendered(training)
    assert set(direct_body["room"]) == set(training_body["room"])
    assert "boundary_quality" not in direct_body["room"]
    assert "boundary_quality" in direct["room"]  # only the rendering drops it; the schema field is untouched


# --- C10: augmentation ----------------------------------------------------------

def test_eval_path_and_zero_augmentation_are_identity():
    s = sample()
    plain = collate_samples([s], TinyTokenizer())
    zero = collate_samples([s], TinyTokenizer(), augment=NO_AUGMENT, generator=torch.Generator().manual_seed(0))
    assert torch.equal(plain["input_ids"], zero["input_ids"])
    assert torch.equal(plain["targets"]["position_normalized"], zero["targets"]["position_normalized"])
    assert plain["conditions"] == zero["conditions"] == [sample()["condition"]]
    with pytest.raises(ValueError, match="unknown augmentation"):
        augment_sample(s, {"rotate": True})
    with pytest.raises(ValueError, match="probability"):
        augment_sample(s, {"drop_support_p": 1.5})


def test_augmentation_is_deterministic_given_a_generator():
    s = sample()
    runs = [augment_sample(s, {}, torch.Generator().manual_seed(3)) for _ in range(2)]
    assert runs[0] == runs[1]
    assert runs[0] != augment_sample(s, {}, torch.Generator().manual_seed(4))
    assert s == sample()  # input untouched


@pytest.mark.parametrize("mirror", [False, True])
def test_rigid_augmentation_preserves_containment_distances_and_relative_yaw(mirror):
    s = sample()
    before = boxes(s)
    seen, mirrored_runs = set(), set()
    for seed in range(12):
        a = augment_sample(s, {**NO_AUGMENT, "rotate90": True, "mirror": mirror}, torch.Generator().manual_seed(seed))
        polygon = a["condition"]["room"]["floor_polygon_xy_m"]
        seen.add((tuple(map(tuple, polygon))))
        room = Polygon(polygon)
        mirrored = room.exterior.is_ccw != Polygon(L_SHAPE).exterior.is_ccw  # a reflection flips winding, a rotation never does
        mirrored_runs.add(mirrored)
        assert room.is_valid and math.isclose(room.area, Polygon(L_SHAPE).area)
        assert [min(p[q] for p in polygon) for q in range(2)] == [0, 0]
        after = boxes(a)
        assert all(room.buffer(1e-9).covers(footprint(obj)) for obj in after)
        door = [_geometry(x["condition"]["room"]["fixed_objects"][0], "target") for x in (s, a)]
        for i in range(len(after)):
            for j in range(i + 1, len(after)):
                assert math.isclose(math.dist(before[i]["_pos"], before[j]["_pos"]), math.dist(after[i]["_pos"], after[j]["_pos"]), abs_tol=1e-9)
                relative = [rows[i]["_yaw"] - rows[j]["_yaw"] for rows in (before, after)]
                assert abs(wrap_yaw(relative[1] - (-relative[0] if mirrored else relative[0]))) < 1e-9
        assert math.isclose(math.dist(door[0]["_pos"], before[0]["_pos"]), math.dist(door[1]["_pos"], after[0]["_pos"]), abs_tol=1e-9)
        assert abs(wrap_yaw((door[1]["_yaw"] - after[0]["_yaw"]) - (-1 if mirrored else 1) * (door[0]["_yaw"] - before[0]["_yaw"]))) < 1e-9
        assert room.buffer(1e-9).covers(Polygon(a["condition"]["constraints"][4]["polygon_xy_m"]))
        direction = a["condition"]["constraints"][3]["direction_xy"]
        facing = after[2]["_yaw"]  # the table faces +X before augmentation: the facing/direction angle is invariant
        assert math.isclose(math.cos(facing) * direction[0] + math.sin(facing) * direction[1], math.cos(1.), abs_tol=1e-9)
        assert all(obj["target_size_local_m"] == [.6, .5, .7] for obj in a["target"]["objects"])
        collate_samples([a], TinyTokenizer())
    assert len(seen) >= 3 and mirrored_runs == ({True, False} if mirror else {False})


def test_rotation_swaps_null_coordinates_with_their_validity_flags():
    s = sample()
    s["target"]["objects"][2]["bottom_center_m"] = [2., None, 0.]
    s["validity"]["position"][2] = [True, False, True]
    s["target"]["objects"][3]["yaw_rad"] = None
    s["validity"]["yaw"][3] = False
    for seed in range(8):
        a = augment_sample(s, {**NO_AUGMENT, "rotate90": True}, torch.Generator().manual_seed(seed))
        position = a["target"]["objects"][2]["bottom_center_m"]
        assert [v is not None for v in position] == a["validity"]["position"][2]
        assert a["target"]["objects"][3]["yaw_rad"] is None
        b = collate_samples([a], TinyTokenizer())
        assert b["validity"]["position"][0, 2].tolist() == a["validity"]["position"][2]


def test_shuffle_keeps_alignment_references_and_certifiable_groups():
    s = sample()
    s["validity"]["position"][1] = [False, False, False]  # chair_b label becomes unusable
    certify_group(s["condition"]["objects"], s["condition"]["constraints"], [0, 1])
    original = {o["description"]: (t["bottom_center_m"], t["yaw_rad"]) for o, t in zip(s["condition"]["objects"], s["target"]["objects"])}
    orders = set()
    for seed in range(8):
        a = augment_sample(s, {**NO_AUGMENT, "shuffle_objects": True}, torch.Generator().manual_seed(seed))
        objects, targets = a["condition"]["objects"], a["target"]["objects"]
        ids = [o["id"] for o in objects]
        orders.add(tuple(o["description"] for o in objects))
        assert ids == [f"obj_{i:04d}" for i in range(5)] and [t["id"] for t in targets] == ids
        by_id = {o["id"]: o for o in objects}
        for obj, target in zip(objects, targets):
            if obj["category"] != "chair":
                assert (target["bottom_center_m"], target["yaw_rad"]) == original[obj["description"]]
        lamp = next(o for o in objects if o["category"] == "lamp")
        assert by_id[lamp["support_parent"]]["category"] == "table"
        assert next(o for o in objects if o["category"] == "table")["support_parent"] == "floor"
        kinds = {c["type"]: c for c in a["condition"]["constraints"]}
        assert by_id[kinds["near"]["object_id"]]["category"] == "lamp" and by_id[kinds["near"]["target_id"]]["category"] == "table"
        assert by_id[kinds["on"]["parent_id"]]["category"] == "table"
        assert by_id[kinds["between"]["object_id"]]["category"] == "rug"
        assert by_id[kinds["between"]["target_ids"][0]]["category"] == "table" and kinds["between"]["target_ids"][1] == "fixed_0000"
        assert by_id[kinds["faces_direction"]["object_id"]]["category"] == "table"
        b = collate_samples([a], TinyTokenizer())
        groups = b["exchangeable_group"][0]
        members = [i for i, g in enumerate(groups) if g == "anonymous_0"]
        assert [objects[i]["category"] for i in members] == ["chair", "chair"] and len(members) == 2
        certify_group(b["objects"][0], b["conditions"][0]["constraints"], members)
        invalid = [i for i, row in enumerate(a["validity"]["position"]) if not any(row)]
        assert len(invalid) == 1 and objects[invalid[0]]["category"] == "chair"
        assert a["validity"]["yaw_symmetry_order"] == [1 if o["category"] == "lamp" else 2 for o in objects]
        assert b["yaw_symmetry_order"][0].tolist() == a["validity"]["yaw_symmetry_order"]
        assert not b["validity"]["position"][0, invalid[0]].any()
    assert len(orders) > 1


def test_shuffle_rejects_misaligned_validity_and_colliding_ids():
    s = sample()
    s["validity"]["yaw"] = [True]
    with pytest.raises(ValueError, match="aligned with target objects"):
        augment_sample(s, {**NO_AUGMENT, "shuffle_objects": True}, torch.Generator().manual_seed(0))
    s = sample()
    s["condition"]["room"]["fixed_objects"][0]["id"] = "obj_0002"
    with pytest.raises(ValueError, match="collide"):
        augment_sample(s, {**NO_AUGMENT, "shuffle_objects": True}, torch.Generator().manual_seed(0))


def test_drop_augmentations_follow_probabilities():
    s = sample()
    dropped = augment_sample(s, {**NO_AUGMENT, "drop_constraints_p": 1., "drop_support_p": 1., "category_only_description_p": 1.})
    assert dropped["condition"]["constraints"] == []
    assert all("support_parent" not in o and o["description"] == o["category"] for o in dropped["condition"]["objects"])
    assert dropped["target"] == s["target"] and dropped["validity"] == s["validity"]
    collate_samples([dropped], TinyTokenizer())
    kept = augment_sample(s, NO_AUGMENT)
    assert kept["condition"] == s["condition"]


def test_mirror_refuses_asset_local_metadata_and_rotate90_leaves_openings_unturned():
    s = sample()
    s["condition"]["room"]["fixed_objects"][0]["semantic_front_local"] = [1., 0., 0.]
    with pytest.raises(ValueError, match="mirror cannot transform"):
        for seed in range(8):
            augment_sample(s, {**NO_AUGMENT, "mirror": True}, torch.Generator().manual_seed(seed))
    s = sample()
    s["condition"]["room"]["openings"] = [{"kind": "door"}]
    for seed in range(8):  # round 10 review: left unturned instead of raising inside the DataLoader
        out = augment_sample(s, {**NO_AUGMENT, "rotate90": True}, torch.Generator().manual_seed(seed))
        assert out["condition"]["room"] == s["condition"]["room"] and out["target"] == s["target"]


# --- C9 loader side ---------------------------------------------------------------

def test_filter_rows_by_flags_uses_legacy_flag_truthiness():
    rows = [{"provenance": {"legacy_flags": {"oob_objects": 5}}}, {"provenance": {"legacy_flags": {"oob_objects": 0}}},
            {"provenance": {"legacy_flags": {"fixed_collision": 1}}}, {"provenance": {}}]
    assert filter_rows_by_flags(rows, ["oob_objects", "fixed_collision", "overlapping_furniture"]) == rows[1:2] + rows[3:]
    assert filter_rows_by_flags(rows, []) == rows
    with pytest.raises(ValueError, match="exclude_flags"):
        filter_rows_by_flags(rows, "oob_objects")


# --- Real multisource row ----------------------------------------------------------

def test_real_legacy_row_collates_renders_and_augments():
    row = real_row()
    tokenizer = TinyTokenizer()
    b = collate_samples([row], tokenizer, max_length=10**7, max_objects=10**4)
    text = tokenizer.decode(b["input_ids"][0])
    assert "exchangeable_group" not in text and "boundary_quality" not in text
    assert text.index('"room":') < text.index('"constraints":') < text.index('"objects":')
    assert text.rstrip().endswith("]}\nAssistant:")
    expected = [obj.get("exchangeable_group") for obj in row["condition"]["objects"]]
    assert b["exchangeable_group"] == [expected] and sum(g is not None for g in expected) >= 2
    assert all("exchangeable_group" not in obj for obj in b["objects"][0])
    before = boxes(row)
    for seed in range(3):
        a = augment_sample(row, {}, torch.Generator().manual_seed(seed))
        ab = collate_samples([a], tokenizer, max_length=10**7, max_objects=10**4)
        groups = ab["exchangeable_group"][0]
        # Drops may merge requests into wider recomputed groups but never lose a build-time member.
        assert sum(g is not None for g in groups) >= sum(g is not None for g in expected)
        for label in {g for g in groups if g}:
            members = [i for i, g in enumerate(groups) if g == label]
            certify_group(ab["objects"][0], ab["conditions"][0]["constraints"], members)
        assert ab["validity"]["position"].sum() == b["validity"]["position"].sum()
        assert ab["validity"]["size"].sum() == b["validity"]["size"].sum()
        rigid = augment_sample(row, {"shuffle_objects": False, "mirror": False}, torch.Generator().manual_seed(seed))
        after = boxes(rigid)
        assert all(math.isclose(math.dist(before[i]["_pos"], before[j]["_pos"]), math.dist(after[i]["_pos"], after[j]["_pos"]), abs_tol=1e-6)
                   for i in range(len(after)) for j in range(i + 1, len(after)))
        assert all(abs(wrap_yaw((after[i]["_yaw"] - after[0]["_yaw"]) - (before[i]["_yaw"] - before[0]["_yaw"]))) < 1e-9 for i in range(len(after)))
        assert all(x["target_size_local_m"] == y["target_size_local_m"] for x, y in zip(row["target"]["objects"], rigid["target"]["objects"]))
