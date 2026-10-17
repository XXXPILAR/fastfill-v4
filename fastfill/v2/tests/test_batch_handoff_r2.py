"""Round 2 (owner B2): shared three-field projection (K5), box-symmetry collate tier (K1), hand-off place inference (K6)."""
import json
import os
from pathlib import Path

import pytest
import torch

from fastfill.v2.batch import (AUGMENT_DEFAULTS, TinyTokenizer, augment_sample, collate_samples, condition_segments,
                               is_axis_aligned_rectangle, render_minimal_condition, tokenize_condition)
from fastfill.v2.direct_layout import export_handoff, infer_support, layout_to_roomgenbench, request_to_condition

REAL_ROWS = Path("/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local/validation.jsonl")
REFERENCE = Path(os.environ.get("FASTFILL_ROOMGENBENCH_ROOT") or Path(__file__).resolve().parents[3] / "RoomGenBench")
ONLY_MINIMAL = {"rotate90": False, "mirror": False, "shuffle_objects": False, "drop_constraints_p": 0.,
                "drop_support_p": 0., "category_only_description_p": 0., "minimal_form_p": 1.}


def row():
    """Rectangular training row (polygon starts at another corner) with everything the projection must drop."""
    objects = [{"id": "obj_0000", "category": "chair", "description": "wooden chair", "support_parent": "floor"},
               {"id": "obj_0001", "category": "chair", "description": "wooden chair"},
               {"id": "obj_0002", "category": "desk", "description": "oak desk", "support_parent": "floor"},
               {"id": "obj_0003", "category": "lamp", "description": "lamp", "support_parent": "obj_0002"}]
    targets = [([1., 1., 0.], .3), ([3., 1., 0.], 2.), ([2., 2., 0.], 1.), ([2., 2., .75], -2.5)]
    return {"schema_version": "fastfill.v2",
            "condition": {"schema_version": "fastfill.v2", "room": {
                "frame": "right_handed_z_up", "floor_polygon_xy_m": [[4.5, 3.25], [0., 3.25], [0., 0.], [4.5, 0.]],
                "floor_z_m": 0., "height_m": 2.8, "floor_known": True, "boundary_known": True,
                "boundary_quality": "polygon", "room_type": "office",
                "fixed_objects": [{"id": "fixed_0000", "category": "door", "size_local_m": [.1, .9, 2.1],
                                   "bottom_center_m": [4.45, 1., 0.], "yaw_rad": 0.}]},
                "objects": objects,
                "constraints": [{"type": "near", "object_id": "obj_0003", "target_id": "obj_0002", "max_distance_m": .5}]},
            "target": {"objects": [{"id": o["id"], "target_size_local_m": [.6, .5, .7], "bottom_center_m": p, "yaw_rad": y}
                                   for o, (p, y) in zip(objects, targets)]},
            "validity": {"position": [[True] * 3] * 4, "size": [[True] * 3] * 4, "yaw": [True] * 4,
                         "yaw_symmetry_order": [4, 2, 2, 1], "size_axis_swap_allowed": [True, False, False, False],
                         "exchangeable_group": [None] * 4},
            "provenance": {"source": "offline_test", "house_id": "h", "split": "train", "legacy_flags": {}}}


def request_for(condition):
    """The direct three-field request a user would send for this room (origin at the lower corner)."""
    room = condition["room"]
    size = [max(p[q] for p in room["floor_polygon_xy_m"]) for q in range(2)]
    return {"room_type": room["room_type"], "room_size_m": size + ([room["height_m"]] if room.get("height_m") else []),
            "furniture_list": [{key: obj[key] for key in ("id", "category", "description")} for obj in condition["objects"]]}


# --- K5: one renderer for the direct request, the augmentation and minimal evaluation -------------

def test_minimal_projection_renders_byte_identical_to_the_direct_request():
    condition = row()["condition"]
    minimal = render_minimal_condition(condition)
    direct = request_to_condition(request_for(condition))
    assert condition_segments(minimal) == condition_segments(direct)
    assert tokenize_condition(minimal, TinyTokenizer()) == tokenize_condition(direct, TinyTokenizer())
    assert minimal["room"]["floor_polygon_xy_m"] == [[0., 0.], [4.5, 0.], [4.5, 3.25], [0., 3.25]]
    assert minimal["room"]["boundary_known"] is direct["room"]["boundary_known"] is True
    assert minimal["room"]["floor_known"] is direct["room"]["floor_known"] is True
    assert minimal["constraints"] == [] and "fixed_objects" not in minimal["room"]
    assert all(set(obj) == {"id", "category", "description"} for obj in minimal["objects"])
    text = "".join(segment for segment, _ in condition_segments(direct))
    assert "boundary_quality" not in text and direct["room"]["boundary_quality"] == "explicit_rectangular_request"
    assert row()["condition"] == condition  # the projection never mutates its input


def test_minimal_projection_matches_direct_requests_on_real_rectangular_rows():
    if not REAL_ROWS.is_file():
        pytest.skip("external data disk is not mounted")
    compared = 0
    with REAL_ROWS.open() as stream:
        for line in stream:
            condition = json.loads(line)["condition"]
            room = condition["room"]
            if (not is_axis_aligned_rectangle(room["floor_polygon_xy_m"]) or not room.get("room_type")
                    or room.get("floor_z_m") != 0 or room.get("floor_known") is not True):
                continue
            assert condition_segments(render_minimal_condition(condition)) == condition_segments(request_to_condition(
                request_for(condition), max_objects=max(128, len(condition["objects"]))))
            compared += 1
            if compared == 50:
                break
    assert compared == 50


def test_axis_aligned_rectangle_test_uses_a_one_centimetre_tolerance():
    assert is_axis_aligned_rectangle([[0, 0], [4, 0.009], [4, 3], [0.01, 3]])
    assert not is_axis_aligned_rectangle([[0, 0], [4, 0.02], [4, 3], [0, 3]])
    assert not is_axis_aligned_rectangle([[0, 0], [4, 0], [4, 3], [2, 3.5], [0, 3]])
    assert not is_axis_aligned_rectangle([[2, 0], [4, 2], [2, 4], [0, 2]])  # rotated square
    assert not is_axis_aligned_rectangle([[0, 0], [4, 0], [3, 3], [1, 3]])  # trapezoid


def test_minimal_form_replaces_the_condition_drops_floor_fixed_z_and_regroups():
    s = row()
    full = collate_samples([augment_sample(s, {**ONLY_MINIMAL, "minimal_form_p": 0.})], TinyTokenizer())
    a = augment_sample(s, ONLY_MINIMAL, torch.Generator().manual_seed(0))
    minimal = collate_samples([a], TinyTokenizer())
    assert a["condition"] == render_minimal_condition(s["condition"]) and a["target"] == s["target"]
    # The floor declaration fixed obj_0000/obj_0002 z in the full form; the minimal form learns it.
    assert full["fixed_position_mask"][0, :, 2].tolist() == [True, False, True, False]
    assert not minimal["fixed_position_mask"].any()
    # Removing support_parent made the two wooden chairs identical requests: matched by geometry now.
    assert full["exchangeable_group"] == [[None] * 4]
    assert minimal["exchangeable_group"] == [["anonymous_0", "anonymous_0", None, None]]
    assert torch.equal(full["targets"]["position_normalized"], minimal["targets"]["position_normalized"])
    assert s == row()


def test_minimal_form_needs_a_rectangle_and_defaults_to_one_half():
    s = row()
    s["condition"]["room"]["floor_polygon_xy_m"] = [[0., 0.], [4.5, 0.], [4.5, 3.25], [2., 3.25], [2., 4.], [0., 4.]]
    assert augment_sample(s, ONLY_MINIMAL)["condition"] == s["condition"]
    assert AUGMENT_DEFAULTS["minimal_form_p"] == .5
    with pytest.raises(ValueError, match="probability"):
        augment_sample(row(), {"minimal_form_p": 2.})
    minimal_runs = {"fixed_objects" not in augment_sample(row(), {}, torch.Generator().manual_seed(seed))["condition"]["room"]
                    for seed in range(16)}
    assert minimal_runs == {True, False}


# --- K1 collate side ------------------------------------------------------------------------

def test_collate_exposes_size_axis_swap_per_request_slot_and_defaults_old_rows_to_false():
    s = row()
    s["target"]["objects"].reverse()
    s["validity"] = {key: list(reversed(value)) for key, value in s["validity"].items()}
    batch = collate_samples([s], TinyTokenizer())
    assert batch["size_axis_swap_allowed"].dtype == torch.bool
    assert batch["size_axis_swap_allowed"].tolist() == [[True, False, False, False]]
    old = row()
    del old["validity"]["size_axis_swap_allowed"]
    assert collate_samples([old], TinyTokenizer())["size_axis_swap_allowed"].tolist() == [[False] * 4]
    short = {**old, "condition": {**old["condition"], "objects": old["condition"]["objects"][:2], "constraints": []},
             "target": {"objects": old["target"]["objects"][:2]}, "validity": {k: v[:2] for k, v in old["validity"].items()}}
    padded = collate_samples([row(), short], TinyTokenizer())
    assert padded["size_axis_swap_allowed"].tolist() == [[True, False, False, False], [False] * 4]
    bad = row()
    bad["validity"]["size_axis_swap_allowed"][1] = 1
    with pytest.raises(ValueError, match="size_axis_swap_allowed"):
        collate_samples([bad], TinyTokenizer())


def test_shuffle_carries_size_axis_swap_with_its_object():
    for seed in range(6):
        a = augment_sample(row(), {**ONLY_MINIMAL, "minimal_form_p": 0., "shuffle_objects": True},
                           torch.Generator().manual_seed(seed))
        batch = collate_samples([a], TinyTokenizer())
        swapped = [obj.get("support_parent") == "floor" and obj["category"] == "chair" for obj in a["condition"]["objects"]]
        assert batch["size_axis_swap_allowed"][0].tolist() == swapped


# --- K6: hand-off place inference ---------------------------------------------------------------

def scene(*boxes, declared=None):
    """Known 4 x 3 x 2.8 m room; boxes are (id, category, size, bottom_center, yaw)."""
    condition = request_to_condition({"room_type": "living_room", "room_size_m": [4., 3., 2.8], "furniture_list": [
        {"id": ident, "category": category} for ident, category, *_ in boxes]})
    for obj in condition["objects"]:
        if obj["id"] in (declared or {}):
            obj["support_parent"] = declared[obj["id"]]
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": ident, "target_size_local_m": size, "bottom_center_m": bottom, "yaw_rad": yaw}
        for ident, _, size, bottom, yaw in boxes]}
    return condition, layout


def placement(condition, layout):
    return {o["id"]: (o["place"], o["place_id"], o["support_status"]) for o in layout_to_roomgenbench(condition, layout)["objects"]}


def test_shelf_middle_tier_objects_are_never_declared():
    shelf = ("shelf", "shelf", [1., .4, 2.], [1., .2, 0.], 0.)
    condition, layout = scene(shelf, ("book_mid", "book", [.2, .15, .25], [1., .2, 1.], 0.),
                              ("book_low", "book", [.2, .15, .25], [.8, .2, .01], 0.),
                              ("book_top", "book", [.2, .15, .25], [1.2, .2, 2.01], 0.), declared={"shelf": "floor"})
    places = placement(condition, layout)
    assert places["shelf"] == ("floor", "floor", "declared")
    assert places["book_mid"] == ("unknown", None, "unknown")        # middle tier: neither floor nor the shelf top
    assert places["book_low"] == ("floor", "floor", "inferred")      # bottom tier rests within 2 cm of the floor
    assert places["book_top"] == ("on_object", "shelf", "inferred")  # top surface: a candidate, never a declaration


def test_painting_near_plant_is_an_inferred_candidate_or_unknown_never_wall():
    plant = ("plant", "plant", [.5, .5, 1.2], [.3, .3, 0.], 0.)
    painting = ("painting", "painting", [.04, .8, .6], [.3, .3, 1.22], 0.)
    places = placement(*scene(plant, painting))
    assert places["painting"] == ("on_object", "plant", "inferred")
    assert places["plant"] == ("floor", "floor", "inferred")
    beside = ("painting", "painting", [.04, .8, .6], [.3, .6, 1.22], 0.)  # centre outside the plant footprint
    assert placement(*scene(plant, beside))["painting"] == ("unknown", None, "unknown")
    hung = ("painting", "painting", [.04, .8, .6], [.3, .3, 1.5], 0.)
    assert placement(*scene(plant, hung))["painting"] == ("unknown", None, "unknown")
    assert placement(*scene(plant, hung, declared={"painting": "wall"}))["painting"] == ("wall", "wall", "declared")


def test_on_object_candidate_is_the_highest_strictly_lower_box_inside_a_rotated_footprint():
    table = ("table", "table", [1.6, .8, .75], [2., 1.5, 0.], .6)
    tray = ("tray", "tray", [.4, .3, .02], [2., 1.5, .75], 0.)
    cup = ("cup", "cup", [.08, .08, .1], [2., 1.5, .77], 0.)
    places = placement(*scene(table, tray, cup))
    assert places["cup"] == ("on_object", "tray", "inferred") and places["tray"] == ("on_object", "table", "inferred")
    # Two papers on the same plane never point at each other; both rest on the table.
    papers = [("paper_a", "paper", [.3, .2, .001], [1.9, 1.45, .75], 0.), ("paper_b", "paper", [.3, .2, .001], [2., 1.5, .75], 0.)]
    places = placement(*scene(table, *papers))
    assert places["paper_a"][1:] == places["paper_b"][1:] == ("table", "inferred")
    # Inside the table's axis-aligned extent but outside its yawed footprint: no candidate.
    corner = ("cup", "cup", [.08, .08, .1], [2.75, 1.85, .75], 0.)
    assert placement(*scene(table, corner))["cup"] == ("unknown", None, "unknown")
    assert infer_support({"id": "x", "bottom_center_m": [0., 0., 3.]}, None, [], None) == (None, "unknown")


def test_an_inference_never_closes_a_cycle_with_a_declared_parent():
    # Audit C4: chair declared on table, table undeclared and predicted on the chair's top; before: table -> chair.
    chair = ("chair", "chair", [1., 1., 1.], [2., 1.5, 0.], 0.)
    table = ("table", "table", [1., 1., 1.], [2., 1.5, 1.], 0.)
    for boxes in ((chair, table), (table, chair)):
        places = placement(*scene(*boxes, declared={"chair": "table"}))
        assert places == {"chair": ("on_object", "table", "declared"), "table": ("unknown", None, "unknown")}
    # The cycle-closing box is skipped, not the rule: the next lower candidate (the table) still supports the cup.
    desk = ("desk", "desk", [1.6, .8, .75], [2., 1.5, 0.], 0.)
    tray = ("tray", "tray", [.4, .3, .02], [2., 1.5, .75], 0.)
    cup = ("cup", "cup", [.08, .08, .1], [2., 1.5, .77], 0.)
    assert placement(*scene(desk, tray, cup, declared={"tray": "cup"}))["cup"] == ("on_object", "desk", "inferred")


def test_place_and_support_status_reach_assets_and_the_real_assembler_receipt(tmp_path):
    shelf = ("shelf", "shelf", [1., .4, 2.], [1., .2, 0.], 0.)
    plant = ("plant", "plant", [.5, .5, 1.2], [3., .3, 0.], 0.)
    condition, layout = scene(shelf, ("book", "book", [.2, .15, .25], [1., .2, 1.], 0.), plant,
                              ("painting", "painting", [.04, .8, .6], [3., .3, 1.22], 0.), declared={"shelf": "floor"})
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    expected = {"shelf": ("floor", "declared"), "book": ("unknown", "unknown"), "plant": ("floor", "inferred"),
                "painting": ("on_object", "inferred")}
    registry = {entry["type"]: entry for entry in map(json.loads, (handoff / "assets.jsonl").read_text().splitlines())}
    assert {key: (entry["place"], entry["support_status"]) for key, entry in registry.items()} == expected
    downstream = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assert {o["id"]: (o["place"], o["support_status"]) for o in downstream["objects"]} == expected
    if not (REFERENCE / "bench" / "assemble.py").is_file():
        pytest.skip("RoomGenBench reference assembler is unavailable; set FASTFILL_ROOMGENBENCH_ROOT")
    from fastfill.v2.roomgenbench import assemble_handoff
    receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    assert {o["id"]: (o["place"], o["support_status"]) for o in receipt["objects"]} == expected
    assert {o["id"]: o["support_parent"] for o in receipt["objects"]} == {"shelf": "floor", "book": None, "plant": "floor", "painting": "plant"}
    assert receipt["assembly_complete"] and (tmp_path / "assembled" / f'{receipt["scene_key"]}.glb').is_file()
