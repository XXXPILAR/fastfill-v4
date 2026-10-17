"""Audit fix 2026-10-06 (C11): room shell, openings, place and shared asset keys in the handoff.

The real RoomGenBench assembler is imported with ROOT/INPUTS redirected to a
scratch directory and fed a converted real validation room (IL3D 3D-FRONT
bedroom: L-shaped 6-vertex polygon, one door, two windows, four floor objects).
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math

import numpy as np
import pytest
import trimesh

from fastfill.v2.direct_layout import asset_key, export_handoff, layout_to_roomgenbench, request_to_condition
from fastfill.v2.tests.test_roomgenbench import REFERENCE, corners, points_for_node


# selected-v3.2 validation row 88, provenance il3d:b5846dce-86d4-11f0-a478-60cf84ae2082 (condition + target).
REAL_CONDITION = {"schema_version": "fastfill.v2", "room": {"frame": "right_handed_z_up",
    "floor_polygon_xy_m": [[2.7776, 3.5566], [0.0, 3.5566], [0.0, 0.9766], [0.9546, 0.9766], [0.9546, 0.0], [2.7776, 0.0]],
    "boundary_known": True, "boundary_quality": "polygon", "floor_known": True, "floor_z_m": 0.0, "height_m": 2.6,
    "room_type": "bedroom", "fixed_objects": [
        {"id": "fixed_0000", "category": "door", "size_local_m": [0.8, 0.24, 2.0391], "bottom_center_m": [0.5821, 3.6771, 0.0], "yaw_rad": 0.0},
        {"id": "fixed_0001", "category": "window", "size_local_m": [0.57064, 0.1, 1.3584], "bottom_center_m": [0.9546, 0.3942, 0.85], "yaw_rad": -1.5707963267948966},
        {"id": "fixed_0002", "category": "window", "size_local_m": [1.34975, 0.1, 1.3584], "bottom_center_m": [1.789295, 0.0, 0.85], "yaw_rad": 0.0}]},
    "objects": [{"id": "obj_0000", "category": "desk", "description": "desk", "support_parent": "floor"},
                {"id": "obj_0001", "category": "dining chair", "description": "dining chair", "support_parent": "floor"},
                {"id": "obj_0002", "category": "wardrobe", "description": "wardrobe", "support_parent": "floor"},
                {"id": "obj_0003", "category": "king size bed", "description": "king size bed", "support_parent": "floor"}],
    "constraints": []}
REAL_LAYOUT = {"schema_version": "fastfill.v2", "objects": [
    {"id": "obj_0000", "target_size_local_m": [0.6001899838447571, 1.3500990271568298, 0.7599999904632568], "bottom_center_m": [2.409455, 0.700208, 0.0], "yaw_rad": -3.141592653589793},
    {"id": "obj_0001", "target_size_local_m": [0.47391000390052795, 0.4409179985523224, 0.7990180253982544], "bottom_center_m": [1.77713, 0.506304, 0.0], "yaw_rad": 0.0},
    {"id": "obj_0002", "target_size_local_m": [0.5558669865131378, 1.2570939660072327, 2.017936944961548], "bottom_center_m": [2.200992, 3.350458, 0.0], "yaw_rad": -1.5707963267948966},
    {"id": "obj_0003", "target_size_local_m": [2.2521300315856934, 1.8505860567092896, 1.1269512423314154], "bottom_center_m": [1.098646, 2.128323, 0.0], "yaw_rad": -1.5707963267948966}]}
POLYGON = REAL_CONDITION["room"]["floor_polygon_xy_m"]
EDGES = list(zip(POLYGON, POLYGON[1:] + POLYGON[:1]))


def real():
    return deepcopy(REAL_CONDITION), deepcopy(REAL_LAYOUT)


def reference_assembler(root):
    """The real bench/assemble.py with its fixed paths redirected to a scratch root."""
    spec = importlib.util.spec_from_file_location("_rgb_fix_test", REFERENCE / "bench" / "assemble.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.INPUTS = root, root / "bench" / "inputs"
    (module.INPUTS / "scenes").mkdir(parents=True)
    return module


def gltf_to_sage(points):
    return np.column_stack([points[:, 0], -points[:, 2], points[:, 1]])


def along_and_offset(points_xy, a, b):
    a, b = np.array(a), np.array(b)
    u = (b - a) / np.linalg.norm(b - a)
    n = np.array([-u[1], u[0]])
    return (points_xy - a) @ u, (points_xy - a) @ n


def test_shell_walls_trace_polygon_edges_and_openings_sit_on_nearest_wall():
    condition, layout = real()
    room = layout_to_roomgenbench(condition, layout)["room"]
    assert [w["id"] for w in room["walls"]] == [f"wall_{i:02d}" for i in range(6)]
    for wall, (a, b) in zip(room["walls"], EDGES):
        assert [wall["start_point"][k] for k in "xyz"] == a + [0.]
        assert [wall["end_point"][k] for k in "xyz"] == b + [0.]
        assert wall["height"] == 2.6 and wall["thickness"] == .1
    door, = room["doors"]
    assert door["id"] == "fixed_0000" and door["wall_id"] == "wall_00"  # the y=3.5566 wall, 0.12 m behind it
    assert door["position_on_wall"] == pytest.approx((2.7776 - 0.5821) / 2.7776)
    assert door["width"] == pytest.approx(.8) and door["height"] == pytest.approx(2.0391)
    assert "sill_height" not in door
    first, second = room["windows"]
    assert (first["wall_id"], second["wall_id"]) == ("wall_03", "wall_04")
    assert first["position_on_wall"] == pytest.approx((0.9766 - 0.3942) / 0.9766)
    assert first["width"] == pytest.approx(.57064)  # local X runs along the wall after the -pi/2 yaw
    assert second["position_on_wall"] == pytest.approx((1.789295 - 0.9546) / (2.7776 - 0.9546))
    assert second["width"] == pytest.approx(1.34975) and second["sill_height"] == pytest.approx(.85)
    assert first["height"] == pytest.approx(1.3584)


def test_real_room_assembles_with_the_reference_assembler_and_boxes_match_previous_mapping(tmp_path):
    condition, layout = real()
    scene = layout_to_roomgenbench(condition, layout)
    module = reference_assembler(tmp_path / "root")
    (module.INPUTS / "scenes" / f'{scene["scene_key"]}.json').write_text(json.dumps(scene))
    assert module.assemble("layout_boxes", scene["scene_key"], tmp_path / "out") == {"box": 4}
    glb = tmp_path / "out" / f'{scene["scene_key"]}.glb'
    meta = json.loads((tmp_path / "out" / f'{scene["scene_key"]}.json').read_text())
    assert [w["name"] for w in meta["walls"]] == [f"shell_wall_{i}" for i in range(6)]
    assert all(o["place"] == "floor" for o in meta["objects"])
    loaded = trimesh.load(glb, force="scene", process=False)
    nodes = set(loaded.graph.nodes_geometry)
    for index, (a, b) in enumerate(EDGES):
        segments = [n for n in nodes if n.startswith(f"shell_wall_{index}_s")]
        assert segments
        points = gltf_to_sage(np.vstack([points_for_node(glb, n) for n in segments]))
        along, offset = along_and_offset(points[:, :2], a, b)
        # Interior face on the edge line, outer face one thickness outside the room, corners closed by t.
        assert sorted(set(np.round(np.abs(offset), 6))) == [0., .1]
        assert along.min() == pytest.approx(-.1) and along.max() == pytest.approx(math.dist(a, b) + .1)
        assert points[:, 2].min() == pytest.approx(0.) and points[:, 2].max() == pytest.approx(2.6)
        if index == 0:
            cut = [n for n in segments if points_for_node(glb, n)[:, 1].min() > 1.]  # the lintel above the door
            assert len(cut) == 1
            lintel = gltf_to_sage(points_for_node(glb, cut[0]))
            assert lintel[:, 2].min() == pytest.approx(2.0391)
            assert np.sort(np.unique(np.round(lintel[:, 0], 6))) == pytest.approx([0.5821 - .4, 0.5821 + .4])
    panel = gltf_to_sage(points_for_node(glb, "shell_wall_0_door"))
    assert panel[:, 0].mean() == pytest.approx(0.5821) and panel[:, 2].max() == pytest.approx(2.0391)
    # The requested boxes are unchanged versus the previous exporter: same world corners from the same mapping.
    for index, obj in enumerate(layout["objects"]):
        downstream = scene["objects"][index]
        w, d, h = obj["target_size_local_m"]
        assert downstream["dimensions"] == {"width": d, "length": w, "height": h}
        assert [downstream["position"][k] for k in "xyz"] == obj["bottom_center_m"]
        assert downstream["rotation"]["z"] == pytest.approx(math.degrees((obj["yaw_rad"] - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi))
        actual = points_for_node(glb, f"obj_{index:03d}")
        expected = [[x, z, -y] for x, y, z in corners(obj["bottom_center_m"], obj["target_size_local_m"], obj["yaw_rad"])]
        rounded = lambda p: sorted(set(tuple(np.round(x, 6)) for x in p))
        assert rounded(actual) == rounded(expected)


def test_dynamic_entry_uses_reference_shell_lifted_to_floor_and_keeps_windows_as_proxies(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = real()
    lift = .5
    condition["room"]["floor_z_m"] = lift
    for fixed in condition["room"]["fixed_objects"]:
        fixed["bottom_center_m"][2] += lift
    for obj in layout["objects"]:
        obj["bottom_center_m"][2] += lift
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    assert receipt["counts"] == {"box": 4} and receipt["fixed_bbox_proxies"] == 2  # door is shell, windows proxies
    assert [w["name"] for w in receipt["walls"]] == [f"shell_wall_{i}" for i in range(6)]
    assert receipt["room"]["shell_geometry_kind"] == "reference_build_shell_polygon_walls"
    assert receipt["room"]["height"] == 2.6 and receipt["room"]["height_source"] == "condition"
    assert all(o["place"] == "floor" and o["support_status"] == "declared" for o in receipt["objects"])
    glb = tmp_path / "assembled" / f'{receipt["scene_key"]}.glb'
    nodes = set(trimesh.load(glb, force="scene", process=False).graph.nodes_geometry)
    assert "fixed_bbox_000" not in nodes and {"fixed_bbox_001", "fixed_bbox_002", "shell_floor", "shell_wall_0_door"} <= nodes
    floor = gltf_to_sage(points_for_node(glb, "shell_floor"))
    assert floor[:, 2].min() == pytest.approx(lift - .05) and floor[:, 2].max() == pytest.approx(lift)  # slab top at floor
    wall = gltf_to_sage(points_for_node(glb, "shell_wall_1_s0"))
    assert wall[:, 2].min() == pytest.approx(lift) and wall[:, 2].max() == pytest.approx(lift + 2.6)
    assert json.loads((handoff / "roomgenbench_scene.json").read_text())["room"]["walls"][0]["start_point"]["z"] == lift


@pytest.mark.parametrize("category,kind", [("doorframe", "doors"), ("Window frame", "windows"), ("wardrobe", None)])
def test_opening_classification_is_by_category_substring(category, kind):
    condition, layout = real()
    condition["room"]["fixed_objects"] = [{**condition["room"]["fixed_objects"][0], "category": category}]
    room = layout_to_roomgenbench(condition, layout)["room"]
    assert [o["id"] for o in room["doors"]] == (["fixed_0000"] if kind == "doors" else [])
    assert [o["id"] for o in room["windows"]] == (["fixed_0000"] if kind == "windows" else [])


def test_wall_height_defaults_when_unknown_and_closed_polygon_adds_no_zero_length_wall():
    condition, layout = real()
    condition["room"]["height_m"] = None
    condition["room"]["floor_polygon_xy_m"] = POLYGON + [POLYGON[0]]
    room = layout_to_roomgenbench(condition, layout)["room"]
    assert len(room["walls"]) == 6 and {w["height"] for w in room["walls"]} == {2.7}
    assert room["dimensions"]["height"] is None and room["ceiling_height"] is None


def test_minimal_request_floor_standing_objects_export_place_floor_for_threshold_lookup(tmp_path):
    threshold = {"floor": 31, "wall": 31, "on_object": 30}  # RoomGenBench/methods/holodeck_retrieval/retrieve.py
    condition = request_to_condition({"room_type": "living_room", "room_size_m": [5., 4.],
                                      "furniture_list": ["sofa", {"category": "chair", "count": 2}, "rug"]})
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": obj["id"], "target_size_local_m": [1., .8, .5], "bottom_center_m": [1. + index, 2., z], "yaw_rad": 0.}
        for index, (obj, z) in enumerate(zip(condition["objects"], [0., .02, -.02, .005]))]}
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    registry = [json.loads(line) for line in (handoff / "assets.jsonl").read_text().splitlines()]
    assert [threshold[row["place"]] for row in registry] == [31, 31, 31]
    assert {row["support_status"] for row in registry} == {"inferred"}
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assert all(o["place_id"] == "floor" for o in scene["objects"])
    assert [o["asset_key"] for o in scene["objects"]][1:3] == [asset_key("chair", "chair")] * 2
    assert asset_key("Dining Chair", "x") == "dining_chair_" + hashlib.sha1(b"x").hexdigest()[:8]


def test_floor_contact_needs_a_known_floor_and_declared_support_wins():
    condition, layout = real()
    condition["objects"] = [{k: v for k, v in o.items() if k != "support_parent"} for o in condition["objects"]]
    condition["objects"][1]["support_parent"] = "obj_0000"  # declared on the desk although it sits at z=0
    layout["objects"][2]["bottom_center_m"][2] = .021
    objects = layout_to_roomgenbench(condition, layout)["objects"]
    assert [o["place_id"] for o in objects] == ["floor", "obj_0000", None, "floor"]
    assert [o["support_status"] for o in objects] == ["inferred", "declared", "unknown", "inferred"]
    condition["room"].update({"floor_z_m": None, "floor_known": False})
    assert [o["place_id"] for o in layout_to_roomgenbench(condition, layout)["objects"]] == [None, "obj_0000", None, None]
