"""SceneSmith example scenes (179 Room scenes + 31 Houses split into 185 rooms). Eval-only: too small, and
they are the paper's evaluation scenes (prompts 0-99 are the SceneEval-100 benchmark).

Export (kits/scenesmith_aggregate.py): meters, Z-up, furniture_position = DMD model-frame origin, rotation =
AngleAxis -> XYZ Euler deg (R = Rz Ry Rx), size = asset glTF AABB in the Z-up asset frame (x, y, z).
Asset origin, checked on geometry_metadata AABBs (not in the aggregate): furniture/manipuland = bottom center;
wall_mounted = back face (bbox y in [0, L]), vertical center; ceiling_mounted = top center.
Front = local +Y (wall-mounted assets extend into +Y; front test on beds/wardrobes).
Boundary = floor-plan AABB; every room has exactly north/south/east/west walls, so it is the real rectangle.
Free bodies are physics-settled: x/y tilt <= TILT_DEG is settling noise and is zeroed; an axis permutation within
TILT_DEG is an exact upright box; a larger tilt (only furniture and manipulands, whose origin is the bottom centre)
becomes the yaw-aligned upright bounding box of the 8 corners (sage10k.upright_box: same rotation order, origin and
front), flagged tilted, with tilt_deg and rot_src kept. Unlike SAGE the boxes rest on their lowest corner (1,135 tilted
manipulands over furniture tops: lowest corner - top median -0.1..-1.1 cm in every lift bin), so they are not moved.
Composite members are cut at the bottom of what holds them (scenesmith agent_utils/room.py to_drake_directive): a fill
item `<name>_f<k>_<i>` is inside the container `<name>_f<k>_c` of the same room, a stack member `<name>_s<k>_<i>` sits
on `_s<k>_<i-1>`, so nothing of it is below that holder's bottom; yet 205 of 865 tilted fill items and 11 of 278 tilted
stack members reach below it (round fruit, markers leaning in cups, a knife on a cutting board). The holder is listed first
(1,199 of 1,199 fill items, 1,079 of 1,079 stack members), so fastfill.anchors, which visits boxes by bottom height,
settles the holder before them instead of standing it on what it holds.
Openings: the export has no doors, and its 910 windows carry no location: every window position is the room frame
origin (= boundary centre, z = 0; the window glTF AABB is centred on its own origin and the placing pose lives in the
room_geometry SDF, which is not exported). Only the window count is kept (meta.n_windows_unlocated), no box is invented.
"""
import math
import os
import re

from fastfill.adapters.sage10k import upright_box
from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SceneSmith"
TILT_DEG = 2.0
ANCHOR = {"furniture": "floor", "wall_mounted": "wall", "ceiling_mounted": "ceiling"}   # manipuland: inferred


def tilt_deg(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(rot["x"])) * math.cos(math.radians(rot["y"]))))))


def fix(f, cut=None):
    """Origin -> bottom center, settling tilt -> upright, larger tilt -> upright bounding box cut at `cut` (the
    bottom of the composite member holding it). Returns (furniture dict, noise_zeroed)."""
    rot, p, s, g = f.get("furniture_rotation"), f.get("furniture_position"), f.get("furniture_size"), f["source_fields"]["asset_group"]
    if not (rot and p and s):
        return f, False
    p = dict(p)
    yaw = math.radians(rot["z"])
    if g == "wall_mounted":
        p["x"] -= math.sin(yaw) * s["length"] / 2
        p["y"] += math.cos(yaw) * s["length"] / 2
        p["z"] -= s["height"] / 2
    elif g == "ceiling_mounted":
        p["z"] -= s["height"]
    f = {**f, "furniture_position": p}
    if not (rot["x"] or rot["y"]):
        return f, False
    if tilt_deg(rot) <= TILT_DEG:
        return {**f, "furniture_rotation": {"x": 0, "y": 0, "z": rot["z"]}}, True
    return upright_box(f, cut), False


def fix_room(furniture):
    """fix() over one room in order, each composite member cut at the bottom of its holder (module doc; the export
    lists the holder first): fill item `_f<k>_<i>` -> container `_f<k>_c`, stack member `_s<k>_<i>` -> `_s<k>_<i-1>`."""
    fixed, done = [], {}
    for f in furniture:
        m = re.search(r"_([fs]\d+)_(c|\d+)$", f.get("furniture_instance_id") or "")
        g, i = m.groups() if m else (None, None)
        k = (g, "c") if m and g[0] == "f" and i != "c" else (g, str(int(i) - 1)) if m and g[0] == "s" and i != "0" \
            else None
        h = done.get(k)
        fixed.append(fix(f, h["furniture_position"]["z"] if h else None))
        if m:
            done[g, i] = fixed[-1][0]
    return fixed


def load(root):
    base = os.path.join(root, "imChuling__3D_Room_Collections/SceneSmith_exported")
    for fn in ("json_single.json", "multiroom/multiroom_split.json"):
        for scene in iter_records(os.path.join(base, fn)):
            parent = scene["parent_scene_id"]
            for room in scene["rooms"]:
                fixed = fix_room(room["furniture"])
                room = {**room, "furniture": [f for f, _ in fixed]}
                ir = convert_room(
                    room, source=SOURCE, uid=f"SceneSmith::{scene['scene_id']}", group=f"scenesmith:{parent}",
                    subset=scene["provenance"]["subset"], boundary_type="polygon", center_z=False,
                    front_offset_deg=90,
                    extra=lambda f: {"anchor": ANCHOR.get(f["source_fields"]["asset_group"]),
                                     "asset_group": f["source_fields"]["asset_group"], **f.get("_tilt", {})},
                    meta={"eval_only": True, "front_known": True, "file": fn, "scene_id": scene["scene_id"],
                          "parent_scene_id": parent, "tilt_threshold_deg": TILT_DEG,
                          "n_tilt_noise_zeroed": sum(z for _, z in fixed), "room_type_source": "dmd_room_frame",
                          "n_windows_unlocated": len(room.get("windows") or [])})
                yield ir


if __name__ == "__main__":
    def rec(g, rot, p=(1.0, 1.0, 0.0), S=(1.0, 0.2, 0.1)):
        return {"furniture_category": "x", "source_fields": {"asset_group": g}, "furniture_position": dict(zip("xyz", p)),
                "furniture_rotation": rot, "furniture_size": dict(zip(("width", "length", "height"), S))}
    # settling noise (1 deg) -> zeroed, sizes kept
    f, z = fix(rec("manipuland", {"x": 1.0, "y": 0.0, "z": 30.0}))
    assert z and f["furniture_rotation"] == {"x": 0, "y": 0, "z": 30.0} and f["furniture_size"]["length"] == 0.2, f
    # 30 deg about X -> upright bounding box [0.2 c + 0.1 s, 1.0, 0.2 s + 0.1 c] (front first), yaw 90, flagged tilted
    f, z = fix(rec("manipuland", {"x": 30.0, "y": 0.0, "z": 0.0}, p=(1.0, 1.0, 0.5)))
    ir = convert_room({"room_boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "furniture": [f]}, source=SOURCE, uid="t",
                      group="g", boundary_type="polygon", front_offset_deg=90, extra=lambda f: f.get("_tilt", {}))
    o, c30 = ir["objects"][0], math.cos(math.radians(30))
    assert not z and o["tilted"] and abs(o["tilt_deg"] - 30) < 1e-9 and abs(o["yaw"] - math.pi / 2) < 1e-9, o
    assert all(abs(a - b) < 1e-9 for a, b in zip(o["size"] + o["pos"], [0.2 * c30 + 0.05, 1.0, 0.1 + 0.1 * c30,
                                                                           1.0, 0.975, 0.45])), o
    # wall-mounted: back-face / vertical-centre origin -> bottom centre (yaw 90: +Y local = -X world)
    f, _ = fix(rec("wall_mounted", {"x": 0, "y": 0, "z": 90.0}, p=(2.0, 1.0, 1.5), S=(0.6, 0.1, 0.4)))
    assert all(abs(f["furniture_position"][k] - v) < 1e-9 for k, v in zip("xyz", (1.95, 1.0, 1.3))), f
    # tilted composite members are cut at their holder's bottom (0.48): the fill item of container f2 and stack member
    # s1_1 on s1_0; not an item of a group without a holder here (f3), a stack base, a plain manipuland
    box = rec("manipuland", {"x": 0, "y": 0, "z": 0}, p=(1.0, 1.0, 0.48))
    item = rec("manipuland", {"x": 30.0, "y": 0, "z": 0}, p=(1.0, 1.0, 0.5))
    ids = ("s::bowl_0_f2_c", "s::apple_0_f2_0", "s::apple_1_f3_0", "s::pear_0", "s::board_0_s1_0", "s::knife_0_s1_1",
           "s::book_0_s4_0")
    got = [f for f, _ in fix_room([{**g, "furniture_instance_id": i}
                                   for g, i in zip((box, item, item, item, box, item, item), ids)])]
    assert all(abs(g["furniture_position"]["z"] - z) < 1e-9
               for g, z in zip(got, (0.48, 0.48, 0.45, 0.45, 0.48, 0.48, 0.45))), got
    assert abs(got[1]["furniture_size"]["height"] - (0.05 + 0.1 * c30 + 0.02)) < 1e-9, got[1]
    print("scenesmith.py self-check ok")
