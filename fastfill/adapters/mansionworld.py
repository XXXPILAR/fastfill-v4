"""MansionWorld (AI2-THOR / Objathor procedural buildings), room view json_room.json (27,884 rooms, 1,000 building
variants of 300 floor plans; building/floor views are the same rooms regrouped).

Export (kits/convert_mansionworld.py, enrich_wall_objects.py): meters, Z-up, (x, z)_yup -> (x, -z), room polygon
recentred on its area centroid. Floor objects: position = THOR position (asset pivot; footprint center within 5 cm),
z = 0 (bottom), size = padded cm footprint minus 0.10 m, de-rotated (yaws are multiples of 90), height = 2 * center y.
Yaw = THOR rotation.y, which the (x, -z) map turns into CCW yaw about +Z of asset local +X; front = Unity +Z = local -Y;
a full THOR rotation (x, y, z) is rot_zxy(y, x, -z) in this frame (Unity applies z, x, y; checked to 0.0).
Exception: the single custom asset of `toilet_suite` (a stall) is authored backwards: its +Y side touches the wall
and its paired toilet_suite_door lies on the room side, so it gets +180 deg.
Wall objects: z = bottom; xy were recentred on the vertex mean instead of the area centroid (bug), corrected here;
width/length had the floor-footprint padding subtracted although wall footprints are unpadded (bug), added back here.
Surface (small) objects: source_fields.surface_groups (one per parent floor object, parent id = support_surfaces
parent_object_id) -> anchor "object", parent = that id, IR id "<small>|<parent>" (a third of the small names repeat
under another parent of the same room; the 22 exported objects whose raw id repeats get "#k"). Room xy = parent
position + R(parent yaw) position_local and yaw = parent yaw + yaw_deg_local, both equal to the raw position /
rotation.y (3,355 objects, max diff 0.0).
The export keeps only rotation.y; the raw floor_*.json small_objects (same download, joined by "<small>|<parent>"
within the room) give x/z: 92.2 % of 201,302 are 0, the rest exact 90-deg axis flips (pens, notebooks, phones, frames
lying flat), none at another angle. Flipped boxes become upright boxes with permuted sizes (internscenes.upright;
counted in meta.n_permuted_axes); their source front is vertical, so their yaw is only a box orientation. The export
bottom (support height_m + z_local = raw center y - exported height / 2) is corrected by (exported - true height) / 2:
flipped bottoms then match the nearest upright neighbour on the same parent to 0.2 mm median (was 6.6 cm). Surface
objects without a raw record are not emitted (meta.surface_objects_excluded). build.prep keeps the ones whose bottom
is within 5 cm of the parent box top (the rest stand on shelves / inside furniture).
Desc: the export's Objathor asset description of floor, wall and surface objects is passed as "desc".
Room type: every raw floor has exactly one roomType "corridor" room (3,719 of 3,719 floors), its open hub, whatever
it holds (living_dining_room, open_office_area, master_bedroom, hub_room, ...); for these the room name (room_id minus
"F<n>_" = raw portable_source_id, 27,884 of 27,884) is the room type. The source value is kept in meta.room_type_src.
Doors / windows: room["doors"] / ["windows"] = {center_xy, width_m, ...} from the raw doorSegment / windowSegment,
already in the room's recentred frame (all door centres lie on the polygon ring). Emitted as "structure" boxes:
local X along the wall = opening width, 0.10 m across the wall, yaw = CCW direction of the nearest polygon edge.
Windows: z = sill_height_m, height = height_m (from the raw holePolygon). Doors: the export dropped the hole height;
raw holePolygon / door-database.json give 2.07-2.13 m for every door asset, so DOOR_H = 2.1 (clearance_depth_m is
the swing clearance, not a thickness). Openings > 0.15 m off the ring (1.8 % of windows: same-named room on another
floor, matched by the converter with the F<n>_ prefix stripped) are dropped and counted in meta.n_openings_dropped.
"""
import collections
import functools
import json
import math
import os
import re

from shapely.geometry import LineString, Point

from fastfill.adapters.internscenes import rot_zxy, upright
from fastfill.adapters.unified import _num, convert_room, iter_records, signed_area

SOURCE = "MansionWorld"
DIR = "imChuling__3D_Room_Collections/MansionWorld_exported"
FLOOR_PREFIX = re.compile(r"^F\d+_")
PAD, MIN_DIM = 0.10, 0.01   # enrich_wall_objects.py FOOTPRINT_PADDING_M / MIN_DIM_M
DOOR_H, OPENING_T, OFF_WALL = 2.1, 0.10, 0.15


def fix(f, vx, vy):
    f = dict(f)
    if f["source_fields"].get("mount_type") == "wall" and f.get("furniture_position"):
        p = f["furniture_position"]
        f["furniture_position"] = {**p, "x": p["x"] + vx, "y": p["y"] + vy}
        # enrich_wall_objects.py also subtracts the 0.10 m floor-footprint padding, but wall footprints are unpadded
        # (raw sample: footprint == annotation bbox for 1,652/1,652). Values clamped to 0.01 stay (true extent <= 0.11).
        s = f.get("furniture_size")
        if s:
            f["furniture_size"] = {**s, **{k: s[k] + PAD for k in ("width", "length") if s[k] > MIN_DIM + 1e-6}}
    if f["furniture_category"] == "toilet_suite" and f.get("furniture_rotation"):
        f["furniture_rotation"] = {**f["furniture_rotation"], "z": f["furniture_rotation"]["z"] + 180}
    if f["furniture_category"].startswith("vertical_core_"):   # stair / elevator cores: let the structure filter see them
        f["furniture_category"] = f["furniture_category"].replace("_", " ")
    return f


def openings(room):
    """Doors / windows -> (furniture dicts on the wall line, n_dropped). Rotation z = wall direction + 90 so that
    convert_room(front_offset_deg=-90) yields yaw = wall direction and size = [width_m, OPENING_T, height]."""
    b = room.get("room_boundary") or []
    if len(b) < 3:
        return [], 0
    if signed_area(b) < 0:
        b = b[::-1]
    edges = [LineString(s) for s in zip(b, b[1:] + b[:1])]
    out, dropped = [], 0
    for kind, key, items in (("door", "door_id", room.get("doors")), ("window", "window_id", room.get("windows"))):
        for e in items or []:
            x, y = e.get("center_xy") or (None, None)
            z, h = (0.0, DOOR_H) if kind == "door" else (e.get("sill_height_m"), e.get("height_m"))
            edge = min(edges, key=lambda ln: ln.distance(Point(x, y))) if _num(x) and _num(y) else None
            if not all(_num(v) for v in (x, y, e.get("width_m"), z, h)) or edge.distance(Point(x, y)) > OFF_WALL:
                dropped += 1
                continue
            (x0, y0), (x1, y1) = edge.coords
            out.append({"furniture_category": kind, "furniture_instance_id": e[key], "structure": True,
                        "furniture_position": {"x": x, "y": y, "z": z},
                        "furniture_rotation": {"x": 0.0, "y": 0.0, "z": math.degrees(math.atan2(y1 - y0, x1 - x0)) + 90},
                        "furniture_size": {"width": OPENING_T, "length": e["width_m"], "height": h}})
    return out, dropped


@functools.lru_cache(maxsize=2)          # json_room.json is grouped by building
def raw_rotations(root, building):
    """{room id: {"<small>|<parent>": [raw rotation {x, y, z}, ...]}} from the building's floor_*.json small_objects
    (id "<small>|<parent> (<room>)"; roomId may lack the F<n>_ prefix, which is unique per floor once stripped)."""
    out, d = {}, os.path.join(root, DIR, "data/MansionWorld/mansionworld", building)
    for fn in os.listdir(d):
        if fn.startswith("floor_") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as fh:
                raw = json.load(fh)
            rid = {FLOOR_PREFIX.sub("", r["id"]): r["id"] for r in raw["rooms"]}
            for s in raw.get("small_objects") or []:
                r = rid.get(FLOOR_PREFIX.sub("", s.get("roomId", "")))
                if r:
                    out.setdefault(r, {}).setdefault(s["id"].split(" (", 1)[0], []).append(s["rotation"])
    return out


def surface(room, rots):
    """surface_groups -> (furniture dicts with "parent", n without a raw rotation); rots = raw_rotations()[room id].
    Positions / yaws are composed from the parent's unmodified export pose (see module doc)."""
    furn = {f["furniture_instance_id"]: f for f in room["furniture"]}
    tops = {t["surface_id"]: t for t in room["source_fields"].get("support_surfaces") or []}
    out, missing, seen = [], 0, collections.Counter()
    for g in room["source_fields"].get("surface_groups") or []:
        top = tops[g["surface_id"]]
        pid = top["parent_object_id"]
        pp, pyaw = furn[pid]["furniture_position"], furn[pid]["furniture_rotation"]["z"]
        c, s = math.cos(math.radians(pyaw)), math.sin(math.radians(pyaw))
        for o in g["objects"]:
            key = f"{o['object_id']}|{pid}"
            seen[key] += 1
            rs = rots.get(key, [])
            if len(rs) < seen[key]:
                missing += 1
                continue
            r = rs[seen[key] - 1]
            R = rot_zxy(*(math.radians(v) for v in (pyaw + o["yaw_deg_local"], r["x"], -r["z"])))
            (sx, sy, sz), yaw, k = upright(R, o["dimensions"])
            (lx, ly), h = o["position_local"], o["dimensions"][2]
            out.append({"furniture_category": o["category"], "parent": pid, "description": o.get("description"),
                        "furniture_instance_id": key if seen[key] == 1 else f"{key}#{seen[key]}",
                        "furniture_position": {"x": pp["x"] + c * lx - s * ly, "y": pp["y"] + s * lx + c * ly,
                                               "z": top["height_m"] + o["z_local"] + (h - sz) / 2},
                        "furniture_rotation": {"x": 0.0, "y": 0.0, "z": yaw},
                        "furniture_size": {"width": sx, "length": sy, "height": sz}, "vertical_axis": k})
    return out, missing


def extra(f):
    if f.get("structure"):
        return {"structure": True}
    if f.get("parent"):
        return {"anchor": "object", "parent": f["parent"], "desc": f.get("description")}
    wall = f["source_fields"].get("mount_type") == "wall"
    return {"anchor": "wall" if wall else "floor", "desc": f.get("description")}


def to_ir(room, scene, rots):
    prov = scene["provenance"]
    building = prov["source_house_id"]
    b = room.get("room_boundary") or []
    vx, vy = (sum(p[0] for p in b) / len(b), sum(p[1] for p in b) / len(b)) if b else (0.0, 0.0)
    boxes, n_dropped = openings(room)
    smalls, missing = surface(room, rots)
    rt = room.get("room_type")
    return convert_room(
        {**room, "furniture": [fix(f, vx, vy) for f in room["furniture"]] + smalls + boxes,
         "room_type": FLOOR_PREFIX.sub("", room["room_id"]) if rt == "corridor" else rt}, source=SOURCE,
        uid=f"MansionWorld::{scene['scene_id']}", group="mansionworld:" + building.split("#")[0],
        boundary_type="polygon", center_z=False, front_offset_deg=-90, extra=extra,
        meta={"building": building, "room_id": room["room_id"], "room_type_src": rt,
              "geometry_hash": prov.get("geometry_hash"), "front_known": True, "n_openings_dropped": n_dropped,
              "surface_objects_excluded": missing, "n_permuted_axes": sum(f["vertical_axis"] != 2 for f in smalls),
              "license_note": "CC BY 4.0 + HF gate: non-commercial research"})


def load(root):
    for scene in iter_records(os.path.join(root, DIR, "json_room.json")):
        rots = raw_rotations(root, scene["provenance"]["source_house_id"])
        for room in scene["rooms"]:
            yield to_ir(room, scene, rots.get(room["room_id"], {}))


if __name__ == "__main__":
    from fastfill.anchors import fixed_geometry
    scene = {"scene_id": "s", "provenance": {"source_house_id": "b#0"}}
    room = {"room_id": "r", "room_type": "office", "room_height": 3.2, "furniture": [],
            "room_boundary": [[-1, -1], [1, -1], [1, 1], [-1, 1]], "source_fields": {},
            "doors": [{"door_id": "d", "center_xy": [1.0, 0.0], "width_m": 0.9, "clearance_depth_m": 0.9}],
            "windows": [{"window_id": "w", "center_xy": [0.0, 1.0], "width_m": 1.2, "sill_height_m": 0.9, "height_m": 1.2},
                        {"window_id": "far", "center_xy": [-5.0, 1.0], "width_m": 1.2, "sill_height_m": 0.9, "height_m": 1.2}]}
    ir = to_ir(room, scene, {})
    d, w = ir["objects"]
    assert d["category"] == "door" and d["structure"] and d["size"] == [0.9, OPENING_T, DOOR_H], d
    assert d["pos"] == [2.0, 1.0, 0.0] and abs(d["yaw"] - math.pi / 2) < 1e-9, d       # right wall runs +Y (CCW)
    assert w["size"] == [1.2, OPENING_T, 1.2] and w["pos"] == [1.0, 2.0, 0.9] and abs(w["yaw"] - math.pi) < 1e-9, w
    assert ir["meta"]["n_openings_dropped"] == 1 and ir["meta"]["n_incomplete"] == 0
    assert [o["id"] for o in fixed_geometry(ir)] == ["d", "w"]
    assert ir["room_type"] == "office" and ir["meta"]["room_type_src"] == "office"

    # surface objects: desk (THOR yaw 90) at (0.5, 0.2) holding an upright cup, a notebook stored Z-up and flipped
    # x = 90 (exported dims [w, d, h] = [0.2, 0.02, 0.3]: its 0.02 side is the vertical one) and two "pen-0" whose
    # raw id repeats; a pen on a shelf has no raw record. Export bottom = height_m + z_local = raw y - dims[2] / 2.
    desk = {"furniture_category": "desk", "furniture_instance_id": "desk-0", "description": "A wooden desk.",
            "furniture_position": {"x": 0.5, "y": 0.2, "z": 0.0}, "furniture_rotation": {"x": 0, "y": 0, "z": 90},
            "furniture_size": {"width": 1.2, "length": 0.6, "height": 0.75}, "source_fields": {"mount_type": "floor"}}
    obj = lambda n, dims, loc, yl, zl: {"object_id": n, "category": n[:-2], "dimensions": dims, "position_local": loc,
                                        "yaw_deg_local": yl, "z_local": zl, "description": f"A {n[:-2]}."}
    room = {**room, "room_id": "F2_living_dining_room", "room_type": "corridor", "doors": [], "windows": [],
            "furniture": [desk], "source_fields": {
                "support_surfaces": [{"surface_id": "desk-0/top", "parent_object_id": "desk-0", "height_m": 0.75}],
                "surface_groups": [{"surface_id": "desk-0/top", "objects": [
                    obj("cup-0", [0.1, 0.08, 0.12], [0.2, 0.0], 10.0, 0.0),
                    obj("notebook-0", [0.2, 0.02, 0.3], [-0.3, 0.1], 0.0, -0.14),
                    obj("pen-0", [0.14, 0.01, 0.01], [0.0, 0.0], 0.0, 0.0),
                    obj("pen-0", [0.14, 0.01, 0.01], [0.1, 0.0], 0.0, 0.0),
                    obj("pen-1", [0.14, 0.01, 0.01], [0.0, 0.0], 0.0, -0.4)]}]}}
    rots = {"cup-0|desk-0": [{"x": 0, "y": 100, "z": 0}], "notebook-0|desk-0": [{"x": 90, "y": 90, "z": 0}],
            "pen-0|desk-0": [{"x": 0, "y": 90, "z": 0}] * 2}
    ir = to_ir(room, scene, rots)
    assert ir["room_type"] == "living_dining_room" and ir["meta"]["room_type_src"] == "corridor", ir["room_type"]
    assert ir["meta"]["surface_objects_excluded"] == 1 and ir["meta"]["n_permuted_axes"] == 1, ir["meta"]
    by = {o["id"]: o for o in ir["objects"]}
    assert list(by) == ["desk-0", "cup-0|desk-0", "notebook-0|desk-0", "pen-0|desk-0", "pen-0|desk-0#2"], list(by)
    assert by["desk-0"]["anchor"] == "floor" and by["desk-0"]["desc"] == "A wooden desk."
    close = lambda u, v: all(abs(x - y) < 1e-9 for x, y in zip(u, v))
    cup, nb = by["cup-0|desk-0"], by["notebook-0|desk-0"]
    assert (cup["anchor"], cup["parent"], cup["desc"], cup["tilted"]) == ("object", "desk-0", "A cup.", False), cup
    # R(90) (0.2, 0) = (0, 0.2): room (0.5, 0.4) -> AABB-shifted by (-1, -1); yaw 90 + 10 - 90 (front offset); d, w, h
    assert close(cup["pos"], [1.5, 1.4, 0.75]) and abs(cup["yaw"] - math.radians(10)) < 1e-9, cup
    assert close(cup["size"], [0.08, 0.1, 0.12]), cup
    # R(90) (-0.3, 0.1) = (-0.1, -0.3); lying flat: true height 0.02, bottom 0.61 + (0.3 - 0.02) / 2 = 0.75
    assert close(nb["pos"], [1.4, 0.9, 0.75]) and close(nb["size"], [0.3, 0.2, 0.02]), nb
    assert abs(math.sin(nb["yaw"])) < 1e-9 < math.cos(nb["yaw"]) and nb["parent"] == "desk-0", nb   # heading of local X
    print("mansionworld.py self-check ok")
