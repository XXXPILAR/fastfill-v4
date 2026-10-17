"""Shared reader for the common export schema used by most collections:
scene{scene_id, provenance, rooms[{room_id, room_type, room_boundary, room_boundary_type, room_height,
furniture[{furniture_category, furniture_instance_id, furniture_position{x,y,z},
furniture_rotation{x,y,z} (deg), furniture_size{width,length,height}, source_fields}]}]}.

Conventions differ per source (box center vs bottom, which local axis is the front, boundary quality),
so each adapter passes them explicitly to `convert_room`.
"""
import json
import math

import ijson


def iter_records(path):
    """Scene records from a .jsonl or a top-level JSON array, streamed (some files are >8 GB)."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "rb") as f:
            yield from ijson.items(f, "item", use_float=True)


def signed_area(poly):
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1])) / 2


def _num(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def convert_room(room, *, source, uid, group, subset=None, boundary_type=None, center_z=False,
                 front_offset_deg=0.0, size_keys=("width", "length", "height"), extra=None, meta=None):
    """One exported room -> IR room.

    boundary_type   IR boundary type for this source ("polygon" | "hull" | "proxy" | None)
    center_z        True if furniture_position.z is the box center (converted to bottom)
    front_offset_deg  angle of the object's semantic front in its exported local frame, measured CCW from
                    local +X; the IR frame is rotated so the front becomes local +X (sizes swap for +-90)
    size_keys       keys of furniture_size along exported local X, Y, Z
    extra(f)        optional per-furniture dict merged into the IR object (anchor, parent, desc, ...)
    Objects with a missing category/size/position/yaw are counted in meta.n_incomplete; tilted objects
    (rotation x/y null or non-zero) get tilted=True. fastfill.build rejects rooms with incomplete objects
    and rooms whose in-scope objects are tilted.
    """
    objs, n_tilted, n_incomplete = [], 0, 0
    swap = round(front_offset_deg) % 180 == 90
    for k, f in enumerate(room.get("furniture") or []):
        s, p, r = f.get("furniture_size"), f.get("furniture_position"), f.get("furniture_rotation")
        if not (f.get("furniture_category") and s and p and r):
            n_incomplete += 1
            continue
        size = [s.get(key) for key in size_keys]
        pos = [p.get("x"), p.get("y"), p.get("z")]
        if not all(_num(v) for v in size + pos) or not _num(r.get("z")):
            n_incomplete += 1
            continue
        tilted = r.get("x") != 0 or r.get("y") != 0
        n_tilted += tilted
        if center_z:
            pos[2] -= size[2] / 2
        if swap:
            size[0], size[1] = size[1], size[0]
        o = {"id": f.get("furniture_instance_id") or f"obj_{k}", "category": f["furniture_category"],
             "size": size, "pos": pos, "yaw": math.radians(r["z"] + front_offset_deg) % (2 * math.pi),
             "anchor": None, "parent": None, "tilted": tilted}
        if extra:
            o.update(extra(f))
        objs.append(o)

    boundary = room.get("room_boundary")
    if boundary and len(boundary) >= 3 and all(_num(v) for pt in boundary for v in pt[:2]):
        boundary = [[float(pt[0]), float(pt[1])] for pt in boundary]
        if signed_area(boundary) < 0:
            boundary = boundary[::-1]
        x0, y0 = min(p[0] for p in boundary), min(p[1] for p in boundary)
        boundary = [[x - x0, y - y0] for x, y in boundary]
    else:
        boundary, boundary_type = None, None
        x0 = min((o["pos"][0] for o in objs), default=0.0)
        y0 = min((o["pos"][1] for o in objs), default=0.0)
    for o in objs:
        o["pos"] = [o["pos"][0] - x0, o["pos"][1] - y0, o["pos"][2]]

    height = room.get("room_height")
    return {"uid": uid, "source": source, "subset": subset, "group": group,
            "room_type": room.get("room_type"), "boundary": boundary, "boundary_type": boundary_type,
            "height": height if _num(height) else None, "objects": objs,
            "meta": {"n_tilted": n_tilted, "n_incomplete": n_incomplete, "room_id": room.get("room_id"),
                     **(meta or {})}}


if __name__ == "__main__":
    room = {"room_id": "r", "room_type": "bedroom", "room_boundary": [[1, 1], [1, 4], [5, 4], [5, 1]],
            "furniture": [
                {"furniture_category": "bed", "furniture_instance_id": "b",
                 "furniture_position": {"x": 2, "y": 2, "z": 0.5}, "furniture_rotation": {"x": 0, "y": 0, "z": 0},
                 "furniture_size": {"width": 2.0, "length": 1.6, "height": 1.0}},
                {"furniture_category": "lamp", "furniture_position": {"x": 2, "y": 2, "z": 0.5},
                 "furniture_rotation": {"x": None, "y": None, "z": 10}, "furniture_size": {"width": 1, "length": 1, "height": 1}},
                {"furniture_category": None}]}
    ir = convert_room(room, source="t", uid="t::r", group="g", boundary_type="polygon", center_z=True,
                      front_offset_deg=-90)
    assert [0.0, 0.0] in ir["boundary"] and signed_area(ir["boundary"]) > 0
    bed = ir["objects"][0]
    assert bed["pos"] == [1.0, 1.0, 0.0] and bed["size"] == [1.6, 2.0, 1.0]
    assert abs(bed["yaw"] - math.radians(270)) < 1e-9
    assert ir["meta"]["n_tilted"] == 1 and ir["meta"]["n_incomplete"] == 1
    print("unified.py self-check ok")
