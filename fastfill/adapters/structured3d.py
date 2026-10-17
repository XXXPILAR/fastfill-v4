"""Structured3D (imChuling export, structured3d.jsonl: 22,894 rooms / 3,488 houses, official train/val/test).

Export (kits/convert_structured3d.py): meters, Z-up, room-local (floor polygon centroid, z from floor_z);
position = bbox_3d centroid (box center); the exported size/yaw come from a heuristic axis canonicalization
(row most aligned with world X), which discards the model frame. The raw bbox_3d fields are kept per object:
raw_basis rows = box axes in world, raw_coeffs = half extents (mm) along those rows. Row 2 is the model's up;
wall-contact statistics on beds / TVs / pictures / toilets put the back on raw +Y, so front = raw -Y.
Objects whose raw row 2 is not world +Z (~31k, almost all 'unknown'/'otherprop') are tilted relative to their
model pose -> tilted=True (raw_basis kept), boxed by the upright envelope of their rotated corners. Yaw follows
raw -Y when horizontal; when that front is near vertical, use raw X and mark front_known=False.
Category 'unknown' (41%: occluded in every view, or the corrupt
perspective_full_09.zip for scenes 1600-1799) keeps its box with category "unknown": build.prep drops it as a
generic label and rejects the room only if dropped floor boxes leave a real hole (hidden_floor_obstacle).
room_type 'undefined' -> None.
Boundary = floor polygon traced from annotation_3d junctions/lines.
Doors/windows: room_context.doors/windows (same room frame, position = opening centre on the wall centreline,
~0.1 m outside the floor polygon; width, height, no direction/thickness) -> structure boxes, see `openings`.
"""
import math
import os

import numpy as np

from fastfill.adapters.internscenes import tilted_bbox
from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "Structured3D"


def _furniture(o):
    B = np.array(o["raw_basis"], float)
    ext = 2 * np.abs(np.array(o["raw_coeffs"], float)) / 1000
    f = {"furniture_category": o["category"],
         "furniture_instance_id": str(o["instance_id"]), "furniture_position": o["position"],
         "furniture_size": {"width": ext[1], "length": ext[0], "height": ext[2]}, "raw_basis": o["raw_basis"]}
    if B[2, 2] < 0.999:                                   # model up is not world up
        R = np.stack([-B[1], B[0], B[2]], axis=1)          # semantic front first
        size, yaw, k, _ = tilted_bbox(R, ext[[1, 0, 2]])
        return {**f, "furniture_size": dict(zip(("width", "length", "height"), size)),
                "furniture_rotation": {"x": None, "y": None, "z": yaw},
                **({"front_known": False} if k == 0 else {})}
    front = -B[1]
    return {**f, "furniture_rotation": {"x": 0, "y": 0, "z": math.degrees(math.atan2(front[1], front[0]))}}


OPENING_T = 0.10     # opening box thickness (the export gives none)
SAME_M = 0.05        # an object box centred this close (xy and z) to an opening, same size, is that opening's box


def _edge_deg(poly, x, y):
    """Direction (deg) of the floor-polygon edge nearest to (x, y)."""
    def dist(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = min(1.0, max(0.0, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy or 1)))
        return math.hypot(x - a[0] - t * dx, y - a[1] - t * dy)
    a, b = min(zip(poly, poly[1:] + poly[:1]), key=lambda e: dist(*e))
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def openings(rc, objs):
    """room_context doors/windows -> (structure furniture, objs without the boxes they replace). Box = [width along
    the wall, OPENING_T, height] centred at the given position (z = centre -> bottom = sill), yaw = nearest floor
    edge. ~15% of openings already carry an object box (unknown / otherstructure, same centre and size): that box is
    replaced by the labelled opening instead of duplicated."""
    out, dup = [], set()
    for kind in ("door", "window"):
        for k, d in enumerate(rc.get(kind + "s") or []):
            p, wh = d["position"], sorted([d["width"], d["height"]])
            for i, o in enumerate(objs):
                q, ext = o["position"], sorted(2 * abs(c) / 1000 for c in o["raw_coeffs"])[1:]
                if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) < SAME_M and abs(q["z"] - p["z"]) < SAME_M \
                        and all(abs(u - v) < 0.3 for u, v in zip(ext, wh)):
                    dup.add(i)
            out.append({"furniture_category": kind, "furniture_instance_id": f"{kind}_{k}", "structure": True,
                        "furniture_position": p,
                        "furniture_rotation": {"x": 0, "y": 0, "z": _edge_deg(rc["floor_polygon"], p["x"], p["y"])},
                        "furniture_size": {"width": d["width"], "length": OPENING_T, "height": d["height"]}})
    return out, [o for i, o in enumerate(objs) if i not in dup]


def convert(rec):
    rc, prov = rec["room_context"], rec["provenance"]
    ops, objs = openings(rc, rec["layout"]["floor_layout"]["objects"])
    room = {"room_id": rc["room_id"], "room_type": None if rc["room_type"] == "undefined" else rc["room_type"],
            "room_boundary": rc["floor_polygon"], "room_height": rc["ceiling_height_m"],
            "furniture": [_furniture(o) for o in objs] + ops}
    return convert_room(room, source=SOURCE, uid=rec["sample_id"], group=f"structured3d:{prov['source_house_id']}",
                        subset=prov["split"], boundary_type="polygon", center_z=True,
                        extra=lambda f: {"structure": True} if f.get("structure") else
                        {**({"raw_basis": f["raw_basis"]} if f["furniture_rotation"]["x"] is None else {}),
                         **({"front_known": False} if f.get("front_known") is False else {})},
                        meta={"front_known": True, "room_origin_house_m": rec["room_origin_house_m"],
                              "n_nearest_assigned": sum(o.get("assignment_method") == "nearest" for o in objs),
                              "n_unknown_category": sum(o["category"] == "unknown" for o in objs)})


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/Structured3D_exported/structured3d.jsonl")
    for rec in iter_records(path):
        yield convert(rec)


if __name__ == "__main__":
    # bed rotated 90 deg: raw row0 = world +Y, row1 = world -X (back on raw +Y -> world -X) -> front world +X, yaw 0
    f = _furniture({"category": "bed", "instance_id": 1, "position": {"x": 0, "y": 0, "z": 0.5},
                    "raw_basis": [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], "raw_coeffs": [800, 1000, 500]})
    assert abs(f["furniture_rotation"]["z"]) < 1e-9 and f["furniture_size"]["width"] == 2.0, f
    f = _furniture({"category": "unknown", "instance_id": 2, "position": {"x": 0, "y": 0, "z": 1},
                    "raw_basis": [[0, 0, -1], [0, 1, 0], [1, 0, 0]], "raw_coeffs": [1, 1, 1]})
    assert f["furniture_category"] == "unknown" and f["furniture_rotation"]["x"] is None
    # 4 x 3 room; door on the bottom wall already boxed as 'otherstructure' (replaced, not duplicated), window on the
    # right wall (x = 4) 0.1 m outside the floor, centre z 1.5, height 1.4 -> sill 0.8
    rec = {"sample_id": "t", "provenance": {"source_house_id": 0, "split": "train"}, "room_origin_house_m": [0, 0, 0],
           "room_context": {"room_id": "r", "room_type": "undefined", "ceiling_height_m": 2.8,
                            "floor_polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                            "doors": [{"position": {"x": 1, "y": -0.06, "z": 1.1}, "width": 0.7, "height": 2.2}],
                            "windows": [{"position": {"x": 4.1, "y": 1.5, "z": 1.5}, "width": 1.6, "height": 1.4}]},
           "layout": {"floor_layout": {"objects": [
               {"category": "otherstructure", "instance_id": 9, "position": {"x": 1, "y": -0.06, "z": 1.1},
                "raw_basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "raw_coeffs": [350, 60, 1100]}]}}}
    ir = convert(rec)
    door, win = ir["objects"]
    close = lambda a, b: all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    assert door["category"] == "door" and door["structure"] and close(door["pos"], [1, -0.06, 0]), door
    assert close(door["size"], [0.7, OPENING_T, 2.2]) and abs(door["yaw"]) < 1e-9, door
    assert win["category"] == "window" and close(win["pos"], [4.1, 1.5, 0.8]) and close(win["size"], [1.6, OPENING_T, 1.4])
    assert abs(win["yaw"] - math.pi / 2) < 1e-9 and not win["tilted"], win
    from fastfill.anchors import fixed_geometry
    assert {o["category"] for o in fixed_geometry(ir)} == {"door", "window"}
    print("structured3d.py self-check ok")
