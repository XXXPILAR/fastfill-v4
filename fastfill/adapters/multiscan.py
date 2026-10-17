"""MultiScan (XXXpilar/multiscan-clean repackaging): 273 scans of 117 real spaces, one scan = one space.

objects.csv: obb_center (box CENTER), obb_half_extents (HALF sizes along the box axes), obb_axes = 9 floats =
the three box axes one after another (world directions), plus annotated `front` / `up` vectors; `up` is box
axis 1 for 10,957/10,957 objects and `front` is one of the box axes. Z-up meters, scan-local frame whose floor
is NOT at z = 0. regions.csv floor_height is the TOP of the floor-slab AABB (slabs are 0.07-0.46 m thick scan
noise) and furniture bottoms sit a median 0.09 m below it, so the floor level used is the median bottom of the
objects that relations.csv marks rests_on_floor (>= 1 per scan; geometric relation of the repackaging).
Boundary: regions.csv poly_loop is the convex hull of the floor-slab OBBs, but floors are annotated only where
the scan saw them (22% of in-scope objects stick out of it by a median 0.24 m). The boundary used is the convex
hull of poly_loop + the wall-segment endpoints (structure only) -> "hull" when every floor-standing
non-architectural object fits inside it (+5 cm); otherwise the room gets boundary_type "proxy" with that hull
extended by those objects' footprints (object extent, cannot prove in-bounds validity).
height = ceiling_height - floor level, only when height_reliable. Group = multiscan:<scene_id> (rescans _NN share it).
"""
import csv
import json
import math
import os

import numpy as np
from shapely.geometry import MultiPoint, Polygon

from fastfill.adapters.internscenes import TILT_TOL_DEG, tilted_bbox, upright
from fastfill.adapters.unified import convert_room
from fastfill.scene import footprint

SOURCE = "MultiScan"
DIR = "XXXpilar__multiscan-clean"


def box(o):
    """objects.csv row -> unified furniture dict (bottom-center position, upright yaw from the annotated front)."""
    axes = np.array(json.loads(o["obb_axes"]), dtype=float).reshape(3, 3)       # rows = box axes
    size = [2 * v for v in json.loads(o["obb_half_extents"])]
    c = json.loads(o["obb_center"])
    fr = np.array(json.loads(o["front"]), dtype=float)
    d = axes @ fr
    f = int(np.argmax(np.abs(d)))
    order = [f] + [i for i in range(3) if i != f]                               # front axis first = local +X
    R = np.stack([axes[i] * (np.sign(d[f]) if i == f else 1) for i in order], axis=1)
    up = upright(R, [size[i] for i in order])
    if up is None:                                                              # tilted: keep a non-zero tilt
        sz, yaw, k, tilt = tilted_bbox(R, [size[i] for i in order])
        rot = {"x": tilt, "y": 0.0, "z": yaw}
    else:
        sz, yaw, k = up
        rot = {"x": 0.0, "y": 0.0, "z": yaw}
    bottom = c[2] - sz[2] / 2
    return {"furniture_category": o["category"].replace("_", " "), "furniture_instance_id": f"o{o['object_id']}",
            "furniture_position": {"x": c[0], "y": c[1], "z": bottom}, "furniture_rotation": rot,
            "furniture_size": {"width": sz[0], "length": sz[1], "height": sz[2]},
            "front_vertical": k == 0,
            "structure": o["is_architectural"] == "True" or o["is_opening"] == "True"}


def boundary(g, furn, tol=0.05):
    """(polygon, type, n_outside) - see module docstring."""
    walls = json.loads(g["wall_segments"] or "[]")
    pts = json.loads(g["poly_loop"]) + [w[k] for w in walls for k in ("start", "end")]
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull
    fps = [Polygon(footprint({"pos": [f["furniture_position"]["x"], f["furniture_position"]["y"]],
                              "size": [f["furniture_size"]["width"], f["furniture_size"]["length"]],
                              "yaw": math.radians(f["furniture_rotation"]["z"])}))
           for f in furn if not f["structure"] and f["furniture_position"]["z"] < tol]
    out = [fp for fp in fps if not hull.buffer(tol).contains(fp)]
    if out:
        hull = MultiPoint(list(hull.exterior.coords) + [c for fp in out for c in fp.exterior.coords]).convex_hull
    return [list(c) for c in hull.exterior.coords[:-1]], ("proxy" if out else "hull"), len(out)


def load(root):
    base = os.path.join(root, DIR)
    scans = {r["scan_id"]: r for r in csv.DictReader(open(os.path.join(base, "scans.csv")))}
    objs, on_floor = {}, {}
    for o in csv.DictReader(open(os.path.join(base, "objects.csv"))):
        objs.setdefault(o["scan_id"], []).append(o)
    for r in csv.DictReader(open(os.path.join(base, "relations.csv"))):
        if r["relation"] == "rests_on_floor":
            on_floor.setdefault(r["scan_id"], set()).add(r["subject_id"])
    for g in csv.DictReader(open(os.path.join(base, "regions.csv"))):
        sid, s = g["scan_id"], scans[g["scan_id"]]
        bottoms = [json.loads(o["aabb_min"])[2] for o in objs.get(sid, []) if o["object_id"] in on_floor.get(sid, ())]
        fz = float(np.median(bottoms)) if bottoms else float(g["floor_height"])
        furn = [box(o) for o in objs.get(sid, [])]
        for f in furn:
            f["furniture_position"]["z"] -= fz
        poly, btype, n_out = boundary(g, furn)
        # source flags is_architectural (wall/floor/ceiling/beam/pillar/windowsill/...) and is_opening are kept as
        # "structure": build never places them but must see them (a floor pillar rejects the room)
        room = {"room_id": sid, "room_type": s["room_type"].lower(), "room_boundary": poly,
                # extrusion_height = ceiling - floor_height (slab top); measure from the floor level used for z
                "room_height": float(g["ceiling_height"]) - fz if g["height_reliable"] == "True" else None,
                "furniture": furn}
        yield convert_room(room, source=SOURCE, uid=f"{SOURCE}::{sid}", group=f"multiscan:{s['scene_id']}",
                           boundary_type=btype, extra=lambda f: {
                               **({"structure": True} if f["structure"] else {}),
                               **({"front_known": False} if f["front_vertical"] else {})},
                           meta={"scan_id": sid, "scene_id": s["scene_id"], "device": s["device"],
                                 "floor_z": fz, "floor_z_source": "median bottom of rests_on_floor objects" if bottoms
                                 else "regions.floor_height", "floor_height": float(g["floor_height"]), "poly_source": g["poly_source"],
                                 "boundary_source": "convex hull of floor slabs + wall segments"
                                 + (" + floor objects outside it" if n_out else ""), "n_floor_objects_outside_structure": n_out,
                                 "height_reliable": g["height_reliable"] == "True", "front_known": True,
                                 "front_source": "annotated front vector", "tilt_tol_deg": TILT_TOL_DEG,
                                 "n_front_vertical": sum(f["front_vertical"] for f in furn)})


if __name__ == "__main__":
    # bag from scene_00000_00: up = axis 1, front = axis 2 -> yaw of the front vector, size [2*h2, 2*h0, 2*h1]
    row = {"obb_axes": "[0.9861942396120026, 0.16559263798884283, 0, 0, 0, 1, 0.16559263798884283, -0.9861942396120026, 0]",
           "obb_half_extents": "[0.126, 0.166, 0.0795]", "obb_center": "[0, 0, 1]", "category": "shoe_box",
           "front": "[0.16559263798884283, -0.9861942396120026, 0]", "object_id": "2",
           "is_architectural": "False", "is_opening": "False"}
    b = box(row)
    assert b["furniture_category"] == "shoe box" and b["furniture_rotation"]["x"] == 0
    assert abs(b["furniture_rotation"]["z"] - math.degrees(math.atan2(-0.98619, 0.16559))) < 1e-3
    assert [round(v, 3) for v in b["furniture_size"].values()] == [0.159, 0.252, 0.332]
    assert abs(b["furniture_position"]["z"] - (1 - 0.166)) < 1e-9
    print("multiscan.py self-check ok")
