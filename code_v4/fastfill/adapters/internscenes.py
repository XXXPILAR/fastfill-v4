"""InternScenes layouts. This module = InternScenes-Gen (SOURCE InternScenes_gen) plus the shared reader for the
Real2Sim re-layouts of real scans (modules internscenes_{scannet,3rscan,arkit,mp3d}).

Export (kits/extract_internscenes*_json.py): meters, Z-up, bbox = [x, y, z, sx, sy, sz, a_z, a_x, a_y] with
position = box CENTER, sizes along the box local axes, ZXY Euler (R = Rz(a_z) Rx(a_x) Ry(a_y), EmbodiedScan /
pytorch3d convention). Floor is at z = 0.
Boundary: Real2Sim = 2D convex hull of the floor mesh (-> "hull"); Gen = boundary_points.json floor polygon.
Some boxes are stored with a 90/180 deg X/Y rotation (Y-up assets, books lying flat); they are upright boxes
with permuted axes and are re-expressed as yaw-only boxes (`upright`). Boxes tilted by more than TILT_TOL_DEG
are replaced by the yaw-aligned bounding box of their 8 corners (`tilted_bbox`: true footprint, height and bottom,
yaw = heading of the local X axis) with rotation x/y = None so convert_room flags them tilted; "tilt_deg" keeps
the angle between the local axis closest to vertical and world Z.
"""
import csv
import io
import json
import math
import os
import zipfile

import numpy as np
from shapely.geometry import Polygon

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "InternScenes_gen"
DIR = "imChuling__3D_Room_Collections/InternScenes_exported"
TILT_TOL_DEG = 5.0
MAX_SIZE_M = 20.0
# ponytail: one offset for all Real2Sim/Gen assets, set from the inspect_ir front test (see module docstrings)
FRONT_OFFSET_DEG = 0.0


def rot_zxy(az, ax, ay):
    cz, sz, cx, sx, cy, sy = math.cos(az), math.sin(az), math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def upright(R, size, tol_deg=TILT_TOL_DEG):
    """World-from-local rotation + local sizes -> (sizes [sx, sy, sz], yaw_deg, k) of the same box as an upright
    box, k = index of the source local axis that is vertical; None if no local axis is within tol of vertical.
    The new local X is the source local X when it stays horizontal (keeps its front), else the source local Y."""
    R = np.asarray(R, dtype=float)
    k = int(np.argmax(np.abs(R[2])))
    if abs(R[2, k]) < math.cos(math.radians(tol_deg)):
        return None
    hx, hy = [i for i in range(3) if i != k]          # k=2: (0,1)  k=1: (0,2)  k=0: (1,2)
    v = R[:, hx]
    return [size[hx], size[hy], size[k]], math.degrees(math.atan2(v[1], v[0])), k


def tilted_bbox(R, size):
    """World-from-local rotation + local sizes of a box that is NOT upright within tolerance -> (sizes [w, l, h] of
    the yaw-aligned bounding box of its 8 corners, yaw_deg, k, tilt_deg); k = local axis closest to vertical, tilt =
    its angle from world Z; yaw = heading of local X, or of local Y when X is the near-vertical axis (as upright()
    does: X's small horizontal part is then only the lean direction, and a box aligned to it is up to 3x too big)."""
    R = np.asarray(R, dtype=float)
    k = int(np.argmax(np.abs(R[2])))
    a = next((c for c in ((R[:2, 1], R[:2, 0]) if k == 0 else (R[:2, 0], R[:2, 1])) if np.hypot(*c) >= 1e-6), (1.0, 0.0))
    yaw = math.atan2(a[1], a[0])
    D = np.array([[math.cos(yaw), math.sin(yaw), 0], [-math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])   # u, v, z
    ext = np.abs(D @ R) @ np.asarray(size, dtype=float)      # max - min of the (origin-symmetric) corners along u, v, z
    return ext.tolist(), math.degrees(yaw), k, math.degrees(math.acos(min(1.0, abs(R[2, k]))))


def canon(f):
    """Exported furniture -> same dict with an upright {x:0,y:0,z:yaw} rotation and permuted sizes when the box is
    upright within tolerance; otherwise its yaw-aligned bounding box (`tilted_bbox`) with {x:None,y:None,z:yaw}
    (convert_room flags it tilted) and "tilt_deg". The centre is unchanged either way."""
    b = f["source_fields"]["bbox"]
    if max(b[3:6]) > MAX_SIZE_M:                 # corrupt box (2 Gen beds are 1910 m / 803014 m tall)
        return {**f, "furniture_size": None}
    R = rot_zxy(b[6], b[7], b[8])
    up = upright(R, b[3:6])
    if up is None:
        (sx, sy, sz), yaw, k, tilt = tilted_bbox(R, b[3:6])
        rot, extra = {"x": None, "y": None, "z": yaw}, {"tilt_deg": tilt}
    else:
        (sx, sy, sz), yaw, k = up
        rot, extra = {"x": 0.0, "y": 0.0, "z": yaw}, {}
    return {**f, "furniture_size": {"width": sx, "length": sy, "height": sz}, "furniture_rotation": rot,
            "vertical_axis": k, **extra}


def convert(scene, source, uid, group, boundary_type, meta):
    room = scene["rooms"][0]
    room = {**room, "furniture": [canon(f) for f in room["furniture"]]}
    ir = convert_room(room, source=source, uid=uid, group=group, subset=scene.get("source_subset"),
                      boundary_type=boundary_type, center_z=True, front_offset_deg=FRONT_OFFSET_DEG,
                      meta={"scene_id": scene["scene_id"], "source_layout": scene["source_layout"],
                            "source_dataset": scene["source_dataset"], "is_true_room": room.get("is_true_room"),
                            "tilt_tol_deg": TILT_TOL_DEG, "front_known": True,
                            "n_permuted_axes": sum(f.get("vertical_axis", 2) != 2 for f in room["furniture"]),
                            **meta})
    return ir


def load_real(root, fname, source, group_of, extra_meta=None):
    for s in iter_records(os.path.join(root, DIR, fname)):
        yield convert(s, source, uid=f"{source}::{s['scene_id']}", group=group_of(s), boundary_type="hull",
                      meta=(extra_meta(s) if extra_meta else {}))


def rscan_reference(root):
    """3RScan scan id (reference or rescan) -> reference scan id."""
    m = json.load(open(os.path.join(root, "imChuling__3D_Room_Collections/3RScan_exported/source_metadata/3RScan.json")))
    ref = {}
    for env in m:
        ref[env["reference"]] = env["reference"]
        for sc in env.get("scans", []):
            ref[sc["reference"]] = env["reference"]
    return ref


def arkit_visits(root):
    """ARKitScenes video_id -> visit_id (None when 'NA')."""
    z = zipfile.ZipFile(os.path.join(root, "imChuling__3D_Room_Collections/ARKitScenes_exported/source.zip"))
    rows = csv.DictReader(io.TextIOWrapper(z.open("source/metadata.csv"), encoding="utf-8"))
    return {r["video_id"]: (None if r["visit_id"] in ("", "NA") else r["visit_id"]) for r in rows}


def load(root):
    """Gen regions: the floor is not at z = 0 (0.10-0.15 m, floor slab); every region has >= 1 door and all
    doors of a region share one bottom height, so z is shifted by the median door bottom."""
    for s in iter_records(os.path.join(root, DIR, "internscenes_gen_exported.json")):
        rt, rid = s["rooms"][0]["room_type"], s["source_scene_name"]
        # boundary_points are wall samples every 0.7 m with the first point repeated: keep the corners only
        # (median 27 -> 6 vertices; every polygon is valid, max shift 1 cm)
        poly = Polygon(s["rooms"][0]["room_boundary"]).simplify(0.01)
        s["rooms"][0]["room_boundary"] = [list(c) for c in poly.exterior.coords[:-1]]
        ir = convert(s, SOURCE, uid=f"{SOURCE}::{rt}/{rid}", group=f"internscenes:gen/{rid}",   # one id = one generated house across room-type folders
                     boundary_type="polygon", meta={"asset_source": "Infinigen", "boundary_simplify_m": 0.01})
        doors = sorted(o["pos"][2] for o in ir["objects"] if o["category"] == "door")
        fz = doors[len(doors) // 2] if doors else None
        for o in ir["objects"]:
            o["pos"][2] -= fz or 0.0
        ir["meta"].update(floor_z=fz, floor_z_source="median door bottom" if doors else None)
        yield ir


if __name__ == "__main__":
    # book lying flat: local X vertical -> upright box, height = source width
    R = rot_zxy(0.3, 0.0, math.pi / 2)
    size, yaw, k = upright(R, [0.03, 0.2, 0.3])
    assert k == 0 and size == [0.2, 0.3, 0.03] and abs(yaw - math.degrees(0.3) - 90) < 1e-6, (size, yaw, k)
    # Y-up chair stored with a_x = -90: local Y vertical, local X keeps the yaw
    size, yaw, k = upright(rot_zxy(0.0, -math.pi / 2, math.radians(40)), [0.7, 0.8, 0.6])
    assert k == 1 and size == [0.7, 0.6, 0.8] and abs(yaw + 40) < 1e-6, (size, yaw)
    assert upright(rot_zxy(0.0, math.radians(10), 0.0), [1, 1, 1]) is None

    # canon: sizes [1.0, 0.2, 0.1], ZXY angles in degrees, centre z
    S = [1.0, 0.2, 0.1]

    def rec(ax, ay, az=0.0, z=0.0):
        return {"furniture_category": "box", "furniture_position": {"x": 1.0, "y": 1.0, "z": z},
                "source_fields": {"bbox": [1.0, 1.0, z, *S, math.radians(az), math.radians(ax), math.radians(ay)]}}

    def sz(c):
        return [c["furniture_size"][key] for key in ("width", "length", "height")]

    # (1) upright box yawed 30 deg: unchanged sizes, yaw 30
    c = canon(rec(0, 0, az=30))
    assert sz(c) == S and abs(c["furniture_rotation"]["z"] - 30) < 1e-9 and "tilt_deg" not in c, c
    # (2) tilted 30 deg about X: yaw-aligned bounding box, yaw 0
    s30, c30 = math.sin(math.radians(30)), math.cos(math.radians(30))
    h, l = 0.2 * s30 + 0.1 * c30, 0.2 * c30 + 0.1 * s30              # 0.1866, 0.2232
    c = canon(rec(30, 0))
    assert np.allclose(sz(c), [1.0, l, h]) and abs(c["furniture_rotation"]["z"]) < 1e-9, c
    assert c["furniture_rotation"]["x"] is None and c["vertical_axis"] == 2 and abs(c["tilt_deg"] - 30) < 1e-9, c
    # (3) lying flat (a_x = 90): permuted upright box via upright(), not the tilted path
    c = canon(rec(90, 0))
    assert np.allclose(sz(c), [1.0, 0.1, 0.2]) and c["vertical_axis"] == 1 and "tilt_deg" not in c, c
    assert c["furniture_rotation"]["x"] == 0.0 and abs(c["furniture_rotation"]["z"]) < 1e-9, c
    # (4) tilted box centred at z = 0.2 -> convert_room bottom = 0.2 - h/2, flagged tilted
    scene = {"scene_id": "s", "source_layout": "l", "source_dataset": "d",
             "rooms": [{"room_type": "t", "room_boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "furniture": [rec(30, 0, z=0.2)]}]}
    ir = convert(scene, "t", uid="u", group="g", boundary_type="polygon", meta={})
    o = ir["objects"][0]
    assert o["tilted"] and abs(o["pos"][2] - (0.2 - h / 2)) < 1e-9 and ir["meta"]["n_tilted"] == 1, (o, ir["meta"])
    print("internscenes.py self-check ok")
