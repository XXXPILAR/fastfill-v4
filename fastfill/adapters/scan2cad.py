"""Scan2CAD CAD alignments on ScanNet v2 scans (SceneCAD_Scan2CAD_exported). Auxiliary tier.

Export (kits aggregate_scenecad_scan2cad.py): meters, Z-up, ScanNet axisAlignment-applied scan frame.
scan2cad_alignments[].furniture_position = oriented CAD bbox CENTER (position_type), size = full CAD bbox
extents along the object's canonical Z-up axes [2*bx*sx, 2*bz*sz, 2*by*sy], rotation = Rz*Ry*Rx Euler of
rotation_matrix_zup_scan_local, whose columns are the object's local X / Y / Z (local +Y = ShapeNet -Z).
Front: ShapeNet canonical front = CAD -Z = local +Y (front test: at -90 beds/bookshelves touched the wall with +X), so front_offset_deg = +90.

Boundary: the SceneCAD floor plane polygon (human layout annotation, 994 scans) -> "polygon"; otherwise the
exported convex hull of ScanNet floor vertices, which misses occluded floor (38% of CAD footprints leave it vs 14%
for the SceneCAD polygon, mostly < 10 cm wall penetration) and so is typed "proxy".
Limitations (why B_aux): one record = one whole ScanNet scan (is_true_room=false), and only the ~30% of objects
that were matched to a ShapeNet CAD are present.
Scan data noise: per-object tilt median 3 deg (9-DoF fit); tilts <= MAX_TILT_DEG are treated as upright.
Bottoms scatter +-10 cm around the floor, so z is measured from the per-scan median bottom of floor-standing
categories and bottoms within FLOOR_SNAP of it are set to 0. Only bottoms within FLOOR_BAND of z = 0 vote: the
axisAlignment frame already puts the floor there (SceneCAD floor plane z p1..p99 = -0.20..0.07), and an unbanded
median followed stacked/wall-mounted CADs (scene0074_01: one shelf at 2.0 m -> floor_z 2.0; 47 scans |floor_z| > 0.2). CAD fits with an axis < 1 mm (a trs.scale axis of
1e-14..1e-8, 75 objects, mostly displays) have no usable size: their size is dropped, so the room counts as incomplete.
"""
import math
import os
import statistics

from shapely.geometry import Polygon

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "Scan2CAD"
PATH = "imChuling__3D_Room_Collections/SceneCAD_Scan2CAD_exported/scenecad_scan2cad_combined.json"
MAX_TILT_DEG = 10.0     # p95 of |tilt| over all 14,225 alignments is 9.7 deg
FLOOR_SNAP = 0.15
FLOOR_BAND = 0.25
FLOOR_CATS = {"chair", "table", "sofa", "bed", "bookshelf", "file_cabinet", "trash_can", "washer",
              "dishwasher", "stove", "piano", "bench"}


def tilt_deg(a):
    return math.degrees(math.acos(max(-1.0, min(1.0, a["source_fields"]["rotation_matrix_zup_scan_local"][2][2]))))


def shell_floor(shell):
    """XY ring of the single horizontal SceneCAD plane at the bottom of the shell, or None."""
    if not shell:
        return None
    v = shell["vertices_zup_m"]
    zmin = min(p[2] for p in v)
    lows = []
    for q in shell["planes"]:
        a, b, c = (v[i] for i in q[:3])
        n = [(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
             (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
             (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])]
        if abs(n[2]) > 0.9 * math.hypot(*n) and sum(v[i][2] for i in q) / len(q) < zmin + 0.3:
            lows.append([[v[i][0], v[i][1]] for i in q])
    if len(lows) != 1 or not Polygon(lows[0]).is_valid:
        return None
    return lows[0]


def load(root):
    for s in iter_records(os.path.join(root, PATH)):
        room = s["rooms"][0]
        al = room["scan2cad_alignments"]
        if not al:          # scenecad_only scans: shell without objects
            continue
        bottoms = [a["furniture_position"]["z"] - a["furniture_size"]["height"] / 2 for a in al
                   if a["furniture_category"] in FLOOR_CATS and tilt_deg(a) <= MAX_TILT_DEG]
        bottoms = [b for b in bottoms if abs(b) < FLOOR_BAND]
        floor = statistics.median(bottoms) if bottoms else None
        furn, snapped = [], 0
        for a in al:
            p, r = a["furniture_position"], a["furniture_rotation"]
            z = p["z"] - a["furniture_size"]["height"] / 2 - (floor or 0.0)
            if floor is not None and abs(z) < FLOOR_SNAP:
                z, snapped = 0.0, snapped + 1
            upright = tilt_deg(a) <= MAX_TILT_DEG
            degenerate = min(a["furniture_size"].values()) < 1e-3
            furn.append({**a, "furniture_size": None if degenerate else a["furniture_size"], "furniture_position": {"x": p["x"], "y": p["y"], "z": z},
                         "furniture_rotation": {"x": 0 if upright else r["x"], "y": 0 if upright else r["y"], "z": r["z"]}})
        sid = s["scene_id"]
        poly = shell_floor(room["scene_shell"])
        yield convert_room(
            {**room, "furniture": furn, "room_boundary": poly or room["room_boundary"]}, source=SOURCE, uid=f"{SOURCE}::{sid}", group=f"scannet:{sid.rsplit('_', 1)[0]}",
            subset=s.get("source_subset"), boundary_type="polygon" if poly else "proxy", center_z=False, front_offset_deg=90,
            extra=lambda f: {"sym": f["source_fields"]["scan2cad_annotation"]["sym"]},
            meta={"scene_id": sid, "front_known": True, "is_true_room": False, "boundary_source": "scenecad_floor_plane" if poly else room["room_boundary"] and "scannet_floor_vertex_convex_hull",
                  "source_coverage": s["provenance"]["source_coverage"], "floor_z": floor, "n_floor_snapped": snapped})


if __name__ == "__main__":
    shell = {"vertices_zup_m": [[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0], [0, 0, 2.5], [4, 0, 2.5]],
             "planes": [[0, 1, 2, 3], [0, 1, 5, 4]]}      # floor quad + one wall
    assert shell_floor(shell) == [[0, 0], [4, 0], [4, 3], [0, 3]]
    assert shell_floor({**shell, "planes": [[0, 1, 5, 4]]}) is None
    print("scan2cad.py self-check ok")
