"""ScanNet++ v2 instance OBBs (ScanNet++_exported). Auxiliary tier: NO semantic front.

Export (kits/convert_scannetpp.py): meters, Z-up, scene-local; furniture_position = OBB centroid; size = OBB
axesLengths reordered so the axis closest to world Z is local Z (frame kept right-handed); rotation = Rz*Ry*Rx
Euler of that frame. The OBBs are fitted per instance, so local X is an arbitrary horizontal box axis: yaw is
the box orientation modulo the box symmetry, not a facing direction (meta front_known=False on every room).
Boundary = XY AABB of all floor-OBB cross-sections (a superset of the floor; whole apartments stay one record).
Tilts <= MAX_TILT_DEG (vertical_axis_dot_z) are treated as upright; z is measured from the per-scene 5th percentile
bottom of all upright boxes and bottoms within FLOOR_SNAP of it are set to 0. (A median over "floor" categories
followed wall cabinets/shelves: b73f5cdc41 floor 1.46 m; against the annotated thin floor OBBs it was > 0.2 m off in
108/748 scenes, the p5 bottom in 5/772.)
"""
import math
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "ScanNetpp"
PATH = "BillLin66__3D_Room_Collections/ScanNet++_exported/scannetpp.jsonl"
MAX_TILT_DEG = 10.0
FLOOR_SNAP = 0.10
COS_TILT = math.cos(math.radians(MAX_TILT_DEG))


def load(root):
    for s in iter_records(os.path.join(root, PATH)):
        room = s["rooms"][0]
        upright = [f["source_fields"]["vertical_axis_dot_z"] >= COS_TILT for f in room["furniture"]]
        bottoms = sorted(f["furniture_position"]["z"] - f["furniture_size"]["height"] / 2
                         for f, u in zip(room["furniture"], upright) if u)
        floor = bottoms[len(bottoms) // 20] if bottoms else None
        furn, snapped = [], 0
        for f, u in zip(room["furniture"], upright):
            p, r = f["furniture_position"], f["furniture_rotation"]
            z = p["z"] - f["furniture_size"]["height"] / 2 - (floor or 0.0)
            if floor is not None and abs(z) < FLOOR_SNAP:
                z, snapped = 0.0, snapped + 1
            furn.append({**f, "furniture_position": {"x": p["x"], "y": p["y"], "z": z},
                         "furniture_rotation": {"x": 0 if u else r["x"], "y": 0 if u else r["y"], "z": r["z"]}})
        sid = s["scene_id"].removeprefix("scannetpp_")
        yield convert_room(
            {**room, "furniture": furn}, source=SOURCE, uid=f"{SOURCE}::{sid}", group=f"scannetpp:{sid}",
            subset=s.get("source_subset"), boundary_type="hull", center_z=False,
            meta={"scene_id": sid, "front_known": False, "is_true_room": False,
                  "boundary_source": room["room_boundary"] and "floor_obb_xy_aabb", "floor_z": floor, "n_floor_snapped": snapped})
