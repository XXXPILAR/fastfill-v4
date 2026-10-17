"""SpatialGen public test set (48 designer rooms). Eval-only: the source release is a test split.

Export: Z-up meters, scene-local, furniture_position = box center, size = scaled local XYZ extents,
yaw about +Z (deg). Boundary = floor polygon. Front = local -Y (front test: beds/wardrobes touch the wall with +Y).
"""
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SpatialGen"


def load(root):
    path = os.path.join(root, "BillLin66__3D_Room_Collections/SpatialGen_exported/spatialgen.jsonl")
    for scene in iter_records(path):
        for room in scene["rooms"]:
            yield convert_room(room, source=SOURCE, uid=room["room_id"], group=f"spatialgen:{scene['scene_id']}",
                               subset="test", boundary_type="polygon", center_z=True,
                               front_offset_deg=-90, meta={"eval_only": True, "front_known": True})
