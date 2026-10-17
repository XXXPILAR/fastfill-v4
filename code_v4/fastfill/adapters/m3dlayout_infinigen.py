"""M3DLayout, Infinigen subset (procedural houses, one room per record). Aux only: boundary is a proxy.

Export (extract_m3dlayout_json.py): source Y-up (location = OBB center, size = half extents, rotation = yaw about +Y)
-> furniture_position = (x, -z, y - sy) = BOTTOM center, size = (2sx, 2sz, 2sy), rotation.z = degrees(yaw);
the layout format is yaw-only (detail objects: 1 of 19,643 tilted > 5.7 deg in a 300-scene sample).
XY origin = furniture AABB SW corner; room_boundary = that AABB (object_extent_aabb) -> boundary_type "proxy";
room_height is an object-extent proxy too -> None. The layout frame is the Infinigen scene mirrored in Y
(layout_y = C - native_y, checked against the detail objects); a mirrored room is still a valid layout.
Front = layout local -Y (front test vs the proxy rectangle: toilet +Y 0.84 / -Y 0.00, bed 0.73 / 0.00).
room_type = M3DLayout's explicit-keyword parse of the source description (None / 'unknown' -> None).
Records without layout furniture (detail-only) are skipped; repeated (scene_id, layout) records are emitted once.
"""
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "M3DLayout_infinigen"
DIR = "imChuling__3D_Room_Collections/M3DLayout_exported"


def furniture(f):
    """Layout furniture -> unified schema: yaw-only format means upright; snake_case category -> words."""
    return {**f, "furniture_category": f["furniture_category"].replace("_", " "),
            "furniture_rotation": {**f["furniture_rotation"], "x": 0, "y": 0}}


def records(root, name):
    """(scene, room) for records that have layout furniture, each distinct (scene_id, layout) once."""
    seen = set()
    for sc in iter_records(os.path.join(root, DIR, name)):
        room = sc["rooms"][0]
        if not room["furniture"]:
            continue
        key = (sc["scene_id"], tuple(sorted((f["furniture_category"], round(f["furniture_position"]["x"], 3),
                                             round(f["furniture_position"]["y"], 3), round(f["furniture_position"]["z"], 3),
                                             round(f["furniture_rotation"]["z"], 1)) for f in room["furniture"])))
        if key in seen:
            continue
        seen.add(key)
        yield sc, room


def room_type(room):
    return None if room.get("room_type") in (None, "unknown") else room["room_type"].replace("_", " ")


def load(root):
    for sc, room in records(root, "infinigen.json"):
        yield convert_room({**room, "room_height": None, "room_type": room_type(room),
                            "furniture": [furniture(f) for f in room["furniture"]]},
                           source=SOURCE, uid=sc["sample_id"], group="infinigen:" + sc["scene_id"].split("_", 1)[1],
                           subset=sc["source_subset"], boundary_type="proxy", front_offset_deg=-90,
                           meta={"scene_id": sc["scene_id"], "front_known": True, "mirrored_frame": True,
                                 "room_type_source": room.get("room_type_source"), "height_source": "dropped_proxy"})
