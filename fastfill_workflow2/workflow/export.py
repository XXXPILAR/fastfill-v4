"""Convert canonical layout to RoomGenBench's SAGE-style scene envelope."""

from __future__ import annotations

import hashlib
import math


def roomgenbench_scene(condition: dict, layout: dict, *, scene_key: str = "fastfill_workflow") -> dict:
    room = condition["room"]
    polygon = room["floor_polygon_xy_m"]
    floor = room["floor_z_m"]
    xmin, xmax = min(p[0] for p in polygon), max(p[0] for p in polygon)
    ymin, ymax = min(p[1] for p in polygon), max(p[1] for p in polygon)
    walls = [{"id": f"wall_{i}",
              "start_point": {"x": a[0], "y": a[1], "z": floor},
              "end_point": {"x": b[0], "y": b[1], "z": floor},
              "height": room["height_m"], "thickness": .1}
             for i, (a, b) in enumerate(zip(polygon, polygon[1:] + polygon[:1]))]
    doors, windows = [], []
    for index, opening in enumerate(room.get("openings", [])):
        shared = {"id": f"opening_{index}", "wall_id": f"wall_{opening['wall_index']}",
                  "position_on_wall": opening["position_on_wall"],
                  "width": opening["width_m"], "height": opening["height_m"]}
        if opening["type"] == "door":
            doors.append({**shared, "door_type": opening.get("door_type", "door")})
        elif opening["type"] == "window":
            windows.append({**shared, "sill_height": opening.get("sill_height_m", 0)})
    by_id = {o["id"]: o for o in condition["objects"]}
    objects = []
    for box in layout["objects"]:
        req = by_id[box["id"]]
        sx, sy, sz = box["target_size_local_m"]
        x, y, z = box["bottom_center_m"]
        # FastFill local +X is front; SAGE local +Y is front. Swap local XY and rotate by -90 degrees.
        sage_yaw = math.degrees((box["yaw_rad"] - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi)
        key = "".join(ch if ch.isalnum() else "_" for ch in req["category"].lower()).strip("_")[:24]
        digest = hashlib.sha1(req["description"].encode("utf-8")).hexdigest()[:8]
        objects.append({"id": box["id"], "type": req["category"],
                        "description": req["description"], "asset_key": f"{key}_{digest}",
                        "position": {"x": x, "y": y, "z": z},
                        "rotation": {"x": 0, "y": 0, "z": sage_yaw},
                        "dimensions": {"width": sy, "length": sx, "height": sz},
                        "place_id": box["support_parent"]})
    return {"scene_key": scene_key, "room_type": room.get("room_type", "room"),
            "prompt": room.get("description", ""),
            "room": {"dimensions": {"width": xmax - xmin, "length": ymax - ymin,
                                    "height": room["height_m"]},
                     "position": {"x": xmin, "y": ymin, "z": floor},
                     "ceiling_height": room["height_m"], "walls": walls,
                     "doors": doors, "windows": windows},
            "objects": objects}

