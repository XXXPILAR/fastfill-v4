"""Aria Synthetic Environments (ASE_exported, 8,373 train scenes / 16,375 wall-ring rooms).

Export (kits/convert_ase.py): Z-up meters, scene-global; furniture_position = ATEK ts_world_object translation
= OBB center; furniture_size = object_dimensions full extents along local X/Y/Z; rotation = ZYX Euler of the
ATEK rotation matrix, z = yaw of local +X. Boundary = closed ring of make_wall endpoints (thickness 0);
the 53 open chains were closed by a guessed straight edge -> dropped here. Front = local -Y (beds touch the
wall with +Y). Category: original object_instances_to_classes string, else the ATEK category name of the
same instance (kept in meta). room_type in the export is inferred from furniture -> kept in meta only.

Completeness (v4): GT boxes exist only for instances seen in the selected ATEK frames (~40% of each scene's
instances); the others are listed in source_inventory.json with no position at all, so they cannot be assigned
to a room and would be invisible obstacles. Every room of a scene with such a furniture instance gets
meta.n_incomplete (build rejects it, "incomplete_objects"). Labels that are not objects (empty_space,
background) do not count. Instances with a box but no class (no object_instances_to_classes entry, ATEK
"other") are kept as category "object", which build shows as an unlabelled fixed box like any source's
"unknown"/"otherprop"; the export left the positioned ones unassigned, so each joins the room containing its
centre. A positioned instance near a room but in none (or a named one the export left out) marks that room
incomplete. v4 front test on this IR: beds touch the wall with -X 100% (n=13,537),
storage/sofas 0.0-0.1% facing the wall.
Openings (v4): room.doors / room.windows = [{position (CENTER), width, height}] without a wall reference; each
is shown as a structure box along the nearest boundary edge, WALL_THICKNESS deep (the export walls have none).
"""
import json
import math
import os

from shapely.geometry import LineString, Point, Polygon

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "ASE"
DIR = "BillLin66__3D_Room_Collections/ASE_exported"
NOT_OBJECTS = {"empty_space", "background"}
WALL_THICKNESS = 0.10
BORDER_M = 0.10          # a positioned unassigned instance within this of a room still counts as in it


def _atek_name(f):
    for rec in (f.get("source_fields") or {}).get("obb3_gt", {}).values():
        if rec.get("category_names"):
            return rec["category_names"][0]
    return None


def missing_instances(root):
    """scene_id -> (number of unplaceable furniture instances, [positioned unassigned instances])."""
    out = {}
    for s in json.load(open(os.path.join(root, DIR, "source_inventory.json"))):
        blind, placed = 0, []
        for o in s["unassigned_objects"]:
            if o["furniture_category"] in NOT_OBJECTS:
                continue
            if o["furniture_position"] is None:
                blind += 1
            else:
                placed.append(o)
        out[s["scene_id"]] = (blind, placed)
    return out


def openings(room, prefix):
    """Doors and windows as unified-schema furniture: centre on the opening, long side along the nearest edge.
    convert_room(front_offset_deg=-90) swaps width/length and subtracts 90 deg, so these come out as size
    [opening width, WALL_THICKNESS, height] with yaw = edge direction."""
    ring = room.get("room_boundary")
    if not ring or len(ring) < 3:
        return []
    edges = [LineString([ring[i], ring[(i + 1) % len(ring)]]) for i in range(len(ring))]
    out = []
    for kind in ("door", "window"):
        for k, d in enumerate(room.get(kind + "s") or []):
            p = d["position"]
            e = min(edges, key=lambda e: e.distance(Point(p["x"], p["y"])))
            (x0, y0), (x1, y1) = e.coords
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0))
            out.append({"furniture_category": kind, "furniture_instance_id": f"{prefix}::{kind}_{k}", "structure": True,
                        "furniture_position": p, "furniture_rotation": {"x": 0, "y": 0, "z": ang + 90},
                        "furniture_size": {"width": WALL_THICKNESS, "length": d["width"], "height": d["height"]}})
    return out


def load(root):
    missing = missing_instances(root)
    for scene in iter_records(os.path.join(root, DIR, "ase.jsonl")):
        sid = scene["scene_id"]  # ase_<split>_<7-digit index>
        blind, placed = missing.get(sid, (0, []))
        rings = [Polygon(r["room_boundary"]).buffer(0) if r["room_boundary"] and len(r["room_boundary"]) >= 3 else None
                 for r in scene["rooms"]]
        for k, room in enumerate(scene["rooms"]):
            furn, n_atek, n_unlabelled = [], 0, 0
            for f in room["furniture"]:
                if f["furniture_category"] is None:
                    name = _atek_name(f)
                    if name and name != "other":
                        f = {**f, "furniture_category": name}
                        n_atek += 1
                    else:
                        f = {**f, "furniture_category": "object"}
                        n_unlabelled += 1
                furn.append(f)
            open_chain = bool(room["room_boundary"]) and len(room["room_boundary"]) != len(room["source_fields"]["make_wall"])
            stray, joined = 0, 0
            if rings[k] is not None and not open_chain:
                near = rings[k].buffer(BORDER_M)
                for o in placed:
                    c = Point(o["furniture_position"]["x"], o["furniture_position"]["y"])
                    if not near.contains(c):
                        continue
                    homes = [j for j, g in enumerate(rings) if g is not None and g.contains(c)]
                    if o["furniture_category"] is None and homes == [k]:
                        furn.append({**o, "furniture_category": "object"})
                        joined += 1
                    else:                  # in no room or several, or a named instance the export left out
                        stray += 1
                furn += openings(room, room["room_id"])
            r = {**room, "furniture": furn, "room_type": None,
                 "room_boundary": None if open_chain else room["room_boundary"]}
            ir = convert_room(r, source=SOURCE, uid=room["room_id"], group=f"ase:{sid[len('ase_'):]}",
                              subset=scene["source_subset"], boundary_type="polygon", center_z=True,
                              front_offset_deg=-90,
                              meta={"room_type_inferred": room["room_type"], "n_category_from_atek": n_atek,
                                    "open_chain_dropped": open_chain, "front_known": True,
                                    "scene_unplaced_instances": blind, "room_unassigned_instances": stray,
                                    "n_unlabelled": n_unlabelled, "n_unassigned_joined": joined})
            ir["meta"]["n_incomplete"] += blind + stray
            yield ir
