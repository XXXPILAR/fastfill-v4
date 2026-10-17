"""Aria Synthetic Environments (ASE_exported, 8,373 train scenes / 16,375 wall-ring rooms).

Export (kits/convert_ase.py): Z-up meters, scene-global; furniture_position = ATEK ts_world_object translation
= OBB center; furniture_size = object_dimensions full extents along local X/Y/Z; rotation = ZYX Euler of the
ATEK rotation matrix, z = yaw of local +X. Boundary = closed ring of make_wall endpoints (thickness 0);
the 53 open chains were closed by a guessed straight edge -> dropped here. Front = local -Y (beds touch the
wall with +Y). Category: original object_instances_to_classes string, else the ATEK category name of the
same instance (kept in meta). room_type in the export is inferred from furniture -> kept in meta only.
Limitation: GT exists only for instances seen in the selected ATEK frames (~40% of each scene's instances).
"""
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "ASE"


def _atek_name(f):
    for rec in (f.get("source_fields") or {}).get("obb3_gt", {}).values():
        if rec.get("category_names"):
            return rec["category_names"][0]
    return None


def load(root):
    path = os.path.join(root, "BillLin66__3D_Room_Collections/ASE_exported/ase.jsonl")
    for scene in iter_records(path):
        sid = scene["scene_id"]  # ase_<split>_<7-digit index>
        for room in scene["rooms"]:
            furn, n_atek = [], 0
            for f in room["furniture"]:
                if f["furniture_category"] is None:
                    name = _atek_name(f)
                    if name and name != "other":
                        f = {**f, "furniture_category": name}
                        n_atek += 1
                furn.append(f)
            open_chain = bool(room["room_boundary"]) and len(room["room_boundary"]) != len(room["source_fields"]["make_wall"])
            r = {**room, "furniture": furn, "room_type": None,
                 "room_boundary": None if open_chain else room["room_boundary"]}
            yield convert_room(r, source=SOURCE, uid=room["room_id"], group=f"ase:{sid[len('ase_'):]}",
                               subset=scene["source_subset"], boundary_type="polygon", center_z=True,
                               front_offset_deg=-90,
                               meta={"room_type_inferred": room["room_type"], "n_category_from_atek": n_atek,
                                     "open_chain_dropped": open_chain, "front_known": True,
                                     "objects_partial": True})
