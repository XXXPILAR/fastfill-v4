"""OptiScene / 3D-SynthPlace, HOLODECK subset: 9,977 generated single rooms (Objaverse assets, floor objects only).

Raw fields (source_fields.layout_object, see OptiScene/README.md and gprompt.py): Unity/AI2-THOR frame = y-up,
LEFT-handed; position {x, y, z} m with y = 0 for every object (bottom) and x/z = box center; rotation.y = yaw in
degrees (clockwise seen from above); bbox = [height, width, depth] cm along the mesh y, x, z.
LH (x, y, z) -> RH Z-up (x, z, y) is a proper change of frame (what the export stored), but the yaw sign flips:
yaw of mesh +X in the IR = -rotation.y. Holodeck meshes face mesh -Z = IR local -Y -> front_offset_deg = -90.
The export's furniture_rotation.z (= +rotation.y) and size mapping are therefore not used.
"""
import os

from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "OptiScene_holodeck"
OPTI = "imChuling__3D_Room_Collections/OptiScene_exported"


def raw_furniture(f):
    lo = f["source_fields"]["layout_object"]
    p, r, b = lo.get("position") or {}, lo.get("rotation") or {}, lo.get("bbox")
    return {**f,
            "furniture_position": {"x": p.get("x"), "y": p.get("z"), "z": p.get("y")},
            "furniture_rotation": {"x": r.get("x", 0), "y": r.get("z", 0),
                                   "z": -r["y"] if isinstance(r.get("y"), (int, float)) else None},
            "furniture_size": {"width": b[1] / 100, "length": b[2] / 100, "height": b[0] / 100} if b and len(b) == 3 else None}


def load(root):
    for scene in iter_records(os.path.join(root, OPTI, "holodeck.json")):
        for room in scene["rooms"]:
            yield convert_room({**room, "furniture": [raw_furniture(f) for f in room["furniture"]]},
                               source=SOURCE, uid=f"optiscene:{scene['scene_id']}",
                               group=f"holodeck:{scene['scene_id']}", subset="HOLODECK", boundary_type="polygon",
                               front_offset_deg=-90, extra=lambda f: {"desc": f.get("description")},
                               meta={"source_file": scene["provenance"]["source_file"], "front_known": True,
                                     "size_note": "Holodeck bbox, padded ~11 cm in width/depth (see report)"})


if __name__ == "__main__":
    f = {"source_fields": {"layout_object": {"position": {"x": 1, "y": 0, "z": 2}, "rotation": {"x": 0, "y": 90, "z": 0},
                                             "bbox": [75, 77, 174]}}}
    g = raw_furniture(f)
    assert g["furniture_position"] == {"x": 1, "y": 2, "z": 0} and g["furniture_rotation"]["z"] == -90
    assert g["furniture_size"] == {"width": 0.77, "length": 1.74, "height": 0.75}
    print("optiscene_holodeck self-check ok")
