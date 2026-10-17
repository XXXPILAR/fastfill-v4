"""SpatialLM-Dataset layouts (Manycore/Kujiale designer houses: 12,328 scenes, 54,775 rooms).
View used: spatiallm.json only (scene records, every room once); json_room.json and multiroom/*.json are
re-packagings of the same rooms (split_views.py) and are ignored.

Export (aggregate_spatiallm.py): source Bbox(category, px,py,pz, angle_z, sx,sy,sz) -> furniture_position = box
CENTER, furniture_size = full scale along local x/y/z, rotation.z = degrees(angle_z); the format is yaw-only, so
the exported x/y = null mean "not tilted" (set to 0 here). Meters, Z-up, scene-global XY.
room_boundary = ring of wall start points (wall_polygon). 2152 of the 54,376 rings are not valid polygons; 2080 of them
are only a zero-area spur, a wall overshooting a corner (the source wall chain crosses itself and closes a sliver lobe:
scene_000060::room_3, 2.5 x 6.5 cm) or a vertex touching the ring, and are repaired (`repair`, recorded in
meta.boundary_repaired; ring moved by median 6 cm, p90 12 cm, max 1.2 m for a zero-area spur; dropped lobes p99
0.05 m2, max 0.24 m2, holding no placeable object centre; kept with the lobe, 3 of the 314 repaired rooms build flags
oob_objects would lose that flag). Doors/windows keep their source pose: 2 of the 8,792 in repaired rooms stood on a
dropped spur/lobe wall and now sit 5 / 12 cm outside the ring, no footprint inside it (scene_011725::room_2,
scene_002826::room_4); the other 170,616 door/window centres in polygon rooms are within 3 cm of their ring. The
other 72 fall apart into several polygons (63) or change area by >= 1% (9): boundary_type None.
Front = local -Y (front test: toilets/beds touch the wall with local +Y), same as the other Manycore exports.
Openings: room.doors / room.windows = [{wall_index, position{x,y,z} (CENTER of the opening, on the wall line),
width (along the wall), height, sill_height (windows only; doors null)}], walls[wall_index] = {start, end, thickness}
(thickness is always 0.0 in the export). Emitted as "structure" boxes width x 0.10 m x height with yaw = wall
direction; center z passes through center_z (door bottom = 0 for 99.7%, the rest are real raised openings).

Cross-export duplicates (same Manycore design; exact floor area / boundary edges + identical pairwise distances of
floor furniture, see the g2-layout-mixed assessment): scenes holding a SpatialGen test room are dropped (SpatialGen
is eval-only), scenes matching a Structured3D / InteriorGS house take that house's group key.
"""
import math
import os

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from fastfill.adapters.unified import _num, convert_room, iter_records, signed_area

SOURCE = "SpatialLM"
SPATIALGEN_DUP = {f"scene_{n:06d}" for n in (
    646, 692, 701, 713, 734, 738, 773, 792, 793, 806, 828, 832, 1569, 1585, 1591, 1599, 1601, 1610, 1612, 1758,
    4021, 4053, 7537, 7629, 7989, 8143, 8147, 8729, 8815, 11483,
    # review pass: boundary-free match (>= 3 floor objects with identical dims at identical relative positions under
    # one axis-aligned rigid map, >= half of the smaller set) or same floor area +-0.1 m2 with >= 70% object match
    82, 537, 644, 667, 671, 740, 745, 822, 827, 841, 843, 1579, 1606, 1608, 1624, 1626, 1725, 1745, 4027, 4099,
    4111, 4112, 4165, 5211, 6058, 6252, 7362, 7478, 7498, 7622, 7729, 7736, 9248, 9360, 9771, 10350, 10394, 11383,
    11482, 12106,
    4359)}   # pre-launch review: design twin of spatialgen_scene_00039::room_0
GROUP = {"scene_003636": "structured3d:scene_00729", "scene_007839": "structured3d:scene_02605",
         "scene_009460": "structured3d:scene_01438", "scene_011931": "structured3d:scene_00402",
         "scene_012118": "structured3d:scene_01556", "scene_004716": "interiorgs:0304_840560",
         "scene_011878": "interiorgs:0553_840921",
         # review pass: same floor area to 0.01 m2 and 4-6 floor objects with identical dims at identical relative positions
         "scene_005720": "interiorgs:0785_841229", "scene_006253": "interiorgs:0536_840884",
         "scene_010373": "interiorgs:0817_841276"}


WALL_THICKNESS = 0.10   # the export's wall thickness is 0.0 everywhere
REPAIR_TOL = 0.01       # a repaired wall ring keeps the ring's shoelace area to within 1%


def repair(ring):
    """Invalid wall ring -> the exterior of Polygon(ring).buffer(0) when that is ONE polygon without holes whose area
    is within REPAIR_TOL of the ring's shoelace area (the floor the walls enclose is then well defined: buffer(0) only
    drops the spur / sliver lobe), else None (several parts, or the area changes: the room is not the ring's).
    Not make_valid: it keeps a crossed corner's sliver lobe as a second polygon (one polygon for 1457 of the 2152)."""
    p, a = Polygon(ring).buffer(0), abs(signed_area(ring))
    if p.geom_type == "Polygon" and not p.interiors and abs(p.area - a) < REPAIR_TOL * a:
        return [list(q) for q in p.exterior.coords[:-1]]


def openings(room, prefix):
    """Doors and windows as unified-schema furniture in the room's frame. convert_room(front_offset_deg=-90)
    swaps width/length and subtracts 90 deg, so width=thickness / length=opening width / rotation=wall+90 come out
    as size [opening width, thickness, height] with yaw = wall direction (long side along the wall)."""
    walls = room.get("walls") or []
    out = []
    for kind in ("door", "window"):
        for k, d in enumerate(room.get(kind + "s") or []):
            wi = d.get("wall_index")
            if not (isinstance(wi, int) and 0 <= wi < len(walls)):
                continue
            w = walls[wi]
            ang = math.degrees(math.atan2(w["end"]["y"] - w["start"]["y"], w["end"]["x"] - w["start"]["x"]))
            out.append({"furniture_category": kind, "furniture_instance_id": f"{prefix}::{kind}_{k}", "structure": True,
                        "furniture_position": d["position"], "furniture_rotation": {"x": 0, "y": 0, "z": ang + 90},
                        "furniture_size": {"width": w.get("thickness") or WALL_THICKNESS, "length": d["width"],
                                           "height": d["height"]}})
    return out


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/SpatialLM_exported/spatiallm.json")
    for scene in iter_records(path):
        sid = scene["scene_id"]
        if sid in SPATIALGEN_DUP:
            continue
        for room in scene["rooms"]:
            furn = [{**f, "furniture_category": f["furniture_category"].replace("_", " "),
                     "furniture_rotation": {**f["furniture_rotation"], "x": 0, "y": 0}}
                    for f in room["furniture"]]
            furn += openings(room, f"{sid}::{room['room_id']}")
            ring = room["room_boundary"]
            usable = ring and len(ring) >= 3 and all(_num(v) for pt in ring for v in pt[:2])   # convert_room's test
            fix = repair(ring) if usable and not Polygon(ring).is_valid else None
            ir = convert_room({**room, "room_boundary": fix or ring, "furniture": furn}, source=SOURCE,
                              uid=f"spatiallm:{sid}::{room['room_id']}",
                              group=GROUP.get(sid, f"spatiallm:{sid}"), boundary_type="polygon", center_z=True, front_offset_deg=-90,
                              extra=lambda f: {"structure": True} if f.get("structure") else {},
                              meta={"scene_id": sid, "front_known": True, "source_splits": room.get("source_splits"),
                                    "view": "spatiallm.json"})
            if fix:                                # provenance: the source ring in the room frame, its defect, both areas
                x0, y0 = min(x for x, _ in fix), min(y for _, y in fix)
                ir["meta"]["boundary_repaired"] = {
                    "method": "buffer(0)", "defect": explain_validity(Polygon(ring)).split("[")[0],
                    "area_src": round(abs(signed_area(ring)), 4), "area": round(Polygon(fix).area, 4),
                    "ring_src": [[x - x0, y - y0] for x, y in ring]}
            b = ir["boundary"]
            if b and not Polygon(b).is_valid:     # wall ring crosses/touches itself: cannot prove in-bounds
                ir["boundary_type"] = None
                ir["meta"]["boundary_invalid"] = True
            yield ir


if __name__ == "__main__":
    # 4 x 3 room at (10, 20); door centred on the bottom wall (+X direction), window on the right wall (+Y direction)
    room = {"room_id": "r", "room_type": "bedroom", "room_height": 2.6, "furniture": [],
            "room_boundary": [[10, 20], [14, 20], [14, 23], [10, 23]],
            "walls": [{"wall_index": i, "start": {"x": a[0], "y": a[1], "z": 0}, "end": {"x": b[0], "y": b[1], "z": 0},
                       "height": 2.6, "thickness": 0.0}
                      for i, (a, b) in enumerate([([10, 20], [14, 20]), ([14, 20], [14, 23]), ([14, 23], [10, 23]),
                                                  ([10, 23], [10, 20])])],
            "doors": [{"wall_index": 0, "position": {"x": 12, "y": 20, "z": 1.05}, "width": 0.9, "height": 2.1,
                       "sill_height": None}],
            "windows": [{"wall_index": 1, "position": {"x": 14, "y": 21.5, "z": 1.6}, "width": 1.2, "height": 1.4,
                         "sill_height": 0.9}]}
    ir = convert_room({**room, "furniture": openings(room, "s::r")}, source=SOURCE, uid="t", group="g",
                      boundary_type="polygon", center_z=True, front_offset_deg=-90,
                      extra=lambda f: {"structure": True} if f.get("structure") else {})
    d, w = ir["objects"]
    close = lambda a, b: all(abs(x - y) < 1e-6 for x, y in zip(a, b))
    assert d["category"] == "door" and d["structure"] and not d["tilted"], d
    assert close(d["size"], [0.9, 0.10, 2.1]) and close(d["pos"], [2, 0, 0]) and abs(d["yaw"]) < 1e-9, d
    assert w["category"] == "window" and w["structure"], w
    assert close(w["size"], [1.2, 0.10, 1.4]) and close(w["pos"], [4, 1.5, 0.9]) and abs(w["yaw"] - math.pi / 2) < 1e-9, w

    # wall rings through load() (iter_records stubbed with one scene): a zero-area spur at the min-x side (the frame
    # comes from the repaired ring), the real scene_000060::room_3 corner overshoot (2.5 x 6.5 cm lobe), two squares
    # touching at a vertex (two parts), a bow-tie (shoelace 0), a loop touching the ring from inside (a hole the
    # exterior would fill) and a loop inside traced the same way (shoelace 17, buffer(0) 16): the first two are
    # repaired and stay "polygon", the other four get boundary_type None; a 2-point ring and a ring with a null
    # coordinate reach convert_room untouched (boundary None) instead of raising in Polygon()
    chair = {"furniture_category": "chair", "furniture_instance_id": "c", "furniture_rotation": {"x": None, "y": None, "z": 0},
             "furniture_position": {"x": 11, "y": 21, "z": 0.4}, "furniture_size": {"width": 0.5, "length": 0.5, "height": 0.8}}
    rings = {"spur": [[10, 20], [14, 20], [14, 23], [10, 23], [10, 22], [9.5, 22], [10, 22]],
             "overshoot": [[6.1625, -0.275], [6.1375, -0.275], [6.1375, 1.4025], [3.25, 1.4025], [3.25, -0.21],
                           [6.1625, -0.21]],
             "touch": [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [1, 2], [1, 1], [0, 1]],
             "bowtie": [[0, 0], [2, 2], [2, 0], [0, 2]],
             "hole": [[0, 0], [4, 0], [4, 4], [2, 4], [3, 3], [1, 3], [2, 4], [0, 4]],
             "twice": [[0, 0], [4, 0], [4, 4], [2, 4], [2, 3], [1, 2], [3, 2], [2, 3], [2, 4], [0, 4]],
             "short": [[0, 0], [1, 0]], "null": [[0, 0], [1, 0], [None, 1]]}
    iter_records = lambda path: [{"scene_id": "s", "rooms": [{"room_id": k, "room_boundary": r, "furniture": [chair]}
                                                             for k, r in rings.items()]}]
    irs = {ir["meta"]["room_id"]: ir for ir in load("")}
    for k in ("spur", "overshoot"):
        ir, rep = irs[k], irs[k]["meta"]["boundary_repaired"]
        P = Polygon(ir["boundary"])
        assert ir["boundary_type"] == "polygon" and P.is_valid and P.bounds[:2] == (0, 0), ir
        assert rep["defect"] in ("Self-intersection", "Ring Self-intersection") and abs(rep["area"] - P.area) < 1e-3, rep
        assert abs(rep["area"] - rep["area_src"]) < REPAIR_TOL * rep["area_src"] and len(rep["ring_src"]) == len(rings[k])
    assert close(irs["spur"]["objects"][0]["pos"], [1, 1, 0]) and abs(Polygon(irs["spur"]["boundary"]).area - 12) < 1e-9
    assert min(x for x, _ in irs["spur"]["meta"]["boundary_repaired"]["ring_src"]) == -0.5
    assert abs(irs["overshoot"]["meta"]["boundary_repaired"]["area"] - 2.8875 * 1.6125) < 1e-4
    for k in ("touch", "bowtie", "hole", "twice"):
        assert irs[k]["boundary_type"] is None and irs[k]["meta"]["boundary_invalid"], irs[k]
        assert "boundary_repaired" not in irs[k]["meta"] and irs[k]["boundary"] is not None
    for k in ("short", "null"):
        assert irs[k]["boundary"] is None and irs[k]["boundary_type"] is None and len(irs[k]["objects"]) == 1, irs[k]
        assert not {"boundary_repaired", "boundary_invalid"} & set(irs[k]["meta"]), irs[k]
    print("spatiallm.py self-check ok")
