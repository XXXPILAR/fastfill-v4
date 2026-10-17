"""IL3D, 3D-FRONT subset (20,274 designer rooms): the canonical copy of 3D-FRONT rooms.

Verified conventions (scratch checks against IL3D's own USDZ meshes and the 3D-FRONT export):
- position = the 3D-FUTURE asset origin (IL3D copies the raw 3D-FRONT `pos`); every sampled mesh has its origin
  at the bottom center (y_min = 0, x/z centered), so it is the IR bottom center. XY are room-local (IL3D moved each
  room so its own floor mesh AABB starts at 0).
- size: the layout `bbox` is NOT usable as-is: for int-scale placements it is [x, z, y] of the mesh, otherwise
  [x, y, z] unscaled, and the IL3D export reads it as [w, h, l] for all. assets.json meta_data width/length/height
  equal the mesh extents along x / y(up) / z for all sampled assets, and the official USD script applies `scale` in
  the asset frame, so the placed local extents are meta x |scale|.
- rotation: the export's furniture_rotation.z = yaw of mesh +X (from the full rotateXYZ matrix; tilted -> x/y null).
  3D-FUTURE meshes face +Z (y-up) = IR local -Y, hence front_offset_deg = -90. Only x-mirrors (scale x < 0) occur,
  which do not flip the front.
- tilted placements (76, x/y null; 2 of them also z null): the export keeps the full world-from-local Z-up matrix
  (source_fields.rotation_matrix_zup; = the raw 3D-FRONT quaternion within 0.014 for all 76, IL3D rounds to whole
  degrees), local X / Y / Z = mesh x / -z / y(up) = placed_size's width / length / height, origin = bottom centre.
  `untilt` turns them into upright boxes by the internscenes.canon rules (see there): 44 (tilt p10/p50/p90 7/11/31
  deg) become the yaw-aligned bounding box of their 8 corners (flagged tilted, tilt_deg), 32 (22 lying on a side, 10
  tilted <= 3 deg) the exact box with permuted sizes; the 2 z-null ones (a 1 mm beam) no longer make 2 rooms incomplete.
- rooms: IL3D groups a house's objects by room TYPE, so 1,300 IL3D "rooms" merge objects of 2-10 source rooms
  (e.g. three bedrooms in one), and its single floor rectangle may belong to yet another room. Every IL3D object is
  matched to its 3D-FRONT export placement (same asset id and position under one per-room frame shift), which gives
  its true source room; each IL3D room is split into one IR room per source room.
- boundary / height: the exact floor polygon of the source room: the union of its Floor-mesh triangles from the raw
  house JSON (data/3D-FRONT/<house>.json, same [x, y, z] -> [x, -z] frame as the export), moved into IL3D's frame by
  the shift, boundary_type "polygon". The export's room_boundary is only the convex hull of those vertices (its README:
  "not the exact floor plan"). Its area is within 5% of the source room size for 13,496 of the 13,859 rooms that state
  one (the hull: 8,085). Gaps and spikes narrower than SPIKE_M (seams between meshes, zero-width slivers of
  degenerate triangles, up to 12 cm long) are closed / cut off, pieces and holes < ARTEFACT_M2 dropped; a floor that
  is still several pieces (21 rooms), has a real hole (20, stair voids), or whose ring is invalid as is or on the
  model's cm grid (`cm_valid`, 4) keeps the export hull, boundary_type "hull" (meta.boundary_from says which).
  Height: the export's floor-to-ceiling height (IL3D's supplemented height when the export has none and the room is
  IL3D's own floor room).
- anchors: assets.json placement flags are per ASSET, so they are used only when a single flag agrees with z.
- doors / windows: the 3D-FRONT export lists them per source room (room["doors"] / ["windows"]: position = opening
  centre, width, height, sill_height (windows), profile = the door's floor footprint [[x, y] x4] incl. the frame
  depth, or the window's vertical quad [[x, y, z] x4]); emitted as "structure" boxes in IL3D's frame (export minus
  the frame shift): local X along the wall = width (the profile side closest to it gives the wall direction), depth =
  door footprint area / width, 0.10 m for windows, bottom = sill_height or centre - height / 2. The export lists a
  door on only one of the two rooms it joins (1 of 5,999 door positions in the first 1,500 houses is on two lists),
  so a door listed on another room of the house is added when it opens into this one (`on_edge`, tagged listed_on):
  10,768 doors, rooms with a door 7,470 -> 13,465 of 21,832; doors > 0.3 m inside the boundary 5,165 -> 19.
  The export splits one door / bay window into several mesh pieces (its README: "the entry count is not the number
  of real doors/windows"): pieces lower than MIN_H (sill slabs) are dropped, and a piece whose footprint is >= half
  covered by a bigger piece of the same kind is merged into it (dropped), so each opening is shown once.
"""
import collections
import json
import math
import os

import numpy as np
from shapely.geometry import LinearRing, Polygon
from shapely.ops import unary_union

from fastfill.adapters.internscenes import tilted_bbox, upright
from fastfill.adapters.unified import _num, convert_room, iter_records
from fastfill.scene import clean_boundary, footprint

OPENING_T = 0.10
MIN_H = 0.15          # = build.FLAT_Z: a lower piece is a sill slab, not an opening
ARTEFACT_M2 = 0.01    # floor pieces / holes smaller than this are triangulation slivers (seen: <= 0.0005 m2)
SPIKE_M = 0.02        # narrower gaps / spikes (mesh seams, degenerate triangles) do not survive the model's cm grid
DOOR_TOL = 0.05       # a door footprint ends within 5 cm of the floor edges it opens onto
LEVEL_TOL = 0.3       # own doors start at the room's floor (98% exactly, else <= 0.2 m steps); storeys are >2 m apart

SOURCE = "IL3D_3dfront"
IL3D = "imChuling__3D_Room_Collections/IL3D_exported"
F3D = "BillLin66__3D_Room_Collections/3DFront_exported/3dfront.jsonl"
HOUSES = "BillLin66__3D_Room_Collections/3DFront_exported/data/3D-FRONT"


def load_assets(root):
    with open(os.path.join(root, IL3D, "IL3D/assets.json")) as f:
        return {a["model_id"]: a for a in json.load(f)}


def placed_size(f, asset):
    """Placed extents along mesh x, z, y = IR local X, Y, Z (before the front offset); None if unknown."""
    m = (asset or {}).get("meta_data") or {}
    s = f["source_fields"]["layout_object"].get("scale")
    if not (s and len(s) == 3 and all(isinstance(m.get(k), (int, float)) for k in ("width", "length", "height"))):
        return None
    return {"width": m["width"] * abs(s[0]), "length": m["height"] * abs(s[2]), "height": m["length"] * abs(s[1])}


def flag_anchor(f):
    fl = {k for k, v in (f.get("placement_flags") or {}).items() if v}
    z = (f.get("furniture_position") or {}).get("z")
    if z is None or len(fl) != 1:
        return None
    if fl == {"on_floor"} and z < 0.05:
        return "floor"
    if fl == {"on_ceiling"} and z > 1.0:
        return "ceiling"
    if fl == {"on_wall"} and z >= 0.05:
        return "wall"
    return None


def il3d_extra(f):
    if f.get("structure"):
        return {"structure": True, **({"listed_on": f["listed_on"]} if f.get("listed_on") else {})}
    return {"anchor": flag_anchor(f), "desc": f.get("description"),
            **({"tilt_deg": f["tilt_deg"]} if "tilt_deg" in f else {})}


def untilt(f):
    """Tilted placement (rotation x/y null) -> the same dict as an upright box (module doc for the matrix): a local axis
    within internscenes.TILT_TOL_DEG of vertical -> the exact box with permuted sizes, rotation {0, 0, yaw}; otherwise
    the yaw-aligned bounding box of its 8 corners, rotation {None, None, yaw} (convert_room flags it tilted) and
    tilt_deg. The box centre is kept; position = the new box's bottom centre."""
    r, p, s = f.get("furniture_rotation") or {}, f.get("furniture_position"), f.get("furniture_size")
    R = f["source_fields"].get("rotation_matrix_zup")
    if (r.get("x") == 0 and r.get("y") == 0) or not (p and all(_num(p.get(k)) for k in "xyz") and s and R):
        return f
    R, S = np.asarray(R, dtype=float), [s["width"], s["length"], s["height"]]
    c = np.array([p["x"], p["y"], p["z"]]) + R[:, 2] * S[2] / 2
    up = upright(R, S)
    if up:
        (w, l, h), yaw, _ = up
        rot, extra = {"x": 0.0, "y": 0.0, "z": yaw}, {}
    else:
        (w, l, h), yaw, _, tilt = tilted_bbox(R, S)
        rot, extra = {"x": None, "y": None, "z": yaw}, {"tilt_deg": tilt}
    return {**f, "furniture_position": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2]) - h / 2},
            "furniture_rotation": rot, "furniture_size": {"width": w, "length": l, "height": h}, **extra}


def export_rooms(root):
    """house uid -> [(source room id, hull in house frame, height, [(jid, x, y), ...], (doors, windows))]."""
    out = collections.defaultdict(list)
    for line in open(os.path.join(root, F3D)):
        d = json.loads(line)
        r = d["room"]
        pts = [(f["source_fields"]["jid"], f["furniture_position"]["x"], f["furniture_position"]["y"])
               for f in r["furniture"] if f["source_fields"].get("jid") and f["furniture_position"]]
        out[d["house_uid"]].append((d["provenance"]["source_room_id"], r["room_boundary"], r["room_height"], pts,
                                    (r.get("doors") or [], r.get("windows") or [])))
    return out


def floor_polygon(tris):
    """Floor triangles [(x, y) x3] -> the exact floor ring [[x, y], ...], or None unless it is one hole-free polygon."""
    u = unary_union([t for t in map(Polygon, tris) if t.area > 0])
    e = SPIKE_M / 2                   # mitre closing then opening: exact except for features narrower than SPIKE_M
    u = u.buffer(e, join_style="mitre").buffer(-2 * e, join_style="mitre").buffer(e, join_style="mitre")
    parts = [g for g in getattr(u, "geoms", [u]) if g.area >= ARTEFACT_M2]
    if len(parts) != 1 or any(Polygon(r).area >= ARTEFACT_M2 for r in parts[0].interiors):
        return None
    ring = [[round(x, 6), round(y, 6)] for x, y in Polygon(parts[0].exterior).simplify(1e-4).exterior.coords[:-1]]
    return ring if len(ring) >= 3 and Polygon(ring).is_valid else None


def cm_valid(b):
    """The boundary as the model sees it (convert_room's min-corner frame, then scene.clean_boundary) is valid."""
    x0, y0 = min(p[0] for p in b), min(p[1] for p in b)
    return clean_boundary([[x - x0, y - y0] for x, y in b]) is not None


def floor_polygons(path):
    """Raw 3D-FRONT house JSON -> {source room id (instanceid): (floor_polygon of its Floor meshes, floor z)}, house
    frame; (None, None) without Floor meshes."""
    with open(path) as f:
        h = json.load(f)
    mesh = {m.get("uid"): m for m in h.get("mesh") or []}
    out = {}
    for r in (h.get("scene") or {}).get("room") or []:
        tris, zs = [], []
        for c in r.get("children") or []:
            m = mesh.get(c.get("ref"))
            if m and m.get("type") == "Floor":      # mesh children carry identity transforms (as the export assumes)
                v, fc = m["xyz"], m["faces"]
                p = [(v[i], -v[i + 2]) for i in range(0, len(v) - 2, 3)]
                tris += [(p[fc[i]], p[fc[i + 1]], p[fc[i + 2]]) for i in range(0, len(fc) - 2, 3)]
                zs += v[1::3]
        out[r.get("instanceid")] = (floor_polygon(tris) if tris else None, min(zs) if zs else None)
    return out


def on_edge(door, ring, floor_z):
    """True if the door starts at this room's floor level (not another storey of the house) and its footprint runs
    along the room's floor edge for >= half its width, i.e. it opens into the room. On 1,571 sampled rooms 877 of 881
    own doors pass; other rooms' doors cover either >= 0.9 of their width (the far side of the same wall) or <= 0.2
    (a corner touch)."""
    pr, p, h = [q[:2] for q in door.get("profile") or []], door.get("position") or {}, door.get("height")
    if len(pr) < 3 or not (_num(door.get("width")) and _num(p.get("z")) and _num(h) and _num(floor_z)):
        return False
    if abs(p["z"] - h / 2 - floor_z) > LEVEL_TOL:
        return False
    fp = Polygon(pr).buffer(DOOR_TOL, join_style="mitre")
    return LinearRing(ring).intersection(fp).length >= 0.5 * door["width"]


def openings(doors_windows, shift, prefix):
    """Export doors / windows of one source room -> (structure furniture dicts in IL3D's frame, n_skipped).
    rotation z = wall direction + 90 so that convert_room(front_offset_deg=-90) gives yaw = wall direction and
    size [width, depth, height]."""
    out, skipped = [], 0
    for kind, items in zip(("door", "window"), doors_windows):
        boxes = []
        for k, o in enumerate(items):
            p, w, h, pr = o.get("position"), o.get("width"), o.get("height"), o.get("profile") or []
            xy = [q[:2] for q in pr]
            segs = [(a, b) for a, b in zip(xy, xy[1:] + xy[:1]) if math.dist(a, b) > 1e-3]
            if not (p and all(_num(p.get(c)) for c in "xyz") and _num(w) and _num(h) and w > 0 and h > 0 and segs):
                skipped += 1
                continue
            a, b = min(segs, key=lambda s: abs(math.dist(*s) - w))       # the side along the wall
            depth = Polygon(xy).area / w if len(pr[0]) == 2 and len(xy) >= 3 else 0.0
            z = o["sill_height"] if _num(o.get("sill_height")) else p["z"] - h / 2
            if h < MIN_H:
                skipped += 1
                continue
            ang = math.atan2(b[1] - a[1], b[0] - a[0])
            d = depth if depth > 0.01 else OPENING_T
            fp = Polygon(footprint({"pos": [p["x"], p["y"]], "size": [w, d], "yaw": ang}))
            boxes.append((fp.area * h, fp, {
                "furniture_category": kind, "furniture_instance_id": f"{prefix}::{kind}_{k}", "structure": True,
                **({"listed_on": o["listed_on"]} if o.get("listed_on") else {}),
                "furniture_position": {"x": p["x"] - shift[0], "y": p["y"] - shift[1], "z": z},
                "furniture_rotation": {"x": 0.0, "y": 0.0, "z": math.degrees(ang) + 90},
                "furniture_size": {"width": d, "length": w, "height": h}}))
        kept = []                       # biggest piece first; a piece half inside the kept ones is the same opening
        for _, fp, f in sorted(boxes, key=lambda t: -t[0]):
            if kept and unary_union([g for g, _ in kept]).intersection(fp).area >= 0.5 * fp.area:
                skipped += 1
                continue
            kept.append((fp, f))
        out += [f for _, f in kept]
    return out, skipped


def frame_shift(furniture, pts, tol=0.02):
    """Shift s with export_pos = il3d_pos + s supported by the most shared placements; (s, votes) or (None, 0)."""
    pairs = [((f["furniture_position"]["x"], f["furniture_position"]["y"]), pts[f["furniture_asset_id"]])
             for f in furniture if f.get("furniture_position") and f.get("furniture_asset_id") in pts]
    best, votes = None, 0
    for p, cands in pairs:
        for q in cands:
            s = (q[0] - p[0], q[1] - p[1])
            n = sum(any(abs(c[0] - a[0] - s[0]) < tol and abs(c[1] - a[1] - s[1]) < tol for c in cs) for a, cs in pairs)
            if n > votes:
                best, votes = s, n
    return best, votes


def split_by_source_room(furniture, rooms, own=None, tol=0.02):
    """-> (shift, {export room index: [furniture]}, [unassigned furniture])."""
    pts = collections.defaultdict(list)
    for k, r in enumerate(rooms):
        for jid, x, y in r[3]:
            pts[jid].append((x, y, k))
    shift, _ = frame_shift(furniture, {j: [(x, y) for x, y, _ in v] for j, v in pts.items()}, tol)
    hits = []
    for f in furniture:
        p = f.get("furniture_position")
        hits.append({k for x, y, k in pts.get(f.get("furniture_asset_id"), ())
                     if shift and p and abs(x - p["x"] - shift[0]) < tol and abs(y - p["y"] - shift[1]) < tol})
    # a placement listed in several source rooms goes to the room most of this room's objects come from,
    # ties to IL3D's own room
    votes = collections.Counter(k for h in hits for k in h)
    for k, r in enumerate(rooms):
        votes[k] += 0.5 * (r[0] == own)
    parts, rest = collections.defaultdict(list), []
    for f, h in zip(furniture, hits):
        (parts[max(sorted(h), key=votes.__getitem__)] if h else rest).append(f)
    return shift, parts, rest


def load(root):
    assets = load_assets(root)
    exp = export_rooms(root)
    floors = {}
    for scene in iter_records(os.path.join(root, IL3D, "3dfront.json")):
        for room in scene["rooms"]:
            sf = room["source_fields"]
            house = sf["original_house_file"][:-len(".json")]
            if house not in floors:
                path = os.path.join(root, HOUSES, house + ".json")
                floors[house] = floor_polygons(path) if os.path.exists(path) else {}
            furniture = []
            for f in room["furniture"]:
                f = dict(f)
                a = assets.get(f.get("furniture_asset_id"))
                # 3D-FUTURE category; for the ~12k assets without one (IL3D label "Others": potted plants, fridges,
                # pianos ...) the asset's own assets.json meta_data category
                f["furniture_category"] = (f.get("asset_category") or ((a or {}).get("meta_data") or {}).get("category")
                                           or f.get("furniture_category"))
                f["furniture_size"] = placed_size(f, a)
                furniture.append(f)
            rooms = exp.get(house, [])
            shift, parts, rest = split_by_source_room(furniture, rooms, sf.get("original_roomId"))
            if len(parts) == 1:          # unassigned objects (NaN position) can only belong to the one source room
                parts = {k: v + rest for k, v in parts.items()}
            for k, fl in (parts or {None: furniture}).items():
                srid, hull, height = rooms[k][:3] if k is not None else (None, None, None)
                if height is None and srid == sf.get("original_roomId"):
                    height = room.get("room_height")
                poly, floor_z = floors[house].get(srid, (None, None))
                if poly and not cm_valid([[x - shift[0], y - shift[1]] for x, y in poly]):
                    poly = None
                ring = poly or hull
                boundary = [[x - shift[0], y - shift[1]] for x, y in ring] if ring else None
                uid = f"il3d:{scene['scene_id']}" + (f"::{srid}" if len(parts) > 1 else "")
                ops, n_skip = [], 0
                if k is not None and len(rooms[k]) > 4:
                    doors, windows = rooms[k][4]
                    doors = doors + [{**o, "listed_on": r[0]} for j, r in enumerate(rooms) if j != k and ring
                                     for o in r[4][0] if on_edge(o, ring, floor_z)]
                    ops, n_skip = openings((doors, windows), shift, uid)
                ir = convert_room({**room, "furniture": [untilt(f) for f in fl] + ops, "room_boundary": boundary,
                                   "room_height": height},
                                  source=SOURCE, uid=uid, group=f"3dfront:{house.lower()}", subset="3D-FRONT",
                                  boundary_type="polygon" if poly else "hull", front_offset_deg=-90, extra=il3d_extra,
                                  meta={"house": house, "source_room_id": srid, "il3d_room_id": sf.get("original_roomId"),
                                        "il3d_scene_id": scene["scene_id"], "split_from_il3d_room": len(parts) > 1,
                                        "boundary_from": "3dfront_floor_mesh_polygon" if poly else "3dfront_export_floor_hull",
                                        "front_known": True, "n_openings": len(ops), "n_openings_skipped": n_skip,
                                        "n_doors_listed_on_neighbour": sum("listed_on" in o for o in ops),
                                        "size_from": "assets.json mesh extents x layout scale"})
                if len(parts) > 1 and rest:  # objects whose source room is unknown: the split rooms are incomplete
                    ir["meta"]["n_incomplete"] += len(rest)
                yield ir


if __name__ == "__main__":
    f = {"furniture_position": {"x": 1.0, "y": 2.0, "z": 0.0}, "furniture_asset_id": "a",
         "placement_flags": {"on_floor": True}, "source_fields": {"layout_object": {"scale": [-2, 1, 1]}}}
    assert placed_size(f, {"meta_data": {"width": 1, "length": 0.5, "height": 3}}) == {"width": 2, "length": 3, "height": 0.5}
    assert flag_anchor(f) == "floor"
    g = {**f, "furniture_position": {"x": 3.0, "y": 2.0, "z": 0.0}, "furniture_asset_id": "b"}
    h = {**f, "furniture_position": None}
    assert frame_shift([f, g], {"a": [(6.0, 7.0), (0.0, 0.0)], "b": [(8.0, 7.0)]}) == ((5.0, 5.0), 2)
    shift, parts, rest = split_by_source_room([f, g, h], [("r0", None, None, [("a", 6.0, 7.0)]),
                                                          ("r1", None, None, [("b", 8.0, 7.0), ("a", 0.0, 0.0)])])
    assert shift == (5.0, 5.0) and parts == {0: [f], 1: [g]} and rest == [h]
    # a door on a wall running +Y (footprint 0.9 along the wall, 0.24 deep) and a window on a wall running +X
    door = {"position": {"x": 6.0, "y": 7.45, "z": 1.05}, "width": 0.9, "height": 2.1, "sill_height": None,
            "profile": [[5.88, 7.0], [5.88, 7.9], [6.12, 7.9], [6.12, 7.0]]}
    win = {"position": {"x": 8.0, "y": 5.0, "z": 1.65}, "width": 1.2, "height": 1.5, "sill_height": 0.9,
           "profile": [[7.4, 5.0, 0.9], [7.4, 5.0, 2.4], [8.6, 5.0, 2.4], [8.6, 5.0, 0.9]]}
    pane = {**win, "profile": [[7.5, 5.0, 0.9], [7.5, 5.0, 2.4], [8.5, 5.0, 2.4], [8.5, 5.0, 0.9]], "width": 1.0}
    sill = {**win, "height": 0.05}
    ops, skip = openings(([door], [pane, win, sill, {"position": None}]), (5.0, 5.0), "t")
    assert skip == 3 and [o["furniture_category"] for o in ops] == ["door", "window"], ops   # pane merged, sill dropped
    assert ops[1]["furniture_size"]["length"] == 1.2
    ir = convert_room({"furniture": ops, "room_boundary": [[0, 0], [4, 0], [4, 3], [0, 3]]}, source="t", uid="t", group="g",
                      boundary_type="hull", front_offset_deg=-90, extra=il3d_extra)
    d, w = ir["objects"]
    assert d["structure"] and abs(d["size"][0] - 0.9) < 1e-9 and abs(d["size"][1] - 0.24) < 1e-6 and d["size"][2] == 2.1, d
    assert abs(d["pos"][0] - 1) < 1e-9 and abs(d["pos"][1] - 2.45) < 1e-9 and abs(d["pos"][2]) < 1e-9, d
    assert abs(d["yaw"] - math.pi / 2) < 1e-9 and w["size"] == [1.2, OPENING_T, 1.5] and w["pos"][2] == 0.9 and abs(w["yaw"]) < 1e-9, w
    # the same door on the neighbour's list is merged into the own one; a neighbour-only door keeps its provenance
    nb = {**door, "position": {**door["position"], "y": 8.45}, "profile": [[5.88, 8.0], [5.88, 8.9], [6.12, 8.9], [6.12, 8.0]]}
    ops, skip = openings(([door, {**door, "listed_on": "r1"}, {**nb, "listed_on": "r1"}], []), (5.0, 5.0), "t")
    assert skip == 1 and [o.get("listed_on") for o in ops] == [None, "r1"], ops
    ir = convert_room({"furniture": ops, "room_boundary": [[0, 0], [4, 0], [4, 3], [0, 3]]}, source="t", uid="t", group="g",
                      boundary_type="polygon", front_offset_deg=-90, extra=il3d_extra)
    assert "listed_on" not in ir["objects"][0] and ir["objects"][1]["listed_on"] == "r1"
    # exact floor: an L from two meshes with a 0.05 mm seam and a sliver -> one 6-vertex ring; a real hole or two
    # separate floors -> None (the hull is kept)
    rect = lambda x0, y0, x1, y1: [((x0, y0), (x1, y0), (x1, y1)), ((x0, y0), (x1, y1), (x0, y1))]
    ring = floor_polygon(rect(0, 0, 4, 2) + rect(0, 2.00005, 2, 4) + [((5, 5), (5.01, 5), (5, 5.01))])
    assert len(ring) == 6 and abs(Polygon(ring).area - 12) < 1e-3 and Polygon(ring).is_valid, ring
    # a 5 mm slit between two meshes and a 12 cm zero-width spike (both collapse on the cm grid) -> the 4 x 3 room
    ring = floor_polygon(rect(0, 0, 2, 3) + rect(2.005, 0, 4, 3) + [((4, 1), (4.12, 1.0005), (4, 1.001))])
    assert len(ring) == 4 and abs(Polygon(ring).area - 12) < 1e-6, ring
    assert floor_polygon(rect(0, 0, 3, 1) + rect(0, 2, 3, 3) + rect(0, 1, 1, 2) + rect(2, 1, 3, 2)) is None
    assert floor_polygon(rect(0, 0, 1, 1) + rect(2, 0, 3, 1)) is None
    # room il3d:b68c68fc: a vertex lands on another edge once rounded to cm in the room frame
    assert cm_valid(ring) and not cm_valid([[5.076, 3.18], [5.076, 0], [6, 0], [6, 4], [5.12, 4], [5.12, 3.14], [5, 3.26], [5, 3.18]])
    # a door on the far side of this room's wall opens into it; a door that only touches its corner does not
    room, dz = [[0, 0], [4, 0], [4, 3], [0, 3]], {"position": {"x": 0, "y": 0, "z": 1.05}, "height": 2.1}
    assert on_edge({**dz, "width": 0.9, "profile": [[1, 3], [1.9, 3], [1.9, 3.24], [1, 3.24]]}, room, 0.0)
    assert not on_edge({**dz, "width": 0.9, "profile": [[1, 3], [1.9, 3], [1.9, 3.24], [1, 3.24]]}, room, -2.9)  # upstairs
    assert not on_edge({**dz, "width": 0.9, "profile": [[4, 3], [4.24, 3], [4.24, 3.9], [4, 3.9]]}, room, 0.0)
    # tilted placement (Z-up matrix, 30 deg about local X) -> yaw-aligned bounding box, flagged tilted; lying flat
    # (90 deg about X: local Y vertical) -> the exact box with permuted sizes, not tilted; yaw-only -> unchanged
    rx = lambda a: [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
    t = {"furniture_category": "vase", "furniture_position": {"x": 1.0, "y": 1.0, "z": 0.5},
         "furniture_rotation": {"x": None, "y": None, "z": 0.0}, "furniture_size": {"width": 1.0, "length": 0.2, "height": 0.1},
         "source_fields": {"rotation_matrix_zup": rx(math.pi / 6)}}
    c30, s30 = math.cos(math.pi / 6), math.sin(math.pi / 6)
    lying = untilt({**t, "source_fields": {"rotation_matrix_zup": rx(math.pi / 2)}})
    assert untilt({**t, "furniture_rotation": {"x": 0.0, "y": 0.0, "z": 0.0}})["furniture_position"]["z"] == 0.5
    ir = convert_room({"furniture": [untilt(t), lying], "room_boundary": [[0, 0], [3, 0], [3, 3], [0, 3]]}, source="t",
                      uid="t", group="g", front_offset_deg=-90, extra=il3d_extra)
    o, q = ir["objects"]
    close = lambda a, b: all(abs(x - y) < 1e-9 for x, y in zip(a, b))
    assert o["tilted"] and abs(o["tilt_deg"] - 30) < 1e-9 and close(o["size"], [0.2 * c30 + 0.1 * s30, 1.0, 0.2 * s30 + 0.1 * c30]), o
    assert close(o["pos"], [1, 1 - 0.05 * s30, 0.5 - 0.1 * s30]) and abs(o["yaw"] - 1.5 * math.pi) < 1e-9, o   # centre kept
    assert not q["tilted"] and "tilt_deg" not in q and close(q["size"], [0.1, 1.0, 0.2]) and close(q["pos"], [1, 0.95, 0.4]), q
    print("il3d_3dfront self-check ok")
