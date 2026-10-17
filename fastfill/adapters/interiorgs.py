"""InteriorGS (Manycore designer scenes, 1,000 houses / 6,607 rooms). View used: interiorgs.json only (full
scenes, every room once); json_single.json + multiroom/*.json repackage the same rooms and are ignored.

Upstream room assignment can lose objects at repeated/degenerate profile edges. For scene.unassigned_objects,
recover only a unique room covering the mean of the eight source_fields.bounding_box corners, using the same
conservative ring repair as conversion. Existing assignments stay authoritative. Shared edges, overlapping rooms,
outside centers, invalid geometry and duplicate IDs remain excluded with IDs/reasons in meta.membership_recovery;
no nearest-room assignment. A missing/invalid room polygon prevents proving uniqueness, so recovery is withheld.

Export (aggregate_scenes.py): meters, Z-up, scene-global XY, room_boundary = structure.rooms[].profile (floor
polygon), room_height from the room_box. The exported furniture_rotation/size are longest-edge geometry
([0, 180) yaw, no front), so this adapter re-derives pose from source_fields.bounding_box instead:
corners 0-3 are the bottom face in a fixed local order and edge c1-c2 is the object's BACK (front test on all
1,000 scenes: bed 0.85, toilet 0.94, painting 0.90 touch the wall with c1-c2, c3-c0 ~0.00), so
front = midpoint(c3, c0) - midpoint(c1, c2); IR size = [|c0c1| (depth, along front), |c1c2|, height].
Caveat: custom cabinetry (labels wardrobe / cabinet / wall cabinet) does not follow this frame (wall side ~uniform),
so their yaw is only reliable up to the box axes: those objects get front_known=False.
Boxes whose top face is shifted from the bottom face or whose bottom is not level (7,355, all rectangular) go
through internscenes.upright / tilted_bbox as internscenes.canon does, with local X = c0 - c1 (front), Y = c2 - c1,
Z = c4 - c0: 3,704 are upright boxes with permuted axes (corners 0-3 on a side face: strip lights, wall panels,
lying boxes) and 33 lean < 5 deg, re-expressed as yaw-only boxes (yaw = heading of local X); the other 3,618 (cigars
racked at 30 deg, leaning books, storage racks) become the yaw-aligned bounding box of their 8 corners, tilted
(rotation x/y None) with "tilt_deg". Bottom = box centre - height / 2. When local X is the (near-)vertical axis (2,683
boxes: vents, ceiling mouldings, wall designs, lying food boxes) yaw is the heading of local Y and front_known=False:
the front points up or down, so the source gives no horizontal front.
123 boxes are level but upside down (local Z = c4 - c0 points down: 103 rotated, 20 mirrored; wall designs, vents,
paintings, hanging wine glasses): same path, so height is the real one, not the ~0 of c0-c3 minus themselves, and yaw
stays the heading of c0 - c1 (the world image of local +X, i.e. of the front, whichever way the box was flipped).
Doors / windows: scene["openings"] (= structure.holes, profile = vertical polygon in the wall, scene Z; every room
floor is at z = 0) of type DOOR / WINDOW, matched to a room geometrically, not through room.doors/windows (most
openings have room_indices []): the profile lies on the room's wall line (the ring is the wall's inner face) or on
the wall's centre line or far face, so an opening belongs to every room with a ring edge parallel to it (profile
spread < PAR_TOL off the edge line), between WALL_TOL inside and thickness + WALL_TOL outside it, covering more than
half its width (a door in a shared wall goes to both rooms). Emitted as "structure" boxes on that edge (the wall line):
local X along the wall = the profile's extent along the edge (the export's width = max(X span, Y span) is wrong for
oblique openings), OPENING_T across it (as the other sources' openings; the export's thickness is the wall's),
bottom = min profile z, height = profile z span. 5,714 / 5,716 doors and 6,211 / 6,273 windows match a room (4,554 doors
and 233 windows two); 50 of the unmatched windows are flat chords across curved walls (ring = 0.2 m segments), not
handled. HOLE / OPENING types (passages without a leaf) are not emitted. The export repeats some holes (0293_840751
holes 15-18 are one window; 0395_840187 holes 23/24 are one door on both faces of a 6 cm wall): a box more than half
inside a bigger one of its kind on the same edge is skipped (32 boxes in 29 rooms: 27 doors, 5 windows;
meta.n_openings_repeated).
Wall rings that are invalid only through a corner overshoot (the ring crosses itself around a few-cm lobe), a zero-area
spur or a self-touching vertex (169 of the 171 invalid rings, area change <= 0.06 m2) are repaired with buffer(0) when
it gives one hole-free polygon within REPAIR_TOL of the ring's shoelace area (meta.boundary_repaired); other invalid
rings get boundary_type None.
"""
import math
import os
import re
from collections import Counter

import numpy as np
from shapely.geometry import Point, Polygon, box as rect
from shapely.ops import unary_union

from fastfill.adapters.internscenes import tilted_bbox, upright
from fastfill.adapters.unified import _num, convert_room, iter_records, signed_area
from fastfill.scene import footprint, head

SOURCE = "InteriorGS"
GROUP = {"0755_840824": "structured3d:scene_01482",   # same design as a Structured3D house (room boundary + furniture match)
         # duplicated InteriorGS houses that split.py's content_key does not link (rooms differ by a few objects)
         "0565_840943": "interiorgs:0478_840944", "0684_841508": "interiorgs:0521_840344"}
TILT_TOL = 0.02   # m
OPENING_T, PAR_TOL, WALL_TOL = 0.10, 0.02, 0.05   # m
REPAIR_TOL = 0.01   # relative area change allowed when repairing a wall ring
# Only the generic custom-cabinetry labels: TV / basin / shoe / wine / display / storage cabinets follow the corner
# order (back edge c1-c2 on the wall 0.51-0.82, front edge c3-c0 <= 0.10), plain wardrobe / cabinet / wall cabinet do not
UNRELIABLE_FRONT = re.compile(r"^(wardrobe|cabinet|wall cabinet)s?( doors)?$", re.I)


def _pose(f):
    """Export furniture -> unified-schema furniture with front-aware yaw/size, or None fields if no bbox."""
    bb = (f.get("source_fields") or {}).get("bounding_box")
    if not bb or len(bb) != 8:
        return {**f, "furniture_rotation": None}
    c = [(p["x"], p["y"], p["z"]) for p in bb]
    bz = [p[2] for p in c[:4]]
    if (max(bz) - min(bz) > TILT_TOL or c[4][2] < c[0][2] - TILT_TOL or      # bottom not level, or local Z down
            max(math.hypot(c[i][0] - c[i + 4][0], c[i][1] - c[i + 4][1]) for i in range(4)) > TILT_TOL):
        return _upright(f, np.array(c))
    fx = (c[3][0] + c[0][0] - c[1][0] - c[2][0]) / 2
    fy = (c[3][1] + c[0][1] - c[1][1] - c[2][1]) / 2
    return {**f,
            "furniture_position": {"x": sum(p[0] for p in c[:4]) / 4, "y": sum(p[1] for p in c[:4]) / 4, "z": min(bz)},
            "furniture_rotation": {"x": 0.0, "y": 0.0, "z": math.degrees(math.atan2(fy, fx))},
            "furniture_size": {"width": math.hypot(fx, fy),
                               "length": math.hypot(c[2][0] - c[1][0], c[2][1] - c[1][1]),
                               "height": max(p[2] for p in c) - min(bz)}}


def _upright(f, c):
    """8 corners (array 8x3) of a box whose bottom face is not level -> the upright box (permuted axes) or the
    yaw-aligned bounding box of the corners (tilted: rotation x/y None, "tilt_deg"), as internscenes.canon."""
    e = np.array([c[0] - c[1], c[2] - c[1], c[4] - c[0]]).T          # local X (front), Y, Z edges as columns
    size = np.linalg.norm(e, axis=0)
    R = e / np.maximum(size, 1e-12)
    up = upright(R, size)
    if up is None:
        (sx, sy, sz), yaw, k, tilt = tilted_bbox(R, size)
        rot, extra = None, {"tilt_deg": tilt}
    else:
        (sx, sy, sz), yaw, k = up
        rot, extra = 0.0, {}
    x, y, z = c.mean(axis=0)
    return {**f, "furniture_position": {"x": x, "y": y, "z": z - sz / 2},
            "furniture_rotation": {"x": rot, "y": rot, "z": yaw},
            "furniture_size": {"width": sx, "length": sy, "height": sz}, "vertical_axis": k, **extra}


def openings(scene_openings, ring, prefix):
    """Scene DOOR / WINDOW openings in this room's walls -> (structure furniture on the wall line (see module doc),
    n_repeated). Rotation z = edge direction, size [extent along the edge, OPENING_T, height] (front_offset_deg 0).
    The export repeats some holes (exact profile copies, or the same hole on both faces of a thin wall, a few cm
    apart): biggest first, an opening more than half inside the kept ones of its kind on the same edge (in the wall
    plane: along-edge x z) is a repeat and is skipped, as il3d_3dfront.openings does."""
    b = [tuple(p[:2]) for p in ring]
    if signed_area(b) < 0:
        b = b[::-1]
    cand = []
    for o in scene_openings:
        kind = {"DOOR": "door", "WINDOW": "window"}.get(o.get("opening_type"))
        pr, t = o.get("profile") or [], o.get("thickness")
        if not (kind and pr and _num(t)):
            continue
        best = None
        for i, ((x0, y0), (x1, y1)) in enumerate(zip(b, b[1:] + b[:1])):
            L = math.hypot(x1 - x0, y1 - y0)
            if L < 1e-9:
                continue
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            s = [(p[0] - x0) * ux + (p[1] - y0) * uy for p in pr]        # along the edge
            n = [(p[1] - y0) * ux - (p[0] - x0) * uy for p in pr]        # across it, > 0 inside the (CCW) room
            ov = min(max(s), L) - max(min(s), 0)
            if max(n) - min(n) < PAR_TOL and min(n) >= -t - WALL_TOL and max(n) <= WALL_TOL and \
                    ov > 0.5 * (max(s) - min(s)) and (best is None or ov > best[0]):
                best = (ov, i, x0, y0, ux, uy, min(s), max(s))
        if best is None:
            continue
        _, i, x0, y0, ux, uy, s0, s1 = best
        z0, z1 = min(p[2] for p in pr), max(p[2] for p in pr)
        cand.append((kind, i, rect(s0, z0, s1, z1), {
            "furniture_category": kind, "furniture_instance_id": f"{prefix}::hole_{o['source_hole_index']}",
            "structure": True,
            "furniture_position": {"x": x0 + ux * (s0 + s1) / 2, "y": y0 + uy * (s0 + s1) / 2, "z": z0},
            "furniture_rotation": {"x": 0.0, "y": 0.0, "z": math.degrees(math.atan2(uy, ux))},
            "furniture_size": {"width": s1 - s0, "length": OPENING_T, "height": z1 - z0}}))
    kept, keep = {}, set()
    for kind, i, r, f in sorted(cand, key=lambda t: -t[2].area):
        ks = kept.setdefault((kind, i), [])
        if not (ks and unary_union(ks).intersection(r).area > 0.5 * r.area):
            ks.append(r)
            keep.add(id(f))
    out = [f for *_, f in cand if id(f) in keep]          # source hole order
    return out, len(cand) - len(out)


def repair_ring(ring):
    """Invalid wall ring -> (buffer(0) exterior, meta) when that is one hole-free polygon within REPAIR_TOL of the
    ring's shoelace area (corner overshoot, spur, self-touching vertex); else (ring, {})."""
    if not ring or len(ring) < 3 or Polygon(ring).is_valid:
        return ring, {}
    a, q = abs(signed_area([p[:2] for p in ring])), Polygon(ring).buffer(0)
    if q.geom_type != "Polygon" or q.interiors or not abs(q.area - a) < REPAIR_TOL * a:
        return ring, {}
    return [list(p) for p in q.exterior.coords[:-1]], {"boundary_repaired": "buffer(0)",
                                                      "boundary_area_change_m2": round(q.area - a, 4)}


def drop_labelled(objs):
    """IR objects minus the opening boxes the source already has as a labelled furniture box: a 'window' / 'door' box
    of the same head noun that overlaps the opening's footprint and covers more than half of its wall-plane rectangle
    (along the wall x z). 207 windows + 4 doors in 127 rooms: mostly boxes centred on the same wall line 0.13 m deep,
    7 window panes 2-5 cm thin a few cm off the line; a bay-window platform labelled 'window' below the opening does
    not count. The source's own box stays (meta.n_openings_dropped)."""
    own = [(head(o), Polygon(footprint(o)), o) for o in objs if not o.get("structure") and head(o) in ("door", "window")]

    def covered(o):
        fp, ux, uy = Polygon(footprint(o)), math.cos(o["yaw"]), math.sin(o["yaw"])
        c = o["pos"][0] * ux + o["pos"][1] * uy
        for h, p, q in own:
            if h != o["category"] or not p.intersection(fp).area > 0:
                continue
            s = [x * ux + y * uy for x, y in p.exterior.coords]
            ds = min(max(s), c + o["size"][0] / 2) - max(min(s), c - o["size"][0] / 2)
            dz = min(q["pos"][2] + q["size"][2], o["pos"][2] + o["size"][2]) - max(q["pos"][2], o["pos"][2])
            if ds > 0 and dz > 0 and ds * dz > 0.5 * o["size"][0] * o["size"][2]:
                return True
        return False
    return [o for o in objs if not (o.get("structure") and covered(o))]


def _extra(f):
    if f.get("structure"):
        return {"structure": True}
    unknown = UNRELIABLE_FRONT.search(f["furniture_category"]) or f.get("vertical_axis") == 0   # front up / down
    x = {"front_known": False} if unknown else {}
    return {**x, "tilt_deg": f["tilt_deg"]} if "tilt_deg" in f else x


def _membership_id(f):
    """Scene-local export identity, with the raw instance/index metadata as fallbacks."""
    fields = f.get("source_fields") or {}
    for prefix, value in (("", f.get("furniture_instance_id")), ("ins:", fields.get("ins_id")),
                          ("source_index:", f.get("source_index"))):
        if value is not None and value != "":
            return prefix + str(value)
    return None


def _membership_polygon(room):
    ring = room.get("room_boundary")
    if not isinstance(ring, list) or len(ring) < 3 or not all(
            isinstance(p, (list, tuple)) and len(p) >= 2 and all(_num(v) for v in p[:2]) for p in ring):
        return None
    fixed, _ = repair_ring([p[:2] for p in ring])
    poly = Polygon(fixed)
    return poly if poly.is_valid and not poly.is_empty and poly.area > 0 else None


def _membership_center(f):
    corners = (f.get("source_fields") or {}).get("bounding_box")
    if not isinstance(corners, list) or len(corners) != 8 or not all(
            isinstance(p, dict) and all(_num(p.get(k)) for k in "xyz") for p in corners):
        return None
    return Point(*(sum(p[k] / 8 for p in corners) for k in "xy"))


def _membership_rooms(scene):
    """Return (room, recovery metadata) pairs without modifying the export or inventing room membership."""
    rooms, pending = scene["rooms"], scene.get("unassigned_objects") or []
    if not pending:
        return [(r, {}) for r in rooms]
    polygons = [_membership_polygon(r) for r in rooms]
    assigned = {_membership_id(f) for r in rooms for f in r.get("furniture") or []}
    counts = Counter(_membership_id(f) for f in pending)

    def locate(f):
        identity = _membership_id(f)
        if identity is None:
            return None, "missing_id"
        if identity in assigned:
            return None, "already_assigned"
        if counts[identity] > 1:
            return None, "duplicate_id"
        center = _membership_center(f)
        if center is None:
            return None, "invalid_bbox"
        if any(p is None for p in polygons):
            return None, "invalid_boundary"
        hits = [i for i, p in enumerate(polygons) if p.covers(center)]
        return (hits[0], None) if len(hits) == 1 else (None, "ambiguous" if hits else "outside")

    decisions = [(f, _membership_id(f) or f"unassigned:{i}", *locate(f)) for i, f in enumerate(pending)]
    reasons = sorted({reason for _, _, _, reason in decisions if reason and reason != "already_assigned"})
    common = {"method": "unique_bbox_center", "scene_unassigned": len(pending),
              "scene_recovered": sum(reason is None for _, _, _, reason in decisions),
              "scene_already_assigned": [key for _, key, _, reason in decisions if reason == "already_assigned"],
              "scene_unresolved": {reason: [key for _, key, _, why in decisions if why == reason] for reason in reasons}}
    return [({**room, "furniture": list(room.get("furniture") or []) + [f for f, _, j, _ in decisions if j == i]},
             {"membership_recovery": {**common, "recovered_ids": [key for _, key, j, _ in decisions if j == i]}})
            for i, room in enumerate(rooms)]


def load(root):
    path = os.path.join(root, "imChuling__3D_Room_Collections/InteriorGS_exported/interiorgs.json")
    for scene in iter_records(path):
        sid = scene["scene_id"]
        for room, recovery in _membership_rooms(scene):
            ring, fix = repair_ring(room["room_boundary"])
            furn = [_pose(f) for f in room["furniture"]]
            ops, n_rep = openings(scene.get("openings") or [], ring or [], sid)
            furn += ops
            ir = convert_room({**room, "room_boundary": ring, "furniture": furn},
                              source=SOURCE, uid=f"interiorgs:{sid}::{room['room_id']}", group=GROUP.get(sid, f"interiorgs:{sid}"),
                              boundary_type="polygon", extra=_extra,
                              meta={"scene_id": sid, "front_known": True, "front_source": "bbox corner order",
                                    "room_type_available": False, "view": "interiorgs.json",
                                    "n_permuted_axes": sum(f.get("vertical_axis", 2) != 2 for f in furn),
                                    "n_openings_repeated": n_rep, **fix, **recovery})
            objs = drop_labelled(ir["objects"])
            ir["meta"]["n_openings_dropped"] = len(ir["objects"]) - len(objs)
            ir["objects"] = objs
            b = ir["boundary"]
            if b and not Polygon(b).is_valid:     # self-intersecting source profile: cannot prove in-bounds
                ir["boundary_type"] = None
                ir["meta"]["boundary_invalid"] = True
            yield ir


if __name__ == "__main__":
    # bed-like box whose back edge c1-c2 lies on the wall x=0 (clockwise corners): front = +X, depth 0.6, width 2
    f = {"source_fields": {"bounding_box": [{"x": x, "y": y, "z": z} for z in (0.0, 1.0) for x, y in
                                            ((0.6, -1.0), (0.0, -1.0), (0.0, 1.0), (0.6, 1.0))]}}
    p = _pose(f)
    assert abs(p["furniture_rotation"]["z"]) < 1e-9 and p["furniture_rotation"]["x"] == 0.0
    assert abs(p["furniture_size"]["width"] - 0.6) < 1e-9 and abs(p["furniture_size"]["length"] - 2.0) < 1e-9
    assert p["furniture_position"] == {"x": 0.3, "y": 0.0, "z": 0.0}

    from fastfill.adapters.internscenes import rot_zxy

    def box(ctr, R, size):       # export corner order: c0 - c1 = local X, c2 - c1 = local Y, c4 - c0 = local Z
        loc = [(sx, sy, sz) for sz in (-1, 1) for sx, sy in ((1, -1), (-1, -1), (-1, 1), (1, 1))]
        pts = np.asarray(ctr) + (np.array(loc) * np.asarray(size) / 2) @ np.asarray(R).T
        return {"source_fields": {"bounding_box": [dict(zip("xyz", q)) for q in pts.tolist()]}}
    close = lambda a, b, tol=1e-6: all(abs(x - y) < tol for x, y in zip(a, b))
    pose = lambda p: ([p["furniture_position"][k] for k in "xyz"], p["furniture_rotation"],
                      [p["furniture_size"][k] for k in ("width", "length", "height")])
    # book lying flat, its local X (front) vertical: upright box with permuted axes, yaw = heading of local Y
    X, Y = np.array([0.0, 0, 1]), np.array([math.cos(0.5), math.sin(0.5), 0])
    R = np.stack([X, Y, np.cross(X, Y)], axis=1)
    pos, rot, size = pose(_pose(box([1, 2, 0.8], R, [0.3, 0.2, 0.05])))
    assert rot["x"] == 0.0 and abs(rot["z"] - math.degrees(0.5)) < 1e-6 and close(size, [0.2, 0.05, 0.3]), (rot, size)
    assert close(pos, [1, 2, 0.65])
    assert _extra({"furniture_category": "book", **_pose(box([1, 2, 0.8], R, [0.3, 0.2, 0.05]))}) == {"front_known": False}
    # level box turned upside down (180 deg about local X: corners 4-7 below 0-3): real height, bottom, front kept
    p = _pose(box([1, 2, 1.5], np.diag([1.0, -1, -1]), [0.8, 0.4, 0.6]))
    pos, rot, size = pose(p)
    assert rot["x"] == 0.0 and abs(rot["z"]) < 1e-6 and close(size, [0.8, 0.4, 0.6]) and close(pos, [1, 2, 1.2]), p
    assert _extra({"furniture_category": "wall design", **p}) == {}
    # truly tilted (30 deg about local X, yaw 90): yaw-aligned bounding box of the 8 corners, flagged via x/y None
    p = _pose(box([0, 0, 1], rot_zxy(math.pi / 2, math.radians(30), 0), [1.0, 0.4, 0.2]))
    pos, rot, size = pose(p)
    c, s_ = math.cos(math.radians(30)), math.sin(math.radians(30))
    assert rot["x"] is None and abs(rot["z"] - 90) < 1e-6 and abs(p["tilt_deg"] - 30) < 1e-6, (rot, p)
    assert close(size, [1.0, 0.4 * c + 0.2 * s_, 0.4 * s_ + 0.2 * c]) and close(pos, [0, 0, 1 - size[2] / 2])

    # 4 x 3 room at (10, 20), ring given clockwise; wall thickness 0.24 / 0.12
    ring = [[10, 20], [10, 23], [14, 23], [14, 20]]
    quad = lambda a, b, z0, z1: [[*a, z1], [*b, z1], [*b, z0], [*a, z0]]
    ops = [{"opening_type": "DOOR", "source_hole_index": 0, "thickness": 0.24,        # on the far face of the south wall
            "profile": quad((11.9, 19.76), (11.0, 19.76), 0.0, 2.1)},
           {"opening_type": "WINDOW", "source_hole_index": 1, "thickness": 0.12,      # on the east wall's centre line
            "profile": quad((14.06, 21.0), (14.06, 22.2), 0.9, 2.3)},
           {"opening_type": "DOOR", "source_hole_index": 2, "thickness": 0.12,        # the next room's wall, 1 m away
            "profile": quad((11.0, 19.0), (11.9, 19.0), 0.0, 2.1)},
           {"opening_type": "DOOR", "source_hole_index": 3, "thickness": 0.12,        # perpendicular to the west wall
            "profile": quad((9.5, 21.0), (9.9, 21.0), 0.0, 2.1)},
           {"opening_type": "HOLE", "source_hole_index": 4, "thickness": 0.24,        # passage: not emitted
            "profile": quad((11.0, 23.0), (12.0, 23.0), 0.0, 2.1)}]
    rep = [{**ops[0], "source_hole_index": 5},                                            # exact repeat of hole 0
           {**ops[0], "source_hole_index": 6, "profile": quad((11.0, 19.82), (11.9, 19.82), 0.0, 2.1)},  # other face
           {**ops[1], "source_hole_index": 7, "profile": quad((14.06, 21.0), (14.06, 22.2), 2.4, 2.7)}]  # window above
    op, n_rep = openings(ops + rep, ring, "s")
    assert n_rep == 2 and [f["furniture_instance_id"] for f in op] == ["s::hole_0", "s::hole_1", "s::hole_7"], op
    ir = convert_room({"room_id": "r", "room_boundary": ring, "furniture": op[:2]}, source=SOURCE,
                      uid="t", group="g", boundary_type="polygon", extra=_extra)
    d, w = ir["objects"]
    assert d["id"] == "s::hole_0" and d["category"] == "door" and d["structure"] and not d["tilted"], d
    assert close(d["size"], [0.9, OPENING_T, 2.1]) and close(d["pos"], [1.45, 0, 0]) and abs(d["yaw"]) < 1e-9, d
    assert w["category"] == "window" and w["structure"], w
    assert close(w["size"], [1.2, OPENING_T, 1.4]) and close(w["pos"], [4, 1.6, 0.9]) and abs(w["yaw"] - math.pi / 2) < 1e-9, w
    # the source's own 'window' box on the same wall line (0.13 m deep) replaces the opening box; a lamp does not
    own = {"id": "s::ins_1", "category": "window", "size": [1.2, 0.13, 1.4], "pos": [4, 1.6, 0.9], "yaw": math.pi / 2}
    lamp = {**own, "id": "s::ins_2", "category": "lamp"}
    assert drop_labelled([d, w, own, lamp]) == [d, own, lamp] and drop_labelled([d, w, lamp]) == [d, w, lamp]
    pane = {**own, "size": [0.02, 1.5, 1.6], "pos": [3.97, 1.5, 0.8], "yaw": 0.0}     # thin pane 3 cm off the line
    sill = {**own, "size": [1.2, 0.5, 0.4], "pos": [3.8, 1.6, 0.0]}                   # platform below the window
    assert drop_labelled([w, pane]) == [pane] and drop_labelled([w, sill]) == [w, sill]

    # wall ring overshooting a corner (crosses itself around a 2.4 x 6.5 cm lobe): repaired, area kept; a bow-tie is not
    spur = [[2.912, 0], [2.888, 0], [2.888, 1.678], [0, 1.678], [0, 0.065], [2.912, 0.065]]
    fixed, m = repair_ring(spur)
    assert not Polygon(spur).is_valid and Polygon(fixed).is_valid and m["boundary_repaired"] == "buffer(0)", m
    assert abs(Polygon(fixed).area - abs(signed_area(spur))) < REPAIR_TOL * abs(signed_area(spur))
    bow = [[0, 0], [1, 1], [1, 0], [0, 1]]
    assert repair_ring(bow) == (bow, {}) and repair_ring(ring) == (ring, {})
    print("interiorgs.py self-check ok")
