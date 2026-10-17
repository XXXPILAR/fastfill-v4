"""SAGE-10k (agentic generation, TRELLIS assets): 9,987 single-room scenes + 13 multi-room scenes (40 rooms).

Aggregate json_single.json / json_multi_split.json (aggregate_scenes.py): meters, Z-up, room_boundary = ordered wall
centreline endpoints (all rooms are 4-corner rectangles), position = bottom center (floor objects z = 0, equals the
cm footprint center of placement_constraints), rotation = Euler deg applied as Rz Ry Rx (kits/tex_utils_local.py),
dimensions = local X/Y/Z extents. Front = local +Y (front test), so front_offset_deg = 90.
Support: the aggregate drops place_id; it is read from the per-scene layout_*.json when present
(floor | wall | <instance id>), otherwise anchors are left to fastfill.anchors.
Doors: the aggregate drops walls, so doors come from the per-scene layout (same frame): centre = wall start +
position_on_wall * (end - start), bottom = wall base z, extent width x wall thickness x height (kits/tex_utils_local.py
create_door_mesh). Emitted as structure boxes. Layout `windows` are empty in the export; windows exist only as
furniture objects, which are kept as they are.
Descriptions: every layout object has one ("description"); passed through as desc when the layout is id-aligned.
Tilt: floor/wall objects are exactly upright; objects resting on objects carry x/y tilt from the physics check.
Tilt <= TILT_DEG is treated as settling noise (zeroed); an axis permutation within TILT_DEG (1,237 objects turned
90 deg onto a side) is an exact upright box. A larger tilt becomes the yaw-aligned upright bounding box of the 8
corners (`upright_box`), flagged tilted, with tilt_deg and the source rotation rot_src kept. The rotation is about the
object origin = bottom centre of its local box (SAGE server code, EmbodiedGen/gentask_baseline/sage/server:
objects/object_generation.py centres x/y and sets min z = 0; the physics check returns that origin's transform,
isaac_sim_mcp_extension/extension.py, scipy 'xyz' = Rz Ry Rx).
Support (`rest`): yet the stored pose keeps the origin on the support whatever the tilt, so the rotated box sinks in:
over 11,855 tilted (>= 5 deg) children whose parent top is confirmed by an untilted sibling, origin - top has median
-0.3..+0.4 cm in every lift bin up to 12 cm (corr 0.07), and a wine bottle lying on a hutch has its axis on the hutch
top. The box is kept whole (the source's extents) and stood on its support: its bottom is raised to the place_id
parent's top, or to the origin (the stored contact height) when the parent top is higher (a shelf or recess inside the
parent's box); "z_lift" records the rise. Build's 5 cm top-surface test then sees the source's contact height, not a
corner that sank 4-9 cm into the desk. Boxes turned exactly onto a side rest on their lowest face (90% of 107 on
confirmed tops), so min(parent top, origin) leaves them as they are.
"""
import collections
import json
import math
import os

import numpy as np

from fastfill.adapters.internscenes import tilted_bbox, upright
from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "SAGE-10k"
TILT_DEG = 2.0


def tilt_deg(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, math.cos(math.radians(rot["x"])) * math.cos(math.radians(rot["y"]))))))


def rot_zyx(r):
    """Euler deg {x, y, z} -> world-from-local R = Rz Ry Rx."""
    (cx, sx), (cy, sy), (cz, sz) = [(math.cos(math.radians(r[a])), math.sin(math.radians(r[a]))) for a in "xyz"]
    return np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ \
        np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])


def upright_box(f, cut=None):
    """Tilted furniture (Euler deg Rz Ry Rx about the bottom centre of its local box, size = local X/Y/Z extents,
    front = local +Y; SAGE and SceneSmith) -> the same dict as an upright box, front passed as local X to the
    internscenes rules (yaw = front heading, or local X heading when the front is the near-vertical axis):
    a local axis within TILT_DEG of vertical -> the exact box with permuted sizes (`upright`), rotation {0, 0, yaw - 90},
    not tilted; otherwise the yaw-aligned bounding box of the 8 corners (as `tilted_bbox`) minus the part below `cut`
    (the height below which the source says the object has nothing: its support), rotation {x: None, y: None,
    z: yaw - 90} so that convert_room(front_offset_deg=90) gives yaw back and flags it tilted. Position = bottom centre
    of the new box; "_tilt" = provenance (rot_src, and tilt_deg for tilted boxes)."""
    r, p, s = f.get("furniture_rotation"), f.get("furniture_position"), f.get("furniture_size")
    if not (r and p and s):
        return f
    R, S, o = rot_zyx(r)[:, [1, 0, 2]], [s["length"], s["width"], s["height"]], np.array([p["x"], p["y"], p["z"]])
    up = upright(R, S, TILT_DEG)
    if up:
        (l, w, h), yaw, _ = up
        c = o + R[:, 2] * S[2] / 2                                               # box centre
        return {**f, "furniture_position": {"x": float(c[0]), "y": float(c[1]), "z": float(c[2]) - h / 2},
                "furniture_rotation": {"x": 0, "y": 0, "z": yaw - 90},
                "furniture_size": {"width": w, "length": l, "height": h}, "_tilt": {"rot_src": dict(r)}}
    _, yaw, _, tilt = tilted_bbox(R, S)
    C = np.array([[x, y, z] for x in (-S[0] / 2, S[0] / 2) for y in (-S[1] / 2, S[1] / 2) for z in (0, S[2])]) @ R.T
    z0 = cut - o[2] if cut is not None and C[:, 2].min() < cut - o[2] < C[:, 2].max() else C[:, 2].min()
    edges = [(i, i | b) for i in range(8) for b in (1, 2, 4) if not i & b]
    P = [c for c in C if c[2] >= z0] + [C[i] + (C[j] - C[i]) * (z0 - C[i, 2]) / (C[j, 2] - C[i, 2])
                                        for i, j in edges if (C[i, 2] - z0) * (C[j, 2] - z0) < 0]
    D = np.array([[math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), 0],
                  [-math.sin(math.radians(yaw)), math.cos(math.radians(yaw)), 0], [0, 0, 1]])        # u, v, z rows
    lo, hi = (np.array(P) @ D.T).min(0), (np.array(P) @ D.T).max(0)
    b = o + D.T @ [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]              # bottom centre
    return {**f, "furniture_position": {"x": float(b[0]), "y": float(b[1]), "z": float(b[2])},
            "furniture_rotation": {"x": None, "y": None, "z": yaw - 90},
            "furniture_size": {"width": float(hi[1] - lo[1]), "length": float(hi[0] - lo[0]),
                               "height": float(hi[2] - lo[2])},
            "_tilt": {"tilt_deg": tilt, "rot_src": dict(r)}}


def rest(f, parent):
    """upright_box(f) stood on its support (module doc): its bottom raised to min(place_id parent's top, origin z), never
    lowered. The box as it is when the parent is missing or not upright (its box top is no surface)."""
    g = upright_box(f)
    if parent is None or tilt_deg(parent["furniture_rotation"]) > TILT_DEG:
        return g
    s = min(parent["furniture_position"]["z"] + parent["furniture_size"]["height"], f["furniture_position"]["z"])
    b = g["furniture_position"]["z"]
    if b >= s:
        return g
    return {**g, "furniture_position": {**g["furniture_position"], "z": s}, "_tilt": {**g["_tilt"], "z_lift": s - b}}


def doors(lroom):
    """Layout doors as unified-schema furniture. convert_room(front_offset_deg=90) swaps width/length and adds 90 deg,
    so width=thickness / length=door width / rotation=wall-90 come out as size [door width, thickness, height] with
    yaw = wall direction (long side along the wall)."""
    walls = {w["id"]: w for w in lroom.get("walls") or []}
    out = []
    for d in lroom.get("doors") or []:
        w = walls.get(d.get("wall_id"))
        if w is None:
            continue
        s, e, t = w["start_point"], w["end_point"], d["position_on_wall"]
        out.append({"furniture_category": "door", "furniture_instance_id": d["id"], "_ir": {"structure": True},
                    "furniture_position": {"x": s["x"] + (e["x"] - s["x"]) * t, "y": s["y"] + (e["y"] - s["y"]) * t,
                                           "z": s["z"]},
                    "furniture_rotation": {"x": 0, "y": 0,
                                           "z": math.degrees(math.atan2(e["y"] - s["y"], e["x"] - s["x"])) - 90},
                    "furniture_size": {"width": w.get("thickness") or 0.10, "length": d["width"], "height": d["height"]}})
    return out


def load(root):
    base = os.path.join(root, "liantian__3D_Room_Scene_Collections/SAGE10k")
    sdir = os.path.join(base, "sage10k/scenes")
    layouts = {d.split("_", 2)[-1]: os.path.join(sdir, d, d.split("_", 2)[-1] + ".json")
               for d in (os.listdir(sdir) if os.path.isdir(sdir) else [])}
    for fn in ("json_single.json", "json_multi_split.json"):
        for rec in iter_records(os.path.join(base, fn)):
            sid, k = rec["scene_id"], rec["room_index"]
            fids = [f["furniture_instance_id"] for f in rec["furniture"]]
            count = collections.Counter(fids)
            place, lroom = None, {}
            if sid in layouts and os.path.exists(layouts[sid]):
                with open(layouts[sid]) as fh:
                    lroom = json.load(fh)["rooms"][k]
                if [o["id"] for o in lroom["objects"]] == fids:
                    place = [o.get("place_id") for o in lroom["objects"]]
            n_noise = 0
            furn, seen = [], collections.Counter()
            for j, f in enumerate(rec["furniture"]):
                p = place[j] if place else None
                if p in ("floor", "wall"):
                    sup = {"anchor": p}
                elif count.get(p) == 1:
                    sup = {"anchor": "object", "parent": p}
                else:
                    sup = {}               # no evidence, dangling or repeated parent id: inferred downstream
                if place is not None and lroom["objects"][j].get("description"):
                    sup["desc"] = lroom["objects"][j]["description"]
                f = {**f, "_ir": sup}
                seen[fids[j]] += 1
                if seen[fids[j]] > 1:      # raw ids repeat for copied small objects (663 rooms): keep ids unique
                    f["furniture_instance_id"] = f"{fids[j]}#{seen[fids[j]]}"
                r = f.get("furniture_rotation")
                if r and (r["x"] or r["y"]):
                    if tilt_deg(r) <= TILT_DEG:
                        f["furniture_rotation"] = {"x": 0, "y": 0, "z": r["z"]}
                        n_noise += 1
                    else:
                        f = rest(f, rec["furniture"][fids.index(p)] if sup.get("parent") else None)
                furn.append(f)
            furn += doors(lroom)

            yield convert_room(
                {**rec, "furniture": furn, "room_boundary": [[q["x"], q["y"]] for q in rec.get("room_boundary") or []]},
                source=SOURCE, uid=f"SAGE-10k::{sid}::{k}", group=f"sage:{sid}",
                subset="multi" if fn == "json_multi_split.json" else "single", boundary_type="polygon",
                center_z=False, front_offset_deg=90, extra=lambda f: {**f["_ir"], **f.get("_tilt", {})},
                meta={"scene_id": sid, "room_index": k, "front_known": True, "tilt_threshold_deg": TILT_DEG,
                      "n_tilt_noise_zeroed": n_noise, "anchor_source": "place_id" if place is not None else "inferred",
                      "building_style": rec.get("building_style")})


if __name__ == "__main__":
    # 4 x 3 room at (10, 20); door on the bottom wall (+X direction), door on the right wall (+Y direction)
    walls = [{"id": f"w{i}", "start_point": {"x": a[0], "y": a[1], "z": 0.0}, "end_point": {"x": b[0], "y": b[1], "z": 0.0},
              "thickness": 0.1} for i, (a, b) in enumerate([((10, 20), (14, 20)), ((14, 20), (14, 23))])]
    lroom = {"walls": walls, "doors": [{"id": "d0", "wall_id": "w0", "position_on_wall": 0.5, "width": 0.9, "height": 2.1},
                                       {"id": "d1", "wall_id": "w1", "position_on_wall": 0.25, "width": 0.8, "height": 2.0},
                                       {"id": "dx", "wall_id": "missing", "position_on_wall": 0.5, "width": 1, "height": 2}]}
    ir = convert_room({"room_boundary": [[10, 20], [14, 20], [14, 23], [10, 23]], "furniture": doors(lroom)},
                      source=SOURCE, uid="t", group="g", boundary_type="polygon", center_z=False, front_offset_deg=90,
                      extra=lambda f: f["_ir"])
    a, b = ir["objects"]
    close = lambda u, v: all(abs(x - y) < 1e-6 for x, y in zip(u, v))
    assert a["category"] == "door" and a["structure"] and not a["tilted"], a
    assert close(a["pos"], [2, 0, 0]) and close(a["size"], [0.9, 0.1, 2.1]) and abs(math.sin(a["yaw"])) < 1e-9, a
    assert close(b["pos"], [4, 0.75, 0]) and close(b["size"], [0.8, 0.1, 2.0]) and abs(b["yaw"] - math.pi / 2) < 1e-9, b

    # tilted box -> upright bounding box. Size [1.0, 0.2, 0.1] (front = local +Y), origin (1, 1, 0.5), 30 deg about X:
    # IR size [along front, width, height] = [0.2 c + 0.1 s, 1.0, 0.2 s + 0.1 c], yaw 90, bottom centre (1, 0.975, 0.45)
    def ir_obj(f, cut=None):
        return convert_room({"room_boundary": [[0, 0], [9, 0], [9, 9], [0, 9]], "furniture": [upright_box(f, cut)]},
                            source=SOURCE, uid="t", group="g", boundary_type="polygon", center_z=False,
                            front_offset_deg=90, extra=lambda f: {**f["_ir"], **f.get("_tilt", {})})["objects"][0]

    def rec(r, S, p=(1, 1, 0.5)):
        return {"furniture_category": "book", "furniture_position": dict(zip("xyz", p)), "furniture_rotation": r,
                "furniture_size": dict(zip(("width", "length", "height"), S)), "_ir": {"desc": "a book"}}
    s30, c30 = 0.5, math.cos(math.radians(30))
    x30 = rec({"x": 30, "y": 0, "z": 0}, [1.0, 0.2, 0.1])
    o = ir_obj(x30)
    assert o["tilted"] and close(o["size"], [0.2 * c30 + 0.1 * s30, 1.0, 0.2 * s30 + 0.1 * c30]), o
    assert abs(o["yaw"] - math.pi / 2) < 1e-9 and close(o["pos"], [1, 0.975, 0.45]) and o["desc"] == "a book", o
    assert abs(o["tilt_deg"] - 30) < 1e-9 and o["rot_src"] == {"x": 30, "y": 0, "z": 0}, o
    # resting on a support at the origin (0.5) or 2 cm below it: the corners under it are cut, footprint unchanged
    for cut in (0.5, 0.48):
        o = ir_obj(x30, cut)
        assert close(o["size"], [0.2 * c30 + 0.1 * s30, 1.0, 0.1 * s30 + 0.1 * c30 + 0.5 - cut]), o
        assert close(o["pos"], [1, 0.975, cut]) and o["tilted"], o
    assert close(ir_obj(x30, 0.3)["pos"], [1, 0.975, 0.45]) and close(ir_obj(x30, 0.9)["pos"], [1, 0.975, 0.45])  # not through it
    # standing on the support: raised to a parent top at/below the origin, to the origin under a higher top (also when
    # rolled onto its side), never lowered, not raised under a tilted or missing parent
    par = lambda t, r=None: {"furniture_position": {"x": 1, "y": 1, "z": t - 0.4}, "furniture_size": {"height": 0.4},
                             "furniture_rotation": r or {"x": 0, "y": 0, "z": 10}}
    rolled, side = rec({"x": 70, "y": 0, "z": 0}, [1.0, 0.2, 0.1]), rec({"x": 90, "y": 0, "z": 0}, [1.0, 0.2, 0.1])
    for f, pa, z in ((x30, par(0.48), 0.48), (x30, par(0.53), 0.5), (x30, par(0.3), 0.45), (x30, None, 0.45),
                     (x30, par(0.48, {"x": 20, "y": 0, "z": 0}), 0.45), (rolled, par(0.53), 0.5),
                     (rolled, par(0.48), 0.48), (side, par(0.4), 0.4)):      # exact side box on its lowest face: as is
        g, h = rest(f, pa), upright_box(f)
        assert g["furniture_size"] == h["furniture_size"] and g["furniture_position"]["x"] == h["furniture_position"]["x"], g
        assert abs(g["furniture_position"]["z"] - z) < 1e-9, (f, pa, g)
        assert abs(g["_tilt"].get("z_lift", 0) - (z - h["furniture_position"]["z"])) < 1e-9, g
    # 90 deg about X (front vertical): exact upright box, sizes [1.0, 0.1, 0.2], yaw 0, centre (1, 0.95, 0.5), not tilted
    o = ir_obj(rec({"x": 90, "y": 0, "z": 0}, [1.0, 0.2, 0.1]))
    assert not o["tilted"] and "tilt_deg" not in o and o["rot_src"]["x"] == 90, o
    assert close(o["size"], [1.0, 0.1, 0.2]) and close(o["pos"], [1, 0.95, 0.4]) and abs(math.sin(o["yaw"])) < 1e-9, o
    # any rotation and cut: the IR box contains every point of the source box above the cut, is tight to it (points
    # sampled on the 12 edges, where the clipped box's vertices lie), its bottom is the cut, and its +X is the front's
    # heading unless the front (local Y) is the near-vertical axis
    rng = np.random.default_rng(0)
    edges = [(i, i | b) for i in range(8) for b in (1, 2, 4) if not i & b]
    t = np.linspace(0, 1, 401)[:, None]
    for k in range(400):
        r, S = dict(zip("xyz", rng.uniform(-180, 180, 3))), rng.uniform(0.02, 1.0, 3)
        R = rot_zyx(r)
        W = np.array([[x, y, z] for x in (-S[0] / 2, S[0] / 2) for y in (-S[1] / 2, S[1] / 2) for z in (0, S[2])]) @ R.T
        cut = 0.7 + rng.uniform(W[:, 2].min(), W[:, 2].max()) if k % 2 else None
        o = ir_obj(rec(r, S, (4, 4, 0.7)), cut)
        if not o["tilted"]:
            continue                                  # an axis within TILT_DEG of vertical: exact box, checked above
        E = np.vstack([W[i] + t * (W[j] - W[i]) for i, j in edges]) + [4, 4, 0.7]
        E = E[E[:, 2] >= (cut if cut is not None else -1.0)]
        cy, sy = math.cos(o["yaw"]), math.sin(o["yaw"])
        L = (E - o["pos"]) @ np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]]).T
        lo, hi = np.array([-o["size"][0] / 2, -o["size"][1] / 2, 0]), np.array([o["size"][0] / 2, o["size"][1] / 2, o["size"][2]])
        assert (L.min(0) > lo - 1e-9).all() and (L.max(0) < hi + 1e-9).all(), (r, S, cut, o)
        assert np.allclose(L.min(0), lo, atol=3e-3) and np.allclose(L.max(0), hi, atol=3e-3), (r, S, cut, o)
        assert cut is None or abs(o["pos"][2] - cut) < 1e-9, (r, S, cut, o)
        if np.argmax(np.abs(R[2])) != 1:
            assert abs((math.atan2(R[1, 1], R[0, 1]) - o["yaw"] + math.pi) % (2 * math.pi) - math.pi) < 1e-9, (r, o)
    print("sage10k.py self-check ok")
