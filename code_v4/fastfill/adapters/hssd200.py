"""HSSD-200 (XXXpilar/hssd_clean rebuild of hssd-hab): 168 designer scenes, 2,351 annotated regions.

Source (README / dataset_manifest.json): meters, Y-up, right-handed, scene-global. objects.csv gives the tight
OBB of every rigid instance: obb_center (box center), obb_half_extents (already scaled) along the asset .glb
axes, obb_rotation_wxyz (world <- glb). regions.csv gives the official region polygon in XZ, floor/ceiling height
and the human region label. Converted here to Z-up by (x, y, z) -> (x, -z, y), z measured from the region floor.

Object frame: the glb axis that maps to world up is the asset's up. Wall-contact statistics (front test on
beds/toilets/cabinets/wardrobes, also per template) show two authoring frames: up=+Y with front=+Z, and the
same frame turned -90 deg about X (up=-Z, front=+Y). Other up axes (33 objects: 24 standing books, 5 stools, 2
upside-down carpets ...) have no known front: yaw along a horizontal glb axis, front_known False (build shows them as
built-in fixed boxes); objects whose no axis is vertical (94: books, boxes, pillows) are tilted. Their upright
envelope retains the center and the projected front of the nearest recognized authoring frame; an unrecognized
frame keeps a geometric heading with front_known=False.
Articulated instances (3,626 cabinets, wardrobes, fridges, ...) and 14 rigid rows of templates without a box have no box
and no region in the rebuild; they are assigned to regions by their translation and boxed from their model (URDF, else
asset_catalog / .glb bounds; see _rebuild, tagged box_src). 3,290 are the articulated copy of a boxed rigid row at the
same place (_twin: 3,062 of the same template, 93 a sibling template of the same category, 135 of another category) ->
dropped, counted in meta; 336 are kept. Only rows with no model stay incomplete, so build rejects their room: 1
articulated (render path '.ao_config.json.glb', no URDF) and the 13 rigid rows of templates 3759 / 3760 / 5804 (absent
from the source, outside every region). is_architectural instances (doors, windows, stairs ...) are kept as "structure"
objects (never placed; a floor staircase rejects the room in build); boxless ones are dropped. Category 'unknown*'
(lexicon id 0) or empty is kept as the label 'unknown*' / 'unknown' (was: incomplete, rejecting 125 rooms): the box is
real, build shows it as a generic fixed box or ignores it when raised.
Boxes lying wholly below the region floor belong to a lower storey without a region at that XY (the source assigns by
XY alone) -> dropped, counted in meta.
Openings boxed from metadata sit off their mesh in the rebuild (half a height too high, some shifted along the wall);
their centre is recomputed from the opening .glb bounds (see _center).
"""
import csv
import functools
import itertools
import json
import math
import os
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon

from fastfill.adapters.internscenes import tilted_bbox
from fastfill.adapters.unified import convert_room

SOURCE = "HSSD200"
T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])      # Y-up -> Z-up
FRONT = {(1, 1): (2, 1), (2, -1): (1, 1)}              # (up axis, sign) -> (front axis, sign) in glb frame


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


@functools.lru_cache(maxsize=None)
def _glb_bounds(path):
    """(lo, hi) of a .glb in its asset frame: every mesh primitive's POSITION min/max through the node transforms
    (corners of each primitive box, so exact without node rotation, as in all 841 opening .glbs). None if absent."""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        fh.seek(12)                                      # 12-byte header, then the JSON chunk (length, type)
        g = json.loads(fh.read(struct.unpack("<I4s", fh.read(8))[0]))
    pts = []

    def walk(i, M):
        nd = g["nodes"][i]
        L = np.eye(4)
        if "matrix" in nd:
            L = np.array(nd["matrix"], float).reshape(4, 4).T     # column-major
        else:
            x, y, z, w = nd.get("rotation", [0, 0, 0, 1])
            L[:3, :3] = _rot([w, x, y, z]) * nd.get("scale", [1, 1, 1])
            L[:3, 3] = nd.get("translation", [0, 0, 0])
        M = M @ L
        for p in g["meshes"][nd["mesh"]]["primitives"] if "mesh" in nd else []:
            a = g["accessors"][p["attributes"]["POSITION"]]
            pts.append(np.array(list(itertools.product(*zip(a["min"], a["max"])))) @ M[:3, :3].T + M[:3, 3])
        for c in nd.get("children", []):
            walk(c, M)

    for i in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(i, np.eye(4))
    pts = np.vstack(pts)
    return pts.min(0), pts.max(0)


def _center(r):
    """Box centre of a boxed row in the source (Y-up) frame.

    bbox_source 'metadata' rows (the 2,956 boxed openings: doors, windows, gates; no other category) were boxed by
    the rebuild from the opening metadata as base-at-origin assets (asset_catalog local_bbox y in [0, H], README
    convention 3), so obb_center = translation + R (s * (0, H/2, 0)). Their .glb meshes are not: most are centred on
    the origin (door 204-0: y in [-1.1, 1.1]), some are not centred at all (window 3759-0: y in [-0.6, 0.9]; door
    3762-0: x in [-0.51, 2.37]). The box centre is translation + R (s * (lo + hi) / 2), [lo, hi] the mesh bounds
    (r["mesh_bounds"], from _glb_bounds; s * (hi - lo) / 2 = obb_half_extents on all 2,956 rows within 2 mm).
    Moved: every row by ~H/2 down from obb_center; vs. plain translation, 586 rows by > 5 cm vertically (3759-0
    +0.15 m, 3760-x -0.10 m, 3756-x -0.32 m) and 3762-0 doors 0.93 m along the wall. Check against the stage walls
    (no stage vertex inside the box shrunk 2 cm in-plane, +-0.3 m deep): doors 98.3% / windows 97.5% clear, vs 83% /
    75% for translation and 3% / 7% for obb_center; door bottoms p10..p90 +0.02 m above the region floor (438 of 467
    within 5 cm, the rest in another storey's region). Without mesh_bounds (.glb missing) the translation is used,
    counted in the room meta.
    Mesh-sourced rows (all furniture, stairs, fences, the 26 mesh-boxed openings) keep obb_center: box already right."""
    if r.get("bbox_source") != "metadata" or r.get("box_src"):     # (rebuilt boxes: see _rebuild)
        return json.loads(r["obb_center"])
    t = np.array(json.loads(r["translation"]))
    if r.get("mesh_bounds") is None:
        return t
    lo, hi = r["mesh_bounds"]
    return t + _rot(json.loads(r["obb_rotation_wxyz"])) @ (np.array(json.loads(r["non_uniform_scale"])) * (lo + hi) / 2)


def _pose(o):
    """URDF <origin xyz rpy> (rpy about fixed x, y, z) -> 4x4; identity if absent."""
    M = np.eye(4)
    if o is not None:
        r, p, y = (float(v) for v in o.get("rpy", "0 0 0").split())
        M[:3, :3] = _rot([math.cos(y / 2), 0, 0, math.sin(y / 2)]) @ _rot([math.cos(p / 2), 0, math.sin(p / 2), 0]) \
            @ _rot([math.cos(r / 2), math.sin(r / 2), 0, 0])
        M[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return M


@functools.lru_cache(maxsize=None)
def _urdf_bounds(path):
    """(lo, hi) of an articulated model in its URDF root frame with every joint at its origin (closed: 3,609 of the
    3,626 instances have all joint positions 0, the rest < 2e-5): each link's visual .glb bounds through the joint
    chain. The links are Z-up exports and 1,508 of the 1,509 URDFs turn them Y-up in the root joint (rpy -90 deg about
    x), so the root frame is the scene's Y-up frame with the model's origin. None if the URDF or a mesh is missing, or
    the root joint does not turn link +Z to +Y (kitchenette0002: its 6 instances are yawed Y-up like their rigid copies,
    so that model would lie on its back)."""
    if not os.path.isfile(path):
        return None
    x = ET.parse(path).getroot()
    up = {j.find("child").get("link"): (j.find("parent").get("link"), _pose(j.find("origin"))) for j in x.iter("joint")}
    root = next(link.get("name") for link in x.iter("link") if link.get("name") not in up)
    if not all(np.allclose(M[:3, 2], [0, 1, 0], atol=1e-3) for parent, M in up.values() if parent == root):
        return None
    pose = lambda link: pose(up[link][0]) @ up[link][1] if link in up else np.eye(4)
    pts = []
    for link in x.iter("link"):
        for v in link.findall("visual"):
            if v.find("geometry/mesh") is None:
                continue
            b = _glb_bounds(os.path.join(os.path.dirname(path), v.find("geometry/mesh").get("filename")))
            if b is None:
                return None
            M = pose(link.get("name")) @ _pose(v.find("origin"))
            pts.append(np.array(list(itertools.product(*zip(*b)))) @ M[:3, :3].T + M[:3, 3])
    pts = np.vstack(pts)
    return pts.min(0), pts.max(0)


def _rebuild(r, lo, hi, src):
    """Box a boxless row in place from its model bounds [lo, hi] in the frame its rotation applies to: centre
    translation + R (s * (lo + hi) / 2), half extents |s * (hi - lo)| / 2, box_src = src.
    Articulated rows: the URDF root frame (_urdf_bounds). Their translation_origin says 'COM' but the translation is
    the root origin (the root link has no mass): on the 1,480 unit-scale rows of up=+Y templates with a same-template
    rigid copy within 10 cm, the rebuilt centre minus the copy's obb_center equals the two translations' difference
    within 1 cm for 99.0%, the world half extents agree within 1 cm for 99.4%. For templates authored up=-Z the render
    .glb (asset_catalog bounds) is not that frame: it is Z-up and centred, the URDF links are re-exported (Cabinet0004:
    .glb z +-0.72, URDF base link z 0..1.45), so catalog bounds would lay their 467 yawed articulated rows on their
    back; on 118 of them with an overlapping unit-scale rigid copy the URDF box has the copy's extents (p90 diff 0,
    catalog p50 0.18 m) and centre (horizontal p50 2 cm; vertically the source places them 0.1-0.3 m off each other).
    26 articulated rows are in the render .glb frame (asset_catalog): 20 carry the rigid copy's up=-Z rotation (root +Y
    not vertical) and 6 are kitchenette0002 (see _urdf_bounds); so are the rigid rows (asset_catalog, else the .glb).
    Kept rebuilt boxes pass the wall-contact front test like source boxes (IR floor cabinets, wardrobes, dressers,
    fridges, nightstands within 15 cm of a wall: back to the wall 136 of 141 vs 2,581 of 2,714); of the 326 in rooms,
    219 stand within 5 cm of the floor and 94 are above 0.5 m (wall units)."""
    s = np.array(json.loads(r["non_uniform_scale"]))
    c = np.array(json.loads(r["translation"])) + _rot(json.loads(r["obb_rotation_wxyz"])) @ (s * (lo + hi) / 2)
    r.update(obb_center=json.dumps(c.tolist()), obb_half_extents=json.dumps((abs(s * (hi - lo)) / 2).tolist()),
             box_src=src)


def _model(r, base, catalog):
    """(lo, hi, src) of a boxless row's model in the frame its rotation applies to (see _rebuild), or None: the URDF
    when the row's rotation keeps its root +Y up, else the asset_catalog bounds, else the render .glb."""
    if r["urdf_path"] and abs(_rot(json.loads(r["obb_rotation_wxyz"]))[1, 1]) > 0.999:
        b = _urdf_bounds(os.path.join(base, "meshes", r["urdf_path"]))
        if b:
            return (*b, "urdf")
    b = catalog.get(r["template_name"])
    if b:
        return (*b, "catalog")
    b = _glb_bounds(os.path.join(base, "meshes", r["render_asset"]))
    return b and (*b, "glb")


def _box(r):
    """(XZ footprint, y0, y1) of a boxed row's OBB in the source frame (footprint = hull of the 8 corners)."""
    p = np.array(list(itertools.product((-1, 1), repeat=3))) * json.loads(r["obb_half_extents"])
    p = p @ _rot(json.loads(r["obb_rotation_wxyz"])).T + _center(r)
    return MultiPoint(p[:, [0, 2]].tolist()).convex_hull, p[:, 1].min(), p[:, 1].max()


def _twin(a, b, rel):
    """True if boxed row b (from _box) is the piece of furniture that rebuilt box a enters again: they share over 30% of
    the smaller one's volume (as build's overlapping_furniture test: two solid pieces cannot) and, by how b's row
    relates to a's (rel, see _rel): 'model' (same template): always, the rigid row may be scaled down (8 rows at
    0.23-0.33 of the volume: Cullen Chest, Carla Wardrobe, ...); 'category' (same category): unless b is a small part
    inside a (under a third of its volume); 'other' (b not architectural, of a category the source ships articulated
    models of): only at similar size (volumes within 3x); None: never.
    The articulated copy differs from its rigid row in scale (articulated rows are unit scale), pivot, translation
    (same-size kitchen units 0.2-0.5 m apart: Oven 0.37 m) and often template (Raleigh Wide / Tall Dresser), then often
    in category too ('Kitchen tall cabinet' wardrobe vs '... with oven' cabinet, Gretna Base Cabinet Double
    kitchen_lower_cabinet vs Gretna Drawers cabinet, Melbourne vs Xena 3 Door Wardrobe). Of the 479 rebuilt boxes the
    same-category test keeps, 150 share over 30% with a similar-size box of another category (3.1% of the rigid rows of
    the articulated categories); 'other' takes 135 (rigid rows 2.5%). The 13 left share with a plant, lamp, TV, toilet,
    cooker hood, basket, toy bin or child's bed. In build output the placed rebuilt objects that share over 30% with a
    similar-size placed object of another category go from 135 of 360 to 9 of 227 (source objects 173 of 5,911). Also
    taken: a desk and two islands whose partner is a chair / stools / a wine cart tucked under them (seat and cart have
    articulated models: storage benches, trolleys)."""
    if rel is None:
        return False
    (fa, a0, a1), (fb, b0, b1) = a, b
    va, vb = fa.area * (a1 - a0), fb.area * (b1 - b0)
    return fa.intersection(fb).area * max(0.0, min(a1, b1) - max(a0, b0)) > 0.3 * min(va, vb) and \
        (rel == "model" or vb >= va / 3 and (rel == "category" or vb <= 3 * va))


def _rel(r, o, kinds):
    """How boxed row o relates to boxless row r, for _twin (kinds: the categories of articulated rows). Not the asset
    name: collections share it ('Champagne' bed and nightstand, 'Kitchen Furniture Collection' modules)."""
    if o["template_name"] == r["template_name"]:
        return "model"
    if o["category"] == r["category"]:
        return "category"
    return "other" if o["category"] in kinds and o["is_architectural"] != "True" else None


def _furniture(r, floor):
    """objects.csv row -> unified-schema furniture dict (Z-up, center z above the region floor)."""
    cat = r["category"] or "unknown"                 # lexicon id 0 ('unknown', 'unknown_wall') or none: no name
    f = {"furniture_category": cat, "furniture_instance_id": f"{r['scene_id']}:{r['instance_id']}",
         "furniture_position": None, "furniture_rotation": None, "furniture_size": None}
    if not r["obb_center"]:
        return f                                         # articulated / missing template: no box
    C = T @ _rot(json.loads(r["obb_rotation_wxyz"]))
    ext = 2 * np.array(json.loads(r["obb_half_extents"]))
    c = T @ np.array(_center(r))
    k = int(np.argmax(abs(C[2])))
    f["furniture_position"] = {"x": c[0], "y": c[1], "z": c[2] - floor}
    fr = FRONT.get((k, int(np.sign(C[2, k]))))
    if fr is None:                                       # unknown authoring frame: the box is known, its front not
        fr, f["front_known"] = (min({0, 1, 2} - {k}), 1), False
    if abs(C[2, k]) < 0.999:                             # no vertical axis: tilted
        a, s = fr
        order = [a, 3 - a - k, k]
        R = C[:, order] * np.array([s, 1, 1])
        size, yaw, _, _ = tilted_bbox(R, ext[order])
        return {**f, "furniture_rotation": {"x": None, "y": None, "z": yaw},
                "furniture_size": dict(zip(("width", "length", "height"), size)),
                "rot_wxyz_yup": json.loads(r["obb_rotation_wxyz"])}
    a, s = fr
    o = 3 - a - k
    d = s * C[:, a]
    f["furniture_rotation"] = {"x": 0, "y": 0, "z": math.degrees(math.atan2(d[1], d[0]))}
    f["furniture_size"] = {"width": float(ext[a]), "length": float(ext[o]), "height": float(ext[k])}
    return f


def load(root):
    base = os.path.join(root, "XXXpilar__hssd_clean")
    regions = {}
    for g in csv.DictReader(open(os.path.join(base, "regions.csv"))):
        g["poly"] = [[x, -z] for x, z in json.loads(g["poly_loop_xz"])]
        regions[(g["scene_id"], g["region_id"])] = g
    by_scene = defaultdict(list)
    for key, g in regions.items():
        by_scene[key[0]].append((key, Polygon(g["poly"]), float(g["floor_height"]), float(g["ceiling_height"])))

    bounds = {k["template_name"]: (np.array(json.loads(k["local_bbox_min"])), np.array(json.loads(k["local_bbox_max"])))
              for k in csv.DictReader(open(os.path.join(base, "asset_catalog.csv"))) if k["local_bbox_min"]}
    rows, boxed = list(csv.DictReader(open(os.path.join(base, "objects.csv")))), defaultdict(list)
    kinds = {r["category"] for r in rows if r["instance_kind"] == "articulated"}
    for r in rows:
        if r["bbox_source"] == "metadata" and r["obb_center"]:
            r["mesh_bounds"] = _glb_bounds(os.path.join(base, "meshes", r["render_asset"]))
        if r["obb_center"]:
            boxed[r["scene_id"]].append((_box(r), r))
    furn, n_arch, n_below, n_nob, n_twin = defaultdict(list), *(defaultdict(int) for _ in range(4))
    for r in rows:
        boxless, twin = not r["obb_center"], False
        if boxless and (m := _model(r, base, bounds)):
            _rebuild(r, *m)
            a = _box(r)
            twin = any(_twin(a, b, _rel(r, o, kinds)) for b, o in boxed[r["scene_id"]])
        if r["region_source"] in ("inside", "nearest"):
            keys = [(r["scene_id"], r["region_id"])]
            # the source assigns by XY only when one polygon matches: boxes wholly below this region's
            # floor stand on a lower storey that has no region there -> not in this room (aabb shifted like _center)
            if r["obb_center"] and json.loads(r["aabb_max"])[1] - (json.loads(r["obb_center"])[1] - _center(r)[1]) \
                    < float(regions[keys[0]]["floor_height"]) - 0.05:
                n_below[keys[0]] += 1
                continue
        elif boxless:                                    # no region in the source: locate by translation
            x, y, z = json.loads(r["translation"])
            keys = [k for k, p, fl, ce in by_scene[r["scene_id"]] if p.contains(Point(x, -z)) and fl - 0.3 <= y <= ce]
        else:
            continue                                     # outdoor props outside every region
        for key in keys:
            if twin:                                     # its rigid copy carries the box already
                n_twin[key] += 1
                continue
            n_nob[key] += "mesh_bounds" in r and r["mesh_bounds"] is None
            f = _furniture(r, float(regions[key]["floor_height"]))
            if r["is_architectural"] == "True":
                # kept as "structure" (never placed) so build sees floor stairs; without a box it cannot
                # be checked and would read as an incomplete object: dropped and counted, as before
                if f["furniture_rotation"] is None:
                    n_arch[key] += 1
                    continue
                f["structure"] = True
            f["box_src"] = r.get("box_src")
            furn[key].append(f)

    for key, g in regions.items():
        room = {"room_id": f"{key[0]}:{key[1]}", "room_type": g["room_type"], "room_boundary": g["poly"],
                "room_height": float(g["extrusion_height"]), "furniture": furn[key]}
        yield convert_room(room, source=SOURCE, uid=f"{SOURCE}:{key[0]}:{key[1]}", group=f"hssd:{key[0]}",
                           boundary_type="polygon", center_z=True,
                           extra=lambda f: {**({"rot_wxyz_yup": f["rot_wxyz_yup"]} if "rot_wxyz_yup" in f else {}),
                                            **({"structure": True} if f.get("structure") else {}),
                                            **({"box_src": f["box_src"]} if f.get("box_src") else {}),
                                            **({"front_known": False} if f.get("front_known") is False else {})},
                           meta={"region_name": g["region_name"], "floor_height": float(g["floor_height"]),
                                 "n_architectural_dropped": n_arch[key], "n_below_floor_dropped": n_below[key],
                                 "n_articulated_twin_dropped": n_twin[key],
                                 "n_opening_no_mesh_bounds": n_nob[key], "front_known": True})


if __name__ == "__main__":
    # glb frame turned -90 deg about X (up=-Z), yawed 90 deg in the scene: front must point to world +Y (Z-up)
    q_x = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]                # Rx(+90): glb -Z -> +Y up
    R = _rot([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]) @ _rot(q_x)  # then yaw 90 about +Y
    w = math.sqrt(1 + np.trace(R)) / 2
    q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    row = {"category": "toilet", "scene_id": "s", "instance_id": "0", "obb_center": "[1, 0.4, -2]",
           "obb_half_extents": "[0.2, 0.35, 0.4]", "obb_rotation_wxyz": json.dumps(q)}
    f = _furniture(row, 0.0)
    # canonical front +Z yawed 90 about +Y -> Habitat +X -> Z-up +X; sizes: front 0.7, side 0.4, up 0.8
    assert abs(f["furniture_rotation"]["z"]) < 1e-6, f
    assert np.allclose([f["furniture_size"][k] for k in ("width", "length", "height")], [0.7, 0.4, 0.8]), f
    assert np.allclose([f["furniture_position"][k] for k in "xyz"], [1, 2, 0.4]), f
    # metadata-boxed opening: obb_center is one half height too high, translation is the true centre
    door = {"category": "door", "scene_id": "s", "instance_id": "1", "bbox_source": "metadata",
            "obb_center": "[1, 2.2, -2]", "translation": "[1, 1.1, -2]", "obb_half_extents": "[0.5, 1.1, 0.1]",
            "obb_rotation_wxyz": "[1, 0, 0, 0]"}
    f = _furniture(door, 0.0)
    assert np.allclose([f["furniture_position"][k] for k in "xyz"], [1, 2, 1.1]) and f["furniture_size"]["height"] == 2.2, f
    # ... unless its mesh bounds are known: mesh x in [-0.5, 2.5] (centre +1.0), y in [-0.6, 1.6] (centre +0.5),
    # mirrored in x (s = -1) and yawed 90 deg about +Y: offset R (s * c) = Ry90 (-1, 0.5, 0) = (0, 0.5, 1) Y-up
    door.update(mesh_bounds=(np.array([-0.5, -0.6, -0.1]), np.array([2.5, 1.6, 0.1])), non_uniform_scale="[-1, 1, 1]",
                obb_half_extents="[1.5, 1.1, 0.1]", obb_rotation_wxyz=json.dumps([math.cos(math.pi / 4), 0,
                                                                                     math.sin(math.pi / 4), 0]))
    f = _furniture(door, 0.0)
    assert np.allclose([f["furniture_position"][k] for k in "xyz"], [1, 1, 1.6]), f   # (1, 1.6, -1) Y-up
    # _glb_bounds: minimal .glb, one node translated (+1, 0, 0) and scaled 2 in y over a unit-cube accessor
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    js = json.dumps({"scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "translation": [1, 0, 0], "scale": [1, 2, 1]}],
                     "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                     "accessors": [{"min": [-0.5, 0, -0.5], "max": [0.5, 1, 0.5]}]}).encode()
    with open(os.path.join(tmp, "body.glb"), "wb") as fh:
        fh.write(struct.pack("<4sII", b"glTF", 2, 20 + len(js)) + struct.pack("<I4s", len(js), b"JSON") + js)
    lo, hi = _glb_bounds(os.path.join(tmp, "body.glb"))
    assert np.allclose(lo, [0.5, 0, -0.5]) and np.allclose(hi, [1.5, 2, 0.5]), (lo, hi)
    # _urdf_bounds: root joint Rx(-90) turns the Z-up link frame Y-up, (x, y, z) -> (x, z, -y): base link x 0.5..1.5,
    # y -0.5..0.5, z -2..0; a child link on a joint 1 up the base's +Z = root +Y: y 0.5..1.5. Union y -0.5..1.5
    urdf = lambda rpy, mesh: f"""<robot><link name="root"/><joint name="r" type="fixed">
        <origin rpy="{rpy}" xyz="0 0 0"/><parent link="root"/><child link="base"/></joint><link name="base"><visual>
        <origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh}"/></geometry></visual><collision><geometry>
        <box size="1 1 1"/></geometry></collision></link><joint name="d" type="revolute"><origin xyz="0 0 1"/>
        <parent link="base"/><child link="door"/></joint><link name="door"><visual><geometry>
        <mesh filename="body.glb"/></geometry></visual></link></robot>"""
    for name, rpy, mesh in (("ok", -math.pi / 2, "body.glb"), ("flat", 0, "body.glb"), ("miss", -math.pi / 2, "x.glb")):
        with open(os.path.join(tmp, name + ".urdf"), "w") as fh:
            fh.write(urdf(f"{rpy} 0 0", mesh))
    lo, hi = _urdf_bounds(os.path.join(tmp, "ok.urdf"))
    assert np.allclose(lo, [0.5, -0.5, -2]) and np.allclose(hi, [1.5, 1.5, 0]), (lo, hi)
    assert _urdf_bounds(os.path.join(tmp, "flat.urdf")) is None and _urdf_bounds(os.path.join(tmp, "miss.urdf")) is None
    shutil.rmtree(tmp)
    # _rebuild: articulated cabinet, origin at its bottom centre, yawed 90 deg about +Y (front +Z -> +X), box from the
    # model bounds; _center keeps the rebuilt centre even for a 'metadata' row; _furniture then as for rigid rows
    y90 = json.dumps([math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
    cab = {"category": "", "scene_id": "s", "instance_id": "2", "bbox_source": "metadata", "obb_center": "",
           "translation": "[1, 0, -2]", "obb_rotation_wxyz": y90, "non_uniform_scale": "[1, 1, 1]"}
    _rebuild(cab, np.array([-0.3, 0, -0.2]), np.array([0.3, 0.9, 0.2]), "urdf")
    assert np.allclose(_center(cab), [1, 0.45, -2]) and cab["box_src"] == "urdf", cab
    f = _furniture(cab, 0.0)
    assert f["furniture_category"] == "unknown" and abs(f["furniture_rotation"]["z"]) < 1e-6, f   # '' -> 'unknown'
    assert np.allclose([f["furniture_size"][k] for k in ("width", "length", "height")], [0.4, 0.6, 0.9]), f
    assert np.allclose([f["furniture_position"][k] for k in "xyz"], [1, 2, 0.45]), f
    # unknown authoring frame (glb +X up): box kept, yaw along glb +Y (-> Y-up -X -> Z-up yaw 180), front_known False
    book = {**row, "category": "unknown_wall", "obb_rotation_wxyz": json.dumps([math.cos(math.pi / 4), 0, 0,
                                                                                  math.sin(math.pi / 4)])}
    f = _furniture(book, 0.0)
    assert f["front_known"] is False and f["furniture_category"] == "unknown_wall", f
    assert abs(f["furniture_rotation"]["z"] % 360 - 180) < 1e-6, f
    assert np.allclose([f["furniture_size"][k] for k in ("width", "length", "height")], [0.7, 0.8, 0.4]), f
    # _twin: a same-size unit 0.37 m off (38% shared) is the same piece; a neighbour touching it or a small part
    # inside it is not, unless it is the same model (scaled-down rigid row); across categories only at similar size
    box = lambda x, h: _box({"obb_center": json.dumps([x, h, 0]), "obb_half_extents": json.dumps([0.3, h, 0.3]),
                             "obb_rotation_wxyz": "[1, 0, 0, 0]"})
    small = _box({"obb_center": "[0, 0.1, 0]", "obb_half_extents": "[0.1, 0.1, 0.1]",
                  "obb_rotation_wxyz": "[1, 0, 0, 0]"})
    assert _twin(box(0, 0.45), box(0.37, 0.45), "category") and not _twin(box(0, 0.45), box(0.6, 0.45), "category")
    assert not _twin(box(0, 0.45), small, "category") and _twin(small, box(0, 0.45), "category")
    assert _twin(box(0, 0.45), small, "model") and _twin(box(0, 0.45), box(0.37, 0.45), "other")
    assert not _twin(box(0, 0.45), small, "other") and not _twin(small, box(0, 0.45), "other")
    assert not _twin(box(0, 0.45), box(0, 0.45), None)
    # _rel: same model > same category > another category with articulated models (not architectural) > none
    cab = {"template_name": "t1", "asset_name": "Gretna Base", "category": "kitchen_lower_cabinet"}
    rel = lambda **o: _rel(cab, {**cab, "template_name": "t2", "asset_name": "", "is_architectural": "False", **o},
                           {"cabinet", "kitchen_lower_cabinet"})
    assert rel(template_name="t1") == "model" and rel(asset_name="Gretna Base") == "category"
    assert rel(category="cabinet") == "other" and rel(category="cabinet", asset_name="Gretna Base") == "other"
    assert rel(category="plant") is None and rel(category="cabinet", is_architectural="True") is None
    print("hssd200.py self-check ok")
