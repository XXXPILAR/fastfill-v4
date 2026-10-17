"""Box renders, anchors, RoomGen hand-offs and a contact sheet for the VLM-judge pilot (render-and-judge design,
sections B "Rendering" and C "Implement first"). CPU only, deterministic, no network and no model calls; the saved
comparison runs are read only.

    python -m fastfill.v2.ops.vlm_judge render  --comparison-dir B --out DIR [--rows 111,114] [--methods gt,random]
    python -m fastfill.v2.ops.vlm_judge anchors --comparison-dir B --out DIR
    python -m fastfill.v2.ops.vlm_judge handoff --comparison-dir B --out DIR --roomgenbench-root .../RoomGenBench
    python -m fastfill.v2.ops.vlm_judge contact --out DIR

render: DIR/panels/<hash>.png, one 1536x1024 panel per (room, method): a 1024 px top-down plan (black walls, 0.5 m
grid, 1 m scale bar, boxes filled by RoomGen place and drawn in order of their top height, the front edge dark with an
arrow, "n category" labels, numbers only above 25 objects) and two 512 px views from opposite upper corners (numpy
z-buffer, near walls removed, flat shading, front faces darker). The image names no method; DIR/manifest.json maps
hash -> (row, method) with the programmatic checks, and keeps each room's numbered legend for the judge prompt.
anchors: DIR/anchors.jsonl. handoff: DIR/handoff/row<r>-<method>/{export,assembly} and DIR/handoff/results.json
(export_handoff, then ``fastfill.v2.roomgenbench --method layout_boxes --require-placement``). contact:
DIR/contact.html, panels by relative path with method names and checks. contact.html, handoff/ and manifest.json name
the methods: for the operator, never for the judge (the judge gets panels/<hash>.png and the legend only).

Conventions (fastfill.v2): target_size_local_m [sx, sy, sz] with local +X the front, bottom_center_m in metres, yaw_rad
counter-clockwise from world +X; footprints as validation.footprint; place (floor / on_object / wall / unknown) from
direct_layout.layout_to_roomgenbench, coloured as RoomGenBench assemble.PLACE_RGBA (grey for unknown).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from fastfill.v2.direct_layout import export_handoff, layout_to_roomgenbench
from fastfill.v2.geometry import wrap_yaw
from fastfill.v2.validation import validate_scene


REPO = Path(__file__).resolve().parents[3]
COMPARISON = REPO / "outputs/fastfill_v2/comparison-20261007b"
ROWS_FILE = "llm-harness-300/rows.jsonl"  # byte-identical to llm-prompt-300/rows.jsonl
SOURCES = {"ff-spread": "llmrows-main7-cell05-main-20261007b-e5/outcomes.jsonl",  # old step-6357 FastFill; the
           "ff-argmax": "llmrows-main7-cell05-main-20261007b-e5-argmax/outcomes.jsonl",  # decoded layout is raw_prediction
           "llm-prompt": "llm-prompt-300/predictions.jsonl", "llm-harness": "llm-harness-300/predictions.jsonl"}
METHODS = ("gt", *SOURCES)
ANCHORS = ("room-centre", "random", "shuffled", "yaw-flipped", "mirrored", "identical")
NAMES = {"gt": "Ground truth", "ff-spread": "FastFill spread (old step-6357)", "ff-argmax": "FastFill argmax (old step-6357)",
         "llm-prompt": "LLM prompt", "llm-harness": "LLM harness", **{a: f"Anchor: {a}" for a in ANCHORS}}
# Pilot (every method returned a layout; ground truth without boundary violations unless noted): bedroom 119
# (MansionWorld, 14 objects), bathroom 104 (InternScenes, 8, height unknown), kitchen 156 (SAGE-10k, 14), living room
# 59 (Holodeck, 12, height unknown; its ground truth overshoots the walls by centimetres), dining room 133 (3D-FRONT,
# 10, chairs tucked under the table), small meeting room 16 (MansionWorld, 14), bedroom 197 (Structured3D, 5) and
# gym 249 (SAGE-10k, 46 objects, ground truth clean).
PILOT_ROWS = (119, 104, 156, 59, 133, 16, 197, 249)
KEYS = ("id", "target_size_local_m", "bottom_center_m", "yaw_rad")
SALT = "fastfill-vlm-judge-v1"

DEFAULT_HEIGHT_M = 2.8
PLACE_RGB = {"floor": (77, 128, 204), "on_object": (90, 170, 100), "wall": (214, 120, 60), "unknown": (150, 150, 150)}
FLOOR_RGB, WALL_RGB = (168, 150, 126), (236, 233, 226)  # RoomGenBench assemble.FLOOR_RGBA / WALL_RGBA
FRONT = "#1a1a1a"
NUMBERS_ONLY_ABOVE = 25
PANEL_W, PANEL_H, PLAN_PX, VIEW_PX, DPI = 1536, 1024, 1024, 512, 100
PLAN_MARGIN_M = .9
VIEW_FOV_DEG, NEAR_M = 50., .05
LIGHT = np.array((.3, .5, .8)) / np.linalg.norm((.3, .5, .8))


def _jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load(comparison):
    """The 300 frozen rows and, per method, each row's layout (None when the method returned none)."""
    base = Path(comparison)
    rows = _jsonl(base / ROWS_FILE)
    layouts = {"gt": [{"schema_version": r["target"]["schema_version"],
                       "objects": [{k: o[k] for k in KEYS} for o in r["target"]["objects"]]} for r in rows]}
    for method, name in SOURCES.items():
        records = _jsonl(base / name)
        if method.startswith("ff-"):
            by_row = {r["row"]: r.get("raw_prediction") for r in records}
            records = [by_row.get(i) for i in range(len(rows))]
        else:
            records = [r.get("layout") for r in records]
        if len(records) != len(rows):
            raise ValueError(f"{name}: {len(records)} records for {len(rows)} rows")
        layouts[method] = [x if isinstance(x, dict) else None for x in records]
    return rows, layouts


def _room(row):
    room = row["condition"]["room"]
    polygon = np.asarray(room["floor_polygon_xy_m"], float)
    return polygon, polygon.min(0), polygon.max(0), room.get("floor_z_m") or 0., room.get("height_m") or DEFAULT_HEIGHT_M


def anchors(row, index, gt, seed=0):
    """Layouts derived from the ground truth (its sizes kept); the same (seed, row) gives the same layouts.

    room-centre: every bottom centre at the floor-level room centre. random: uniform XY whose rotated footprint fits
    the room's XY bounds, uniform yaw, ground-truth height above the floor. shuffled: the objects pass their bottom
    centres and yaws along one random cycle (nobody keeps its own). yaw-flipped: yaw + pi. mirrored: x -> xmin + xmax - x,
    yaw -> pi - yaw. identical: the ground truth itself.
    """
    _, lo, hi, floor, _ = _room(row)
    objs = gt["objects"]
    centre = [float(v) for v in (lo + hi) / 2]
    rng = np.random.default_rng([seed, index, ANCHORS.index("random")])
    random = []
    for o in objs:
        yaw = float(rng.uniform(-math.pi, math.pi))
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        sx, sy = o["target_size_local_m"][:2]
        half = (.5 * (c * sx + s * sy), .5 * (s * sx + c * sy))
        xy = [float(rng.uniform(a + h, b - h)) if b - a > 2 * h else float((a + b) / 2) for a, b, h in zip(lo, hi, half)]
        random.append({**o, "bottom_center_m": [*xy, o["bottom_center_m"][2]], "yaw_rad": yaw})
    order = np.random.default_rng([seed, index, ANCHORS.index("shuffled")]).permutation(len(objs))
    donor = {int(a): objs[int(b)] for a, b in zip(order, np.roll(order, -1))}
    layouts = {
        "room-centre": [{**o, "bottom_center_m": [*centre, floor]} for o in objs],
        "random": random,
        "shuffled": [{**o, "bottom_center_m": list(donor[i]["bottom_center_m"]), "yaw_rad": donor[i]["yaw_rad"]}
                     for i, o in enumerate(objs)],
        "yaw-flipped": [{**o, "yaw_rad": wrap_yaw(o["yaw_rad"] + math.pi)} for o in objs],
        "mirrored": [{**o, "bottom_center_m": [float(lo[0] + hi[0]) - o["bottom_center_m"][0], *o["bottom_center_m"][1:]],
                      "yaw_rad": wrap_yaw(math.pi - o["yaw_rad"])} for o in objs],
        "identical": deepcopy(objs)}
    return {name: {"schema_version": gt["schema_version"], "objects": objects} for name, objects in layouts.items()}


def method_layout(rows, layouts, index, method, seed):
    if method in layouts:
        return layouts[method][index]
    return anchors(rows[index], index, layouts["gt"][index], seed)[method]


def corners_xy(size, pos, yaw):
    """validation.footprint's corners: local (-sx/2,-sy/2), (sx/2,-sy/2), (sx/2,sy/2), (-sx/2,sy/2) rotated by yaw
    about the bottom centre; corners 1 -> 2 are the front edge (local +X)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy = size[0] / 2, size[1] / 2
    return np.array([(pos[0] + c * px - s * py, pos[1] + s * px + c * py)
                     for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])


def front_arrow(size, pos, yaw):
    """(tail, head): from the footprint centre to the middle of the front edge, i.e. along local +X."""
    tail = np.asarray(pos[:2], float)
    return tail, tail + size[0] / 2 * np.array((math.cos(yaw), math.sin(yaw)))


def by_height(its):
    """Drawing order: by top height, then request number."""
    return sorted(its, key=lambda it: (it["pos"][2] + it["size"][2], it["n"]))


def items(row, layout):
    """Request-ordered boxes (n from 1) with their RoomGen place, and the RoomGen scene; invalid layouts raise."""
    scene = layout_to_roomgenbench(row["condition"], layout)
    place = {o["id"]: o["place"] for o in scene["objects"]}
    by_id = {o["id"]: o for o in layout["objects"]}
    return [{"n": n, "category": r["category"], "place": place[r["id"]], "size": by_id[r["id"]]["target_size_local_m"],
             "pos": by_id[r["id"]]["bottom_center_m"], "yaw": by_id[r["id"]]["yaw_rad"]}
            for n, r in enumerate(row["condition"]["objects"], 1)], scene


def facing(row, layout):
    """The wall-facing statistic of the investigation's facing.py: floor objects (bottom within 5 cm of the floor)
    whose axis-aligned footprint bound lies within 0.15 m of exactly one wall of the room's XY bounds and whose front
    is within ~45 deg of that wall's normal; returns (front into the room, scored)."""
    _, lo, hi, floor, _ = _room(row)
    into = total = 0
    for o in layout["objects"]:
        (sx, sy, _), (x, y, z), t = o["target_size_local_m"], o["bottom_center_m"], o["yaw_rad"]
        if z - floor > .05:
            continue
        c, s = abs(math.cos(t)), abs(math.sin(t))
        hx, hy = .5 * (c * sx + s * sy), .5 * (s * sx + c * sy)
        gaps = {(1, 0): x - hx - lo[0], (-1, 0): hi[0] - x - hx, (0, 1): y - hy - lo[1], (0, -1): hi[1] - y - hy}
        near = [n for n, g in gaps.items() if abs(g) <= .15 + 1e-9]  # rounded coordinates land exactly on 0.15
        if len(near) != 1:
            continue
        dot = math.cos(t) * near[0][0] + math.sin(t) * near[0][1]
        if abs(dot) < .7:
            continue
        total, into = total + 1, into + (dot > 0)
    return into, total


def checks(row, layout, scene):
    report = validate_scene(row["condition"], layout["objects"])
    violations = Counter(c["code"] for c in report["checks"] if c["status"] == "violation")
    places = Counter(o["place"] for o in scene["objects"])
    return {"clean": not violations, "violations": dict(sorted(violations.items())), "places": dict(sorted(places.items())),
            "require_placement": "unknown" not in places, "wall_backed_floor_facing_in": list(facing(row, layout))}


def plan_extent(row):
    """(x0, x1, y0, y1) of the square plan: the room's XY bounds plus PLAN_MARGIN_M around the longer side."""
    _, lo, hi, _, _ = _room(row)
    c, half = (lo + hi) / 2, float((hi - lo).max()) / 2 + PLAN_MARGIN_M
    return c[0] - half, c[0] + half, c[1] - half, c[1] + half


def cameras(row):
    """[(eye, target)] at two opposite upper corners, aimed at the room centre at 0.4 * height (height 2.8 m when
    unknown): along the line through the design's eye (1 m outside the corner on the room diagonal, at room height
    + 1.2 m), backed off as RoomGenBench render.py does until the room's bounding sphere fills the field of view, so
    the whole floor is in frame (from the design's eye itself, up to 32% of the floor fell outside the view)."""
    _, lo, hi, floor, height = _room(row)
    centre = (lo + hi) / 2
    target = np.array([*centre, floor + .4 * height])
    dist = .5 * math.hypot(*(hi - lo), height) / math.sin(math.radians(VIEW_FOV_DEG) / 2)
    out = []
    for corner in (lo, hi):
        d = (corner - centre) / np.linalg.norm(corner - centre)
        ray = np.array([*(corner + d), floor + height + 1.2]) - target
        out.append((target + ray / np.linalg.norm(ray) * dist, target))
    return out


def project(points, eye, target, size=VIEW_PX):
    """World points -> (pixel x, pixel y, depth along the view axis): pinhole, square image, VIEW_FOV_DEG vertical."""
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, (0., 0., 1.))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    rel = np.asarray(points, float) - eye
    depth = rel @ f
    k = size / 2 / math.tan(math.radians(VIEW_FOV_DEG) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return size / 2 + k * (rel @ r) / depth, size / 2 - k * (rel @ u) / depth, depth


def _shade(rgb, normal, factor=1.):
    return np.asarray(rgb) / 255 * (.62 + .38 * max(0., float(np.dot(normal, LIGHT)))) * factor


def view_primitives(row, its, eye):
    """[(convex polygon (k, 3), rgb 0-1)]: floor, the walls whose inside faces the eye, then the boxes (top + 4 sides,
    in top-height order; the front face darker)."""
    polygon, _, _, floor, height = _room(row)
    prims = [(np.c_[polygon, np.full(len(polygon), floor)], _shade(FLOOR_RGB, (0, 0, 1)))]
    ccw = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, 0))) > 0
    for a, b in zip(polygon, np.roll(polygon, -1, 0)):
        dx, dy = b - a
        outward = np.array((dy, -dx) if ccw else (-dy, dx)) / math.hypot(dx, dy)
        if np.dot(outward, eye[:2] - a) > 0:  # the eye is outside this wall: removed, as RoomGenBench render.py
            continue
        prims.append((np.array([[*a, floor], [*b, floor], [*b, floor + height], [*a, floor + height]]),
                      _shade(WALL_RGB, (*-outward, 0))))
    for it in by_height(its):
        xy = corners_xy(it["size"], it["pos"], it["yaw"])
        z0, z1 = it["pos"][2], it["pos"][2] + it["size"][2]
        v = np.r_[np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z1)]]
        c, s = math.cos(it["yaw"]), math.sin(it["yaw"])
        rgb = PLACE_RGB[it["place"]]
        prims += [(v[[4, 5, 6, 7]], _shade(rgb, (0, 0, 1))), (v[[1, 2, 6, 5]], _shade(rgb, (c, s, 0), .55)),
                  (v[[3, 0, 4, 7]], _shade(rgb, (-c, -s, 0))), (v[[2, 3, 7, 6]], _shade(rgb, (-s, c, 0))),
                  (v[[0, 1, 5, 4]], _shade(rgb, (s, -c, 0)))]
    return prims


def _fill(img, inv, ids, x, y, iz, rgb, ident):
    """One triangle into the buffers; inv holds 1/depth (exactly linear in screen space), larger is nearer."""
    h, w = inv.shape
    x0, x1 = max(int(math.floor(x.min())), 0), min(int(math.ceil(x.max())), w)
    y0, y1 = max(int(math.floor(y.min())), 0), min(int(math.ceil(y.max())), h)
    area = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0])
    if x0 >= x1 or y0 >= y1 or abs(area) < 1e-9:
        return
    X, Y = np.meshgrid(np.arange(x0, x1) + .5, np.arange(y0, y1) + .5)
    w0 = ((x[2] - x[1]) * (Y - y[1]) - (y[2] - y[1]) * (X - x[1])) / area
    w1 = ((x[0] - x[2]) * (Y - y[2]) - (y[0] - y[2]) * (X - x[2])) / area
    w2 = 1 - w0 - w1
    depth = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]
    window = np.s_[y0:y1, x0:x1]
    # at equal depth (coplanar faces, e.g. two overlapping tops of one height) the later face wins, as in the plan,
    # instead of rounding noise picking pixel by pixel
    win = (w0 >= 0) & (w1 >= 0) & (w2 >= 0) & (depth >= inv[window] * (1 - 1e-9))
    inv[window][win], img[window][win], ids[window][win] = depth[win], rgb, ident


def rasterize(prims, eye, target, size=VIEW_PX):
    """z-buffer of convex polygons (fan triangulated) on white, 1 px dark lines where the visible face changes;
    returns (uint8 image, face id buffer: the index into prims, -1 for background)."""
    img, inv, ids = np.ones((size, size, 3)), np.zeros((size, size)), np.full((size, size), -1)
    for ident, (poly, rgb) in enumerate(prims):
        x, y, d = project(poly, eye, target, size)
        if (d < NEAR_M).any():  # ponytail: drops a polygon crossing the near plane instead of clipping it
            continue
        for t in range(1, len(poly) - 1):
            tri = [0, t, t + 1]
            _fill(img, inv, ids, x[tri], y[tri], 1 / d[tri], rgb, ident)
    edge = np.zeros(ids.shape, bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[1:] |= ids[1:] != ids[:-1]
    img[edge] *= .3
    return (img * 255).round().astype(np.uint8), ids


def view(row, its, eye, target):
    """(image, [(n, x, y)]): one corner view and the boxes whose top-face centre shows their own top face."""
    prims = view_primitives(row, its, eye)
    img, ids = rasterize(prims, eye, target)
    first = len(prims) - 5 * len(its)  # after the floor and walls, each box adds its top then its four sides
    top = {it["n"]: first + 5 * k for k, it in enumerate(by_height(its))}
    tops = [[*it["pos"][:2], it["pos"][2] + it["size"][2]] for it in its]
    return img, [(it["n"], x, y) for it, x, y, d in zip(its, *project(tops, eye, target))
                 if d > NEAR_M and 0 <= x < VIEW_PX and 0 <= y < VIEW_PX and ids[int(y), int(x)] == top[it["n"]]]


def draw_plan(ax, row, its):
    from matplotlib import patheffects
    from matplotlib.patches import Polygon
    polygon, lo, hi, _, _ = _room(row)
    x0, x1, y0, y1 = plan_extent(row)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ppm = PLAN_PX / (x1 - x0)
    for g in np.arange(math.ceil(lo[0] / .5) * .5, hi[0], .5):
        ax.plot([g, g], [lo[1], hi[1]], color="#dddddd", lw=.6, zorder=1)
    for g in np.arange(math.ceil(lo[1] / .5) * .5, hi[1], .5):
        ax.plot([lo[0], hi[0]], [g, g], color="#dddddd", lw=.6, zorder=1)
    ax.add_patch(Polygon(polygon, closed=True, fill=False, ec="black", lw=3, zorder=2))
    for z, it in enumerate(by_height(its), start=3):
        corners = corners_xy(it["size"], it["pos"], it["yaw"])
        ax.add_patch(Polygon(corners, closed=True, fc=np.asarray(PLACE_RGB[it["place"]]) / 255, ec="#303030", lw=.8, zorder=z))
        ax.plot(*corners[1:3].T, color=FRONT, lw=2.5, solid_capstyle="butt", zorder=z)
        tail, head = front_arrow(it["size"], it["pos"], it["yaw"])
        ax.annotate("", head, tail, zorder=z, arrowprops={"arrowstyle": "-|>", "color": FRONT, "lw": 1.2, "shrinkA": 0,
                    "shrinkB": 0, "mutation_scale": float(np.clip(.25 * it["size"][0] * ppm, 4, 12))})
    renderer, pad = ax.figure.canvas.get_renderer(), 2 / ppm  # pad: the labels' white halo

    def extent(artist):  # (centre x, centre y, half width, half height) in metres, the text as rendered plus pad
        b = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        return (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, b.width / 2 + pad, b.height / 2 + pad

    cx, yb = (lo[0] + hi[0]) / 2, lo[1] - PLAN_MARGIN_M / 2
    ax.plot([cx - .5, cx + .5], [yb, yb], color="black", lw=4, solid_capstyle="butt")
    placed = [(cx, yb, .5 + pad, 3 / ppm), extent(ax.text(cx, yb - .06, "1 m", ha="center", va="top", fontsize=10))]
    centre, half = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
    for k, (eye, _) in enumerate(cameras(row), 1):  # 0.45 m outside the room corner the view looks from
        x, y = centre + (eye[:2] - centre) / np.linalg.norm(eye[:2] - centre) * (half + .45)
        placed.append(extent(ax.text(x, y, f"V{k}", ha="center", va="center", fontsize=11, weight="bold", color="#555555")))
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    numbers_only = len(its) > NUMBERS_ONLY_ABOVE
    size = 7.5 if numbers_only else 8.5
    grid = [(i, j) for i in range(-8, 9) for j in range(-30, 31)]
    for it in by_height(its):  # at its box centre, else the nearest free slot (whole label widths / lines away)
        x, y = it["pos"][:2]
        label = ax.text(x, y, str(it["n"]) if numbers_only else f'{it["n"]} {it["category"]}', fontsize=size,
                        ha="center", va="center", zorder=10_000, path_effects=halo)
        _, _, w, h = extent(label)
        for i, j in sorted(grid, key=lambda o: math.hypot(o[0] * w, o[1] * h)):
            sx, sy = x + 2 * i * w, y + 2 * j * h
            if (x0 + w <= sx <= x1 - w and y0 + h <= sy <= y1 - h
                    and not any(abs(sx - px) < w + pw and abs(sy - py) < h + ph for px, py, pw, ph in placed)):
                break
        else:  # ponytail: no free slot within 8 widths / 30 lines: overlap at the box
            sx, sy = x, y
        label.set_position((sx, sy))
        placed.append((sx, sy, w, h))


def render_panel(path, row, its):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import patheffects
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(PANEL_W / DPI, PANEL_H / DPI), dpi=DPI)
    draw_plan(fig.add_axes((0, 0, PLAN_PX / PANEL_W, 1)), row, its)
    halo = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    for k, (eye, target) in enumerate(cameras(row)):
        ax = fig.add_axes((PLAN_PX / PANEL_W, .5 - .5 * k, VIEW_PX / PANEL_W, VIEW_PX / PANEL_H))
        img, labels = view(row, its, eye, target)
        ax.imshow(img, interpolation="nearest", extent=(0, VIEW_PX, VIEW_PX, 0))
        ax.set_xlim(0, VIEW_PX)
        ax.set_ylim(VIEW_PX, 0)
        ax.axis("off")
        for n, x, y in labels:
            ax.text(x, y, str(n), fontsize=7, ha="center", va="center", path_effects=halo)
        ax.text(8, 8, f"V{k + 1}", ha="left", va="top", fontsize=11, weight="bold", color="#555555")
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def panel_name(row_index, method):
    return hashlib.sha256(f"{SALT}|{row_index}|{method}".encode()).hexdigest()[:16]


def cmd_render(args):
    rows, layouts = load(args.comparison_dir)
    out = Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "fastfill.vlm-judge-render.v1", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "comparison_dir": str(Path(args.comparison_dir).resolve()), "seed": args.seed, "rows": {}, "panels": {}, "missing": []}
    for r in args.rows:
        row = rows[r]
        polygon, lo, hi, _, _ = _room(row)
        manifest["rows"][str(r)] = {
            "room_type": row["condition"]["room"].get("room_type"), "source": row["provenance"]["source"],
            "size_m": [float(v) for v in (*(hi - lo), _room(row)[4])], "height_assumed": row["condition"]["room"].get("height_m") is None,
            "legend": [{"n": n, "id": o["id"], "category": o["category"], "description": o["description"][:80]}
                       for n, o in enumerate(row["condition"]["objects"], 1)]}
        for method in args.methods:
            layout = method_layout(rows, layouts, r, method, args.seed)
            try:
                if layout is None:
                    raise ValueError("no layout")
                its, scene = items(row, layout)
            except ValueError as error:
                manifest["missing"].append({"row": r, "method": method, "reason": str(error)})
                continue
            name = panel_name(r, method)
            render_panel(out / "panels" / f"{name}.png", row, its)
            manifest["panels"][name] = {"row": r, "method": method, "file": f"panels/{name}.png", "checks": checks(row, layout, scene)}
            print(f"row {r} {method} -> {name}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def cmd_anchors(args):
    rows, layouts = load(args.comparison_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "anchors.jsonl").open("w") as stream:
        for r in args.rows:
            for name, layout in anchors(rows[r], r, layouts["gt"][r], args.seed).items():
                stream.write(json.dumps({"row": r, "anchor": name, "seed": args.seed, "layout": layout}) + "\n")


def cmd_handoff(args):
    rows, layouts = load(args.comparison_dir)
    root = Path(args.out) / "handoff"
    root.mkdir(parents=True, exist_ok=False)
    results = []
    for r in args.rows:
        for method in args.methods:
            entry, layout = {"row": r, "method": method}, layouts[method][r]
            if layout is None:
                results.append({**entry, "pass": False, "stage": "no_layout"})
                continue
            directory = root / f"row{r:03d}-{method}"
            try:
                export_handoff(directory / "export", rows[r]["condition"], layout)
            except ValueError as error:
                results.append({**entry, "pass": False, "stage": "export", "message": str(error)})
                continue
            command = [sys.executable, "-m", "fastfill.v2.roomgenbench", "--handoff", str(directory / "export"),
                       "--output-dir", str(directory / "assembly"), "--method", "layout_boxes", "--require-placement",
                       "--roomgenbench-root", str(Path(args.roomgenbench_root).resolve())]
            done = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
            lines = (done.stdout if done.returncode == 0 else done.stderr).strip().splitlines()
            results.append({**entry, "pass": done.returncode == 0, "stage": "assembly", "returncode": done.returncode,
                            "message": lines[-1] if lines else "", "directory": str(directory)})
            print(json.dumps(results[-1]), flush=True)
    (root / "results.json").write_text(json.dumps(results, indent=1) + "\n")


def cmd_contact(args):
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text())
    handoff_file = out / "handoff" / "results.json"
    handoff = {(e["row"], e["method"]): e for e in json.loads(handoff_file.read_text())} if handoff_file.is_file() else {}
    order = METHODS + ANCHORS
    sections = []
    for r, info in manifest["rows"].items():
        r = int(r)
        panels = sorted((p | {"hash": h} for h, p in manifest["panels"].items() if p["row"] == r), key=lambda p: order.index(p["method"]))
        cards = []
        for p in panels:
            c = p["checks"]
            into, total = c["wall_backed_floor_facing_in"]
            lines = [f"validate_scene: {'clean' if c['clean'] else ', '.join(f'{k} {v}' for k, v in c['violations'].items())}",
                     "places: " + ", ".join(f"{k} {v}" for k, v in c["places"].items()),
                     f"require-placement: {'pass' if c['require_placement'] else 'FAIL'}",
                     f"wall-backed floor objects facing in: {into}/{total}"]
            if (r, p["method"]) in handoff:
                e = handoff[(r, p["method"])]
                lines.append("RoomGen layout_boxes --require-placement assembly: " + (
                    "pass" if e["pass"] else f"FAIL ({html.escape(e.get('message', e['stage']))})"))
            cards.append(f'<div class="card"><h3>{html.escape(NAMES[p["method"]])} <small>{p["hash"]}</small></h3>'
                         f'<a href="{p["file"]}"><img src="{p["file"]}" loading="lazy"></a><ul>'
                         + "".join(f"<li>{line}</li>" for line in lines) + "</ul></div>")
        cards += [f'<div class="card missing"><h3>{html.escape(NAMES[m["method"]])}</h3><p>not rendered: {html.escape(m["reason"])}</p></div>'
                  for m in manifest["missing"] if m["row"] == r]
        legend = "".join(f"<li>{html.escape(o['category'])}: {html.escape(o['description'])}</li>" for o in info["legend"])
        sx, sy, sz = info["size_m"]
        sections.append(f'<section><h2>Row {r}: {html.escape(str(info["room_type"]))} ({html.escape(info["source"])}), '
                        f'{sx:.2f} x {sy:.2f} x {sz:.2f} m{" (height assumed)" if info["height_assumed"] else ""}, '
                        f'{len(info["legend"])} objects</h2><details><summary>legend</summary><ol>{legend}</ol></details>'
                        f'<div class="grid">{"".join(cards)}</div></section>')
    page = ("<!doctype html><html><head><meta charset='utf-8'><title>VLM judge pilot panels</title><style>"
            "body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:12px}"
            ".card{background:#fff;border:1px solid #ddd;padding:8px}.card img{width:100%;height:auto}"
            ".card h3{margin:0 0 6px;font-size:15px}.card small{color:#999;font-weight:normal}.card ul{margin:6px 0 0;padding-left:18px}"
            ".missing{color:#a33}section{margin-bottom:32px}</style></head><body><h1>VLM judge pilot: rendered layouts</h1>"
            "<p>Colours: blue floor, green on an object, orange wall, grey unknown place. Dark edge and arrow: front (local +X). "
            "V1/V2: the corner views. Operator sheet: method names are shown here only, never to the judge.</p>"
            + "".join(sections) + "</body></html>\n")
    (out / "contact.html").write_text(page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("render", "anchors", "handoff", "contact"))
    parser.add_argument("--out", required=True, help="output directory (panels/, manifest.json, anchors.jsonl, handoff/, contact.html)")
    parser.add_argument("--comparison-dir", default=str(COMPARISON))
    parser.add_argument("--rows", type=lambda s: [int(x) for x in s.split(",")], default=list(PILOT_ROWS))
    parser.add_argument("--methods", type=lambda s: s.split(","), help="default: every method and anchor (handoff: methods only)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roomgenbench-root", default=str(REPO / "RoomGenBench"))
    args = parser.parse_args(argv)
    allowed = METHODS if args.command == "handoff" else METHODS + ANCHORS
    args.methods = args.methods or list(allowed)
    if set(args.methods) - set(allowed):
        parser.error(f"--methods must be among {allowed}")
    {"render": cmd_render, "anchors": cmd_anchors, "handoff": cmd_handoff, "contact": cmd_contact}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
