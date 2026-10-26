"""OptiScene's three automatic evaluations (NeurIPS 2025, supplementary D.2 and Table 3), re-implemented: their
repository (github.com/PolySummit/OptiScene) has training and inference code only, no evaluation code.

  OOR   object overlap rate, their formula (supplementary D.2) per scene: 2D boxes centred at each object's (x, y)
        with its width and depth, the sum of the pairwise intersection areas over the sum of the box areas;
        averaged over scenes. As written, the boxes are axis-aligned (no rotation) and every object counts.
        ``--oor-rotated`` rotates the boxes by the yaw; ``--oor-floor-only`` keeps floor-standing objects only
        (their scenes hold furniture only; objects on furniture overlap their support by construction).
  FID   "following DiffuScene": top-down renders of generated and real layouts of the same rooms, standard FID
        (pytorch-fid's Inception-v3, pool3) and KID on the same features. They render textured assets in Isaac Sim;
        we have boxes only, so the renders are semantic (floor, fixed objects, objects coloured by category), the
        ATISS / DiffuScene / DeBaRA convention. Values are comparable within this table, not with their paper.
  GPT   their GPT-4o prompt (supplementary Table 3) verbatim, scoring functionality, layout and aesthetics 0-10 plus
        an overall score, from the layout renders and the "user preferences". Like their figures (a front and a top
        view), two renders go with each request: a top view and an oblique front view of the boxes, coloured and
        labelled by category. Our rows have no text prompt, so the preferences are written from the request: room
        type and size, the objects with their descriptions, the input's constraints. ``--legend`` adds one line
        explaining the diagram (not in their prompt; off by default).

    python experiments/optiscene_eval.py --oor --set test300
    python experiments/optiscene_eval.py --gpt --set 22 --repeats 1 --budget-usd 1   # one room, every method
    python experiments/optiscene_eval.py --gpt --set 20 --dry-run          # writes prompts and images only
    python experiments/optiscene_eval.py --fid --set test300                # downloads pytorch-fid's weights once
"""
import argparse, base64, hashlib, io, json, math, shutil, statistics, sys, time, urllib.error, urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))
from workflow import contract  # noqa: E402,F401  (puts fastfill on sys.path)
del contract
from shapely.geometry import Polygon  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402
from fastfill.v2.ops import layout_quality as lq  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402
from big_table import ROOT, sources  # noqa: E402

EXP = HERE / "experiments"
OUT = EXP / "optiscene_eval"  # workflow2's scores; the earlier methods' scores stay in Fastfill workflow/experiments
JUDGE = "gpt-4o"  # the paper's judge; --judge picks another
# USD per 1M input / cached input / output tokens; a judge not listed is costed at gpt-5.6-sol's price (an estimate)
JUDGE_PRICES = {"gpt-4o": (2.5, 1.25, 10.0), "gpt-5.6-sol": (4.0, .4, 20.0)}

PROMPT = """You are an expert in interior design and human-centric spatial planning. Your task is to evaluate \
the quality of the following room layout renders based on how well they match the user's design \
preferences, which are provided below (in triple backquotes).
Please assign a numerical score from **0 to 10** (0 = completely inconsistent, 10 = perfectly \
aligned) considering the following three aspects:
1. Functionality and Activity-based Alignment
- Does the layout support natural and efficient use of the space for daily activities (e.g., sleeping, \
working, relaxing, walking)?
- Are key object groupings (e.g., desk and chair, bed and nightstand) placed functionally and \
accessibly?
- Is there sufficient circulation space to ensure human accessibility?
2. Layout and Furniture Placement
- Are the furniture pieces arranged in a logical, practical way within the room boundaries?
- Are objects positioned with respect to common design principles (e.g., not blocking windows \
or doors, aligned to walls where appropriate)?
- Are there overlaps or unnatural collisions between objects?
3. Aesthetic Coherence
- Is the overall layout visually balanced and spacious?
- Does the arrangement exhibit good proportions, symmetry or asymmetry, and grouping where \
needed?
- Is the furniture distribution harmonious and pleasing to the eye, according to the user's stated \
aesthetic preferences?{legend}
User Preferences:
```{preferences}```
After considering all the above, return your evaluation in the following JSON format:
{{
```{{
"functionality_score": X,
"layout_score": Y,
"aesthetics_score": Z,
"overall_score": M,
"comments": "Brief explanation of your judgment."
}}```
}}
Scoring Guidelines:
- Scores should be integers from 0 to 10.
- The overall score can be the average or holistic assessment across the three criteria.
- Include a brief justification of your scores in the "comments" field (1-3 sentences).
Note: The goal is to measure how well the generated layout adheres to functional, spatial, and \
aesthetic expectations given the user's input. Be fair and critical."""


# I-Design (Celen et al., 2024), gpt_v_as_evaluator.py at github.com/atcelen/IDesign (main, 7bc891c): the prompt and the
# example JSON verbatim; they send two Blender renders and grade three times per scene (mean and std per aspect)
IDESIGN_EXAMPLE = """
{
  "realism_and_3d_geometric_consistency": {
    "grade": 8,
    "comment": "The renders appear to have appropriate 3D geometry and lighting that is fairly consistent with real-world expectations. The proportions and perspective look realistic."
  },
  "functionality_and_activity_based_alignment": {
    "grade": 7,
    "comment": "The room includes a workspace, sleeping area, and living area as per the user preference. The L-shaped couch facing the bed partially meets the requirement for watching TV comfortably. However, there does not appear to be a TV depicted in the render, so it's not entirely clear if the functionality for TV watching is fully supported."
  },
  "layout_and_furniture": {
    "grade": 7,
    "comment": "The room has a bed that’s not centered and with space at the foot, and a large desk with a chair. However, it's unclear if the height of the bed meets the user's preference, and the layout does not clearly show the full-length mirror in relation to the wardrobe, so its placement in accordance to user preferences is uncertain."
  },
  "color_scheme_and_material_choices": {
    "grade": 9,
    "comment": "The room adheres to a light color scheme with blue and white tones as preferred by the user, without a nautical feel. The bed and other furniture choices are aligned with the color scheme specified."
  },
  "overall_aesthetic_and_atmosphere": {
    "grade": 8,
    "comment": "The room's general aesthetic is bright, clean, and relatively minimalistic, which could align with the user's preference for a light color scheme and a modern look. The chandelier is present as opposed to bright, hospital-like lighting."
  }
}
"""
IDESIGN_ASPECTS = ("realism_and_3d_geometric_consistency", "functionality_and_activity_based_alignment",
                   "layout_and_furniture", "color_scheme_and_material_choices", "overall_aesthetic_and_atmosphere")
IDESIGN_PROMPT = f"""
          Give a grade from 1 to 10 or unknown to the following room renders based on how well they correspond together to the user preference (in triple backquotes) in the following aspects: 
          - Realism and 3D Geometric Consistency
          - Functionality and Activity-based Alignment
          - Layout and furniture  
          - Color Scheme and Material Choices
          - Overall Aesthetic and Atmosphere         
          User Preference:
          ```{{preferences}}```
          Return the results in the following JSON format:
          ```json
          {IDESIGN_EXAMPLE.replace("{", "{{").replace("}", "}}")}
          ```
          """


def parse_idesign(text):
    import re
    m = re.findall(r"```json(.*?)```", text, re.DOTALL)
    body = m[0].strip() if m else text[text.find("{"):text.rfind("}") + 1]
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        return None
    out = {}
    for k in IDESIGN_ASPECTS:
        g = (d.get(k) or {}).get("grade") if isinstance(d.get(k), dict) else None
        try:
            out[k] = float(g)
        except (TypeError, ValueError):
            out[k] = None  # "unknown"
    return out

LEGEND = ("The renders are a top view and an oblique front view of the room. In the top view the black outline is the floor, red shaded areas are the "
          "door clearance zones, blue segments are windows, dark grey boxes are fixed structures, each furniture "
          "piece is a coloured box labelled with its category, an arrow shows the direction its front faces, and "
          "dashed boxes are objects standing on furniture or hung on a wall; a 1 m scale bar is drawn. The front view shows the same boxes with their heights.")


# ---------------------------------------------------------------------------------------------------- OOR
def oor(row, layout, rotated=False, floor_only=False):
    """Supplementary D.2: sum over pairs of Area(B_i ∩ B_j) / sum of Area(B_i), B_i the box centred at the object's
    (x, y) with its width and depth (axis-aligned as written; ``rotated``: turned by the yaw)."""
    bs = [b for b in lq.boxes(row, layout) if not floor_only or lq.obstacle(b)]
    if rotated:
        polys = [b["poly"] for b in bs]
    else:
        from shapely.geometry import box as abox
        polys = [abox(b["centre"][0] - b["size"][0] / 2, b["centre"][1] - b["size"][1] / 2,
                      b["centre"][0] + b["size"][0] / 2, b["centre"][1] + b["size"][1] / 2) for b in bs]
    polys = [p for p in polys if p.is_valid and p.area > 0]
    total = sum(p.area for p in polys)
    if not total:
        return None
    tree, inter = STRtree(polys), 0.
    for i, p in enumerate(polys):
        for j in tree.query(p):
            if j > i:
                inter += p.intersection(polys[j]).area
    return inter / total


# ---------------------------------------------------------------------------------------------------- renders
def colour(cat):
    h = hashlib.md5(cat.encode()).digest()
    return tuple(60 + b % 196 for b in h[:3])


def semantic_render(row, layout, size=256):
    """ATISS / DiffuScene-style top-down semantic image (floor, fixed objects, category colours), room fit to the
    image (the same transform for every method of a room)."""
    from PIL import Image, ImageDraw
    floor = Polygon(row["condition"]["room"]["floor_polygon_xy_m"])
    x0, y0, x1, y1 = floor.bounds
    scale = (size - 8) / max(x1 - x0, y1 - y0, 1e-6)
    ox, oy = (size - scale * (x1 - x0)) / 2, (size - scale * (y1 - y0)) / 2
    tf = lambda pts: [(ox + (x - x0) * scale, size - (oy + (y - y0) * scale)) for x, y in pts]  # noqa: E731
    img = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon(tf(floor.exterior.coords), fill=(128, 128, 128))
    for p in lq.fixed_obstacles(row):
        d.polygon(tf(p.exterior.coords), fill=(40, 40, 40))
    for b in sorted(lq.boxes(row, layout), key=lambda b: (b["bottom"], b["size"][2])):
        d.polygon(tf(b["poly"].exterior.coords), fill=colour(b["cat"]))
    return img


def _png(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def top_render(row, layout):
    """Top view (PNG bytes): floor outline, door zones, windows, fixed objects, every object as a box coloured and
    labelled by category, an arrow for its front, dashed when it stands on furniture or hangs on a wall."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPoly
    from fastfill.v2.cell_costs import door_zones
    room = row["condition"]["room"]
    floor = Polygon(room["floor_polygon_xy_m"])
    x0, y0, x1, y1 = floor.bounds
    w = 7.
    fig, ax = plt.subplots(figsize=(w, max(2.5, min(w * 1.6, w * (y1 - y0) / max(x1 - x0, 1e-6)))))
    ax.add_patch(MPoly(list(floor.exterior.coords), closed=True, fc="#f2f2f2", ec="k", lw=2))
    for z in door_zones(room, .8):
        ax.add_patch(MPoly(z, closed=True, fc="#ffbbbb", ec="r", lw=.8, alpha=.7))
    coords = room["floor_polygon_xy_m"]
    for o in room.get("openings", []):
        if o.get("type") == "window" and o.get("wall_index") is not None and o["wall_index"] < len(coords):
            (ax_, ay_), (bx_, by_) = coords[o["wall_index"]], coords[(o["wall_index"] + 1) % len(coords)]
            L = math.hypot(bx_ - ax_, by_ - ay_) or 1.
            t, half = o.get("position_on_wall", .5), min(o.get("width_m", 1.), L) / 2
            ux, uy = (bx_ - ax_) / L, (by_ - ay_) / L
            cx, cy = ax_ + (bx_ - ax_) * t, ay_ + (by_ - ay_) * t
            ax.plot([cx - ux * half, cx + ux * half], [cy - uy * half, cy + uy * half], color="#1f6fd1", lw=5)
    for p in lq.fixed_obstacles(row):
        ax.add_patch(MPoly(list(p.exterior.coords), closed=True, fc="#555555", ec="k"))
    rgb = lambda c: tuple(v / 255 for v in colour(c))  # noqa: E731
    for b in sorted(lq.boxes(row, layout), key=lambda b: (b["bottom"], b["size"][2])):
        raised = b["bottom"] > lq.FLOOR_M
        ax.add_patch(MPoly(list(b["poly"].exterior.coords), closed=True, fc=rgb(b["cat"]), ec="k", lw=.8,
                           alpha=.55 if raised else .9, ls="--" if raised else "-"))
        cx, cy = b["centre"]
        L = min(b["size"][0], b["size"][1]) * .45 + .05
        ax.arrow(cx, cy, b["front"][0] * L, b["front"][1] * L, head_width=.08, color="k", lw=.7,
                 length_includes_head=True)
        ax.text(cx, cy, b["cat"][:14], fontsize=7, ha="center", va="center")
    ax.plot([x0, x0 + 1], [y0 - .25, y0 - .25], color="k", lw=3)
    ax.text(x0 + .5, y0 - .45, "1 m", ha="center", fontsize=8)
    ax.set_xlim(x0 - .4, x1 + .4)
    ax.set_ylim(y0 - .7, y1 + .4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Top view", fontsize=10)
    return _png(fig)


def front_render(row, layout):
    """Oblique front view (PNG bytes): the floor and every object as a solid box with its real height, coloured and
    labelled by category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    room = row["condition"]["room"]
    floor = Polygon(room["floor_polygon_xy_m"])
    x0, y0, x1, y1 = floor.bounds
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(projection="3d")
    ax.add_collection3d(Poly3DCollection([[(x, y, 0) for x, y in floor.exterior.coords]], facecolor="#e6e6e6",
                                         edgecolor="k", linewidths=1.2))
    rgb = lambda c: tuple(v / 255 for v in colour(c))  # noqa: E731
    for b in sorted(lq.boxes(row, layout), key=lambda b: (b["bottom"], b["size"][2])):
        z0, z1 = b["bottom"], b["bottom"] + b["size"][2]
        ring = list(b["poly"].exterior.coords)[:4]
        faces = [[(x, y, z0) for x, y in ring], [(x, y, z1) for x, y in ring]]
        faces += [[(*ring[k], z0), (*ring[(k + 1) % 4], z0), (*ring[(k + 1) % 4], z1), (*ring[k], z1)] for k in range(4)]
        ax.add_collection3d(Poly3DCollection(faces, facecolor=rgb(b["cat"]), edgecolor="k", linewidths=.4, alpha=.9))
        ax.text(b["centre"][0], b["centre"][1], z1 + .05, b["cat"][:12], fontsize=6, ha="center")
    span = max(x1 - x0, y1 - y0, 1.)
    ax.set_xlim(x0, x0 + span)
    ax.set_ylim(y0, y0 + span)
    ax.set_zlim(0, max(float(room.get("height_m") or 2.8), 1.))
    ax.set_box_aspect((1, 1, max(float(room.get("height_m") or 2.8), 1.) / span))
    ax.view_init(elev=35, azim=-60)
    ax.set_axis_off()
    ax.set_title("Front view", fontsize=10)
    return _png(fig)


def preferences(row):
    c = row["condition"]
    room = c["room"]
    floor = Polygon(room["floor_polygon_xy_m"])
    x0, y0, x1, y1 = floor.bounds
    counts = {}
    for o in c["objects"]:
        key = (o.get("category") or "object", (o.get("description") or "").strip())
        counts[key] = counts.get(key, 0) + 1
    items = [f"{n} x {cat}" + (f" ({desc})" if desc and desc.lower() != cat.lower() else "") for (cat, desc), n in counts.items()]
    names = {o["id"]: o.get("category") for o in c["objects"]}
    cons = []
    for k in c.get("constraints", []):
        ids = [k.get("object_id"), k.get("target_id"), *(k.get("target_ids") or [])]
        cons.append(k["type"] + " " + ", ".join(f"{names.get(i, i)}" for i in ids if i))
    doors = sum(1 for o in room.get("openings", []) if o.get("type") == "door")
    windows = sum(1 for o in room.get("openings", []) if o.get("type") == "window")
    lines = [f"Room type: {room.get('room_type') or 'unspecified'}.",
             f"Floor: about {x1 - x0:.1f} m x {y1 - y0:.1f} m ({floor.area:.1f} m2), {doors} door(s), {windows} window(s).",
             "Objects: " + "; ".join(items) + "."]
    if cons:
        lines.append("Requirements: " + "; ".join(cons) + ".")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------- judge
def ask_judge(prompt, pngs, settings, judge=JUDGE, attempts=5):
    content = [{"type": "input_text", "text": prompt}] + [
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(p).decode()} for p in pngs]
    body = {"model": judge, "store": False, "input": [{"role": "user", "content": content}]}
    price = JUDGE_PRICES.get(judge, JUDGE_PRICES["gpt-5.6-sol"])
    url = settings["base_url"].rstrip("/") + "/responses"
    problem = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {
                "Authorization": f"Bearer {settings['api_key']}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                ans = json.load(r)
            text = "".join(p.get("text", "") for it in ans.get("output") or [] if it.get("type") == "message"
                           for p in it.get("content") or [] if p.get("type") == "output_text")
            u = ans.get("usage") or {}
            cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
            cost = ((u.get("input_tokens", 0) - cached) * price[0] + cached * price[1]
                    + u.get("output_tokens", 0) * price[2]) / 1e6
            if text:
                return text, cost, u
            problem = "empty answer"
        except urllib.error.HTTPError as e:
            problem = f"HTTP {e.code}"
            if e.code in (400, 401, 403, 404):
                raise RuntimeError(f"judge request rejected: {problem}") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            problem = type(e).__name__
        time.sleep(min(60, 5 * 2 ** k))
    raise RuntimeError(f"judge request failed: {problem}")


def parse_scores(text):
    first, last = text.find("{"), text.rfind("}")
    for a in range(first, last):
        if text[a] != "{":
            continue
        try:
            d = json.loads(text[a:last + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "overall_score" in d:
            return {k: d.get(k) for k in ("functionality_score", "layout_score", "aesthetics_score", "overall_score", "comments")}
    m = {}
    for key in ("functionality_score", "layout_score", "aesthetics_score", "overall_score"):
        i = text.find(f'"{key}"')
        if i >= 0:
            num = "".join(ch for ch in text[i + len(key) + 3:i + len(key) + 8] if ch.isdigit() or ch == ".")
            m[key] = float(num) if num else None
    return m or None


# ---------------------------------------------------------------------------------------------------- FID / KID
def inception_features(images):
    import torch
    from pytorch_fid.inception import InceptionV3
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).eval()
    feats = []
    with torch.no_grad():
        for k in range(0, len(images), 32):
            x = torch.stack([torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1) for im in images[k:k + 32]])
            feats.append(model(x)[0].squeeze(-1).squeeze(-1).numpy())
    return np.concatenate(feats)


def fid(a, b):
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid(a, b, rng, subsets=100, n=100):
    n = min(n, len(a), len(b) )
    if n == min(len(a), len(b)):  # every draw would hold every room: no spread to report, one exact value
        subsets = 1
    d = a.shape[1]
    vals = []
    for _ in range(subsets):
        x, y = a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)) * 1000, float(np.std(vals)) * 1000


# ---------------------------------------------------------------------------------------------------- main
def room_set(name, src):
    if name == "test300":
        return list(range(300))
    if name == "test50":
        return json.loads((EXP / "test50_rooms.json").read_text())["indices"]  # copied from the original workflow
    if name == "20":
        return sorted(i for i, v in src["ours staged (old time budget)"].items() if v[0])
    return [int(x) for x in name.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="20", help="test300, test50, 20 (the staged rooms) or comma-separated indices")
    ap.add_argument("--methods", nargs="*", help="method names as in big_table (default: all with layouts)")
    ap.add_argument("--oor", action="store_true")
    ap.add_argument("--oor-rotated", action="store_true", help="OOR boxes turned by the yaw (the paper: axis-aligned)")
    ap.add_argument("--oor-floor-only", action="store_true", help="OOR over floor-standing objects only")
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--fid-min-rooms", type=int, default=50,
                    help="FID/KID only with this many rooms (with fewer the 2048-d covariance is singular: FID is not "
                         "meaningful, KID is noisy)")
    ap.add_argument("--gpt", action="store_true")
    ap.add_argument("--repeats", type=int, default=1, help="GPT scores per layout (averaged)")
    ap.add_argument("--protocol", choices=("optiscene", "idesign"), default="optiscene",
                    help="optiscene: their Table 3 prompt (0-10, 3 aspects + overall); idesign: I-Design's evaluator "
                         "(1-10 or unknown, 5 aspects; run with --repeats 3 as they do)")
    ap.add_argument("--judge", default=JUDGE, help="judge model (the paper: gpt-4o); results in gpt/ or gpt_<judge>/")
    ap.add_argument("--workers", type=int, default=6, help="judge requests in parallel")
    ap.add_argument("--dry-run", action="store_true", help="GPT: write the prompts and images, call nothing")
    ap.add_argument("--legend", action="store_true", help="GPT: add one line explaining the diagram (not in the paper)")
    ap.add_argument("--budget-usd", type=float, default=3.)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    rows = [migrate_legacy_row(json.loads(l)) for l in open(ROOT / "test300/rows.jsonl") if l.strip()]
    src = sources(rows)
    src.pop("shuffled ground truth (anchor)", None) if a.gpt else None
    rooms = room_set(a.set, src)
    names = [m for m in src if (not a.methods or m in a.methods)]
    a.out.mkdir(parents=True, exist_ok=True)
    tag = a.set.replace(",", "_") + ("" if not a.gpt or a.judge == JUDGE else "_" + a.judge) + \
        ("_idesign" if a.gpt and a.protocol == "idesign" else "")
    report = {}

    if a.oor:
        res = {}
        for m in names:
            vals = [oor(rows[i], src[m][i][0], a.oor_rotated, a.oor_floor_only) for i in rooms
                    if i in src[m] and src[m][i][0]]
            vals = [v for v in vals if v is not None]
            res[m] = {"rooms": len(vals), "OOR": statistics.mean(vals) if vals else None}
        report["OOR"] = res
        print("OOR (" + ("rotated" if a.oor_rotated else "axis-aligned") + " boxes, "
              + ("floor-standing objects" if a.oor_floor_only else "all objects") + f", {a.set})")
        for m, r in res.items():
            print(f"  {m:<45} rooms {r['rooms']:>3}  OOR {r['OOR']:.4f}" if r["OOR"] is not None else f"  {m}: –")

    if a.fid:
        if shutil.disk_usage(Path.home()).free < 400e6:
            raise SystemExit("FID: less than 400 MB free; pytorch-fid downloads ~92 MB of Inception weights once")
        rng = np.random.default_rng(0)
        res = {}
        gt = {i: semantic_render(rows[i], rows[i]["target"]) for i in rooms}
        gt_feat = dict(zip(rooms, inception_features([gt[i] for i in rooms])))
        for m in names:
            if m == "ground truth":
                continue
            have = [i for i in rooms if i in src[m] and src[m][i][0]]
            if len(have) < a.fid_min_rooms:
                res[m] = {"rooms": len(have), "FID": None, "KID": None}
                continue
            f = inception_features([semantic_render(rows[i], src[m][i][0]) for i in have])
            g = np.stack([gt_feat[i] for i in have])
            res[m] = {"rooms": len(have), "FID": fid(g, f), "KID": kid(g, f, rng)}
            print(f"  {m:<45} rooms {len(have):>3}  FID {res[m]['FID']:.2f}  KID×1000 {res[m]['KID'][0]:.2f} ± {res[m]['KID'][1]:.2f}", flush=True)
        report["FID"] = res

    if a.gpt:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        gdir = a.out / (("gpt" if a.judge == JUDGE else "gpt_" + "".join(ch if ch.isalnum() else "_" for ch in a.judge))
                        + ("_idesign" if a.protocol == "idesign" else ""))
        gdir.mkdir(exist_ok=True)
        settings = None
        if not a.dry_run:
            from workflow.credentials import official_api
            settings = official_api(model=a.judge)
        tasks = []  # renders first (matplotlib is not thread-safe), then the requests in parallel
        for m in names:
            mdir = gdir / "".join(ch if ch.isalnum() else "_" for ch in m)
            mdir.mkdir(exist_ok=True)
            for i in rooms:
                lay = src[m].get(i, (None,))[0]
                if not lay:
                    continue
                prompt = (IDESIGN_PROMPT.format(preferences=preferences(rows[i])) if a.protocol == "idesign" else
                          PROMPT.format(legend="\n" + LEGEND if a.legend else "", preferences=preferences(rows[i])))
                todo = [k for k in range(a.repeats) if not (mdir / f"room_{i:03d}_{k}.json").exists()]
                if not todo and not a.dry_run:
                    continue
                pngs = [top_render(rows[i], lay), front_render(rows[i], lay)]
                if a.dry_run:
                    for view, png in zip(("top", "front"), pngs):
                        (mdir / f"room_{i:03d}_{view}.png").write_bytes(png)
                    (mdir / f"room_{i:03d}.prompt.txt").write_text(prompt)
                    continue
                tasks += [(m, i, k, prompt, pngs, mdir / f"room_{i:03d}_{k}.json") for k in todo]
        spent, failed = 0., 0
        if tasks:
            print(f"{len(tasks)} judge requests to {a.judge}, {a.workers} in parallel", flush=True)

            def run(t):
                m, i, k, prompt, pngs, f = t
                text, cost, usage = ask_judge(prompt, pngs, settings, a.judge)
                rec = {"room": i, "method": m, "repeat": k, "protocol": a.protocol,
                       "scores": parse_idesign(text) if a.protocol == "idesign" else parse_scores(text), "raw": text,
                       "cost_usd_estimate": cost, "usage": usage, "judge": a.judge}
                f.write_text(json.dumps(rec, indent=1))
                return cost
            with ThreadPoolExecutor(a.workers) as pool:
                futures = [pool.submit(run, t) for t in tasks]
                for n, fut in enumerate(as_completed(futures), 1):
                    try:
                        spent += fut.result()
                    except RuntimeError as exc:
                        failed += 1
                        print(f"  request failed: {exc}", flush=True)
                    if spent >= a.budget_usd:
                        print(f"budget reached (${spent:.2f} estimated); cancelling the rest", flush=True)
                        for other in futures:
                            other.cancel()
                        break
                    if n % 20 == 0:
                        print(f"  {n}/{len(tasks)} done, ~${spent:.2f}", flush=True)
        res = {}
        for m in names:
            mdir = gdir / "".join(ch if ch.isalnum() else "_" for ch in m)
            scores = [r["scores"] for f in sorted(mdir.glob("room_*_*.json"))
                      for r in [json.loads(f.read_text())] if r["room"] in rooms and r.get("scores")]
            if scores and a.protocol == "idesign":
                # I-Design: per scene, mean (and std) over the repeats; here the mean over scenes of those means
                per = {}
                for f in sorted(mdir.glob("room_*_*.json")):
                    r = json.loads(f.read_text())
                    if r["room"] in rooms and r.get("scores"):
                        per.setdefault(r["room"], []).append(r["scores"])
                row = {"rooms": len(per), "requests": len(scores)}
                for k in IDESIGN_ASPECTS:
                    means = [statistics.mean(g) for g in ([s[k] for s in reps if s.get(k) is not None] for reps in per.values()) if g]
                    stds = [float(np.std(g)) for g in ([s[k] for s in reps if s.get(k) is not None] for reps in per.values()) if len(g) > 1]
                    row[k] = {"mean": statistics.mean(means) if means else None, "within_scene_std": statistics.mean(stds) if stds else None,
                              "unknown": sum(1 for reps in per.values() for s in reps if s.get(k) is None)}
                res[m] = row
                print(f"  {m:<45} rooms {row['rooms']:>3}  " + "  ".join(
                    f"{k.split('_')[0][:5]} {row[k]['mean']:.2f}" if row[k]['mean'] is not None else f"{k.split('_')[0][:5]} –"
                    for k in IDESIGN_ASPECTS), flush=True)
                continue
            if scores:
                res[m] = {"n": len(scores), **{k: statistics.mean(float(s[k]) for s in scores if s.get(k) is not None)
                                               for k in ("functionality_score", "layout_score", "aesthetics_score",
                                                         "overall_score")}}
                print(f"  {m:<45} n {res[m]['n']:>3}  Func {res[m]['functionality_score']:.2f}  Layout "
                      f"{res[m]['layout_score']:.2f}  Aes {res[m]['aesthetics_score']:.2f}  Overall "
                      f"{res[m]['overall_score']:.2f}", flush=True)
        report["GPT"] = {"judge": a.judge, "scores": res}
        if a.dry_run:
            print(f"dry run: prompts and images in {gdir}")
        else:
            print(f"judge spend this run: ~${spent:.2f} (estimated; {failed} failed requests)")

    (a.out / f"report_{tag}.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
