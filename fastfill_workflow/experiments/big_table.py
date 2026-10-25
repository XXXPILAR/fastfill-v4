"""One table over every saved run: distribution metrics that the critic never sees, usability, legality, and the
critic-shaped metrics (kept apart), on three room sets (test300, test50, the 20 rooms run with ``staged``).

  KID ×1000  kernel inception distance between top-down semantic renders of a method's layouts and of the ground
             truth of the SAME rooms (features: ImageNet ResNet-50 pool, the weights cached locally, not Inception-v3,
             so values are comparable within this table only); 100 draws of 50 rooms each; only with >= 100 rooms
             (below that the estimate is noise: negative values, a shuffled ground truth scoring better than models)
  SCA %      a classifier (PCA 32 + logistic regression, 5-fold grouped by room) telling the ground truth of the same
             rooms from the method (balanced classes); 50 = indistinguishable; only with >= 50 rooms (unstable below
             100)
  SG, NAV    layout_quality (scene-graph recall against the ground truth; largest free floor component)
  REACH0.5   share of floor obstacles >= 0.3 m tall a 0.5 m wide person reaches from a door (the critic uses 0.4 m)
  COL, OPC   layout_quality; OOB: share of objects with more than 1 % of the footprint outside the floor
  legal %    rooms with no collision, no blocked door and nothing out of bounds
  REL, ACC   layout_quality: the critic optimises the same definitions (ACC the same rule, REL templates from the other
             half of the training data): optimisation targets, not independent evidence
A method is scored on the rooms where it returned a layout ("rooms"); a missing layout lowers "rooms" only.

    ../.venv/bin/python experiments/big_table.py --out experiments/big_table
"""
import argparse, hashlib, json, statistics, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import workflow.contract  # noqa: E402,F401  (puts fastfill on sys.path)
import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from fastfill.v2.cell_costs import door_zones  # noqa: E402
from fastfill.v2.ops import layout_quality as lq  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402
from workflow import critic  # noqa: E402
from workflow.geometry import DOOR_DEPTH_M  # noqa: E402

ROOT = HERE.parent
EXP = HERE / "experiments"
BASE = ROOT / "related work code/baselines/runs/full_agentsize"
SIZE = 224


def jsonl(path):
    return [json.loads(l).get("layout") for l in open(path) if l.strip()]


def room_file(dirs, i):
    for d in dirs:
        p = EXP / d / f"room_{i:03d}.json"
        if p.exists():
            return json.loads(p.read_text())
    return None


def sources(rows):
    """name -> {room: (layout or None, seconds or None, usd or None)}"""
    out = {}
    n = len(rows)
    out["ground truth"] = {i: (rows[i]["target"], None, None) for i in range(n)}
    out["shuffled ground truth (anchor)"] = {i: (lq.shuffled(rows[i], i), None, None) for i in range(n)}
    for m in ("base", "D", "E"):
        lays = jsonl(ROOT / f"server_runs/test300-v4/test300_{m}/predictions_spread.jsonl")
        out[f"FastFill {m} (spread)"] = {i: (lays[i], .08, 0.) for i in range(n)}
    gpt = jsonl(ROOT / "test300/gpt56/predictions.jsonl")
    out["GPT-5.6 one-shot"] = {}
    for i in range(n):
        p = ROOT / f"test300/gpt56/room_{i:03d}.json"
        sec = json.loads(p.read_text()).get("latency_s") if p.exists() else None
        out["GPT-5.6 one-shot"][i] = (gpt[i], sec, None)
    for m, name in (("layoutgpt", "LayoutGPT"), ("holodeck", "Holodeck"), ("layoutvlm", "LayoutVLM")):
        out[name] = {}
        for i in range(n):
            p = BASE / m / f"room_{i:03d}" / "result.json"
            if p.exists():
                r = json.loads(p.read_text())
                out[name][i] = (r.get("layout"), r.get("seconds"), None)
    runs = {"workflow v4 (draft + local repair)": (["test50_v4_server"], "wall_s", "cost_usd"),
            "ours A: all at once + critic loop": (["orders_test50_server/A", "orders_AC_rest30_server/A"], "e2e_s", "e2e_usd"),
            "ours C: by group": (["orders_test50_server/C", "orders_AC_rest30_server/C"], "e2e_s", "e2e_usd"),
            "ours C_final: by group + whole-room loop": (["orders_test50_server/C_final", "orders_AC_rest30_server/C_final"],
                                                          "e2e_s", "e2e_usd"),
            "ours staged (old time budget)": (["local_staged20"], "wall_s", "cost_usd")}
    for name, (dirs, sk, ck) in runs.items():
        out[name] = {}
        for i in range(n):
            r = room_file(dirs, i)
            if r is not None:
                out[name][i] = (r.get("layout"), r.get(sk), r.get(ck))
    return out


def colour(cat):
    h = hashlib.md5(cat.encode()).digest()
    return tuple(60 + b % 196 for b in h[:3])


def render(row, layout):
    """Top-down semantic render: floor grey, fixed objects dark, every object filled with its category's colour,
    lower objects first; the room's floor bounds fill the image (same transform for every method of a room)."""
    floor = Polygon(row["condition"]["room"]["floor_polygon_xy_m"])
    x0, y0, x1, y1 = floor.bounds
    scale = (SIZE - 8) / max(x1 - x0, y1 - y0, 1e-6)
    ox, oy = (SIZE - scale * (x1 - x0)) / 2, (SIZE - scale * (y1 - y0)) / 2
    tf = lambda pts: [(ox + (x - x0) * scale, SIZE - (oy + (y - y0) * scale)) for x, y in pts]  # noqa: E731
    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon(tf(floor.exterior.coords), fill=(128, 128, 128))
    for p in lq.fixed_obstacles(row):
        d.polygon(tf(p.exterior.coords), fill=(40, 40, 40))
    if layout:
        for b in sorted(lq.boxes(row, layout), key=lambda b: (b["bottom"], b["size"][2])):
            d.polygon(tf(b["poly"].exterior.coords), fill=colour(b["cat"]), outline=(255, 255, 255))
    return img


class Features:
    def __init__(self):
        net = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        self.net = net.eval()
        self.mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images):
        out = []
        for k in range(0, len(images), 32):
            x = torch.stack([torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
                             for im in images[k:k + 32]])
            out.append(self.net((x - self.mean) / self.std).numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def kid(real, fake, rng, subsets=100, n=50):
    if min(len(real), len(fake)) < 100:
        return None, None
    d = real.shape[1]
    vals = []
    for _ in range(subsets):
        x, y = real[rng.choice(len(real), n, replace=False)], fake[rng.choice(len(fake), n, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        m = n
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)) * 1000, float(np.std(vals)) * 1000


def sca(real, fake, groups_real, groups_fake):
    if len(fake) < 50:
        return None
    x = np.concatenate([real, fake])
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    g = np.r_[groups_real, groups_fake]
    accs = []
    for tr, te in GroupKFold(n_splits=5).split(x, y, g):
        k = min(32, len(tr) - 1, x.shape[1])
        clf = make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(max_iter=2000, C=.1))
        clf.fit(x[tr], y[tr])
        accs.append(float((clf.predict(x[te]) == y[te]).mean()))
    return 100 * float(np.mean(accs))


def per_room(row, layout, templates):
    bs = lq.boxes(row, layout)
    floor = Polygon(row["condition"]["room"]["floor_polygon_xy_m"]).buffer(0)
    out_n = sum(1 for b in bs if b["poly"].area > 0 and b["poly"].difference(floor).area > max(.01 * b["poly"].area, 5e-3))
    try:
        col = lq.col(row, layout)
    except Exception:
        col = (None, None)
    opc = lq.opc(row, bs)
    doors = [Polygon(z).intersection(floor) for z in door_zones(row["condition"]["room"], DOOR_DEPTH_M)]
    targets = [b for b in bs if lq.obstacle(b) and b["size"][2] >= lq.ACC_MIN_HEIGHT_M]
    missed = critic.unreachable(floor, bs, lq.fixed_obstacles(row), doors, .25, blockers=False) if floor.geom_type == "Polygon" else []
    return {"nav": lq.nav(row, bs), "acc": lq.acc(row, bs), "rel": lq.rel(bs, templates), "sg": lq.sg(row, bs),
            "opc": opc, "col": col, "oob": (out_n, len(bs)), "reach": (len(targets) - len(missed), len(targets)),
            "legal": col[0] == 0 and opc[0] == 0 and out_n == 0 if col[0] is not None else False}


def pooled(recs, key):
    h = sum(r[key][0] for r in recs if r[key][0] is not None)
    n = sum(r[key][1] for r in recs if r[key][1] is not None)
    return 100 * h / n if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EXP / "big_table")
    ap.add_argument("--templates", default=str(EXP / "quality50/eval_templates_odd.json"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = [migrate_legacy_row(json.loads(l)) for l in open(ROOT / "test300/rows.jsonl") if l.strip()]
    templates = json.loads(Path(a.templates).read_text())["templates"]
    src = sources(rows)
    sets = {"test300": list(range(len(rows))),
            "test50": json.loads((EXP / "test50_rooms.json").read_text())["indices"],
            "20 rooms (staged)": sorted(i for i, v in src["ours staged (old time budget)"].items() if v[0])}
    feats = Features()
    cache, metrics = {}, {}
    for name, rooms in src.items():
        keep = [i for i, (lay, _, _) in rooms.items() if lay]
        images = [render(rows[i], rooms[i][0]) for i in keep]
        cache[name] = dict(zip(keep, feats(images)))
        metrics[name] = {i: per_room(rows[i], rooms[i][0], templates) for i in keep}
        print(f"{name}: {len(keep)} layouts, {time.perf_counter() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    result, lines = {}, []
    for set_name, rooms in sets.items():
        lines += [f"\n## {set_name} ({len(rooms)} rooms)\n",
                  "| method | rooms | KID×1000 ↓ | SCA % (→50) | SG ↑ | NAV ↑ | REACH0.5 ↑ | COL ↓ | OOB ↓ | OPC ↓ | "
                  "legal % ↑ | REL* ↑ | ACC* ↑ | s / room (median) | USD / room |",
                  "|" + "---|" * 15]
        result[set_name] = {}
        for name, rooms_src in src.items():
            have = [i for i in rooms if i in cache[name]]
            if not have:
                continue
            recs = [metrics[name][i] for i in have]
            fake = np.stack([cache[name][i] for i in have])
            real = np.stack([cache["ground truth"][i] for i in have])  # the same rooms' ground truth
            k = (None, None) if name == "ground truth" else kid(real, fake, rng)
            s = None if name == "ground truth" else sca(real, fake, np.array(have), np.array(have))
            secs = [rooms_src[i][1] for i in have if rooms_src[i][1] is not None]
            usd = [rooms_src[i][2] for i in have if rooms_src[i][2] is not None]
            row = {"rooms": len(have), "of": len(rooms), "KID": k[0], "KID_std": k[1], "SCA": s,
                   "SG": pooled(recs, "sg"), "NAV": 100 * statistics.mean(r["nav"] for r in recs),
                   "REACH0.5": pooled(recs, "reach"), "COL": pooled(recs, "col"), "OOB": pooled(recs, "oob"),
                   "OPC": pooled(recs, "opc"), "legal": 100 * sum(r["legal"] for r in recs) / len(recs),
                   "REL": pooled(recs, "rel"), "ACC": pooled(recs, "acc"),
                   "seconds_median": statistics.median(secs) if secs else None,
                   "usd_mean": statistics.mean(usd) if usd else None}
            result[set_name][name] = row
            f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
            kid_s = "–" if k[0] is None else f"{k[0]:.1f} ± {k[1]:.1f}"
            lines.append(f"| {name} | {len(have)}/{len(rooms)} | {kid_s} | {f(s)} | {f(row['SG'])} | {f(row['NAV'])} | "
                         f"{f(row['REACH0.5'])} | {f(row['COL'])} | {f(row['OOB'])} | {f(row['OPC'])} | {f(row['legal'])} | "
                         f"{f(row['REL'])} | {f(row['ACC'])} | {f(row['seconds_median'], 0)} | {f(row['usd_mean'], 2)} |")
    text = ("# Every saved run on the same metrics\n\n" + (__doc__ or "").split("\n\n    ../")[0] + "\n" + "\n".join(lines) +
            "\n\n\\* REL and ACC: the critic optimises the same definitions; read them as targets reached.\n")
    (a.out / "table.md").write_text(text)
    (a.out / "metrics.json").write_text(json.dumps(result, indent=1))
    print(text)


if __name__ == "__main__":
    main()
