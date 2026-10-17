"""Per-object grid-head distribution + input ablations on validation rows (minimal projection), plus an input-blind
per-category cell prior from the TRAIN split.

Rows: validation.jsonl in file order, after the run's validation exclude_flags, whose three-field projection exists
(boundary-known axis-aligned rectangle) and has <= max_objects objects; the first --rows that collate.
Variants of the same rows (model inputs only; targets/normalization/groups of (a) unless noted):
  a original projection
  b room rectangle extents (W, H) replaced by another selected room's (seeded permutation), lower corner kept;
    text-only: targets stay in (a)'s normalized frame
  c room_type replaced by another selected room's (seeded permutation); text-only
  d descriptions := category (augment_sample with minimal_form_p=1, category_only_description_p=1: groups recomputed)
  e request order permuted (batch._shuffle_objects: reorders, renames obj_%04d in the new order like training)
Per object (prediction slot matched to its label by GeometryCriterion's own assignment): cell CE, predicted cell
entropy, p(GT cell), top-1/top-2 prob, GT-cell rank, argmax hit, GT-cell residual L1, argmax-decoded XY error (m),
size error (criterion's per-slot log-L1/3 with its swap choice), yaw CE + residual of the criterion's candidate,
decoded yaw error to the nearest symmetry candidate; for b-e also the total-variation distance of the cell / yaw
distributions to (a)'s for the same label object. Per-room sums are cross-checked against criterion term_sums.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from fastfill.v2.batch import _shuffle_objects, augment_sample, collate_samples, load_tokenizer
from fastfill.v2.data import filter_rows_by_flags
from fastfill.v2.evaluate import MINIMAL_PROJECTION, project_minimal
from fastfill.v2.geometry import decode_yaw, encode_grid_position, encode_yaw, wrap_yaw
from fastfill.v2.io import load_checkpoint_config, to_device
from fastfill.v2.losses import GeometryCriterion, LossConfig, _elementwise, _gather
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.objective import supervision_masks
from fastfill.v2.schema import normalize_room

VARIANTS = ("a", "b", "c", "d", "e")
DESC_ONLY = {**MINIMAL_PROJECTION, "category_only_description_p": 1.}
METRICS = ("cell_ce", "cell_entropy", "p_gt", "top1", "top2", "gt_rank", "argmax_hit", "residual_l1", "xy_err_m",
           "size_err", "yaw_ce", "yaw_reg", "yaw_err_rad", "yaw_choice_eq_cls_best", "cell_tv_vs_a", "yaw_tv_vs_a", "size_logdiff_vs_a")


def mask_all(m):
    return all(m) if isinstance(m, list) else bool(m)


def per_object(pred, batch, crit, cfg):
    """Records for slot i of the single scene; also returns criterion term sums for the cross-check."""
    res = crit(pred, batch, global_counts=False)
    A = res["assignment"]
    tg = {k: _gather(v, A) for k, v in batch["targets"].items()}
    va = {k: _gather(v, A) for k, v in batch["validity"].items() if k in ("position", "size", "yaw")}
    masks = {k: v[0] for k, v in supervision_masks(batch, va).items()}
    n = int(batch["slot_mask"][0].sum())
    recs = [{"slot": i, "label": int(A[0, i])} for i in range(n)]
    logits = pred["position_cell_logits"][0].float()
    resid = pred["position_cell_residuals"][0].float()
    G = math.isqrt(logits.shape[-1])
    learn = ~batch["fixed_position_mask"][0]
    gt = tg["position_normalized"][0]
    scale = batch["scale"][0]
    sums = defaultdict(float)
    for i in range(n):
        if not (masks["position"][i] and learn[i, 0] and learn[i, 1]):
            continue
        cell, target = encode_grid_position(gt[i, :2][None], G)
        cell = int(cell)
        logp = F.log_softmax(logits[i], -1)
        p = logp.exp()
        top = p.topk(2).values
        r = recs[i]
        r.update(cell_ce=float(-logp[cell]), cell_entropy=float(-(p * logp).sum()), p_gt=float(p[cell]),
                 top1=float(top[0]), top2=float(top[1]), gt_rank=int((logits[i] > logits[i, cell]).sum()),
                 argmax_hit=float(int(logits[i].argmax()) == cell),
                 residual_l1=float(_elementwise(resid[i, cell] - target[0], cfg.position_type, cfg.smooth_l1_beta).mean()),
                 xy_err_m=float(((pred["position_normalized"][0, i, :2].float() - gt[i, :2]) * scale[:2]).norm()),
                 _cell_p=p.detach().cpu().numpy(), _gt_cell=cell)
        sums["position_cell"] += r["cell_ce"]
    # size / yaw: GeometryCriterion's own per-slot terms and joint candidate choice (losses.py), replicated per slot
    size_learn = ~batch["fixed_size_mask"][0]
    sp, sg = pred["size"][0].float(), tg["size"][0]
    err = lambda t: torch.where(size_learn, _elementwise(sp.log() - t.log(), cfg.size_type, cfg.smooth_l1_beta), 0.).sum(-1) / 3
    plain, swapped = err(sg), err(sg[:, [1, 0, 2]])
    swap = _gather(batch["size_axis_swap_allowed"], A)[0] & batch["slot_mask"][0]
    pinned = swap & batch["fixed_size_mask"][0][..., :2].any(-1)
    swap = swap & ~pinned
    odd = swap & (swapped < plain)
    orders = _gather(batch["yaw_symmetry_order"], A)[0]
    yl, yr = pred["yaw_logits"][0].float(), pred["yaw_residuals"][0].float()
    ydec = decode_yaw(yl, yr)
    for i in range(n):
        if not masks["yaw"][i]:
            continue
        k = 4 if swap[i] else 2 if pinned[i] else int(orders[i])
        y = tg["yaw"][0, i]
        cands = y + torch.arange(k, device=y.device, dtype=y.dtype) * (2 * math.pi / k)
        bins, gres = encode_yaw(cands, yl.shape[-1])
        cls = F.cross_entropy(yl[i].expand(k, -1), bins, reduction="none")
        reg = F.smooth_l1_loss(yr[i][bins], gres, reduction="none", beta=1.)
        cost = cfg.yaw_cls * cls + cfg.yaw_reg * reg
        if swap[i] and masks["size"][i]:
            cost = cost + cfg.size * torch.stack((plain[i], swapped[i])).repeat(2)
        c = int(cost.argmin())
        if swap[i] and masks["size"][i]:
            odd[i] = bool(c % 2)
        recs[i].update(yaw_ce=float(cls[c]), yaw_reg=float(reg[c]),
                       yaw_err_rad=float(wrap_yaw(ydec[i] - cands).abs().min()),
                       _yaw_p=F.softmax(yl[i], -1).detach().cpu().numpy())
        if k > 1:  # did the joint (cls, reg, size) choice pick the classifier's own best candidate?
            recs[i]["yaw_choice_eq_cls_best"] = float(c == int(cls.argmin()))
        sums["yaw_cls"] += recs[i]["yaw_ce"]
    for i in range(n):
        if masks["size"][i]:
            recs[i]["size_err"] = float(swapped[i] if odd[i] else plain[i])
            recs[i]["_logsize"] = sp[i].log().detach().cpu().numpy()
            sums["size"] += recs[i]["size_err"]
    check = {k: (sums[k], float(res["term_sums"][k])) for k in ("position_cell", "size", "yaw_cls")}
    return recs, check


def prior_counts(train_path, G, max_rows=None):
    """Cell counts of label XY (normalized by the row's own room bounds) per category, per (category, room_type), global."""
    cat, catrt, glob, rows = defaultdict(lambda: np.zeros(G * G)), defaultdict(lambda: np.zeros(G * G)), np.zeros(G * G), 0
    with open(train_path) as f:
        for line in f:
            row = json.loads(line)
            rows += 1
            room = row["condition"]["room"]
            origin, scale = normalize_room(room)
            cats = {o["id"]: o["category"] for o in row["condition"]["objects"]}
            pv = row.get("validity", {}).get("position", [])
            for j, t in enumerate(row["target"]["objects"]):
                m = pv[j] if j < len(pv) else False
                ok = (m[0] and m[1]) if isinstance(m, list) else bool(m)
                if not ok or t.get("bottom_center_m") is None:
                    continue
                xy = [(t["bottom_center_m"][q] - origin[q]) / scale[q] for q in range(2)]
                ix, iy = (min(max(int(math.floor(v * G)), 0), G - 1) for v in xy)
                cell = ix * G + iy
                c = cats.get(t["id"])
                cat[c][cell] += 1
                catrt[(c, room.get("room_type"))][cell] += 1
                glob[cell] += 1
            if max_rows and rows >= max_rows:
                break
    return {"cat": dict(cat), "catrt": dict(catrt), "glob": glob, "rows": rows, "objects": int(glob.sum())}


def c4(h, G):
    """Average of the histogram over the 4 quarter turns of the normalized grid (training uses rotate90)."""
    m = h.reshape(G, G)
    return np.mean([np.rot90(m, k) for k in range(4)], 0).reshape(-1)


def prior_logp(counts, level, category, room_type, alpha, sym, G):
    if level == "glob":
        h = counts["glob"]
    elif level == "cat":
        h = counts["cat"].get(category, np.zeros(G * G))
    else:  # category+room type, backing off to the category histogram when the pair is unseen
        h = counts["catrt"].get((category, room_type))
        h = counts["cat"].get(category, np.zeros(G * G)) if h is None else h
    if sym:
        h = c4(h, G)
    return np.log((h + alpha) / (h.sum() + alpha * G * G))


def boot(S, C, idx):
    """Object-weighted mean and room-bootstrap CI from per-room sums S and counts C."""
    mean = S.sum() / max(C.sum(), 1)
    b = S[idx].sum(1) / np.maximum(C[idx].sum(1), 1)
    return [float(mean), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--train-max-rows", type=int)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ckpt = Path(a.checkpoint)
    run_start = ckpt.parent / "run_manifest_start.json"
    resolved = json.loads(run_start.read_text())["resolved_config"] if run_start.exists() else {}
    cfg = LossConfig(**resolved.get("loss", {}))
    exclude = resolved.get("validation", {}).get("exclude_flags", [])
    max_length = load_checkpoint_config(ckpt)["max_length"] or 4096
    model = load_model(ckpt, device=a.device).eval()
    tok = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(ckpt.parent / "tokenizer"), local_files_only=True)
    G = model.config.position_grid
    crit = GeometryCriterion(cfg)
    collate = lambda s: to_device(collate_samples([s], tok, max_length=max_length, max_objects=model.config.max_objects), a.device)

    rows = [json.loads(l) for l in open(a.validation) if l.strip()]
    assert all(r["provenance"].get("split") == "validation" for r in rows), "validation split only"
    flagged = len(rows)
    rows = filter_rows_by_flags(rows, exclude)
    flagged -= len(rows)
    sel, skipped = [], defaultdict(int)
    for r in rows:
        m = project_minimal(r)
        if m is None:
            skipped["non_rectangular"] += 1
            continue
        if len(m["condition"]["objects"]) > model.config.max_objects:
            skipped["over_128_objects"] += 1
            continue
        for t in m["target"]["objects"]:  # hidden tag: survives _shuffle_objects renaming, never rendered
            t["_orig_id"] = t["id"]
        try:
            collate(m)
        except ValueError as e:
            skipped["collate:" + str(e)[:60]] += 1
            continue
        sel.append((r, m))
        if len(sel) == a.rows:
            break
    R = len(sel)
    rng = np.random.default_rng(a.seed)

    def derange(k):
        p = rng.permutation(k)
        for i in range(k):
            if p[i] == i:
                j = (i + 1) % k
                p[i], p[j] = p[j], p[i]
        return p

    size_donor, type_donor = derange(R), derange(R)
    rect = lambda m: m["condition"]["room"]["floor_polygon_xy_m"]
    changed = {"b": 0, "c": 0, "d_objects_with_description_ne_category": 0, "objects": 0}
    records = {v: [] for v in VARIANTS}
    checks, times = [], defaultdict(float)
    with torch.no_grad():
        for ri, (row, m) in enumerate(sel):
            batch_a = collate(m)
            ref = {}
            for v in VARIANTS:
                t0 = time.perf_counter()
                if v == "a":
                    batch, inputs = batch_a, batch_a
                elif v in ("b", "c"):
                    s = copy.deepcopy(m)
                    room = s["condition"]["room"]
                    if v == "b":
                        (x0, y0), (x1, y1) = rect(m)[0], rect(m)[2]
                        (u0, w0), (u1, w1) = rect(sel[size_donor[ri]][1])[0], rect(sel[size_donor[ri]][1])[2]
                        X, Y = round(x0 + (u1 - u0), 6), round(y0 + (w1 - w0), 6)
                        room["floor_polygon_xy_m"] = [[x0, y0], [X, y0], [X, Y], [x0, Y]]
                        changed["b"] += (abs((u1 - u0) - (x1 - x0)) > 1e-6) or (abs((w1 - w0) - (y1 - y0)) > 1e-6)
                    else:
                        room["room_type"] = sel[type_donor[ri]][1]["condition"]["room"]["room_type"]
                        changed["c"] += room["room_type"] != m["condition"]["room"]["room_type"]
                    text = collate(s)
                    batch = batch_a
                    inputs = {**batch_a, **{k: text[k] for k in ("input_ids", "attention_mask", "object_spans")}}
                elif v == "d":
                    s = augment_sample(row, DESC_ONLY, torch.Generator().manual_seed(0))
                    changed["d_objects_with_description_ne_category"] += sum(o["description"] != o["category"] for o in m["condition"]["objects"])
                    changed["objects"] += len(m["condition"]["objects"])
                    for t in s["target"]["objects"]:
                        t["_orig_id"] = t["id"]
                    batch = inputs = collate(s)
                else:
                    s = copy.deepcopy(m)
                    _shuffle_objects(s["condition"], s["target"], s["validity"], torch.Generator().manual_seed(a.seed * 100003 + ri))
                    batch = inputs = collate(s)
                pred = model(**model_inputs(inputs))
                recs, check = per_object(pred, batch, crit, cfg)
                checks.append(check)
                labels = batch["objects"][0]
                tmap = {t["id"]: t.get("_orig_id", t["id"]) for t in (s if v in ("d", "e") else m)["target"]["objects"]}
                for r in recs:
                    lab = labels[r["label"]]
                    r.update(room=ri, variant=v, category=lab["category"], room_type=m["condition"]["room"]["room_type"],
                             orig_label=tmap.get(lab["id"], lab["id"]))
                    key = r["orig_label"]
                    if v == "a":
                        ref[key] = r
                    elif key in ref:
                        if "_cell_p" in r and "_cell_p" in ref[key]:
                            r["cell_tv_vs_a"] = float(.5 * np.abs(r["_cell_p"] - ref[key]["_cell_p"]).sum())
                        if "_yaw_p" in r and "_yaw_p" in ref[key]:
                            r["yaw_tv_vs_a"] = float(.5 * np.abs(r["_yaw_p"] - ref[key]["_yaw_p"]).sum())
                        if "_logsize" in r and "_logsize" in ref[key]:
                            r["size_logdiff_vs_a"] = float(np.abs(r["_logsize"] - ref[key]["_logsize"]).mean())
                records[v].extend(recs)
                times[v] += time.perf_counter() - t0
            if ri % 50 == 0:
                print(json.dumps({"room": ri, "s": dict(times)}), flush=True)
    worst = max(abs(x - y) / max(1., abs(y)) for c in checks for x, y in c.values())

    # input-blind prior on variant (a)'s position-scored objects
    t0 = time.perf_counter()
    counts = prior_counts(a.train, G, a.train_max_rows)
    prior_time = time.perf_counter() - t0
    priors = [(lvl, alpha, sym) for lvl in ("glob", "cat", "catrt") for alpha in (1., .1, .01) for sym in (False, True)]
    for r in records["a"]:
        if "_gt_cell" in r:
            for lvl, alpha, sym in priors:
                r[f"prior_{lvl}_a{alpha}_{'c4' if sym else 'raw'}"] = float(
                    -prior_logp(counts, lvl, r["category"], r["room_type"], alpha, sym, G)[r["_gt_cell"]])
    unseen_cat = sum("_gt_cell" in r and r["category"] not in counts["cat"] for r in records["a"])
    unseen_pair = sum("_gt_cell" in r and (r["category"], r["room_type"]) not in counts["catrt"] for r in records["a"])

    idx = np.random.default_rng(a.seed + 1).integers(0, R, size=(a.boot, R))
    def room_sums(v, key):
        S, C = np.zeros(R), np.zeros(R)
        for r in records[v]:
            if key in r:
                S[r["room"]] += r[key]
                C[r["room"]] += 1
        return S, C
    summary = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        keys = list(METRICS) + (sorted({k for r in records["a"] for k in r if k.startswith("prior_")}) if v == "a" else [])
        for key in keys:
            S, C = room_sums(v, key)
            if C.sum():
                summary[v][key] = boot(S, C, idx) + [int(C.sum())]
    diffs = {}
    for v in ("b", "c", "d", "e"):
        diffs[f"{v}-a"] = {}
        for key in ("cell_ce", "cell_entropy", "p_gt", "argmax_hit", "xy_err_m", "size_err", "yaw_ce", "yaw_err_rad"):
            Sv, Cv = room_sums(v, key)
            Sa, Ca = room_sums("a", key)
            d = Sv[idx].sum(1) / np.maximum(Cv[idx].sum(1), 1) - Sa[idx].sum(1) / np.maximum(Ca[idx].sum(1), 1)
            diffs[f"{v}-a"][key] = [float(Sv.sum() / Cv.sum() - Sa.sum() / Ca.sum()), float(np.percentile(d, 2.5)),
                                    float(np.percentile(d, 97.5))]
    model_minus_prior = {}
    Sa, Ca = room_sums("a", "cell_ce")
    for key in summary["a"]:
        if key.startswith("prior_"):
            Sp, Cp = room_sums("a", key)
            d = Sa[idx].sum(1) / np.maximum(Ca[idx].sum(1), 1) - Sp[idx].sum(1) / np.maximum(Cp[idx].sum(1), 1)
            model_minus_prior[key] = [float(Sa.sum() / Ca.sum() - Sp.sum() / Cp.sum()), float(np.percentile(d, 2.5)),
                                      float(np.percentile(d, 97.5))]
    report = {"checkpoint": str(ckpt), "rows": R, "validation_rows_after_flags": len(rows), "flag_excluded": flagged,
              "exclude_flags": exclude, "skipped_before_quota": dict(skipped), "loss_config": cfg.__dict__,
              "changed_rooms": changed, "criterion_crosscheck_max_rel_err": worst, "seconds": dict(times),
              "prior": {"train_rows": counts["rows"], "train_objects": counts["objects"], "categories": len(counts["cat"]),
                        "pairs": len(counts["catrt"]), "unseen_category_objects": unseen_cat,
                        "unseen_category_roomtype_objects": unseen_pair, "seconds": prior_time,
                        "smoothing": "add-alpha (Laplace) over G*G cells: (count+alpha)/(N+alpha*G*G); catrt backs off "
                                     "to the category histogram when the pair is unseen; c4 = average over 4 quarter turns",
                        "uniform_ce": math.log(G * G)},
              "summary_mean_ci95_count": summary, "diff_vs_a_mean_ci95": diffs, "model_a_minus_prior_cell_ce": model_minus_prior}
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    with open(out / "objects.jsonl", "w") as f:
        for v in VARIANTS:
            for r in records[v]:
                f.write(json.dumps({k: x for k, x in r.items() if not k.startswith("_")}) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "changed_rooms", "criterion_crosscheck_max_rel_err", "prior")}, indent=1))
    for v in VARIANTS:
        print(v, json.dumps({k: [round(x, 4) for x in summary[v][k][:3]] + [summary[v][k][3]] for k in summary[v]}))
    print(json.dumps(diffs, indent=1))
    print(json.dumps(model_minus_prior, indent=1))


if __name__ == "__main__":
    main()
