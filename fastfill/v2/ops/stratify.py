"""Per-object errors of one evaluate output, split by target selection rule, declared place, source family and
object-count bin, with room-level bootstrap 95% CIs and validator collision / clean-room rates per stratum.

    python stratify.py --eval-dir runs/test-X [--outcomes outcomes.jsonl] --rows data/D/test.jsonl --out S.json

Inputs: an ``evaluate`` output directory (report.json + outcomes*.jsonl) and the rows file it scored (its sha256 must
equal report.json ``data_sha256``; ``subset_limit`` is honoured). ``outcomes.jsonl`` is the report's first projection,
``outcomes-<p>.jsonl`` projection p; minimal rows are rebuilt with ``evaluate.project_minimal``. For a
``--from-head-outputs`` run, meta ``checkpoint`` / ``forward_implementation_sha256`` come from its head-output manifest
(the saving run), ``implementation_sha256`` is the code that decoded and scored.

Metrics (no new definitions): per object, ``evaluate.reference_metrics``' own loop on its own helpers (``_matched``,
``_log_size_error``, ``_yaw_error``, ``_box_equivalent_errors``, ``bev_iou``), kept per label; the label-only baselines
are ``evaluate.baseline_metrics``' loop per object on ``_label_rows`` / ``_estimate`` / ``fit_baselines`` (the run's
leave-one-out fit). Both are checked per request against the stored outcome (exact float equality) and the pooled
reference against report.json via ``evaluate._model_summary`` (exact). Validator: ``validation.validate_scene`` (code
on disk) on the layout and on the labels, as evaluate does; ``evaluate._room_validation`` per room, compared with the
stored ``target_validation`` (rooms that differ = validator drift since the evaluation).

Attribution: errors and baselines belong to the LABEL object (the legal matching's target); validator flags to the
PREDICTED object id that a check names. Strata: ``rule`` = target ``provenance.field_evidence[i].selection_rule``
(zipped with target.objects; missing = frozen_prep, so old data is all frozen_prep); ``place`` = the full row's
effective support declaration (floor / wall / on_object / undeclared; the minimal projection does not show it to the
model); ``family`` = provenance.source up to its first "_"; ``objects`` = requested-object count bin.
Means are object-weighted (rates: per object or per room as named) over rooms with a layout, except
interface_pass_rate / strict_ok_rate / clean_room_rate_incl_failures (every room). CI: percentile bootstrap over
rooms (multinomial room weights), ``--boot`` resamples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from fastfill.v2 import evaluate as E  # noqa: E402
from fastfill.v2.io import fingerprint, read_samples  # noqa: E402
from fastfill.v2.validation import effective_support_requests, validate_scene  # noqa: E402

RULES = ("frozen_prep", "wall_anchor", "support_inside_parent", "support_on_added_parent")
PLACES = ("floor", "wall", "on_object", "undeclared")
BINS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 128), (129, 10**9))  # 129+: beyond the baseline's max_objects
BASELINES = tuple(name for name, _ in E.BASELINE_METRICS)
OBJECT_KINDS, ROOM_KINDS = ("rule", "place"), ("family", "objects")
METRICS = (E.REFERENCE_METRICS + tuple(f"baseline:{b}" for b in BASELINES)
           + ("object_collision_rate", "object_hard_violation_rate", "gt_object_collision_rate",
              "room_clean_rate", "room_collision_rate", "gt_room_clean_rate", "gt_room_collision_rate",
              "interface_pass_rate", "strict_ok_rate", "clean_room_rate_incl_failures"))
COLLISIONS = ("collision", "fixed_collision")


def bin_name(n):
    return next(f"{lo}-{hi}" if hi < 10**9 else f"{lo}+" for lo, hi in BINS if lo <= n <= hi)


def object_errors(layout, sample):
    """reference_metrics' loop kept per label: [(label id, {metric: value or None})] in slot order, plus the
    per-request summary that reference_metrics would return for the pooled keys."""
    batch, assignment = E._matched(layout, sample, True)
    objects = batch["objects"][0]
    predicted = {o["id"]: o for o in layout["objects"]}
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    valid = batch["validity"]
    records, by_order, box_equivalent = [], {}, 0
    for i, j in enumerate(assignment[:len(objects)]):
        p, t = predicted[objects[i]["id"]], labels.get(objects[j]["id"], {})
        pv, sv, yv = bool(valid["position"][0, j].all()), bool(valid["size"][0, j].all()), bool(valid["yaw"][0, j])
        e = dict.fromkeys(E.REFERENCE_METRICS)
        if pv:
            e["bottom_center_error_m"] = float(np.linalg.norm(np.subtract(p["bottom_center_m"], t["bottom_center_m"])))
        order, swap = int(batch["yaw_symmetry_order"][0, j]), bool(batch["size_axis_swap_allowed"][0, j])
        size_error = E._log_size_error(p["target_size_local_m"], t["target_size_local_m"]) if sv else None
        yaw_error = E._yaw_error(p["yaw_rad"], t["yaw_rad"], 2 if swap else order) if yv else None
        e["log_size_error_plain_convention"], e["yaw_error_rad_plain_convention"] = size_error, yaw_error
        if (sv or yv) and swap:
            size_error, yaw_error = E._box_equivalent_errors(p, t, sv, yv)
            box_equivalent += 1
        e["log_size_error"], e["yaw_error_rad"] = size_error, yaw_error
        if yv:
            by_order.setdefault(order, []).append(yaw_error)
        if pv and sv and yv:
            e["bev_iou"] = float(E.bev_iou(*(torch.tensor(v, dtype=torch.float64) for v in (
                p["bottom_center_m"], p["target_size_local_m"], p["yaw_rad"],
                t["bottom_center_m"], t["target_size_local_m"], t["yaw_rad"]))))
        records.append((objects[j]["id"], e))
    summary = {**{key: E._stat([e[key] for _, e in records if e[key] is not None]) for key in E.REFERENCE_METRICS},
               "yaw_error_rad_by_symmetry_order": {str(o): E._stat(v) for o, v in sorted(by_order.items())},
               "box_equivalent_objects": box_equivalent}
    return records, summary


def object_baselines(sample, fit, exclude_row):
    """baseline_metrics' loop kept per object: {id: {baseline name: value, pv, sv, yv}} plus the per-request summary."""
    objects, geometry, origin, scale = E._label_rows(sample)
    labels = {o["id"]: o for o in sample["target"]["objects"]}
    center = np.array([origin[0] + scale[0] / 2, origin[1] + scale[1] / 2, origin[2]])
    own = {fit["keys"][(exclude_row, o["id"])] for o in objects if (exclude_row, o["id"]) in fit["keys"]}
    out, lists = {}, {name: [] for name in BASELINES}
    for i, obj in enumerate(objects):
        target = labels.get(obj["id"], {})
        b = {"pv": all(geometry["position_valid"][i]), "sv": all(geometry["size_valid"][i]), "yv": bool(geometry["yaw_valid"][i])}
        if b["pv"]:
            gt = np.asarray(target["bottom_center_m"], dtype=np.float64)
            b["room_center_position"] = float(np.linalg.norm(center - gt))
            estimate, _ = E._estimate(fit, "position", obj["category"], own, np.mean)
            if estimate is not None:
                b["category_mean_position"] = float(np.linalg.norm(np.asarray(origin) + np.asarray(scale) * estimate - gt))
        if b["sv"]:
            estimate, _ = E._estimate(fit, "size", obj["category"], own, np.median)
            if estimate is not None:
                label = target["target_size_local_m"]
                error = E._log_size_error(estimate, label)
                if geometry["swap"][i]:
                    error = min(error, E._log_size_error(estimate, (label[1], label[0], label[2])))
                b["category_median_size"] = error
        if b["yv"]:
            b["uniform_yaw"] = E._uniform_yaw_error(geometry["symmetry"][i], geometry["swap"][i],
                                                    target["target_size_local_m"] if b["sv"] else None)
        for name in BASELINES:
            if name in b:
                lists[name].append(b[name])
        out[obj["id"]] = b
    return out, {name: {metric: E._stat(lists[name])} for name, metric in E.BASELINE_METRICS}


def _flags(validation, ids):
    collision, hard = dict.fromkeys(ids, False), dict.fromkeys(ids, False)
    for c in validation["checks"]:
        for i in c["object_ids"]:
            if i in collision and c["status"] == "violation":
                collision[i] |= c["code"] in COLLISIONS
                hard[i] |= c["hard"]
    return collision, hard


_STATE = {}


def _room(line):
    """One outcome line -> its room record (worker)."""
    o = json.loads(line)
    row, projection, fit, exclude = o["row"], _STATE["projection"], _STATE["fit"], _STATE["exclude_row"]
    full = _STATE["rows"][row]
    if o.get("provenance", {}).get("scene_id") != full["provenance"].get("scene_id"):
        raise ValueError(f"outcome row {row} is scene {o.get('provenance', {}).get('scene_id')}, rows file has another")
    sample = E.project_minimal(full) if projection == "minimal" else full
    evidence = full["provenance"].get("field_evidence") or [None] * len(full["target"]["objects"])
    if len(evidence) != len(full["target"]["objects"]):  # zip would silently attribute rules to the wrong objects
        raise ValueError(f"row {row}: {len(evidence)} field_evidence entries for {len(full['target']['objects'])} target objects")
    rules = {t["id"]: (e or {}).get("selection_rule") or "frozen_prep" for t, e in zip(full["target"]["objects"], evidence)}
    parents = {r["id"]: r.get("support_parent") for r in effective_support_requests(full["condition"])}
    ids = [obj["id"] for obj in sample["condition"]["objects"]]
    objects = {i: {"rule": rules.get(i, "frozen_prep"),
                   "place": {None: "undeclared", "floor": "floor", "wall": "wall"}.get(parents[i], "on_object"),
                   "err": {}, "base": {}} for i in ids}
    base, base_summary = object_baselines(sample, fit, None if exclude is None else row)
    for i, b in base.items():
        objects[i]["base"] = b
    gt = validate_scene(sample["condition"], sample["target"]["objects"])
    gt_collision, gt_hard = _flags(gt, ids)
    room = {"row": row, "scene_id": full["provenance"].get("scene_id"), "source": str(full["provenance"].get("source")),
            "n_objects": len(ids), "layout": "error" not in o, "gt_room": E._room_validation(gt),
            "over_capacity": E.OBJECT_BUDGET_ERROR in o.get("error", {}).get("message", ""),
            "latency_ms": o.get("fastfill_latency_ms"), "objects": objects}
    room["family"], room["objects_bin"] = room["source"].split("_")[0], bin_name(len(ids))
    stored_base = {name: o["baselines"][name] for name in BASELINES} if o.get("baselines") else None
    check = {"baselines_exact": stored_base == base_summary if stored_base else None, "reference_exact": None,
             "validation_same": None,
             "gt_validation_same": (o["ground_truth_validation"] == room["gt_room"]) if o.get("ground_truth_validation") else None}
    for i in ids:
        objects[i]["gt_collision"], objects[i]["gt_hard"] = gt_collision[i], gt_hard[i]
    if room["layout"]:
        layout = o["raw_prediction"]
        records, summary = object_errors(layout, sample)
        for label, e in records:
            objects[label]["err"] = e
        validation = validate_scene(sample["condition"], layout["objects"])
        collision, hard = _flags(validation, ids)
        for i in ids:
            objects[i]["collision"], objects[i]["hard"] = collision[i], hard[i]
        room["model_room"], room["strict_ok"] = E._room_validation(validation), validation["ok"]
        stored = o["model"]["reference"]
        check["reference_exact"] = stored is not None and all(stored[k] == summary[k] for k in summary)
        check["validation_same"] = E._room_validation(o["target_validation"]) == room["model_room"]
        room["summary"] = summary
    else:
        room["model_room"], room["strict_ok"], room["summary"] = None, False, None
    room["model_flags"] = {k: o["model"].get(k) for k in ("schema_success", "requested_ids_exactly_once",
                                                          "positive_valid_size", "target_geometry_valid")}
    room["base_summary"], room["check"] = base_summary, check
    return room


def _init(state):
    _STATE.update(state)
    torch.set_num_threads(1)


def run_info(eval_dir, outcomes="outcomes.jsonl"):
    """(report.json, projection of the outcomes file, the report section that summarizes it)."""
    report = json.loads((Path(eval_dir) / "report.json").read_text())
    projection = report["projection"] if outcomes == "outcomes.jsonl" else outcomes[len("outcomes-"):-len(".jsonl")]
    return report, projection, report if outcomes == "outcomes.jsonl" else report["projections"][projection]


def collect(eval_dir, outcomes="outcomes.jsonl", rows_path=None, workers=6, cache=None):
    """Room records of one evaluate output plus the reproduction checks. Refuses rows the run did not score.
    ``cache`` (a directory): reuse the result of the same outcomes file, rows and code (this script, fastfill/v2)."""
    eval_dir = Path(eval_dir)
    report, projection, section = run_info(eval_dir, outcomes)
    rows_path = Path(rows_path or report["data_path"])
    sha = fingerprint(rows_path)
    if sha != report["data_sha256"]:
        raise SystemExit(f"refused: {rows_path} sha256 {sha[:12]} is not the scored data {report['data_sha256'][:12]}")
    if cache:
        code = [fingerprint(p) for p in (Path(__file__), *sorted(Path(E.__file__).parent.glob("*.py")))]
        key = json.dumps([str(eval_dir.resolve()), outcomes, fingerprint(eval_dir / outcomes), sha, code])
        cached = Path(cache) / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.pkl"
        if cached.is_file():
            return pickle.loads(cached.read_bytes())
        result = collect(eval_dir, outcomes, rows_path, workers)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.with_suffix(".tmp").write_bytes(pickle.dumps(result))
        cached.with_suffix(".tmp").rename(cached)
        return result
    rows = read_samples(rows_path, max_samples=report.get("subset_limit"))
    fit_source = (section.get("baselines") or {}).get("fit") or "evaluation_set_leave_one_out"
    if fit_source.startswith("baseline_fit_file:"):
        with open(fit_source.split(":", 1)[1]) as stream:
            fit = E.fit_baselines(json.loads(line) for line in stream if line.strip())
        exclude = None
    else:
        fit, exclude = E.fit_baselines(rows), "row"
    with open(eval_dir / outcomes) as stream:
        lines = [line for line in stream if line.strip()]
    state = {"rows": rows, "fit": fit, "exclude_row": exclude, "projection": projection}
    if workers > 1:
        _STATE.update(state)  # forked workers inherit the rows and the fit
        with mp.get_context("fork").Pool(workers, initializer=_init, initargs=({},)) as pool:
            rooms = pool.map(_room, lines, chunksize=8)
    else:
        _init(state)
        rooms = [_room(line) for line in lines]
    # pooled reference and baselines rebuilt from the per-object values with evaluate's own summarizers
    pseudo = [{"model": {**r["model_flags"], "reference": r["summary"]}} for r in rooms]
    summary = E._model_summary(pseudo)["reference"]
    stored = section["model"]["reference"]
    baselines = E._baseline_summary([{"baselines": {**r["base_summary"], "category_fallback_objects": 0}} for r in rooms])
    stored_b = section.get("baselines") or {}
    checks = {
        "requests": len(rooms), "requests_with_layout": sum(r["layout"] for r in rooms),
        "over_capacity_requests": sum(r["over_capacity"] for r in rooms),
        "request_reference_mismatches": sum(r["check"]["reference_exact"] is False for r in rooms),
        "request_baseline_mismatches": sum(r["check"]["baselines_exact"] is False for r in rooms),
        "rooms_validation_differs_from_stored": sum(r["check"]["validation_same"] is False for r in rooms),
        "rooms_gt_validation_differs_from_stored": sum(r["check"]["gt_validation_same"] is False for r in rooms),
        "rooms_gt_validation_stored": sum(r["check"]["gt_validation_same"] is not None for r in rooms),
        "report_reference_exact": {k: summary[k] == stored[k] for k in stored},
        "report_baselines_exact": {name: baselines[name] == stored_b.get(name) for name in BASELINES},
        "recomputed_reference": {k: summary[k] for k in E.REFERENCE_METRICS}}
    checks["reproduces_report_exactly"] = (all(checks["report_reference_exact"].values())
                                           and all(checks["report_baselines_exact"].values())
                                           and not checks["request_reference_mismatches"] and not checks["request_baseline_mismatches"])
    manifest = report.get("head_outputs_manifest") or {}  # --from-head-outputs: the forward pass ran in the saving run
    meta = {"eval_dir": str(eval_dir.resolve()), "outcomes": outcomes, "projection": projection, "rows": str(rows_path.resolve()),
            "data_sha256": sha, "subset_limit": report.get("subset_limit"),
            **{k: report.get(k) for k in ("grid_decode", "baseline", "code_commit", "implementation_sha256", "head_outputs")},
            "checkpoint": report.get("checkpoint") or manifest.get("checkpoint"),
            "forward_implementation_sha256": manifest.get("implementation_sha256") if report.get("head_outputs")
            else report.get("implementation_sha256") if report.get("checkpoint") else None,
            "baseline_fit": fit_source, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return meta, rooms, checks


def _rate(values):
    return sum(map(float, values)), len(values)


def room_values(room, metric, objs, *, fallback=False):
    """(sum, count) of one metric over the kept objects ``objs`` of one room, or None if the room does not count.
    ``fallback`` (llm_compare): a room without a layout scores its label-only baselines (position: room centre,
    size: category median, yaw: uniform-random expectation, IoU 0; clean room 0) instead of being left out."""
    if not objs:
        return None
    layout = room["layout"]
    if metric in ("interface_pass_rate", "strict_ok_rate", "clean_room_rate_incl_failures"):
        value = {"interface_pass_rate": layout, "strict_ok_rate": room["strict_ok"],
                 "clean_room_rate_incl_failures": layout and not room["model_room"]["hard_violation"]}[metric]
        return float(value), 1
    if not layout and not fallback:
        return None
    if not layout:  # fallback
        if metric == "room_clean_rate":
            return 0., 1
        key = {"bottom_center_error_m": "room_center_position", "log_size_error": "category_median_size",
               "yaw_error_rad": "uniform_yaw"}.get(metric)
        if key:
            return _rate([o["base"][key] for o in objs if key in o["base"]])
        if metric == "bev_iou":
            return _rate([0. for o in objs if o["base"]["pv"] and o["base"]["sv"] and o["base"]["yv"]])
        return None
    if metric in E.REFERENCE_METRICS:
        return _rate([o["err"][metric] for o in objs if o["err"].get(metric) is not None])
    if metric.startswith("baseline:"):
        key = metric.split(":", 1)[1]
        return _rate([o["base"][key] for o in objs if key in o["base"]])
    if metric in ("object_collision_rate", "object_hard_violation_rate", "gt_object_collision_rate"):
        key = {"object_collision_rate": "collision", "object_hard_violation_rate": "hard", "gt_object_collision_rate": "gt_collision"}[metric]
        return _rate([o[key] for o in objs])
    validation = room["gt_room"] if metric.startswith("gt_") else room["model_room"]
    if metric.endswith("clean_rate"):
        return float(not validation["hard_violation"]), 1
    return float(validation["collision_pairs"] + validation["fixed_collision_pairs"] > 0), 1  # *_room_collision_rate


def bootstrap(num, den, boot, seed, chunk=200):
    """Point ratios sum(num)/sum(den) per column and ``boot`` resampled ratios (rooms resampled with replacement)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rooms = len(num)
    rng = np.random.default_rng(seed)
    samples = []
    with np.errstate(invalid="ignore", divide="ignore"):
        point = num.sum(0) / den.sum(0)
        for start in range(0, boot, chunk):
            w = rng.multinomial(rooms, np.full(rooms, 1 / rooms), size=min(chunk, boot - start)).astype(float)
            samples.append((w @ num) / (w @ den))
    return point, np.concatenate(samples)


def ci(samples):
    with warnings.catch_warnings():  # a stratum absent from a resample is NaN there; all-NaN columns stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(samples, [2.5, 97.5], axis=0)


def strata(room):
    """(kind, value, kept objects) for every stratum the room belongs to."""
    objs = list(room["objects"].values())
    yield "all", "all", objs
    for kind in OBJECT_KINDS:
        groups = {}
        for o in objs:
            groups.setdefault(o[kind], []).append(o)
        yield from ((kind, value, kept) for value, kept in groups.items())
    yield "family", room["family"], objs
    yield "objects", room["objects_bin"], objs


def aggregate(rooms, boot=2000, seed=0):
    columns, cells = {}, []  # (kind, value, metric) -> column index; per room: [(column, num, den)]
    counts = {}
    for r, room in enumerate(rooms):
        for kind, value, objs in strata(room):
            c = counts.setdefault((kind, value), {"rooms": 0, "objects": 0, "rooms_with_layout": 0})
            c["rooms"] += 1
            c["objects"] += len(objs)
            c["rooms_with_layout"] += room["layout"]
            for metric in METRICS:
                value_pair = room_values(room, metric, objs)
                if value_pair is not None:
                    cells.append((r, columns.setdefault((kind, value, metric), len(columns)), *value_pair))
    num, den = np.zeros((len(rooms), len(columns))), np.zeros((len(rooms), len(columns)))
    for r, col, n, d in cells:
        num[r, col] += n
        den[r, col] += d
    point, samples = bootstrap(num, den, boot, seed)
    lo, hi = ci(samples)
    order = {"rule": RULES, "place": PLACES, "objects": tuple(bin_name(lo_) for lo_, _ in BINS)}
    rank = lambda kind, value: (order[kind].index(value) if value in order.get(kind, ()) else len(order.get(kind, ())), value)
    finite = lambda x: float(x) if np.isfinite(x) else None
    out = {}
    for kind, value in sorted(counts, key=lambda kv: (kv[0], *rank(*kv))):
        entry = out.setdefault(kind, {}).setdefault(value, {**counts[kind, value], "metrics": {}})
        for metric in METRICS:
            col = columns.get((kind, value, metric))
            if col is not None and den[:, col].sum():
                entry["metrics"][metric] = {"mean": float(point[col]), "ci95": [finite(lo[col]), finite(hi[col])],
                                            "n": int(den[:, col].sum())}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, type=Path)
    p.add_argument("--outcomes", default="outcomes.jsonl", help="outcomes.jsonl (first projection) or outcomes-<p>.jsonl")
    p.add_argument("--rows", type=Path, help="the scored rows file (default: report.json data_path)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache", type=Path, help="directory of collected room records, reused by compare.py --cache")
    a = p.parse_args(argv)
    meta, rooms, checks = collect(a.eval_dir, a.outcomes, a.rows, a.workers, a.cache)
    result = {"meta": {**meta, "boot": a.boot, "seed": a.seed, "metrics": METRICS,
                       "attribution": "errors/baselines -> label object; validator flags -> predicted object id; "
                                      "place = full row's declaration; rule = target field_evidence selection_rule"},
              "checks": checks, "strata": aggregate(rooms, a.boot, a.seed),
              "latency_ms": {"observed": sum(r["latency_ms"] is not None for r in rooms),
                             **({"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
                                if (v := [r["latency_ms"] for r in rooms if r["latency_ms"] is not None]) else {})}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"out": str(a.out), **{k: checks[k] for k in checks if k != "recomputed_reference"}}, indent=1))


if __name__ == "__main__":
    main()
