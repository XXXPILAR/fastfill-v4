"""Paired room-level comparison of two evaluate outputs on the same rows (A minus B).

    python compare.py --a runs/test-X --b runs/test-X-argmax --label-a spread --label-b argmax --out C.json \
        [--a-outcomes outcomes.jsonl] [--b-outcomes outcomes.jsonl] [--rows ROWS] [--where rule=frozen_prep] \
        [--max-objects 128] [--boot 10000] [--cache DIR]

Refuses unless both runs scored the same rows: equal report ``data_sha256`` and projection, and the same outcome rows
with the same ``provenance.scene_id`` in the same order; and unless both were decoded and scored by the same code:
equal report ``implementation_sha256`` and ``baseline`` (online / supplied_head_outputs / ...) and, for head outputs,
equal forward-pass code (the head-output manifests' ``implementation_sha256``). Per-object values come from
``stratify.collect`` (evaluate's own metric code, reproduction-checked). ``--where KEY=V1[,V2]`` (repeatable; rule /
place on objects, family / objects on rooms) keeps a subset; ``--max-objects N`` keeps rooms of at most N requested
objects (rooms both models can answer); room-level validator metrics are reported only without an object filter.

Per metric, over the rooms where both runs have a layout and the metric has a value: PRIMARY the mean of per-room
differences with its paired room-bootstrap 95% CI (``--boot`` resamples of rooms); significant when the CI excludes 0,
direction from its sign. ROBUSTNESS an exact two-sided sign test on the per-room differences (ties dropped), Holm's
correction across the metrics of this comparison; ``verdict.agree`` false flags a disagreement of the two. The pooled
(object-weighted) means of A and B (``a`` / ``b``) and their difference with CI, and the per-room means
(``a_room_mean`` / ``b_room_mean``, whose difference is ``room_mean_diff``) are descriptive. Lower is better except
bev_iou and room_clean_rate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import stratify as S

ERROR_METRICS = S.E.REFERENCE_METRICS + ("object_collision_rate", "object_hard_violation_rate")
ROOM_METRICS = ("room_clean_rate", "room_collision_rate")
HIGHER_IS_BETTER = {"bev_iou", "room_clean_rate", "clean_room_rate_incl_failures", "interface_pass_rate"}


def sign_test(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n, k = pos + neg, min(pos, neg)
    p = min(1., 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.
    return {"a_greater": pos, "b_greater": neg, "ties": int(len(d) - n), "p": p}


def holm(ps):
    order, adjusted, running = sorted(range(len(ps)), key=lambda i: ps[i]), [None] * len(ps), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def keep_fn(where, max_objects=None):
    """Object filter and room filter from ``--where`` pairs and ``--max-objects``."""
    obj = {k: v for k, v in where.items() if k in S.OBJECT_KINDS}
    room = {k: v for k, v in where.items() if k in S.ROOM_KINDS}
    room_key = {"family": "family", "objects": "objects_bin"}
    return (lambda o: all(o[k] in v for k, v in obj.items())), (lambda r: all(r[room_key[k]] in v for k, v in room.items())
                                                               and (max_objects is None or r["n_objects"] <= max_objects)), bool(obj)


def paired(rooms_a, rooms_b, metrics, *, boot, seed, keep=lambda o: True, fallback=False):
    """Paired statistics of aligned room lists; a metric counts a room only when both runs give it a value."""
    out = {}
    for m, metric in enumerate(metrics):
        cells = []
        for ra, rb in zip(rooms_a, rooms_b):
            va = S.room_values(ra, metric, [o for o in ra["objects"].values() if keep(o)], fallback=fallback)
            vb = S.room_values(rb, metric, [o for o in rb["objects"].values() if keep(o)], fallback=fallback)
            if va and vb and va[1] and vb[1]:
                cells.append((*va, *vb))
        if not cells:
            continue
        c = np.array(cells)
        d = c[:, 0] / c[:, 1] - c[:, 2] / c[:, 3]
        # one set of room weights for pooled A, pooled B and the per-room differences: the resamples stay paired
        point, samples = S.bootstrap(np.column_stack((c[:, 0], c[:, 2], d)),
                                     np.column_stack((c[:, 1], c[:, 3], np.ones(len(d)))), boot, seed + m)
        out[metric] = {"rooms": len(d), "objects_a": int(c[:, 1].sum()), "objects_b": int(c[:, 3].sum()),
                       "a": float(point[0]), "b": float(point[1]), "pooled_diff": float(point[0] - point[1]),
                       "a_room_mean": float(np.mean(c[:, 0] / c[:, 1])), "b_room_mean": float(np.mean(c[:, 2] / c[:, 3])),
                       "pooled_diff_ci95": [float(x) for x in S.ci(samples[:, 0] - samples[:, 1])],
                       "room_mean_diff": float(point[2]), "room_mean_diff_ci95": [float(x) for x in S.ci(samples[:, 2])],
                       "sign_test": sign_test(d), "higher_is_better": metric in HIGHER_IS_BETTER}
    adjusted = holm([v["sign_test"]["p"] for v in out.values()])
    for v, p in zip(out.values(), adjusted):
        v["sign_test"]["p_holm"] = p
    return verdicts(out)


def verdicts(out):
    """primary: room-mean-difference CI excludes 0; sign_test: Holm p < .05; each a_better / a_worse / None."""
    for v in out.values():
        lo, hi = v["room_mean_diff_ci95"]
        st = v["sign_test"]
        side = lambda higher: None if higher is None else "a_better" if higher == v["higher_is_better"] else "a_worse"
        v["verdict"] = {"primary": side(True if lo > 0 else False if hi < 0 else None),
                        "sign_test": side(st["a_greater"] > st["b_greater"] if st["p_holm"] < .05 else None)}
        v["verdict"]["agree"] = v["verdict"]["primary"] == v["verdict"]["sign_test"]
    return out


def same_code(runs):
    """Refuse unless every (eval dir, outcomes file) was decoded and scored by the same code from the same kind of
    source, and head outputs were produced by the same forward-pass code."""
    codes = []
    for d, o in runs:
        report = S.run_info(d, o)[0]
        manifest = report.get("head_outputs_manifest") or {}
        codes.append((report.get("implementation_sha256"), report.get("baseline"), manifest.get("implementation_sha256")))
    if len(set(codes)) > 1:
        raise SystemExit("refused: the runs were not decoded by the same code (implementation_sha256, source, forward code): "
                         + "; ".join(f"{d}/{o}: {(k[0] or '-')[:12]} {k[1]} {(k[2] or '-')[:12]}" for (d, o), k in zip(runs, codes)))


def same_inputs(runs):
    """Refuse (before any work) unless every (eval dir, outcomes file) scored the same data in the same projection."""
    info = [S.run_info(d, o) for d, o in runs]
    keys = [(r["data_sha256"], projection, r.get("subset_limit")) for r, projection, _ in info]
    if len(set(keys)) > 1:
        raise SystemExit("refused: the runs did not score the same rows (data_sha256, projection, subset_limit): "
                         + "; ".join(f"{d}/{o}: {k[0][:12]} {k[1]} {k[2]}" for (d, o), k in zip(runs, keys)))


def aligned(*room_lists):
    """Refuse unless every run holds the same outcome rows with the same scene ids, in the same order."""
    if len({tuple((r["row"], r["scene_id"]) for r in rooms) for rooms in room_lists}) > 1:
        raise SystemExit("refused: the runs did not score the same rows (outcome rows / scene ids differ)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--a-outcomes", default="outcomes.jsonl")
    p.add_argument("--b-outcomes", default="outcomes.jsonl")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--rows", type=Path, help="the scored rows file (default: report.json data_path)")
    p.add_argument("--where", action="append", default=[], metavar="KEY=V1[,V2]")
    p.add_argument("--max-objects", type=int, help="keep rooms of at most N requested objects")
    p.add_argument("--cache", type=Path, help="stratify.collect cache directory")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    where = {}
    for item in a.where:
        key, _, values = item.partition("=")
        if key not in S.OBJECT_KINDS + S.ROOM_KINDS or not values:
            raise SystemExit(f"--where {item}: KEY must be one of {S.OBJECT_KINDS + S.ROOM_KINDS}")
        where[key] = set(values.split(","))
    same_inputs([(a.a, a.a_outcomes), (a.b, a.b_outcomes)])
    same_code([(a.a, a.a_outcomes), (a.b, a.b_outcomes)])
    meta_a, rooms_a, checks_a = S.collect(a.a, a.a_outcomes, a.rows, a.workers, a.cache)
    meta_b, rooms_b, checks_b = S.collect(a.b, a.b_outcomes, a.rows, a.workers, a.cache)
    aligned(rooms_a, rooms_b)
    keep, keep_room, object_filter = keep_fn(where, a.max_objects)
    excluded = sum(not keep_room(r) for r in rooms_a)
    pairs = [(ra, rb) for ra, rb in zip(rooms_a, rooms_b) if keep_room(ra)]
    rooms_a, rooms_b = [ra for ra, _ in pairs], [rb for _, rb in pairs]
    metrics = ERROR_METRICS + (() if object_filter else ROOM_METRICS)
    result = {"meta": {"a": {**meta_a, "label": a.label_a}, "b": {**meta_b, "label": a.label_b},
                       "where": {k: sorted(v) for k, v in where.items()}, "max_objects": a.max_objects, "boot": a.boot,
                       "seed": a.seed, "difference": "A minus B",
                       "primary": "paired room-bootstrap 95% CI of the mean per-room difference (direction from it)",
                       "robustness": "exact two-sided sign test on per-room differences, Holm across metrics",
                       "pooled": "object-weighted means and their difference: descriptive only"},
              "checks": {"a_reproduces_report": checks_a["reproduces_report_exactly"],
                         "b_reproduces_report": checks_b["reproduces_report_exactly"]},
              "rooms": {"compared": len(pairs), "excluded_by_room_filter": excluded,
                        "layout_both": sum(ra["layout"] and rb["layout"] for ra, rb in pairs),
                        "layout_only_a": sum(ra["layout"] and not rb["layout"] for ra, rb in pairs),
                        "layout_only_b": sum(rb["layout"] and not ra["layout"] for ra, rb in pairs),
                        "layout_neither": sum(not ra["layout"] and not rb["layout"] for ra, rb in pairs)},
              "metrics": paired(rooms_a, rooms_b, metrics, boot=a.boot, seed=a.seed, keep=keep)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps(result["rooms"]))
    for metric, v in result["metrics"].items():
        print(f"{metric:34s} room_mean_diff={v['room_mean_diff']:+.4f} [{v['room_mean_diff_ci95'][0]:+.4f},"
              f"{v['room_mean_diff_ci95'][1]:+.4f}] rooms={v['rooms']} sign {v['sign_test']['a_greater']}/"
              f"{v['sign_test']['b_greater']} p_holm={v['sign_test']['p_holm']:.3g} {v['verdict']}")


if __name__ == "__main__":
    main()
