"""FastFill (raw argmax / spread) vs the LLM agents on the frozen LLM rows, with paired room-level statistics.

    python llm_compare.py --method "FastFill spread=runs/X-spread,runs/X-online" --method "FastFill argmax=runs/X-argmax,runs/X-online" \
        --method "LLM prompt=runs/llm-prompt-300-eval-4125b82,runs/llm-prompt-300/summary.json" ... \
        [--reference "FastFill spread" --reference "FastFill argmax"] [--selection-cohort FILE] --out L.json

``--method LABEL=EVAL_DIR[,SUMMARY_JSON | ,LATENCY_EVAL_DIR]``: an evaluate output on the rows (outcomes.jsonl) and,
for an LLM, its llm_baseline / llm_structured summary.json (calls, tokens; the per-room ``latency_s`` and the
``error`` of every unanswered row come from the predictions.jsonl beside it); for FastFill, optionally the online
evaluate run whose ``fastfill_latency_ms`` gives the latency (a ``--from-head-outputs`` run observes none; its
grid_decode is recorded as ``latency_grid_decode``). Latency is reported over every room and over rooms with a layout;
rooms without a layout are listed with their object count and rank among the rows (``failed_rooms``). Every run must have scored the
same rows file (same data_sha256, projection, rows and scene ids), and the FastFill runs must have been decoded by the
same code (``compare.same_code``), else it refuses. References default to the first two methods.

Two scenarios, each with per-method object-weighted means (room bootstrap 95% CI) and, for every non-reference method
against every reference, ``compare.paired`` (difference = method minus reference; exact sign test on per-room
differences, Holm across the metrics of that pair, as robustness; the primary verdict is the paired room-bootstrap CI of
the mean per-room difference, see compare.py):
  common: only rooms where every method returned a layout (a failure of any method removes the room for all);
  failures_as_failures: every room; a room without a layout scores its label-only baselines from the same evaluate
    output (position: floor-level room centre; size: per-category median, leave-one-out over these rows; yaw: the
    uniform-random expectation ``evaluate._uniform_yaw_error``; BEV IoU 0; clean room 0).
``--selection-cohort FILE`` (repeatable) counts how many of these rows' scene ids occur in a checkpoint-selection
cohort file: rows that chose the checkpoint are not held out from it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import compare as C
import stratify as S

COMMON = S.E.REFERENCE_METRICS[:4] + ("object_collision_rate", "room_clean_rate", "room_collision_rate")
FAILURES = S.E.REFERENCE_METRICS[:4] + ("clean_room_rate_incl_failures", "interface_pass_rate")
COHORT_NOTE = ("A row whose scene is in a checkpoint-selection cohort took part in choosing that checkpoint: a FastFill "
               "checkpoint selected on that cohort is not held out on it; the LLM agents are.")


def pooled(rooms, metrics, *, boot, seed, fallback=False):
    out = {}
    for m, metric in enumerate(metrics):
        cells = [v for room in rooms if (v := S.room_values(room, metric, list(room["objects"].values()), fallback=fallback))]
        if cells:
            c = np.array(cells)
            point, samples = S.bootstrap(c[:, :1], c[:, 1:], boot, seed + m)
            out[metric] = {"mean": float(point[0]), "ci95": [float(x) for x in S.ci(samples[:, 0])], "rooms": len(c),
                           "n": int(c[:, 1].sum())}
    return out


def online_latency(eval_dir, rooms):
    """Per-room fastfill_latency_ms (s, None if absent) of the online run ``eval_dir`` on the same rows (scene ids in
    order), else refuses."""
    with open(Path(eval_dir) / "outcomes.jsonl") as stream:
        outcomes = [json.loads(line) for line in stream if line.strip()]
    if [o["provenance"].get("scene_id") for o in outcomes] != [r["scene_id"] for r in rooms]:
        raise SystemExit(f"refused: {eval_dir} did not score the same rows")
    return [None if o.get("fastfill_latency_ms") is None else o["fastfill_latency_ms"] / 1000 for o in outcomes]


def predictions(summary_path, rooms):
    """The predictions.jsonl beside summary.json, one per row, else refuses."""
    with open(Path(summary_path).parent / "predictions.jsonl") as stream:
        out = [json.loads(line) for line in stream if line.strip()]
    if len(out) != len(rooms):
        raise SystemExit(f"refused: {summary_path} predictions do not match the rows")
    return out


def failed_rooms(rooms):
    """Rooms without a layout: requested objects and rank among all rows by object count (1 = most objects), and the
    chance that all k fall among the R rooms with at least the smallest failed count if failure ignored size."""
    failed = [r for r in rooms if not r["layout"]]
    if not failed:
        return None
    sizes = [r["n_objects"] for r in rooms]
    top = sum(n >= min(r["n_objects"] for r in failed) for n in sizes)
    return {"rooms": [{"row": r["row"], "scene_id": r["scene_id"], "objects": r["n_objects"],
                       "rank": 1 + sum(n > r["n_objects"] for n in sizes)} for r in failed],
            "largest_rooms_holding_all": top, "of_rows": len(rooms),
            "p_all_in_largest_if_size_independent": math.comb(top, len(failed)) / math.comb(len(rooms), len(failed))}


def latency_stats(per_room, rooms):
    """Mean / p50 / p95 over every room with a latency, and the mean over rooms with a layout only."""
    every = [x for x in per_room if x is not None]
    layout = [x for x, r in zip(per_room, rooms) if x is not None and r["layout"]]
    return {"mean_latency_s": float(np.mean(every)) if every else None,
            "p50_latency_s": float(np.percentile(every, 50)) if every else None,
            "p95_latency_s": float(np.percentile(every, 95)) if every else None,
            "mean_latency_s_layout_rooms": float(np.mean(layout)) if layout else None, "latency_rooms_layout": len(layout)}


def cost(label, rooms, eval_dir, summary_path):
    if not summary_path or summary_path.is_dir():  # FastFill: one forward pass per room, latency observed by evaluate
        per_room = online_latency(summary_path, rooms) if summary_path else \
            [None if r["latency_ms"] is None else r["latency_ms"] / 1000 for r in rooms]
        return {"method": label, "kind": "fastfill", "rooms": len(rooms), "calls": None, "calls_per_room": 1,
                **latency_stats(per_room, rooms), "tokens": None, "unanswered": sum(not r["layout"] for r in rooms),
                "failure_causes": {k: v for k, v in (("over_capacity", sum(not r["layout"] and r["over_capacity"] for r in rooms)),
                                                     ("other", sum(not r["layout"] and not r["over_capacity"] for r in rooms))) if v},
                "failed_rooms": failed_rooms(rooms), "latency_source": str(summary_path or "this run"),
                "latency_grid_decode": json.loads((Path(summary_path or eval_dir) / "report.json").read_text()).get("grid_decode"),
                "latency_scope": "evaluate's fastfill_latency_ms: one GPU forward pass plus decoding per room"}
    s = json.loads(Path(summary_path).read_text())
    usage, preds = s.get("usage"), predictions(summary_path, rooms)
    causes = {}
    for room, prediction in zip(rooms, preds):  # error type (text before ':') of every row without a layout
        if not room["layout"]:
            kind = str(prediction.get("error") or "no error recorded").split(":")[0]
            causes[kind] = causes.get(kind, 0) + 1
    return {"method": label, "kind": "llm", "model": s.get("model"), "mode": s.get("mode"),
            "reasoning_effort": (s.get("request_parameters") or {}).get("reasoning_effort"), "rooms": s["rows"],
            "calls": s["api_calls"], "calls_per_room": s["api_calls"] / s["rows"],
            **latency_stats([p["latency_s"] for p in preds], rooms), "summary_mean_latency_s": s["mean_latency_s"],
            "tokens": None if not usage else {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
            "tokens_per_room": None if not usage else usage["total_tokens"] / s["rows"],
            "unanswered": s.get("unanswered"), "failed": s.get("failed"), "first_answer_invalid": s.get("first_answer_invalid"),
            "repairs": s.get("repairs"), "summary": str(summary_path), "failure_causes": causes, "failed_rooms": failed_rooms(rooms),
            "latency_scope": "predictions.jsonl latency_s (its mean is summary mean_latency_s): wall time per room over all "
                             "its API calls, including client rate-limit waits, retries and backoff (and repair calls)"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", action="append", required=True, metavar="LABEL=EVAL_DIR[,SUMMARY_JSON]")
    p.add_argument("--reference", action="append", metavar="LABEL", help="default: the first two methods")
    p.add_argument("--selection-cohort", action="append", default=[], type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args(argv)
    methods = {}
    for item in a.method:
        label, _, paths = item.rpartition("=")
        eval_dir, _, summary = paths.partition(",")
        methods[label] = (Path(eval_dir), Path(summary) if summary else None)
    references = a.reference or list(methods)[:2]
    if set(references) - set(methods):
        raise SystemExit(f"unknown reference {set(references) - set(methods)}")
    C.same_inputs([(d, "outcomes.jsonl") for d, _ in methods.values()])
    C.same_code([(d, "outcomes.jsonl") for d, s in methods.values() if not s or s.is_dir()])  # FastFill decodes
    collected = {label: S.collect(d, "outcomes.jsonl", None, a.workers) for label, (d, _) in methods.items()}
    C.aligned(*(rooms for _, rooms, _ in collected.values()))
    rooms = {label: c[1] for label, c in collected.items()}
    n = len(next(iter(rooms.values())))
    common = [i for i in range(n) if all(rooms[label][i]["layout"] for label in rooms)]
    scenarios = {"common": (common, COMMON, False), "failures_as_failures": (list(range(n)), FAILURES, True)}
    result = {"meta": {"methods": {label: {**collected[label][0], "summary": str(s) if s else None}
                                   for label, (_, s) in methods.items()},
                       "references": references, "boot": a.boot, "seed": a.seed, "difference": "method minus reference",
                       "fallback": "no layout -> room centre position, category-median size (leave-one-out over these rows), "
                                   "uniform-random yaw expectation scored like the method's yaw error (evaluate._uniform_yaw_error), BEV IoU 0, clean room 0",
                       "selection_cohort_note": COHORT_NOTE},
              "checks": {label: c[2]["reproduces_report_exactly"] for label, c in collected.items()},
              "rows": n, "common_rooms": len(common),
              "layouts": {label: sum(r["layout"] for r in rs) for label, rs in rooms.items()},
              "cost": [cost(label, rooms[label], d, s) for label, (d, s) in methods.items()], "scenarios": {}}
    scene_ids = {r["scene_id"] for r in next(iter(rooms.values()))}
    result["selection_cohort"] = []
    for path in a.selection_cohort:
        with open(path) as stream:
            cohort = {json.loads(line)["provenance"].get("scene_id") for line in stream if line.strip()}
        result["selection_cohort"].append({"file": str(path.resolve()), "cohort_rows": len(cohort),
                                           "rows_in_cohort": len(scene_ids & cohort), "rows": len(scene_ids)})
    for name, (index, metrics, fallback) in scenarios.items():
        pick = {label: [rs[i] for i in index] for label, rs in rooms.items()}
        result["scenarios"][name] = {
            "rooms": len(index),
            "methods": {label: pooled(pick[label], metrics, boot=a.boot, seed=a.seed, fallback=fallback) for label in rooms},
            "pairs": {f"{label} vs {ref}": C.paired(pick[label], pick[ref], metrics, boot=a.boot, seed=a.seed, fallback=fallback)
                      for ref in references for label in rooms if label not in references}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("rows", "common_rooms", "layouts", "checks", "selection_cohort")}, indent=1))
    for name, scenario in result["scenarios"].items():
        print(f"== {name}: {scenario['rooms']} rooms")
        for label, stats in scenario["methods"].items():
            print(f"  {label:28s}", " ".join(f"{m}={v['mean']:.3f}" for m, v in stats.items()))


if __name__ == "__main__":
    main()
