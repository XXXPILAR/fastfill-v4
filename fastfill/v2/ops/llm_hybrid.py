"""FastFill + LLM on saved FastFill predictions (no checkpoint, no GPU: the model ran elsewhere), full condition.

    --method sizes   (method 1) FastFill's sizes are locked; the LLM places every object (x, y, z, facing_deg).
    --method repair  (method 2) the LLM gets FastFill's whole layout as a draft with its problems and moves as few
                     objects as it can; a draft without problems is kept as it is (no request, counted in ``kept``).

Every answer is parsed with the locked sizes (``llm_workflow.parse``), tidied (``tidy``: near-contact snaps, small
wall pushes) and checked with ``workflow_problems`` (the strict scorer's checks); the problems go back to the LLM up
to ``--repairs`` times and the layout with the fewest problems is kept, never the draft's when the draft has fewer
(method 2 compares with the draft too). No usable answer: FastFill's own layout (counted in ``fallback``), so every
room has a layout. Prompt: ``llm_workflow.instructions`` / ``request_text``; credentials from the env file
(``load_env``), never printed. A rerun keeps the rooms already done (room_<k>.json) and asks only the others.

    python -m fastfill.v2.ops.llm_hybrid --rows fastfill/v2/ops/test300/rows.jsonl \\
        --fastfill server_runs/test300-v4/test300_E/predictions_spread.jsonl --method sizes \\
        --env ~/.fastfill_api.env --output runs/hybrid/sizes-E
    python -m fastfill.v2.evaluate --data <rows> --predictions runs/hybrid/sizes-E/predictions.jsonl --output <new dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from fastfill.v2.llm_baseline import RateLimiter, load_env
from fastfill.v2.llm_structured import chat
from fastfill.v2.llm_workflow import SHOWN, instructions, parse, request_text, tidy, workflow_problems
from fastfill.v2.schema import migrate_legacy_row

DRAFT = ("\n\n[Draft Layout]\n{draft}\n[/Draft Layout]\n\nThe draft above already places every object with the given sizes. "
         "It has these problems:\n{problems}\n\nFix every problem. Keep each object that is not part of a problem exactly "
         "where it is; move as few objects as possible, by as little as possible, and keep functional groups together. "
         "Report the complete design (every id) in the answer format.")
AGAIN = ("Your layout has these problems:\n{problems}\n\nFix them, keep the other objects where they are, and report "
         "the complete design (every id) again in the same format.")
UNREADABLE = "Your answer could not be read ({error}). Answer again in the required format, with every id exactly once."


def listed(problems):
    shown = "\n".join(f"- {p}" for p in problems[:SHOWN])
    return shown + (f"\n- ... and {len(problems) - SHOWN} more" if len(problems) > SHOWN else "")


def draft_text(layout):
    return json.dumps({"objects": [{"id": o["id"], "x": round(o["bottom_center_m"][0], 3), "y": round(o["bottom_center_m"][1], 3),
                                    "z": round(o["bottom_center_m"][2], 3), "facing_deg": round(math.degrees(o["yaw_rad"]) % 360, 1)}
                                   for o in layout["objects"]]})


def checked(layout, condition):
    layout, moved = tidy(layout, condition)
    return layout, workflow_problems(layout, condition), moved


def solve(env, model, limiter, condition, fastfill, method, repairs):
    """(layout, record): the best checked layout and what happened."""
    sizes = {o["id"]: o["target_size_local_m"] for o in fastfill["objects"]}
    record = {"turns": [], "usage": [], "kept": False, "fallback": False}
    draft, draft_problems, _ = checked(fastfill, condition)
    record["draft_problems"] = len(draft_problems)
    if method == "repair" and not draft_problems:
        record["kept"] = True
        return draft, record
    user = request_text(condition, sizes)
    if method == "repair":
        user += DRAFT.format(draft=draft_text(draft), problems=listed(draft_problems))
    messages = [{"role": "system", "content": instructions(condition, True)}, {"role": "user", "content": user}]
    best = (draft, draft_problems) if method == "repair" else None
    for turn in range(repairs + 1):
        reply = chat(env, model, messages, limiter=limiter, usage=record["usage"])
        messages.append({"role": "assistant", "content": reply})
        try:
            layout, problems, moved = checked(parse(reply, condition, sizes), condition)
        except (ValueError, KeyError, TypeError) as error:
            record["turns"].append({"turn": turn, "error": f"{type(error).__name__}: {error}"})
            messages.append({"role": "user", "content": UNREADABLE.format(error=type(error).__name__)})
            continue
        record["turns"].append({"turn": turn, "problems": len(problems), "tidied": moved})
        if best is None or len(problems) < len(best[1]):
            best = (layout, problems)
        if not problems:
            break
        messages.append({"role": "user", "content": AGAIN.format(problems=listed(problems))})
    if best is None:
        record["fallback"] = True
        return fastfill, record
    record["final_problems"] = len(best[1])
    record["used_draft"] = best[0] is draft
    return best[0], record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--fastfill", type=Path, required=True, help="FastFill predictions.jsonl, one {layout} per row")
    p.add_argument("--method", choices=("sizes", "repair"), required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repairs", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--rpm", type=float, help="requests per minute for this process (default: FASTFILL_LLM_RPM or 30); "
                   "split the budget when several runs share one key")
    a = p.parse_args(argv)
    env = load_env(a.env.expanduser())
    model = env["OPENAI_MODEL"]
    limiter = RateLimiter(a.rpm or float(env.get("FASTFILL_LLM_RPM", 30)))
    rows = [migrate_legacy_row(json.loads(l)) for l in a.rows.read_text().splitlines() if l.strip()][:a.limit]
    drafts = [json.loads(l).get("layout") for l in a.fastfill.read_text().splitlines() if l.strip()][:len(rows)]
    if len(drafts) != len(rows):
        raise SystemExit(f"{a.fastfill}: {len(drafts)} predictions for {len(rows)} rows")
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "RUN.json").write_text(json.dumps({"method": a.method, "model": model, "repairs": a.repairs, "rows": str(a.rows),
        "fastfill": str(a.fastfill), "fastfill_sha256": hashlib.sha256(a.fastfill.read_bytes()).hexdigest(),
        "rows_sha256": hashlib.sha256(a.rows.read_bytes()).hexdigest()}, indent=1) + "\n")
    path = lambda k: a.output / f"room_{k:03d}.json"

    def one(k):
        condition, start, error, layout, record = rows[k]["condition"], time.perf_counter(), None, None, {}
        if drafts[k] is None:
            error = "no FastFill layout"
        else:
            try:
                layout, record = solve(env, model, limiter, condition, drafts[k], a.method, a.repairs)
            except Exception as e:  # network / gateway: FastFill's layout stands, the room is redone on a rerun
                error, layout, record = f"{type(e).__name__}: {e}", drafts[k], {"fallback": True}
        out = {"k": k, "source": rows[k]["provenance"].get("source"), "latency_s": time.perf_counter() - start,
               "error": error, "layout": layout, **record}
        path(k).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        print(k, out["source"], f"{out['latency_s']:.0f}s", error or ("kept" if record.get("kept") else
              f"draft {record.get('draft_problems')} -> {record.get('final_problems', 'fallback')} problems"), flush=True)

    done = lambda k: path(k).is_file() and not json.loads(path(k).read_text())["error"]
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, [k for k in range(len(rows)) if not done(k)]))
    results = [json.loads(path(k).read_text()) for k in range(len(rows))]
    with (a.output / "predictions.jsonl").open("w") as f:
        f.writelines(json.dumps({"layout": r["layout"], "error": r["error"], "latency_s": r["latency_s"]}) + "\n" for r in results)
    summary = {"method": a.method, "model": model, "rooms": len(results), "errors": sum(bool(r["error"]) for r in results),
               "kept_draft": sum(bool(r.get("kept")) for r in results), "fallback": sum(bool(r.get("fallback")) for r in results),
               "zero_problems": sum(r.get("final_problems", r.get("draft_problems") if r.get("kept") else None) == 0 for r in results),
               "requests": sum(len(r.get("turns") or []) for r in results),
               "median_latency_s": sorted(r["latency_s"] for r in results)[len(results) // 2]}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
