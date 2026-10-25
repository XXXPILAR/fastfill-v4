"""Compare ways of placing a room on the same rooms, plan and sizes (``workflow.orders``): A all at once, B one by one,
C group by group, and the ablations B_final / C_final. Official OpenAI API (settings in ``.env`` beside
``workflow/``), as ``run_draft.py``.

  A        one LLM call for the whole room, then the whole-room critic loop
  B / C    one LLM call per object / per functional group, each followed at once by a repair of only the new objects,
           which are then frozen (the full step-by-step workflows)
  B_final / C_final   the same step-by-step result (no new LLM placement calls) followed by a whole-room loop with A's
           budget: A vs B_final / C_final differs only in how the layout was generated, B vs B_final only in
           whether a final whole-room repair may move earlier objects (``--no-final-repair`` skips these)

Per room: step ② (FastFill sizes and yaws: ``--checkpoint`` on the GPU in this process, timed per room, or
``--sizes-file``), then in a worker process step ① once (the LLM plan, saved to <out>/plans/room_XXX.json with its
cost and seconds, reused by every mode), then each mode: <out>/<variant>/room_XXX.json.

Resuming: <out>/config.json holds the settings that change results (rows, model, size source, batch, GPT repairs,
search seconds, final repair, a hash of the workflow code and the critic templates). A rerun with any of them changed
stops: use another --out. Modes may be added, budgets and workers changed. A saved plan is reused, a saved variant
skipped. Budget as ``run_draft.py``: a room starts only when spent + ``--reserve-usd`` x modes per room in flight stays
within ``--budget-usd``.

Reported per variant (summary.json, quality.md): hard-valid rooms, hard violations and soft cost left, the number of
steps (B: objects / batch; C: groups, with the rooms where C had a single step, i.e. placed like A), LLM calls,
tokens, and cost and seconds twice: incremental (placement, repair, polish of this variant) and end-to-end (+ the
shared plan's cost and seconds, + the room's size-prediction seconds). The search budget is checked between repair
attempts, not a hard wall-clock limit: the seconds reported are measured. ``--quality`` adds the layout_quality
table (REL with ``--templates``, the odd half's templates by default), ``--evaluate`` runs fastfill.v2.evaluate.

    python fastfill_workflow/experiments/run_orders.py --rows fastfill/v2/ops/test300/rows.jsonl \\
        --rooms-file fastfill_workflow/experiments/test50_rooms.json \\
        --checkpoint runs/main4-cell05-v4fixed-e5-base/model-step-6552 --device cuda \\
        --out runs/draft-workflow/orders-test50 --budget-usd 60 --workers 8 --quality --evaluate
After a stop, ``--summary-only`` (no API, no model, no new rooms) writes the summary of the rooms already done.
Pilot first (``--indices ...`` or ``--limit-rooms 6``) to measure the cost per room; B makes one LLM call per object
(``--batch`` k places k objects per call).
"""
import argparse, hashlib, json, multiprocessing as mp, shutil, statistics, subprocess, sys, time, traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "experiments"))
from run_draft import pick, preflight  # noqa: E402
from workflow.contract import _main as FASTFILL_ROOT  # noqa: E402
from workflow.contract import check_request, fill_room_defaults  # noqa: E402
from workflow.credentials import FatalAPIError, official_api  # noqa: E402
from workflow.loop import LoopBudget  # noqa: E402
from workflow.orders import MODES, plan_from_dict, plan_to_dict, run_order_variants, variant_name  # noqa: E402
from workflow.planning import make_plan  # noqa: E402


def variants_of(mode: str, final: bool) -> list[str]:
    return [mode] + ([variant_name(mode, True)] if final and mode != "A" else [])


def usage_of(calls):
    calls = [c for c in calls if c.get("usage")]
    usage = [c["usage"] for c in calls]
    return (sum(c.get("cost_usd") or 0 for c in calls), len(calls),
            {"prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
             "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
             "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                                     for u in usage)})


def code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted((HERE / "workflow").glob("*.py")) + [HERE / "critic_templates.json", HERE / "size_catalog.json"]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def run_room(index, condition, sizes, yaws, size_error, size_s, catalog, settings, modes, out, gpt_repairs, batch,
             final, search_seconds):
    """One room in a worker: the plan (once), then each mode and its variants. Returns the new spending:
    {"plan": (usd, calls, tokens), mode: (usd, calls, tokens)} (a mode's LLM calls counted once for its variants)."""
    out = Path(out)
    spent = {}
    plan_path = out / "plans" / f"room_{index:03d}.json"
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        spent["plan"] = (0., 0, {})
    else:
        start = time.perf_counter()
        checked = check_request(fill_room_defaults(condition)[0])
        plan = make_plan(checked, llm=True, model=settings["model"], base_url=settings["base_url"],
                         api_key=settings["api_key"], semantic_only=True)
        spent["plan"] = usage_of(plan.llm.get("calls", []))
        saved = {"plan": plan_to_dict(plan), "seconds": round(time.perf_counter() - start, 2),
                 "cost_usd": round(spent["plan"][0], 4)}
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(saved, ensure_ascii=False))
    plan = plan_from_dict(saved["plan"])
    shared = {"plan_usd": saved["cost_usd"], "plan_s": saved["seconds"], "size_s": size_s}
    for mode in modes:
        names = variants_of(mode, final)
        if all((out / n / f"room_{index:03d}.json").exists() for n in names):
            continue
        try:
            results = run_order_variants(
                condition, sizes, plan, mode, yaws=yaws, catalog_path=catalog, model=settings["model"],
                base_url=settings["base_url"], api_key=settings["api_key"], gpt_repairs=gpt_repairs,
                loop_budget=LoopBudget(gpt_repairs=gpt_repairs, search_seconds=search_seconds), batch=batch,
                final_repair=(False, True) if len(names) > 1 else (False,))
        except FatalAPIError:
            raise
        except Exception as exc:  # noqa: BLE001
            error = {"status": "error", "layout": None, "error": f"{type(exc).__name__}: {exc}"[:500],
                     "traceback": traceback.format_exc()[-2000:]}
            results = {n: dict(error) for n in names}
        for name, result in results.items():
            d = result.get("diagnostics") or {}
            if size_error:
                d["size_error"] = size_error
            cost, ncalls, tokens = usage_of(d.get("draft_calls", []) + d.get("gpt_repair_calls", []))
            wall = (result.get("timing_s") or {}).get("total", 0.)
            result.update(room=index, wall_s=wall, cost_usd=round(cost, 4), tokens=tokens, llm_calls=ncalls,
                          e2e_usd=round(cost + shared["plan_usd"], 4),
                          e2e_s=round(wall + shared["plan_s"] + (size_s or 0.), 2), shared=shared)
            path = out / name / f"room_{index:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False))
        last = results[names[-1]].get("diagnostics") or {}  # the final variant's calls include the shared ones
        spent[mode] = usage_of(last.get("draft_calls", []) + last.get("gpt_repair_calls", []))
    return spent


def summarise(out, rows, done, names, a):
    med = lambda v: round(statistics.median(v), 1) if v else None  # noqa: E731
    lines = ["| variant | rooms | hard-valid | hard violations left | soft cost left | steps / room (median, max) | "
             "single-step rooms | LLM calls / room | USD / room (incremental) | USD / room (end-to-end) | "
             "s / room median (incremental) | s / room median (end-to-end) | s / room max (end-to-end) |",
             "|" + "---|" * 13]
    summary = {"variants": {}}
    for name in names:
        results = {i: json.loads((out / name / f"room_{i:03d}.json").read_text()) for i in done}
        (out / name / "rows.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in done))
        (out / name / "predictions.jsonl").write_text("".join(json.dumps({"layout": results[i].get("layout")}) + "\n"
                                                              for i in done))
        diags = [r.get("diagnostics") or {} for r in results.values()]
        finals = [d.get("final") or {} for d in diags]
        steps = [d.get("step_count") for d in diags if d.get("step_count")]
        n = max(len(done), 1)
        s = {"rooms": len(done), "status": dict(Counter(r["status"] for r in results.values())),
             "hard_valid": sum(f.get("hard_count") == 0 for f in finals),
             "hard_violations_left": sum(f.get("hard_count") or 0 for f in finals),
             "soft_cost_left": round(sum(f.get("soft_total") or 0 for f in finals), 3),
             "steps": {"median": med(steps), "max": max(steps) if steps else None,
                       "single_step_rooms": sum(x == 1 for x in steps)},
             "llm_calls": sum(r.get("llm_calls", 0) for r in results.values()),
             "usd_incremental": round(sum(r.get("cost_usd", 0) for r in results.values()), 4),
             "usd_end_to_end": round(sum(r.get("e2e_usd", 0) for r in results.values()), 4),
             "tokens": {k: sum(r.get("tokens", {}).get(k, 0) for r in results.values())
                        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
             "seconds_incremental": {"median": med([r["wall_s"] for r in results.values()]),
                                     "sum": round(sum(r["wall_s"] for r in results.values()), 1)},
             "seconds_end_to_end": {"median": med([r["e2e_s"] for r in results.values()]),
                                    "max": max((r["e2e_s"] for r in results.values()), default=None)}}
        summary["variants"][name] = s
        (out / name / "summary.json").write_text(json.dumps(s, indent=1))
        lines.append(f"| {name} | {len(done)} | {s['hard_valid']} | {s['hard_violations_left']} | {s['soft_cost_left']} | "
                     f"{s['steps']['median']}, {s['steps']['max']} | {s['steps']['single_step_rooms']} | "
                     f"{s['llm_calls'] / n:.1f} | {s['usd_incremental'] / n:.3f} | {s['usd_end_to_end'] / n:.3f} | "
                     f"{s['seconds_incremental']['median']} | {s['seconds_end_to_end']['median']} | "
                     f"{s['seconds_end_to_end']['max']} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    text = ("# Placement order: A all at once / B one by one / C group by group (+ _final: whole-room repair after)\n\n"
            f"Rooms done in every variant: {len(done)}. End-to-end adds the shared plan (cost, seconds) and the room's "
            "size-prediction seconds to each variant. B_final / C_final reuse B's / C's placement calls; their LLM "
            "calls and USD include them.\n\n" + "\n".join(lines) + "\n")
    if a.quality and done:
        sys.path.insert(0, str(FASTFILL_ROOT))
        from fastfill.v2.ops import layout_quality as lq
        from fastfill.v2.schema import migrate_legacy_row
        templates = json.loads(Path(a.templates).read_text())["templates"]
        for lo, hi, label in ((1, 10 ** 6, "all rooms"), (1, 20, "1-20 objects"), (21, 10 ** 6, "21+ objects")):
            chosen = [i for i in done if lo <= len(rows[i]["condition"]["objects"]) <= hi]
            if not chosen:
                continue
            text += (f"\n### {label} ({len(chosen)} rooms)\n\n| variant | hard-valid | NAV ↑ | ACC ↑ | REL ↑ | SG ↑ | "
                     "COL ↓ | OPC ↓ |\n|---|---|---|---|---|---|---|---|\n")
            mig = [migrate_legacy_row(rows[i]) for i in chosen]
            loaded = {name: [json.loads((out / name / f"room_{i:03d}.json").read_text()) for i in chosen] for name in names}
            for name, layouts in [("ground truth", [r["target"] for r in mig])] + \
                    [(name, [r.get("layout") for r in loaded[name]]) for name in names]:
                q = lq.score_method(mig, layouts, templates)
                f = lambda v: "–" if v is None else f"{100 * v:.1f}"  # noqa: E731
                valid = "–" if name == "ground truth" else sum(
                    ((r.get("diagnostics") or {}).get("final") or {}).get("hard_count") == 0 for r in loaded[name])
                text += (f"| {name} | {valid} | {f(q['NAV'])} | {f(q['ACC'])} | {f(q['REL'])} | {f(q['SG'])} | "
                         f"{f(q['COL'])} | {f(q['OPC'])} |\n")
        text += f"\nREL templates: {a.templates}\n"
    (out / "quality.md").write_text(text)
    print(text, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="FastFill model dir (RUN/model-step-N, tokenizer beside it)")
    source.add_argument("--sizes-file", type=Path, help="predictions jsonl aligned with --rows (sizes and yaws read)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--reserve-usd", type=float, default=0.6, help="kept free per mode of a room in flight")
    ap.add_argument("--rooms-file", type=Path, help="JSON with an 'indices' list (e.g. experiments/test50_rooms.json)")
    ap.add_argument("--indices", type=int, nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--limit-rooms", type=int, help="only the first N picked rooms (a pilot)")
    ap.add_argument("--per-bin", type=int, help="this many rooms per object-count bin (seed 0) instead of all")
    ap.add_argument("--catalog", type=Path, default=HERE / "size_catalog.json")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--batch", type=int, default=1, help="B: objects per LLM call (1: one at a time)")
    ap.add_argument("--gpt-repairs", type=int, default=2, help="GPT repair requests per room and variant's loop")
    ap.add_argument("--search-seconds", type=float, default=180., help="program search budget per loop budget")
    ap.add_argument("--no-final-repair", action="store_true", help="skip the B_final / C_final ablations")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--mp-start", default="spawn", help=argparse.SUPPRESS)
    ap.add_argument("--quality", action="store_true", help="layout_quality table per variant in <out>/quality.md")
    ap.add_argument("--templates", type=Path, default=HERE / "experiments" / "quality50" / "eval_templates_odd.json")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--summary-only", action="store_true",
                    help="no API, no model, no new rooms: summarise the rooms already done in --out (after a stop)")
    a = ap.parse_args()
    if a.rooms_file and not a.indices:  # explicit --indices (a pilot) win over the rooms file
        a.indices = json.loads(a.rooms_file.read_text())["indices"]
    final = not a.no_final_repair
    if a.summary_only:  # the run's own settings (config.json), whatever this command line says
        config = json.loads((a.out / "config.json").read_text())
        rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
        picked = json.loads((a.out / "rooms.json").read_text())
        final = config["fingerprint"]["final_repair"]
        names = [n for m in config["modes"] for n in variants_of(m, final)]
        done = [i for i in picked if all((a.out / n / f"room_{i:03d}.json").exists() for n in names)]
        print(f"{len(done)} of {len(picked)} rooms done in every variant {names}", flush=True)
        summarise(a.out, rows, done, names, a)
        return

    settings = official_api()
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    picked = pick(rows, a)[:a.limit_rooms] if a.limit_rooms else pick(rows, a)
    fingerprint = {"rows": str(a.rows.resolve()), "model": settings["model"],
                   "sizes": str(a.checkpoint.resolve()) if a.checkpoint else str(a.sizes_file.resolve()),
                   "batch": a.batch, "gpt_repairs": a.gpt_repairs, "search_seconds": a.search_seconds,
                   "final_repair": final, "catalog": str(a.catalog.resolve()), "code": code_hash()}
    config_path = a.out / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text()).get("fingerprint", {})
        changed = {k: (old.get(k), v) for k, v in fingerprint.items() if old.get(k) != v}
        if changed:
            raise SystemExit(f"{a.out} holds results made with other settings {changed}: use another --out")
    preflight(settings)
    print(f"model {settings['model']} at {settings['base_url']}: reachable", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rooms.json").write_text(json.dumps(picked))
    sizer, file_sizes = None, None
    if a.checkpoint:
        from workflow.fastfill_sizes import FastFillSizer
        sizer = FastFillSizer(a.checkpoint, device=a.device)
        print(f"FastFill loaded: {sizer.describe()}", flush=True)
    else:
        file_sizes = [json.loads(l)["layout"] for l in a.sizes_file.read_text().splitlines() if l.strip()]
        if len(file_sizes) != len(rows):
            raise SystemExit(f"--sizes-file has {len(file_sizes)} lines, --rows {len(rows)}")
    config_path.write_text(json.dumps({"fingerprint": fingerprint, "base_url": settings["base_url"],
                                       "sizes": sizer.describe() if sizer else {"file": str(a.sizes_file)},
                                       "modes": a.modes, "budget_usd": a.budget_usd, "rooms": len(picked)}, indent=1))

    names = [n for m in a.modes for n in variants_of(m, final)]
    ledger_path = a.out / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"rooms": {}, "spent_usd": 0.0}
    pending = [i for i in picked if not all((a.out / n / f"room_{i:03d}.json").exists() for n in names)]
    sizes, yaws, size_errors, size_s = {}, {}, {}, {}
    for i in pending:
        t0 = time.perf_counter()
        if sizer is not None:
            try:
                sizes[i], yaws[i] = sizer.predict(rows[i]["condition"])
            except ValueError as exc:  # over the model's budget: the catalog's sizes, no suggested yaws
                sizes[i], yaws[i], size_errors[i] = None, None, f"{type(exc).__name__}: {exc}"[:300]
            size_s[i] = round(time.perf_counter() - t0, 3)
        else:  # read from a file: the prediction's own time is not known here
            layout = file_sizes[i]
            sizes[i] = {b["id"]: b["target_size_local_m"] for b in layout["objects"]} if layout else None
            yaws[i] = {b["id"]: b["yaw_rad"] for b in layout["objects"]} if layout else None
            size_s[i] = None
    if sizer is not None:
        del sizer
    running, stop = {}, False
    reserve = a.reserve_usd * len(a.modes)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context(a.mp_start)) as pool:
        while pending or running:
            while pending and not stop and len(running) < a.workers:
                if ledger["spent_usd"] + (len(running) + 1) * reserve > a.budget_usd:
                    print(f"budget stop: spent ${ledger['spent_usd']:.2f} of ${a.budget_usd:.2f}", flush=True)
                    stop = True
                    break
                i = pending.pop(0)
                running[pool.submit(run_room, i, rows[i]["condition"], sizes[i], yaws[i], size_errors.get(i), size_s[i],
                                    str(a.catalog), settings, a.modes, str(a.out), a.gpt_repairs, a.batch, final,
                                    a.search_seconds)] = i
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                i = running.pop(future)
                try:
                    spent = future.result()
                except FatalAPIError as exc:
                    print(f"stopped: {exc} (no new rooms; {len(running)} still finishing)", flush=True)
                    stop = True
                    continue
                room = ledger["rooms"].setdefault(str(i), {})
                for part, (cost, ncalls, tokens) in spent.items():
                    if part == "plan" and not ncalls and "plan" in room:
                        continue  # a saved plan was reused: keep the cost recorded when it was made
                    room[part] = {"cost_usd": round(cost, 4), "calls": ncalls, **tokens}
                    ledger["spent_usd"] = round(ledger["spent_usd"] + cost, 4)
                ledger_path.write_text(json.dumps(ledger, indent=1))
                status = {n: json.loads((a.out / n / f"room_{i:03d}.json").read_text())["status"] for n in names
                          if (a.out / n / f"room_{i:03d}.json").exists()}
                print(f"room {i:3d} objects {len(rows[i]['condition']['objects']):3d} {status} "
                      f"${sum(c for c, _, _ in spent.values()):.3f} (total ${ledger['spent_usd']:.2f})", flush=True)

    done = [i for i in picked if all((a.out / n / f"room_{i:03d}.json").exists() for n in names)]
    summarise(a.out, rows, done, names, a)
    if a.evaluate and done:
        for name in names:
            shutil.rmtree(a.out / name / "eval", ignore_errors=True)
            with open(a.out / name / "eval.log", "w") as log:
                code = subprocess.call([sys.executable, "-m", "fastfill.v2.evaluate",
                                        "--data", str((a.out / name / "rows.jsonl").resolve()),
                                        "--predictions", str((a.out / name / "predictions.jsonl").resolve()),
                                        "--output", str((a.out / name / "eval").resolve())],
                                       cwd=FASTFILL_ROOT, stdout=log, stderr=log)
            print(f"{name}: evaluate exit {code}", flush=True)


if __name__ == "__main__":
    main()
