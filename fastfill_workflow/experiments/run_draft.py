"""Run the draft workflow over a rows file: ① LLM plan, ② FastFill size, ③ LLM draft, ④ local repair (⑤ off).

LLM calls go to the official OpenAI API only (settings in ``.env`` beside ``workflow/``; ``workflow/credentials.py``).
Step ② either runs a FastFill checkpoint in this process (``--checkpoint``, GPU recommended) or reads sizes from a
predictions file aligned with the rows (``--sizes-file``, e.g. ``remask_ablation``'s output).

Before anything is loaded a free preflight (GET /v1/models/<model>) checks network, key and model, so a node without
internet access fails in seconds, not after the checkpoint load. One room at a time; resumable (rooms with a saved
``room_XXX.json`` are skipped). Budget: every room's cost (official prices) goes to ``ledger.json``, and the run stops
before a room when spent + ``--reserve-usd`` would pass ``--budget-usd``; a rejected key or exhausted quota stops it
at once. At the end: ``rows.jsonl`` / ``predictions.jsonl`` (the rooms run, aligned, ``layout`` null for a failure)
for ``fastfill.v2.evaluate``, and ``summary.json``; ``--evaluate`` also runs the evaluator into ``eval/``.

    python fastfill_workflow/experiments/run_draft.py --rows fastfill/v2/ops/test300/rows.jsonl \\
        --checkpoint runs/main4-cell05-v4fixed-e5-base/model-step-6552 --device cuda \\
        --out runs/draft-workflow/test300 --budget-usd 55 --evaluate
"""
import argparse, json, random, shutil, subprocess, sys, time, traceback, urllib.error, urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # the workflow folder (holds workflow/, size_catalog.json, .env)
sys.path.insert(0, str(HERE))
from workflow.contract import _main as FASTFILL_ROOT  # noqa: E402  (also puts the fastfill package on sys.path)
from workflow.credentials import FatalAPIError, official_api  # noqa: E402
from workflow.pipeline import run_draft_workflow  # noqa: E402


def preflight(settings: dict) -> None:
    request = urllib.request.Request(f"{settings['base_url']}/models/{settings['model']}",
                                     headers={"Authorization": f"Bearer {settings['api_key']}",
                                              "User-Agent": "fastfill-workflow/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"preflight: OpenAI answered HTTP {exc.code} for model {settings['model']} "
                         f"(key or model access): {exc.read()[:200]!r}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"preflight: cannot reach {settings['base_url']} from this node ({exc}); run where there is "
                         "internet access, or make sizes on a GPU node first and pass --sizes-file")


def pick(rows: list[dict], a) -> list[int]:
    if a.indices:
        return a.indices
    if a.per_bin:
        rng, picked = random.Random(0), []
        for lo, hi in [(1, 5), (6, 10), (11, 20), (21, 40), (41, 10 ** 6)]:
            pool = [i for i, r in enumerate(rows) if lo <= len(r["condition"]["objects"]) <= hi]
            picked += sorted(rng.sample(pool, min(a.per_bin, len(pool))))
        return picked
    return list(range(len(rows)))[:a.limit]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="FastFill model dir (RUN/model-step-N, tokenizer beside it)")
    source.add_argument("--sizes-file", type=Path, help="predictions jsonl aligned with --rows (sizes are read)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--reserve-usd", type=float, default=0.6, help="kept free for the next room (a large one ~0.4)")
    ap.add_argument("--indices", type=int, nargs="+")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per-bin", type=int, help="this many rooms per object-count bin (seed 0) instead of all")
    ap.add_argument("--catalog", type=Path, default=HERE / "size_catalog.json")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()

    settings = official_api()
    preflight(settings)
    print(f"model {settings['model']} at {settings['base_url']}: reachable", flush=True)
    rows = [json.loads(l) for l in a.rows.read_text().splitlines() if l.strip()]
    picked = pick(rows, a)
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
    (a.out / "config.json").write_text(json.dumps({
        "rows": str(a.rows), "model": settings["model"], "base_url": settings["base_url"],
        "sizes": sizer.describe() if sizer else {"file": str(a.sizes_file)}, "budget_usd": a.budget_usd,
        "rooms": len(picked)}, indent=1))

    ledger_path = a.out / "ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"rooms": {}, "spent_usd": 0.0}
    for i in picked:
        path = a.out / f"room_{i:03d}.json"
        if path.exists():
            continue
        if ledger["spent_usd"] + a.reserve_usd > a.budget_usd:
            print(f"budget stop: spent ${ledger['spent_usd']:.2f} of ${a.budget_usd:.2f}", flush=True)
            break
        sizes = None
        if file_sizes is not None and file_sizes[i] is not None:
            sizes = {b["id"]: b["target_size_local_m"] for b in file_sizes[i]["objects"]}
        start = time.perf_counter()
        try:
            result = run_draft_workflow(rows[i]["condition"], sizes, catalog_path=str(a.catalog), polish=False,
                                        model=settings["model"], base_url=settings["base_url"],
                                        api_key=settings["api_key"], sizer=sizer)
        except FatalAPIError as exc:
            print(f"stopped: {exc}", flush=True)
            break
        except Exception as exc:
            result = {"status": "error", "layout": None, "error": f"{type(exc).__name__}: {exc}"[:500],
                      "traceback": traceback.format_exc()[-2000:]}
        d = result.get("diagnostics") or {}
        calls = d.get("plan_llm", {}).get("calls", []) + d.get("draft", {}).get("calls", [])
        cost = sum(c.get("cost_usd") or 0 for c in calls)
        usage = [c.get("usage") or {} for c in calls]
        tokens = {"prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                  "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                  "reasoning_tokens": sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
                                          for u in usage)}
        result.update(room=i, wall_s=round(time.perf_counter() - start, 2), cost_usd=round(cost, 4), tokens=tokens)
        path.write_text(json.dumps(result, ensure_ascii=False))
        ledger["rooms"][str(i)] = {"cost_usd": round(cost, 4), **tokens, "calls": len(calls)}
        ledger["spent_usd"] = round(ledger["spent_usd"] + cost, 4)
        ledger_path.write_text(json.dumps(ledger, indent=1))
        print(f"room {i:3d} objects {len(rows[i]['condition']['objects']):3d} {result['status']:12s} "
              f"{result['wall_s']:6.1f}s ${cost:.3f} (total ${ledger['spent_usd']:.2f}) plan {d.get('plan_source')} "
              f"sizes {d.get('size_source')} {d.get('size_error') or ''}", flush=True)

    done = [i for i in picked if (a.out / f"room_{i:03d}.json").exists()]
    results = {i: json.loads((a.out / f"room_{i:03d}.json").read_text()) for i in done}
    (a.out / "rows.jsonl").write_text("".join(json.dumps(rows[i]) + "\n" for i in done))
    (a.out / "predictions.jsonl").write_text("".join(json.dumps({"layout": results[i].get("layout")}) + "\n"
                                                     for i in done))
    walls = sorted(r["wall_s"] for r in results.values())
    summary = {"rooms_picked": len(picked), "rooms_done": len(done),
               "status": dict(Counter(r["status"] for r in results.values())),
               "spent_usd": ledger["spent_usd"],
               "tokens": {k: sum(r.get("tokens", {}).get(k, 0) for r in results.values())
                          for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")},
               "wall_s": {"median": walls[len(walls) // 2] if walls else None, "max": walls[-1] if walls else None,
                          "sum": round(sum(walls), 1)},
               "size_fallback_rooms": sum(1 for r in results.values() if (r.get("diagnostics") or {}).get("size_error"))}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("summary:", json.dumps(summary), flush=True)
    if a.evaluate and done:
        shutil.rmtree(a.out / "eval", ignore_errors=True)  # evaluate refuses an existing output
        log = open(a.out / "eval.log", "w")
        code = subprocess.call([sys.executable, "-m", "fastfill.v2.evaluate", "--data", str((a.out / "rows.jsonl").resolve()),
                                "--predictions", str((a.out / "predictions.jsonl").resolve()),
                                "--output", str((a.out / "eval").resolve())], cwd=FASTFILL_ROOT, stdout=log, stderr=log)
        print(f"evaluate exit {code}; report: {a.out / 'eval' / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
