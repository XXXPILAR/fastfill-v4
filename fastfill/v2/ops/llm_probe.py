"""Pure-LLM probe: one GPT request per room, full condition (polygon, doors/windows, constraints), LLM chooses sizes.

No tidy, no repair: the raw first answer. Uses llm_workflow's prompt (instructions + request_text) and
llm_structured.chat; credentials are read from the env file by load_env and never printed. A rerun keeps the rooms
already answered without an error (room_<k>.json) and asks only the others. predictions.jsonl is in row order, for
``evaluate --predictions``.

    python -m fastfill.v2.ops.llm_probe --rows data/test300/rows.jsonl --env ~/.fastfill_api.env --output DIR [--limit 1]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from fastfill.v2.llm_baseline import RateLimiter, load_env
from fastfill.v2.llm_structured import chat
from fastfill.v2.llm_workflow import instructions, parse, request_text, workflow_problems
from fastfill.v2.schema import migrate_legacy_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=int, nargs="*", help="rerun these room indices, keep the other room files")
    a = p.parse_args()
    env = load_env(a.env.expanduser())
    model = env["OPENAI_MODEL"]
    limiter = RateLimiter(float(env.get("FASTFILL_LLM_RPM", 30)))
    samples = [migrate_legacy_row(json.loads(l)) for l in a.rows.read_text().splitlines()][:a.limit]
    a.output.mkdir(parents=True, exist_ok=True)

    def one(k):
        sample = samples[k]
        condition = sample["condition"]
        messages = [{"role": "system", "content": instructions(condition, False)},
                    {"role": "user", "content": request_text(condition, None)}]
        usage, start, reply, error, layout, problems = [], time.perf_counter(), None, None, None, None
        try:
            reply = chat(env, model, messages, limiter=limiter, usage=usage)
            layout = parse(reply, condition, None)
            problems = workflow_problems(layout, condition)
        except Exception as e:  # keep the raw reply for inspection
            error = f"{type(e).__name__}: {e}"
        out = {"k": k, "source": sample.get("provenance", {}).get("source"), "room_type": condition["room"].get("room_type"),
               "latency_s": time.perf_counter() - start, "usage": usage, "error": error, "problems": problems,
               "layout": layout, "reply": reply, "request": messages[1]["content"]}
        (a.output / f"room_{k:03d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
        print(k, out["source"], out["room_type"], f"{out['latency_s']:.0f}s", "error" if error else f"{len(problems or [])} problems", flush=True)
        return out

    with ThreadPoolExecutor(a.workers) as pool:
        done = lambda k: (a.output / f"room_{k:03d}.json").is_file() and not json.loads((a.output / f"room_{k:03d}.json").read_text())["error"]
        todo = a.only if a.only else [k for k in range(len(samples)) if not done(k)]
        list(pool.map(one, todo))
    results = [json.loads((a.output / f"room_{k:03d}.json").read_text()) for k in range(len(samples))]
    with (a.output / "predictions.jsonl").open("w") as f:
        f.writelines(json.dumps({"layout": r["layout"], "error": r["error"], "latency_s": r["latency_s"]}) + "\n" for r in results)
    print(json.dumps({"model": model, "rooms": len(results), "errors": sum(bool(r["error"]) for r in results)}))


if __name__ == "__main__":
    main()
