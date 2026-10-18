"""Token and cost ledger of LLM runs, from the per-room files they write (llm_probe / llm_hybrid room_<k>.json, each
with the API's ``usage`` of every request). Requests that died without a response (gateway timeouts, dropped
connections) returned no usage and are not in it: the provider may still bill them.

    python -m fastfill.v2.ops.llm_cost RUN_DIR [RUN_DIR ...] [--price-in USD_PER_1M --price-out USD_PER_1M] [--json FILE]

Prices are per million tokens; reasoning tokens are billed as output (they are part of completion_tokens). Without
prices only tokens are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def ledger(run):
    rooms = sorted(Path(run).glob("room_*.json"))
    t = {"run": str(run), "rooms": len(rooms), "rooms_with_error": 0, "requests": 0, "prompt_tokens": 0, "cached_tokens": 0,
         "completion_tokens": 0, "reasoning_tokens": 0, "latency_s": 0.}
    for path in rooms:
        r = json.loads(path.read_text())
        t["rooms_with_error"] += bool(r.get("error"))
        t["latency_s"] += r.get("latency_s") or 0.
        for u in r.get("usage") or []:
            t["requests"] += 1
            t["prompt_tokens"] += u.get("prompt_tokens") or 0
            t["completion_tokens"] += u.get("completion_tokens") or 0
            t["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            t["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    t["total_tokens"] = t["prompt_tokens"] + t["completion_tokens"]
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--price-in", type=float, help="USD per 1M input tokens")
    p.add_argument("--price-out", type=float, help="USD per 1M output tokens (reasoning included)")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    rows = [ledger(r) for r in a.runs]
    if a.price_in is not None and a.price_out is not None:
        for t in rows:
            t["cost_usd"] = round((t["prompt_tokens"] * a.price_in + t["completion_tokens"] * a.price_out) / 1e6, 2)
    head = f"{'run':42s} {'rooms':>5s} {'err':>4s} {'reqs':>5s} {'input':>10s} {'output':>10s} {'reasoning':>10s} {'total':>11s}"
    print(head + ("  cost_usd" if "cost_usd" in rows[0] else ""))
    for t in rows:
        print(f"{Path(t['run']).name[:42]:42s} {t['rooms']:5d} {t['rooms_with_error']:4d} {t['requests']:5d} {t['prompt_tokens']:10,d} "
              f"{t['completion_tokens']:10,d} {t['reasoning_tokens']:10,d} {t['total_tokens']:11,d}"
              + (f"  {t['cost_usd']:8.2f}" if "cost_usd" in t else ""))
    if len(rows) > 1:
        keys = ("rooms", "requests", "prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
        s = {k: sum(t[k] for t in rows) for k in keys}
        print(f"{'TOTAL':42s} {s['rooms']:5d} {'':4s} {s['requests']:5d} {s['prompt_tokens']:10,d} {s['completion_tokens']:10,d} "
              f"{s['reasoning_tokens']:10,d} {s['total_tokens']:11,d}"
              + (f"  {sum(t['cost_usd'] for t in rows):8.2f}" if "cost_usd" in rows[0] else ""))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
