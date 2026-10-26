"""CLI for catalog construction and one-room or JSONL workflow runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import build_catalog
from .export import roomgenbench_scene
from .pipeline import run_workflow


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m workflow")
    command = parser.add_subparsers(dest="command", required=True)
    catalog = command.add_parser("catalog", help="Build train-only category size priors")
    catalog.add_argument("--train", required=True)
    catalog.add_argument("--output", required=True)
    run = command.add_parser("run", help="Plan and solve one condition JSON")
    run.add_argument("--input", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--catalog")
    run.add_argument("--no-llm", action="store_true", help="Rule plan only (ablation); default: one LLM planning call")
    run.add_argument("--model")
    run.add_argument("--base-url")
    run.add_argument("--beam-width", type=int, default=32)
    run.add_argument("--candidate-limit", type=int, default=48)
    run.add_argument("--repair-rounds", type=int, default=2)
    run.add_argument("--roomgenbench-output")
    run.add_argument("--draft-layout", help="Optional model-generated layout used only as a proposer")
    run.add_argument("--allow-thin-overlap", action="store_true")
    batch = command.add_parser("batch", help="Generate one prediction per evaluation JSONL row")
    batch.add_argument("--data", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--catalog")
    batch.add_argument("--limit", type=int)
    batch.add_argument("--no-llm", action="store_true", help="Rule plan only (ablation); default: one LLM planning call")
    batch.add_argument("--model")
    batch.add_argument("--base-url")
    batch.add_argument("--beam-width", type=int, default=32)
    batch.add_argument("--candidate-limit", type=int, default=48)
    batch.add_argument("--allow-thin-overlap", action="store_true")
    args = parser.parse_args()
    if args.command == "catalog":
        result = build_catalog(args.train, args.output)
        print(f"Wrote {len(result['categories'])} categories to {args.output}")
        return
    if args.command == "batch":
        counts = {"ok": 0, "failed": 0}
        with Path(args.data).open(encoding="utf-8") as source, Path(args.output).open("w", encoding="utf-8") as output:
            for index, line in enumerate(source):
                if args.limit is not None and index >= args.limit:
                    break
                try:
                    condition = json.loads(line)["condition"]
                    result = run_workflow(condition, catalog_path=args.catalog, llm=not args.no_llm,
                                          model=args.model, base_url=args.base_url,
                                          beam_width=args.beam_width, candidate_limit=args.candidate_limit,
                                          allow_thin_overlap=args.allow_thin_overlap)
                    counts["ok" if result["status"] == "ok" else "failed"] += 1
                    record = {"layout": result["layout"], "status": result["status"],
                              "timing_s": result["timing_s"],
                              "plan_source": result["diagnostics"]["plan_source"],
                              "llm": result["diagnostics"]["llm"]}
                except (ValueError, TypeError, KeyError) as exc:
                    counts["failed"] += 1
                    record = {"layout": None, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(counts))
        return
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    condition = data.get("condition", data)
    draft = None
    if args.draft_layout:
        draft_data = json.loads(Path(args.draft_layout).read_text(encoding="utf-8"))
        draft = draft_data.get("layout", draft_data)
    result = run_workflow(condition, catalog_path=args.catalog, llm=not args.no_llm,
                          model=args.model, base_url=args.base_url,
                          draft_layout=draft,
                          beam_width=args.beam_width, candidate_limit=args.candidate_limit,
                          repair_rounds=args.repair_rounds,
                          allow_thin_overlap=args.allow_thin_overlap)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.roomgenbench_output and result["layout"] is not None:
        scene = roomgenbench_scene(condition, result["layout"])
        Path(args.roomgenbench_output).write_text(json.dumps(scene, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"status": result["status"], "timing_s": result["timing_s"],
                      "objects": len(result["layout"]["objects"]) if result["layout"] else 0}))


if __name__ == "__main__":
    main()
