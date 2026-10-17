"""Direct bbox checkpoint inference; the legacy asset loop is explicitly optional.

The checkpoint binds its own settings: ``load_model`` applies ``model_config.json``
(position head and grid, size reference, yaw bins, ...) and ``--max-length``
defaults to the training ``max_length`` recorded beside the export
(``io.load_checkpoint_config``); only a model directory with no run record falls
back to 4096. An explicit ``--max-length`` always wins.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from fastfill.v2.batch import load_tokenizer
from fastfill.v2.evaluate import DEFAULT_MAX_LENGTH, GRID_DECODES, predict_layout
from fastfill.v2.io import load_checkpoint_config, safe_output
from fastfill.v2.model import load_model
from fastfill.v2.schema import validate_condition, validate_layout


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--condition", type=Path)
    inputs.add_argument("--request", type=Path, help="room_type + room_size_m + furniture_list; no assets")
    p.add_argument("--room-size-semantics", choices=["rectangular", "reference_extent"], default="rectangular",
                   help="For --request: known rectangular room, or a source XY reference range with unknown physical boundary/floor")
    p.add_argument("--output", type=Path, required=True, help="New JSON file, never overwrite input")
    p.add_argument("--baseline", choices=["structured", "text"], default="structured")
    p.add_argument("--catalog", type=Path)
    p.add_argument("--export-dir", type=Path, help="New bbox JSON/GLB/SVG handoff directory; no assets")
    p.add_argument("--device", default="cpu")
    p.add_argument("--grid-decode", choices=GRID_DECODES, default="spread",
                   help="grid position head: collision-aware spread (default) or plain per-object argmax")
    p.add_argument("--max-length", type=int, help="default: the checkpoint's training max_length")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-asset-retries", type=int, default=2)
    p.add_argument("--max-repair-calls", type=int, default=0)
    p.add_argument("--repair-step-m", type=float, default=.25)
    p.add_argument("--max-seconds", type=float, default=10.)
    p.add_argument("--commit-in-memory", action="store_true")
    args = p.parse_args(argv)
    if args.condition and args.room_size_semantics != "rectangular":
        raise ValueError("--room-size-semantics applies only to --request; --condition declares its own room semantics")
    output = safe_output(args.output)
    if args.request and args.catalog:
        raise ValueError("direct furniture requests export bbox geometry and do not resolve assets")
    if args.export_dir and args.catalog:
        raise ValueError("bbox handoff exports raw predictions; omit --catalog")
    if args.export_dir:
        handoff = safe_output(args.export_dir)
        if output == handoff or output.is_relative_to(handoff) or handoff.is_relative_to(output):
            raise ValueError("--output and --export-dir must not overlap or contain each other")
    if args.request:
        from fastfill.v2.direct_layout import request_to_condition
        condition = request_to_condition(json.loads(args.request.read_text()),
                                         room_size_semantics=args.room_size_semantics)
    else:
        condition = json.loads(args.condition.read_text())
    validate_condition(condition)
    if args.commit_in_memory and args.catalog is None:
        raise ValueError("commit requires an actual asset catalog")
    if args.max_length is None:
        args.max_length = load_checkpoint_config(args.checkpoint)["max_length"] or DEFAULT_MAX_LENGTH
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))  # CPU inference may use more threads
    start = time.perf_counter()
    if args.baseline == "structured":
        model = load_model(args.checkpoint, device=args.device)
        tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(args.checkpoint.parent / "tokenizer"), local_files_only=True)
        inference_start = time.perf_counter()
        layout = predict_layout(model, tokenizer, condition, max_length=args.max_length, device=args.device,
                                grid_decode=args.grid_decode)
    else:
        from fastfill.v2.text_sft import generate_text, load_text_model
        model, tokenizer = load_text_model(args.checkpoint, device=args.device)
        inference_start = time.perf_counter()
        layout = json.loads(generate_text(model, tokenizer, condition, max_length=args.max_length,
                                         max_new_tokens=args.max_new_tokens, device=args.device))
        validate_layout(layout, condition)
    inference_ms = (time.perf_counter() - inference_start) * 1000
    result = layout
    if args.catalog:
        from fastfill.v2.runtime import AtomicMemoryHost, BoundedTranslationRepair, RuntimeBudget, run_pipeline
        from fastfill.v2.serve import load_catalog
        host = AtomicMemoryHost() if args.commit_in_memory else None
        result = run_pipeline(condition, layout, load_catalog(args.catalog), host=host,
            budget=RuntimeBudget(max_asset_retries=args.max_asset_retries, max_repair_calls=args.max_repair_calls, max_seconds=args.max_seconds),
            repair=BoundedTranslationRepair(args.repair_step_m) if args.max_repair_calls else None,
            expected_world_version=0 if host else None, idempotency_key="cli-prediction")
        result = {**result, "fastfill_latency_ms": inference_ms,
                  "cold_start_total_ms": (time.perf_counter() - start) * 1000}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if args.export_dir:
        from fastfill.v2.direct_layout import export_handoff
        export_handoff(args.export_dir, condition, layout)
    print(json.dumps({"output": str(output), "requested_objects": len(condition["objects"]),
                      "fastfill_latency_ms": inference_ms,
                      "runtime_ok": result["ok"] if args.catalog else None,
                      "committed": result.get("committed") if args.catalog else None}))
    return 0 if args.catalog is None or result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
