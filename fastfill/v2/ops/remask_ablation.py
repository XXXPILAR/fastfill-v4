"""One checkpoint, several decodes: one-shot argmax / spread against remasking decoding (``remask``).

Every mode decodes the same rows (full condition, as deployed) from the same checkpoint and is scored by
``evaluate --predictions``; the table compares the raw layouts:

    argmax        one forward pass, argmax cell (the raw head)
    spread        one forward pass, ``spread_grid_xy`` post-processing (the deployed decode)
    remaskR       R rounds of remasking decoding, argmax cells (needs pose_conditioning when R > 1)
    remaskRL      R rounds, cells restricted to the least-cost cells (outside / door zone / fixed objects /
                  committed footprints)
    remaskRLX     ... and SceneNAT retraction: low-confidence committed objects are masked again (``remask``)

``--strip-support`` removes every support declaration (``support_parent`` and ``on`` constraints) from the
conditions before decoding. ``support_accuracy`` scores a support-head model's predicted ``support_parent`` only
against declarations the decoded condition does not show (a shown one is copied, not predicted): without
``--strip-support`` there are none and it is None; ``declared_in_input`` counts the shown ones. Overall and for
floor / wall / object parents; None without a support head.

    python -m fastfill.v2.ops.remask_ablation --checkpoint RUN/model --data rows.jsonl --output DIR \\
        --modes argmax spread remask1L remask4 remask4L remask4LX [--max-samples N] [--strip-support] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

from fastfill.v2.batch import collate_samples, load_tokenizer
from fastfill.v2.evaluate import serialize_predictions
from fastfill.v2.io import load_checkpoint_config, to_device
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.remask import remask_predictions
from fastfill.v2.schema import migrate_legacy_row

MODE = re.compile(r"^(argmax|spread|remask(\d+)(L?)(X?))$")


def decode(model, batch, mode):
    m = MODE.match(mode)
    if not m:
        raise ValueError(f"unknown mode {mode}")
    if m.group(1) in ("argmax", "spread"):
        with torch.no_grad():
            return serialize_predictions(model(**model_inputs(batch)), batch, grid_decode=m.group(1))[0]
    out = remask_predictions(model, batch, rounds=int(m.group(2)), legal=bool(m.group(3)), retract=bool(m.group(4)))
    return serialize_predictions(out, batch, grid_decode="argmax")[0]


def strip_support(row: dict) -> dict:
    """The row with no support declaration left in its condition."""
    condition = {**row["condition"], "objects": [{k: v for k, v in obj.items() if k != "support_parent"}
                                                 for obj in row["condition"]["objects"]],
                 "constraints": [c for c in row["condition"].get("constraints", []) if c.get("type") != "on"]}
    return {**row, "condition": condition}


def support_labels(row: dict) -> dict:
    """id -> declared parent (support_parent or hard on), for the requests that declare one."""
    from fastfill.v2.validation import effective_support_requests
    return {r["id"]: r["support_parent"] for r in effective_support_requests(row["condition"])
            if r.get("support_parent") is not None}


def support_accuracy(predictions: list, labels: list, shown: list | None = None) -> dict | None:
    """Predicted support_parent against the declarations hidden from the decoded condition (``shown``: the
    declarations it still carries, per row): overall and by kind of declared parent."""
    hits = {"all": [], "floor": [], "wall": [], "object": []}
    visible = 0
    for k, (prediction, label) in enumerate(zip(predictions, labels)):
        objects = {o["id"]: o for o in (prediction["layout"] or {}).get("objects", [])}
        for ident, parent in label.items():
            if ident not in objects:
                continue
            if "support_parent" not in objects[ident]:
                return None
            if shown is not None and ident in shown[k]:
                visible += 1
                continue
            hit = objects[ident]["support_parent"] == parent
            hits["all"].append(hit)
            hits[parent if parent in ("floor", "wall") else "object"].append(hit)
    return {k: (sum(v) / len(v) if v else None) for k, v in hits.items()} | {"objects": len(hits["all"]),
                                                                          "declared_in_input": visible}


def summarize(report: dict) -> dict:
    collapse = report["collapse"]["predicted_matched"]
    checks = report["target_validation_checks"]
    reference = report["model"]["reference"]
    rate = lambda key: (checks[key]["pass"] / (checks[key]["pass"] + checks[key]["violation"])
                        if key in checks and checks[key]["pass"] + checks[key]["violation"] else None)
    return {"position_m": reference["bottom_center_error_m"]["mean"], "log_size": reference["log_size_error"]["mean"],
            "yaw_rad": reference["yaw_error_rad"]["mean"], "bev_iou": reference["bev_iou"]["mean"],
            "footprint_outside": collapse["footprint_outside_fraction"],
            "door_blocking": collapse["door_blocking_fraction_of_floor_objects"],
            "overlap_iou_0.3": collapse["bev_overlap_rate_iou_gt_0.3"], "central": collapse["central_quarter_fraction"],
            "collision_pairs_per_room": (report["validation"]["model"] or {}).get("collision_pairs", 0) / max(report["requests"], 1),
            "constraint_pass": rate("constraint"), "clean_room_rate": (report["validation"]["model"] or {}).get("clean_room_rate")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True, help="RUN/model (RUN/tokenizer beside it)")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", default=["argmax", "spread", "remask1L", "remask4", "remask4L", "remask4LX"])
    p.add_argument("--max-samples", type=int)
    p.add_argument("--strip-support", action="store_true", help="decode without any support declaration")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    model = load_model(a.checkpoint, device=a.device).eval()
    tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(a.checkpoint.parent / "tokenizer"),
                               local_files_only=True)
    max_length = load_checkpoint_config(a.checkpoint)["max_length"] or 4096
    lines = [l for l in a.data.read_text().splitlines() if l.strip()][:a.max_samples]
    labels = [support_labels(migrate_legacy_row(json.loads(l))) for l in lines]
    if a.strip_support:
        lines = [json.dumps(strip_support(migrate_legacy_row(json.loads(l)))) for l in lines]
    shown = [support_labels(migrate_legacy_row(json.loads(l))) for l in lines]  # what the decoded condition declares
    rows = a.output / "rows.jsonl"
    rows.write_text("\n".join(lines) + "\n")
    batches = [to_device(collate_samples([migrate_legacy_row(json.loads(l))], tokenizer, max_length=max_length,
                                         max_objects=model.config.max_objects), a.device) for l in lines]
    table = {}
    for mode in a.modes:
        predictions, seconds = [], []
        for batch in batches:
            start = time.perf_counter()
            try:
                layout, error = decode(model, batch, mode), None
            except ValueError as e:
                layout, error = None, str(e)
            seconds.append(time.perf_counter() - start)
            predictions.append({"layout": layout, "error": error, "latency_s": seconds[-1]})
        path = a.output / f"predictions_{mode}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in predictions))
        scored = a.output / f"eval_{mode}"
        subprocess.run([sys.executable, "-m", "fastfill.v2.evaluate", "--data", str(rows), "--predictions", str(path),
                        "--output", str(scored)], check=True, stdout=subprocess.DEVNULL)
        report = json.loads((scored / "report.json").read_text())
        table[mode] = {**summarize(report),
                       "decode_ms": 1000 * sum(seconds) / len(seconds), "failed": sum(r["layout"] is None for r in predictions),
                       "support_accuracy": support_accuracy(predictions, labels, shown)}
        print(mode, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in table[mode].items()}), flush=True)
        table[mode]["by_constraints"] = {name: {"requests": cohort["requests"], **summarize(cohort)}
                                         for name, cohort in report.get("by_constraints", {}).items()}
    ground_truth = json.loads((a.output / f"eval_{a.modes[0]}" / "report.json").read_text())["collapse"]["ground_truth"]
    table["ground_truth"] = {"footprint_outside": ground_truth["footprint_outside_fraction"],
                             "door_blocking": ground_truth["door_blocking_fraction_of_floor_objects"],
                             "overlap_iou_0.3": ground_truth["bev_overlap_rate_iou_gt_0.3"],
                             "central": ground_truth["central_quarter_fraction"]}
    (a.output / "ablation.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
