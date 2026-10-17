"""Anchor oracle probe: does a pose-conditioned model use other furniture when it is given?

Each validation room is predicted twice by the same checkpoint (raw argmax decode, no spread):
A reveals nothing; B reveals the reference poses of its anchors (bed, sofa, desk, dining table,
TV stand, ...: ``is_anchor``). Only the other fully labelled requests ("dependents") are scored,
with the anchors at their reference poses in both layouts:

  position_m        XY distance of the decoded bottom centre to its (matched) reference, m
  position_cell     grid cell cross entropy (grid_residual heads)
  anchor_distance_m |distance to the nearest anchor - the reference's distance to it|, m: a
                    relation proxy that needs no mined templates
  overlap           floor-standing dependent whose footprint has BEV IoU > 0.3 with another
                    floor-standing object
  outside           dependent footprint more than FOOTPRINT_VIOLATION_FRACTION outside the floor

Rooms without an anchor or a dependent are skipped and counted. Differences B - A carry a 95%
paired bootstrap interval over rooms (rooms of one house are not independent: indicative only).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from fastfill.v2.batch import collate_samples, load_tokenizer
from fastfill.v2.boxes import bev_iou
from fastfill.v2.cell_costs import FLOOR_TOLERANCE_M
from fastfill.v2.evaluate import FOOTPRINT_VIOLATION_FRACTION
from fastfill.v2.geometry import decode_yaw
from fastfill.v2.io import load_checkpoint_config, read_samples, to_device
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.model import load_model, model_inputs
from fastfill.v2.validation import footprint

# The keyword must be the head noun (last word): "desk lamp", "bed pillow" and "desk chair" are not anchors.
ANCHOR = re.compile(r"\b(bed|sofa|couch|desk|table|tv stand|tv cabinet)$")
NOT_ANCHOR = re.compile(r"\b(bedside|nightstand|side|end|coffee|console|dressing)\b")
EXCLUDE_FLAGS = {"oob_objects", "fixed_collision", "overlapping_furniture"}
METRICS = ("position_m", "position_cell", "anchor_distance_m", "overlap", "outside")


def is_anchor(category):
    category = category.lower().replace("_", " ")
    return bool(ANCHOR.search(category)) and not NOT_ANCHOR.search(category)


def _boxes(prediction, batch, assignment):
    """World (bottom centre, size, yaw) of predictions per slot, and of the references each slot is matched to."""
    scale, origin = batch["scale"][0], batch["origin"][0]
    predicted = (prediction["position_normalized"][0] * scale + origin, prediction["size"][0],
                 decode_yaw(prediction["yaw_logits"][0], prediction["yaw_residuals"][0]))
    index = assignment[0]
    targets = batch["targets"]
    reference = (targets["position_normalized"][0][index] * scale + origin, targets["size"][0][index], targets["yaw"][0][index])
    return predicted, reference


def _room_metrics(prediction, batch, result, anchors, dependents, floor):
    predicted, reference = _boxes(prediction, batch, result["assignment"])
    anchor_xy = reference[0][anchors, :2]
    rows = {key: [] for key in METRICS}
    world = {i: (reference[0][i], reference[1][i], reference[2][i]) for i in anchors}  # anchors at reference poses
    world.update({i: (predicted[0][i], predicted[1][i], predicted[2][i]) for i in dependents})
    on_floor = {i for i in world if abs(float(reference[0][i][2]) - floor) <= FLOOR_TOLERANCE_M}
    polygon = _floor_polygon(batch)
    for i in dependents:
        rows["position_m"].append(float((predicted[0][i, :2] - reference[0][i, :2]).norm()))
        nearest = int((anchor_xy - reference[0][i, :2]).norm(dim=-1).argmin())
        rows["anchor_distance_m"].append(abs(float((predicted[0][i, :2] - anchor_xy[nearest]).norm()) -
                                             float((reference[0][i, :2] - anchor_xy[nearest]).norm())))
        shape = footprint({"_pos": world[i][0].tolist(), "_size": world[i][1].tolist(), "_yaw": float(world[i][2])})
        rows["outside"].append(float(shape.area > 0 and shape.difference(polygon).area > FOOTPRINT_VIOLATION_FRACTION * shape.area))
        if i in on_floor:
            rows["overlap"].append(float(any(float(bev_iou(*world[i], *world[j])) > .3 for j in on_floor if j != i)))
    if "position_cell" in result["term_sums"] and result["term_counts"]["position_cell"]:
        rows["position_cell"].append(float(result["term_sums"]["position_cell"]) / result["term_counts"]["position_cell"])
    return {key: (float(np.mean(values)) if values else None) for key, values in rows.items()}


def _floor_polygon(batch):
    from shapely.geometry import Polygon
    return Polygon(batch["conditions"][0]["room"]["floor_polygon_xy_m"])


def _bootstrap(differences, rounds=2000, seed=0):
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (rounds, len(values)))].mean(1)
    return float(values.mean()), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


@torch.no_grad()
def probe(checkpoint, data, *, device="cpu", max_rows=None):
    model = load_model(checkpoint, device=device).eval()
    if not model.config.pose_conditioning:
        raise ValueError("the anchor probe needs a checkpoint trained with model.pose_conditioning")
    tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(Path(checkpoint).parent / "tokenizer"),
                               local_files_only=True)
    max_length = load_checkpoint_config(checkpoint)["max_length"] or 8192
    criterion = GeometryCriterion(LossConfig())
    rooms, skipped = [], {"flagged": 0, "no_anchor": 0, "no_dependent": 0, "rejected": 0}
    for sample in read_samples(data)[:max_rows]:
        if EXCLUDE_FLAGS & {k for k, v in sample.get("provenance", {}).get("legacy_flags", {}).items() if v}:
            skipped["flagged"] += 1
            continue
        try:
            plain = collate_samples([sample], tokenizer, max_length=max_length, max_objects=model.config.max_objects)
        except ValueError:
            skipped["rejected"] += 1
            continue
        labelled = (plain["validity"]["position"].all(-1) & plain["validity"]["size"].all(-1) & plain["validity"]["yaw"])[0]
        objects = sample["condition"]["objects"]
        anchors = [i for i, obj in enumerate(objects) if labelled[i] and is_anchor(obj["category"])]
        dependents = [i for i in range(len(objects)) if labelled[i] and i not in anchors]
        if not anchors or not dependents:
            skipped["no_anchor" if not anchors else "no_dependent"] += 1
            continue
        given = to_device(collate_samples([{**sample, "_given_pose_ids": [objects[i]["id"] for i in anchors]}], tokenizer,
                                          max_length=max_length, max_objects=model.config.max_objects), device)
        floor = float(sample["condition"]["room"].get("floor_z_m") or 0.)
        room = {"scene": sample.get("provenance", {}).get("scene_id"), "anchors": len(anchors), "dependents": len(dependents)}
        for name, inputs in (("A", {k: v for k, v in model_inputs(given).items() if not k.startswith("given_")}),
                             ("B", model_inputs(given))):
            prediction = model(**inputs)
            result = criterion(prediction, given, global_counts=False)  # anchors leave supervision in both
            room[name] = _room_metrics({k: v.float().cpu() if torch.is_tensor(v) else v for k, v in prediction.items()},
                                       to_device(given, "cpu"), {**result, "assignment": result["assignment"].cpu()},
                                       anchors, dependents, floor)
        rooms.append(room)
    summary = {}
    for key in METRICS:
        pairs = [(r["A"][key], r["B"][key]) for r in rooms if r["A"][key] is not None and r["B"][key] is not None]
        if not pairs:
            continue
        a, b = map(np.asarray, zip(*pairs))
        mean, interval = _bootstrap(b - a)
        summary[key] = {"A_nothing_given": float(a.mean()), "B_anchors_given": float(b.mean()),
                        "B_minus_A": mean, "B_minus_A_95ci": interval, "rooms": len(pairs)}
    return {"checkpoint": str(checkpoint), "data": str(data), "rooms": len(rooms), "skipped": skipped,
            "decode": "raw argmax (no spread)", "summary": summary, "per_room": rooms}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="report JSON path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-rows", type=int)
    a = p.parse_args(argv)
    report = probe(a.checkpoint, a.data, device=a.device, max_rows=a.max_rows)
    Path(a.output).write_text(json.dumps(report, indent=2) + "\n")
    print(f"rooms {report['rooms']}  skipped {report['skipped']}  ({report['decode']})")
    print(f"{'metric':<20}{'A nothing':>12}{'B anchors':>12}{'B-A':>10}   95% CI")
    for key, row in report["summary"].items():
        lo, hi = row["B_minus_A_95ci"]
        print(f"{key:<20}{row['A_nothing_given']:>12.4f}{row['B_anchors_given']:>12.4f}{row['B_minus_A']:>10.4f}   [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
