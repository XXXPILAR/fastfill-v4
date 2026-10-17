"""How the labels sit against the expected-geometry costs (``cell_costs``), before training with them.

For a random sample of rows: how many rooms carry doors; how often label floor objects cover a door's clearance
zone (per zone depth, per category and source); how often label footprints leave the floor polygon; and the cost of
each label's own cell, with the border-inclusive least-cost candidates and with cell centres only. A label cell that
is not free would make the loss fight the position cross-entropy.

    python -m fastfill.v2.ops.door_zone_stats --data ../data_v4_v2/train.jsonl ../data_v4_v2/validation.jsonl \\
        --rows 6000 2000 --output ../door_zone_stats.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import random
import time

import numpy as np

DEPTHS = (.6, .8)
FRACTION = .1


def _sample_lines(path: Path, rows: int, seed: int) -> list[str]:
    with path.open() as f:
        total = sum(1 for _ in f)
    keep = set(random.Random(seed).sample(range(total), min(rows, total)))
    with path.open() as f:
        return [line for i, line in enumerate(f) if i in keep]


def _room(line: str) -> dict:
    from shapely.geometry import Polygon
    import torch
    from fastfill.v2.batch import _geometry_rows, room_normalization
    from fastfill.v2.cell_costs import FLOOR_TOLERANCE_M, cell_cost_maps, door_zones
    from fastfill.v2.geometry import encode_grid_position
    from fastfill.v2.schema import migrate_legacy_row
    from fastfill.v2.validation import footprint

    sample = migrate_legacy_row(json.loads(line))
    room = sample["condition"]["room"]
    source = sample.get("provenance", {}).get("source", "?")
    origin, scale = room_normalization(room)
    geom = _geometry_rows(sample, origin, scale)
    objects = sample["condition"]["objects"]
    floor = Polygon(room["floor_polygon_xy_m"])
    floor_z = room.get("floor_z_m")
    zones = {d: [Polygon(z).intersection(floor) for z in door_zones(room, d)] for d in DEPTHS}
    out = {"source": source, "doors": len(zones[DEPTHS[0]]), "objects": []}
    boxes, cells = [], []
    for i, obj in enumerate(objects):
        complete = all(geom["position_valid"][i]) and all(geom["size_valid"][i][:2]) and geom["yaw_valid"][i]
        if not complete:
            boxes.append((None, None, False))
            cells.append(None)
            continue
        p = [v * s + o for v, s, o in zip(geom["position"][i], scale, origin)]
        size, yaw = geom["size"][i], geom["yaw"][i]
        on_floor = floor_z is not None and abs(p[2] - floor_z) <= FLOOR_TOLERANCE_M
        poly = footprint({"_pos": p, "_size": size, "_yaw": yaw})
        area = poly.area
        record = {"category": obj["category"], "floor": on_floor,
                  "outside": bool(area > 0 and poly.difference(floor).area > FRACTION * area)}
        for d in DEPTHS:
            blocked = sum(poly.intersection(z).area for z in zones[d]) if area > 0 else 0.
            record[f"door_{d}"] = bool(on_floor and blocked > FRACTION * area)
        out["objects"].append(record)
        boxes.append((size, yaw, on_floor))
        index, _ = encode_grid_position(torch.tensor([geom["position"][i][:2]]), 16)
        cells.append(int(index[0]))
    started = time.perf_counter()
    border, mask, split = cell_cost_maps(room, boxes, {"door_depth_m": .8}, parts=True)
    out["seconds"] = time.perf_counter() - started
    centre, _ = cell_cost_maps(room, boxes, {"door_depth_m": .8, "sub": 1})
    k = 0
    for i, cell in enumerate(cells):
        if cell is None:
            continue
        record = out["objects"][k]
        k += 1
        if not mask[i]:
            continue
        record.update(label_cell_cost=float(border[i, cell]), label_cell_cost_centre=float(centre[i, cell]),
                      label_cell_outside=float(split["outside"][i, cell]), label_cell_door=float(split["door"][i, cell]),
                      costly_cells=float((border[i] > .5).mean()), uniform_cost=float(border[i].mean()))
    return out


def summarize(rooms: list[dict]) -> dict:
    objects = [o for r in rooms for o in r["objects"]]
    floor = [o for o in objects if o["floor"]]
    costed = [o for o in objects if "label_cell_cost" in o]
    rate = lambda items, key: sum(bool(o[key]) for o in items) / len(items) if items else None
    with_doors = [r for r in rooms if r["doors"]]
    summary = {
        "rooms": len(rooms), "rooms_with_doors": len(with_doors), "doors_per_room_with_doors":
            float(np.mean([r["doors"] for r in with_doors])) if with_doors else None,
        "label_complete_objects": len(objects), "floor_objects": len(floor),
        "footprint_outside_rate": rate(objects, "outside"),
        "door_blocking_rate_floor_objects": {str(d): rate(floor, f"door_{d}") for d in DEPTHS},
        "door_blocking_rate_floor_objects_in_rooms_with_doors": {
            str(d): rate([o for r in with_doors for o in r["objects"] if o["floor"]], f"door_{d}") for d in DEPTHS},
        "rooms_with_any_blocking_label": {str(d): sum(any(o[f"door_{d}"] for o in r["objects"]) for r in with_doors)
                                          / len(with_doors) if with_doors else None for d in DEPTHS},
        "label_cell_cost": {
            "objects": len(costed),
            "zero_rate_border_candidates": rate([{"z": o["label_cell_cost"] == 0} for o in costed], "z"),
            "zero_rate_centre_only": rate([{"z": o["label_cell_cost_centre"] == 0} for o in costed], "z"),
            "gt_0.25_rate_border_candidates": rate([{"z": o["label_cell_cost"] > .25} for o in costed], "z"),
            "gt_0.25_rate_centre_only": rate([{"z": o["label_cell_cost_centre"] > .25} for o in costed], "z"),
            "mean_border_candidates": float(np.mean([o["label_cell_cost"] for o in costed])) if costed else None,
            "mean_centre_only": float(np.mean([o["label_cell_cost_centre"] for o in costed])) if costed else None,
            "nonzero_due_to_outside": rate([{"z": o["label_cell_outside"] > 0} for o in costed], "z"),
            "nonzero_due_to_door": rate([{"z": o["label_cell_door"] > 0} for o in costed], "z"),
        },
        "grid_informativeness": {
            "mean_fraction_of_cells_cost_gt_0.5": float(np.mean([o["costly_cells"] for o in costed])) if costed else None,
            "mean_cost_under_uniform_cells": float(np.mean([o["uniform_cost"] for o in costed])) if costed else None,
        },
        "cost_seconds_per_room": {"mean": float(np.mean([r["seconds"] for r in rooms])),
                                  "p95": float(np.percentile([r["seconds"] for r in rooms], 95))},
    }
    blockers = Counter(o["category"] for o in floor if o["door_0.8"])
    floor_by_category = Counter(o["category"] for o in floor)
    summary["top_blocking_categories_0.8"] = [
        {"category": c, "blocking": n, "floor_objects": floor_by_category[c], "rate": n / floor_by_category[c]}
        for c, n in blockers.most_common(25)]
    by_source = defaultdict(list)
    for r in rooms:
        by_source[r["source"]].append(r)
    summary["by_source"] = {
        s: {"rooms": len(rs), "rooms_with_doors": sum(bool(r["doors"]) for r in rs),
            "door_blocking_rate_0.8": rate([o for r in rs for o in r["objects"] if o["floor"]], "door_0.8"),
            "footprint_outside_rate": rate([o for r in rs for o in r["objects"]], "outside"),
            "label_cell_nonzero_rate": rate([{"z": o["label_cell_cost"] > 0} for r in rs for o in r["objects"]
                                             if "label_cell_cost" in o], "z")}
        for s, rs in sorted(by_source.items())}
    worst = Counter()
    for o in costed:
        if o["label_cell_cost"] > .25:
            worst[o["category"]] += 1
    summary["top_categories_label_cell_cost_gt_0.25"] = worst.most_common(15)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--rows", type=int, nargs="+", required=True, help="rows sampled from each --data file")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    if len(a.rows) != len(a.data):
        p.error("--rows needs one count per --data file")
    report = {}
    for path, rows in zip(a.data, a.rows):
        lines = _sample_lines(path, rows, a.seed)
        with Pool(a.workers) as pool:
            rooms = pool.map(_room, lines, chunksize=16)
        report[path.name] = summarize(rooms)
        print(path.name, json.dumps({k: report[path.name][k] for k in (
            "rooms", "rooms_with_doors", "door_blocking_rate_floor_objects", "label_cell_cost")}, indent=1), flush=True)
    a.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
