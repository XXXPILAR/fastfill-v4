"""Results-first workflow: FastFill sizes, LLM placement, checked repair.

Per request:
  1. sizes (``--sizes``): ``checkpoint`` runs a FastFill checkpoint (``evaluate.predict_layout``, spread decode); its
     sizes are locked and its layout is the fallback. ``ground_truth`` locks the row's labelled sizes: an upper bound
     for diagnosis, never a result to report. ``none`` lets the LLM choose sizes (the structured-harness setting).
  2. the LLM places every object. With locked sizes it answers only x, y, z and facing_deg, so it cannot change them.
  3. ``workflow_problems`` checks the answer: ``llm_structured.structured_problems`` (below the floor, above the ceiling,
     declared supports, beyond a wall, overlaps, floating) plus footprints outside the true floor polygon, floor objects
     inside a door's clearance zone (``cell_costs.door_zones``, ``evaluate.DOOR_ZONE_DEPTH_M`` deep) and footprints
     overlapping fixed objects. The problems go back to the LLM up to ``--repairs`` times; the valid layout with the
     fewest problems is kept, not the last one.
  3b. before every check ``tidy`` (no LLM) snaps near-contact bottoms onto the floor or the object under them and
     pushes footprints that cross a wall by at most 0.3 m back inside: the strict scorer counts millimetre gaps.
  4. no valid LLM layout: the FastFill layout when there is one (counted in ``fallback``), else a failure row.

``--condition minimal`` uses the three-field projection (``evaluate.project_minimal``: the rows of the paid
``harness`` / ``structured-harness`` baselines, no doors); ``full`` the row's own condition (floor polygon, openings,
fixed objects, supports, constraints). Writes ``rows.jsonl``, ``predictions.jsonl`` (one row each, ``layout`` null
on failure) and ``summary.json`` like ``llm_baseline``; score with::

    python -m fastfill.v2.evaluate --data <out>/rows.jsonl --predictions <out>/predictions.jsonl --output <new dir>

``llm_baseline`` and ``llm_structured`` stay unchanged (autorun binds paid runs to their sha256); this module only
imports from them. Credentials come from the same env file (OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, ...).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import threading
import time

from shapely.geometry import Polygon

from fastfill.v2.cell_costs import FLOOR_TOLERANCE_M, blocks_doors, door_zones
from fastfill.v2.evaluate import DOOR_ZONE_DEPTH_M, FOOTPRINT_VIOLATION_FRACTION, project_minimal
from fastfill.v2.io import fingerprint, read_samples, safe_output
from fastfill.v2.llm_baseline import RateLimiter, load_env, request_parameters
from fastfill.v2.llm_structured import CONVENTIONS, chat, design_text, sections, structured_problems, tally
from fastfill.v2.schema import migrate_legacy_row, validate_layout
from fastfill.v2.validation import footprint

SIZES = ("checkpoint", "ground_truth", "none")
CONDITIONS = ("minimal", "full")
SHOWN = 40  # problems listed in one repair message

GUIDANCE = """You are a skilled interior layout designer. Place every object of [Task Objects] in the room described below.
(1) Place every listed object exactly once, under its own id. Do not add or drop objects.
(2) {sizes}
(3) Avoid overlaps: footprints must not intersect, except an object resting on another (its z is the top of the object under it and it stays within that object's footprint). An object hung on a wall has its back against a wall and z is the height of its bottom above the floor. An object with a support_parent stands on that object, on the floor when it says floor, or hangs on a wall when it says wall.
(4) Place the large furniture first (beds, wardrobes, sofas, desks, tables, cabinets) and prefer the walls and edges of the room.
(5) Align objects parallel or perpendicular to the walls.
(6) Keep functional groups together: chairs at their table or desk and facing it, nightstands beside the bed, a sofa facing the TV or coffee table, a desk chair facing its desk.
(7) Keep walkways clear so that every object can be reached and used{doors}.
(8) Keep every object inside the floor polygon and its top below the ceiling{fixed}.
(9) Before giving coordinates, reason briefly about the general arrangement: what needs the most space or a fixed place, what belongs together, and how people move through the room. Then report the design.

""" + CONVENTIONS + """

Answer in this format, with exactly one entry per id and plain numbers as values:

<reasoning>
[Reason]
...
[/Reason]
</reasoning>

<answer>
[Design]
{{"objects": [{{{fields}}}, ...]}}
[/Design]
</answer>"""
LOCKED = ("Each object's width_m, depth_m and height_m are given and fixed: do not change them; choose only its "
          "position (x, y, z) and facing_deg.")
FREE = "No sizes are given: choose a realistic width, depth and height for each object from its category and description."


def _openings(room):
    """Doors and windows of the floor polygon as world-coordinate centres on their walls."""
    polygon, rows = room["floor_polygon_xy_m"], []
    for o in room.get("openings") or []:
        if not isinstance(o.get("wall_index"), int) or o["wall_index"] >= len(polygon):
            continue
        (ax, ay), (bx, by) = polygon[o["wall_index"]], polygon[(o["wall_index"] + 1) % len(polygon)]
        t = o["position_on_wall"]
        row = {"type": o["type"], "centre_xy": [round(ax + t * (bx - ax), 3), round(ay + t * (by - ay), 3)],
               "wall_from": [round(ax, 3), round(ay, 3)], "wall_to": [round(bx, 3), round(by, 3)],
               "width_m": round(o["width_m"], 3), "height_m": round(o["height_m"], 3)}
        if o.get("sill_height_m") is not None:
            row["sill_height_m"] = round(o["sill_height_m"], 3)
        rows.append(row)
    return rows


def request_text(condition, sizes):
    """The request: room type / size / objects (``llm_structured.sections``), each object with its locked
    width/depth/height when ``sizes`` maps id -> local [depth, width, height], then doors and windows, fixed objects
    and constraints when the condition has them."""
    text = sections(condition)
    if sizes is not None:
        rows = []
        for o in condition["objects"]:
            depth, width, height = sizes[o["id"]]
            rows.append({**{k: o[k] for k in ("id", "category", "description", "support_parent") if o.get(k) is not None},
                         "width_m": round(width, 3), "depth_m": round(depth, 3), "height_m": round(height, 3)})
        start = text.index("[Task Objects]")
        text = text[:start] + "[Task Objects]\n[\n" + ",\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n]\n[/Task Objects]"
    room = condition["room"]
    extra = []
    if _openings(room):
        extra.append(("Doors and Windows", json.dumps(_openings(room))))
    if room.get("fixed_objects"):
        extra.append(("Fixed Objects", json.dumps([{"id": f["id"], "category": f.get("category"),
                       "width_m": round(f["size_local_m"][1], 3), "depth_m": round(f["size_local_m"][0], 3),
                       "height_m": round(f["size_local_m"][2], 3), "bottom_centre": [round(v, 3) for v in f["bottom_center_m"]],
                       "facing_deg": round(math.degrees(f["yaw_rad"]) % 360, 1)} for f in room["fixed_objects"]])))
    if condition.get("constraints"):
        extra.append(("Constraints", json.dumps(condition["constraints"])))
    return text + "".join(f"\n\n[Task {name}]\n{body}\n[/Task {name}]" for name, body in extra)


def instructions(condition, locked):
    room = condition["room"]
    fields = '"id": "...", ' + ('' if locked else '"width_m": ..., "depth_m": ..., "height_m": ..., ') + \
             '"x": ..., "y": ..., "z": ..., "facing_deg": ...'
    return GUIDANCE.format(sizes=LOCKED if locked else FREE, fields=fields,
                           doors="; leave the floor in front of every door free" if _openings(room) else "",
                           fixed="; fixed objects are already in the room and must not be overlapped" if room.get("fixed_objects") else "")


def parse(reply, condition, sizes):
    """Answer -> layout. Locked sizes come from ``sizes`` (local [depth, width, height]), never from the answer."""
    design = design_text(reply)
    answers = json.loads(design[design.find("{"):design.rfind("}") + 1])["objects"]
    ids, requested = [str(o["id"]) for o in answers], [o["id"] for o in condition["objects"]]
    if sorted(ids) != sorted(requested):
        raise ValueError(f"answer must hold each requested id exactly once: duplicated {sorted({i for i in ids if ids.count(i) > 1})}, "
                         f"unknown {sorted(set(ids) - set(requested))}, missing {sorted(set(requested) - set(ids))}")
    by_id, objects = {str(o["id"]): o for o in answers}, []
    for request in condition["objects"]:
        o = by_id[request["id"]]
        size = list(sizes[request["id"]]) if sizes is not None else [float(o["depth_m"]), float(o["width_m"]), float(o["height_m"])]
        values = [float(o[k]) for k in ("x", "y", "z", "facing_deg")]
        if not all(map(math.isfinite, values + size)) or min(size) <= 0:
            raise ValueError(f"{request['id']}: numbers must be finite and sizes positive")
        objects.append({"id": request["id"], "target_size_local_m": size, "bottom_center_m": values[:3],
                        "yaw_rad": (math.radians(values[3]) + math.pi) % (2 * math.pi) - math.pi})
    layout = {"schema_version": "fastfill.v2", "objects": objects}
    validate_layout(layout, condition)
    return layout


SNAP_M = .03       # tidy: a bottom this close to the floor or to a supporting top is put exactly on it
PUSH_MAX_M = .3    # tidy: a footprint at most this far through a wall is pushed back inside; farther is a real error
PUSH_MARGIN_M = 1e-3


def _shape(o):
    return footprint({"_pos": o["bottom_center_m"], "_size": o["target_size_local_m"], "_yaw": o["yaw_rad"]})


def tidy(layout, condition):
    """Deterministic last step (no LLM): snap near-contact bottoms onto the floor or onto the top of the object
    under them, and push footprints that cross a wall by at most ``PUSH_MAX_M`` back inside. The strict scorer
    (``validation.validate_scene``, 0.1 mm tolerance) counts the millimetre gaps and wall crossings an LLM leaves
    when it rounds coordinates or puts furniture flush against a wall. Returns (layout, number of objects moved)."""
    room = condition["room"]
    floor_polygon, floor = Polygon(room["floor_polygon_xy_m"]), room.get("floor_z_m") or 0.
    objects = [{**o, "bottom_center_m": list(o["bottom_center_m"])} for o in layout["objects"]]
    moved = set()
    for o in sorted(objects, key=lambda o: o["bottom_center_m"][2]):  # lower objects settle first
        x, y, z = o["bottom_center_m"]
        tops = [p["bottom_center_m"][2] + p["target_size_local_m"][2] for p in objects
                if p is not o and abs(p["bottom_center_m"][2] + p["target_size_local_m"][2] - z) <= SNAP_M
                and _shape(p).buffer(1e-9).contains(_shape(o).centroid)]
        target = max(tops) if tops else floor if abs(z - floor) <= SNAP_M else None
        if target is not None and target != z:
            o["bottom_center_m"][2] = target
            moved.add(o["id"])
    for o in objects:
        shape = _shape(o)
        if floor_polygon.covers(shape):
            continue
        centre = floor_polygon.representative_point()
        dx, dy = centre.x - o["bottom_center_m"][0], centre.y - o["bottom_center_m"][1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            continue
        lo, hi = 0., PUSH_MAX_M  # smallest shift towards the room's inside that covers the footprint
        shifted = lambda d: footprint({"_pos": [o["bottom_center_m"][0] + dx / norm * d, o["bottom_center_m"][1] + dy / norm * d, 0.],
                                       "_size": o["target_size_local_m"], "_yaw": o["yaw_rad"]})
        if not floor_polygon.covers(shifted(hi)):
            continue
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if floor_polygon.covers(shifted(mid)) else (mid, hi)
        d = hi + PUSH_MARGIN_M if floor_polygon.covers(shifted(hi + PUSH_MARGIN_M)) else hi
        o["bottom_center_m"][0] += dx / norm * d
        o["bottom_center_m"][1] += dy / norm * d
        moved.add(o["id"])
    return {**layout, "objects": objects}, len(moved)


def workflow_problems(layout, condition):
    """What the strict scorer would fail, said to the LLM: ``validate_scene``'s boundary, collision, fixed-object,
    ceiling and constraint violations, ``structured_problems``' supports, floor and floating checks (its own
    tolerant wall / overlap lines are replaced by the strict ones), and floor objects in a door's clearance zone."""
    from fastfill.v2.validation import validate_scene
    room = condition["room"]
    names = {o["id"]: o["category"] for o in condition["objects"]}
    names.update({f["id"]: f.get("category") or "fixed object" for f in room.get("fixed_objects", [])})
    found = [p for p in structured_problems(layout, condition) if " extends " not in p and " overlap (" not in p]
    for c in validate_scene(condition, layout["objects"])["checks"]:
        if c["status"] != "violation" or not c["hard"]:
            continue
        ids = [f"{i} ({names.get(i, '?')})" for i in c["object_ids"]]
        if c["code"] == "boundary":
            found.append(f"{ids[0]} crosses a wall: its footprint must lie completely inside the floor polygon")
        elif c["code"] in ("collision", "fixed_collision"):
            found.append(f"{ids[0]} and {ids[1]} intersect ({c['overlap_area_m2']:.3f} m2 of footprint while their heights "
                         "overlap): separate them, or put one exactly on top of the other (z = the lower one's top)")
        elif c["code"] == "ceiling":
            found.append(f"{ids[0]} reaches above the ceiling")
        elif c["code"] not in ("support", "floor_lower_bound"):  # those are already worded by structured_problems
            found.append(f"{', '.join(ids) or 'the layout'} violates {c['code']}" + (f": {c['message']}" if c.get("message") else ""))
    floor_polygon, floor = Polygon(room["floor_polygon_xy_m"]), room.get("floor_z_m") or 0.
    doors = [Polygon(zone).intersection(floor_polygon) for zone in door_zones(room, DOOR_ZONE_DEPTH_M)]
    for o in layout["objects"]:
        shape, on_floor = _shape(o), abs(o["bottom_center_m"][2] - floor) <= FLOOR_TOLERANCE_M
        if shape.area > 0 and blocks_doors(on_floor, o["target_size_local_m"][2]) and \
                any(shape.intersection(d).area > FOOTPRINT_VIOLATION_FRACTION * shape.area for d in doors):
            found.append(f"{o['id']} ({names[o['id']]}) stands in front of a door; keep about {DOOR_ZONE_DEPTH_M:g} m in front of every door free")
    return found


def _fastfill(checkpoint, device):
    from fastfill.v2.batch import load_tokenizer
    from fastfill.v2.evaluate import predict_layout
    from fastfill.v2.io import load_checkpoint_config
    from fastfill.v2.model import load_model
    model = load_model(checkpoint, device=device).eval()
    tokenizer = load_tokenizer("tiny" if model.config.backbone == "tiny" else str(Path(checkpoint).parent / "tokenizer"),
                               local_files_only=True)
    max_length = load_checkpoint_config(checkpoint)["max_length"] or 4096
    lock = threading.Lock()  # one GPU model shared by the worker threads

    def predict(condition):
        with lock:
            return predict_layout(model, tokenizer, condition, max_length=max_length, device=device)
    return predict


def run(data, output, env_file, *, sizes="checkpoint", condition_form="minimal", checkpoint=None, device="cpu", model=None,
        repairs=2, workers=4, max_samples=None, retries=1, ask=chat):
    if sizes not in SIZES or condition_form not in CONDITIONS:
        raise ValueError(f"sizes must be one of {SIZES}, condition one of {CONDITIONS}")
    if (sizes == "checkpoint") != (checkpoint is not None):
        raise ValueError("--checkpoint is required by, and only used with, --sizes checkpoint")
    target = safe_output(output, create=True)
    env = load_env(env_file)
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise ValueError("pass --model or set OPENAI_MODEL in the env file")
    limiter = RateLimiter(float(env.get("FASTFILL_LLM_RPM", 30)))
    samples = [migrate_legacy_row(s) for s in read_samples(data)]
    rows = [p for p in (map(project_minimal, samples) if condition_form == "minimal" else samples) if p is not None][:max_samples]
    predict = _fastfill(checkpoint, device) if sizes == "checkpoint" else None
    calls, usage = [0], []

    def call(messages):
        calls[0] += 1
        return ask(env, model, messages) if ask is not chat else chat(env, model, messages, limiter=limiter, usage=usage)

    def one(row):
        start, condition = time.perf_counter(), row["condition"]
        fallback = predict(condition) if predict else None
        if sizes == "checkpoint":
            locked = {o["id"]: o["target_size_local_m"] for o in fallback["objects"]}
        elif sizes == "ground_truth":
            locked = {o["id"]: o["target_size_local_m"] for o in row["target"]["objects"]}
        else:
            locked = None
        system = instructions(condition, locked is not None)
        best = error = None
        trace, tidied = [], []  # problems after each valid answer (after tidy), objects tidy moved in it
        for attempt in range(1, retries + 2):  # an answer that is no layout is asked again from scratch
            messages = [{"role": "system", "content": system}, {"role": "user", "content": request_text(condition, locked)}]
            try:
                reply = call(messages)
                layout, moved = tidy(parse(reply, condition, locked), condition)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:500]
                continue
            found = workflow_problems(layout, condition)
            best, trace, tidied = (layout, found), [len(found)], [moved]
            rounds = 0
            while found and rounds < repairs:
                rounds += 1
                messages += [{"role": "assistant", "content": reply}, {"role": "user", "content":
                             "Your design has these problems:\n- " + "\n- ".join(found[:SHOWN]) +
                             (f"\n(and {len(found) - SHOWN} more)" if len(found) > SHOWN else "") +
                             "\nFix them and return the complete corrected design for every id in the same format."}]
                try:
                    reply = call(messages)
                    layout, moved = tidy(parse(reply, condition, locked), condition)
                except Exception:  # an unreadable repair keeps the best layout so far
                    trace.append(None)
                    break
                found = workflow_problems(layout, condition)
                trace.append(len(found))
                tidied.append(moved)
                if len(found) < len(best[1]):
                    best = (layout, found)
            break
        source = "llm" if best else "fastfill_fallback" if fallback else None
        layout = best[0] if best else fallback
        return {"layout": layout, "source": source, "error": None if layout else error, "attempts": attempt,
                "problems_trace": trace, "tidied_objects": tidied, "problems_final": len(best[1]) if best else None,
                "problems_final_list": best[1][:SHOWN] if best else None, "latency_s": time.perf_counter() - start}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, rows))
    with (target / "rows.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with (target / "predictions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    llm = [r for r in results if r["source"] == "llm"]
    summary = {"model": model, "sizes": sizes, "condition": condition_form, "checkpoint": str(checkpoint) if checkpoint else None,
               "repairs": repairs, "request_parameters": request_parameters(env, model), "rows": len(rows),
               "llm_layouts": len(llm), "fallback": sum(r["source"] == "fastfill_fallback" for r in results),
               "failed": sum(r["layout"] is None for r in results), "api_calls": calls[0], "usage": tally(usage),
               "mean_problems_first": sum(r["problems_trace"][0] for r in llm) / max(1, len(llm)),
               "mean_problems_final": sum(r["problems_final"] for r in llm) / max(1, len(llm)),
               "rows_without_problems": sum(r["problems_final"] == 0 for r in llm),
               "mean_latency_s": sum(r["latency_s"] for r in results) / max(1, len(results)),
               "data": str(data), "data_sha256": fingerprint(data), "implementation_sha256": fingerprint(Path(__file__))}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="new directory")
    p.add_argument("--env", required=True, help="env file with OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL")
    p.add_argument("--sizes", choices=SIZES, default="checkpoint")
    p.add_argument("--condition", choices=CONDITIONS, default="minimal")
    p.add_argument("--checkpoint", help="FastFill model directory (--sizes checkpoint)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model")
    p.add_argument("--repairs", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int)
    a = p.parse_args(argv)
    print(json.dumps(run(a.data, a.output, a.env, sizes=a.sizes, condition_form=a.condition, checkpoint=a.checkpoint,
                         device=a.device, model=a.model, repairs=a.repairs, workers=a.workers, max_samples=a.max_samples), indent=2))


if __name__ == "__main__":
    main()
