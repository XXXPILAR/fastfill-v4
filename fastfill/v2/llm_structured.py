"""OptiScene-style LLM baseline (huggingface/example.py): modes ``structured`` and ``structured-harness``.

A module of its own so that llm_baseline.py, whose sha256 autorun binds today's paid ``prompt`` / ``harness`` runs to,
stays byte-identical to commit 510f1e0. Same rows (``evaluate.project_minimal`` of ``--data``, first ``--max-samples``),
same answer format (llm_baseline ``to_layout``) and the same output files as llm_baseline.

The request: one fixed instruction with one train demonstration (``STRUCTURED``, the system message, the same for every
request) and the request as [Task Room Type] / [Task Room Size] / [Task Objects] sections (``sections``: exactly the
fields FastFill gets). The answer: <reasoning>[Reason]...[/Reason]</reasoning><answer>[Design]{json}[/Design]</answer>.
OptiScene gives retrieved asset sizes; this request has none, so the LLM predicts them.

``structured`` asks once; ``structured-harness`` sends ``check_answer``'s problems back up to ``--repairs`` times and
keeps the last valid layout (a later unreadable answer or failed request keeps it). The problems use only the request
and the answer: an unreadable answer, ids not exactly once, non-finite or non-positive numbers, below the floor, above
the ceiling, a declared support not met, then llm_baseline ``problems`` (beyond a wall, overlapping footprints, a raised
object with nothing under it unless it hangs on a wall). In both modes an answer that is still no valid layout is asked again from scratch once.

Each prediction row records ``checks_first`` / ``checks_final`` (problems of the first / kept answer; for an unreadable
first answer the parse or number errors) and ``usage``; summary.json ``mean_checks_*``, ``prompt_sha256``, ``repairs``
and ``usage``. ``checks_*`` are harness diagnostics, not llm_baseline's ``problems_*``: score with evaluate only::

    python -m fastfill.v2.evaluate --data <out>/rows.jsonl --predictions <out>/predictions.jsonl --output <new dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from fastfill.v2.evaluate import project_minimal
from fastfill.v2.io import fingerprint, read_samples, safe_output
from fastfill.v2.llm_baseline import SYSTEM, USER_AGENT, RateLimiter, load_env, problems, request_parameters, to_layout
from fastfill.v2.validation import on_wall

CONVENTIONS = SYSTEM[SYSTEM.index("Coordinates"):SYSTEM.index(" Use realistic")]  # the prompt modes' own wording

# The one demonstration: train split, scene_id il3d:b580392a-86d4-11f0-a478-60cf84ae2082 (IL3D_3dfront), line 1372 of
# rebuild-main-20261007b/main/train.jsonl, the first row there whose three-field projection is a bedroom with known
# height and floor, 4-8 objects, every position/size/yaw labelled, every description beyond its category, and whose
# target passes problems() and the ceiling. Sections by sections(); design = its target rounded to mm / 0.1 deg;
# the reason is written by hand. test_llm_structured rebuilds it from that row.
DEMONSTRATION = """[Example Room Type]
bedroom
[/Example Room Type]

[Example Room Size]
width (x) x length (y) x height (z): 4.178 m x 3.242 m x 2.600 m
floor polygon (x, y): [[0.0, 0.0], [4.178, 0.0], [4.178, 3.242], [0.0, 3.242]]
floor z: 0.000 m
[/Example Room Size]

[Example Objects]
[
{"id": "obj_0000", "category": "king size bed", "description": "A modern bed with a beige upholstered headboard and base. It features a neatly made bed with gray bedding and two pillows. The bed has a sleek design with minimalistic legs."},
{"id": "obj_0001", "category": "nightstand", "description": "A cylindrical container with a smooth surface, featuring two vertical black stripes and a black lid. It has a minimalist design with a matte finish."},
{"id": "obj_0002", "category": "wardrobe", "description": "A tall wardrobe with sliding doors featuring floral patterns. The body of the wardrobe has a textured, wood-like finish."},
{"id": "obj_0003", "category": "nightstand", "description": "A cylindrical container with a smooth surface, featuring two vertical black stripes and a black lid. It has a minimalist design with a matte finish."}
]
[/Example Objects]

[Example Reason]
The king size bed needs the most space, so it goes first: its headboard rests against a long wall, set in from the corners so that a nightstand fits on each side, and it faces into the room. The two nightstands flank the head of the bed against the same wall and face the same way as the bed. The wardrobe stands against the short wall beyond one nightstand, facing into the room with free space in front of its sliding doors. The floor around the foot of the bed and in front of the wardrobe stays open as the walkway.
[/Example Reason]

[Example Design]
{"room type": "bedroom", "objects": [
{"id": "obj_0000", "width_m": 2.06, "depth_m": 2.344, "height_m": 1.062, "x": 1.897, "y": 1.204, "z": 0.0, "facing_deg": 90.0},
{"id": "obj_0001", "width_m": 0.375, "depth_m": 0.375, "height_m": 0.53, "x": 0.602, "y": 0.226, "z": 0.0, "facing_deg": 90.0},
{"id": "obj_0002", "width_m": 2.342, "depth_m": 0.643, "height_m": 2.327, "x": 3.855, "y": 1.182, "z": 0.0, "facing_deg": 180.0},
{"id": "obj_0003", "width_m": 0.375, "depth_m": 0.375, "height_m": 0.53, "x": 3.185, "y": 0.214, "z": 0.0, "facing_deg": 90.0}
]}
[/Example Design]"""

# OptiScene's instruction (huggingface/example.py) adapted: sizes are predicted, ids replace its object strings.
STRUCTURED = """You are a skilled room layout designer. Your task is to place every object of [Task Objects] in a room of the given [Task Room Type] and [Task Room Size], choosing each object's size, position and facing. Follow this guidance:
(1) Place every listed object exactly once, under its own id; objects sharing a category are still separate objects. Do not add or drop objects.
(2) No sizes are given: choose a realistic width, depth and height for each object from its category and description.
(3) Avoid overlaps: bounding boxes must not intersect, except an object resting on another (its z is the top of the object under it and it stays within that object's footprint). An object hung on a wall (a painting, mirror, wall shelf or cabinet, wall-mounted TV) has its back against a wall and z is the height of its bottom above the floor. An object with a support_parent stands on that object, on the floor when it says floor, or hangs on a wall when it says wall.
(4) Place the large furniture first (beds, wardrobes, sofas, tables, cabinets) and prefer the walls and edges of the room, which keeps it spacious.
(5) Align objects parallel or perpendicular to the walls.
(6) Keep functional groups together: chairs at their table or desk and facing it, nightstands beside the bed, a sofa facing the TV or coffee table.
(7) Keep walkways clear so that every object can be reached and used.
(8) Keep every object inside the floor polygon and its top below the ceiling.
(9) Before giving coordinates, reason briefly step by step about the general arrangement: which objects need the most space or a fixed place, which belong together, and how people move through the room. Then report the design.

""" + CONVENTIONS + """

The response must follow this format, with exactly one entry per id of [Task Objects] and plain numbers as values:

<reasoning>
[Reason]
...
[/Reason]
</reasoning>

<answer>
[Design]
{"room type": "...", "objects": [{"id": "...", "width_m": ..., "depth_m": ..., "height_m": ..., "x": ..., "y": ..., "z": ..., "facing_deg": ...}, ...]}
[/Design]
</answer>

First carefully read this example:

""" + DEMONSTRATION + """

Before submitting your design, verify:
- every id appears exactly once
- all objects are inside the room and below the ceiling
- no objects overlap except objects resting on others
- walkways stay clear and the layout is practical
Now design the room given in [Task Room Type], [Task Room Size] and [Task Objects]."""
NUMBERS = ("width_m", "depth_m", "height_m", "x", "y", "z", "facing_deg")
MODES = ("structured", "structured-harness")
USAGE = ("prompt_tokens", "completion_tokens", "total_tokens")
SHOWN = 40  # problems listed in one repair message (all are counted)


def sections(condition, tag="Task"):
    """[<tag> Room Type], [<tag> Room Size], [<tag> Objects]: room type, room size (width x length x height, floor
    polygon, floor z) and the furniture list (id, category, description; support_parent only when declared)."""
    room = condition["room"]
    xs, ys = [p[0] for p in room["floor_polygon_xy_m"]], [p[1] for p in room["floor_polygon_xy_m"]]
    height, floor = room.get("height_m"), room.get("floor_z_m")
    size = (f"width (x) x length (y) x height (z): {max(xs) - min(xs):.3f} m x {max(ys) - min(ys):.3f} m x "
            f"{f'{height:.3f} m' if height else 'unknown'}\n"
            f"floor polygon (x, y): {json.dumps([[round(x, 3), round(y, 3)] for x, y in room['floor_polygon_xy_m']])}\n"
            f"floor z: {'unknown' if floor is None else f'{floor:.3f} m'}")
    objects = [{k: o[k] for k in ("id", "category", "description", "support_parent") if o.get(k) is not None}
               for o in condition["objects"]]
    return "\n\n".join(f"[{tag} {name}]\n{body}\n[/{tag} {name}]" for name, body in (
        ("Room Type", room.get("room_type") or "unknown"), ("Room Size", size),
        ("Objects", "[\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in objects) + "\n]")))


def structured_problems(layout, condition, *, margin=.05):
    """Every problem the request alone shows: the vertical room bounds and declared supports, then ``problems``
    (uncapped, so a crowded room's overlaps never hide these) except "floats" for an object declared on or hung on
    a wall (``validation.on_wall``: a footprint side within ``WALL_GAP_M`` of a known boundary), which ``problems``
    (byte-pinned in llm_baseline) would send to the floor."""
    room, found = condition["room"], []
    floor, height = room.get("floor_z_m") or 0., room.get("height_m")
    names = {o["id"]: o["category"] for o in condition["objects"]}
    placed = {o["id"]: o for o in layout["objects"]}
    hung = {i for i, o in placed.items() if room.get("boundary_known") is not False and on_wall(o, room)}
    for i, o in placed.items():
        z, top = o["bottom_center_m"][2], o["bottom_center_m"][2] + o["target_size_local_m"][2]
        if z < floor - margin:
            found.append(f"{i} ({names[i]}) is {floor - z:.2f} m below the floor")
        if height and top > floor + height + margin:
            found.append(f"{i} ({names[i]}) reaches {top - floor:.2f} m, above the {height:.2f} m ceiling")
    for request in condition["objects"]:
        i, parent = request["id"], request.get("support_parent")
        x, y, z = placed[i]["bottom_center_m"]
        if parent == "floor" and abs(z - floor) > margin:
            found.append(f"{i} ({names[i]}) must stand on the floor (z = {floor:.2f}), not at z = {z:.2f}")
        elif parent == "wall" and i not in hung:
            found.append(f"{i} ({names[i]}) must hang on a wall: one side of its footprint against a wall")
        elif parent in placed:  # its bottom centre on the parent's top, inside the parent's axis-aligned footprint
            (d, w, h), (px, py, pz), yaw = placed[parent]["target_size_local_m"], placed[parent]["bottom_center_m"], placed[parent]["yaw_rad"]
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            if abs(z - (pz + h)) > margin or abs(x - px) > .5 * (c * d + s * w) or abs(y - py) > .5 * (s * d + c * w):
                found.append(f"{i} ({names[i]}) must rest on {parent} ({names[parent]}): z = {pz + h:.2f} m and inside its footprint")
    walls = hung | {r["id"] for r in condition["objects"] if r.get("support_parent") == "wall"}
    floats = tuple(f"{i} ({names[i]}) floats " for i in walls)
    return found + [p for p in problems(layout, condition, limit=None) if not p.startswith(floats)]


def design_text(reply):
    """The [Design] JSON of a tagged answer: after the last <answer> (the reasoning may hold braces) and [Design]."""
    text = reply.rsplit("<answer>", 1)[-1].rsplit("[Design]", 1)[-1]
    return text.split("[/Design]", 1)[0].split("</answer>", 1)[0]


def check_answer(reply, condition):
    """A structured answer -> (layout, problems), layout None when the answer is no valid layout."""
    def bad(value, positive):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return True
        return not math.isfinite(value) or positive and value <= 0
    try:
        design = design_text(reply)
        answers = json.loads(design[design.find("{"):design.rfind("}") + 1])["objects"]
        numbers = [f"{o.get('id')}: {key} must be a {'positive' if key.endswith('_m') else 'finite'} number, got {o.get(key)!r}"
                   for o in answers for key in NUMBERS if bad(o.get(key), key.endswith("_m"))]
    except Exception as error:
        return None, [f"the [Design] JSON could not be read ({type(error).__name__}: {error})"]
    try:
        layout = to_layout(design, condition)
    except Exception as error:  # ids not exactly once (to_layout's message), or a number listed above
        ids = "exactly once" in str(error)
        return None, ([str(error)] if ids else [] if numbers else [f"no valid layout ({type(error).__name__}: {error})"]) + numbers
    return layout, structured_problems(layout, condition)


def chat(env, model, messages, *, limiter=None, timeout=300., attempts=6, usage=None):
    """llm_baseline.chat with the system message inside ``messages``; ``usage`` collects the response's token usage."""
    body = json.dumps({**request_parameters(env, model), "messages": messages}).encode()
    for attempt in range(attempts):
        if limiter is not None:
            limiter.wait()
        request = urllib.request.Request(env["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {env['OPENAI_API_KEY']}", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                answer = json.loads(response.read())
            if usage is not None and answer.get("usage"):
                usage.append(answer["usage"])
            return answer["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            if error.code not in (408, 409, 425, 429) and error.code < 500 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            if attempt == attempts - 1:
                raise
        time.sleep(min(120., 5. * 2 ** attempt))


def tally(usage):
    """Summed token usage of the calls whose response reported one; None when none did (e.g. an injected ``ask``)."""
    return {"calls_reported": len(usage), **{k: sum(u.get(k) or 0 for u in usage) for k in USAGE}} if usage else None


def run(data, output, env_file, *, model=None, mode="structured", repairs=2, workers=4, max_samples=None, retries=1, ask=chat):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    target = safe_output(output, create=True)
    env = load_env(env_file)
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise ValueError("pass --model or set OPENAI_MODEL in the env file")
    limiter = RateLimiter(float(env.get("FASTFILL_LLM_RPM", 30)))
    identity = {"data_sha256": fingerprint(data), "implementation_sha256": fingerprint(Path(__file__))}
    rows = [p for p in map(project_minimal, read_samples(data)) if p is not None][:max_samples]
    calls = [0]

    def call(messages, usage):
        calls[0] += 1  # one chat call (its own HTTP retries included); one that raises reports no usage
        return ask(env, model, messages) if ask is not chat else chat(env, model, messages, limiter=limiter, usage=usage)

    def one(row):
        start, error, condition, valid, usage = time.perf_counter(), None, row["condition"], [], []
        for attempt in range(1, retries + 2):  # still no valid layout (after the repairs): asked again from scratch
            try:
                messages = [{"role": "system", "content": STRUCTURED}, {"role": "user", "content": sections(condition)}]
                reply = call(messages, usage)
                layout, found = check_answer(reply, condition)
                valid.append(layout is not None)
                kept, first, rounds = None if layout is None else (layout, found), len(found), 0
                while mode == "structured-harness" and found and rounds < repairs:
                    messages += [{"role": "assistant", "content": reply}, {"role": "user", "content":
                                 "Your design has these problems:\n- " + "\n- ".join(found[:SHOWN])
                                 + "\nFix them and answer again in the same format, with the complete design for every id."}]
                    try:
                        reply = call(messages, usage)
                    except Exception:
                        if kept is None:
                            raise
                        break  # a failed repair request keeps the last valid layout
                    rounds += 1
                    layout, found = check_answer(reply, condition)
                    kept = kept if layout is None else (layout, found)
                if kept is None:
                    raise ValueError("no valid layout: " + "; ".join(found))
                return {"layout": kept[0], "error": None, "latency_s": time.perf_counter() - start, "attempts": attempt,
                        "first_answer_valid": valid[0], "repair_rounds": rounds, "checks_first": first,
                        "checks_final": len(kept[1]), "usage": tally(usage)}
            except Exception as exc:  # malformed or incomplete answers are failures, never dropped
                error = f"{type(exc).__name__}: {exc}"[:500]
        return {"layout": None, "error": error, "latency_s": time.perf_counter() - start, "attempts": retries + 1,
                "first_answer_valid": valid[0] if valid else None, "usage": tally(usage)}

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, rows))
    with (target / "rows.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with (target / "predictions.jsonl").open("w") as stream:
        stream.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    ok = [r for r in results if r["layout"] is not None]
    summary = {"model": model, "mode": mode, "request_parameters": request_parameters(env, model),
               "endpoint": env.get("OPENAI_BASE_URL", "").rstrip("/") + "/chat/completions",
               "rows": len(rows), "failed": len(results) - len(ok), "api_calls": calls[0],
               "first_answer_invalid": sum(r["first_answer_valid"] is False for r in results),
               "unanswered": sum(r["first_answer_valid"] is None for r in results),
               "mean_latency_s": sum(r["latency_s"] for r in results) / max(1, len(results)),
               "mean_checks_first": sum(r["checks_first"] for r in ok) / max(1, len(ok)),
               "mean_checks_final": sum(r["checks_final"] for r in ok) / max(1, len(ok)),
               "prompt_sha256": hashlib.sha256(STRUCTURED.encode()).hexdigest(),
               "repairs": repairs if mode == "structured-harness" else 0,
               "usage": {k: sum((r["usage"] or {}).get(k, 0) for r in results) for k in ("calls_reported", *USAGE)},
               "data": str(data), **identity}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True, help="file with OPENAI_BASE_URL / OPENAI_API_KEY [/ OPENAI_MODEL / FASTFILL_LLM_RPM]")
    p.add_argument("--model")
    p.add_argument("--mode", choices=MODES, default="structured")
    p.add_argument("--repairs", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int)
    a = p.parse_args(argv)
    print(json.dumps(run(a.data, a.output, a.env, model=a.model, mode=a.mode, repairs=a.repairs, workers=a.workers,
                         max_samples=a.max_samples)))


if __name__ == "__main__":
    main()
