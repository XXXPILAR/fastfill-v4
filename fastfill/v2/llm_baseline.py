"""Prompt-engineering and harness baselines: an OpenAI-compatible chat model places the three-field request.

Keeps the evaluation rows whose room is a boundary-known rectangle and projects them to the same
three-field condition the structured model is selected on (``evaluate.project_minimal``). The LLM sees
room type, room extent and the furniture list (id, category, description) only, and answers every
object's width/depth/height, bottom centre and facing angle.

Modes: ``prompt`` asks once; ``harness`` then checks the answer (objects outside the room, overlapping
objects that share a height interval, raised objects resting on nothing) and asks for a corrected
layout with the concrete problems listed, up to ``--repairs`` times. Requests are paced to the env
file's FASTFILL_LLM_RPM (default 30) and retried with backoff on rate limits and server errors.

An answer that is no valid layout (unparseable, or ids duplicated/unknown/missing) is asked again once from
scratch, as is a request that fails (after ``chat``'s own retries): ``attempts`` per row counts the first answers
asked (those needed for its layout, else all) and ``first_answer_valid`` whether the first answer received was a
valid layout (null: no request got one). summary.json's ``first_answer_invalid`` counts the rows whose first answer
was no valid layout (raw id compliance; failed requests never count there), ``unanswered`` the rows no request
answered, and ``failed`` the rows still without a layout.

Writes ``rows.jsonl`` (the projected rows) and ``predictions.jsonl`` (one row each, ``layout`` null on
failure) for::

    python -m fastfill.v2.evaluate --data <out>/rows.jsonl --predictions <out>/predictions.jsonl --output <new dir>

Credentials come from an env file (OPENAI_BASE_URL, OPENAI_API_KEY, optional OPENAI_MODEL,
FASTFILL_LLM_RPM, FASTFILL_LLM_REASONING_EFFORT), never argv; summary.json records the parameters sent and
the sha256 of the input rows file and of this module (``data_sha256``, ``implementation_sha256``).
"""
from __future__ import annotations

import argparse
import http.client
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.request

from fastfill.v2.evaluate import project_minimal
from fastfill.v2.io import fingerprint, read_samples, safe_output
from fastfill.v2.schema import validate_layout

SYSTEM = ("You are an interior layout designer. Place every listed object in the room and answer with JSON only. "
          "Coordinates are metres in the room frame: x and y on the floor, z up. width_m is the object's "
          "left-right extent, depth_m its front-back extent, height_m its vertical extent. x, y, z is the centre "
          "of the object's bottom face (z = height of the surface it stands on). facing_deg is the direction its "
          "front faces, counter-clockwise from +x (0 faces +x, 90 faces +y). Use realistic sizes, keep objects "
          "inside the room, avoid overlaps except objects resting on others.")


def load_env(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip().removeprefix("export ").strip()] = value.strip().strip("'\"")
    return values


def prompt(condition):
    room = condition["room"]
    xs, ys = [p[0] for p in room["floor_polygon_xy_m"]], [p[1] for p in room["floor_polygon_xy_m"]]
    height = room.get("height_m")
    objects = [{"id": o["id"], "category": o["category"], "description": o["description"]} for o in condition["objects"]]
    return (f"Room type: {room.get('room_type', 'unknown')}\n"
            f"Floor: x from {min(xs):.3f} to {max(xs):.3f}, y from {min(ys):.3f} to {max(ys):.3f}, floor z = {room['floor_z_m']}\n"
            f"Ceiling height: {f'{height:.3f} m' if height else 'unknown'}\n"
            f"Objects ({len(objects)}):\n{json.dumps(objects, ensure_ascii=False)}\n"
            'Answer {"objects": [{"id", "width_m", "depth_m", "height_m", "x", "y", "z", "facing_deg"}, ...]} '
            "with exactly one entry per id.")


def to_layout(reply, condition):
    """LLM JSON -> fastfill.v2 layout: local +X is the front, so size = [depth, width, height] and yaw = facing."""
    text = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()
    text = text[text.find("{"):text.rfind("}") + 1]
    ids = [str(o["id"]) for o in json.loads(text)["objects"]]
    requested = [str(o["id"]) for o in condition["objects"]]
    if sorted(ids) != sorted(requested):  # duplicates, unknown or missing ids make the answer invalid, not cleaned
        raise ValueError(f"answer must hold each requested id exactly once: duplicated {sorted({i for i in ids if ids.count(i) > 1})}, "
                         f"unknown {sorted(set(ids) - set(requested))}, missing {sorted(set(requested) - set(ids))}")
    answers = {str(o["id"]): o for o in json.loads(text)["objects"]}
    objects = []
    for request in condition["objects"]:
        o = answers[request["id"]]
        facing = math.radians(float(o["facing_deg"]))
        objects.append({"id": request["id"],
                        "target_size_local_m": [float(o["depth_m"]), float(o["width_m"]), float(o["height_m"])],
                        "bottom_center_m": [float(o["x"]), float(o["y"]), float(o["z"])],
                        "yaw_rad": (facing + math.pi) % (2 * math.pi) - math.pi})
    layout = {"schema_version": "fastfill.v2", "objects": objects}
    validate_layout(layout, condition)
    return layout


def problems(layout, condition, *, margin=.05, max_overlap=.15, limit=40):
    """Concrete, checkable defects the harness reports back: out of room, overlaps, floating items."""
    # ponytail: rotated footprints by their axis-aligned bounds, the same approximation as spread decoding
    room = condition["room"]
    polygon = room["floor_polygon_xy_m"]
    lo = (min(p[0] for p in polygon), min(p[1] for p in polygon))
    hi = (max(p[0] for p in polygon), max(p[1] for p in polygon))
    floor = room.get("floor_z_m") or 0.
    names = {o["id"]: o["category"] for o in condition["objects"]}
    boxes = []
    for o in layout["objects"]:
        (d, w, h), (x, y, z), yaw = o["target_size_local_m"], o["bottom_center_m"], o["yaw_rad"]
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        boxes.append((o["id"], x, y, z, .5 * (c * d + s * w), .5 * (s * d + c * w), h))
    found = []
    for i, x, y, z, hx, hy, h in boxes:
        beyond = max(lo[0] - (x - hx), (x + hx) - hi[0], lo[1] - (y - hy), (y + hy) - hi[1])
        if beyond > margin:
            found.append(f"{i} ({names[i]}) extends {beyond:.2f} m beyond a wall")
        if z - floor > .15 and not any(j != i and abs(z - (zj + hj)) <= margin and abs(x - xj) <= hxj and abs(y - yj) <= hyj
                                       for j, xj, yj, zj, hxj, hyj, hj in boxes):
            found.append(f"{i} ({names[i]}) floats at z={z:.2f} m with nothing under it; put it on a surface or the floor")
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            i, xi, yi, zi, hxi, hyi, hi_ = boxes[a]
            j, xj, yj, zj, hxj, hyj, hj = boxes[b]
            if min(zi + hi_, zj + hj) - max(zi, zj) <= margin:
                continue
            ox = min(xi + hxi, xj + hxj) - max(xi - hxi, xj - hxj)
            oy = min(yi + hyi, yj + hyj) - max(yi - hyi, yj - hyj)
            if ox > 0 and oy > 0:
                share = ox * oy / max(1e-9, min(4 * hxi * hyi, 4 * hxj * hyj))
                if share > max_overlap:
                    found.append(f"{i} ({names[i]}) and {j} ({names[j]}) overlap ({share:.0%} of the smaller footprint)")
    return found[:limit]


class RateLimiter:
    def __init__(self, rpm):
        self.interval, self.lock, self.next = 60. / rpm, threading.Lock(), 0.

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(start - now)


def request_parameters(env, model):
    """Everything sent besides the messages: chat-completions ``reasoning_effort`` from FASTFILL_LLM_REASONING_EFFORT
    (default medium; empty omits it). No temperature: reasoning models reject it."""
    effort = env.get("FASTFILL_LLM_REASONING_EFFORT", "medium")
    return {"model": model, **({"reasoning_effort": effort} if effort else {})}


USER_AGENT = "fastfill-llm-baseline/1.0"


def chat(env, model, messages, *, limiter=None, timeout=300., attempts=6):
    body = json.dumps({**request_parameters(env, model), "messages": [{"role": "system", "content": SYSTEM}, *messages]}).encode()
    for attempt in range(attempts):
        if limiter is not None:
            limiter.wait()
        request = urllib.request.Request(env["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions", data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {env['OPENAI_API_KEY']}",
            "User-Agent": USER_AGENT})  # Cloudflare-fronted relays answer urllib's default agent with 403
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as error:
            if error.code not in (408, 409, 425, 429) and error.code < 500 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):  # relay cut the body short
            if attempt == attempts - 1:
                raise
        time.sleep(min(120., 5. * 2 ** attempt))


def run(data, output, env_file, *, model=None, mode="prompt", repairs=2, workers=4, max_samples=None, retries=1, ask=chat):
    if mode not in ("prompt", "harness"):
        raise ValueError("mode must be prompt or harness")
    target = safe_output(output, create=True)
    env = load_env(env_file)
    model = model or env.get("OPENAI_MODEL")
    if not model:
        raise ValueError("pass --model or set OPENAI_MODEL in the env file")
    limiter = RateLimiter(float(env.get("FASTFILL_LLM_RPM", 30)))
    identity = {"data_sha256": fingerprint(data), "implementation_sha256": fingerprint(Path(__file__))}  # what ran, for caches
    rows = [p for p in map(project_minimal, read_samples(data)) if p is not None][:max_samples]
    calls = [0]

    def call(messages):
        calls[0] += 1
        return ask(env, model, messages) if ask is not chat else chat(env, model, messages, limiter=limiter)

    def one(row):
        start, error, condition, valid = time.perf_counter(), None, row["condition"], []  # valid: per answer received
        for attempt in range(1, retries + 2):  # a malformed first answer is asked again from scratch
            try:
                messages = [{"role": "user", "content": prompt(condition)}]
                reply = call(messages)
                valid.append(False)
                layout = to_layout(reply, condition)
                valid[-1] = True
                found, rounds, first = problems(layout, condition), 0, None
                first = len(found)
                while mode == "harness" and found and rounds < repairs:
                    messages += [{"role": "assistant", "content": reply}, {"role": "user", "content":
                                 "Your layout has these problems:\n- " + "\n- ".join(found)
                                 + "\nReturn the complete corrected JSON for every id."}]
                    reply = call(messages)
                    try:
                        candidate = to_layout(reply, condition)
                    except Exception:  # keep the last valid layout when a repair answer is malformed
                        break
                    rounds += 1
                    layout, found = candidate, problems(candidate, condition)
                return {"layout": layout, "error": None, "latency_s": time.perf_counter() - start, "attempts": attempt,
                        "first_answer_valid": valid[0], "repair_rounds": rounds, "problems_first": first,
                        "problems_final": len(found)}
            except Exception as exc:  # malformed or incomplete answers are failures, never dropped
                error = f"{type(exc).__name__}: {exc}"[:500]
        return {"layout": None, "error": error, "latency_s": time.perf_counter() - start, "attempts": retries + 1,
                "first_answer_valid": valid[0] if valid else None}

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
               "mean_problems_first": sum(r["problems_first"] for r in ok) / max(1, len(ok)),
               "mean_problems_final": sum(r["problems_final"] for r in ok) / max(1, len(ok)), "data": str(data), **identity}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--env", type=Path, required=True, help="file with OPENAI_BASE_URL / OPENAI_API_KEY [/ OPENAI_MODEL / FASTFILL_LLM_RPM]")
    p.add_argument("--model")
    p.add_argument("--mode", choices=("prompt", "harness"), default="prompt")
    p.add_argument("--repairs", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-samples", type=int)
    a = p.parse_args(argv)
    print(json.dumps(run(a.data, a.output, a.env, model=a.model, mode=a.mode, repairs=a.repairs, workers=a.workers,
                         max_samples=a.max_samples)))


if __name__ == "__main__":
    main()
