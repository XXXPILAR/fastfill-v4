"""Step ③ of the draft workflow: one LLM call places every object at its fixed size.

Sizes come from step ② (one per object, e.g. FastFill's prediction); support, priority, groups and relations from
step ① (``make_plan(..., semantic_only=True)``). The LLM answers positions and yaws only; heights follow from the
support (the floor, a wall-mounted default, or the top of the parent), and sizes stay the given ones whatever the
answer says. An object the answer misses or gets wrong is still placed (on its parent, else at the room's
representative point) and counted in ``missing``, so the draft is always complete.
"""

from __future__ import annotations

import json
import math

from . import llm as llm_client
from .geometry import Geometry
from .planning import Plan, _number
from .solver import _pose

WALL_BOTTOM_M = 1.15

DRAFT_INSTRUCTIONS = """You place furniture in a room. The user message is JSON: the room (floor polygon in metres, \
world x/y, z up; openings on its walls; fixed objects that cannot move, with bottom centre, size and yaw), the \
objects to place and the constraints, plus a plan: each object's support, placement priority, functional groups and \
soft relations. Every object's size is already fixed (size_m: width = left-right as seen from its front, depth = \
front-back, height); do not change sizes.

Answer with one JSON object only, no prose:
{"objects": {"<id>": {"x": x, "y": y, "yaw_deg": a}}}
optionally with "z" (bottom height above the floor, metres) for objects supported by "wall".

Rules:
- Include every object id exactly once; do not add or remove objects.
- x, y: the object's bottom-centre position in world metres. yaw_deg: the direction its front faces, counter-clockwise \
degrees from the world +x axis; the depth axis points along it.
- An object supported by another object sits on top of it: give x, y within the parent's top.
- Hard constraints must hold. Keep objects inside the floor polygon, off each other and off fixed objects, and keep \
doors clear. Follow the plan's groups and relations where you can.
- Wall-supported objects hang with their back against a wall, facing into the room."""

# appended to DRAFT_INSTRUCTIONS when the objects carry a learned facing (``suggested_yaw_deg``)
SUGGESTED_YAW_RULE = """
- suggested_yaw_deg, when given, is the facing a model trained on real rooms predicts for the object (same convention \
as yaw_deg). Use it as the object's yaw and choose the position to suit it (e.g. the wall its back goes against). \
Depart from it only where it cannot work: it would push the object through a wall, block a door, or collide; then \
choose a yaw that fits."""


def request_text(condition: dict, plan: Plan, sizes: dict[str, list[float]],
                 yaws: dict[str, float] | None = None) -> str:
    objects = []
    for obj in condition["objects"]:
        ident = obj["id"]
        depth, width, height = sizes[ident]
        entry = {"id": ident, "category": obj["category"], "description": obj.get("description", ""),
                 "support": plan.parents[ident],
                 "size_m": {"width": round(width, 3), "depth": round(depth, 3), "height": round(height, 3)}}
        if ident in plan.priority:
            entry["priority"] = plan.priority[ident]
        if yaws and ident in yaws:
            entry["suggested_yaw_deg"] = round(math.degrees(math.remainder(float(yaws[ident]), 2 * math.pi)), 1)
        objects.append(entry)
    relations = [{k: v for k, v in r.items() if k not in ("origin", "hard")}
                 for r in plan.relations if r.get("origin") != "group"]
    payload = {"room": condition["room"], "objects": objects, "constraints": condition["constraints"],
               "plan": {"groups": plan.groups, "relations": relations}}
    return json.dumps(payload, ensure_ascii=False)


def parse_poses(text: str, ids: set[str]) -> tuple[dict[str, dict], dict]:
    first, last = text.find("{"), text.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("LLM response does not contain a JSON object")
    answer = json.loads(text[first:last + 1])
    entries = answer.get("objects") if isinstance(answer, dict) else None
    if not isinstance(entries, dict):
        raise ValueError("LLM response has no objects map")
    poses, dropped = {}, {"unknown_object": 0, "pose_invalid": 0}
    for ident, entry in entries.items():
        if ident not in ids:
            dropped["unknown_object"] += 1
            continue
        if not isinstance(entry, dict):
            dropped["pose_invalid"] += 1
            continue
        x, y = _number(entry.get("x"), -1e4, 1e4), _number(entry.get("y"), -1e4, 1e4)
        yaw = _number(entry.get("yaw_deg"), -1e4, 1e4)
        if None in (x, y, yaw):
            dropped["pose_invalid"] += 1
            continue
        poses[ident] = {"x": x, "y": y, "yaw_rad": math.remainder(math.radians(yaw), 2 * math.pi),
                        "z": _number(entry.get("z"), -1e3, 1e3)}
    return poses, {k: v for k, v in dropped.items() if v}


def assemble(geometry: Geometry, plan: Plan, sizes: dict[str, list[float]],
             poses: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """Boxes for every request, parents before children; returns (placed, ids without a usable pose)."""
    placed, missing = {}, []
    anchor = geometry.polygon.representative_point()

    def put(ident: str) -> None:
        if ident in placed:
            return
        parent = plan.parents.get(ident, "floor")
        if parent in geometry.requests:
            put(parent)
        parent_box = placed.get(parent, geometry.fixed.get(parent))
        if parent not in ("floor", "wall") and parent_box is None:
            parent = "floor"
        size, pose = sizes[ident], poses.get(ident)
        if pose is None:
            missing.append(ident)
            if parent_box is not None:
                x, y = parent_box["bottom_center_m"][:2]
                yaw = parent_box["yaw_rad"]
            else:
                x, y, yaw = anchor.x, anchor.y, 0.0
        else:
            x, y, yaw = pose["x"], pose["y"], pose["yaw_rad"]
        if parent == "floor":
            z = geometry.floor
        elif parent == "wall":
            top = geometry.ceiling - size[2] - .03
            given = pose.get("z") if pose else None
            z = geometry.floor + given if given is not None and 0 <= given <= top - geometry.floor else \
                min(geometry.floor + WALL_BOTTOM_M, top)
            z = max(z, geometry.floor)
        else:
            z = geometry._support_height(_pose(ident, size, x, y, 0, yaw, parent), parent, parent_box)
        placed[ident] = _pose(ident, size, x, y, z, yaw, parent)

    for ident in geometry.requests:
        put(ident)
    return placed, missing


def make_draft(condition: dict, plan: Plan, sizes: dict[str, list[float]], geometry: Geometry, *,
               yaws: dict[str, float] | None = None,
               model: str | None = None, base_url: str | None = None,
               api_key: str | None = None) -> tuple[dict[str, dict], dict]:
    """Returns (placed, info) with info = {calls, dropped, missing, error}. ``yaws`` (rad, e.g. FastFill's) go to
    the LLM as ``suggested_yaw_deg``: it decides each position for that facing instead of having it swapped in."""
    info: dict = {"calls": [], "dropped": {}, "missing": [], "error": None}
    poses: dict[str, dict] = {}
    try:
        instructions = DRAFT_INSTRUCTIONS + (SUGGESTED_YAW_RULE if yaws else "")
        text, call = llm_client.ask(instructions, request_text(condition, plan, sizes, yaws),
                                    model=model, base_url=base_url, api_key=api_key)
        info["calls"].append(call)
        poses, info["dropped"] = parse_poses(text, set(geometry.requests))
    except (RuntimeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and "requires --model" in str(exc):
            raise
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
    placed, info["missing"] = assemble(geometry, plan, sizes, poses)
    return placed, info
