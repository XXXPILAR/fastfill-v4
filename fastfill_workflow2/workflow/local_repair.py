"""Step ④ of the draft workflow: local constraint solving around the draft.

Only objects named by a hard issue move, together with whatever stands on them; every other object stays where the
draft put it and acts as an obstacle. Sizes and supports are fixed (steps ② and ①); the search chooses position and
yaw with the existing beam search in "local" mode: candidates nearest the draft first (5 and 10 cm nudges, 0.2 and
0.4 m rings), then wall slots, relation and group positions, the room grid, and a bounded shrink only when nothing
else fits. The score keeps hard constraints, rewards soft relations, and charges ``draft_weight`` per metre (and per
quarter turn) away from the draft.

When a level fails the moving set grows: the issue objects, then their whole functional groups, then the room (the
same beam search over every object, still at fixed sizes). If the room level fails too, the result is the version
with the fewest hard issues among the draft and each level's best partial assignment (unplaced objects at their
draft poses), so a layout always comes back.
"""

from __future__ import annotations

import time

from .catalog import SizeCatalog, _key
from .geometry import Geometry
from .planning import Plan
from .solver import beam_search

DRAFT_WEIGHT = 2.0          # per metre away from the draft; one violated soft relation costs 3 in the beam rank
SHRINK_FACTORS = {1: .95, 2: .90}


class FixedSizes:
    """Catalog interface for the solver with one size per object; ``shrunk`` scales it uniformly by 5 or 10 %,
    never below the train catalog's 15th percentile on an axis (and never growing an axis already below it).
    ``allow_shrink=False`` (the critic loop's default) keeps every size: a bounded search that finds no full-size
    pose does not show that the size cannot fit, and shrinking would trade size accuracy for fewer collisions."""

    def __init__(self, sizes: dict[str, list[float]], catalog: SizeCatalog, allow_shrink: bool = True):
        self.sizes, self.catalog, self.allow_shrink = sizes, catalog, allow_shrink

    def candidates(self, request: dict, llm_hint=None) -> list[tuple[list[float], float, str]]:
        return [(self.sizes[request["id"]], 0., "fixed")]

    def shrunk(self, request: dict, level: int) -> list[tuple[list[float], float, str]]:
        if not self.allow_shrink:
            return []
        size = self.sizes[request["id"]]
        record = self.catalog.categories.get(_key(request["category"]))
        floor = record["p15"] if record else [0.] * 3
        factor = SHRINK_FACTORS[level]
        shrunk = [round(min(v, max(v * factor, lo)), 4) for v, lo in zip(size, floor)]
        return [(shrunk, -1.5 * level, f"shrink{level}")] if shrunk != size else []


def _with_children(plan: Plan, ids: set[str]) -> set[str]:
    result, changed = set(ids), True
    while changed:
        before = len(result)
        result |= {ident for ident, parent in plan.parents.items() if parent in result}
        changed = len(result) != before
    return result


def local_repair(geometry: Geometry, plan: Plan, sizes: dict[str, list[float]], placed: dict[str, dict],
                 catalog: SizeCatalog, *, draft_weight: float = DRAFT_WEIGHT, beam_width: int = 24,
                 candidate_limit: int = 48) -> tuple[dict[str, dict], dict]:
    """Returns (placed, info); ``plan.draft_poses`` must hold the draft poses the score stays near."""
    issues = geometry.validate(placed)
    info = {"issues_before": len(issues), "levels": [], "result": "clean_draft"}
    if not issues:
        return placed, info
    fixed_sizes = FixedSizes(sizes, catalog)
    bad = {ident for issue in issues for ident in issue.ids if ident in geometry.requests}
    grouped = set(bad)
    for group in plan.groups:
        members = {group["anchor"], *group["members"]} & set(geometry.requests)
        if members & bad:
            grouped |= members
    levels, seen = [], []
    for name, ids in (("objects", bad), ("groups", grouped), ("room", set(geometry.requests))):
        moving = _with_children(plan, ids)
        if moving not in seen:
            seen.append(moving)
            levels.append((name, moving))
    best, best_count = placed, len(issues)
    for name, moving in levels:
        start = time.perf_counter()
        frozen = {ident: box for ident, box in placed.items() if ident not in moving}
        state, trace = beam_search(geometry, fixed_sizes, plan, mode="local", beam_width=beam_width,
                                   candidate_limit=candidate_limit, density=1, initial=frozen, subset=moving,
                                   draft_weight=draft_weight)
        record = {"level": name, "moving": len(moving), "seconds": round(time.perf_counter() - start, 3),
                  "solved": state is not None, "blocked": trace.get("blocked", {})}
        info["levels"].append(record)
        if state is not None:
            info["result"] = f"repaired_{name}"
            info["shrunk"] = sorted(ident for ident, box in state.placed.items()
                                    if box["target_size_local_m"] != sizes[ident])
            return state.placed, info
        partial = {**placed, **trace.get("partial", {})}
        count = len(geometry.validate(partial))
        record["partial_issues"] = count
        if count < best_count:
            best, best_count = partial, count
    info["result"] = "best_effort"
    info["issues_after"] = best_count
    return best, info
