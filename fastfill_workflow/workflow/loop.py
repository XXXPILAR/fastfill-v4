"""Step ⑤ of critic_flowchart_v2: the budgeted check-and-repair loop (program proposals first, GPT on demand).

Each attempt takes the worst issue not tried yet (hard violations by magnitude while any are left, then soft costs of
at least ``min_soft_cost``) and builds a repair scope (flowchart §2 B):
  level 0  the issue's objects, their collision neighbours and their relation neighbours (step ①'s relations and the
           input's constraints, one hop), with everything standing on them
  level 1  + the functional groups they belong to
  level 2  + every object (hard issues only)
Fixed objects never move; objects outside the scope are obstacles, and every candidate is checked on the whole room.

Proposals (flowchart §2 C):
  program  the solver's local beam search over the scope at fixed sizes (no shrinking), near the current poses; a
           template issue (A far from / not facing B) enters the search as a temporary soft relation so the search
           proposes positions beside B. The critic picks among the complete states of the last beam.
  GPT      when every program level failed and ``gpt_repairs`` allows: one request with the room, every object's pose,
           the scope, the evidence (codes, objects, magnitudes) and earlier failed attempts; it answers new poses for
           scope objects, and may re-pick the support of an object whose support it inferred (the input's supports and
           hard ``on`` constraints stay locked). Children move with their parent.
Every candidate goes through ``Critic.better`` (flowchart §2 D, §3): accepted, or rolled back with the reason kept.
The loop stops when nothing is left to try, after ``stale_limit`` attempts in a row without an accepted change,
after ``max_attempts``, or when the program search has used ``search_seconds``. Each GPT request counts against
``gpt_repairs`` whether its candidate is accepted or not.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass

from . import llm as llm_client
from .catalog import SizeCatalog
from .critic import WEIGHTS, Assessment, Critic, Violation  # WEIGHTS' keys are the soft codes
from .drafting import WALL_BOTTOM_M
from .geometry import Geometry
from .local_repair import DRAFT_WEIGHT, FixedSizes, _with_children
from .planning import Plan, _number
from .solver import _pose, _wrap, beam_search

CANDIDATES_ASSESSED = 8  # complete beam states the critic scores per proposal
# the beam's cost per metre (and quarter turn) away from the current pose: hard issues keep step ④'s weight (stay
# near the draft); a soft issue usually needs a longer move (a member 2 m from its group) than one soft relation (3
# in the beam rank) pays for at that weight. The critic still decides whether the move is accepted.
SOFT_DRAFT_WEIGHT = .5


@dataclass
class LoopBudget:
    gpt_repairs: int = 2          # K_repair: GPT repair requests, accepted or not
    max_attempts: int = 24        # issues taken up
    stale_limit: int = 4          # attempts in a row without an accepted change
    search_seconds: float = 180.  # program search wall clock
    min_soft_cost: float = .05    # soft violations below this weighted cost are left alone


REPAIR_INSTRUCTIONS = """You repair part of a furniture layout. The user message is JSON: the room (floor polygon in \
metres, world x/y, z up; openings; fixed objects that cannot move), every object with its fixed size (width = \
left-right as seen from its front, depth = front-back, height), current pose and support, the objects you may move \
("movable"), the problems a program check found (with the objects involved and how large they are), the input's \
constraints, and earlier repair attempts that were rejected.

Answer with one JSON object only, no prose:
{"objects": {"<id>": {"x": x, "y": y, "yaw_deg": a}}}
Only movable objects may appear; leave out those that should stay. x, y: bottom-centre in world metres; yaw_deg: the \
direction the object's front faces, counter-clockwise degrees from world +x. An object whose "support_changeable" is \
true may also get "support": "floor", "wall" or another object's id when its current support cannot hold it; \
objects on a moved parent move with it unless you give them a pose. Do not change sizes.

Fix the listed problems without creating new ones: stay inside the floor polygon, off other objects and fixed \
objects, keep door strips clear, keep the input's constraints. Prefer small moves; keep related furniture together \
(seats with their table, nightstands beside the bed, etc.)."""


def _describe(v: Violation) -> str:
    unit = {"template_faces": "of facing error (normalised)", "relation_plan": "(normalised gap)",
            "relation_user": "(normalised gap)", "access": "(missing free share, normalised)"}
    return f"{v.code} {list(v.ids)}: {v.magnitude:.3f}{' m' if v.code in HARD_M else ' ' + unit.get(v.code, '')}" + \
        (f" ({v.detail})" if v.detail else "")


HARD_M = {"collision", "boundary", "vertical_bounds", "floor_support", "wall_support", "object_support",
          "door_clearance", "window_clearance", "keepout", "template_near"}


class Loop:
    def __init__(self, geometry: Geometry, plan: Plan, sizes: dict[str, list[float]], catalog: SizeCatalog,
                 critic: Critic, *, locked_parents: dict[str, str], ask=None, budget: LoopBudget | None = None,
                 beam_width: int = 24, candidate_limit: int = 48, draft_weight: float = DRAFT_WEIGHT):
        self.g, self.plan, self.sizes, self.catalog, self.critic = geometry, plan, sizes, catalog, critic
        self.locked = locked_parents
        self.ask = ask
        self.budget = budget or LoopBudget()
        self.beam_width, self.candidate_limit, self.draft_weight = beam_width, candidate_limit, draft_weight
        self.fixed_sizes = FixedSizes(sizes, catalog, allow_shrink=False)
        self.search_s = 0.
        self.gpt_calls: list[dict] = []
        self.failed: list[str] = []

    # ---------------------------------------------------------------- scope
    def scope(self, ids: tuple[str, ...], level: int, current: Assessment) -> set[str]:
        requests = set(self.g.requests)
        base = {i for i in ids if i in requests}
        result = set(base)
        for v in current.hard:
            if v.code == "collision" and set(v.ids) & base:
                result |= set(v.ids) & requests
        for r in self.plan.relations + self.g.explicit:
            linked = {r.get("object_id"), r.get("target_id"), r.get("parent_id"), *(r.get("target_ids") or [])}
            if linked & base:
                result |= linked & requests
        if level >= 1:
            for group in self.plan.groups:
                members = {group["anchor"], *group["members"]} & requests
                if members & result:
                    result |= members
        if level >= 2:
            result = set(requests)
        return _with_children(self.plan, result)

    # ---------------------------------------------------------------- program proposal
    def _extra_relations(self, target: Violation) -> list[dict]:
        if target.code == "template_near":
            return [{"type": "near", "object_id": target.ids[0], "target_id": target.ids[1], "max_distance_m": .25,
                     "hard": False, "origin": "critic"}]
        if target.code == "template_faces":
            return [{"type": "faces", "object_id": target.ids[0], "target_id": target.ids[1], "hard": False,
                     "tolerance_rad": math.radians(40), "origin": "critic"}]
        return []

    def program(self, scope: set[str], target: Violation, placed: dict[str, dict]) -> list[dict[str, dict]]:
        geometry = self.g
        extra = self._extra_relations(target)
        if extra:
            geometry = copy.copy(self.g)
            geometry.soft = self.g.soft + extra
        saved = self.plan.draft_poses
        self.plan.draft_poses = {i: {"x": b["bottom_center_m"][0], "y": b["bottom_center_m"][1], "yaw_rad": b["yaw_rad"]}
                                 for i, b in placed.items()}
        start = time.perf_counter()
        try:
            frozen = {i: b for i, b in placed.items() if i not in scope}
            _state, trace = beam_search(geometry, self.fixed_sizes, self.plan, mode="local", beam_width=self.beam_width,
                                        candidate_limit=self.candidate_limit, density=1, initial=frozen, subset=scope,
                                        draft_weight=SOFT_DRAFT_WEIGHT if target.code in WEIGHTS else self.draft_weight)
        finally:
            self.plan.draft_poses = saved
            self.search_s += time.perf_counter() - start
        candidates = list(trace.get("complete") or [])[:CANDIDATES_ASSESSED]
        if not candidates and trace.get("partial"):
            candidates = [{**placed, **trace["partial"]}]
        return candidates

    # ---------------------------------------------------------------- GPT proposal
    def _payload(self, scope: set[str], target: Violation, current: Assessment, placed: dict[str, dict]) -> str:
        objects = []
        for obj in self.g.condition["objects"]:
            ident, box = obj["id"], placed[obj["id"]]
            depth, width, height = box["target_size_local_m"]
            objects.append({"id": ident, "category": obj["category"], "support": self.plan.parents[ident],
                            "size_m": {"width": round(width, 3), "depth": round(depth, 3), "height": round(height, 3)},
                            "x": round(box["bottom_center_m"][0], 3), "y": round(box["bottom_center_m"][1], 3),
                            "z": round(box["bottom_center_m"][2], 3),
                            "yaw_deg": round(math.degrees(box["yaw_rad"]), 1), "movable": ident in scope,
                            "support_changeable": ident in scope and ident not in self.locked})
        involved = [v for v in current.hard + current.soft if set(v.ids) & scope]
        involved.sort(key=lambda v: (v not in current.hard, -v.magnitude))
        room = {k: v for k, v in self.g.room.items() if k in ("floor_polygon_xy_m", "floor_z_m", "height_m", "openings",
                                                               "fixed_objects")}
        return json.dumps({"room": room, "objects": objects, "constraints": self.g.condition["constraints"],
                           "problems": [_describe(target)] + [_describe(v) for v in involved if v != target][:15],
                           "rejected_attempts": self.failed[-4:]}, ensure_ascii=False)

    def _apply(self, answer: dict, scope: set[str], placed: dict[str, dict]) -> tuple[dict[str, dict], dict[str, str]]:
        """New boxes from the answer: poses for scope objects, supports re-picked where allowed, children carried."""
        parents = dict(self.plan.parents)
        poses: dict[str, dict] = {}
        for ident, entry in (answer.get("objects") or {}).items():
            if ident not in scope or not isinstance(entry, dict):
                continue
            x, y, yaw = (_number(entry.get(k), -1e4, 1e4) for k in ("x", "y", "yaw_deg"))
            if None in (x, y, yaw):
                continue
            poses[ident] = {"x": x, "y": y, "yaw_rad": math.remainder(math.radians(yaw), 2 * math.pi),
                            "z": _number(entry.get("z"), -1e3, 1e3)}
            support = entry.get("support")
            if isinstance(support, str) and ident not in self.locked and support != ident and \
                    (support in ("floor", "wall") or support in self.g.requests or support in self.g.fixed):
                trial = {**parents, ident: support}
                seen, cur = {ident}, support
                while cur in trial and cur not in ("floor", "wall"):
                    if cur in seen:
                        break
                    seen.add(cur)
                    cur = trial[cur]
                else:
                    parents[ident] = support
        new = dict(placed)
        order, done = [], set()

        def visit(ident: str) -> None:
            if ident in done:
                return
            parent = parents.get(ident)
            if parent in self.g.requests:
                visit(parent)
            done.add(ident)
            order.append(ident)

        moved = _with_children(Plan(parents), set(poses))
        for ident in moved:
            visit(ident)
        for ident in order:
            if ident not in moved:
                continue
            parent, old, size = parents[ident], placed[ident], self.sizes[ident]
            pose = poses.get(ident)
            if pose is None:  # carried by a moved parent: same offset and turn in the parent's frame
                anchor_old, anchor_new = placed.get(parent), new.get(parent)
                if anchor_old is None or anchor_new is None:
                    continue
                turn = anchor_new["yaw_rad"] - anchor_old["yaw_rad"]
                dx = old["bottom_center_m"][0] - anchor_old["bottom_center_m"][0]
                dy = old["bottom_center_m"][1] - anchor_old["bottom_center_m"][1]
                c, s = math.cos(turn), math.sin(turn)
                pose = {"x": anchor_new["bottom_center_m"][0] + c * dx - s * dy,
                        "y": anchor_new["bottom_center_m"][1] + s * dx + c * dy,
                        "yaw_rad": _wrap(old["yaw_rad"] + turn), "z": None}
            if parent == "floor":
                z = self.g.floor
            elif parent == "wall":
                top = self.g.ceiling - size[2] - .03
                given = pose.get("z")
                z = given if given is not None and self.g.floor <= given <= top else \
                    (old["bottom_center_m"][2] if self.plan.parents[ident] == "wall" else min(self.g.floor + WALL_BOTTOM_M, top))
            else:
                parent_box = new.get(parent, self.g.fixed.get(parent))
                if parent_box is None:
                    continue
                z = self.g._support_height(_pose(ident, size, pose["x"], pose["y"], 0, pose["yaw_rad"], parent),
                                           parent, parent_box)
            new[ident] = _pose(ident, size, pose["x"], pose["y"], z, pose["yaw_rad"], parent)
        return new, parents

    def gpt(self, scope: set[str], target: Violation, current: Assessment,
            placed: dict[str, dict]) -> tuple[dict[str, dict] | None, dict[str, str] | None, str]:
        try:
            text, call = self.ask(REPAIR_INSTRUCTIONS, self._payload(scope, target, current, placed))
        except (RuntimeError, ValueError) as exc:
            self.gpt_calls.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
            return None, None, "request failed"
        self.gpt_calls.append(call)
        first, last = text.find("{"), text.rfind("}")
        try:
            answer = json.loads(text[first:last + 1]) if 0 <= first < last else None
        except json.JSONDecodeError:
            answer = None
        if not isinstance(answer, dict):
            return None, None, "answer is not a JSON object"
        new, parents = self._apply(answer, scope, placed)
        if new == placed and parents == self.plan.parents:
            return None, None, "answer moved nothing"
        return new, parents, ""

    # ---------------------------------------------------------------- loop
    def _targets(self, current: Assessment, tried: set) -> list[Violation]:
        pool = sorted(current.hard, key=lambda v: -v.magnitude) if current.hard else \
            sorted((v for v in current.soft if v.cost >= self.budget.min_soft_cost), key=lambda v: -v.cost)
        return [v for v in pool if (v.code, v.ids) not in tried]

    def _pick(self, candidates: list[dict[str, dict]], critic: Critic) -> tuple[dict[str, dict] | None, Assessment | None]:
        scored = [(critic.assess(c), c) for c in candidates]
        if not scored:
            return None, None
        best = min(scored, key=lambda t: t[0].key())
        return best[1], best[0]

    def run(self, placed: dict[str, dict]) -> tuple[dict[str, dict], dict]:
        start = time.perf_counter()
        current = self.critic.assess(placed)
        info = {"initial": current.summary(), "attempts": [], "termination": "max_attempts"}
        tried, stale = set(), 0
        for _ in range(self.budget.max_attempts):
            targets = self._targets(current, tried)
            if not targets:
                info["termination"] = "no issue left to try"
                break
            if self.search_s >= self.budget.search_seconds:
                info["termination"] = "search time budget"
                break
            target = targets[0]
            tried.add((target.code, target.ids))
            accepted, seen = False, []
            levels = (0, 1, 2) if current.hard else (0, 1)
            for level in levels:
                scope = self.scope(target.ids, level, current)
                if scope in seen or not scope:
                    continue
                seen.append(scope)
                t0 = time.perf_counter()
                candidate, after = self._pick(self.program(scope, target, placed), self.critic)
                ok, reason = self.critic.better(after, current) if after else (False, "no candidate")
                info["attempts"].append({"target": target.as_dict(), "level": level, "scope": len(scope),
                                         "source": "program", "accepted": ok, "reason": reason,
                                         "before": current.key(), "after": after.key() if after else None,
                                         "seconds": round(time.perf_counter() - t0, 3)})
                if ok:
                    placed, current, accepted = candidate, after, True
                    break
            if not accepted and self.ask is not None and len(self.gpt_calls) < self.budget.gpt_repairs:
                scope = self.scope(target.ids, 1, current)
                t0 = time.perf_counter()
                candidate, parents, problem = self.gpt(scope, target, current, placed)
                critic = self.critic
                if candidate is not None and parents != self.plan.parents:
                    critic = self.critic.with_geometry(_regeometry(self.g, parents))
                after = critic.assess(candidate) if candidate is not None else None
                ok, reason = critic.better(after, current) if after else (False, problem)
                info["attempts"].append({"target": target.as_dict(), "level": 1, "scope": len(scope), "source": "gpt",
                                         "accepted": ok, "reason": reason, "before": current.key(),
                                         "after": after.key() if after else None,
                                         "seconds": round(time.perf_counter() - t0, 3)})
                if ok:
                    if parents != self.plan.parents:  # the geometry and critic read plan.parents (same dict)
                        changed = {k: (self.plan.parents[k], v) for k, v in parents.items() if self.plan.parents[k] != v}
                        self.plan.parents.update(parents)
                        info.setdefault("support_changes", {}).update({k: list(v) for k, v in changed.items()})
                    placed, current, accepted = candidate, after, True
                else:
                    self.failed.append(f"moving {sorted(scope)} for {_describe(target)} was rejected: {reason}")
            stale = 0 if accepted else stale + 1
            if stale >= self.budget.stale_limit:
                info["termination"] = "no improvement"
                break
        info.update(final=current.summary(), gpt_calls=self.gpt_calls, search_seconds=round(self.search_s, 3),
                    seconds=round(time.perf_counter() - start, 3))
        return placed, info


class FocusedLoop(Loop):
    """The loop restricted to ``movable`` objects (``orders``' step-by-step modes): only issues involving them are
    taken up, only they move; everything else is frozen."""

    def __init__(self, *args, movable: set[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.movable = set(movable)

    def _targets(self, current: Assessment, tried: set) -> list[Violation]:
        mine = [v for v in current.hard if set(v.ids) & self.movable and (v.code, v.ids) not in tried]
        if mine:
            return sorted(mine, key=lambda v: -v.magnitude)
        return sorted((v for v in current.soft if set(v.ids) & self.movable and (v.code, v.ids) not in tried
                       and v.cost >= self.budget.min_soft_cost), key=lambda v: -v.cost)

    def scope(self, ids: tuple[str, ...], level: int, current: Assessment) -> set[str]:
        return super().scope(ids, level, current) & self.movable & set(self.g.requests)


def _regeometry(geometry: Geometry, parents: dict[str, str]) -> Geometry:
    """The same geometry with another support map (a GPT candidate that re-picks supports is checked with it)."""
    other = copy.copy(geometry)
    other.parents = parents
    return other


def critic_loop(geometry: Geometry, plan: Plan, sizes: dict[str, list[float]], placed: dict[str, dict],
                catalog: SizeCatalog, critic: Critic, *, locked_parents: dict[str, str], ask=None,
                budget: LoopBudget | None = None, beam_width: int = 24, candidate_limit: int = 48,
                draft_weight: float = DRAFT_WEIGHT) -> tuple[dict[str, dict], dict]:
    return Loop(geometry, plan, sizes, catalog, critic, locked_parents=locked_parents, ask=ask, budget=budget,
                beam_width=beam_width, candidate_limit=candidate_limit, draft_weight=draft_weight).run(placed)


def make_ask(model: str | None, base_url: str | None, api_key: str | None):
    """The GPT repair request through the workflow's official-API client."""
    def ask(instructions: str, prompt: str):
        return llm_client.ask(instructions, prompt, model=model, base_url=base_url, api_key=api_key)
    return ask
