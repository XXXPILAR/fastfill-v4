"""Three orders of generating and checking a layout from one plan and one set of sizes (experiments/run_orders.py).

  A  all at once      one LLM call places every object (step ③, ``drafting.make_draft``); the critic loop then repairs
                      the whole room (``loop.Loop``)
  B  one by one       objects in the solver's order (supports first, then the plan's priority), ``batch`` objects per
                      LLM call (1: one at a time). Each call sees the objects placed so far (they stay) and the ones
                      still to come (sizes only, so it can leave room); right after it the critic checks the room
                      placed so far and repairs only the new objects (``loop.FocusedLoop``), which are then frozen
  C  group by group   the plan's functional groups (an anchor and its members); an object in no group joins the group
                      of what it stands on, and the floor and wall objects left over form one more group. Groups go by
                      their most important member (step ①'s priority, 1 = first; unranked groups by the solver's
                      order), and an object whose support would come in a later group joins the support's group.
                      Within a group, objects keep the solver's order (supports first). One LLM call per group, then
                      check and repair of that group only, then frozen
Shared by the three: the plan (made once, passed in, so the modes differ in nothing else), the sizes and suggested yaws,
the critic (doors hard under the evaluator's rule, the even half's templates), fixed sizes, one GPT repair budget per
room (``gpt_repairs``, shared by B's and C's steps), the program repair (by default every step runs until its issues
are tried, ``loop.LoopBudget``; an explicit ``loop_budget.search_seconds`` is split over the steps still to run, as in
the recorded order experiments), and at the end the critic-judged polish and the final assessment.
``run_order_variants`` also returns B_final / C_final: the same step-by-step state followed by a whole-room loop with
A's budget (an ablation separating "how the layout was generated" from "who may move in the repair").
The search budget is checked between attempts, so one beam search can run past it: report measured seconds.
"""

from __future__ import annotations

import copy
import dataclasses
import time

from .catalog import SizeCatalog
from .contract import check_request, fill_room_defaults, validate_condition, validate_layout
from .critic import Critic, load_templates
from .drafting import assemble, make_draft, make_partial_draft
from .geometry import Geometry
from .local_repair import DRAFT_WEIGHT, FixedSizes
from .loop import FocusedLoop, Loop, LoopBudget, make_ask
from .planning import Plan, _required_parents
from .refine import refine
from .solver import _order

MODES = ("A", "B", "C")


def plan_to_dict(plan: Plan) -> dict:
    return dataclasses.asdict(plan)


def plan_from_dict(data: dict) -> Plan:
    return Plan(**copy.deepcopy(data))


def critic_geometry(condition: dict, plan: Plan) -> Geometry:
    return Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False, door_rule="evaluator")


def sub_room(condition: dict, plan: Plan, ids, templates: list[dict]) -> tuple[Geometry, Critic]:
    """Geometry and critic of the room holding only ``ids`` (plus its fixed objects): the input's constraints and
    the plan's relations whose objects are all there."""
    ids = set(ids)
    sub = copy.deepcopy(condition)
    sub["objects"] = [o for o in sub["objects"] if o["id"] in ids]
    known = ids | {f["id"] for f in sub["room"].get("fixed_objects", [])}

    def inside(r: dict) -> bool:
        refs = {r.get("object_id"), r.get("target_id"), r.get("parent_id"), *(r.get("target_ids") or [])} - {None}
        return refs <= known
    sub["constraints"] = [r for r in sub["constraints"] if inside(r)]
    geometry = critic_geometry(sub, plan)
    geometry.soft = [r for r in geometry.soft if inside(r)]
    critic = Critic(geometry, [r for r in geometry.explicit if not r.get("hard", True)],
                    [r for r in plan.relations if inside(r)], templates)
    return geometry, critic


def functional_steps(geometry: Geometry, plan: Plan, order: list[str]) -> list[list[str]]:
    requests, position = set(geometry.requests), {ident: k for k, ident in enumerate(order)}
    group_of: dict[str, int] = {}
    for k, group in enumerate(plan.groups):
        for member in [group["anchor"], *group["members"]]:
            if member in requests:
                group_of.setdefault(member, k)

    def key(ident: str, seen: tuple = ()) -> tuple:
        if ident in group_of:
            return ("group", group_of[ident])
        parent = plan.parents.get(ident)
        if parent in requests and parent not in seen:
            return key(parent, seen + (ident,))
        return ("rest",)
    assign = {ident: key(ident) for ident in order}
    unranked = len(order) + 1

    def rank(members: list[str]) -> tuple:
        """Groups go by their most important member (the plan's priority, 1 = first), then by solver order."""
        return (min((plan.priority.get(m, unranked) for m in members), default=unranked),
                min(position[m] for m in members))

    def grouped() -> dict[tuple, list[str]]:
        groups: dict[tuple, list[str]] = {}
        for ident in order:
            groups.setdefault(assign[ident], []).append(ident)
        return groups
    changed = True
    while changed:  # an object never comes before its support: it joins the support's group if that goes later
        changed, groups = False, grouped()
        for ident in order:
            parent = plan.parents.get(ident)
            if parent in assign and assign[ident] != assign[parent] and \
                    rank(groups[assign[parent]]) > rank(groups[assign[ident]]):
                assign[ident], changed = assign[parent], True
                break
    return sorted(grouped().values(), key=rank)  # within a group: solver order (supports first)


def steps_for(mode: str, geometry: Geometry, plan: Plan, sizes: dict[str, list[float]], catalog: SizeCatalog,
              batch: int = 1) -> list[list[str]]:
    order = _order(geometry, FixedSizes(sizes, catalog), plan, set(geometry.requests))
    if mode == "A":
        return [order]
    if mode == "B":
        return [order[k:k + batch] for k in range(0, len(order), batch)]
    return functional_steps(geometry, plan, order)


def run_order(condition: dict, sizes: dict[str, list[float]] | None, plan: Plan, mode: str, *,
              final_repair: bool = False, **kwargs) -> dict:
    """One mode: {status, layout, diagnostics, timing_s} like ``pipeline.run_draft_workflow``."""
    results = run_order_variants(condition, sizes, plan, mode, final_repair=(final_repair,), **kwargs)
    return next(iter(results.values()))


def variant_name(mode: str, final_repair: bool) -> str:
    return f"{mode}_final" if final_repair and mode != "A" else mode


def run_order_variants(condition: dict, sizes: dict[str, list[float]] | None, plan: Plan, mode: str, *,
                       yaws: dict[str, float] | None = None, catalog_path: str | None = None, llm: bool = True,
                       model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                       gpt_repairs: int = 2, loop_budget: LoopBudget | None = None, batch: int = 1,
                       final_repair: tuple[bool, ...] = (False,), polish: bool = True,
                       templates_path: str | None = None, beam_width: int = 24,
                       candidate_limit: int = 48) -> dict[str, dict]:
    """The mode's steps once, then one result per ``final_repair`` value, keyed by ``variant_name``: B and C with
    ``final_repair=True`` add a whole-room loop after their last step, from the same step-by-step state and with the
    same budget as A's whole-room loop (``loop_budget``: GPT repairs, search seconds, attempts), so B vs B_final
    differs only in that repair and A vs B_final / C_final only in how the layout was generated. ``plan`` is copied."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    start = time.perf_counter()
    validate_condition(condition)
    condition, notes = fill_room_defaults(condition)
    condition = check_request(condition)
    plan = plan_from_dict(plan_to_dict(plan))
    catalog, fixed_sizes, size_source = SizeCatalog(catalog_path), {}, {"given": 0, "catalog": 0}
    for obj in condition["objects"]:
        size = (sizes or {}).get(obj["id"])
        if size is not None and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size):
            fixed_sizes[obj["id"]], size_source["given"] = [float(v) for v in size], size_source["given"] + 1
        else:
            fixed_sizes[obj["id"]], size_source["catalog"] = catalog.candidates(obj)[0][0], size_source["catalog"] + 1
    templates = load_templates(templates_path)
    geometry = critic_geometry(condition, plan)
    critic = Critic.for_plan(geometry, plan, templates)
    locked = _required_parents(condition)
    budget = loop_budget or LoopBudget(gpt_repairs=gpt_repairs)
    ask = make_ask(model, base_url, api_key) if llm and budget.gpt_repairs > 0 else None
    common = {"locked_parents": locked, "beam_width": beam_width, "candidate_limit": candidate_limit,
              "draft_weight": DRAFT_WEIGHT}
    steps = steps_for(mode, geometry, plan, fixed_sizes, catalog, batch)
    records, draft_calls, gpt_calls, search_left = [], [], [], budget.search_seconds or 0.
    placed: dict[str, dict] = {}
    for k, ids in enumerate(steps):
        t0 = time.perf_counter()
        if not llm:  # offline: every new object at its fallback (on its parent, else the room's representative point)
            placed, missing = assemble(geometry, plan, fixed_sizes, {}, ids=ids, placed=placed)
            draft = {"calls": [], "dropped": {}, "missing": missing, "error": "llm disabled"}
        elif mode == "A":
            placed, draft = make_draft(condition, plan, fixed_sizes, geometry, yaws=yaws, model=model,
                                       base_url=base_url, api_key=api_key)
        else:
            placed, draft = make_partial_draft(condition, plan, fixed_sizes, geometry, ids, placed, yaws=yaws,
                                               model=model, base_url=base_url, api_key=api_key)
        drafted = time.perf_counter()
        draft_calls += draft["calls"]
        if mode == "A":
            step_geometry, step_critic, cls, extra = geometry, critic, Loop, {}
        else:
            step_geometry, step_critic = sub_room(condition, plan, placed, templates)
            cls, extra = FocusedLoop, {"movable": set(ids)}
        left = max(budget.gpt_repairs - len(gpt_calls), 0)
        if budget.search_seconds is None:  # the default: every step runs until its issues are tried (loop.LoopBudget)
            step_budget = dataclasses.replace(budget, gpt_repairs=left)
        else:  # an explicit search budget (the recorded order experiments): shared out between the steps as before
            step_budget = dataclasses.replace(budget, gpt_repairs=left, search_seconds=search_left / (len(steps) - k),
                                              max_attempts=budget.max_attempts if mode == "A" else
                                              min(budget.max_attempts or 24, 3 * len(ids) + 4))
        before = step_critic.assess(placed)
        loop = cls(step_geometry, plan, fixed_sizes, catalog, step_critic, ask=ask if left else None,
                   budget=step_budget, **common, **extra)
        placed, info = loop.run(placed)
        gpt_calls += loop.gpt_calls
        search_left = max(search_left - loop.search_s, 0.)
        records.append({"ids": ids, "draft_error": draft["error"], "draft_missing": draft["missing"],
                        "draft_dropped": draft["dropped"], "before": before.summary(), "after": info["final"],
                        "attempts": info["attempts"], "termination": info["termination"],
                        "gpt_calls": len(loop.gpt_calls), "search_s": round(loop.search_s, 3),
                        "support_changes": info.get("support_changes", {}),
                        "seconds": {"draft": round(drafted - t0, 3), "repair": round(time.perf_counter() - drafted, 3)}})
    stepped = time.perf_counter()
    snapshot, parents = dict(placed), dict(plan.parents)
    step_state = critic.assess(snapshot).summary()
    results = {}
    for whole in final_repair:
        t0 = time.perf_counter()
        placed = dict(snapshot)
        plan.parents.clear()
        plan.parents.update(parents)  # the geometry and the critic read this dict: undo another variant's re-picks
        final_loop, final_calls = None, []
        if whole and mode != "A":  # A's own loop already was the whole-room repair
            loop = Loop(geometry, plan, fixed_sizes, catalog, critic, ask=ask, budget=budget, **common)
            placed, final_loop = loop.run(placed)
            final_calls = loop.gpt_calls
        repaired = time.perf_counter()
        polish_info = {"result": "skipped"}
        if polish:
            placed, polish_info = refine(geometry, plan, placed, critic=critic)
        final = critic.assess(placed)
        layout = {"schema_version": condition["schema_version"],
                  "objects": [placed[obj["id"]] for obj in condition["objects"]]}
        validate_layout(layout, condition)
        end = time.perf_counter()
        results[variant_name(mode, whole)] = {
            "status": "clean" if not final.hard else "best_effort", "layout": layout,
            "diagnostics": {"mode": mode, "variant": variant_name(mode, whole), "batch": batch if mode == "B" else None,
                            "final_repair": bool(whole and mode != "A"), "notes": notes, "plan_source": plan.source,
                            "size_source": size_source, "sizes": fixed_sizes, "steps": records,
                            "step_count": len(steps), "draft_calls": draft_calls,
                            "gpt_repair_calls": gpt_calls + final_calls, "step_gpt_calls": len(gpt_calls),
                            "final_gpt_calls": len(final_calls), "after_steps": step_state, "final_loop": final_loop,
                            "polish": polish_info, "final": final.summary(),
                            "unresolved": [v.as_dict() for v in final.hard],
                            "status_detail": "hard_valid" if not final.hard else "violations_remaining",
                            "support_parents": dict(plan.parents)},
            # seconds of this variant: the shared steps plus its own final repair and polish
            "timing_s": {"steps": round(stepped - start, 3), "final_repair": round(repaired - t0, 3),
                         "polish": round(end - repaired, 3), "total": round(stepped - start + end - t0, 3)}}
    return results
