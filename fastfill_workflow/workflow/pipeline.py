"""Orchestrate planning, size proposals, full-room search, and checked repair."""

from __future__ import annotations

import math
import time

from .catalog import SizeCatalog
from .critic import Critic, load_templates
from .contract import check_request, fill_room_defaults, validate_condition, validate_layout
from .drafting import assemble, make_draft
from .local_repair import DRAFT_WEIGHT, local_repair
from .loop import LoopBudget, critic_loop, make_ask
from .refine import refine
from .geometry import Geometry
from .planning import _required_parents, make_plan
from .solver import repair, solve


def run_draft_workflow(condition: dict, sizes: dict[str, list[float]] | None, *, catalog_path: str | None = None,
                       llm: bool = True, model: str | None = None, base_url: str | None = None,
                       api_key: str | None = None, repair: bool = True, polish: bool = True,
                       draft_weight: float = DRAFT_WEIGHT, beam_width: int = 24, candidate_limit: int = 48,
                       sizer=None, yaws: dict[str, float] | None = None, yaw_mode: str = "prompt",
                       critic: bool = True, gpt_repairs: int = 2, loop_budget: LoopBudget | None = None,
                       templates_path: str | None = None) -> dict:
    """Draft workflow, steps ①–④: ① LLM semantic plan, ② one size per object (``sizes``, or when it is None
    ``sizer(condition)``, a ``FastFillSizer`` run on the request as given; an id without one gets the train
    catalog's, and a room the sizer cannot take is recorded in ``size_error``), ③ LLM draft at those sizes, the hard checks, and ④ local
    repair of the objects with issues (``local_repair``; ``repair=False`` returns the checked draft), and ⑤ a polish
    that snaps to walls and axes and tightens soft relations without new hard issues (``polish=False`` skips it). A layout is
    always returned; status is clean, best_effort (issues left after repair) or needs_repair (repair off).

    ``yaw_mode`` (with FastFill yaws: ``yaws``, or the sizer's): "prompt" (default) gives them to step ③ as
    ``suggested_yaw_deg``, so the LLM picks each position for that facing and may depart from it where it cannot
    work; "llm" gives no suggestion; "fastfill" swaps FastFill's yaws into the finished draft; "hybrid" swaps them
    in and lets repair turn an object back to the LLM's yaw (``solver.ALT_YAW_PENALTY_TURNS``). The swap modes keep
    the LLM's positions for another facing, which pushes objects through walls (49 of 50 test rooms got more issues),
    so they are kept for comparison only. Without FastFill yaws every mode is "llm".

    ``critic=True`` (default; critic_flowchart_v2 ⑤–⑦) replaces step ④ with the critic loop (``loop.critic_loop``):
    door clearance becomes a hard check under the evaluator's rule (windows stay soft), every candidate is judged
    by ``critic.Critic`` on the whole room (hard violations with magnitudes first, then the soft costs: relations,
    relation templates, access), sizes never shrink, and up to ``gpt_repairs`` GPT repair requests are made when the
    program search fails (none without ``llm``). Polish moves go through the same rule, and the final assessment is
    recorded. ``critic=False`` is the v4 workflow (soft doors, local repair, shrinking allowed)."""
    if yaw_mode not in ("prompt", "llm", "fastfill", "hybrid"):
        raise ValueError(f"unknown yaw_mode {yaw_mode!r}")
    start = time.perf_counter()
    validate_condition(condition)
    request = condition
    condition, notes = fill_room_defaults(condition)
    condition = check_request(condition)
    plan = make_plan(condition, llm=llm, model=model, base_url=base_url, api_key=api_key, semantic_only=True)
    planned = time.perf_counter()
    size_error = None
    if sizes is None and sizer is not None:
        try:
            sizes, yaws = sizer.predict(request)
        except ValueError as exc:  # over the model's object or context budget
            size_error = f"{type(exc).__name__}: {exc}"[:300]
    sized = time.perf_counter()
    catalog, fixed_sizes, size_source = SizeCatalog(catalog_path), {}, {"given": 0, "catalog": 0}
    for obj in condition["objects"]:
        size = (sizes or {}).get(obj["id"])
        if size is not None and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size):
            fixed_sizes[obj["id"]] = [float(v) for v in size]
            size_source["given"] += 1
        else:
            fixed_sizes[obj["id"]] = catalog.candidates(obj)[0][0]
            size_source["catalog"] += 1
    geometry = Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False,
                        door_rule="evaluator") if critic else \
        Geometry(condition, plan.parents, plan.relations, hard_openings=False)
    if llm:
        placed, draft = make_draft(condition, plan, fixed_sizes, geometry, model=model, base_url=base_url,
                                   api_key=api_key, yaws=yaws if yaw_mode == "prompt" else None)
    else:  # offline tests: no poses, every object at its fallback
        placed, missing = assemble(geometry, plan, fixed_sizes, {})
        draft = {"calls": [], "dropped": {}, "missing": missing, "error": "llm disabled"}
    llm_yaws = {ident: box["yaw_rad"] for ident, box in placed.items()}
    swapped = 0
    if yaws and yaw_mode in ("fastfill", "hybrid"):
        for ident, box in placed.items():
            if ident in yaws and ident not in draft["missing"]:
                placed[ident] = {**box, "yaw_rad": math.remainder(float(yaws[ident]), 2 * math.pi)}
                swapped += 1
    drafted = time.perf_counter()
    draft_issues = geometry.validate(placed)
    draft_layout = {"schema_version": condition["schema_version"],
                    "objects": [placed[obj["id"]] for obj in condition["objects"]]}
    draft_bad = sorted({ident for issue in draft_issues for ident in issue.ids if ident in geometry.requests})
    repair_info = {"result": "skipped"}
    judge, loop_info = None, None
    if critic:
        judge = Critic.for_plan(geometry, plan, load_templates(templates_path))
        draft_summary = judge.assess(placed).summary()  # before the loop can re-pick a support
        if repair:
            placed, loop_info = critic_loop(geometry, plan, fixed_sizes, placed, catalog, judge,
                                            locked_parents=_required_parents(condition),
                                            ask=make_ask(model, base_url, api_key) if llm and gpt_repairs > 0 else None,
                                            budget=loop_budget or LoopBudget(gpt_repairs=gpt_repairs),
                                            beam_width=beam_width, candidate_limit=candidate_limit,
                                            draft_weight=draft_weight)
            repair_info = {"result": "critic_loop"}
    elif repair and draft_issues:
        plan.draft_poses = {ident: {"x": box["bottom_center_m"][0], "y": box["bottom_center_m"][1],
                                    "yaw_rad": box["yaw_rad"],
                                    "alt_yaw_rad": llm_yaws[ident] if yaw_mode == "hybrid" and swapped else None}
                            for ident, box in placed.items() if ident not in draft["missing"]}
        placed, repair_info = local_repair(geometry, plan, fixed_sizes, placed, catalog, draft_weight=draft_weight,
                                           beam_width=beam_width, candidate_limit=candidate_limit)
    repaired = time.perf_counter()
    polish_info = {"result": "skipped"}
    if polish:
        placed, polish_info = refine(geometry, plan, placed, critic=judge)
    polished = time.perf_counter()
    issues = geometry.validate(placed)
    layout = {"schema_version": condition["schema_version"],
              "objects": [placed[obj["id"]] for obj in condition["objects"]]}
    validate_layout(layout, condition)
    critic_info = None
    if judge is not None:  # ⑦ final acceptance: the same checks on the result of every step
        final = judge.assess(placed)
        critic_info = {"draft": draft_summary,
                       "loop": loop_info, "final": final.summary(),
                       "unresolved": [v.as_dict() for v in final.hard],
                       "status_detail": "hard_valid" if not final.hard else "violations_remaining"}
    end = time.perf_counter()
    status = "clean" if not issues else ("best_effort" if repair else "needs_repair")
    return {"status": status, "layout": layout,
            "diagnostics": {"notes": notes, "plan_source": plan.source, "plan_llm": plan.llm,
                            "support_parents": plan.parents, "functional_groups": plan.groups,
                            "priority": plan.priority, "size_source": size_source, "size_error": size_error,
                            "sizes": fixed_sizes, "draft": draft,
                            "draft_hard_issues": [issue.as_dict() for issue in draft_issues],
                            "draft_objects_with_issues": draft_bad, "draft_layout": draft_layout,
                            "yaw": {"mode": yaw_mode if (swapped or (yaw_mode == "prompt" and yaws)) else "llm",
                                    "fastfill_yaws_used": swapped, "llm_yaws": llm_yaws,
                                    "suggested": len(yaws or {}) if yaw_mode == "prompt" else 0,
                                    "followed": sum(1 for k, v in llm_yaws.items() if yaw_mode == "prompt" and yaws
                                                    and k in yaws and abs(math.remainder(v - yaws[k], 2 * math.pi))
                                                    <= math.radians(10))},
                            "repair": repair_info, "polish": polish_info, "critic": critic_info,
                            "plan_relations": plan.relations,
                            "hard_issues": [issue.as_dict() for issue in issues]},
            "timing_s": {"plan": round(planned - start, 3), "size": round(sized - planned, 3),
                         "draft": round(drafted - sized, 3),
                         "repair": round(repaired - drafted, 3), "polish": round(polished - repaired, 3),
                         "check": round(end - polished, 3),
                         "total": round(end - start, 3)}}


def run_workflow(condition: dict, *, catalog_path: str | None = None,
                 llm: bool = True, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 draft_layout: dict | None = None,
                 beam_width: int = 32, candidate_limit: int = 48,
                 repair_rounds: int = 2, allow_thin_overlap: bool = False) -> dict:
    start = time.perf_counter()
    condition = check_request(condition)
    if beam_width < 1 or candidate_limit < 1 or repair_rounds < 0:
        raise ValueError("beam_width and candidate_limit must be positive; repair_rounds nonnegative")
    catalog = SizeCatalog(catalog_path)
    plan = make_plan(condition, llm=llm, model=model, base_url=base_url, api_key=api_key)
    if draft_layout is not None:
        validate_layout(draft_layout, condition)
        for box in draft_layout["objects"]:
            ident = box["id"]
            plan.size_hints[ident] = box["target_size_local_m"]
            plan.draft_poses[ident] = {"x": box["bottom_center_m"][0],
                                      "y": box["bottom_center_m"][1],
                                      "yaw_rad": box["yaw_rad"]}
        plan.source += "+external_draft"
    planned_s = time.perf_counter() - start
    geometry = Geometry(condition, plan.parents, plan.relations,
                        allow_thin_overlap=allow_thin_overlap)
    state, search = solve(geometry, catalog, plan, beam_width=beam_width,
                          candidate_limit=candidate_limit)
    solved_s = time.perf_counter() - start
    if state is None:
        return {"status": "infeasible_or_search_exhausted", "layout": None,
                "diagnostics": {"search": search, "plan_source": plan.source, "llm": plan.llm,
                                "reason": "No complete hard-valid assignment was found within the search budget."},
                "timing_s": {"plan": round(planned_s, 3), "search": round(solved_s - planned_s, 3),
                             "total": round(solved_s, 3)}}
    state, repair_history = repair(geometry, catalog, plan, state, rounds=repair_rounds)
    layout = {"schema_version": condition["schema_version"],
              "objects": [state.placed[obj["id"]] for obj in condition["objects"]]}
    validate_layout(layout, condition)
    issues = geometry.validate(state.placed)
    if issues:
        raise RuntimeError(f"internal error: solver returned invalid layout: {issues}")
    soft_score, soft_issues = geometry.soft_score(state.placed)
    elapsed = time.perf_counter() - start
    return {"status": "ok", "layout": layout,
            "diagnostics": {
                "plan_source": plan.source,
                "llm": plan.llm,
                "support_parents": plan.parents,
                "functional_groups": plan.groups,
                "priority": plan.priority,
                "hard_issues": [], "soft_issues": [issue.as_dict() for issue in soft_issues],
                "soft_score": soft_score,
                "bbox_level_only": True,
                "allow_thin_overlap": allow_thin_overlap,
                "search": search, "repair": repair_history,
            },
            "timing_s": {"plan": round(planned_s, 3), "search": round(solved_s - planned_s, 3),
                         "repair_validate": round(elapsed - solved_s, 3), "total": round(elapsed, 3)}}
