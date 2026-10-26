"""The staged workflow (``placement="staged"``): big furniture first, then small items, each stage checked by the
program for legality and by the LLM for sense.

  stage one   every floor-standing object, one functional group at a time: the LLM places the group beside what is
              placed (``make_partial_draft``), the program checks and repairs only that group (``FocusedLoop``,
              GPT repairs on demand), and the group is frozen. Then the program checks the whole stage (reach and
              access need every big object), polish snaps to walls and axes, and the LLM reviews the stage once
              (``semantic.SemanticPass``). The big furniture is then frozen.
  stage two   the objects on furniture, on fixed objects and on walls, grouped with their support's group: the
              LLM places them on the support step ① chose, the program checks and repairs only them (the search may
              rearrange them on that support; a GPT repair may also re-pick it), sizes never shrink. Then the program
              checks every small item on the whole room, and the LLM reviews the small items with the room as
              context.
  escape      at most ``escape_moves`` moves of big furniture in stage two: when hard violations of small items are
              left after the program check (e.g. a support too small), the program may move the big objects
              involved; when the review names a big object, its fix may move it. Either is kept only if stage
              one's program checks still pass on the big furniture (``semantic.admissible`` against stage one's
              final state).
  safeguard   the final program check, no LLM: the result must have no more hard violations than the layout after
              stage two's program check, or that layout is returned instead (every committed change is already
              checked, so this only catches gaps).

Every LLM change is a proposal: the program repairs it and accepts or rejects it, so the committed layout is always
one that passed the program checks.
"""

from __future__ import annotations

import dataclasses
import time
import traceback

from .catalog import SizeCatalog
from .contract import check_request, fill_room_defaults, validate_condition, validate_layout
from .credentials import FatalAPIError
from .critic import Critic, load_templates
from .drafting import assemble, make_partial_draft
from .local_repair import DRAFT_WEIGHT, FixedSizes
from .loop import FocusedLoop, LoopBudget, make_ask
from .orders import critic_geometry, functional_steps, plan_from_dict, plan_to_dict, sub_room
from .planning import Plan, _required_parents
from .refine import refine
from .semantic import SOFT_TOLERANCE, SemanticPass, admissible
from .solver import _order


def split_stages(plan: Plan, order: list[str]) -> tuple[list[str], list[str]]:
    """Stage one: objects on the floor; stage two: the rest (on objects, fixed objects, walls), in solver order."""
    one = [i for i in order if plan.parents.get(i) == "floor"]
    return one, [i for i in order if i not in set(one)]


def run_staged(condition: dict, sizes: dict[str, list[float]] | None, plan: Plan, *,
               yaws: dict[str, float] | None = None, catalog_path: str | None = None, llm: bool = True,
               model: str | None = None, base_url: str | None = None, api_key: str | None = None,
               gpt_repairs: int = 2, loop_budget: LoopBudget | None = None, polish: bool = True,
               templates_path: str | None = None, beam_width: int = 24, candidate_limit: int = 48,
               semantic: bool = True, semantic_rounds: int = 2, semantic_fixes: int = 3,
               soft_tolerance: float = SOFT_TOLERANCE, escape_moves: int = 1, semantic_ask=None) -> dict:
    """{status, layout, diagnostics, timing_s} like ``orders.run_order``. ``semantic_ask`` replaces the LLM client
    for the reviews (tests); the semantic review runs only with ``llm`` or ``semantic_ask``. ``plan`` is copied."""
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
    review_ask = semantic_ask or (make_ask(model, base_url, api_key) if llm else None)
    review_ask = review_ask if semantic else None
    common = {"beam_width": beam_width, "candidate_limit": candidate_limit}

    order = _order(geometry, FixedSizes(fixed_sizes, catalog), plan, set(geometry.requests))
    one, two = split_stages(plan, order)
    groups_one = functional_steps(geometry, plan, one)  # stage two's groups are made after stage one
    # every loop runs until its issues are tried (``loop.LoopBudget``, no time limit), so a big room gets the repair
    # it needs; ``budget.search_seconds``, when given (tests), is each loop's own time limit
    state = {"placed": {}, "draft_calls": [], "gpt_calls": [], "search_s": 0., "time_limited": [],
             "steps": [], "reviews": {}, "live_loop": None, "phase": "stage one"}

    def run_loop(step_geometry, step_critic, placed, movable, *, allow_gpt=True, label=""):
        left = max(budget.gpt_repairs - len(state["gpt_calls"]), 0) if allow_gpt else 0
        step_budget = dataclasses.replace(budget, gpt_repairs=left)
        loop = FocusedLoop(step_geometry, plan, fixed_sizes, catalog, step_critic, locked_parents=locked,
                           ask=ask if left else None, budget=step_budget, movable=set(movable),
                           draft_weight=DRAFT_WEIGHT, **common)
        state["live_loop"] = loop  # its GPT calls are counted even if it fails (``_recover``)
        placed, info = loop.run(placed)
        state["live_loop"] = None
        state["gpt_calls"] += loop.gpt_calls
        state["search_s"] += loop.search_s
        if info["termination"] == "time limit":
            state["time_limited"].append(label)
        return placed, info

    def place_groups(groups, stage):
        for ids in groups:
            t0 = time.perf_counter()
            if not llm:  # offline: every new object at its fallback (on its parent, else the room's representative point)
                placed, missing = assemble(geometry, plan, fixed_sizes, {}, ids=ids, placed=state["placed"])
                draft = {"calls": [], "dropped": {}, "missing": missing, "error": "llm disabled"}
            else:
                placed, draft = make_partial_draft(condition, plan, fixed_sizes, geometry, ids, state["placed"],
                                                   yaws=yaws, model=model, base_url=base_url, api_key=api_key)
            drafted = time.perf_counter()
            state["draft_calls"] += draft["calls"]
            step_geometry, step_critic = sub_room(condition, plan, placed, templates)
            before = step_critic.assess(placed)
            calls_before = len(state["gpt_calls"])
            placed, info = run_loop(step_geometry, step_critic, placed, ids, label=f"{stage} group {ids[0]}")
            state["placed"] = placed
            state["steps"].append({"stage": stage, "ids": ids, "draft_error": draft["error"],
                            "draft_missing": draft["missing"], "draft_dropped": draft["dropped"],
                            "before": before.summary(), "after": info["final"], "attempts": info["attempts"],
                            "termination": info["termination"], "gpt_calls": len(state["gpt_calls"]) - calls_before,
                            "search_s": info["search_seconds"], "support_changes": info.get("support_changes", {}),
                            "seconds": {"draft": round(drafted - t0, 3),
                                        "repair": round(time.perf_counter() - drafted, 3)}})

    def review(step_geometry, step_critic, stage, review_ids, movable, stage_locked, escape=None):
        if review_ask is None or not review_ids:
            return {"result": "skipped"}, 0
        reviewer = SemanticPass(step_geometry, step_critic, plan, fixed_sizes, catalog, locked_parents=stage_locked,
                                ask=review_ask, stage=stage, soft_tolerance=soft_tolerance, condition=condition,
                                **common)
        parents = dict(plan.parents)
        try:
            state["placed"], info = reviewer.run(state["placed"], set(review_ids), set(movable),
                                                 rounds=semantic_rounds, max_fixes=semantic_fixes, escape=escape)
        except FatalAPIError:
            raise
        except Exception as exc:  # the review is optional: keep the layout as the program checks left it
            plan.parents.clear()
            plan.parents.update(parents)
            info = {"stage": stage, "result": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        info["llm_calls"] = reviewer.calls
        state["reviews"][stage] = info
        return info, len(reviewer.calls)

    def body() -> dict:
        timing, semantic_calls = {}, 0
        # ------------------------------------------------------------------ stage one: big furniture
        place_groups(groups_one, "floor")
        big = set(state["placed"])
        g1, c1 = sub_room(condition, plan, state["placed"], templates)
        state["placed"], check_one = run_loop(g1, c1, state["placed"], big, label="stage one check")
        polish_info = {"result": "skipped"}
        if polish:
            state["placed"], polish_info = refine(g1, plan, state["placed"], critic=c1)
        timing["stage_one_program"] = time.perf_counter() - start
        review_one, n = review(g1, c1, "floor", big, big, {**locked, **{i: plan.parents[i] for i in big}})
        semantic_calls += n
        stage_one = c1.assess(state["placed"])
        timing["stage_one"] = time.perf_counter() - start

        def stage_one_still_ok(placed: dict[str, dict]) -> tuple[bool, str]:
            """Stage one's program checks on the big furniture, against stage one's final state."""
            boxes = {i: placed[i] for i in big}
            geometry_one, critic_one = sub_room(condition, plan, boxes, templates)
            ok, reason = admissible(critic_one.assess(boxes), stage_one, soft_tolerance)
            return ok, ("stage one checks still pass" if ok else f"stage one: {reason}")

        # ------------------------------------------------------------------ stage two: small items
        # again from the plan as stage one left it (a GPT repair may have re-picked a support)
        groups_two = functional_steps(geometry, plan, [i for i in order if i not in big])
        escape_info, check_two, review_two = {"result": "skipped"}, None, {"result": "skipped"}
        escape = {"ids": big, "left": escape_moves, "recheck": stage_one_still_ok}
        state["phase"] = "stage two"
        place_groups(groups_two, "small")
        small = set(state["placed"]) - big
        if small:
            state["placed"], check_two = run_loop(geometry, critic, state["placed"], small, label="stage two check")
            current = critic.assess(state["placed"])
            stuck = {i for v in current.hard if set(v.ids) & small for i in v.ids}
            involved = {i for i in stuck if i in big} | {plan.parents[i] for i in stuck & small if plan.parents[i] in big}
            if involved and escape["left"] > 0:
                escape["left"] -= 1
                snapshot, parents = dict(state["placed"]), dict(plan.parents)
                moved, loop_info = run_loop(geometry, critic, state["placed"], small | involved, allow_gpt=False,
                                            label="escape")
                after = critic.assess(moved)
                ok, reason = critic.better(after, current)
                if ok:
                    ok, reason = stage_one_still_ok(moved)
                if ok:
                    state["placed"] = moved
                else:
                    plan.parents.clear()
                    plan.parents.update(parents)
                    state["placed"] = snapshot
                escape_info = {"result": "accepted" if ok else "rejected", "reason": reason, "moved_big": sorted(involved),
                               "before": current.key(), "after": after.key(), "attempts": loop_info["attempts"]}
        timing["stage_two_program"] = time.perf_counter() - start
        checkpoint, checkpoint_parents = dict(state["placed"]), dict(plan.parents)
        after_steps = critic.assess(checkpoint)
        if small:
            review_two, n = review(geometry, critic, "small", small, small, {**locked, **{i: plan.parents[i] for i in big}},
                                   escape=escape)
            semantic_calls += n

        # ------------------------------------------------------------------ safeguard: program only
        state["phase"] = "safeguard"
        final = critic.assess(state["placed"])
        safeguard = "passed"
        if final.hard_count > after_steps.hard_count or (after_steps.hard_count == 0 and final.hard_count):
            state["placed"], final, safeguard = checkpoint, after_steps, "reverted to the layout after stage two's checks"
            plan.parents.clear()
            plan.parents.update(checkpoint_parents)
        placed = state["placed"]
        layout = {"schema_version": condition["schema_version"],
                  "objects": [placed[obj["id"]] for obj in condition["objects"]]}
        validate_layout(layout, condition)
        end = time.perf_counter()
        return {"status": "clean" if not final.hard else "best_effort", "layout": layout,
                "diagnostics": {"mode": "staged", "variant": "staged", "notes": notes, "plan_source": plan.source,
                                "size_source": size_source, "sizes": fixed_sizes, "steps": state["steps"],
                                "step_count": len(state["steps"]), "stage_ids": {"floor": sorted(big), "small": sorted(small)},
                                "draft_calls": state["draft_calls"], "gpt_repair_calls": state["gpt_calls"],
                                "stage_one": {"check": check_one, "polish": polish_info, "semantic": review_one,
                                              "final": stage_one.summary()},
                                "stage_two": {"check": check_two, "escape": escape_info, "semantic": review_two},
                                "semantic_calls": semantic_calls, "after_steps": after_steps.summary(),
                                "final_loop": check_two, "polish": polish_info, "safeguard": safeguard,
                                "final": final.summary(), "unresolved": [v.as_dict() for v in final.hard],
                                "status_detail": "hard_valid" if not final.hard else "violations_remaining",
                                "support_parents": dict(plan.parents), "search_s": round(state["search_s"], 1),
                                "time_limited": state["time_limited"]},
                "timing_s": {"stage_one_program": round(timing["stage_one_program"], 3),
                             "stage_one": round(timing["stage_one"], 3),
                             "stage_two_program": round(timing["stage_two_program"] - timing["stage_one"], 3),
                             "total": round(end - start, 3)}}

    try:
        return body()
    except FatalAPIError:
        raise
    except Exception as exc:  # an unexpected failure: keep what is committed rather than lose the room
        error = f"{type(exc).__name__}: {exc}"[:500]
        trace = traceback.format_exc()[-2000:]
    return _recover(condition, plan, geometry, critic, fixed_sizes, order, state, error, trace, notes=notes,
                    size_source=size_source, start=start)


def _recover(condition: dict, plan: Plan, geometry, critic: Critic, sizes: dict[str, list[float]], order: list[str],
             state: dict, error: str, trace: str, *, notes, size_source: dict, start: float) -> dict:
    """The result after an unexpected failure: the last committed layout (every change in it passed the program
    checks), the objects not placed yet at their fallback (on their support, else the room's representative point),
    status best_effort and ``status_detail`` "error_recovered", with the error and every LLM call made so far."""
    placed = dict(state["placed"])
    missing = [i for i in order if i not in placed]
    placed, _ = assemble(geometry, plan, sizes, {}, ids=missing, placed=placed)
    final = critic.assess(placed)
    layout = {"schema_version": condition["schema_version"],
              "objects": [placed[obj["id"]] for obj in condition["objects"]]}
    validate_layout(layout, condition)
    live = state["live_loop"]
    gpt_calls = state["gpt_calls"] + (live.gpt_calls if live is not None else [])
    review = lambda stage: state["reviews"].get(stage, {"result": "not run"})  # noqa: E731
    return {"status": "best_effort", "layout": layout,
            "diagnostics": {"mode": "staged", "variant": "staged", "notes": notes, "plan_source": plan.source,
                            "size_source": size_source, "sizes": sizes, "steps": state["steps"],
                            "step_count": len(state["steps"]),
                            "stage_ids": {"floor": sorted(i for i in placed if plan.parents.get(i) == "floor"),
                                          "small": sorted(i for i in placed if plan.parents.get(i) != "floor")},
                            "draft_calls": state["draft_calls"], "gpt_repair_calls": gpt_calls,
                            "stage_one": {"semantic": review("floor")}, "stage_two": {"semantic": review("small")},
                            "semantic_calls": sum(len(r.get("llm_calls", [])) for r in state["reviews"].values()),
                            "after_steps": final.summary(), "final_loop": None, "polish": {"result": "not run"},
                            "safeguard": "not run", "final": final.summary(),
                            "unresolved": [v.as_dict() for v in final.hard], "status_detail": "error_recovered",
                            "support_parents": dict(plan.parents), "search_s": round(state["search_s"], 1),
                            "time_limited": state["time_limited"], "error": error, "traceback": trace, "failed_in": state["phase"],
                            "recovered": {"committed_objects": len(state["placed"]), "fallback_objects": missing}},
            "timing_s": {"total": round(time.perf_counter() - start, 3)}}
