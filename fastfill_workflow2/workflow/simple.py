"""Workflow 2 (``placement="simple"``, the default here): one placement, one program check, one LLM check.

  ① plan     one LLM call for supports, groups and relations (``planning.make_plan``, semantic only); ``plan_llm=False``
             uses the rules
  ② sizes    one per object: FastFill's when given, else the catalog's
  ③ place    ONE LLM call places every object of the room (``drafting.make_draft``; FastFill yaws as suggestions)
  ④ check    the program checks once (``critic.Critic``). If there are hard violations (collision, out of bounds, no
             support, blocked door, broken hard constraint), one repair pass takes them up, each once, by the program
             search at fixed sizes; soft costs are not chased and there is no polish, so wherever the placement is legal
             it is left as the LLM made it
  ⑤ review   the LLM reviews the whole room once (``semantic.SemanticPass``, one round): only severe issues are fixed;
             each fix is repaired by the program and kept only if the program gate passes (no new hard violation;
             reach, access and doors no worse) and a blind pairwise comparison prefers it in both orders
  ⑥ result   the layout, plus the layout after each of ③ ④ ⑤ (``diagnostics.stages``), so a bad result can be traced

Compared with ``placement="staged"`` (workflow 1): no groups, no stages, no freezing, no soft-cost search, no polish,
one review round. LLM calls: 1 plan + 1 placement + 1 review + 3 per severe issue fixed (proposal, two pairwise votes).
"""

from __future__ import annotations

import math
import time

from .catalog import SizeCatalog
from .contract import check_request, fill_room_defaults, validate_condition, validate_layout
from .credentials import FatalAPIError
from .critic import Critic, load_templates
from .drafting import assemble, make_draft
from .loop import Loop, LoopBudget, make_ask
from .orders import critic_geometry
from .planning import Plan, _required_parents, make_plan
from .semantic import SOFT_TOLERANCE, SemanticPass


def _layout(condition: dict, placed: dict[str, dict]) -> dict:
    return {"schema_version": condition["schema_version"], "objects": [placed[o["id"]] for o in condition["objects"]]}


def run_simple(condition: dict, sizes: dict[str, list[float]] | None, *, plan: Plan | None = None,
               yaws: dict[str, float] | None = None, catalog_path: str | None = None, llm: bool = True,
               model: str | None = None, base_url: str | None = None, api_key: str | None = None,
               plan_llm: bool = True, repair: bool = True, semantic: bool = True, semantic_fixes: int = 3,
               soft_tolerance: float = SOFT_TOLERANCE, templates_path: str | None = None, beam_width: int = 24,
               candidate_limit: int = 48, semantic_ask=None) -> dict:
    """{status, layout, diagnostics, timing_s}. ``plan``: one made already (the pipeline's), else made here.
    ``semantic_ask`` replaces the LLM client for the review (tests); the review runs only with ``llm`` or it."""
    start = time.perf_counter()
    validate_condition(condition)
    condition, notes = fill_room_defaults(condition)
    condition = check_request(condition)
    if plan is None:
        plan = make_plan(condition, llm=llm and plan_llm, model=model, base_url=base_url, api_key=api_key,
                         semantic_only=True)
    planned = time.perf_counter()
    catalog, fixed_sizes, size_source = SizeCatalog(catalog_path), {}, {"given": 0, "catalog": 0}
    for obj in condition["objects"]:
        size = (sizes or {}).get(obj["id"])
        if size is not None and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size):
            fixed_sizes[obj["id"]], size_source["given"] = [float(v) for v in size], size_source["given"] + 1
        else:
            fixed_sizes[obj["id"]], size_source["catalog"] = catalog.candidates(obj)[0][0], size_source["catalog"] + 1
    geometry = critic_geometry(condition, plan)
    critic = Critic.for_plan(geometry, plan, load_templates(templates_path))
    locked = _required_parents(condition)

    # ③ one placement call
    if llm:
        placed, draft = make_draft(condition, plan, fixed_sizes, geometry, yaws=yaws, model=model, base_url=base_url,
                                   api_key=api_key)
    else:  # offline: every object at its fallback (on its support, else the room's representative point)
        placed, missing = assemble(geometry, plan, fixed_sizes, {})
        draft = {"calls": [], "dropped": {}, "missing": missing, "error": "llm disabled"}
    placed_s = time.perf_counter()
    stages = {"draft": _layout(condition, placed)}
    assessed = {"draft": critic.assess(placed)}

    # ④ one program check (and one repair pass over the hard violations only)
    check = {"result": "clean" if not assessed["draft"].hard else "not repaired"}
    if repair and assessed["draft"].hard:
        loop = Loop(geometry, plan, fixed_sizes, catalog, critic, locked_parents=locked,
                    budget=LoopBudget(gpt_repairs=0, min_soft_cost=math.inf), beam_width=beam_width,
                    candidate_limit=candidate_limit)
        placed, check = loop.run(placed)
        check["result"] = "repaired"
    stages["after_check"] = _layout(condition, placed)
    assessed["after_check"] = critic.assess(placed)
    checked_s = time.perf_counter()

    # ⑤ one LLM review of the whole room
    review = {"result": "skipped"}
    review_ask = semantic_ask or (make_ask(model, base_url, api_key) if llm else None)
    if semantic and review_ask is not None:
        reviewer = SemanticPass(geometry, critic, plan, fixed_sizes, catalog, locked_parents=locked, ask=review_ask,
                                stage="room", soft_tolerance=soft_tolerance, condition=condition,
                                beam_width=beam_width, candidate_limit=candidate_limit)
        parents = dict(plan.parents)
        everything = set(placed)
        try:
            placed, review = reviewer.run(placed, everything, everything, rounds=1, max_fixes=semantic_fixes)
        except FatalAPIError:
            raise
        except Exception as exc:  # the review is optional: keep the checked layout
            plan.parents.clear()
            plan.parents.update(parents)
            review = {"result": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
        review["llm_calls"] = reviewer.calls
    stages["final"] = _layout(condition, placed)
    final = assessed["final"] = critic.assess(placed)

    layout = _layout(condition, placed)
    validate_layout(layout, condition)
    end = time.perf_counter()
    return {"status": "clean" if not final.hard else "best_effort", "layout": layout,
            "diagnostics": {"placement": "simple", "notes": notes, "plan_source": plan.source, "plan_llm": plan.llm,
                            "support_parents": dict(plan.parents), "functional_groups": plan.groups,
                            "plan_relations": plan.relations, "size_source": size_source, "sizes": fixed_sizes,
                            "draft": draft, "check": check, "semantic": review, "stages": stages,
                            "assessments": {k: v.summary() for k, v in assessed.items()},
                            "critic": {"loop": {"gpt_calls": []}, "final": final.summary(),
                                       "unresolved": [v.as_dict() for v in final.hard],
                                       "status_detail": "hard_valid" if not final.hard else "violations_remaining"},
                            "hard_issues": [v.as_dict() for v in final.hard]},
            "timing_s": {"plan": round(planned - start, 3), "place": round(placed_s - planned, 3),
                         "check": round(checked_s - placed_s, 3), "review": round(end - checked_s, 3),
                         "total": round(end - start, 3)}}
