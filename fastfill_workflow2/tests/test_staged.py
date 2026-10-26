from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog
from workflow.contract import check_request, fill_room_defaults
from workflow.critic import Assessment, Critic, Violation
from workflow.geometry import Geometry
from workflow.loop import LoopBudget
from workflow.orders import sub_room
from workflow.pipeline import run_draft_workflow
from workflow.planning import make_plan
from workflow.semantic import FIX_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS, REVIEW_INSTRUCTIONS, SemanticPass, admissible
from workflow.solver import _pose
from workflow.staged import run_staged, split_stages

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"
SIZES = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
         "lamp": [.25, .25, .45]}


def _room():
    condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
    return check_request(fill_room_defaults(condition)[0])


def _layout(chair_yaw: float) -> dict:
    placed = {"bed": _pose("bed", SIZES["bed"], 2.5, 2.95, 0., -math.pi / 2, "floor"),
              "nightstand": _pose("nightstand", SIZES["nightstand"], 1.15, 3.7, 0., -math.pi / 2, "floor"),
              "desk": _pose("desk", SIZES["desk"], 4.6, 1.5, 0., math.pi, "floor"),
              "chair": _pose("chair", SIZES["chair"], 3.95, 1.5, 0., chair_yaw, "floor")}
    placed["lamp"] = _pose("lamp", SIZES["lamp"], 4.6, 1.5, .75, math.pi, "desk")
    return placed


class FakeLLM:
    """Answers the three semantic requests: a review naming the chair, a fix, and pairwise votes."""

    def __init__(self, severity="severe", fix=None, vote="faithful"):
        self.severity, self.fix, self.vote = severity, fix or {"x": 3.95, "y": 1.5, "yaw_deg": 0}, vote
        self.seen: list[str] = []
        self.payloads: list[tuple[str, dict]] = []

    def __call__(self, instructions: str, prompt: str):
        payload = json.loads(prompt)
        self.payloads.append((instructions, payload))
        if instructions == REVIEW_INSTRUCTIONS:
            self.seen.append("review")
            chair = next(o for o in payload["objects"] if o["id"] == "chair")
            issues = [] if abs(chair["yaw_deg"]) < 5 else \
                [{"ids": ["chair"], "severity": self.severity, "problem": "the chair faces away from the desk",
                  "fix": "turn it to the desk"}]
            return json.dumps({"issues": issues}), {"usage": {}}
        if instructions == FIX_INSTRUCTIONS:
            self.seen.append("fix")
            return json.dumps({"objects": {"chair": self.fix}}), {"usage": {}}
        assert instructions == PAIRWISE_INSTRUCTIONS
        self.seen.append("pairwise")
        if self.vote == "always_2":
            return json.dumps({"better": "2"}), {"usage": {}}
        yaw = {k: next(o["yaw_deg"] for o in payload[k] if o["id"] == "chair") for k in ("layout_1", "layout_2")}
        return json.dumps({"better": "1" if abs(yaw["layout_1"]) < abs(yaw["layout_2"]) else "2"}), {"usage": {}}


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.condition = _room()
        self.plan = make_plan(self.condition, llm=False, semantic_only=True)
        self.g = Geometry(self.condition, self.plan.parents, self.plan.relations, hard_doors=True, hard_windows=False,
                          door_rule="evaluator")
        self.critic = Critic.for_plan(self.g, self.plan, templates=[])

    def _pass(self, llm):
        return SemanticPass(self.g, self.critic, self.plan, SIZES, SizeCatalog(CATALOG), locked_parents={}, ask=llm,
                            stage="floor")

    def test_admissible(self):
        clean = Assessment([], [Violation("reach", ("bed",), .2)])
        self.assertFalse(admissible(Assessment([Violation("collision", ("a", "b"), .01)], []), clean)[0])
        self.assertFalse(admissible(Assessment([], [Violation("reach", ("bed",), .5)]), clean)[0])
        self.assertFalse(admissible(Assessment([], [Violation("reach", ("bed",), .2), Violation("template_near", ("a", "b"), .2)]),
                                    clean, soft_tolerance=.1)[0])
        self.assertTrue(admissible(Assessment([], [Violation("reach", ("bed",), .2),
                                                   Violation("template_near", ("a", "b"), .08)]), clean)[0])

    def test_fix_kept_when_preferred_in_both_orders(self):
        llm = FakeLLM()
        placed, info = self._pass(llm).run(_layout(math.pi), {"bed", "nightstand", "desk", "chair"},
                                           {"bed", "nightstand", "desk", "chair"})
        self.assertAlmostEqual(placed["chair"]["yaw_rad"], 0., places=3)
        self.assertEqual(info["accepted"], 1)
        self.assertEqual(llm.seen[:4], ["review", "fix", "pairwise", "pairwise"])
        self.assertEqual(llm.seen[-1], "review")  # the second round re-reviews and finds nothing

    def test_split_vote_keeps_the_layout(self):
        start = _layout(math.pi)
        placed, info = self._pass(FakeLLM(vote="always_2")).run(start, {"chair"}, {"chair"})
        self.assertEqual(placed, start)
        self.assertEqual(info["accepted"], 0)

    def test_minor_issue_is_not_fixed(self):
        llm = FakeLLM(severity="minor")
        start = _layout(math.pi)
        placed, _ = self._pass(llm).run(start, {"chair"}, {"chair"})
        self.assertEqual(placed, start)
        self.assertEqual(llm.seen, ["review"])

    def test_illegal_proposal_is_never_committed(self):
        llm = FakeLLM(fix={"x": 2.5, "y": 2.95, "yaw_deg": 0})  # into the bed
        placed, _ = self._pass(llm).run(_layout(math.pi), {"chair"}, {"chair"})
        self.assertEqual(self.critic.assess(placed).hard, [])

    def test_frozen_objects_do_not_move(self):
        llm = FakeLLM()
        start = _layout(math.pi)
        placed, info = self._pass(llm).run(start, {"chair"}, {"lamp"})  # the chair is not movable here
        self.assertEqual(placed, start)
        self.assertEqual(info["rounds"][0]["fixes"][0]["reason"], "no movable object in the issue")


    def test_requests_carry_the_requirements(self):
        """Stage one's review sees descriptions, the user's constraints, the plan as suggestions and the pending lamp;
        the pairwise judge gets the same context but not the reviewer's problem text."""
        start = _layout(math.pi)
        floor = {i: b for i, b in start.items() if i != "lamp"}
        g1, c1 = sub_room(self.condition, self.plan, floor, [])
        llm = FakeLLM()
        SemanticPass(g1, c1, self.plan, SIZES, SizeCatalog(CATALOG), locked_parents={}, ask=llm, stage="floor",
                     condition=self.condition).run(floor, set(floor), set(floor), rounds=1)
        kinds = [k for k, _ in llm.payloads]
        self.assertEqual(kinds, [REVIEW_INSTRUCTIONS, FIX_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS, PAIRWISE_INSTRUCTIONS])
        review, fix, pairwise = llm.payloads[0][1], llm.payloads[1][1], llm.payloads[2][1]
        objects = {o["id"]: o for o in review["objects"]}
        self.assertEqual(objects["bed"]["description"], "a queen bed with upholstered headboard")
        self.assertEqual(objects["lamp"]["status"], "pending")
        self.assertEqual(objects["chair"]["status"], "review")
        self.assertTrue(all(c["hard"] in (True, False) for c in review["requirements"]["constraints"]))
        self.assertIn("not user requirements", review["requirements"]["plan"]["note"])
        self.assertTrue(all({"kind", "finding"} <= set(e) for e in review["evidence"]))
        self.assertEqual([p["id"] for p in fix["pending"]], ["lamp"])
        self.assertTrue(all("description" in o for o in fix["objects"]))
        self.assertIn("requirements", pairwise)
        self.assertEqual([p["id"] for p in pairwise["pending"]], ["lamp"])
        self.assertNotIn("faces away", json.dumps(pairwise))  # blind: no problem text
        self.assertNotIn("problem", pairwise)


    def test_malformed_answers_reject_only_the_candidate(self):
        start = _layout(math.pi)

        class Odd(FakeLLM):
            def __init__(self, fix_answer, **kw):
                super().__init__(**kw)
                self.fix_answer = fix_answer

            def __call__(self, instructions, prompt):
                if instructions == FIX_INSTRUCTIONS:
                    self.seen.append("fix")
                    if isinstance(self.fix_answer, Exception):
                        raise self.fix_answer
                    return json.dumps(self.fix_answer), {"usage": {}}
                return super().__call__(instructions, prompt)

        # a list instead of the map: read as the same poses
        listed = Odd({"objects": [{"id": "chair", "x": 3.95, "y": 1.5, "yaw_deg": 0}]})
        placed, info = self._pass(listed).run(start, {"chair"}, {"chair"})
        self.assertAlmostEqual(placed["chair"]["yaw_rad"], 0., places=3)
        for answer in ({"objects": "chair"}, {"objects": [1, 2]}, {"objects": {"chair": [3.95, 1.5]}},
                       TypeError("unexpected failure")):
            placed, info = self._pass(Odd(answer)).run(start, {"chair"}, {"chair"})
            self.assertEqual(placed, start)
            self.assertFalse(info["rounds"][0]["fixes"][0]["accepted"])
        self.assertIn("error: TypeError", info["rounds"][0]["fixes"][0]["reason"])

    def test_malformed_review_is_read_safely(self):
        for issues in ({"chair": "bad"}, "none", [{"ids": "chair", "severity": "minor", "problem": "x"}]):
            llm = lambda ins, prompt, issues=issues: (json.dumps({"issues": issues}), {"usage": {}})  # noqa: E731
            placed, info = self._pass(llm).run(_layout(math.pi), {"chair"}, {"chair"})
            self.assertEqual(placed, _layout(math.pi))
        self.assertEqual(info["rounds"][0]["issues"][0]["ids"], ["chair"])


class PolishTests(unittest.TestCase):
    def test_polish_before_the_small_items_exist(self):
        """Stage one polishes the furniture while the lamp on the desk is not placed yet: the desk moves alone."""
        from workflow.refine import refine
        condition = _room()
        plan = make_plan(condition, llm=False, semantic_only=True)
        floor = {i: b for i, b in _layout(0.).items() if i != "lamp"}
        floor["desk"] = _pose("desk", SIZES["desk"], 4.6, 1.5, 0., math.pi, "floor")  # 0.1 m off the wall: snaps
        g1, c1 = sub_room(condition, plan, floor, [])
        placed, _ = refine(g1, plan, floor, critic=c1)
        self.assertNotIn("lamp", placed)
        self.assertGreater(placed["desk"]["bottom_center_m"][0], 4.65)  # moved to the wall


class BudgetTests(unittest.TestCase):
    """The repair is bounded by work (every issue tried, a number of attempts), not by time."""

    def setUp(self):
        self.condition = _room()
        self.plan = make_plan(self.condition, llm=False, semantic_only=True)
        self.g = Geometry(self.condition, self.plan.parents, self.plan.relations, hard_doors=True,
                          hard_windows=False, door_rule="evaluator")
        self.critic = Critic.for_plan(self.g, self.plan, templates=[])
        crowded = _layout(math.pi)
        crowded["chair"] = _pose("chair", SIZES["chair"], 4.3, 1.5, 0., math.pi, "floor")  # into the desk
        self.crowded = crowded

    def _loop(self, budget, movable=None):
        from workflow.loop import FocusedLoop, Loop
        kw = {"locked_parents": {}, "budget": budget}
        if movable is None:
            return Loop(self.g, self.plan, SIZES, SizeCatalog(CATALOG), self.critic, **kw)
        return FocusedLoop(self.g, self.plan, SIZES, SizeCatalog(CATALOG), self.critic, movable=movable, **kw)

    def test_default_attempts_scale_with_the_issues_and_no_time_limit(self):
        current = self.critic.assess(self.crowded)
        loop = self._loop(LoopBudget(gpt_repairs=0))
        self.assertEqual(loop.max_attempts(current), 24 + 3 * len(loop._targets(current, set())))
        self.assertIsNone(LoopBudget().search_seconds)

    def test_every_issue_is_tried_without_a_time_budget(self):
        placed, info = self._loop(LoopBudget(gpt_repairs=0)).run(self.crowded)
        self.assertEqual(info["termination"], "no issue left to try")
        self.assertIsNone(info["time_limit_s"])
        self.assertEqual(self.critic.assess(placed).hard, [])

    def test_an_explicit_time_limit_is_reported(self):
        _, info = self._loop(LoopBudget(gpt_repairs=0, search_seconds=0.)).run(self.crowded)
        self.assertEqual(info["termination"], "time limit")


class StagedTests(unittest.TestCase):
    def test_split_stages(self):
        plan = make_plan(_room(), llm=False, semantic_only=True)
        one, two = split_stages(plan, ["bed", "lamp", "desk", "chair", "nightstand"])
        self.assertEqual(two, ["lamp"])
        self.assertEqual(one, ["bed", "desk", "chair", "nightstand"])

    def test_offline_staged_run(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        plan = make_plan(_room(), llm=False, semantic_only=True)
        result = run_staged(condition, None, plan, catalog_path=str(CATALOG), llm=False,
                            loop_budget=LoopBudget(gpt_repairs=0, search_seconds=30))
        d = result["diagnostics"]
        self.assertEqual({b["id"] for b in result["layout"]["objects"]}, {o["id"] for o in condition["objects"]})
        self.assertEqual(d["stage_ids"]["small"], ["lamp"])
        stages = [s["stage"] for s in d["steps"]]
        self.assertEqual(stages, sorted(stages, key=lambda s: s != "floor"))  # every floor group before any small one
        self.assertEqual(d["stage_one"]["semantic"], {"result": "skipped"})
        self.assertEqual(result["status"] == "clean", d["final"]["hard_count"] == 0)
        self.assertLessEqual(d["final"]["hard_count"], d["after_steps"]["hard_count"])

    def test_failed_review_keeps_the_layout(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        plan = make_plan(_room(), llm=False, semantic_only=True)

        def broken(instructions, prompt):
            raise KeyError("boom")  # not a request failure: escapes SemanticPass's own handling
        budget = LoopBudget(gpt_repairs=0, search_seconds=30)
        plain = run_staged(condition, None, plan, catalog_path=str(CATALOG), llm=False, loop_budget=budget)
        result = run_staged(condition, None, plan, catalog_path=str(CATALOG), llm=False, loop_budget=budget,
                            semantic_ask=broken)
        d = result["diagnostics"]
        self.assertEqual(d["stage_one"]["semantic"]["result"], "error")
        self.assertIsNotNone(result["layout"])
        self.assertEqual(d["final"]["hard_count"], plain["diagnostics"]["final"]["hard_count"])

    def test_unexpected_failure_keeps_the_committed_layout(self):
        """A failure in stage two's placement: the big furniture placed and checked so far is kept, the rest is put
        at its fallback, and the result says so instead of being dropped."""
        import workflow.staged as staged
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        original = staged.assemble
        def failing(geometry, plan_, sizes, poses, *, ids=None, placed=None):
            if ids and "lamp" in ids and placed and len(placed) == 4 and not getattr(failing, "done", False):
                failing.done = True
                raise RuntimeError("boom")
            return original(geometry, plan_, sizes, poses, ids=ids, placed=placed)
        staged.assemble = failing
        try:
            result = run_draft_workflow(condition, None, catalog_path=str(CATALOG), llm=False, placement="staged",
                                        loop_budget=LoopBudget(gpt_repairs=0, search_seconds=20))
        finally:
            staged.assemble = original
        d = result["diagnostics"]
        self.assertEqual(result["status"], "best_effort")
        self.assertEqual({b["id"] for b in result["layout"]["objects"]}, {o["id"] for o in condition["objects"]})
        self.assertEqual(d["failed_in"], "stage two")
        self.assertEqual(d["recovered"], {"committed_objects": 4, "fallback_objects": ["lamp"]})
        self.assertIn("RuntimeError: boom", d["error"])
        self.assertTrue(d["steps"] and all(s_["stage"] == "floor" for s_ in d["steps"]))  # stage one's record kept

    def test_pipeline_staged_with_reviews(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        llm = FakeLLM()
        result = run_draft_workflow(condition, None, catalog_path=str(CATALOG), llm=False, semantic_ask=llm, placement="staged",
                                    loop_budget=LoopBudget(gpt_repairs=0, search_seconds=30))
        d = result["diagnostics"]
        self.assertEqual(d["placement"], "staged")
        self.assertIn("review", llm.seen)
        self.assertEqual(d["stages"]["stage_one"]["semantic"]["stage"], "floor")
        self.assertEqual(d["stages"]["stage_two"]["semantic"]["stage"], "small")
        self.assertEqual(result["status"] == "clean", d["critic"]["final"]["hard_count"] == 0)


if __name__ == "__main__":
    unittest.main()
