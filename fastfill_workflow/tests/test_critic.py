from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog
from workflow.contract import check_request, fill_room_defaults
from workflow.critic import Assessment, Critic, Violation
from workflow.geometry import Geometry
from workflow.local_repair import FixedSizes
from workflow.loop import Loop, LoopBudget
from workflow.pipeline import run_draft_workflow
from workflow.planning import _required_parents, make_plan
from workflow.solver import _pose

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"


def _room():
    condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
    condition, _ = fill_room_defaults(condition)
    return check_request(condition)


def _geometry(condition, plan):
    return Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False, door_rule="evaluator")


class CriticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.condition = _room()
        cls.plan = make_plan(cls.condition, llm=False, semantic_only=True)
        cls.g = _geometry(cls.condition, cls.plan)
        cls.critic = Critic.for_plan(cls.g, cls.plan, templates=[])
        # a clean layout: bed against the top wall, the rest spread out; the lamp on the desk
        cls.placed = {
            "bed": _pose("bed", [2.1, 1.6, .5], 2.5, 2.95, 0., -math.pi / 2, "floor"),
            "nightstand": _pose("nightstand", [.45, .45, .55], 1.15, 3.7, 0., -math.pi / 2, "floor"),
            "desk": _pose("desk", [.6, 1.2, .75], 4.6, 1.5, 0., math.pi, "floor"),
            "chair": _pose("chair", [.5, .5, .9], 3.95, 1.5, 0., 0., "floor"),
        }
        cls.placed["lamp"] = _pose("lamp", [.25, .25, .45], 4.6, 1.5, .75, math.pi, "desk")

    def test_clean_layout_has_no_hard_violation(self):
        self.assertEqual(self.critic.assess(self.placed).hard, [], self.critic.assess(self.placed).summary())

    def test_smaller_overlap_counts_as_progress(self):
        deep = {**self.placed, "chair": _pose("chair", [.5, .5, .9], 4.2, 1.5, 0., 0., "floor")}
        shallow = {**self.placed, "chair": _pose("chair", [.5, .5, .9], 4.1, 1.5, 0., 0., "floor")}
        a, b = self.critic.assess(deep), self.critic.assess(shallow)
        self.assertEqual(a.hard_count, b.hard_count)
        self.assertGreater(a.hard_magnitude, b.hard_magnitude)
        self.assertTrue(Critic.better(b, a)[0])
        self.assertFalse(Critic.better(a, b)[0])

    def test_hard_valid_layout_rejects_new_violation(self):
        clean = self.critic.assess(self.placed)
        worse = Assessment(hard=[Violation("collision", ("a", "b"), .02)], soft=[])
        self.assertFalse(Critic.better(worse, clean)[0])
        self.assertFalse(Critic.not_worse(worse, clean))

    def test_door_rule_follows_the_evaluator(self):
        # the door strip is x 0.15-1.05, y 0-0.8: a chair there blocks it, a rug does not
        chair = {**self.placed, "chair": _pose("chair", [.5, .5, .9], .6, .4, 0., 0., "floor")}
        self.assertIn("door_clearance", {v.code for v in self.critic.assess(chair).hard})
        condition = copy.deepcopy(self.condition)
        condition["objects"][3]["category"] = "rug"
        g = _geometry(condition, self.plan)
        rug = {**self.placed, "chair": _pose("chair", [1., 1., .01], .6, .4, 0., 0., "floor")}
        self.assertNotIn("door_clearance", {v.code for v in Critic.for_plan(g, self.plan, templates=[]).assess(rug).hard})

    def test_templates_score_distance_continuously(self):
        templates = [{"a": "nightstand", "b": "bed", "kind": "near", "max_gap_m": .05, "rooms": 1}]
        critic = Critic.for_plan(self.g, self.plan, templates=templates)
        near = critic.assess(self.placed)
        far = critic.assess({**self.placed, "nightstand": _pose("nightstand", [.45, .45, .55], .5, 2.0, 0., 0., "floor")})
        cost = lambda a: sum(v.cost for v in a.soft if v.code == "template_near")  # noqa: E731
        self.assertGreater(cost(far), cost(near))

    def test_fixed_sizes_do_not_shrink_in_the_loop(self):
        sizes = FixedSizes({"bed": [2.1, 1.6, .5]}, SizeCatalog(CATALOG), allow_shrink=False)
        self.assertEqual(sizes.shrunk({"id": "bed", "category": "bed"}, 1), [])


class LoopTests(unittest.TestCase):
    def test_offline_workflow_with_critic(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        result = run_draft_workflow(condition, None, catalog_path=str(CATALOG), llm=False,
                                    loop_budget=LoopBudget(gpt_repairs=0, search_seconds=60))
        critic = result["diagnostics"]["critic"]
        self.assertGreater(critic["draft"]["hard_count"], 0)  # every object starts at one point
        self.assertLess(critic["final"]["hard_count"], critic["draft"]["hard_count"])
        self.assertEqual(result["status"] == "clean", critic["final"]["hard_count"] == 0)
        sizes = result["diagnostics"]["sizes"]
        for box in result["layout"]["objects"]:  # sizes fixed: nothing shrunk
            self.assertEqual(box["target_size_local_m"], sizes[box["id"]])
        self.assertTrue(all("accepted" in a for a in critic["loop"]["attempts"]))

    def test_gpt_answer_respects_locked_supports_and_carries_children(self):
        condition = _room()
        plan = make_plan(condition, llm=False, semantic_only=True)
        g = _geometry(condition, plan)
        sizes = {"bed": [2.1, 1.6, .5], "nightstand": [.45, .45, .55], "desk": [.6, 1.2, .75], "chair": [.5, .5, .9],
                 "lamp": [.25, .25, .45]}
        placed = CriticTests.placed if hasattr(CriticTests, "placed") else None
        if placed is None:
            CriticTests.setUpClass()
            placed = CriticTests.placed
        loop = Loop(g, plan, sizes, SizeCatalog(CATALOG), Critic.for_plan(g, plan, templates=[]),
                    locked_parents=_required_parents(condition), budget=LoopBudget(gpt_repairs=1))
        answer = {"objects": {"desk": {"x": 4.6, "y": 2.2, "yaw_deg": 180},
                              "lamp": {"x": 1.0, "y": 1.0, "yaw_deg": 0, "support": "floor"}}}
        new, parents = loop._apply(answer, {"desk", "lamp"}, placed)
        self.assertEqual(parents["lamp"], "desk")  # the input's support_parent is locked
        self.assertAlmostEqual(new["lamp"]["bottom_center_m"][2], .75)
        self.assertAlmostEqual(new["desk"]["bottom_center_m"][1], 2.2)
        answer = {"objects": {"desk": {"x": 4.6, "y": 2.2, "yaw_deg": 180}}}
        new, _ = loop._apply(answer, {"desk", "lamp"}, placed)  # no pose for the lamp: carried with the desk
        self.assertAlmostEqual(new["lamp"]["bottom_center_m"][1], 2.2)


if __name__ == "__main__":
    unittest.main()
