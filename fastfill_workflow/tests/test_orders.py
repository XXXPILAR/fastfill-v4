from __future__ import annotations

import json
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog
from workflow.contract import check_request, fill_room_defaults
from workflow.drafting import ALL_OBJECTS_RULE, DRAFT_INSTRUCTIONS, PARTIAL_INSTRUCTIONS, partial_request_text
from workflow.orders import critic_geometry, functional_steps, run_order, run_order_variants, steps_for
from workflow.loop import LoopBudget
from workflow.planning import Plan, make_plan
from workflow.solver import _pose

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"


def _room():
    condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
    return check_request(fill_room_defaults(condition)[0])


class OrderTests(unittest.TestCase):
    def test_functional_steps_keep_supports_first(self):
        condition = _room()
        # the lamp stands on the desk but is grouped with the bed, which comes first: it must wait for the desk
        plan = Plan(parents={"bed": "floor", "nightstand": "floor", "desk": "floor", "chair": "floor", "lamp": "desk"},
                    groups=[{"anchor": "bed", "members": ["nightstand", "lamp"]}, {"anchor": "desk", "members": ["chair"]}],
                    priority={"bed": 1, "desk": 2})
        g = critic_geometry(condition, plan)
        steps = functional_steps(g, plan, ["bed", "nightstand", "desk", "lamp", "chair"])
        where = {i: k for k, step in enumerate(steps) for i in step}
        self.assertLessEqual(where["desk"], where["lamp"])
        self.assertEqual(sorted(i for s in steps for i in s), sorted(g.requests))

    def test_steps_per_mode(self):
        condition = _room()
        plan = make_plan(condition, llm=False, semantic_only=True)
        g = critic_geometry(condition, plan)
        sizes = {o["id"]: SizeCatalog(CATALOG).candidates(o)[0][0] for o in condition["objects"]}
        self.assertEqual(len(steps_for("A", g, plan, sizes, SizeCatalog(CATALOG))), 1)
        self.assertEqual(len(steps_for("B", g, plan, sizes, SizeCatalog(CATALOG))), len(condition["objects"]))
        self.assertEqual(len(steps_for("B", g, plan, sizes, SizeCatalog(CATALOG), batch=2)), 3)

    def test_partial_request_marks_status(self):
        condition = _room()
        plan = make_plan(condition, llm=False, semantic_only=True)
        sizes = {o["id"]: [.5, .5, .5] for o in condition["objects"]}
        placed = {"bed": _pose("bed", [2.1, 1.6, .5], 2.5, 2.95, 0., -1.5708, "floor")}
        payload = json.loads(partial_request_text(condition, plan, sizes, ["desk", "lamp"], placed))
        status = {o["id"]: o["status"] for o in payload["objects"]}
        self.assertEqual(status["bed"], "placed")
        self.assertEqual((status["desk"], status["lamp"]), ("place_now", "place_now"))
        self.assertEqual(status["chair"], "later")

    def test_every_mode_returns_a_complete_layout(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        plan = make_plan(check_request(fill_room_defaults(condition)[0]), llm=False, semantic_only=True)
        for mode in ("A", "B", "C"):
            result = run_order(condition, None, plan, mode, catalog_path=str(CATALOG), llm=False,
                               loop_budget=LoopBudget(gpt_repairs=0, search_seconds=30))
            self.assertEqual({b["id"] for b in result["layout"]["objects"]}, {o["id"] for o in condition["objects"]})
            self.assertEqual(result["diagnostics"]["mode"], mode)
            self.assertEqual(result["status"] == "clean", result["diagnostics"]["final"]["hard_count"] == 0)

    def test_partial_prompt_does_not_ask_for_every_object(self):
        self.assertIn(ALL_OBJECTS_RULE, DRAFT_INSTRUCTIONS)
        self.assertNotIn(ALL_OBJECTS_RULE, PARTIAL_INSTRUCTIONS)
        self.assertIn("place_now", PARTIAL_INSTRUCTIONS)

    def test_final_variant_shares_the_steps(self):
        condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        plan = make_plan(check_request(fill_room_defaults(condition)[0]), llm=False, semantic_only=True)
        for mode in ("B", "C"):
            results = run_order_variants(condition, None, plan, mode, catalog_path=str(CATALOG), llm=False,
                                         loop_budget=LoopBudget(gpt_repairs=0, search_seconds=30),
                                         final_repair=(False, True))
            plain, final = results[mode], results[f"{mode}_final"]
            self.assertEqual(plain["diagnostics"]["steps"], final["diagnostics"]["steps"])
            self.assertEqual(plain["diagnostics"]["after_steps"], final["diagnostics"]["after_steps"])
            self.assertIsNone(plain["diagnostics"]["final_loop"])
            self.assertIsNotNone(final["diagnostics"]["final_loop"])
            self.assertLessEqual(final["diagnostics"]["final"]["hard_count"], plain["diagnostics"]["after_steps"]["hard_count"])


if __name__ == "__main__":
    unittest.main()
