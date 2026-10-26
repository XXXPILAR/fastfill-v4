from __future__ import annotations

import json
import unittest
from pathlib import Path

from test_staged import SIZES, FakeLLM, _room  # shared fixtures
from workflow import credentials
from workflow.loop import LoopBudget
from workflow.pipeline import run_draft_workflow
from workflow.planning import make_plan
from workflow.semantic import REVIEW_INSTRUCTIONS
from workflow.simple import run_simple

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "size_catalog.json"


def _request():
    return json.loads((ROOT / "examples" / "bedroom.json").read_text())


class SimpleTests(unittest.TestCase):
    def test_offline_run_keeps_every_stage(self):
        result = run_simple(_request(), None, catalog_path=str(CATALOG), llm=False)
        d = result["diagnostics"]
        self.assertEqual(d["placement"], "simple")
        self.assertEqual(set(d["stages"]), {"draft", "after_check", "final"})
        self.assertEqual({b["id"] for b in result["layout"]["objects"]}, {o["id"] for o in _request()["objects"]})
        self.assertEqual(d["semantic"], {"result": "skipped"})
        # every object starts at one point offline: the one check repairs hard violations only
        self.assertGreater(d["assessments"]["draft"]["hard_count"], 0)
        self.assertLess(d["assessments"]["after_check"]["hard_count"], d["assessments"]["draft"]["hard_count"])
        self.assertEqual(d["check"]["result"], "repaired")
        self.assertEqual(d["check"]["termination"], "no issue left to try")
        self.assertTrue(all(a["target"]["code"] not in ("relation_plan", "template_near", "template_faces",
                                                        "wall_back", "reach", "access")
                            for a in d["check"]["attempts"]))  # soft costs are not chased
        self.assertEqual(result["status"] == "clean", d["critic"]["final"]["hard_count"] == 0)

    def test_one_review_round_of_the_whole_room(self):
        llm = FakeLLM()
        plan = make_plan(_room(), llm=False, semantic_only=True)
        result = run_simple(_request(), SIZES, plan=plan, catalog_path=str(CATALOG), llm=False, semantic_ask=llm)
        review = result["diagnostics"]["semantic"]
        self.assertEqual(review["stage"], "room")
        self.assertEqual(len(review["rounds"]), 1)
        self.assertEqual(llm.seen.count("review"), 1)
        first = next(p for k, p in llm.payloads if k == REVIEW_INSTRUCTIONS)
        self.assertTrue(all(o["status"] == "review" for o in first["objects"]))

    def test_review_never_breaks_legality(self):
        llm = FakeLLM(fix={"x": 2.5, "y": 2.95, "yaw_deg": 0})  # a fix that drops the chair into the bed
        plan = make_plan(_room(), llm=False, semantic_only=True)
        result = run_simple(_request(), SIZES, plan=plan, catalog_path=str(CATALOG), llm=False, semantic_ask=llm)
        d = result["diagnostics"]
        self.assertLessEqual(d["assessments"]["final"]["hard_count"], d["assessments"]["after_check"]["hard_count"])

    def test_pipeline_default_is_simple(self):
        result = run_draft_workflow(_request(), None, catalog_path=str(CATALOG), llm=False,
                                    loop_budget=LoopBudget(gpt_repairs=0))
        self.assertEqual(result["diagnostics"]["placement"], "simple")
        self.assertIn("plan", result["timing_s"])

    def test_key_file_is_the_original_workflows(self):
        """workflow2 has no .env of its own: it reads the original workflow's in place (never a copy)."""
        if not credentials._OWN_ENV_FILE.is_file():
            expected = credentials._V1_ENV_FILE if credentials._V1_ENV_FILE.is_file() else credentials._OWN_ENV_FILE
            self.assertEqual(credentials.DEFAULT_ENV_FILE, expected)


if __name__ == "__main__":
    unittest.main()
