from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from workflow.catalog import SizeCatalog, build_catalog
from workflow.contract import check_request, validate_layout
from workflow.export import roomgenbench_scene
from workflow.geometry import Geometry
from workflow.pipeline import run_workflow
from workflow.planning import make_plan

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.condition = json.loads((ROOT / "examples" / "bedroom.json").read_text())
        cls.result = run_workflow(cls.condition, catalog_path=ROOT / "size_catalog.json", llm=False,
                                  beam_width=12, candidate_limit=32)

    def test_no_furniture_bbox_in_input(self):
        self.assertEqual(self.result["status"], "ok")
        self.assertTrue(all("target_size_local_m" not in obj and "bottom_center_m" not in obj
                            for obj in self.condition["objects"]))
        self.assertEqual({obj["id"] for obj in self.result["layout"]["objects"]},
                         {obj["id"] for obj in self.condition["objects"]})
        validate_layout(self.result["layout"], self.condition)

    def test_hard_geometry_and_support(self):
        plan = make_plan(self.condition, llm=False)
        geometry = Geometry(self.condition, plan.parents, plan.relations)
        placed = {obj["id"]: obj for obj in self.result["layout"]["objects"]}
        self.assertEqual(geometry.validate(placed), [])
        self.assertEqual(placed["lamp"]["support_parent"], "desk")
        self.assertAlmostEqual(placed["lamp"]["bottom_center_m"][2],
                               placed["desk"]["target_size_local_m"][2])

    def test_roomgenbench_axes_and_support(self):
        scene = roomgenbench_scene(self.condition, self.result["layout"])
        by_id = {obj["id"]: obj for obj in scene["objects"]}
        source = {obj["id"]: obj for obj in self.result["layout"]["objects"]}
        self.assertEqual(by_id["lamp"]["place_id"], "desk")
        for ident, obj in by_id.items():
            self.assertAlmostEqual(obj["dimensions"]["width"], source[ident]["target_size_local_m"][1])
            self.assertAlmostEqual(obj["dimensions"]["length"], source[ident]["target_size_local_m"][0])
            diff = math.radians(obj["rotation"]["z"]) + math.pi / 2 - source[ident]["yaw_rad"]
            self.assertAlmostEqual((diff + math.pi) % (2 * math.pi) - math.pi, 0)

    def test_reject_size_as_condition(self):
        wrong = json.loads(json.dumps(self.condition))
        wrong["objects"][0]["fixed_size_local_m"] = [2, 1.5, .6]
        with self.assertRaisesRegex(ValueError, "size must not be an input"):
            check_request(wrong)

    def test_reject_unverifiable_opening(self):
        wrong = json.loads(json.dumps(self.condition))
        wrong["room"]["openings"][0]["wall_index"] = 99
        with self.assertRaisesRegex(ValueError, "wall_index"):
            check_request(wrong)

    def test_infeasible_returns_no_layout(self):
        impossible = json.loads(json.dumps(self.condition))
        impossible["room"]["floor_polygon_xy_m"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        impossible["room"]["openings"] = []
        impossible["room"]["fixed_objects"] = []
        impossible["objects"] = [impossible["objects"][0]]
        impossible["constraints"] = []
        result = run_workflow(impossible, catalog_path=ROOT / "size_catalog.json", llm=False,
                              beam_width=8, candidate_limit=16)
        self.assertEqual(result["status"], "infeasible_or_search_exhausted")
        self.assertIsNone(result["layout"])

    def test_train_only_catalog(self):
        source = {"condition": {"objects": [{"id": "x", "category": "test chair"}]},
                  "target": {"objects": [{"id": "x", "target_size_local_m": [.5, .6, .7]}]},
                  "validity": {"size": [[True, True, True]]}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            train = path / "train.jsonl"
            train.write_text((json.dumps(source) + "\n") * 20)
            build_catalog(train, path / "catalog.json")
            catalog = SizeCatalog(path / "catalog.json")
            self.assertEqual(catalog.categories["test chair"]["median"], [.5, .6, .7])
            fake = json.loads((path / "catalog.json").read_text())
            fake["split"] = "validation"
            (path / "catalog.json").write_text(json.dumps(fake))
            with self.assertRaisesRegex(ValueError, "train split"):
                SizeCatalog(path / "catalog.json")

    def test_external_draft_is_a_proposal(self):
        draft = json.loads(json.dumps(self.result["layout"]))
        draft["objects"][0]["bottom_center_m"][0] = 99
        result = run_workflow(self.condition, catalog_path=ROOT / "size_catalog.json",
                              draft_layout=draft, llm=False, beam_width=8, candidate_limit=24)
        self.assertEqual(result["status"], "ok")
        self.assertNotEqual(result["layout"]["objects"][0]["bottom_center_m"][0], 99)
        self.assertIn("external_draft", result["diagnostics"]["plan_source"])

    def test_thin_overlap_policy_is_explicit(self):
        condition = json.loads(json.dumps(self.condition))
        condition["room"]["openings"] = []
        condition["room"]["fixed_objects"] = []
        condition["objects"] = [
            {"id": "rug", "category": "rug", "description": "rug"},
            {"id": "chair", "category": "chair", "description": "chair"},
        ]
        condition["constraints"] = []
        parents = {"rug": "floor", "chair": "floor"}
        rug = {"id": "rug", "target_size_local_m": [2, 2, .02],
               "bottom_center_m": [2, 2, 0], "yaw_rad": 0, "support_parent": "floor"}
        chair = {"id": "chair", "target_size_local_m": [.5, .5, .8],
                 "bottom_center_m": [2, 2, 0], "yaw_rad": 0, "support_parent": "floor"}
        strict = Geometry(condition, parents, [])
        visual = Geometry(condition, parents, [], allow_thin_overlap=True)
        self.assertIn("collision", [issue.code for issue in strict.check_box(chair, {"rug": rug})])
        self.assertNotIn("collision", [issue.code for issue in visual.check_box(chair, {"rug": rug})])


if __name__ == "__main__":
    unittest.main()
