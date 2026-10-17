"""Evaluation retains malformed predictions and asset/repair failures in denominators."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastfill.v2.evaluate import evaluate_layout, reference_metrics, run_evaluation, summarize
from fastfill.v2.runtime import Asset, CatalogResolver
from fastfill.v2.tests.test_runtime import condition, layout


def sample():
    return {"schema_version": "fastfill.v2", "condition": condition(), "target": layout(),
            "validity": {"position": [[True]*3], "size": [[True]*3], "yaw": [True], "yaw_symmetry_order": [1]},
            "provenance": {"source": "fixture", "house_id": "house", "scene_id": "scene", "split": "test"}}


class EvaluationRuntimeTests(unittest.TestCase):
    def test_all_malformed_prediction_rows_are_counted_and_safe_to_write(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions, catalog = (root / name for name in ("data.jsonl", "predictions.jsonl", "catalog.json"))
            data.write_text((json.dumps(sample())+"\n")*3)
            predictions.write_text("{broken\n[]\n"+json.dumps(layout([{**layout()["objects"][0], "yaw_rad": float("nan")}]))+"\n")
            catalog.write_text(json.dumps({"assets": [{"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, .75]}]}))
            report = run_evaluation(data, root / "report", predictions=predictions, catalog=catalog, commit_in_memory=True)
            self.assertEqual(report["requests"], 3)
            self.assertEqual(report["failed_requests"], 3)
            self.assertEqual(report["inference_failed_requests"], 3)
            self.assertEqual(report["stage_failures"]["model_schema"], 3)
            self.assertEqual(report["stage_failures"]["asset_resolution"], 3)
            self.assertEqual(report["system"]["final_commit"], 0)
            outcomes = [json.loads(line) for line in (root / "report/outcomes.jsonl").read_text().splitlines()]
            self.assertEqual(len(outcomes), 3)
            self.assertTrue(all(isinstance(row["raw_prediction"], str) for row in outcomes))

    def test_system_failure_is_distinct_from_successful_schema_parse(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions, catalog = (root / name for name in ("data.jsonl", "predictions.jsonl", "catalog.json"))
            data.write_text(json.dumps(sample())+"\n")
            predictions.write_text(json.dumps(layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]}]))+"\n")
            catalog.write_text(json.dumps([{ "ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, .75]}]))
            report = run_evaluation(data, root / "report", predictions=predictions, catalog=catalog,
                                    commit_in_memory=True, asset_retries=0)
            self.assertEqual(report["model"]["schema_success"], 1)
            self.assertEqual(report["inference_failed_requests"], 0)
            self.assertEqual(report["system_failed_requests"], 1)
            self.assertEqual(report["failed_requests"], 1)
            self.assertEqual(report["system"]["final_commit"], 0)
            outcome = json.loads((root / "report/outcomes.jsonl").read_text())
            self.assertEqual(outcome["actual_resolved"][0]["asset_ref"], "desk")
            self.assertIsNone(report["latency_ms"]["end_to_end_p50"])
            self.assertIsNotNone(report["latency_ms"]["runtime_p50"])
            self.assertIn("evaluation_wall_time_ms", outcome)

    def test_evaluation_uses_the_selected_bounded_repair_budget(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, predictions, catalog = (root / name for name in ("data.jsonl", "predictions.jsonl", "catalog.json"))
            data.write_text(json.dumps(sample())+"\n")
            predictions.write_text(json.dumps(layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]}]))+"\n")
            catalog.write_text(json.dumps([{ "ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, .75]}]))
            report = run_evaluation(data, root / "report", predictions=predictions, catalog=catalog,
                                    commit_in_memory=True, asset_retries=0, repair_calls=1, repair_step_m=.4)
            self.assertEqual(report["system_failed_requests"], 0)
            self.assertEqual(report["stage_failures"]["target_geometry"], 1)
            self.assertEqual(report["stage_failures"]["actual_first_pass"], 1)
            self.assertEqual(report["system"]["repair_calls"], 1)
            self.assertEqual(report["system"]["final_commit"], 1)
            outcome = json.loads((root / "report/outcomes.jsonl").read_text())
            self.assertEqual(outcome["actual_resolved"][0]["bottom_center_m"], [.2, 2, 0])
            self.assertAlmostEqual(outcome["final_output"][0]["bottom_center_m"][0], .6)

    def test_operational_latency_excludes_reference_evaluation_wall_time(self):
        outcome = evaluate_layout(layout(), sample(), resolver=CatalogResolver((Asset("desk", "desk", (1, 1, .75)),)))
        outcome["fastfill_latency_ms"] = 10
        outcome["evaluation_wall_time_ms"] = 1000
        outcome["system"]["end_to_end_latency_ms"] = 2
        report = summarize([outcome])
        self.assertEqual(report["latency_ms"]["end_to_end_p50"], 12)
        self.assertEqual(report["latency_ms"]["runtime_p50"], 2)
        self.assertEqual(report["latency_ms"]["evaluation_wall_time_p50"], 1000)

    def test_size_incomplete_exchange_labels_match_on_position_and_do_not_skip_runtime(self):
        # Legacy row form: the group label still sits on the condition objects.
        objects = [{"id": ident, "category": "desk", "description": "desk", "support_parent": "floor",
                    "exchangeable_group": "desks"} for ident in ("desk", "other")]
        preds = [{**layout()["objects"][0], "id": "desk", "bottom_center_m": [3, 2, 0]},
                 {**layout()["objects"][0], "id": "other", "bottom_center_m": [1, 2, 0]}]
        row = {**sample(), "condition": condition(objects), "target": layout([
                {**preds[1], "id": "desk"}, {**preds[0], "id": "other", "target_size_local_m": [None, None, None]}]),
               "validity": {"position": [[True]*3, [True]*3], "size": [[True]*3, [False]*3], "yaw": [True, True]}}
        outcome = evaluate_layout(layout(preds), row, resolver=CatalogResolver((Asset("desk", "desk", (1, 1, .75)),)),
                                  commit_in_memory=True)
        self.assertTrue(outcome["model"]["schema_success"])
        self.assertEqual(outcome["model"]["reference"]["matching_scope"], "exchangeable_groups")
        self.assertEqual(outcome["model"]["reference"]["position_only_groups"], ["desks"])
        self.assertEqual(outcome["model"]["reference"]["bottom_center_error_m"]["mean"], 0.)
        self.assertEqual(outcome["model"]["reference"]["log_size_error"]["valid_objects"], 1)
        self.assertTrue(outcome["runtime"]["committed"])

    def test_position_incomplete_group_does_not_disable_matching_for_other_complete_group(self):
        objects, targets, predictions = [], [], []
        for ident, group, x in (("a", "complete", 1.), ("b", "complete", 3.),
                                ("c", "incomplete", 1.), ("d", "incomplete", 3.)):
            objects.append({"id": ident, "category": "desk", "description": "desk",
                            "support_parent": "floor", "exchangeable_group": group})
            targets.append({"id": ident, "target_size_local_m": [1., 1., 1.],
                            "bottom_center_m": [x, 2., 0.] if ident != "d" else [None] * 3, "yaw_rad": 0.})
            predictions.append({"id": ident, "target_size_local_m": [1., 1., 1.],
                                "bottom_center_m": [4. - x if group == "complete" else x, 2., 0.],
                                "yaw_rad": 0.})
        row = {**sample(), "condition": condition(objects), "target": layout(targets),
               "validity": {"position": [[True] * 3] * 3 + [[False] * 3],
                            "size": [[True] * 3] * 4, "yaw": [False] * 4}}
        metrics = reference_metrics(layout(predictions), row, include_iou=False)
        self.assertEqual(metrics["bottom_center_error_m"], {"mean": 0., "valid_objects": 3})
        self.assertEqual(metrics["matching_scope"], "exchangeable_complete_groups_fixed_incomplete_groups")
        self.assertEqual(metrics["incomplete_groups"], ["incomplete"])
        self.assertEqual(metrics["position_only_groups"], [])

    def test_required_mesh_validation_is_consistent_in_evaluation(self):
        outcome = evaluate_layout(layout(), sample(), resolver=CatalogResolver((Asset("desk", "desk", (1, 1, .75)),)),
                                  commit_in_memory=True, required_levels=("bbox", "mesh"))
        self.assertTrue(outcome["model"]["schema_success"])
        self.assertFalse(outcome["runtime"]["committed"])
        self.assertIn("mesh_unchecked", outcome["runtime"]["validation"]["final"]["unknown_checks"])


if __name__ == "__main__":
    unittest.main()
