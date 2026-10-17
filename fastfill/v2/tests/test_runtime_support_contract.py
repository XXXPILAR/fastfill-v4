"""Effective support and validator input-boundary regressions."""
import copy
import unittest

from fastfill.v2.runtime import Asset, AtomicMemoryHost, BoundedTranslationRepair, CatalogResolver, RuntimeBudget, SupportSurface, reconcile, run_pipeline
from fastfill.v2.tests.test_runtime import condition, desk, layout
from fastfill.v2.validation import validate_scene


class EffectiveSupportTests(unittest.TestCase):
    def test_hard_on_floor_alone_is_effective_support_without_mutation(self):
        req = condition([{"id": "desk", "category": "desk"}],
                        [{"type": "on", "object_id": "desk", "parent_id": "floor"}])
        original = copy.deepcopy(req)
        report = validate_scene(req, layout()["objects"])
        self.assertTrue(report["ok"])
        self.assertNotIn("support_unknown", report["unknown_checks"])
        host = AtomicMemoryHost()
        result = run_pipeline(req, layout(), CatalogResolver((desk(),)), host=host,
                              idempotency_key="r", expected_world_version=0)
        self.assertTrue(result["committed"])
        self.assertEqual(result["final_objects"][0]["support_parent"], "floor")
        self.assertEqual(req, original)

    def test_hard_on_asset_updates_actual_height_and_preserves_targets(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp"}],
                        [{"type": "on", "object_id": "lamp", "parent_id": "desk", "surface_id": "top"}])
        pred = layout(layout()["objects"] + [{"id": "lamp", "target_size_local_m": [.2, .2, .4],
                                              "bottom_center_m": [2, 2, .75], "yaw_rad": 0}])
        surface = SupportSurface("top", ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), .7)
        original = copy.deepcopy((req, pred))
        result = run_pipeline(req, pred, CatalogResolver((desk(size=(1, 1, .7), support_surfaces=(surface,)),
                              Asset("lamp", "lamp", (.2, .2, .4)))))
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_objects"][1]["bottom_center_m"][2], .7)
        self.assertEqual(result["final_objects"][1]["target_bottom_center_m"][2], .75)
        self.assertEqual((req, pred), original)

    def test_soft_on_does_not_override_declared_floor_support(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"}],
                        [{"type": "on", "object_id": "desk", "parent_id": "wall", "hard": False}])
        self.assertTrue(validate_scene(req, layout()["objects"])["ok"])
        result = run_pipeline(req, layout(), CatalogResolver((desk(),)))
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_objects"][0]["support_parent"], "floor")

    def test_soft_on_alone_does_not_claim_hard_support_evidence(self):
        req = condition([{"id": "desk", "category": "desk"}],
                        [{"type": "on", "object_id": "desk", "parent_id": "floor", "hard": False}])
        report = validate_scene(req, layout()["objects"])
        self.assertFalse(report["ok"])
        self.assertIn("support_unknown", report["unknown_checks"])

    def test_conflicting_hard_parent_constraints_reject_before_resolution(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"}],
                        [{"type": "on", "object_id": "desk", "parent_id": "wall"}])
        host = AtomicMemoryHost()
        result = run_pipeline(req, layout(), CatalogResolver((desk(),)), host=host,
                              idempotency_key="r", expected_world_version=0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["metrics"]["asset"]["resolver_calls"], 0)
        self.assertIn("conflicting", result["diagnostics"][0]["message"])
        self.assertEqual(host.snapshot()["commits"], 0)

    def test_hard_on_cycle_is_detected_by_direct_validator(self):
        req = condition([{"id": ident, "category": "desk"} for ident in ("desk", "other")],
                        [{"type": "on", "object_id": "desk", "parent_id": "other"},
                         {"type": "on", "object_id": "other", "parent_id": "desk"}])
        pred = [{**layout()["objects"][0], "id": ident, "bottom_center_m": [x, 2, 0]}
                for ident, x in (("desk", 1), ("other", 3))]
        report = validate_scene(req, pred)
        self.assertFalse(report["ok"])
        self.assertIn("cycle", report["checks"][0]["message"])

    def test_conflicting_surface_ids_are_rejected(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor", "support_surface_id": "a"}],
                        [{"type": "on", "object_id": "desk", "parent_id": "floor", "surface_id": "b"}])
        report = validate_scene(req, layout()["objects"])
        self.assertFalse(report["ok"])
        self.assertIn("conflicting", report["checks"][0]["message"])

    def test_translation_repair_follows_hard_on_descendants(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp"}],
                        [{"type": "on", "object_id": "lamp", "parent_id": "desk"}])
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]},
                       {"id": "lamp", "target_size_local_m": [.2, .2, .4], "bottom_center_m": [.2, 2, .75], "yaw_rad": 0}])
        surface = SupportSurface("top", ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), .75)
        result = run_pipeline(req, pred, CatalogResolver((desk(support_surfaces=(surface,)), Asset("lamp", "lamp", (.2, .2, .4)))),
                              budget=RuntimeBudget(max_asset_retries=0, max_repair_calls=1), repair=BoundedTranslationRepair(.4))
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["final_objects"][1]["bottom_center_m"][0], .6)


class ValidatorBoundaryTests(unittest.TestCase):
    def test_unknown_or_malformed_required_levels_are_rejected(self):
        for required in (("meshh",), "mesh", [], ("bbox", "mesh", "mesh"), (None,), None):
            with self.subTest(required=required), self.assertRaises(ValueError):
                validate_scene(condition(), layout()["objects"], required_levels=required)

    def test_pipeline_rejects_unknown_required_level_before_resolution(self):
        result = run_pipeline(condition(), layout(), CatalogResolver((desk(),)), required_levels=("meshh",))
        self.assertFalse(result["ok"])
        self.assertEqual(result["metrics"]["asset"]["resolver_calls"], 0)

    def test_direct_actual_capability_metadata_requires_typed_names(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor", "required_capabilities": ["work_surface"]}])
        obj = {**layout()["objects"][0], "actual_size_local_m": [1, 1, .75]}
        for capabilities in ("work_surface", {"work_surface": False}, {"work_surface"}, [""], [True], ["work_surface", "  "]):
            with self.subTest(capabilities=capabilities):
                report = validate_scene(req, [{**obj, "capabilities": capabilities}], stage="actual")
                self.assertFalse(report["ok"])
                self.assertEqual(report["checks"][0]["code"], "invalid_geometry")

    def test_invalid_capability_metadata_cannot_pass_when_not_requested(self):
        obj = {**layout()["objects"][0], "actual_size_local_m": [1, 1, .75], "capabilities": "work_surface"}
        report = validate_scene(condition(), [obj], stage="actual")
        self.assertFalse(report["ok"])
        self.assertEqual(report["checks"][0]["code"], "invalid_geometry")


class ExistingRelationTests(unittest.TestCase):
    def test_against_wall_requires_one_complete_edge_within_tolerance(self):
        req = condition(constraints=[{"type": "against_wall", "object_id": "desk", "tolerance_m": .1}])
        for x, expected in ((.55, True), (.8, False)):
            with self.subTest(x=x):
                obj = {**layout()["objects"][0], "bottom_center_m": [x, 2, 0]}
                report = validate_scene(req, [obj])
                self.assertEqual(report["ok"], expected)
                check = next(check for check in report["checks"] if check.get("constraint_type") == "against_wall")
                self.assertEqual(check["status"], "pass" if expected else "violation")

    def test_against_wall_does_not_accept_only_nearby_corners(self):
        import math
        req = condition(constraints=[{"type": "against_wall", "object_id": "desk"}])
        obj = {**layout()["objects"][0], "bottom_center_m": [.71, .71, 0], "yaw_rad": math.pi/4}
        report = validate_scene(req, [obj])
        check = next(check for check in report["checks"] if check.get("constraint_type") == "against_wall")
        self.assertEqual(check["status"], "violation")

    def test_against_wall_unknown_boundary_is_not_evidence(self):
        req = condition(constraints=[{"type": "against_wall", "object_id": "desk"}])
        req = {**req, "room": {**req["room"], "boundary_known": False}}
        obj = {**layout()["objects"][0], "bottom_center_m": [.5, 2, 0]}
        report = validate_scene(req, [obj])
        check = next(check for check in report["checks"] if check.get("constraint_type") == "against_wall")
        self.assertEqual(check["status"], "unknown")
        self.assertFalse(report["ok"])

    def test_between_requires_segment_through_footprint_interior(self):
        req = condition([{"id": ident, "category": "desk", "support_parent": "floor"} for ident in ("a", "b", "c")],
                        [{"type": "between", "object_id": "a", "target_ids": ["b", "c"]}])
        for y, expected in ((2, True), (2.2, False), (3, False)):
            with self.subTest(y=y):
                objects = [{"id": ident, "target_size_local_m": [.4, .4, .75], "bottom_center_m": pos, "yaw_rad": 0}
                           for ident, pos in (("a", [2, 2, 0]), ("b", [.5, y, 0]), ("c", [3.5, y, 0]))]
                report = validate_scene(req, objects)
                self.assertEqual(report["ok"], expected)
                check = next(check for check in report["checks"] if check.get("constraint_type") == "between")
                self.assertEqual(check["status"], "pass" if expected else "violation")

    def test_between_missing_references_cannot_pass(self):
        req = condition(constraints=[{"type": "between", "object_id": "desk", "target_ids": ["missing", "desk"]}])
        report = validate_scene(req, layout()["objects"])
        self.assertFalse(report["ok"])
        self.assertIn("constraint_reference", {check["code"] for check in report["checks"]})


if __name__ == "__main__":
    unittest.main()
