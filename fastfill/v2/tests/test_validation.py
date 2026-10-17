"""Geometry validity is reported independently of mesh, physics and Solver."""
import math
import unittest

from fastfill.v2.validation import validate_scene
from fastfill.v2.tests.test_runtime import condition, layout


class ValidationTests(unittest.TestCase):
    def test_target_bbox_validity_does_not_claim_mesh_or_physics(self):
        out = validate_scene(condition(), layout()["objects"])
        self.assertTrue(out["ok"])
        self.assertEqual(out["geometry_level"], "bbox")
        self.assertTrue({"mesh_unchecked", "physics_unchecked", "solver_unchecked"} <= set(out["unknown_checks"]))

    def test_partial_floor_polygon_is_not_boundary_evidence(self):
        req = condition()
        req["room"] = {**req["room"], "boundary_known": False}
        out = validate_scene(req, layout()["objects"])
        self.assertFalse(out["ok"])
        self.assertIn("boundary_unknown", out["unknown_checks"])

    def test_floor_unknown_is_not_treated_as_zero(self):
        req = condition()
        req["room"] = {**req["room"], "floor_known": False}
        out = validate_scene(req, layout()["objects"])
        self.assertFalse(out["ok"])
        self.assertIn("floor_unknown", out["unknown_checks"])

    def test_rotation_boundary_and_collision_use_oriented_local_box(self):
        obj = {**layout()["objects"][0], "target_size_local_m": [2, .4, .75],
               "bottom_center_m": [.3, 2, 0], "yaw_rad": math.pi / 2}
        self.assertTrue(validate_scene(condition(), [obj])["ok"])
        self.assertFalse(validate_scene(condition(), [{**obj, "yaw_rad": 0}])["ok"])

    def test_pair_collision_is_hard_without_semantic_category_exemptions(self):
        req = condition([{"id": "a", "category": "chair", "support_parent": "floor"},
                         {"id": "b", "category": "desk", "support_parent": "floor"}])
        objs = [{**layout()["objects"][0], "id": id} for id in ("a", "b")]
        out = validate_scene(req, objs)
        self.assertFalse(out["ok"])
        self.assertIn("collision", {c["code"] for c in out["checks"]})

    def test_touching_boxes_are_not_collisions(self):
        req = condition([{"id": "a", "category": "desk", "support_parent": "floor"},
                         {"id": "b", "category": "desk", "support_parent": "floor"}])
        objs = [{**layout()["objects"][0], "id": "a", "bottom_center_m": [1, 2, 0]},
                {**layout()["objects"][0], "id": "b", "bottom_center_m": [2, 2, 0]}]
        self.assertTrue(validate_scene(req, objs)["ok"])

    def test_unknown_hard_constraint_blocks_but_soft_constraint_reports(self):
        for hard, ok in ((True, False), (False, True)):
            req = condition(constraints=[{"type": "physics_magic", "hard": hard}])
            out = validate_scene(req, layout()["objects"])
            self.assertEqual(out["ok"], ok)
            self.assertIn("constraint_unknown", out["unknown_checks"])

    def test_fixed_geometry_collision_is_checked(self):
        req = condition()
        req["room"] = {**req["room"], "fixed_objects": [{"id": "column", "size_local_m": [.2, .2, 3],
                          "bottom_center_m": [2, 2, 0], "yaw_rad": 0}]}
        out = validate_scene(req, layout()["objects"])
        self.assertFalse(out["ok"])
        self.assertIn("fixed_collision", {c["code"] for c in out["checks"]})

    def test_invalid_target_is_filtered_before_geometry_arithmetic(self):
        for size in ((float("nan"), 1, 1), (0, 1, 1), (-1, 1, 1)):
            out = validate_scene(condition(), [{**layout()["objects"][0], "target_size_local_m": size}])
            self.assertFalse(out["ok"])
            self.assertEqual(out["checks"][0]["code"], "invalid_geometry")

    def test_actual_capability_requirements_are_rechecked(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor", "required_capabilities": ["work_surface"]}])
        obj = {**layout()["objects"][0], "actual_size_local_m": [1, 1, .75]}
        for capabilities, status in ((None, "unknown"), ([], "violation"), (["work_surface"], "pass")):
            with self.subTest(capabilities=capabilities):
                report = validate_scene(req, [{**obj, "capabilities": capabilities}], stage="actual")
                check = next(check for check in report["checks"] if check["code"] == "capabilities")
                self.assertEqual(check["status"], status)
                self.assertEqual(report["ok"], status == "pass")

    def test_unconverted_openings_are_unknown_and_block_commit(self):
        req = condition()
        req["room"] = {**req["room"], "openings": [{"kind": "door", "position": [2, 0, 0]}]}
        report = validate_scene(req, layout()["objects"])
        self.assertFalse(report["ok"])
        self.assertIn("openings_unchecked", report["unknown_checks"])


if __name__ == "__main__":
    unittest.main()
