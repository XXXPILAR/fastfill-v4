"""Offline end-to-end contract tests; no external host is contacted."""
import copy
import math
import unittest

from fastfill.v2.runtime import (
    Asset, AtomicMemoryHost, BoundedTranslationRepair, CatalogResolver, RuntimeBudget, SupportSurface,
    reconcile, run_pipeline,
)


def condition(objects=None, constraints=None):
    return {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "floor_z_m": 0.0, "height_m": 3.0,
    }, "objects": [{"description": obj.get("category", "object"), **obj} for obj in
                    (objects or [{"id": "desk", "category": "desk", "support_parent": "floor"}])],
        "constraints": constraints or []}


def layout(objects=None):
    return {"schema_version": "fastfill.v2", "objects": objects or [{"id": "desk",
        "target_size_local_m": [1.0, 1.0, 0.75], "bottom_center_m": [2.0, 2.0, 0.0], "yaw_rad": 0.0}]}


def desk(ref="desk_asset", size=(1.0, 1.0, 0.75), **kwargs):
    return Asset(ref=ref, category="desk", actual_size_local_m=size, **kwargs)


class RuntimeContractTests(unittest.TestCase):
    def test_actual_and_target_sizes_stay_distinct_and_inputs_unchanged(self):
        req, pred = condition(), layout()
        before = copy.deepcopy((req, pred))
        result = run_pipeline(req, pred, CatalogResolver((desk(size=(0.9, 1.0, 0.7)),)))
        self.assertTrue(result["ok"])
        obj = result["final_objects"][0]
        self.assertEqual(obj["target_size_local_m"], [1.0, 1.0, 0.75])
        self.assertEqual(obj["actual_size_local_m"], [0.9, 1.0, 0.7])
        self.assertEqual((req, pred), before)
        self.assertTrue(result["validation"]["actual_first_pass"]["ok"])
        self.assertEqual(result["metrics"]["asset"]["retrieval_coverage"], 1.0)

    def test_raw_pivot_transform_is_composed_after_canonical_conversion(self):
        raw_to_canonical = ((2, 0, 0, -1), (0, 2, 0, -2), (0, 0, 2, -3), (0, 0, 0, 1))
        asset = desk(canonical_transform=raw_to_canonical)
        pred = {**layout()["objects"][0], "bottom_center_m": [5, 6, 7], "yaw_rad": math.pi / 2}
        out = reconcile({"desk": asset}, condition(), layout([pred]))[0]
        transform = out["asset_transform"]
        # Raw pivot (.5, 1, 1.5) maps to canonical origin, hence requested bottom center.
        pivot = (.5, 1, 1.5, 1)
        position = [sum(transform[i][j] * pivot[j] for j in range(4)) for i in range(3)]
        for actual, expected in zip(position, (5, 6, 7)):
            self.assertAlmostEqual(actual, expected)

    def test_capability_category_and_fixed_size_are_hard_filters(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                          "required_capabilities": ["work_surface"], "fixed_size_local_m": [1, 1, .75]}])
        assets = (desk("missing"), desk("wrong_size", (2, 1, .75), capabilities=("work_surface",)),
                  Asset("wrong_category", "table", (1, 1, .75), capabilities=("work_surface",)))
        result = run_pipeline(req, layout(), CatalogResolver(assets))
        self.assertFalse(result["ok"])
        self.assertEqual(result["diagnostics"][0]["code"], "asset_unavailable")

    def test_asset_retry_is_bounded_and_uses_actual_geometry(self):
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [0.4, 2, 0]}])
        assets = (desk("bad", (.9, 1, .75)), desk("good", (.7, 1, .75)))
        result = run_pipeline(condition(), pred, CatalogResolver(assets), budget=RuntimeBudget(max_asset_retries=1))
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_objects"][0]["asset_ref"], "good")
        self.assertFalse(result["validation"]["actual_first_pass"]["ok"])
        self.assertEqual(result["metrics"]["system"]["asset_retries"], 1)

    def test_asset_retry_budget_zero_fails_without_partial_commit(self):
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [0.4, 2, 0]}])
        host = AtomicMemoryHost()
        result = run_pipeline(condition(), pred, CatalogResolver((desk("bad", (.9, 1, .75)),
                              desk("good", (.7, 1, .75)))), host=host,
                              budget=RuntimeBudget(max_asset_retries=0), idempotency_key="r1",
                              expected_world_version=0)
        self.assertFalse(result["ok"])
        self.assertEqual(host.snapshot(), {"version": 0, "objects": [], "commits": 0})

    def test_missing_actual_support_evidence_cannot_pass_as_bbox_top(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp", "support_parent": "desk"}])
        pred = layout(layout()["objects"] + [{"id": "lamp", "target_size_local_m": [.2, .2, .4],
                                              "bottom_center_m": [2, 2, .75], "yaw_rad": 0}])
        result = run_pipeline(req, pred, CatalogResolver((desk(), Asset("lamp", "lamp", (.2, .2, .4)))))
        self.assertFalse(result["ok"])
        self.assertIn("support_surface_unknown", {c["code"] for c in result["validation"]["final"]["checks"]})

    def test_verified_support_surface_moves_child_with_actual_height(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp", "support_parent": "desk"}])
        pred = layout(layout()["objects"] + [{"id": "lamp", "target_size_local_m": [.2, .2, .4],
                                              "bottom_center_m": [2, 2, .75], "yaw_rad": 0}])
        surface = SupportSurface("top", ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), .7)
        assets = (desk(size=(1, 1, .7), support_surfaces=(surface,)), Asset("lamp", "lamp", (.2, .2, .4)))
        result = run_pipeline(req, pred, CatalogResolver(assets))
        self.assertTrue(result["ok"])
        self.assertEqual(result["raw_model_output"]["objects"][1]["bottom_center_m"][2], .75)
        self.assertEqual(result["final_objects"][1]["bottom_center_m"][2], .7)

    def test_missing_support_retries_parent_asset_before_child(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp", "support_parent": "desk"}])
        pred = layout(layout()["objects"] + [{"id": "lamp", "target_size_local_m": [.2, .2, .4],
                                              "bottom_center_m": [2, 2, .75], "yaw_rad": 0}])
        surface = SupportSurface("top", ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), .7)
        assets = (desk("missing"), desk("verified", (1, 1, .7), support_surfaces=(surface,)),
                  Asset("lamp", "lamp", (.2, .2, .4)))
        result = run_pipeline(req, pred, CatalogResolver(assets), budget=RuntimeBudget(max_asset_retries=1))
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_objects"][0]["asset_ref"], "verified")
        self.assertEqual(result["attempts"][-1]["object_id"], "desk")

    def test_partial_fixed_size_and_log_tolerance_are_hard_filters(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor",
                          "fixed_size_local_m": [1, None, None], "retrieval_tolerance_log": .1}])
        result = run_pipeline(req, layout(), CatalogResolver((desk("bad", (.9, 1, .75)),
                              desk("good", (1, .95, .75)))))
        self.assertTrue(result["ok"])
        self.assertEqual(result["final_objects"][0]["asset_ref"], "good")

    def test_required_mesh_check_blocks_offline_commit(self):
        host = AtomicMemoryHost()
        result = run_pipeline(condition(), layout(), CatalogResolver((desk(),)), host=host,
                              idempotency_key="r", expected_world_version=0, required_levels=("bbox", "mesh"))
        self.assertFalse(result["ok"])
        self.assertEqual(host.snapshot()["commits"], 0)

    def test_repair_cannot_shrink_objects_or_change_ids(self):
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]}])
        def shrink(_condition, objects, _validation):
            return [{**objects[0], "actual_size_local_m": [.1, .1, .1]}]
        host = AtomicMemoryHost()
        result = run_pipeline(condition(), pred, CatalogResolver((desk(),)), repair=shrink, host=host,
                              budget=RuntimeBudget(max_asset_retries=0, max_repair_calls=1),
                              idempotency_key="r", expected_world_version=0)
        self.assertFalse(result["ok"])
        self.assertIn("invalid_repair", {d["code"] for d in result["diagnostics"]})
        self.assertEqual(host.snapshot()["commits"], 0)

    def test_pose_only_repair_is_validated_and_target_output_preserved(self):
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]}])
        def move(_condition, objects, _validation):
            return [{**objects[0], "bottom_center_m": [2, 2, 0]}]
        result = run_pipeline(condition(), pred, CatalogResolver((desk(),)), repair=move,
                              budget=RuntimeBudget(max_asset_retries=0, max_repair_calls=1))
        self.assertTrue(result["ok"])
        self.assertEqual(result["raw_model_output"]["objects"][0]["bottom_center_m"], [.2, 2, 0])
        self.assertEqual(result["metrics"]["system"]["repair_calls"], 1)

    def test_builtin_translation_repair_preserves_support_children(self):
        req = condition([{"id": "desk", "category": "desk", "support_parent": "floor"},
                         {"id": "lamp", "category": "lamp", "support_parent": "desk"}])
        pred = layout([{**layout()["objects"][0], "bottom_center_m": [.2, 2, 0]},
                       {"id": "lamp", "target_size_local_m": [.2, .2, .4],
                        "bottom_center_m": [.2, 2, .75], "yaw_rad": 0}])
        surface = SupportSurface("top", ((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), .75)
        assets = (desk(support_surfaces=(surface,)), Asset("lamp", "lamp", (.2, .2, .4)))
        result = run_pipeline(req, pred, CatalogResolver(assets), repair=BoundedTranslationRepair(.4),
                              budget=RuntimeBudget(max_asset_retries=0, max_repair_calls=1))
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["final_objects"][0]["bottom_center_m"][0], .6)
        self.assertAlmostEqual(result["final_objects"][1]["bottom_center_m"][0], .6)
        self.assertEqual(result["final_objects"][0]["target_bottom_center_m"], [.2, 2, 0])

    def test_commit_is_idempotent_and_checks_world_version(self):
        host = AtomicMemoryHost()
        resolver = CatalogResolver((desk(),))
        a = run_pipeline(condition(), layout(), resolver, host=host, idempotency_key="a", expected_world_version=0)
        b = run_pipeline(condition(), layout(), resolver, host=host, idempotency_key="a", expected_world_version=0)
        self.assertTrue(a["committed"] and b["committed"])
        self.assertEqual(host.snapshot()["commits"], 1)
        c = run_pipeline(condition(), layout(), resolver, host=host, idempotency_key="c", expected_world_version=0)
        self.assertFalse(c["committed"])
        self.assertEqual(host.snapshot()["commits"], 1)

    def test_duplicate_ids_fail_before_asset_resolution(self):
        result = run_pipeline(condition(), layout(layout()["objects"] * 2), CatalogResolver((desk(),)))
        self.assertFalse(result["ok"])
        self.assertEqual(result["metrics"]["asset"]["resolver_calls"], 0)

    def test_face_constraint_requires_actual_semantic_front(self):
        req = condition(constraints=[{"type": "faces_direction", "object_id": "desk", "direction_xy": [1, 0]}])
        result = run_pipeline(req, layout(), CatalogResolver((desk(),)))
        self.assertFalse(result["ok"])
        known = run_pipeline(req, layout(), CatalogResolver((desk(semantic_front_local=(1, 0, 0)),)))
        self.assertTrue(known["ok"])

    def test_elapsed_budget_is_checked_before_host_commit(self):
        result = run_pipeline(condition(), layout(), CatalogResolver((desk(),)),
                              budget=RuntimeBudget(max_seconds=0))
        self.assertFalse(result["ok"])
        self.assertEqual(result["diagnostics"][0]["code"], "budget_exhausted")

    def test_host_rejects_idempotency_key_reuse_for_different_payload(self):
        host = AtomicMemoryHost()
        first = [{"id": "desk", "bottom_center_m": [2, 2, 0]}]
        host.commit(first, idempotency_key="a", expected_world_version=0)
        with self.assertRaises(ValueError):
            host.commit([{**first[0], "bottom_center_m": [3, 2, 0]}], idempotency_key="a", expected_world_version=1)

    def test_host_rejects_duplicate_world_ids_atomically(self):
        host = AtomicMemoryHost()
        objects = [{"id": "desk", "bottom_center_m": [2, 2, 0]}]
        host.commit(objects, idempotency_key="a", expected_world_version=0)
        before = host.snapshot()
        with self.assertRaises(ValueError):
            host.commit(objects, idempotency_key="b", expected_world_version=1)
        self.assertEqual(host.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
