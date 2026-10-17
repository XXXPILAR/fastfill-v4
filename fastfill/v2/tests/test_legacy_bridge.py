"""Selected corpus migration preserves lineage without leaking geometry via IDs."""
from copy import deepcopy
from types import SimpleNamespace
import json
import tempfile
from pathlib import Path
import unittest

from fastfill.build import prep
from fastfill.scene import canonical, messages, rot90
from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.schema import validate_condition
from fastfill.v2.io import read_samples
from fastfill.v2.batch import collate_samples, TinyTokenizer


def fixture():
    return {"uid": "SpatialLM:fixture", "source": "SpatialLM", "group": "house-1",
            "room_type": "living room", "boundary_type": "polygon", "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]],
            "height": 3., "meta": {"front_known": True}, "objects": [
                {"id": "source-b", "category": "chair", "size": [1.1234, .8, .9], "pos": [4, 3, 0],
                 "yaw": .123456, "anchor": "floor", "parent": None, "tilted": False},
                {"id": "source-a", "category": "chair", "size": [.7, .6, .8], "pos": [1, 1, 0],
                 "yaw": 0., "anchor": "floor", "parent": None, "tilted": False}]}


def prepared(raw):
    args = SimpleNamespace(boundary_types=["polygon", "hull"], anchors=["floor", "object"],
                           source_anchors={}, min_objects=1, max_vertices=0, oob_tol=.1, hidden_max=.3, reject_flagged=[])
    result, reason = prep(deepcopy(raw), args)
    assert reason is None, reason
    return result


def saved(raw, p):
    c = canonical(p)
    return {"uid": raw["uid"], "source": raw["source"], "flags": {}, "messages": messages(c)}


class LegacyBridgeTests(unittest.TestCase):
    def test_migrated_sample_is_readable_and_batchable(self):
        raw = fixture(); p = prepared(raw)
        out = convert_selected_room(raw, p, saved(raw, p), "train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(json.dumps(out) + "\n")
            rows = read_samples(path, training=True)
            batch = collate_samples(rows, TinyTokenizer(), max_length=8192, max_objects=16)
            self.assertEqual(int(batch["slot_mask"].sum()), 2)

    def test_precise_targets_size_hidden_ids_and_split_preserved(self):
        raw = fixture(); p = prepared(raw)
        out = convert_selected_room(raw, p, saved(raw, p), "train")
        validate_condition(out["condition"])
        self.assertEqual(out["provenance"]["legacy_uid"], raw["uid"])
        self.assertEqual(out["provenance"]["split"], "train")
        self.assertTrue(all("size" not in x and "fixed_size_local_m" not in x for x in out["condition"]["objects"]))
        self.assertIn(1.1234, [o["target_size_local_m"][0] for o in out["target"]["objects"]])
        self.assertTrue(all(x["id"].startswith("obj_") for x in out["condition"]["objects"]))

    def test_request_condition_does_not_change_with_target_geometry_or_raw_order(self):
        a = fixture(); b = deepcopy(a)
        b["objects"] = [{**o, "size": [.9, .9, .9], "pos": [3-i, 2, 0], "yaw": .8}
                        for i, o in enumerate(reversed(b["objects"]))]
        pa, pb = prepared(a), prepared(b)
        oa = convert_selected_room(a, pa, saved(a, pa), "test")
        ob = convert_selected_room(b, pb, saved(b, pb), "test")
        self.assertEqual(oa["condition"], ob["condition"])

    def test_tilt_is_masked_not_silently_used_as_local_box(self):
        a = fixture(); a["objects"][0]["tilted"] = True
        p = prepared(a); out = convert_selected_room(a, p, saved(a, p), "train")
        source_ids = out["provenance"]["target_source_ids"]
        idx = source_ids.index("source-b")
        self.assertFalse(any(out["validity"]["size"][idx]))
        self.assertFalse(out["validity"]["yaw"][idx])

    def test_support_unknown_is_not_fabricated_as_floor_from_old_snap(self):
        a = fixture(); a["objects"][0]["pos"][2] = .2; a["objects"][0]["anchor"] = None
        p = prepared(a); out = convert_selected_room(a, p, saved(a, p), "train")
        idx = out["provenance"]["target_source_ids"].index("source-b")
        self.assertNotIn("support_parent", out["condition"]["objects"][idx])
        self.assertEqual(out["target"]["objects"][idx]["bottom_center_m"][2], .2)

    def test_old_rotated_constraint_ids_follow_source_identity(self):
        a = fixture(); p = prepared(a); c = canonical(rot90(p, 1))
        row = {"uid": a["uid"], "source": a["source"], "flags": {},
               "messages": messages(c, [["near", c["objects"][0]["id"], c["objects"][1]["id"], 3.]])}
        out = convert_selected_room(a, p, row, "train")
        ids = {o["id"] for o in out["condition"]["objects"]}
        self.assertEqual(out["condition"]["constraints"][0]["type"], "near")
        self.assertIn(out["condition"]["constraints"][0]["object_id"], ids)
        self.assertIn(out["condition"]["constraints"][0]["target_id"], ids)


if __name__ == "__main__":
    unittest.main()
