import copy

import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples, condition_segments


def sample(n=2):
    objects = [{"id": f"chair_{i}", "category": "chair", "description": "plain chair"} for i in range(n)]
    return {
        "schema_version": "fastfill.v2",
        "condition": {"schema_version": "fastfill.v2", "room": {
            "frame": "right_handed_z_up", "floor_polygon_xy_m": [[2, 4], [7, 4], [7, 8], [2, 8]],
            "floor_z_m": 1.0, "height_m": 3.0}, "objects": objects, "constraints": []},
        "target": {"objects": [{"id": o["id"], "target_size_local_m": [0.6, 0.5, 1.0],
            "bottom_center_m": [3 + i, 5, 1], "yaw_rad": 0.2} for i, o in enumerate(objects)]},
        "validity": {"position": [[True] * 3 for _ in objects], "size": [[True] * 3 for _ in objects],
            "yaw": [True for _ in objects]}, "provenance": {
                "source": "offline_test", "house_id": "offline_house", "split": "train"}}


def test_condition_memory_never_contains_targets():
    s = sample()
    other = copy.deepcopy(s)
    other["target"]["objects"][0]["bottom_center_m"] = [987, 654, 321]
    a = collate_samples([s], TinyTokenizer())
    b = collate_samples([other], TinyTokenizer())
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["origin"], torch.tensor([[2., 4., 1.]]))
    assert torch.equal(a["scale"], torch.tensor([[5., 4., 3.]]))
    assert torch.allclose(a["targets"]["position_normalized"][0, 0], torch.tensor([.2, .25, 0.]))


def test_condition_token_spans_preserve_object_and_constraints():
    s = sample(1)
    s["condition"]["constraints"] = [{"type": "faces_direction", "object_id": "chair_0", "direction_xy": [1, 0]}]
    tokenizer = TinyTokenizer()
    batch = collate_samples([s, sample(3)], tokenizer)
    start, end = batch["object_spans"][0, 0].tolist()
    text = tokenizer.decode(batch["input_ids"][0, start:end])
    assert "chair_0" in text and "plain chair" in text
    assert "faces_direction" in tokenizer.decode(batch["input_ids"][0])
    assert batch["slot_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert not batch["validity"]["position"][0, 1:].any()


def test_overlong_condition_is_rejected_whole():
    with pytest.raises(ValueError, match="context"):
        collate_samples([sample()], TinyTokenizer(), max_length=30)


def test_missing_validity_not_assumed_valid_and_unknown_stays_nan():
    s = sample(1)
    s["validity"] = {}
    s["target"]["objects"][0]["yaw_rad"] = None
    batch = collate_samples([s], TinyTokenizer())
    assert not batch["validity"]["size"].any()
    assert torch.isnan(batch["targets"]["yaw"][0, 0])


def test_fixed_geometry_and_floor_masks_are_explicit():
    s = sample(1)
    s["condition"]["objects"][0].update(support_parent="floor", fixed_size_local_m=[.8, .7, 1.2])
    s["target"]["objects"][0]["target_size_local_m"] = [.8, .7, 1.2]
    b = collate_samples([s], TinyTokenizer())
    assert b["fixed_position_mask"].tolist() == [[[False, False, True]]]
    assert b["fixed_size_mask"].all()
    assert torch.equal(b["fixed_size"][0, 0], torch.tensor([.8, .7, 1.2]))


def test_partial_fixed_size_and_condition_demo_conflicts():
    s = sample(1)
    s["condition"]["objects"][0]["fixed_size_local_m"] = [.6, None, 1.]
    b = collate_samples([s], TinyTokenizer())
    assert b["fixed_size_mask"].tolist() == [[[True, False, True]]]
    assert b["conditions"][0] == s["condition"]
    s["condition"]["objects"][0]["fixed_size_local_m"] = [.8, None, 1.]
    with pytest.raises(ValueError, match="conflicts"):
        collate_samples([s], TinyTokenizer())
    s = sample(1)
    s["condition"]["objects"][0]["support_parent"] = "floor"
    s["target"]["objects"][0]["bottom_center_m"][2] = 1.2
    with pytest.raises(ValueError, match="floor support.*conflicts"):
        collate_samples([s], TinyTokenizer())


def test_validity_arrays_follow_reordered_target_ids():
    s = sample(2)
    s["target"]["objects"].reverse()
    s["validity"]["position"] = [[False] * 3, [True] * 3]
    s["validity"]["yaw_symmetry_order"] = [2, 1]
    b = collate_samples([s], TinyTokenizer())
    assert b["validity"]["position"].tolist() == [[[True] * 3, [False] * 3]]
    assert b["yaw_symmetry_order"].tolist() == [[1, 2]]


def test_schema_rejects_asset_geometry_leakage():
    s = sample(1)
    s["condition"]["objects"][0]["asset_id"] = "source-model-id"
    with pytest.raises(ValueError, match="unknown fields"):
        collate_samples([s], TinyTokenizer())
