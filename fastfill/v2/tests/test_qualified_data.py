"""Main-task regression: qualification must preserve rich conditions and labels."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fastfill.v2.qualified_data import qualify_sample, build_dataset


def sample(source="SpatialLM", split="train"):
    return {"schema_version": "fastfill.v2", "condition": {
        "schema_version": "fastfill.v2", "room": {"frame": "right_handed_z_up",
        "floor_polygon_xy_m": [[0, 0], [5, 0], [5, 4], [0, 4]], "floor_z_m": 0.,
        "floor_known": True, "boundary_known": True, "height_m": 2.8,
        "fixed_objects": [{"id": "fixed", "category": "chair", "size_local_m": [.5, .5, 1.],
                           "bottom_center_m": [3., 3., 0.], "yaw_rad": 0.}]},
        "objects": [{"id": "desk", "category": "desk", "description": "desk", "support_parent": "floor"}],
        "constraints": [{"type": "faces_direction", "object_id": "desk", "direction_xy": [1., 0.]}]},
        "target": {"schema_version": "fastfill.v2", "objects": [{"id": "desk",
            "target_size_local_m": [1.4, .7, .75], "bottom_center_m": [2., 1., 0.], "yaw_rad": .3}]},
        "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [False], "size_axis_swap_allowed": [False]},
        "provenance": {"source": source, "scene_id": source + "::" + split,
            "split": split, "group": source + ":" + split, "house_id": source + ":" + split,
            "target_source_ids": ["raw0"], "source_meta": {"floor_z": -.2, "n_floor_snapped": 1}}}


def test_full_task_is_preserved_and_no_yaw_promotion():
    parent = sample()
    before = deepcopy(parent)
    row = qualify_sample(parent)
    assert parent == before
    assert row["condition"] == parent["condition"]
    assert row["target"] == parent["target"]
    assert row["validity"] == parent["validity"]
    assert row["provenance"]["qualification_changes"] == []


def test_scan2cad_demotes_whole_position_but_retains_xy_size_and_rich_input():
    parent = sample("Scan2CAD")
    row = qualify_sample(parent)
    assert parent["condition"]["room"]["floor_known"] is True
    expected = deepcopy(parent["condition"])
    expected["room"]["floor_known"] = False
    assert row["condition"] == expected
    assert row["target"] == parent["target"]
    assert row["validity"] == {"position": [[False]*3], "size": [[True]*3], "yaw": [False], "size_axis_swap_allowed": [False]}
    assert row["provenance"]["estimated_floor_provenance"]["per_object_snap_membership"].startswith("unknown")


@pytest.mark.parametrize("value", [3e-9, 1e8])
def test_range_masks_whole_size_without_clamping(value):
    parent = sample()
    parent["target"]["objects"][0]["target_size_local_m"][2] = value
    row = qualify_sample(parent)
    assert row["target"] == parent["target"]
    assert row["validity"]["size"] == [[False]*3]
    assert row["validity"]["position"] == [[True]*3]


def test_bad_masked_values_not_promoted():
    parent = sample()
    parent["target"]["objects"][0]["target_size_local_m"][2] = None
    parent["validity"]["size"][0][2] = False
    assert qualify_sample(parent)["validity"]["size"] == parent["validity"]["size"]


@pytest.mark.parametrize("source,split", [("SceneSmith", "train"), ("invented", "train")])
def test_unknown_or_heldout_training_source_fails(source, split):
    with pytest.raises(ValueError):
        qualify_sample(sample(source, split))


def test_duplicate_identity_and_boolean_geometry_fail():
    parent = sample()
    parent["target"]["objects"][0]["bottom_center_m"][0] = True
    with pytest.raises(ValueError):
        qualify_sample(parent)


def test_immutable_build_requires_pins_and_preserves_all_splits(tmp_path):
    from fastfill.v2.io import fingerprint
    parent = tmp_path / "parent"
    parent.mkdir()
    for split in ("train", "validation", "test"):
        (parent / (split + ".jsonl")).write_text(json.dumps(sample(split=split)) + "\n")
    (parent / "manifest.json").write_text('{"schema_version":"fastfill.v2"}\n')
    (parent / "rejections.jsonl").write_text("")
    pins = {p.name: fingerprint(p) for p in parent.iterdir()}
    before = dict(pins)
    output = tmp_path / "new"
    manifest = build_dataset(parent, output, expected_hashes=pins)
    assert manifest["split_samples"] == {"train": 1, "validation": 1, "test": 1}
    assert {p.name: fingerprint(p) for p in parent.iterdir()} == before
    assert manifest["validity_counts"]["yaw"] == 0
    with pytest.raises(FileExistsError):
        build_dataset(parent, output, expected_hashes=pins)
    with pytest.raises(ValueError):
        build_dataset(parent, parent / "child", expected_hashes=pins)
    (parent / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="pinned"):
        build_dataset(parent, tmp_path / "tampered", expected_hashes=pins)


def test_refuse_qualified_parent():
    with pytest.raises(ValueError):
        qualify_sample(qualify_sample(sample()))


def test_mask_review_reverts_entire_exchange_group_to_fixed_identity():
    parent = sample("Scan2CAD")
    obj = parent["condition"]["objects"][0]
    parent["condition"]["objects"].append({**obj, "id": "desk2"})
    parent["target"]["objects"].append({**deepcopy(parent["target"]["objects"][0]), "id": "desk2"})
    parent["provenance"]["target_source_ids"].append("raw1")
    for key in ("position", "size", "yaw", "size_axis_swap_allowed"):
        parent["validity"][key].append(deepcopy(parent["validity"][key][0]))
    parent["validity"]["exchangeable_group"] = ["identical-desks"] * 2
    row = qualify_sample(parent)
    assert row["validity"]["exchangeable_group"] == [None, None]
    assert all("exchangeable_group" not in request for request in row["condition"]["objects"])
    assert row["target"] == parent["target"]
    assert len([c for c in row["provenance"]["qualification_changes"] if c["field"] == "exchangeable_group"]) == 2
    parent["provenance"]["source"] = "SpatialLM"
    kept = qualify_sample(parent)
    assert kept["condition"] == parent["condition"]
    assert kept["validity"]["exchangeable_group"] == ["identical-desks"] * 2
