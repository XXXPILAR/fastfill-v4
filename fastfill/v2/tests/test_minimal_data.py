"""Regression tests for minimal-condition data, not legacy semantic-front labels."""
from copy import deepcopy
import json
import math

import pytest

from fastfill.v2.direct_layout import request_to_condition
from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.io import fingerprint
from fastfill.v2.minimal_data import build_minimal_dataset, project_sample


def pair(uid="room", split="train", floor=1., height=3.):
    target = {"id": "obj_0000", "target_size_local_m": [1., .5, 1.],
              "bottom_center_m": [11., 21., floor + .5], "yaw_rad": 0.}
    room = {"frame": "right_handed_z_up", "room_type": "office", "boundary_known": True,
            "floor_known": True, "floor_z_m": floor, "height_m": height,
            "floor_polygon_xy_m": [[10., 20.], [14., 20.], [14., 23.], [10., 23.]],
            "fixed_objects": [{"id": "fixed_0", "category": "window", "size_local_m": [1., .1, 1.],
                               "bottom_center_m": [10., 21., floor + 1.], "yaw_rad": 0.}]}
    condition = {"schema_version": "fastfill.v2", "room": room,
                 "objects": [{"id": "obj_0000", "category": "chair", "description": "chair",
                              "support_parent": "floor", "exchangeable_group": "anonymous_0"}],
                 "constraints": [{"type": "against_wall", "object_id": "obj_0000"}]}
    provenance = {"source": "SpatialLM", "scene_id": uid, "legacy_uid": uid,
                  "house_id": "house_" + uid, "group": "house_" + uid, "split": split,
                  "target_source_ids": ["source_0"], "field_evidence": [{"tilted": False,
                     "size_semantics": "canonical_source_IR", "front_policy": "strict"}]}
    sample = {"schema_version": "fastfill.v2", "condition": condition,
              "target": {"schema_version": "fastfill.v2", "objects": [target]},
              "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [False]},
              "provenance": provenance}
    source = {"uid": uid, "source": "SpatialLM", "objects": [{"id": "source_0", "size": [1., .5, 1.],
                       "pos": [11., 21., floor + .5], "yaw": 0., "tilted": False}],
              "boundary": room["floor_polygon_xy_m"], "group": provenance["group"],
              "room_type": "office", "height": height, "floor_z_m": floor, "boundary_type": "polygon"}
    return sample, source


def test_projection_uses_same_public_adapter_and_preserves_lineage():
    sample, source = pair()
    before = deepcopy(sample)
    result = project_sample(sample, source, split="train")
    request = {"room_type": "office", "room_size_m": [4., 3.],
               "furniture_list": [{"id": "obj_0000", "category": "chair", "description": "chair", "count": 1}]}
    assert result["condition"] == request_to_condition(request)
    assert result["provenance"]["minimal_request"] == request
    assert result["target"]["objects"][0]["bottom_center_m"] == [1., 1., .5]
    assert result["target"]["objects"][0]["target_size_local_m"] == [1., .5, 1.]
    assert result["provenance"]["frame_translation_m"] == [-10., -20., -1.]
    for key in ("scene_id", "legacy_uid", "house_id", "group", "split", "target_source_ids"):
        assert result["provenance"][key] == sample["provenance"][key]
    assert sample == before
    batch = collate_samples([result], TinyTokenizer())
    assert batch["scale"][0, 2].item() == 3.
    assert result["condition"]["room"]["height_m"] is None
    assert result["provenance"]["qualification_source_height_m"] == 3.


def test_xyz_height_is_explicit_opt_in_not_default():
    sample, source = pair(height=2.5)
    xy = project_sample(sample, source, split="train")
    xyz = project_sample(sample, source, split="train", room_dimension_mode="xyz")
    assert xy["provenance"]["minimal_request"]["room_size_m"] == [4., 3.]
    assert xyz["provenance"]["minimal_request"]["room_size_m"] == [4., 3., 2.5]
    assert xyz["condition"]["room"]["height_m"] == 2.5
    assert xy["target"] == xyz["target"]
    assert collate_samples([xy], TinyTokenizer())["scale"][0, 2].item() == 3.
    assert collate_samples([xyz], TinyTokenizer())["scale"][0, 2].item() == 2.5
    with pytest.raises(ValueError, match="room_dimension_mode"):
        project_sample(sample, source, split="train", room_dimension_mode="invalid")


def test_unknown_height_stays_unknown_and_labels_are_not_input():
    sample, source = pair(height=None)
    result = project_sample(sample, source, split="train")
    assert result["provenance"]["minimal_request"]["room_size_m"] == [4., 3.]
    assert result["condition"]["room"]["height_m"] is None
    assert "fixed_objects" not in result["condition"]["room"]
    assert result["condition"]["constraints"] == []
    assert set(result["condition"]["objects"][0]) == {"id", "category", "description"}
    assert "target_size_local_m" not in json.dumps(result["condition"])
    xyz = project_sample(sample, source, split="train", room_dimension_mode="xyz")
    assert xyz["provenance"]["minimal_request"]["room_size_m"] == [4., 3.]
    assert xyz["condition"]["room"]["height_m"] is None


def test_geometric_yaw_does_not_certify_semantic_front_or_snap_position():
    sample, source = pair(floor=0.)
    sample["target"]["objects"][0]["bottom_center_m"][2] = -.0005
    source["objects"][0]["pos"][2] = -.0005
    result = project_sample(sample, source, split="train")
    assert result["validity"]["yaw"] == [True]
    assert result["provenance"]["yaw_label_semantics"] == "local_bbox_axes_not_certified_semantic_front"
    assert result["provenance"]["parent_yaw_validity"] == [False]
    assert result["target"]["objects"][0]["bottom_center_m"][2] == -.0005
    assert collate_samples([result], TinyTokenizer())["yaw_symmetry_order"].tolist() == [[2]]


def test_object_budget_rejects_whole_scene_without_partial_projection():
    sample, source = pair()
    before = deepcopy(sample)
    with pytest.raises(ValueError, match="max_objects"):
        project_sample(sample, source, split="train", max_objects=0)
    assert sample == before
    second = {**deepcopy(sample["condition"]["objects"][0]), "id": "obj_0001"}
    sample["condition"]["objects"] = sample["condition"]["objects"] + [second]
    with pytest.raises(ValueError, match="max_objects"):
        project_sample(sample, source, split="train", max_objects=1)


def test_source_split_and_nonfinite_yaw_rejected():
    sample, source = pair()
    with pytest.raises(ValueError, match="split"):
        project_sample(sample, source, split="test")
    sample["target"]["objects"][0]["yaw_rad"] = float("nan")
    with pytest.raises(ValueError, match="labels"):
        project_sample(sample, source, split="train")


@pytest.mark.parametrize("field,value,reason", [
    ("room_type", "kitchen", "source_room_type"),
    ("height", 4., "source_room_height"),
    ("floor_z_m", 2., "source_floor_frame"),
    ("boundary_type", "hull", "source_boundary_semantics"),
])
def test_source_room_metadata_must_agree_with_parent(field, value, reason):
    sample, source = pair()
    source[field] = value
    with pytest.raises(ValueError, match=reason):
        project_sample(sample, source, split="train")


def test_parent_cannot_invent_height_or_hide_source_height_without_evidence():
    sample, source = pair(height=None)
    sample["condition"]["room"]["height_m"] = 3.
    with pytest.raises(ValueError, match="source_room_height"):
        project_sample(sample, source, split="train")
    sample, source = pair(height=3.)
    sample["condition"]["room"]["height_m"] = None
    with pytest.raises(ValueError, match="source_room_height"):
        project_sample(sample, source, split="train")
    sample["provenance"]["legacy_height_dropped"] = True
    result = project_sample(sample, source, split="train")
    assert result["condition"]["room"]["height_m"] is None
    assert result["provenance"]["qualification_source_height_m"] == 3.


def test_recorded_parent_height_drop_cannot_bypass_source_ceiling_qualification():
    sample, source = pair(floor=0., height=2.)
    sample["condition"]["room"]["height_m"] = None
    sample["provenance"]["legacy_height_dropped"] = True
    sample["target"]["objects"][0]["bottom_center_m"][2] = 1.5
    source["objects"][0]["pos"][2] = 1.5
    with pytest.raises(ValueError, match="ceiling"):
        project_sample(sample, source, split="train")


def test_spatiallm_source_has_declared_canonical_zero_floor_by_default():
    sample, source = pair(floor=0.)
    del source["floor_z_m"]
    assert project_sample(sample, source, split="train")["condition"]["room"]["floor_z_m"] == 0.
    sample["condition"]["room"]["floor_z_m"] = .1
    with pytest.raises(ValueError, match="source_floor_frame"):
        project_sample(sample, source, split="train")


@pytest.mark.parametrize("change,reason", [
    (lambda s, ir: s["provenance"].update(source="HSSD200"), "source"),
    (lambda s, ir: s["condition"]["room"].update(boundary_known=False), "boundary"),
    (lambda s, ir: s["condition"]["room"].update(floor_known=False), "floor"),
    (lambda s, ir: s["condition"]["room"].update(room_type="unknown"), "room_type"),
    (lambda s, ir: s["condition"]["room"].update(floor_polygon_xy_m=[[10., 20.], [14., 20.], [10., 23.]]), "rectangle"),
    (lambda s, ir: s["validity"]["size"][0].__setitem__(2, False), "labels"),
    (lambda s, ir: s["provenance"]["field_evidence"][0].update(tilted=True), "tilted"),
    (lambda s, ir: ir["objects"][0].update(tilted=True), "tilted"),
    (lambda s, ir: ir["objects"][0].update(yaw=.5), "source_geometry"),
    (lambda s, ir: ir.update(uid="other"), "source_identity"),
])
def test_ineligible_geometry_fails_closed(change, reason):
    sample, source = pair()
    change(sample, source)
    with pytest.raises(ValueError, match=reason):
        project_sample(sample, source, split="train")


def test_obb_corners_and_own_height_determine_qualification():
    sample, source = pair()
    sample["target"]["objects"][0]["bottom_center_m"][0] = 10.25
    source["objects"][0]["pos"][0] = 10.25
    with pytest.raises(ValueError, match="xy_boundary"):
        project_sample(sample, source, split="train")
    sample, source = pair(height=1.)
    with pytest.raises(ValueError, match="ceiling"):
        project_sample(sample, source, split="train")
    sample, source = pair()
    with pytest.raises(ValueError, match="max_objects"):
        project_sample(sample, source, split="train", max_objects=0)


def parent_files(tmp_path, duplicate_group=False):
    parent, source_path = tmp_path / "parent", tmp_path / "source.jsonl"
    parent.mkdir()
    sources = []
    for split in ("train", "validation", "test"):
        row, source = pair(uid=split, split=split)
        if duplicate_group:
            row["provenance"]["house_id"] = row["provenance"]["group"] = "same_house"
            source["group"] = "same_house"
        (parent / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
        sources.append(source)
    (parent / "manifest.json").write_text(json.dumps({"schema_version": "fastfill.v2"}))
    source_path.write_text("".join(json.dumps(r) + "\n" for r in sources))
    return parent, source_path


def test_build_inherits_split_and_does_not_change_parent_bytes(tmp_path):
    parent, source = parent_files(tmp_path)
    before = {p.name: p.read_bytes() for p in parent.iterdir()}
    out = tmp_path / "new-version"
    manifest = build_minimal_dataset(parent, source, out, expected_source_sha256=fingerprint(source))
    assert manifest["split_samples"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["split_objects"] == {"train": 1, "validation": 1, "test": 1}
    assert manifest["room_dimension_mode"] == "xy"
    assert manifest["split_integrity"]["group_cross_split_conflicts"] == 0
    for split in ("train", "validation", "test"):
        row = json.loads((out / (split + ".jsonl")).read_text())
        assert row["provenance"]["scene_id"] == row["provenance"]["split"] == split
    assert before == {p.name: p.read_bytes() for p in parent.iterdir()}
    with pytest.raises(FileExistsError):
        build_minimal_dataset(parent, source, out, expected_source_sha256=fingerprint(source))


def test_build_rejects_source_hash_and_cross_split_group(tmp_path):
    parent, source = parent_files(tmp_path, duplicate_group=True)
    with pytest.raises(ValueError, match="SHA256"):
        build_minimal_dataset(parent, source, tmp_path / "bad-hash", expected_source_sha256="0" * 64)
    assert not (tmp_path / "bad-hash").exists()
    with pytest.raises(ValueError, match="cross.split"):
        build_minimal_dataset(parent, source, tmp_path / "bad-split", expected_source_sha256=fingerprint(source))
