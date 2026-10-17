"""Evidence-preserving regressions for the multi-source minimal task."""
from copy import deepcopy
import json
import math

import pytest

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.direct_layout import request_to_condition
from fastfill.v2.io import fingerprint
from fastfill.v2.multisource_data import SOURCES, build_dataset, project_sample
from fastfill.v2.schema import migrate_legacy_row
from fastfill.v2.tests.test_minimal_data import pair


def source_pair(source="HSSD200", split="train", uid=None):
    sample, raw = pair(uid=uid or source + split, split=split, floor=0.)
    sample = migrate_legacy_row(sample)  # C1: groups live in validity, never in condition objects
    sample["provenance"]["source"] = raw["source"] = source
    if source == "MultiScan":
        raw["meta"] = {"floor_z": 0., "floor_height": 0., "n_floor_objects_outside_structure": 0}
    return sample, raw


@pytest.mark.parametrize("source", SOURCES)
def test_all_original_sources_keep_partial_supervision(source):
    split = "test" if source in {"SceneSmith", "SpatialGen"} else "train"
    sample, raw = source_pair(source, split)
    before = deepcopy(sample)
    sample["validity"]["size"] = [[False, False, False]]
    result = project_sample(sample, raw, split=split)
    assert result["validity"]["size"] == [[False, False, False]]
    assert result["validity"]["position"] == [[source != "Scan2CAD"] * 3]
    assert result["validity"]["yaw"] == [source == "SpatialLM"]
    assert result["provenance"]["source"] == source
    assert result["provenance"]["target_source_ids"] == ["source_0"]
    assert result["target"]["objects"][0]["bottom_center_m"] == [1., 1., .5]
    assert result["condition"] == request_to_condition(result["provenance"]["minimal_request"], room_size_semantics="reference_extent")
    assert result["provenance"]["parent_condition"] == before["condition"]
    assert sample["target"] == before["target"]


def test_unknown_types_nonrectangles_and_negative_z_are_preserved():
    sample, raw = source_pair("InteriorGS")
    sample["condition"]["room"].pop("room_type")
    raw["room_type"] = None
    polygon = [[10., 20.], [14., 20.], [10., 23.]]
    sample["condition"]["room"]["floor_polygon_xy_m"] = raw["boundary"] = polygon
    sample["target"]["objects"][0]["bottom_center_m"][2] = raw["objects"][0]["pos"][2] = -.25
    result = project_sample(sample, raw, split="train")
    assert result["provenance"]["minimal_request"]["room_type"] == "unknown"
    assert result["provenance"]["minimal_request"]["room_size_m"] == [4., 3.]
    assert result["target"]["objects"][0]["bottom_center_m"][2] == -.25
    assert result["validity"]["position"] == [[True] * 3]
    assert result["condition"]["room"]["boundary_known"] is False
    assert result["condition"]["constraints"] == []
    assert "fixed_objects" not in result["condition"]["room"]
    assert set(result["condition"]["objects"][0]) == {"id", "category", "description"}


def test_size_range_masks_whole_vector_without_fixing_geometry():
    sample, raw = source_pair()
    size = [1., 1e5, 1.]  # above the default range; a wider head range admits it (sub-3mm axes never do)
    sample["target"]["objects"][0]["target_size_local_m"] = raw["objects"][0]["size"] = size
    result = project_sample(sample, raw, split="train")
    assert result["validity"]["size"] == [[False] * 3]
    assert result["target"]["objects"][0]["target_size_local_m"] == size
    assert result["validity"]["position"] == [[True] * 3]
    change = next(c for c in result["provenance"]["qualification_changes"] if c["field"] == "size")
    assert change["before"] == [True] * 3 and change["value_m"] == size
    wider = project_sample(sample, raw, split="train", model_config={"size_log_limit": 25})
    assert wider["validity"]["size"] == [[True] * 3]


def test_scan2cad_position_quarantine_reports_missing_raw_residuals():
    sample, raw = source_pair("Scan2CAD")
    result = project_sample(sample, raw, split="train")
    assert result["validity"]["position"] == [[False] * 3]
    assert result["provenance"]["source_floor_evidence"]["floor_source"] == "estimated"
    assert result["provenance"]["source_floor_evidence"]["per_object_pre_snap_z"] == "unavailable_in_frozen_IR"
    assert result["target"]["objects"][0]["bottom_center_m"] == [1., 1., .5]


def test_multiscan_vertical_reference_and_semantic_yaw_retained_as_box_axis():
    sample, raw = source_pair("MultiScan")
    sample["provenance"]["vertical_reframe_m"] = -.1
    raw["meta"] = {"floor_z": -.2, "floor_height": -.1, "n_floor_objects_outside_structure": 0}
    raw["objects"][0]["pos"][2] = .6
    sample["condition"]["room"].update(floor_known=False, floor_z_m=None)
    sample["validity"]["yaw"] = [True]
    result = project_sample(sample, raw, split="train")
    assert result["target"]["objects"][0]["bottom_center_m"][2] == .5
    assert result["validity"]["yaw"] == [True]
    assert result["validity"]["yaw_symmetry_order"] == [2]
    assert result["provenance"]["frame_translation_m"] == [-10., -20., 0.]


def test_null_untrusted_coordinates_stay_null_and_collate():
    sample, raw = source_pair()
    sample["validity"]["position"] = [[False] * 3]
    sample["target"]["objects"][0]["bottom_center_m"][2] = raw["objects"][0]["pos"][2] = None
    result = project_sample(sample, raw, split="train")
    assert result["target"]["objects"][0]["bottom_center_m"] == [1., 1., None]
    batch = collate_samples([result], TinyTokenizer())
    assert not batch["validity"]["position"].any()
    sample["validity"]["position"] = [[True] * 3]
    with pytest.raises(ValueError, match="valid position"):
        project_sample(sample, raw, split="train")


@pytest.mark.parametrize("field", ["size", "pos", "yaw"])
def test_any_source_geometry_mismatch_fails_loudly(field):
    sample, raw = source_pair()
    raw["objects"][0][field] = [1., 2., 3.] if field != "yaw" else .3
    with pytest.raises(ValueError, match="source_geometry_mismatch"):
        project_sample(sample, raw, split="train")


def test_test_only_sources_and_invented_source_are_never_train():
    for source in ("SceneSmith", "SpatialGen", "UnknownSource"):
        sample, raw = source_pair(source)
        with pytest.raises(ValueError, match="source.*(role|taxonomy)"):
            project_sample(sample, raw, split="train")


def test_full_inventory_over_training_budget_is_never_truncated():
    sample, raw = source_pair()
    request, target, obj, evidence = [deepcopy(v) for v in (
        sample["condition"]["objects"][0], sample["target"]["objects"][0], raw["objects"][0], sample["provenance"]["field_evidence"][0])]
    count = 623
    sample["condition"]["objects"] = [{**request, "id": f"obj_{i:04d}"} for i in range(count)]
    sample["condition"]["constraints"] = []
    sample["target"]["objects"] = [{**target, "id": f"obj_{i:04d}"} for i in range(count)]
    raw["objects"] = [{**obj, "id": f"source_{i}"} for i in range(count)]
    sample["provenance"]["target_source_ids"] = [o["id"] for o in raw["objects"]]
    sample["provenance"]["field_evidence"] = [evidence for _ in range(count)]
    sample["validity"] = {"position": [[True] * 3 for _ in range(count)], "size": [[True] * 3 for _ in range(count)], "yaw": [False] * count}
    result = project_sample(sample, raw, split="train")
    assert len(result["condition"]["objects"]) == count
    batch = collate_samples([result], TinyTokenizer(), max_objects=count, max_length=100000)
    assert batch["slot_mask"].sum() == count


def dataset_files(tmp_path, duplicate_group=False):
    parent, ir = tmp_path / "parent", tmp_path / "ir"
    parent.mkdir(); ir.mkdir()
    raw_sources = {name: [] for name in SOURCES}
    for split in ("train", "validation", "test"):
        source = "SpatialLM" if split == "train" else "SceneSmith" if split == "test" else "HSSD200"
        sample, raw = source_pair(source, split, uid=split)
        if duplicate_group:
            sample["provenance"]["house_id"] = sample["provenance"]["group"] = raw["group"] = "samehouse"
        (parent / (split + ".jsonl")).write_text(json.dumps(sample) + "\n")
        raw_sources[source].append(raw)
    for name, rows in raw_sources.items():
        (ir / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (parent / "manifest.json").write_text('{}\n')
    (tmp_path / "frozen-manifest.json").write_text(json.dumps({"ir_sha256": {name + ".jsonl": fingerprint(ir / (name + ".jsonl")) for name in SOURCES}}))
    return parent, ir


def test_build_is_immutable_reproducible_hash_pinned_and_preserves_splits(tmp_path):
    parent, ir = dataset_files(tmp_path)
    before = {str(p): p.read_bytes() for root in (parent, ir) for p in root.iterdir()}
    kwargs = {"expected_spatiallm_sha256": fingerprint(ir / "SpatialLM.jsonl"), "frozen_manifest": tmp_path / "frozen-manifest.json"}
    m1 = build_dataset(parent, ir, tmp_path / "new1", **kwargs)
    m2 = build_dataset(parent, ir, tmp_path / "new2", **kwargs)
    assert m1["split_samples"] == {"train": 1, "validation": 1, "test": 1}
    assert m1["source_split_samples"] == {"SpatialLM:train": 1, "HSSD200:validation": 1, "SceneSmith:test": 1}
    assert m1["split_integrity"]["group_cross_split_conflicts"] == 0
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "changes.jsonl", "exclusions.jsonl"):
        assert (tmp_path / "new1" / name).read_bytes() == (tmp_path / "new2" / name).read_bytes()
    assert before == {str(p): p.read_bytes() for root in (parent, ir) for p in root.iterdir()}
    with pytest.raises(FileExistsError):
        build_dataset(parent, ir, tmp_path / "new1", **kwargs)
    with pytest.raises(ValueError, match="outside"):
        build_dataset(parent, ir, parent / "unsafe", **kwargs)
    with pytest.raises(ValueError, match="SHA256"):
        build_dataset(parent, ir, tmp_path / "badpin", frozen_manifest=tmp_path / "frozen-manifest.json")


def test_parent_group_leakage_strict_json_and_ir_duplication_fail_before_output(tmp_path):
    parent, ir = dataset_files(tmp_path, duplicate_group=True)
    kwargs = {"expected_spatiallm_sha256": fingerprint(ir / "SpatialLM.jsonl"), "frozen_manifest": tmp_path / "frozen-manifest.json"}
    with pytest.raises(ValueError, match="cross-split"):
        build_dataset(parent, ir, tmp_path / "groupbad", **kwargs)
    assert not (tmp_path / "groupbad").exists()
    (parent / "train.jsonl").write_text('{"schema_version":"fastfill.v2","schema_version":"bad"}\n')
    with pytest.raises(ValueError, match="duplicate JSON"):
        build_dataset(parent, ir, tmp_path / "jsonbad", **kwargs)


def test_all_ir_sources_require_canonical_hash_proof(tmp_path):
    parent, ir = dataset_files(tmp_path)
    kwargs = {"expected_spatiallm_sha256": fingerprint(ir / "SpatialLM.jsonl"), "frozen_manifest": tmp_path / "frozen-manifest.json"}
    (ir / "HSSD200.jsonl").write_text((ir / "HSSD200.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="frozen IR SHA256"):
        build_dataset(parent, ir, tmp_path / "tampered", **kwargs)
    assert not (tmp_path / "tampered").exists()
    with pytest.raises(ValueError, match="frozen manifest"):
        build_dataset(parent, ir, tmp_path / "unbound", expected_spatiallm_sha256=kwargs["expected_spatiallm_sha256"])


def test_numeric_qualification_helper_preserves_full_task_semantic_yaw():
    from fastfill.v2.multisource_data import qualify_parent_masks
    sample, raw = source_pair("SpatialLM")
    original = deepcopy(sample)
    masks, changes = qualify_parent_masks(sample)
    assert masks == sample["validity"]
    assert masks["yaw"] == [False]
    assert changes == []
    assert sample == original


@pytest.mark.parametrize('config', [{'size_log_limit': 31}, {'size_reference': [1e-40, 1., 1.]}, {'size_reference': [1e38, 1., 1.]}])
def test_size_range_matches_float32_model_contract(config):
    sample, raw = source_pair()
    with pytest.raises(ValueError, match='size.*range|float32'):
        project_sample(sample, raw, split='train', model_config=config)
