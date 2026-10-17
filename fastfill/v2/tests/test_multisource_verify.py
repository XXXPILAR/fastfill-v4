"""Independent verifier regressions; fixtures never call the data producer."""
from copy import deepcopy
import hashlib
import json
import math

import pytest

from fastfill.v2.multisource_verify import verify_pair, verify_dataset, verify_full_pair


def metadata(parent, derived):
    p, room = parent["provenance"], parent["condition"]["room"]
    estimated = p["source"] == "Scan2CAD"
    floor = room.get("floor_z_m")
    changes = []
    for i, obj in enumerate(parent["target"]["objects"]):
        base = {"object_id": obj["id"], "target_source_id": p["target_source_ids"][i]}
        for field, reason in (("position", "Scan2CAD_estimated_floor_and_upstream_snap_uncertainty"),
            ("size", "outside_configured_size_head_range"), ("yaw", "pinned_SpatialLM_full_local_extents_and_geometric_yaw")):
            old, new = parent["validity"][field][i], derived["validity"][field][i]
            if old != new:
                change = {**base, "field": field, "before": old, "after": new, "reason": reason}
                changes.append({**change, "value_m": obj["target_size_local_m"]} if field == "size" else change)
    changes.append({"field": "bottom_center_m", "operation": "common_input_reference_translation",
        "offset_m": derived["provenance"]["frame_translation_m"], "reason": "structural_input_extent_and_explicit_floor_reference"})
    derived["provenance"]["qualification_changes"] = changes
    derived["provenance"]["source_floor_evidence"] = {
        "floor_source": "estimated" if estimated else "unknown" if floor is None else "source_canonical_reference",
        "parent_floor_known": room.get("floor_known"), "parent_floor_z_m": floor,
        "upstream_z_snap_possible": estimated,
        "per_object_pre_snap_z": "unavailable_in_frozen_IR" if estimated else "not_inferred"}


def pair(source="HSSD200", split="train", uid="room", floor=0.):
    parent = {
        "schema_version": "fastfill.v2",
        "condition": {"schema_version": "fastfill.v2", "room": {
            "frame": "right_handed_z_up", "room_type": "office", "floor_z_m": floor,
            "floor_known": True, "boundary_known": True, "height_m": 3.,
            "floor_polygon_xy_m": [[10., 20.], [14., 20.], [11., 23.]],
            "fixed_objects": [{"id": "fixed_0"}]}, "objects": [
            {"id": "obj_0000", "category": "chair", "description": "chair", "support_parent": "floor"}],
            "constraints": [{"type": "against_wall", "object_id": "obj_0000"}]},
        "target": {"schema_version": "fastfill.v2", "objects": [
            {"id": "obj_0000", "bottom_center_m": [11., 21., .5 + floor],
             "target_size_local_m": [1., .5, 1.], "yaw_rad": 0.}]},
        "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [False], "size_axis_swap_allowed": [False]},
        "provenance": {"source": source, "scene_id": uid, "legacy_uid": uid,
            "house_id": "house_" + uid, "group": "house_" + uid, "split": split,
            "target_source_ids": ["raw_0"], "field_evidence": [
                {"tilted": False, "size_semantics": "canonical_source_IR"}]}}
    request = {"room_type": "office", "room_size_m": [4., 3.], "furniture_list": [
        {"id": "obj_0000", "category": "chair", "description": "chair", "count": 1}]}
    derived = deepcopy(parent)
    derived["condition"] = {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "room_type": "office", "floor_z_m": 0.,
        "floor_known": False, "boundary_known": False, "boundary_quality": "source_reference_extent",
        "height_m": None, "floor_polygon_xy_m": [[0., 0.], [4., 0.], [4., 3.], [0., 3.]]},
        "objects": [{"id": "obj_0000", "category": "chair", "description": "chair"}], "constraints": []}
    derived["target"]["objects"][0]["bottom_center_m"] = [1., 1., .5]
    derived["validity"]["yaw_symmetry_order"] = [2]
    derived["provenance"] = {**deepcopy(parent["provenance"]), "minimal_request": request,
        "parent_condition": deepcopy(parent["condition"]), "parent_validity": deepcopy(parent["validity"]),
        "condition_projection": "room_type_reference_extent_furniture_only-v1",
        "room_dimension_mode": "xy", "room_size_semantics": "reference_extent",
        "frame_translation_m": [-10., -20., -floor], "geometric_yaw_qualified": [False],
        "yaw_label_semantics": "local_bbox_axes_mod_pi_not_certified_semantic_front",
        "correspondence": "fixed_request_identity"}
    source_room = {"source": source, "uid": uid, "group": "house_" + uid, "objects": [
        {"id": "raw_0", "size": [1., .5, 1.], "pos": [11., 21., .5 + floor], "yaw": 0., "tilted": False}],
        "boundary": deepcopy(parent["condition"]["room"]["floor_polygon_xy_m"]), "room_type": "office"}
    metadata(parent, derived)
    return parent, derived, source_room


def test_reference_extent_keeps_estimated_labels_without_repairing():
    parent, derived, ir = pair(source="Scan2CAD", floor=-.2)
    derived["validity"]["position"] = [[False] * 3]
    metadata(parent, derived)
    before = deepcopy((parent, derived, ir))
    stats = verify_pair(parent, derived, "train", ir)
    assert stats["objects"] == 1
    assert stats["position_masks_demoted"] == 1
    assert (parent, derived, ir) == before


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r["target"]["objects"][0]["bottom_center_m"].__setitem__(0, 1.1), "target"),
    (lambda r: r["target"]["objects"][0]["target_size_local_m"].__setitem__(1, .6), "target"),
    (lambda r: r["condition"]["objects"][0].update(target_size_local_m=[1., .5, 1.]), "condition"),
    (lambda r: r["condition"]["room"].update(floor_known=True), "condition"),
    (lambda r: r["validity"].update(yaw=[True]), "validity"),
    (lambda r: r["provenance"].update(split="test"), "split"),
    (lambda r: r["provenance"].update(house_id="different"), "house_id"),
    (lambda r: r["provenance"].update(target_source_ids=["other"]), "target_source_ids"),
    (lambda r: r["provenance"]["parent_condition"]["room"].update(floor_z_m=0.2), "parent_condition"),
])
def test_tampering_is_rejected(mutate, reason):
    parent, derived, ir = pair()
    mutate(derived)
    with pytest.raises(ValueError, match=reason):
        verify_pair(parent, derived, "train", ir)


def test_spatiallm_yaw_needs_exact_sealed_source_identity_and_axes():
    parent, derived, ir = pair(source="SpatialLM")
    derived["validity"]["yaw"] = [True]
    derived["provenance"]["geometric_yaw_qualified"] = [True]
    metadata(parent, derived)
    assert verify_pair(parent, derived, "train", ir)["yaw_masks_promoted"] == 1
    bad_ir = deepcopy(ir)
    bad_ir["objects"][0]["yaw"] = .1
    with pytest.raises(ValueError, match="source_geometry"):
        verify_pair(parent, derived, "train", bad_ir)


def test_size_range_masks_complete_vector_but_never_resizes_label():
    parent, derived, ir = pair()
    parent["target"]["objects"][0]["target_size_local_m"][2] = 1e-9
    derived["target"]["objects"][0]["target_size_local_m"][2] = 1e-9
    derived["validity"]["size"] = [[False] * 3]
    ir["objects"][0]["size"][2] = 1e-9
    metadata(parent, derived)
    assert verify_pair(parent, derived, "train", ir)["size_masks_demoted"] == 1
    derived["target"]["objects"][0]["target_size_local_m"][2] = math.exp(-10)
    with pytest.raises(ValueError, match="target"):
        verify_pair(parent, derived, "train", ir)


def test_parent_null_numeric_labels_are_preserved():
    parent, derived, ir = pair()
    parent["target"]["objects"][0]["bottom_center_m"][2] = None
    parent["validity"]["position"][0][2] = False
    derived["target"]["objects"][0]["bottom_center_m"][2] = None
    derived["validity"]["position"][0][2] = False
    derived["provenance"]["parent_validity"] = deepcopy(parent["validity"])
    ir["objects"][0]["pos"][2] = None
    assert verify_pair(parent, derived, "train", ir)["objects"] == 1


def files(tmp_path, cross_house=False):
    parent_root, data_root, ir_root = [tmp_path / name for name in ("parent", "derived", "ir")]
    for path in (parent_root, data_root, ir_root):
        path.mkdir()
    raw_rows = []
    for split in ("train", "validation", "test"):
        parent, derived, raw = pair(split=split, uid=split)
        raw_rows.append(raw)
        if cross_house:
            for row in (parent, derived):
                row["provenance"]["house_id"] = "common"
        for path, row in ((parent_root, parent), (data_root, derived)):
            (path / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    (parent_root / "manifest.json").write_text("{}")
    ir_path = ir_root / "HSSD200.jsonl"
    ir_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"input_sha256": {str(p): digest(p) for p in [*parent_root.iterdir(), ir_path]},
                "output_sha256": {p.name: digest(p) for p in data_root.iterdir()},
                "size_output_policy": {"size_reference": [1., 1., 1.], "size_log_limit": 10.},
                "source_IR_rows": {"HSSD200": 3},
                "split_samples": {s: 1 for s in ("train", "validation", "test")},
                "split_objects": {s: 1 for s in ("train", "validation", "test")}}
    (data_root / "manifest.json").write_text(json.dumps(manifest))
    return parent_root, data_root, ir_root, {ir_path.name: digest(ir_path)}


def test_full_verifier_checks_all_splits_hashes_and_output_safety(tmp_path):
    parent, data, ir, seals = files(tmp_path)
    report = verify_dataset(parent, data, ir, expected_ir_sha256=seals)
    assert report["ok"]
    assert report["counts"]["samples"] == 3
    assert report["counts"]["objects"] == 3
    assert report["input_hashes_unchanged"]
    with pytest.raises(FileExistsError):
        verify_dataset(parent, data, ir, output=data / "manifest.json", expected_ir_sha256=seals)
    with pytest.raises(ValueError, match="outside"):
        verify_dataset(parent, data, ir, output=ir / "new.json", expected_ir_sha256=seals)


def test_full_verifier_reports_cross_split_house_and_corrupted_hash(tmp_path):
    parent, data, ir, seals = files(tmp_path, cross_house=True)
    (data / "train.jsonl").write_text((data / "train.jsonl").read_text().replace('"yaw_rad": 0.0', '"yaw_rad": 0.1'))
    report = verify_dataset(parent, data, ir, expected_ir_sha256=seals)
    assert not report["ok"]
    assert report["error_counts"]["group_cross_split"] > 0
    assert report["error_counts"]["output_hash"] == 1


def test_full_condition_view_never_projects_or_promotes_yaw():
    parent, _, _ = pair()
    derived = deepcopy(parent)
    derived["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=[])
    assert verify_full_pair(parent, derived, "train")["objects"] == 1
    assert derived["condition"]["room"]["fixed_objects"]
    assert derived["condition"]["constraints"]
    derived["validity"]["yaw"] = [True]
    with pytest.raises(ValueError, match="validity"):
        verify_full_pair(parent, derived, "train")


def test_full_condition_target_numeric_changes_fail_even_with_invalid_mask():
    parent, _, _ = pair()
    derived = deepcopy(parent)
    derived["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=[])
    derived["target"]["objects"][0]["bottom_center_m"][0] = 1.
    derived["validity"]["position"] = [[False] * 3]
    with pytest.raises(ValueError, match="target"):
        verify_full_pair(parent, derived, "train")


def test_full_condition_scan2cad_only_demotes_and_records_estimate():
    parent, _, _ = pair(source="Scan2CAD")
    derived = deepcopy(parent)
    derived["condition"]["room"]["floor_known"] = False
    derived["validity"]["position"] = [[False] * 3]
    derived["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=[
        {"field": "room.floor_known", "before": True, "after": False,
         "reason": "Scan2CAD_floor_is_estimated_not_independent_physical_measurement"},
        {"field": "position", "object_id": "obj_0000", "target_source_id": "raw_0", "before": [True] * 3,
         "after": [False] * 3, "reason": "Scan2CAD_estimated_floor_and_upstream_snap_uncertainty"}],
        estimated_floor_provenance={"floor_source": "estimated", "parent_floor_known": True, "parent_floor_z_m": 0.,
            "source_meta_floor_z": None, "source_meta_n_floor_snapped": None,
            "per_object_pre_snap_z": "unavailable_in_frozen_IR", "per_object_snap_membership": "unknown_do_not_infer_from_zero_z"})
    assert verify_full_pair(parent, derived, "train")["position_masks_demoted"] == 1
    derived["provenance"]["estimated_floor_provenance"]["per_object_snap_membership"] = "known"
    with pytest.raises(ValueError, match="estimated_floor_provenance"):
        verify_full_pair(parent, derived, "train")


def test_full_condition_out_of_range_size_changes_mask_and_leaves_numbers():
    parent, _, _ = pair()
    parent["target"]["objects"][0]["target_size_local_m"][2] = 1e-9
    derived = deepcopy(parent)
    derived["validity"]["size"] = [[False] * 3]
    derived["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=[
        {"field": "size", "object_id": "obj_0000", "target_source_id": "raw_0", "before": [True] * 3,
         "after": [False] * 3, "value_m": [1., .5, 1e-9], "reason": "outside_configured_size_head_range"}])
    assert verify_full_pair(parent, derived, "train")["size_masks_demoted"] == 1
    derived["provenance"]["qualification_changes"] = []
    with pytest.raises(ValueError, match="qualification_changes"):
        verify_full_pair(parent, derived, "train")


def test_full_condition_dataset_machine_report(tmp_path):
    parent, data, ir, seals = files(tmp_path)
    for split in ("train", "validation", "test"):
        row = json.loads((parent / (split + ".jsonl")).read_text())
        row["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=[])
        (data / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["dataset_qualification_policy"] = "full-condition-mask-review-v1"
    manifest["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir() if p.name.endswith(".jsonl")}
    (data / "manifest.json").write_text(json.dumps(manifest))
    report = verify_dataset(parent, data, ir, full_condition=True, expected_ir_sha256=seals)
    assert report["ok"]
    assert report["counts"]["samples"] == 3
    assert report["counts"]["removed_constraints"] == 0
    # A self-consistent hash cannot hide an invented per-record change.
    journal = data / "changes.jsonl"
    journal.write_text('{"invented_change":true}\n')
    manifest["output_sha256"][journal.name] = hashlib.sha256(journal.read_bytes()).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest))
    altered = verify_dataset(parent, data, ir, full_condition=True, expected_ir_sha256=seals)
    assert not altered["ok"]
    assert altered["error_counts"]["changes_journal"] == 1


@pytest.mark.parametrize("field,value", [("position", [[1, True, True]]), ("size", [[True]]), ("yaw", [])])
def test_malformed_parent_masks_fail_before_comparison(field, value):
    parent, derived, ir = pair()
    parent["validity"][field] = value
    with pytest.raises(ValueError, match="parent validity"):
        verify_pair(parent, derived, "train", ir)


def test_active_parent_null_label_is_rejected():
    parent, derived, ir = pair()
    parent["target"]["objects"][0]["bottom_center_m"][0] = None
    with pytest.raises(ValueError, match="invalid active target"):
        verify_pair(parent, derived, "train", ir)


def test_full_condition_partial_geometry_removes_entire_exchange_group():
    parent, _, _ = pair(source="Scan2CAD")
    parent["condition"]["objects"][0]["exchangeable_group"] = "chairs"
    parent["condition"]["objects"].append({**deepcopy(parent["condition"]["objects"][0]), "id": "obj_0001"})
    parent["target"]["objects"].append({**deepcopy(parent["target"]["objects"][0]), "id": "obj_0001"})
    for field in ("position", "size", "yaw", "size_axis_swap_allowed"):
        parent["validity"][field].append(deepcopy(parent["validity"][field][0]))
    parent["provenance"]["target_source_ids"].append("raw_1")
    derived = deepcopy(parent)
    derived["condition"]["room"]["floor_known"] = False
    derived["validity"]["position"] = [[False] * 3] * 2
    changes = [{"field": "room.floor_known", "before": True, "after": False,
        "reason": "Scan2CAD_floor_is_estimated_not_independent_physical_measurement"}]
    for i in (0, 1):
        changes.append({"object_id": f"obj_{i:04d}", "target_source_id": f"raw_{i}", "field": "position",
            "before": [True] * 3, "after": [False] * 3, "reason": "Scan2CAD_estimated_floor_and_upstream_snap_uncertainty"})
    # A pre-C1 parent carries the group inside condition objects; the derived row never does.
    for i in (0, 1):
        derived["condition"]["objects"][i].pop("exchangeable_group")
        changes.append({"object_id": f"obj_{i:04d}", "target_source_id": f"raw_{i}", "field": "exchangeable_group",
            "before": "chairs", "after": None, "reason": "mask_review_removed_complete_position_exchangeability"})
    derived["validity"]["exchangeable_group"] = [None, None]
    derived["provenance"].update(dataset_qualification_policy="full-condition-mask-review-v1", qualification_changes=changes,
        estimated_floor_provenance={"floor_source": "estimated", "parent_floor_known": True, "parent_floor_z_m": 0.,
            "source_meta_floor_z": None, "source_meta_n_floor_snapped": None, "per_object_pre_snap_z": "unavailable_in_frozen_IR",
            "per_object_snap_membership": "unknown_do_not_infer_from_zero_z"})
    assert verify_full_pair(parent, derived, "train")["exchangeable_members_demoted"] == 2
    derived["validity"]["exchangeable_group"] = [None, "chairs"]
    with pytest.raises(ValueError, match="validity"):
        verify_full_pair(parent, derived, "train")
    derived["validity"]["exchangeable_group"] = [None, None]
    derived["condition"]["objects"][1]["exchangeable_group"] = "chairs"
    with pytest.raises(ValueError, match="condition"):
        verify_full_pair(parent, derived, "train")


def test_minimal_journal_orders_all_demotions_before_geometric_yaw():
    parent, derived, ir = pair(source="SpatialLM")
    parent["condition"]["objects"].append({**deepcopy(parent["condition"]["objects"][0]), "id": "obj_0001"})
    parent["target"]["objects"].append({**deepcopy(parent["target"]["objects"][0]), "id": "obj_0001", "target_size_local_m": [1., .5, 1e-9]})
    for field in ("position", "size", "yaw", "size_axis_swap_allowed"):
        parent["validity"][field].append(deepcopy(parent["validity"][field][0]))
    parent["provenance"]["target_source_ids"].append("raw_1")
    parent["provenance"]["field_evidence"].append(deepcopy(parent["provenance"]["field_evidence"][0]))
    ir["objects"].append({**deepcopy(ir["objects"][0]), "id": "raw_1", "size": [1., .5, 1e-9]})
    derived["condition"]["objects"].append({**deepcopy(derived["condition"]["objects"][0]), "id": "obj_0001"})
    derived["target"]["objects"].append({**deepcopy(derived["target"]["objects"][0]), "id": "obj_0001", "target_size_local_m": [1., .5, 1e-9]})
    derived["validity"] = {"position": [[True] * 3] * 2, "size": [[True] * 3, [False] * 3],
                           "yaw": [True, True], "yaw_symmetry_order": [2, 2], "size_axis_swap_allowed": [False, False]}
    derived["provenance"].update(deepcopy(parent["provenance"]))
    derived["provenance"]["parent_condition"] = deepcopy(parent["condition"])
    derived["provenance"]["parent_validity"] = deepcopy(parent["validity"])
    derived["provenance"]["minimal_request"]["furniture_list"].append(
        {"id": "obj_0001", "category": "chair", "description": "chair", "count": 1})
    derived["provenance"]["geometric_yaw_qualified"] = [True, True]
    changes = [{"object_id": "obj_0001", "target_source_id": "raw_1", "field": "size", "before": [True] * 3,
        "after": [False] * 3, "value_m": [1., .5, 1e-9], "reason": "outside_configured_size_head_range"}]
    changes += [{"object_id": f"obj_{i:04d}", "target_source_id": f"raw_{i}", "field": "yaw",
        "before": False, "after": True, "reason": "pinned_SpatialLM_full_local_extents_and_geometric_yaw"} for i in (0, 1)]
    changes.append({"field": "bottom_center_m", "operation": "common_input_reference_translation", "offset_m": [-10., -20., 0.],
        "reason": "structural_input_extent_and_explicit_floor_reference"})
    derived["provenance"]["qualification_changes"] = changes
    assert verify_pair(parent, derived, "train", ir)["yaw_masks_promoted"] == 2
    derived["provenance"]["qualification_changes"] = [changes[1], changes[0], *changes[2:]]
    with pytest.raises(ValueError, match="qualification_changes"):
        verify_pair(parent, derived, "train", ir)
