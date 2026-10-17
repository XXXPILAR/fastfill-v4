"""Frozen source evidence must affect only the copied v2 geometry."""
import json
from copy import deepcopy

import pytest

from fastfill.v2.legacy_evidence import EvidenceIndex


def evidence_root(tmp_path, *, span=0.3716752803):
    designer = tmp_path / "designer"
    intern = tmp_path / "internscenes"
    designer.mkdir()
    intern.mkdir()
    (designer / "opening-target-join.json").write_text(json.dumps({
        "saved_hits": [{"uid": "room", "source_object_id": "window-src",
                        "kind": "windows", "along_wall_span": span}]
    }))
    (intern / "k0_saved_object_exposure.jsonl").write_text(json.dumps({
        "uid": "room", "object_id": "chair-src", "vertical_axis": 0,
        "geometry_class": "upright", "retained_state": "target"
    }) + "\n")
    return tmp_path


def raw_room():
    return {"uid": "room", "source": "IL3D_3dfront", "meta": {"front_known": True},
            "objects": [
                {"id": "window-src", "category": "window", "structure": True,
                 "size": [0.26428, 0.1, 1.5], "pos": [1.0, 2.0, 0.9], "yaw": 0.7},
                {"id": "chair-src", "category": "chair", "front_known": True,
                 "size": [0.5, 0.6, 0.8], "pos": [2.0, 2.0, 0.0], "yaw": 1.0}]}


def test_correction_preserves_original_and_other_geometry(tmp_path):
    index = EvidenceIndex.from_root(evidence_root(tmp_path), require=True)
    raw = raw_room()
    before = deepcopy(raw)
    result = index.apply_evidence(raw)
    assert raw == before
    corrected = result["objects"][0]
    assert corrected["size"] == [0.3716752803, 0.1, 1.5]
    assert corrected["pos"] == raw["objects"][0]["pos"]
    assert corrected["yaw"] == raw["objects"][0]["yaw"]
    assert corrected["v2_evidence"]["corrected_width_m"] == 0.3716752803
    assert corrected["v2_evidence"]["original_width_m"] == 0.26428
    assert result["meta"]["v2_evidence"]["opening_width_corrections"] == 1


def test_k0_marks_front_unknown_without_dropping_or_changing_pose(tmp_path):
    result = EvidenceIndex.from_root(evidence_root(tmp_path)).apply_evidence(raw_room())
    chair = result["objects"][1]
    assert chair["front_known"] is False
    assert chair["size"] == [0.5, 0.6, 0.8]
    assert chair["yaw"] == 1.0
    assert chair["v2_evidence"]["front_unknown_reason"] == (
        "source_local_x_vertical_geometric_heading_fallback")
    assert len(result["objects"]) == 2
    assert result["meta"]["v2_evidence"]["front_unknown_objects"] == 1


def test_index_maps_are_immutable(tmp_path):
    index = EvidenceIndex.from_root(evidence_root(tmp_path))
    with pytest.raises(TypeError):
        index.opening_widths["room"]["window-src"] = 10.0
    with pytest.raises(TypeError):
        index.front_unknown_objects["room"] = frozenset()
    assert isinstance(index.front_unknown_objects["room"], frozenset)


def test_optional_missing_evidence_is_recorded_not_declared_clean(tmp_path):
    index = EvidenceIndex.from_root(tmp_path)
    result = index.apply_evidence(raw_room())
    assert result["objects"] == raw_room()["objects"]
    summary = result["meta"]["v2_evidence"]
    assert summary["status"] == "not_applied"
    assert len(summary["missing_files"]) == 2
    assert summary["scope"] == "recorded_source_evidence_only"


def test_partial_index_applies_available_records_and_reports_absence(tmp_path):
    root = evidence_root(tmp_path)
    (root / "internscenes/k0_saved_object_exposure.jsonl").unlink()
    result = EvidenceIndex.from_root(root).apply_evidence(raw_room())
    assert result["objects"][0]["size"][0] == 0.3716752803
    assert result["objects"][1]["front_known"] is True
    assert result["meta"]["v2_evidence"]["status"] == "partial_evidence"


def test_required_missing_files_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="opening-target-join"):
        EvidenceIndex.from_root(tmp_path, require=True)


@pytest.mark.parametrize("span", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_width_evidence_fails(tmp_path, span):
    with pytest.raises(ValueError, match="positive finite"):
        EvidenceIndex.from_root(evidence_root(tmp_path, span=span))


def test_malformed_present_file_fails_instead_of_becoming_missing(tmp_path):
    root = evidence_root(tmp_path)
    (root / "designer/opening-target-join.json").write_text("broken")
    with pytest.raises(ValueError, match="opening-target-join"):
        EvidenceIndex.from_root(root)


def test_corrected_id_must_reference_window_geometry(tmp_path):
    raw = raw_room()
    raw["objects"][0]["category"] = "chair"
    with pytest.raises(ValueError, match="window"):
        EvidenceIndex.from_root(evidence_root(tmp_path)).apply_evidence(raw)


def test_no_matching_records_is_not_a_complete_geometry_audit(tmp_path):
    raw = raw_room()
    raw["uid"] = "another-room"
    summary = EvidenceIndex.from_root(evidence_root(tmp_path)).apply_evidence(raw)["meta"]["v2_evidence"]
    assert summary["status"] == "applied_no_matching_records"
    assert summary["scope"] == "recorded_source_evidence_only"


def test_applying_recorded_evidence_twice_is_idempotent(tmp_path):
    index = EvidenceIndex.from_root(evidence_root(tmp_path))
    once = index.apply_evidence(raw_room())
    assert index.apply_evidence(once) == once
