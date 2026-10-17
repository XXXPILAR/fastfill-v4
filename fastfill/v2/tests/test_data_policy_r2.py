"""Round-2 data contract: K1 box-symmetry tier, K2 height-conflict flag, K3 source floor declarations."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from unittest.mock import patch

import pytest

from fastfill.v2 import review_data as review
from fastfill.v2.io import fingerprint
from fastfill.v2.legacy_build import build_selected_dataset
from fastfill.v2.legacy_verify import verify_selected_dataset
from fastfill.v2.multisource_data import project_sample
from fastfill.v2.multisource_verify import verify_dataset, verify_full_pair, verify_pair
from fastfill.v2.qualified_data import build_dataset as build_qualified, qualify_sample
from fastfill.v2.schema import migrate_legacy_row
from fastfill.v2.tests import test_legacy_bridge
from fastfill.v2.tests.test_legacy_bridge_geometry import by_source, convert, obj, room
from fastfill.v2.tests.test_legacy_build import miniature_release, rows
from fastfill.v2.tests.test_minimal_data import pair as minimal_pair
from fastfill.v2.tests.test_qualified_data import sample as qualified_parent


def multiscan(objects):
    return room("MultiScan", objects, floor_z=-1., floor_height=-1., height_reliable=True, n_floor_objects_outside_structure=0)


# --- K1: box-symmetry tier --------------------------------------------------

@pytest.mark.parametrize("raw,ident,swap,order", [
    (room("InternScenes_arkit", [obj("a", "table")]), "a", True, 4),
    (room("InternScenes_scannet", [obj("a", "lamp")]), "a", True, 4),
    (room("InteriorGS", [obj("a", "sofa")]), "a", True, 4),
    (room("HSSD200", [obj("a", "semi_chair"), obj("b", "table")]), "a", True, 4),
    (room("HSSD200", [obj("a", "semi_chair"), obj("b", "table")]), "b", False, 2),
    (multiscan([obj("a", "bed"), obj("b", "table")]), "a", True, 4),
    (multiscan([obj("a", "bed"), obj("b", "table")]), "b", False, 1),
    (multiscan([obj("a", "bed net"), obj("b", "table")]), "a", False, 1),
    (room("SAGE-10k", [obj("a", "chair")]), "a", False, 2),
    (room("InternScenes_gen", [obj("a", "chair")]), "a", False, 2),
])
def test_axis_policy_marks_swap_tier_with_order_four(raw, ident, swap, order):
    _, validity = by_source(convert(raw, policy="axis"))[ident]
    assert validity["size_axis_swap_allowed"] is swap and validity["yaw_symmetry_order"] == order


def test_swap_tier_is_axis_policy_only_and_old_rows_migrate_to_false():
    strict = convert(room("InternScenes_arkit", [obj("a", "table")]), policy="strict")
    assert strict["validity"]["size_axis_swap_allowed"] == [False] and strict["validity"]["yaw_symmetry_order"] == [1]
    old = deepcopy(strict)
    del old["validity"]["size_axis_swap_allowed"]
    migrated = migrate_legacy_row(old)
    assert migrated["validity"]["size_axis_swap_allowed"] == [False] and "size_axis_swap_allowed" not in old["validity"]
    assert migrate_legacy_row(migrated) is migrated


# --- K3: floor support from source annotation only --------------------------

def test_source_floor_anchor_declares_and_snaps_within_two_cm_only():
    raw = room("IL3D_3dfront", [obj("exact", anchor="floor"), obj("near", anchor="floor", pos=[2., 1., .015]),
                                obj("float", anchor="floor", pos=[3., 1., .05]),
                                obj("guess", anchor="floor", anchor_inferred=True, pos=[4., 1., .01]),
                                obj("none", pos=[4., 3., 0.])])
    result = convert(raw, policy="axis")
    requests = {source: result["condition"]["objects"][i] for i, source in enumerate(result["provenance"]["target_source_ids"])}
    evidence = {source: result["provenance"]["field_evidence"][i] for i, source in enumerate(result["provenance"]["target_source_ids"])}
    targets = {source: target for source, (target, _) in by_source(result).items()}
    assert {k: v.get("support_parent") for k, v in requests.items()} == {
        "exact": "floor", "near": "floor", "float": None, "guess": None, "none": None}
    assert targets["near"]["bottom_center_m"] == [2., 1., 0.] and evidence["near"]["legacy_z_snap_applied_to_target"] is True
    assert targets["exact"]["bottom_center_m"][2] == 0. and evidence["exact"]["legacy_z_snap_applied_to_target"] is False
    assert targets["float"]["bottom_center_m"][2] == .05 and evidence["float"]["legacy_z_snap_applied_to_target"] is False
    assert raw["objects"][1]["pos"][2] == .015  # source IR is never rewritten
    unknown_floor = convert(multiscan([obj("a", anchor="floor")]), policy="axis")
    assert "support_parent" not in unknown_floor["condition"]["objects"][0]


def snapping_release(tmp_path):
    """Miniature release whose scenes hold one snapped (1.5 cm) and one floating (5 cm) floor anchor."""
    def fixture():
        raw = original()
        raw["objects"][0]["pos"] = [4, 3, .05]
        raw["objects"][1]["pos"] = [1, 1, .015]
        return raw
    original = test_legacy_bridge.fixture
    with patch("fastfill.v2.tests.test_legacy_build.fixture", fixture):
        return miniature_release(tmp_path)


def build(release, evidence, output, **kwargs):
    with redirect_stdout(StringIO()):
        return build_selected_dataset(release, output, evidence_root=evidence, **kwargs)


def assert_failed(report, text):
    assert not report["passed"] and any(text in e["message"] for e in report["errors"]), report["errors"]


def mutate(path, change):
    records = rows(path)
    change(records[0])
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_bridge_manifest_counts_floor_declarations_and_verifier_recomputes_them(tmp_path):
    release, evidence = snapping_release(tmp_path)
    manifest = build(release, evidence, tmp_path / "dataset")
    assert (manifest["floor_declarations_written"], manifest["floor_declaration_z_snapped"],
            manifest["floor_declaration_skipped_floating"]) == (5, 3, 1)  # flagged scenes move object 0 to z=0
    assert manifest["source_size_axis_swap_allowed_objects"] == {}
    report = verify_selected_dataset(tmp_path / "dataset")
    assert report["passed"], report["errors"]
    for row in rows(tmp_path / "dataset/train.jsonl"):
        assert sorted(t["bottom_center_m"][2] for t in row["target"]["objects"]) == [0, .05]

    def undeclare(row):
        for request in row["condition"]["objects"]:
            request.pop("support_parent", None)
    for name, change, message in (
            ("undeclared", undeclare, "snapped to the floor without a floor declaration"),
            ("floating", lambda row: [r.update(support_parent="floor") for r in row["condition"]["objects"]],
             "floor support"),  # caught by the z-consistency check before the anchor check
            ("anchorless", lambda row: [e.update(raw_anchor=None) for e in row["provenance"]["field_evidence"]],
             "floor support declared without a source floor anchor"),
            ("swap", lambda row: row["validity"].update(size_axis_swap_allowed=[True, True], yaw_symmetry_order=[4, 4]),
             "size_axis_swap_allowed differs")):
        release, evidence = snapping_release(tmp_path / name)
        build(release, evidence, tmp_path / name / "dataset")
        mutate(tmp_path / name / "dataset/train.jsonl", change)
        assert_failed(verify_selected_dataset(tmp_path / name / "dataset"), message)


def test_unsnapped_floor_anchor_within_tolerance_is_a_verifier_failure(tmp_path):
    release, evidence = snapping_release(tmp_path)
    build(release, evidence, tmp_path / "dataset")

    def unsnap(row):
        for request, target, item in zip(row["condition"]["objects"], row["target"]["objects"], row["provenance"]["field_evidence"]):
            if item["legacy_z_snap_applied_to_target"]:
                request.pop("support_parent")
                target["bottom_center_m"][2], item["legacy_z_snap_applied_to_target"] = .015, False
    mutate(tmp_path / "dataset/train.jsonl", unsnap)
    assert_failed(verify_selected_dataset(tmp_path / "dataset"), "within the snap tolerance was not declared")


def test_bounded_smoke_build_caps_each_source_exactly(tmp_path):
    release, evidence = miniature_release(tmp_path)
    manifest = build(release, evidence, tmp_path / "dataset", sources=["SpatialLM"], max_scenes_per_source=2)
    assert manifest["samples_written"] == 2 and manifest["bounded_build"] is True
    assert manifest["source_filter"] == ["SpatialLM"] and manifest["max_scenes_per_source"] == 2
    assert verify_selected_dataset(tmp_path / "dataset")["passed"]
    with pytest.raises(ValueError, match="unknown source filter"):
        build(release, evidence, tmp_path / "other", sources=["NoSuchSource"])


def test_snapped_rows_flow_through_review_qualify_and_full_verifier(tmp_path):
    release, evidence = snapping_release(tmp_path)
    build(release, evidence, tmp_path / "bridge")
    with redirect_stdout(StringIO()):
        review.build_reviewed(tmp_path / "bridge", tmp_path / "reviewed")
        pins = {p.name: fingerprint(p) for p in (tmp_path / "reviewed").iterdir()}
        manifest = build_qualified(tmp_path / "reviewed", tmp_path / "main", expected_hashes=pins, holdout_groups=())
    assert manifest["source_size_axis_swap_allowed_objects"] == {"SpatialLM": 0}
    report = verify_dataset(tmp_path / "reviewed", tmp_path / "main", tmp_path, expected_ir_sha256={},
                            full_condition=True, holdout_groups=())
    assert report["ok"], report["errors_first_100"]


# --- K2: height conflict is a flag only -------------------------------------

def test_qualified_build_flags_height_conflict_in_journal_and_verifier_mirrors_it(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    for split in ("train", "validation", "test"):
        row = qualified_parent(split=split)
        row["provenance"]["field_evidence"] = [{"tilted": False}]
        row["target"]["objects"][0]["target_size_local_m"][2] = 2.9 if split == "train" else .75
        (parent / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
    (parent / "manifest.json").write_text('{"schema_version":"fastfill.v2"}\n')
    (parent / "rejections.jsonl").write_text("")
    pins = {p.name: fingerprint(p) for p in parent.iterdir()}
    with redirect_stdout(StringIO()):
        manifest = build_qualified(parent, tmp_path / "main", expected_hashes=pins, holdout_groups=())
    assert manifest["qualification_change_counts"] == {"provenance.height_conflict": 1}
    flagged = rows(tmp_path / "main/train.jsonl")[0]
    assert flagged["condition"]["room"]["height_m"] == 2.8
    assert flagged["provenance"]["height_conflict"]["objects"] == ["desk"]
    journal, = rows(tmp_path / "main/changes.jsonl")
    assert journal["field"] == "provenance.height_conflict" and journal["reason"] == "target_exceeds_declared_height_flag_only"
    report = verify_dataset(parent, tmp_path / "main", tmp_path, expected_ir_sha256={}, full_condition=True, holdout_groups=())
    assert report["ok"], report["errors_first_100"]
    row = json.loads((parent / "train.jsonl").read_text())
    derived = qualify_sample(row)
    derived["provenance"]["height_conflict"]["max_excess_m"] += 1e-3
    with pytest.raises(ValueError, match="height_conflict"):
        verify_full_pair(row, derived, "train", holdout_groups=())


# --- minimal view: snapped targets and the swap tier ------------------------

def test_minimal_view_accepts_bridge_snap_and_keeps_swap_order_four():
    sample, source = minimal_pair(floor=0.)
    sample["provenance"]["source"] = source["source"] = "IL3D_3dfront"  # a source with floor anchors
    sample["target"]["objects"][0]["bottom_center_m"][2] = 0.
    source["objects"][0]["pos"][2] = .015
    sample["provenance"]["field_evidence"][0]["legacy_z_snap_applied_to_target"] = True
    sample["validity"].update(size_axis_swap_allowed=[True], yaw_symmetry_order=[4])
    result = project_sample(sample, source, split="train")
    assert result["target"]["objects"][0]["bottom_center_m"][2] == 0.
    assert result["validity"]["yaw_symmetry_order"] == [4] and result["validity"]["size_axis_swap_allowed"] == [True]
    assert verify_pair(sample, result, "train", source)["objects"] == 1
    far = deepcopy(source)
    far["objects"][0]["pos"][2] = .03
    with pytest.raises(ValueError, match="snap"):
        project_sample(sample, far, split="train")
    with pytest.raises(ValueError, match="snap"):
        verify_pair(sample, result, "train", far)
    unflagged = deepcopy(sample)
    unflagged["provenance"]["field_evidence"][0]["legacy_z_snap_applied_to_target"] = False
    with pytest.raises(ValueError, match="source_geometry"):
        project_sample(unflagged, source, split="train")
