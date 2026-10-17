"""Audit fixes 2026-10-06, data-builder side: C1 schema, C4 axis yaw, C5 groups, C6 descriptions, C7 qualification."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from unittest.mock import patch

import pytest

from fastfill.v2 import review_data as review
from fastfill.v2.legacy_build import build_selected_dataset, main as legacy_main
from fastfill.v2.multisource_data import ROOMGENBENCH_HOLDOUT_GROUPS, project_sample
from fastfill.v2.multisource_data import build_dataset as build_minimal
from fastfill.v2.multisource_verify import verify_dataset, verify_full_pair
from fastfill.v2.qualified_data import build_dataset as build_qualified, main as qualified_main, qualify_sample
from fastfill.v2.schema import migrate_legacy_row, validate_condition
from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.tests.test_legacy_bridge_geometry import by_source, convert, obj, room, saved_row
from fastfill.v2.tests.test_legacy_build import miniature_release, rows
from fastfill.v2.tests.test_multisource_data import dataset_files, source_pair
from fastfill.v2.tests.test_qualified_data import sample as qualified_parent


# --- C1: schema -------------------------------------------------------------

def legacy_row():
    return {"schema_version": "fastfill.v2", "condition": {"schema_version": "fastfill.v2",
        "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 3], [0, 3]]},
        "objects": [{"id": "a", "category": "chair", "description": "chair", "exchangeable_group": "g"},
                    {"id": "b", "category": "chair", "description": "chair", "exchangeable_group": "g"},
                    {"id": "c", "category": "desk", "description": "desk"}], "constraints": []},
        "target": {"schema_version": "fastfill.v2", "objects": [{"id": "c"}, {"id": "b"}, {"id": "a"}]},
        "validity": {"position": [[True] * 3] * 3, "size": [[True] * 3] * 3, "yaw": [False] * 3}}


def test_condition_objects_reject_exchangeable_group():
    with pytest.raises(ValueError, match="exchangeable_group"):
        validate_condition(legacy_row()["condition"])


def test_migrate_legacy_row_moves_groups_into_target_ordered_validity():
    row = legacy_row()
    before = deepcopy(row)
    migrated = migrate_legacy_row(row)
    assert row == before
    assert migrated["validity"]["exchangeable_group"] == [None, "g", "g"]
    assert all("exchangeable_group" not in o for o in migrated["condition"]["objects"])
    validate_condition(migrated["condition"])
    assert migrate_legacy_row(migrated) is migrated
    row["validity"]["exchangeable_group"] = [None] * 3
    with pytest.raises(ValueError, match="both"):
        migrate_legacy_row(row)


# --- C4/C5/C6: legacy bridge ----------------------------------------------

def test_axis_default_admits_front_known_yaw_with_symmetry_order_by_source():
    raw = room("SAGE-10k", [obj("chair", yaw=1.2), obj("tilted", tilted=True), obj("blind", front_known=False)])
    result = convert_selected_room(raw, {**deepcopy(raw), "fixed": []}, saved_row(), "train")  # default policy
    byid = by_source(result)
    assert byid["chair"][1]["yaw"] is True and byid["chair"][1]["yaw_symmetry_order"] == 2
    assert byid["tilted"][1]["yaw"] is False and not any(byid["tilted"][1]["position"])
    assert byid["blind"][1]["yaw"] is False
    assert result["provenance"]["field_evidence"][0]["front_policy"] == "axis"
    scan = convert(room("MultiScan", [obj("chair")], floor_z=-1., floor_height=-1., height_reliable=True,
                        n_floor_objects_outside_structure=0), policy="axis")
    assert scan["validity"]["yaw"] == [True] and scan["validity"]["yaw_symmetry_order"] == [1]
    assert convert(room("SAGE-10k", [obj("chair")]), policy="strict")["validity"]["yaw"] == [False]
    with pytest.raises(ValueError, match="front policy"):
        convert(room("SAGE-10k", [obj("chair")]), policy="axis-free")


def test_groups_need_only_complete_position_and_live_in_validity():
    twins = [obj("a", "chair", desc="A red chair", pos=[1., 1., 0.]), obj("b", "chair", desc="A red chair ", pos=[3., 1., 0.])]
    result = convert(room("MansionWorld", twins + [obj("c", "chair", desc="A blue chair")]), policy="axis")
    assert result["validity"]["size"] == [[False] * 3] * 3  # footprint proxy: no size supervision
    byid = by_source(result)
    assert byid["a"][1]["exchangeable_group"] == byid["b"][1]["exchangeable_group"]
    assert byid["a"][1]["exchangeable_group"].startswith("anonymous_")
    assert byid["c"][1]["exchangeable_group"] is None
    assert all("exchangeable_group" not in o for o in result["condition"]["objects"])
    tilted = convert(room("MansionWorld", [{**twins[0], "tilted": True}, twins[1]]), policy="axis")
    assert tilted["validity"]["exchangeable_group"] == [None, None]


def test_descriptions_come_from_source_desc_or_fall_back_to_category():
    result = convert(room("SAGE-10k", [obj("a", "Table", desc="  A wooden conference table  "), obj("b", "chair", desc=""),
                                      obj("c", "lamp", description="A brass lamp"), obj("d", "sofa")]), policy="axis")
    descriptions = {source: result["condition"]["objects"][i]["description"]
                    for i, source in enumerate(result["provenance"]["target_source_ids"])}
    assert descriptions == {"a": "A wooden conference table", "b": "chair", "c": "A brass lamp", "d": "sofa"}
    assert result["provenance"]["descriptions"] == "source_desc_or_category"


def test_build_manifest_records_yaw_policy_symmetry_and_group_counts(tmp_path):
    release, evidence = miniature_release(tmp_path)
    with redirect_stdout(StringIO()):
        result = build_selected_dataset(release, tmp_path / "dataset", evidence_root=evidence)
    assert result["front_policy"] == "axis"
    assert "axis" in result["yaw_policy"]
    assert result["source_yaw_valid_objects"] == {"SpatialLM": 6}
    assert result["source_yaw_symmetry_order_counts"] == {"SpatialLM:2": 6}
    assert result["exchangeable_group_counts"] == {"exchangeable_groups": 3, "exchangeable_members": 6}  # two identical chairs per room
    assert result["valid_label_counts"]["yaw"] == 6
    for split in ("train", "validation", "test"):
        for row in rows(tmp_path / "dataset" / f"{split}.jsonl"):
            assert row["validity"]["yaw_symmetry_order"] == [2, 2]
            assert row["validity"]["exchangeable_group"] == ["anonymous_0"] * 2
    with redirect_stdout(StringIO()):
        legacy_main(["--release-root", str(release), "--evidence-root", str(evidence),
                     "--output", str(tmp_path / "strict"), "--front-policy", "strict"])
    assert json.loads((tmp_path / "strict/manifest.json").read_text())["valid_label_counts"].get("yaw", 0) == 0


# --- review: facing trust needs a semantic front -------------------------

def test_axis_only_yaw_does_not_certify_source_derived_facing():
    parent = {"schema_version": "fastfill.v2", "condition": {"schema_version": "fastfill.v2",
        "room": {"frame": "right_handed_z_up"}, "objects": [{"id": "chair", "category": "chair"}, {"id": "table", "category": "table"}],
        "constraints": [{"type": "faces", "object_id": "chair", "target_id": "table"}]},
        "target": {"schema_version": "fastfill.v2", "objects": [
            {"id": "chair", "target_size_local_m": [.5, .6, .8], "bottom_center_m": [1., 1., 0.], "yaw_rad": .2},
            {"id": "table", "target_size_local_m": [1., 1., .7], "bottom_center_m": [2., 1., 0.], "yaw_rad": 0.}]},
        "validity": {"position": [[True] * 3] * 2, "size": [[True] * 3] * 2, "yaw": [True, True], "yaw_symmetry_order": [2, 2]},
        "provenance": {"scene_id": "s", "legacy_uid": "s", "source": "SAGE-10k", "split": "train",
                       "constraint_source": "frozen_sparse_legacy_request",
                       "field_evidence": [{"recorded_source_evidence": {}}, {"recorded_source_evidence": {}}]}}
    assert review.review_sample(parent)["condition"]["constraints"] == []
    parent["validity"]["yaw_symmetry_order"] = [1, 2]
    assert review.review_sample(parent)["condition"]["constraints"] == parent["condition"]["constraints"]


# --- C7: qualification ------------------------------------------------------

def test_degenerate_axis_masks_whole_size_with_its_own_reason():
    parent = qualified_parent()
    parent["target"]["objects"][0]["target_size_local_m"] = [1.4, .0029, .75]
    row = qualify_sample(parent)
    assert row["validity"]["size"] == [[False] * 3]
    assert row["target"] == parent["target"]
    change, = [c for c in row["provenance"]["qualification_changes"] if c["field"] == "size"]
    assert change["reason"] == "degenerate_axis_lt_3mm" and change["value_m"] == [1.4, .0029, .75]
    parent["target"]["objects"][0]["target_size_local_m"] = [1.4, .003, .75]
    assert qualify_sample(parent)["validity"]["size"] == [[True] * 3]


def test_target_above_declared_height_is_flagged_never_dropped():
    parent = qualified_parent()
    parent["target"]["objects"][0]["target_size_local_m"][2] = 2.84  # bottom 0 + 2.84 <= 2.8 + 0.05
    assert "height_conflict" not in qualify_sample(parent)["provenance"]
    parent["target"]["objects"][0]["target_size_local_m"][2] = 2.86
    row = qualify_sample(parent)
    assert row["condition"] == parent["condition"]  # K2: height_m stays 2.8
    assert row["target"] == parent["target"] and row["validity"]["size"] == [[True] * 3]
    conflict = {"objects": ["desk"], "max_excess_m": pytest.approx(.06)}
    assert row["provenance"]["height_conflict"] == conflict
    assert {"field": "provenance.height_conflict", "before": None, "after": conflict,
            "reason": "target_exceeds_declared_height_flag_only"} in row["provenance"]["qualification_changes"]
    assert verify_full_pair(parent, row, "train")["objects"] == 1
    dropped = deepcopy(row)
    dropped["condition"]["room"]["height_m"] = None  # the round-1 behaviour is now a verification failure
    with pytest.raises(ValueError, match="condition"):
        verify_full_pair(parent, dropped, "train")
    parent["validity"]["size"] = [[False] * 3]  # an unverified size never flags the room height
    assert "height_conflict" not in qualify_sample(parent)["provenance"]


def test_roomgenbench_rooms_leave_train_for_test_with_reason():
    parent = qualified_parent("SAGE-10k")
    parent["provenance"]["group"] = parent["provenance"]["house_id"] = "sage:layout_61ebde9f"
    row = qualify_sample(parent, holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS)
    assert row["provenance"]["split"] == "test"
    assert row["provenance"]["holdout_reason"] == "roomgenbench_benchmark_room"
    assert row["provenance"]["qualification_changes"][-1] == {
        "field": "provenance.split", "before": "train", "after": "test", "reason": "roomgenbench_benchmark_room"}
    assert verify_full_pair(parent, row, "train")["objects"] == 1
    with pytest.raises(ValueError, match="provenance split"):
        verify_full_pair(parent, row, "train", holdout_groups=())
    assert qualify_sample(parent)["provenance"]["split"] == "train"  # not listed: unchanged
    sample, raw = source_pair("SAGE-10k")
    sample["provenance"]["group"] = raw["group"] = sample["provenance"]["house_id"] = "sage:layout_6b049b06"
    minimal = project_sample(sample, raw, split="train", holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS)
    assert minimal["provenance"]["split"] == "test" and minimal["provenance"]["qualification_changes"][-1]["field"] == "provenance.split"


def test_qualified_build_writes_holdout_to_test_and_verifier_accepts_it(tmp_path):
    from fastfill.v2.io import fingerprint
    parent = tmp_path / "parent"
    parent.mkdir()
    for split in ("train", "validation", "test"):
        first = qualified_parent(split=split)
        second = qualified_parent("SAGE-10k", split)
        second["provenance"].update(scene_id="bench-" + split, group="sage:layout_fef15043" if split == "train" else "other-" + split,
                                    house_id="sage:layout_fef15043" if split == "train" else "other-" + split)
        (parent / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in (first, second)))
    (parent / "manifest.json").write_text('{"schema_version":"fastfill.v2","front_policy":"axis","yaw_policy":"axis policy"}\n')
    (parent / "rejections.jsonl").write_text("")
    pins = {p.name: fingerprint(p) for p in parent.iterdir()}
    with redirect_stdout(StringIO()):
        manifest = build_qualified(parent, tmp_path / "new", expected_hashes=pins)
    assert manifest["split_samples"] == {"train": 1, "validation": 2, "test": 3}
    assert manifest["holdout_samples_by_parent_split"] == {"train": 1, "validation": 0, "test": 0}
    assert manifest["roomgenbench_holdout_groups"] == sorted(ROOMGENBENCH_HOLDOUT_GROUPS)
    assert manifest["qualification_change_counts"]["provenance.split"] == 1
    assert manifest["parent_front_policy"] == "axis" and manifest["parent_yaw_policy"] == "axis policy"
    assert manifest["yaw_policy"].startswith("inherit parent yaw validity")
    moved = rows(tmp_path / "new/test.jsonl")[-1]
    assert moved["provenance"]["scene_id"] == "bench-train" and moved["provenance"]["split"] == "test"
    journal = rows(tmp_path / "new/changes.jsonl")
    assert journal[-1]["split"] == "train" and journal[-1]["parent_line"] == 2 and journal[-1]["field"] == "provenance.split"
    report = verify_dataset(parent, tmp_path / "new", tmp_path, expected_ir_sha256={}, full_condition=True)
    assert report["ok"], report["errors_first_100"]
    assert report["split_counts"]["test"]["samples"] == 3 and report["journal_entries_checked"] == 1
    with pytest.raises(ValueError, match="holdout"):
        verify_dataset(parent, tmp_path / "new", tmp_path, expected_ir_sha256={}, full_condition=True, holdout_groups=())
    stub = {"split_samples": {}, "qualification_change_counts": {}}
    with patch("fastfill.v2.qualified_data.build_dataset", return_value=stub) as build, redirect_stdout(StringIO()):
        qualified_main(["--parent-root", "p", "--output", "o", "--ir-root", "i", "--frozen-manifest", "m",
                        "--holdout-group", "other-validation", "--holdout-group", "x"])
        qualified_main(["--parent-root", "p", "--output", "o", "--ir-root", "i", "--frozen-manifest", "m"])
    assert build.call_args_list[0].kwargs["holdout_groups"] == ("other-validation", "x")
    assert build.call_args_list[1].kwargs["holdout_groups"] == ROOMGENBENCH_HOLDOUT_GROUPS


def test_minimal_view_build_and_independent_verifier_round_trip_with_holdout(tmp_path):
    from fastfill.v2.io import fingerprint
    parent, ir = dataset_files(tmp_path)
    row = json.loads((parent / "validation.jsonl").read_text())
    kwargs = {"expected_spatiallm_sha256": fingerprint(ir / "SpatialLM.jsonl"), "frozen_manifest": tmp_path / "frozen-manifest.json",
              "holdout_groups": (row["provenance"]["group"],)}
    manifest = build_minimal(parent, ir, tmp_path / "view", **kwargs)
    assert manifest["split_samples"] == {"train": 1, "test": 2}
    assert manifest["roomgenbench_holdout_groups"] == [row["provenance"]["group"]]
    seals = json.loads((tmp_path / "frozen-manifest.json").read_text())["ir_sha256"]
    report = verify_dataset(parent, tmp_path / "view", ir, expected_ir_sha256=seals, holdout_groups=kwargs["holdout_groups"])
    assert report["ok"], report["errors_first_100"]
    assert report["split_counts"]["test"]["samples"] == 2


def test_full_pipeline_bridge_review_qualify_verify(tmp_path):
    """New-format rows flow through every builder and the independent verifier end to end."""
    from fastfill.v2.io import fingerprint
    release, evidence = miniature_release(tmp_path)
    with redirect_stdout(StringIO()):
        build_selected_dataset(release, tmp_path / "bridge", evidence_root=evidence)
        review.build_reviewed(tmp_path / "bridge", tmp_path / "reviewed")
        reviewed = json.loads((tmp_path / "reviewed/manifest.json").read_text())
        assert reviewed["front_policy"] == "axis" and "axis" in reviewed["yaw_policy"]
        pins = {p.name: fingerprint(p) for p in (tmp_path / "reviewed").iterdir()}
        holdout = ("house:train-clean",)
        manifest = build_qualified(tmp_path / "reviewed", tmp_path / "main", expected_hashes=pins, holdout_groups=holdout)
    assert manifest["split_samples"] == {"validation": 1, "test": 2}
    assert manifest["source_yaw_symmetry_order_counts"] == {"SpatialLM:2": 6}
    report = verify_dataset(tmp_path / "reviewed", tmp_path / "main", tmp_path, expected_ir_sha256={}, full_condition=True,
                            holdout_groups=holdout)
    assert report["ok"], report["errors_first_100"]
    for split in ("validation", "test"):
        for row in rows(tmp_path / "main" / f"{split}.jsonl"):
            validate_condition(row["condition"])
            assert row["validity"]["yaw_symmetry_order"] == [2, 2]
            assert "exchangeable_group" in row["validity"]
