"""Independent streaming acceptance of immutable selected-corpus migrations."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastfill.v2.tests.test_legacy_build import build, miniature_release, rows
from fastfill.v2.legacy_verify import main, verify_selected_dataset


def corpus(tmp_path, **kwargs):
    release, evidence = miniature_release(tmp_path)
    output = tmp_path / "dataset"
    build(release, evidence, output, front_policy="legacy-convention", **kwargs)
    return output, release, evidence


def replace_rows(path, records):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))


def mutate_sample(output, split, change):
    path = output / f"{split}.jsonl"
    records = rows(path)
    change(records[0])
    replace_rows(path, records)


def assert_failed(report, text):
    assert not report["passed"]
    assert any(text in error["message"] for error in report["errors"]), report["errors"]


def add_detail_counts(manifest, sample, sign):
    """Add (+1) or remove (-1) one sample's yaw / symmetry / group contributions to the recomputed manifest keys."""
    source, validity = sample["provenance"]["source"], sample["validity"]
    manifest["source_yaw_valid_objects"][source] += sign * sum(map(bool, validity["yaw"]))
    for order in validity["yaw_symmetry_order"]:
        manifest["source_yaw_symmetry_order_counts"][f"{source}:{order}"] += sign
    groups = [g for g in validity["exchangeable_group"] if g is not None]
    manifest["exchangeable_group_counts"]["exchangeable_groups"] += sign * len(set(groups))
    manifest["exchangeable_group_counts"]["exchangeable_members"] += sign * len(groups)
    if any(validity["size_axis_swap_allowed"]):
        manifest["source_size_axis_swap_allowed_objects"][source] += sign * sum(validity["size_axis_swap_allowed"])
    manifest["floor_declarations_written"] += sign * sum(o.get("support_parent") == "floor" for o in sample["condition"]["objects"])
    manifest["floor_declaration_z_snapped"] += sign * sum(e["legacy_z_snap_applied_to_target"] for e in sample["provenance"]["field_evidence"])
    manifest["wall_declarations_written"] += sign * sum(o.get("support_parent") == "wall" for o in sample["condition"]["objects"])
    for e in sample["provenance"]["field_evidence"]:
        manifest["selection_rule_counts"][e["selection_rule"]] = manifest["selection_rule_counts"].get(e["selection_rule"], 0) + sign


def test_complete_corpus_counts_and_saved_filter_are_recomputed(tmp_path):
    output, _, _ = corpus(tmp_path)
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]
    assert report["split_counts"]["train"] == {
        "scenes": 1, "targets": 2, "position": 2, "size": 2, "yaw": 2,
        "full_geometry": 2, "full_geometry_scenes": 1, "text_eligible_scenes": 1,
        "learnable_scenes": 1, "zero_supervision_scenes": 0,
    }
    assert report["source_split_counts"]["SpatialLM"]["validation"]["targets"] == 2
    assert report["selection_accounting"] == {
        "selected": 4, "filtered_train": 1, "candidates": 3,
        "kept": 3, "rejected": 0, "unaccounted": 0,
    }
    assert report["hash_checks"]["source"]["checked"] == 4
    assert report["hash_checks"]["evidence"]["checked"] == 2
    assert report["hash_checks"]["implementation"]["checked"] == 3
    assert set(report["dataset_sha256"]) == {
        "manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl", "rejections.jsonl"}


def test_streams_rows_without_read_text_tokenization_or_batches(tmp_path):
    output, _, _ = corpus(tmp_path)
    original = Path.read_text
    def no_jsonl_read_text(path, *args, **kwargs):
        assert path.suffix != ".jsonl", "JSONL must be streamed"
        return original(path, *args, **kwargs)
    with patch.object(Path, "read_text", no_jsonl_read_text), \
            patch("fastfill.v2.batch.tokenize_condition", side_effect=AssertionError("no tokenizer")), \
            patch("fastfill.v2.batch.collate_samples", side_effect=AssertionError("no tensor batch")):
        report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]


def failed_worker(_job):
    raise RuntimeError("injected worker failure before publication")


def test_real_parallel_worker_failure_never_publishes_partial_dataset(tmp_path):
    release, evidence = miniature_release(tmp_path)
    output = tmp_path / "dataset"
    with patch("fastfill.v2.legacy_build._worker_job", failed_worker):
        with pytest.raises(RuntimeError, match="injected worker failure"):
            build(release, evidence, output, workers=2)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".dataset-*"))


@pytest.mark.parametrize("change,expected", [
    (lambda row: row.update(schema_version="wrong"), "schema_version"),
    (lambda row: row["target"]["objects"].append(deepcopy(row["target"]["objects"][0])), "target IDs"),
    (lambda row: row["target"]["objects"].pop(), "target IDs"),
    (lambda row: row["validity"]["position"].pop(), "validity position"),
    (lambda row: row["target"]["objects"][0]["target_size_local_m"].__setitem__(0, True), "coordinates"),
    (lambda row: row["target"]["objects"][0].update(yaw_rad=3.141592653589793), "wrapped"),
    (lambda row: row["provenance"].update(split="test"), "provenance split"),
    (lambda row: row["provenance"].update(source="SceneSmith"), "test-only"),
])
def test_schema_geometry_and_identity_tampering_is_rejected(tmp_path, change, expected):
    output, _, _ = corpus(tmp_path)
    mutate_sample(output, "train", change)
    assert_failed(verify_selected_dataset(output), expected)


def test_duplicate_uid_across_output_rows_is_rejected(tmp_path):
    output, _, _ = corpus(tmp_path)
    path = output / "train.jsonl"
    replace_rows(path, rows(path) * 2)
    assert_failed(verify_selected_dataset(output), "duplicate UID")


@pytest.mark.parametrize("alias", [False, True])
def test_underlying_house_and_global_visit_aliases_cannot_cross_splits(tmp_path, alias):
    output, _, _ = corpus(tmp_path)
    def change(row):
        if alias:
            row["provenance"]["source_meta"]["group_aliases"] = ["same-visit"]
        else:
            row["provenance"]["house_id"] = "same-house"
    for split in ("train", "validation"):
        mutate_sample(output, split, change)
    assert_failed(verify_selected_dataset(output), "split leakage")


def test_removed_scene_is_not_hidden_by_updating_manifest_counts(tmp_path):
    output, _, _ = corpus(tmp_path)
    (output / "test.jsonl").write_text("")
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["samples_written"] = 2
    manifest["split_samples"].pop("test")
    manifest["source_split_samples"].pop("SpatialLM:test")
    for field in ("targets", "position", "size", "yaw", "full_geometry"):
        manifest["valid_label_counts"][field] -= 2
    path.write_text(json.dumps(manifest))
    report = verify_selected_dataset(output)
    assert_failed(report, "unaccounted selected UID")
    assert report["selection_accounting"]["unaccounted"] == 1


def test_explicit_rejection_completes_selected_uid_accounting(tmp_path):
    output, _, _ = corpus(tmp_path)
    sample = rows(output / "test.jsonl")[0]
    (output / "test.jsonl").write_text("")
    replace_rows(output / "rejections.jsonl", [{"uid": sample["provenance"]["legacy_uid"],
        "source": "SpatialLM", "split": "test", "reason": "fixture audited exclusion"}])
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["samples_written"] = 2
    manifest["split_samples"].pop("test")
    manifest["source_split_samples"].pop("SpatialLM:test")
    for field in ("targets", "position", "size", "yaw", "full_geometry"):
        manifest["valid_label_counts"][field] -= 2
    add_detail_counts(manifest, sample, -1)
    manifest["v2_rejections"] = {"fixture audited exclusion": 1}
    path.write_text(json.dumps(manifest))
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]
    assert report["selection_accounting"]["rejected"] == 1
    assert report["selection_accounting"]["unaccounted"] == 0


def test_default_training_flags_cannot_be_removed_from_provenance(tmp_path):
    output, _, _ = corpus(tmp_path)
    mutate_sample(output, "train", lambda row: row["provenance"].update(legacy_flags={"oob_objects": True}))
    assert_failed(verify_selected_dataset(output), "legacy flags differ")


def test_boolean_manifest_count_is_not_an_integer_count(tmp_path):
    output, _, _ = corpus(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split_samples"]["train"] = True
    path.write_text(json.dumps(manifest))
    assert_failed(verify_selected_dataset(output), "split_samples")


def test_include_flagged_is_explicit_and_bounded_coverage_is_not_claimed(tmp_path):
    output, _, _ = corpus(tmp_path, include_flagged=True, max_scenes=2)
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]
    assert report["bounded_build"]
    assert not report["complete_selection_coverage"]
    assert report["legacy_train_flag_filter"] == []
    assert report["selection_accounting"]["filtered_train"] == 0
    assert report["selection_accounting"]["unaccounted"] == 2


@pytest.mark.parametrize("kind", ["source", "evidence", "manifest_count"])
def test_hashes_and_manifest_counts_are_independently_checked(tmp_path, kind):
    output, release, evidence = corpus(tmp_path)
    if kind == "source":
        path = release / "ir/SpatialLM.jsonl"
        path.write_text(path.read_text() + "\n")
    elif kind == "evidence":
        path = evidence / "internscenes/k0_saved_object_exposure.jsonl"
        path.write_text("\n")
    else:
        path = output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["valid_label_counts"]["targets"] += 1
        path.write_text(json.dumps(manifest))
    assert_failed(verify_selected_dataset(output), "hash mismatch" if kind != "manifest_count" else "valid_label_counts")


def test_masked_geometry_counts_text_subset_and_constraint_evidence(tmp_path):
    output, _, _ = corpus(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    def change(row):
        add_detail_counts(manifest, row, -1)
        row["validity"]["yaw"][0] = False
        # Groups live in validity; the near constraint below would break their certification.
        row["validity"]["exchangeable_group"] = [None] * len(row["target"]["objects"])
        row["condition"]["constraints"] = [{"type": "near", "object_id": "obj_0000",
                                               "target_id": "obj_0001", "max_distance_m": 3.}]
        row["provenance"]["omitted_legacy_constraints"] = [
            {"legacy_constraint": ["on", "chair_1", "chair_2"], "reason": "inferred"}]
        row["provenance"]["source_meta"]["v2_evidence"] = {
            "opening_width_corrections": 2, "front_unknown_objects": 1}
        add_detail_counts(manifest, row, 1)
    mutate_sample(output, "train", change)
    manifest["valid_label_counts"]["yaw"] -= 1
    manifest["valid_label_counts"]["full_geometry"] -= 1
    path.write_text(json.dumps(manifest))
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]
    assert report["split_counts"]["train"]["full_geometry"] == 1
    assert report["split_counts"]["train"]["full_geometry_scenes"] == 0
    assert report["split_counts"]["train"]["text_eligible_scenes"] == 0
    assert report["kept_legacy_constraint_types"] == {"near": 1}
    assert report["omitted_legacy_constraint_types"] == {"on": 1}
    assert report["source_corrections"]["opening_width_corrections"] == 2
    assert report["source_corrections"]["front_unknown_objects"] == 1


def test_new_safe_report_and_cli_exit_status(tmp_path):
    output, release, evidence = corpus(tmp_path)
    report_path = tmp_path / "reports/verified.json"
    with redirect_stdout(StringIO()):
        assert main(["--data-root", str(output), "--output", str(report_path)]) == 0
    assert json.loads(report_path.read_text())["passed"]
    with pytest.raises(FileExistsError):
        verify_selected_dataset(output, output=report_path)
    for protected in (output / "report.json", release / "report.json", evidence / "report.json"):
        with pytest.raises(ValueError, match="outside"):
            verify_selected_dataset(output, output=protected)
        assert not protected.exists()
    mutate_sample(output, "train", lambda row: row["provenance"].update(split="test"))
    with redirect_stdout(StringIO()):
        assert main(["--data-root", str(output), "--output", str(tmp_path / "failed.json")]) == 1
