"""A reviewed revision may qualify legacy facing, never rewrite geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from fastfill.v2 import review_data as review


def sample(uid="scene", split="train", *, yaw=True, known_floor=True):
    return {"schema_version": "fastfill.v2", "condition": {
        "schema_version": "fastfill.v2", "room": {
            "frame": "right_handed_z_up", "floor_known": known_floor, "floor_z_m": 0.,
            "fixed_objects": [{"id": "fixed", "bottom_center_m": [0., 0., -2.]}]},
        "objects": [{"id": "chair", "category": "chair", "exchangeable_group": "chairs"},
                    {"id": "table", "category": "table"}],
        "constraints": [{"type": "faces", "object_id": "chair", "target_id": "table"},
                        {"type": "against_wall", "object_id": "table"}]},
        "target": {"schema_version": "fastfill.v2", "objects": [
            {"id": "chair", "target_size_local_m": [.5, .6, .8],
             "bottom_center_m": [1., 1., 0.], "yaw_rad": .2},
            {"id": "table", "target_size_local_m": [1., 1., .7],
             "bottom_center_m": [2., 1., 0.], "yaw_rad": 0.}]},
        "validity": {"position": [[True] * 3] * 2, "size": [[True] * 3] * 2, "yaw": [yaw, True]},
        "provenance": {"scene_id": uid, "legacy_uid": uid, "source": "AuditedSource",
            "split": split, "constraint_source": "frozen_sparse_legacy_request",
            "field_evidence": [{"recorded_source_evidence": {}}, {"recorded_source_evidence": {}}]}}


def line(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def dataset(root, rows=None):
    root.mkdir()
    rows = rows if rows is not None else {s: [sample(s, s)] for s in review.SPLITS}
    for split in review.SPLITS:
        (root / f"{split}.jsonl").write_bytes(b"".join(line(r) for r in rows.get(split, [])))
    (root / "manifest.json").write_bytes(line({"schema_version": "fastfill.v2",
        "builder": "selected-v3.2-bridge", "front_policy": "strict"}))
    (root / "rejections.jsonl").write_bytes(b'{"reason":"old rejection"}\n')
    return root


def hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in review.INPUT_FILES}


@pytest.mark.parametrize("yaw", [False, [], [False], [True, False], None])
def test_unverified_legacy_faces_are_omitted_without_mutation(yaw):
    parent = sample(yaw=yaw)
    before = deepcopy(parent)
    child = review.review_sample(parent)
    assert parent == before
    assert child["condition"]["constraints"] == parent["condition"]["constraints"][1:]
    omitted = child["provenance"]["review_omitted_constraints"]
    assert omitted[0]["constraint"] == parent["condition"]["constraints"][0]
    assert omitted[0]["reason"] == "unverified_source_derived_facing"
    for key in ("target", "validity"):
        assert child[key] == parent[key]
    assert child["condition"]["objects"] == parent["condition"]["objects"]
    assert child["provenance"]["review_parent_scene_id"] == "scene"
    assert "source-derived" in child["provenance"]["review_source_qualification"]


@pytest.mark.parametrize("yaw", [True, [True]])
def test_trusted_yaw_keeps_facing(yaw):
    parent = sample(yaw=yaw)
    child = review.review_sample(parent)
    assert child["condition"] == parent["condition"]
    assert child["provenance"]["review_omitted_constraints"] == []


@pytest.mark.parametrize("reason", ["no semantic front", "", None])
def test_recorded_unknown_front_overrules_true_yaw_mask(reason):
    parent = sample()
    parent["provenance"]["field_evidence"][0]["recorded_source_evidence"]["front_unknown_reason"] = reason
    assert len(review.review_sample(parent)["condition"]["constraints"]) == 1


def test_nonlegacy_user_requirement_is_kept_even_without_yaw_evidence():
    parent = sample(yaw=False)
    parent["provenance"]["constraint_source"] = "explicit_user_request"
    assert review.review_sample(parent)["condition"] == parent["condition"]


def test_absent_constraints_remain_absent():
    parent = sample()
    del parent["condition"]["constraints"]
    assert review.review_sample(parent)["condition"] == parent["condition"]


@pytest.mark.parametrize("corruption", ["condition", "target", "evidence", "recorded-evidence"])
def test_malformed_parent_structures_fail_closed(corruption):
    parent = sample()
    if corruption in ("condition", "target"):
        parent[corruption] = None
    elif corruption == "evidence":
        parent["provenance"]["field_evidence"][0] = None
    else:
        parent["provenance"]["field_evidence"][0]["recorded_source_evidence"] = None
    with pytest.raises(ValueError):
        review.review_sample(parent)


def test_target_id_mapping_is_independent_of_request_order():
    parent = sample(yaw=False)
    parent["target"]["objects"].reverse()
    parent["validity"]["yaw"].reverse()
    assert len(review.review_sample(parent)["condition"]["constraints"]) == 1


def test_floor_conflicts_are_only_diagnostics_not_geometry_edits():
    parent = sample()
    parent["target"]["objects"][0]["bottom_center_m"][2] = -.02
    child = review.review_sample(parent)
    diagnostic = child["provenance"]["review_geometry_diagnostics"]
    assert diagnostic["below_known_floor"] == [{"object_id": "chair", "bottom_minus_floor_m": -.02}]
    assert diagnostic["minimum_bottom_minus_floor_m"] == -.02
    assert child["target"] == parent["target"]
    assert child["condition"] == parent["condition"]


@pytest.mark.parametrize("mode", ["unknown", "missing", "partial-position", "boundary"])
def test_unknown_floor_partial_label_and_tolerance_are_not_false_conflicts(mode):
    parent = sample(known_floor=mode != "unknown")
    parent["target"]["objects"][0]["bottom_center_m"][2] = -.1
    if mode == "missing":
        del parent["condition"]["room"]["floor_known"]
    elif mode == "partial-position":
        parent["validity"]["position"][0] = [False, False, True]
    elif mode == "boundary":
        parent["target"]["objects"][0]["bottom_center_m"][2] = -1e-4
    assert review.review_sample(parent)["provenance"]["review_geometry_diagnostics"]["below_known_floor"] == []


def test_build_and_verify_preserve_all_rows_labels_splits_and_rejections(tmp_path):
    rows = {s: [sample(s + "-ok", s), sample(s + "-unknown", s, yaw=False)] for s in review.SPLITS}
    rows["test"][0]["target"]["objects"][0]["bottom_center_m"][2] = -.2
    source = dataset(tmp_path / "parent", rows)
    before = hashes(source)
    output = tmp_path / "reviewed"
    manifest = review.build_reviewed(source, output)
    assert manifest["front_policy"] == "strict"
    assert hashes(source) == before
    assert manifest["parent_sha256"] == before
    assert manifest["split_samples"] == dict.fromkeys(review.SPLITS, 2)
    assert manifest["split_objects"] == dict.fromkeys(review.SPLITS, 4)
    assert manifest["omitted_faces"] == 3 and manifest["rows_with_omitted_faces"] == 3
    assert manifest["floor_conflict_objects"] == 1 and manifest["rows_with_floor_conflicts"] == 1
    assert manifest["source_omitted_faces"] == {"AuditedSource": 3}
    assert manifest["source_floor_conflict_objects"] == {"AuditedSource": 1}
    assert (output / "rejections.jsonl").read_bytes() == (source / "rejections.jsonl").read_bytes()
    for split in review.SPLITS:
        observed = [json.loads(raw) for raw in (output / f"{split}.jsonl").read_bytes().splitlines()]
        assert observed == [review.review_sample(row) for row in rows[split]]
    verified = review.verify_reviewed(output, source)
    assert verified["verified"] is True and verified["samples_written"] == 6


@pytest.mark.parametrize("destination", ["exists", "child", "ancestor", "raw", "symlink"])
def test_build_never_overwrites_or_writes_inside_parent_or_raw_source(tmp_path, destination):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "new"
    if destination == "exists":
        output.mkdir()
    elif destination == "child":
        output = source / "child"
    elif destination == "ancestor":
        output = tmp_path
    elif destination == "raw":
        output = Path("/Volumes/harddisk/3D_Room_Collections/new-reviewed")
    else:
        output.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises((ValueError, FileExistsError)):
        review.build_reviewed(source, output)


def test_parent_change_aborts_before_atomic_publish(tmp_path):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "reviewed"
    original = review.review_sample
    def race(record):
        result = original(record)
        with (source / "rejections.jsonl").open("ab") as handle:
            handle.write(b"\n")
        return result
    with patch.object(review, "review_sample", race), pytest.raises(ValueError, match="changed"):
        review.build_reviewed(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".reviewed-*"))


@pytest.mark.parametrize("corruption", ["missing-file", "wrong-schema", "split", "duplicate-uid", "request-id", "nan"])
def test_invalid_parent_cannot_produce_a_reviewed_revision(tmp_path, corruption):
    source = dataset(tmp_path / "parent")
    if corruption == "missing-file":
        (source / "rejections.jsonl").unlink()
    elif corruption == "wrong-schema":
        (source / "manifest.json").write_bytes(line({"schema_version": "other"}))
    elif corruption == "nan":
        (source / "train.jsonl").write_text('{"bad":NaN}\n')
    else:
        row = sample()
        if corruption == "split":
            row["provenance"]["split"] = "test"
        elif corruption == "request-id":
            row["condition"]["objects"][0]["id"] = "different"
        else:
            row["provenance"]["legacy_uid"] = "validation"
        (source / "train.jsonl").write_bytes(line(row))
    with pytest.raises((ValueError, FileNotFoundError)):
        review.build_reviewed(source, tmp_path / "reviewed")
    assert not (tmp_path / "reviewed").exists()


@pytest.mark.parametrize("corruption", ["target", "validity", "condition", "extra-row", "missing-row", "manifest-count"])
def test_verifier_checks_deterministic_rows_even_when_output_hash_is_updated(tmp_path, corruption):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "reviewed"
    review.build_reviewed(source, output)
    path = output / "train.jsonl"
    row = json.loads(path.read_bytes())
    if corruption == "target":
        row["target"]["objects"][0]["bottom_center_m"][2] = -.5
    elif corruption == "validity":
        row["validity"]["yaw"][0] = False
    elif corruption == "condition":
        row["condition"]["constraints"].clear()
    if corruption in ("target", "validity", "condition"):
        path.write_bytes(line(row))
    elif corruption == "extra-row":
        path.write_bytes(line(row) * 2)
    elif corruption == "missing-row":
        path.write_bytes(b"")
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["output_sha256"]["train.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if corruption == "manifest-count":
        manifest["samples_written"] = 999
    (output / "manifest.json").write_bytes(line(manifest))
    with pytest.raises(ValueError):
        review.verify_reviewed(output, source)


def test_cli_and_import_use_no_model_dependencies(tmp_path):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "reviewed"
    probe = subprocess.run([sys.executable, "-c", "import sys; import fastfill.v2.review_data; "
        "assert 'torch' not in sys.modules and 'numpy' not in sys.modules"], check=True)
    assert probe.returncode == 0
    for args in (["build", "--parent-root", str(source), "--output", str(output)],
                 ["verify", "--parent-root", str(source), "--data-root", str(output)]):
        run = subprocess.run([sys.executable, "-m", "fastfill.v2.review_data", *args],
                             capture_output=True, text=True, check=True)
        assert json.loads(run.stdout)["samples_written"] == 3
