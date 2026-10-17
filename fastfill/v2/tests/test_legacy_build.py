"""Frozen release migration tests using real v1 preparation and saved messages."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from fastfill.build import prep
from fastfill.scene import canonical, messages
from fastfill.v2.legacy_build import build_selected_dataset, verify_release
from fastfill.v2.legacy_evidence import EvidenceIndex, FRONT_FILE, OPENINGS_FILE
from fastfill.v2.tests.test_legacy_bridge import fixture


V1_FILES = ("build.py", "scene.py", "anchors.py", "validate.py", "split.py")
PACKAGE = Path(__file__).resolve().parents[2]
ARGS = {"sources": ["SpatialLM"], "boundary_types": ["polygon", "hull"], "anchors": ["floor", "object"],
        "source_anchors": {}, "min_objects": 1, "max_vertices": 0, "oob_tol": .1,
        "hidden_max": .3, "reject_flagged": []}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def miniature_release(root, *, tilted=False):
    release, evidence = root / "release", root / "evidence"
    data, ir = release / "data/v3.2", release / "ir"
    data.mkdir(parents=True)
    ir.mkdir()
    raw_rows, saved_rows = [], {"train": [], "dev": [], "test": []}
    for split, suffix, flagged in (("train", "train-clean", False), ("train", "train-flagged", True),
                                   ("dev", "dev-flagged", True), ("test", "test-flagged", True)):
        raw = {**fixture(), "uid": f"SpatialLM:{suffix}", "group": f"house:{suffix}"}
        if flagged:
            raw["objects"] = [{**obj, "pos": [6., 3., 0.]} if i == 0 else obj
                              for i, obj in enumerate(raw["objects"])]
        if tilted:
            raw["objects"] = [{**obj, "tilted": True} for obj in raw["objects"]]
        prepared, reason = prep(deepcopy(raw), SimpleNamespace(**ARGS))
        assert reason is None, reason
        row = {"uid": raw["uid"], "source": raw["source"], "flags": prepared["meta"]["flags"],
               "messages": messages(canonical(prepared))}
        assert bool(row["flags"].get("oob_objects")) == flagged
        raw_rows.append(raw)
        saved_rows[split].append(row)
    for split, rows in saved_rows.items():
        (data / f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    (ir / "SpatialLM.jsonl").write_text("".join(json.dumps(row)+"\n" for row in raw_rows))
    manifest = {"args": ARGS, "files": {f"{split}.jsonl": {"sha256": sha256(data / f"{split}.jsonl")} for split in saved_rows},
                "ir_sha256": {"SpatialLM.jsonl": sha256(ir / "SpatialLM.jsonl")},
                "code_sha256": {name: sha256(PACKAGE / name) for name in V1_FILES}}
    (data / "MANIFEST.json").write_text(json.dumps(manifest))
    openings, fronts = evidence / OPENINGS_FILE, evidence / FRONT_FILE
    openings.parent.mkdir(parents=True)
    fronts.parent.mkdir(parents=True)
    openings.write_text(json.dumps({"saved_hits": []}))
    fronts.write_text("")
    return release, evidence


def build(release, evidence, output, **kwargs):
    with redirect_stdout(StringIO()):
        return build_selected_dataset(release, output, evidence_root=evidence, **kwargs)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rewrite_ir_manifest(release, records):
    ir = release / "ir/SpatialLM.jsonl"
    ir.write_text("".join(json.dumps(row)+"\n" for row in records))
    path = release / "data/v3.2/MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["ir_sha256"][ir.name] = sha256(ir)
    path.write_text(json.dumps(manifest))


def assert_no_output_or_staging(output):
    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}-*"))


def test_verifies_real_v1_hashes_and_preserves_frozen_split_assignments(tmp_path):
    release, evidence = miniature_release(tmp_path)
    manifest, inputs = verify_release(release)
    before = {path: Path(path).read_bytes() for path in inputs}
    result = build(release, evidence, tmp_path / "dataset")
    assert result["split_samples"] == {"train": 1, "validation": 1, "test": 1}
    assert result["legacy_train_filtered_by_source"] == {"SpatialLM": 1}
    expected = {"train": "SpatialLM:train-clean", "validation": "SpatialLM:dev-flagged", "test": "SpatialLM:test-flagged"}
    for split, uid in expected.items():
        records = rows(tmp_path / "dataset" / f"{split}.jsonl")
        assert len(records) == 1
        assert records[0]["provenance"]["legacy_uid"] == uid
        assert records[0]["provenance"]["split"] == split
        if split != "train":
            assert records[0]["provenance"]["legacy_flags"]["oob_objects"]
    assert result["source_hashes"] == inputs
    assert {path: Path(path).read_bytes() for path in inputs} == before
    assert manifest["code_sha256"]["build.py"] == sha256(PACKAGE / "build.py")


def test_output_rows_are_accepted_by_public_v2_reader(tmp_path):
    from fastfill.v2.io import read_samples
    release, evidence = miniature_release(tmp_path)
    build(release, evidence, tmp_path / "dataset")
    for split in ("train", "validation", "test"):
        records = read_samples(tmp_path / "dataset" / f"{split}.jsonl", training=split == "train")
        assert records[0]["schema_version"] == "fastfill.v2"


def test_include_flagged_retains_train_without_resplitting_holdout(tmp_path):
    release, evidence = miniature_release(tmp_path)
    result = build(release, evidence, tmp_path / "dataset", include_flagged=True, seed=987)
    assert result["split_samples"] == {"train": 2, "validation": 1, "test": 1}
    assert result["legacy_train_filtered_by_source"] == {}
    assert {row["provenance"]["legacy_uid"] for row in rows(tmp_path / "dataset/train.jsonl")} == {
        "SpatialLM:train-clean", "SpatialLM:train-flagged"}
    assert rows(tmp_path / "dataset/validation.jsonl")[0]["provenance"]["split"] == "validation"


def test_existing_output_cannot_be_overwritten(tmp_path):
    release, evidence = miniature_release(tmp_path)
    output = tmp_path / "dataset"
    build(release, evidence, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        build(release, evidence, output, seed=999)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_changed_frozen_input_hash_is_rejected_without_final_output(tmp_path):
    release, evidence = miniature_release(tmp_path)
    path = release / "data/v3.2/train.jsonl"
    path.write_text(path.read_text()+"\n")
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="hash mismatch"):
        build(release, evidence, output)
    assert_no_output_or_staging(output)


def test_wrong_legacy_code_hash_is_rejected(tmp_path):
    release, _ = miniature_release(tmp_path)
    path = release / "data/v3.2/MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["code_sha256"]["scene.py"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="code differs"):
        verify_release(release)


@pytest.mark.parametrize("repeat", [False, True])
def test_missing_or_repeated_selected_uid_aborts_atomic_output(tmp_path, repeat):
    release, evidence = miniature_release(tmp_path)
    records = rows(release / "ir/SpatialLM.jsonl")
    rewrite_ir_manifest(release, records + records[-1:] if repeat else records[:-1])
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="missing or repeated"):
        build(release, evidence, output)
    assert_no_output_or_staging(output)


def test_input_change_during_migration_aborts_atomic_output(tmp_path):
    release, evidence = miniature_release(tmp_path)
    original = EvidenceIndex.apply_evidence
    path = release / "data/v3.2/train.jsonl"
    def concurrent_change(index, raw):
        corrected = original(index, raw)
        path.write_text(path.read_text()+"\n")
        return corrected
    output = tmp_path / "dataset"
    with patch.object(EvidenceIndex, "apply_evidence", concurrent_change), pytest.raises(ValueError, match="changed during migration"):
        build(release, evidence, output)
    assert_no_output_or_staging(output)


def test_all_unreliable_fields_leave_no_partial_dataset(tmp_path):
    release, evidence = miniature_release(tmp_path, tilted=True)
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="no selected samples eligible"):
        build(release, evidence, output)
    assert_no_output_or_staging(output)


def test_missing_required_evidence_is_rejected_before_output(tmp_path):
    release, evidence = miniature_release(tmp_path)
    (evidence / FRONT_FILE).unlink()
    output = tmp_path / "dataset"
    with pytest.raises(FileNotFoundError, match="evidence missing"):
        build(release, evidence, output)
    assert_no_output_or_staging(output)


def test_data_cli_defaults_to_frozen_selected_release_not_raw_multiscan(tmp_path):
    from fastfill.v2.data import main
    from fastfill.v2.legacy_build import DEFAULT_EVIDENCE, DEFAULT_RELEASE
    output = tmp_path / "new-dataset"
    result = {"samples_written": 3, "split_samples": {"train": 1, "validation": 1, "test": 1}}
    stdout = StringIO()
    with patch("fastfill.v2.legacy_build.build_selected_dataset", return_value=result) as selected, \
            patch("fastfill.v2.data.build_dataset", side_effect=AssertionError("default must not select raw MultiScan")), \
            redirect_stdout(stdout):
        main(["--output", str(output)])
    selected.assert_called_once_with(DEFAULT_RELEASE, output, evidence_root=DEFAULT_EVIDENCE,
                                     seed=42, max_scenes=None, front_policy="axis", include_flagged=False, workers=1)
    assert json.loads(stdout.getvalue()) == result


def test_data_cli_forwards_selected_release_evidence_and_policy(tmp_path):
    from fastfill.v2.data import main
    release, evidence, output = tmp_path / "frozen", tmp_path / "audit", tmp_path / "new-dataset"
    result = {"samples_written": 2, "split_samples": {"train": 2}}
    with patch("fastfill.v2.legacy_build.build_selected_dataset", return_value=result) as selected, \
            patch("fastfill.v2.data.build_dataset", side_effect=AssertionError("selected release is independent of raw source root")), \
            redirect_stdout(StringIO()):
        main(["--output", str(output), "--release-root", str(release), "--evidence-root", str(evidence),
              "--seed", "17", "--max-scenes", "2", "--front-policy", "legacy-convention", "--include-flagged", "--workers", "2"])
    selected.assert_called_once_with(release, output, evidence_root=evidence, seed=17, max_scenes=2,
                                     front_policy="legacy-convention", include_flagged=True, workers=2)


def test_data_cli_keeps_explicit_multiscan_compatibility(tmp_path):
    from fastfill.v2.data import main
    source, output = tmp_path / "raw-multiscan", tmp_path / "new-dataset"
    result = {"samples_written": 4, "split_samples": {"train": 4}}
    with patch("fastfill.v2.data.build_dataset", return_value=result) as multiscan, \
            patch("fastfill.v2.legacy_build.build_selected_dataset", side_effect=AssertionError("explicit MultiScan must not read frozen release")), \
            redirect_stdout(StringIO()):
        main(["--source", "multiscan", "--source-root", str(source), "--output", str(output),
              "--seed", "99", "--max-scenes", "4"])
    multiscan.assert_called_once_with(source, output, sources=("multiscan",), seed=99, max_scenes=4)
