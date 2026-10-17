"""Parallel frozen migration must preserve serial bytes and source evidence."""
import pytest

from fastfill.v2.tests.test_legacy_build import assert_no_output_or_staging, build, miniature_release


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("include_flagged", [False, True])
def test_parallel_output_is_byte_identical_to_serial_and_inputs_unchanged(tmp_path, include_flagged):
    release, evidence = miniature_release(tmp_path)
    release_before, evidence_before = snapshot(release), snapshot(evidence)
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial_result = build(release, evidence, serial, workers=1, include_flagged=include_flagged)
    parallel_result = build(release, evidence, parallel, workers=2, include_flagged=include_flagged)
    assert serial_result["split_samples"] == parallel_result["split_samples"]
    assert serial_result["valid_label_counts"] == parallel_result["valid_label_counts"]
    for filename in ("train.jsonl", "validation.jsonl", "test.jsonl", "rejections.jsonl"):
        assert (parallel / filename).read_bytes() == (serial / filename).read_bytes()
    assert snapshot(release) == release_before
    assert snapshot(evidence) == evidence_before
    existing = snapshot(parallel)
    with pytest.raises(FileExistsError):
        build(release, evidence, parallel, workers=2, include_flagged=include_flagged)
    assert snapshot(parallel) == existing
    assert snapshot(release) == release_before
    assert snapshot(evidence) == evidence_before


@pytest.mark.parametrize("workers", [0, True, False, -1, 1.5, "2", None])
def test_workers_must_be_positive_integer_with_bool_rejected(tmp_path, workers):
    release, evidence = miniature_release(tmp_path)
    release_before, evidence_before = snapshot(release), snapshot(evidence)
    output = tmp_path / "parallel"
    with pytest.raises(ValueError, match="workers"):
        build(release, evidence, output, workers=workers)
    assert_no_output_or_staging(output)
    assert snapshot(release) == release_before
    assert snapshot(evidence) == evidence_before


def test_parallel_rejections_leave_no_partial_dataset_or_input_changes(tmp_path):
    release, evidence = miniature_release(tmp_path, tilted=True)
    release_before, evidence_before = snapshot(release), snapshot(evidence)
    output = tmp_path / "parallel"
    with pytest.raises(ValueError, match="no selected samples eligible"):
        build(release, evidence, output, workers=2)
    assert_no_output_or_staging(output)
    assert snapshot(release) == release_before
    assert snapshot(evidence) == evidence_before
