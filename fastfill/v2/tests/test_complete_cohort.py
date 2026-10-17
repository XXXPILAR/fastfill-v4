"""A fair complete-label comparison cohort must not rewrite parent samples."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastfill.v2 import cohort


SPLITS = ("train", "validation", "test")


def sample(uid, split="train", *, complete=True):
    return {"schema_version": "fastfill.v2", "condition": {
        "schema_version": "fastfill.v2", "room": {
            "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "floor_z_m": 0., "height_m": None},
        "objects": [{"id": "chair", "category": "chair", "description": "chair 🪑"}],
        "constraints": []}, "target": {"schema_version": "fastfill.v2", "objects": [{
            "id": "chair", "target_size_local_m": [0.5, 0.6, 0.8],
            "bottom_center_m": [1.0, 2.0, 0.0], "yaw_rad": 0.2}]},
        "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [complete]},
        "provenance": {"legacy_uid": uid, "source": "MultiScan", "split": split}}


def line(record):
    return (" \t" + json.dumps(record, ensure_ascii=False, separators=(", ", ": ")) + "  \r\n").encode()


def dataset(root, rows=None):
    root.mkdir()
    rows = rows or {split: [sample(split, split)] for split in SPLITS}
    for split in SPLITS:
        (root / f"{split}.jsonl").write_bytes(b"".join(line(r) for r in rows.get(split, [])))
    (root / "manifest.json").write_text(json.dumps({"schema_version": "fastfill.v2",
        "builder": "selected-v3.2-bridge", "front_policy": "strict", "split_samples": {
            split: len(rows.get(split, [])) for split in SPLITS}}) + "\n")
    return root


def assert_absent(output):
    assert not output.exists()
    assert not tuple(output.parent.glob(f".{output.name}-*"))


def test_preserves_eligible_bytes_order_and_original_split(tmp_path):
    rows = {split: [sample(split + "-first", split), sample(split + "-partial", split, complete=False),
                    sample(split + "-last", split)] for split in SPLITS}
    source = dataset(tmp_path / "parent", rows)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "complete"
    result = cohort.build_complete_cohort(source, output)
    for split in SPLITS:
        assert (output / f"{split}.jsonl").read_bytes() == line(rows[split][0]) + line(rows[split][2])
    assert result["split_samples"] == dict.fromkeys(SPLITS, 2)
    assert result["split_skipped"] == dict.fromkeys(SPLITS, 1)
    assert result["samples_written"] == 6
    assert result["parent_hashes"] == {str((source / name).resolve()): hashlib.sha256(blob).hexdigest()
                                      for name, blob in before.items()}
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    assert json.loads((output / "manifest.json").read_text()) == result


def test_one_incomplete_object_rejects_whole_scene_not_only_that_slot(tmp_path):
    partial = sample("two-objects")
    partial["condition"]["objects"].append({"id": "table", "category": "table", "description": "table"})
    partial["target"]["objects"].append({"id": "table", "target_size_local_m": None,
        "bottom_center_m": [2., 2., 0.], "yaw_rad": 0.})
    partial["validity"] = {"position": [[True] * 3] * 2,
        "size": [[True] * 3, [False] * 3], "yaw": [True, True]}
    source = dataset(tmp_path / "parent", {"train": [partial, sample("complete")]})
    result = cohort.build_complete_cohort(source, tmp_path / "complete")
    assert result["split_samples"]["train"] == 1
    assert result["split_skipped"]["train"] == 1
    assert (tmp_path / "complete/train.jsonl").read_bytes() == line(sample("complete"))


@pytest.mark.parametrize("corruption", ["missing-id", "extra-id", "duplicate-id", "zero-size", "unwrapped-yaw"])
def test_invalid_complete_geometry_aborts_without_output(tmp_path, corruption):
    invalid = sample("invalid")
    targets = invalid["target"]["objects"]
    if corruption == "missing-id":
        targets.clear()
    elif corruption == "extra-id":
        targets.append({**targets[0], "id": "extra"})
    elif corruption == "duplicate-id":
        targets.append(deepcopy(targets[0]))
    elif corruption == "zero-size":
        targets[0]["target_size_local_m"][0] = 0.
    else:
        targets[0]["yaw_rad"] = 10.
    source = dataset(tmp_path / "parent", {"train": [sample("good"), invalid]})
    output = tmp_path / "complete"
    with pytest.raises(ValueError):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_floor_condition_conflict_is_rejected_by_geometry_preflight(tmp_path):
    invalid = sample("floating-floor")
    invalid["condition"]["objects"][0]["support_parent"] = "floor"
    invalid["target"]["objects"][0]["bottom_center_m"][2] = .2
    source = dataset(tmp_path / "parent", {"train": [invalid]})
    output = tmp_path / "complete"
    with pytest.raises(ValueError, match="floor support condition conflicts"):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_no_eligible_training_rows_aborts_even_with_valid_holdout(tmp_path):
    source = dataset(tmp_path / "parent", {"train": [sample("partial", complete=False)],
        "validation": [sample("validation", "validation")], "test": [sample("test", "test")]})
    output = tmp_path / "complete"
    with pytest.raises(ValueError, match="no eligible training"):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_existing_output_is_never_replaced(tmp_path):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "complete"
    output.mkdir()
    (output / "mine.txt").write_text("preserve")
    with pytest.raises(FileExistsError):
        cohort.build_complete_cohort(source, output)
    assert (output / "mine.txt").read_text() == "preserve"


@pytest.mark.parametrize("destination", ["child", "ancestor", "protected-source"])
def test_output_location_guards(tmp_path, destination):
    source = dataset(tmp_path / "parent")
    output = source / "child" if destination == "child" else tmp_path
    if destination == "protected-source":
        output = Path("/Volumes/harddisk/3D_Room_Collections/new-v2-cohort")
    with pytest.raises(ValueError, match="outside source data"):
        cohort.build_complete_cohort(source, output)


@pytest.mark.parametrize("filename", ["train.jsonl", "manifest.json"])
def test_parent_change_during_filter_aborts_atomic_publication(tmp_path, filename):
    source = dataset(tmp_path / "parent")
    original = cohort._geometry_rows
    def concurrent_change(*args):
        value = original(*args)
        with (source / filename).open("ab") as handle:
            handle.write(b"\n")
        return value
    output = tmp_path / "complete"
    with patch.object(cohort, "_geometry_rows", concurrent_change), pytest.raises(ValueError, match="changed"):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_missing_parent_manifest_fails_before_creating_staging(tmp_path):
    source = dataset(tmp_path / "parent")
    (source / "manifest.json").unlink()
    output = tmp_path / "complete"
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_manifest_snapshot_hash_matches_the_decoded_parent_metadata(tmp_path):
    source = dataset(tmp_path / "parent")
    original = cohort._digest
    manifest_path = source / "manifest.json"
    changed = False
    def concurrent_manifest_read(path):
        nonlocal changed
        if Path(path) == source / "train.jsonl" and not changed:
            changed = True
            replacement = json.loads(manifest_path.read_text())
            replacement["builder"] = "changed-parent"
            manifest_path.write_text(json.dumps(replacement))
        return original(path)
    output = tmp_path / "complete"
    with patch.object(cohort, "_digest", concurrent_manifest_read), pytest.raises(ValueError, match="changed"):
        cohort.build_complete_cohort(source, output)
    assert_absent(output)


def test_cli_builds_same_controlled_cohort(tmp_path, capsys):
    source = dataset(tmp_path / "parent")
    output = tmp_path / "complete"
    cohort.main(["--data-root", str(source), "--output", str(output)])
    assert json.loads(capsys.readouterr().out)["split_samples"] == dict.fromkeys(SPLITS, 1)
    assert (output / "train.jsonl").read_bytes() == (source / "train.jsonl").read_bytes()
