import hashlib
import json

import pytest

from fastfill.v2.verify_bundle import verify_bundle


def bundle(tmp_path):
    data = tmp_path / "data/train.jsonl"
    data.parent.mkdir()
    data.write_text('one immutable row\n')
    entry = {"sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "bytes": data.stat().st_size}
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": "fastfill.bundle.v1", "files": {"data/train.jsonl": entry}}))
    return data


def test_verify_accepts_valid_bundle_and_generated_outputs_only(tmp_path):
    data = bundle(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/new-checkpoint").write_text("runtime output")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/cached.pyc").write_bytes(b"cached")
    result = verify_bundle(tmp_path)
    assert result["ok"] is True
    assert result["files_checked"] == 1
    assert result["bytes_checked"] == data.stat().st_size


def test_tampering_or_missing_files_fail(tmp_path):
    data = bundle(tmp_path)
    data.write_text("same bytes not assured")
    with pytest.raises(ValueError, match="mismatch"):
        verify_bundle(tmp_path)
    data.unlink()
    with pytest.raises(FileNotFoundError):
        verify_bundle(tmp_path)


def test_extra_frozen_file_fails(tmp_path):
    bundle(tmp_path)
    (tmp_path / "data/unlisted.jsonl").write_text("extra")
    with pytest.raises(ValueError, match="unlisted"):
        verify_bundle(tmp_path)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "data/../../outside", "data\\outside"])
def test_inventory_escape_is_rejected(tmp_path, path):
    bundle(tmp_path)
    (tmp_path / "BUNDLE_MANIFEST.json").write_text(json.dumps({
        "schema_version": "fastfill.bundle.v1", "files": {path: {"sha256": "0"*64, "bytes": 0}}}))
    with pytest.raises(ValueError, match="path"):
        verify_bundle(tmp_path)


def test_symlink_file_cannot_replace_inventory_entry(tmp_path):
    data = bundle(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.write_bytes(data.read_bytes())
    data.unlink()
    data.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        verify_bundle(tmp_path)
