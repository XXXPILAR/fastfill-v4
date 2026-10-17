"""Text baseline provenance and full-cohort context preflight regressions."""
import json

import pytest

from fastfill.v2 import text_sft
from fastfill.v2.io import fingerprint
from fastfill.v2.tests.test_execution import sample


def _input(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(sample()) + "\n")
    return path


def _args(path, output):
    return ["--data", str(path), "--output", str(output), "--backbone", "tiny",
            "--dry-run", "--batch-size", "1", "--hidden-size", "16"]


def _change(path):
    row = json.loads(path.read_text())
    row = {**row, "target": {**row["target"], "objects": [{
        **row["target"]["objects"][0], "target_size_local_m": [2., 2., 2.]}]}}
    path.write_text(json.dumps(row) + "\n")


def test_text_context_preflight_checks_every_row_before_model_loading(tmp_path, monkeypatch):
    path = _input(tmp_path)
    row = sample()
    long = {**row, "condition": {**row["condition"], "objects": [{
        **row["condition"]["objects"][0], "description": "chair " * 2000}]}}
    path.write_text(json.dumps(row) + "\n" + json.dumps(long) + "\n")
    def forbidden_load(*args, **kwargs):
        raise AssertionError("model must not load before every row passes context preflight")
    monkeypatch.setattr(text_sft, "build_text_model", forbidden_load)
    with pytest.raises(ValueError, match="context budget"):
        text_sft.main(_args(path, tmp_path / "run") + ["--max-length", "1024"])
    assert not (tmp_path / "run").exists()


def test_text_rejects_source_changed_during_read_before_model_loading(tmp_path, monkeypatch):
    path = _input(tmp_path)
    original_read = text_sft._read_samples
    def read(source):
        result = original_read(source)
        _change(source)
        return result
    def forbidden_load(*args, **kwargs):
        raise AssertionError("edited source must fail before model loading")
    monkeypatch.setattr(text_sft, "_read_samples", read)
    monkeypatch.setattr(text_sft, "build_text_model", forbidden_load)
    with pytest.raises((ValueError, RuntimeError), match="changed|fingerprint|hash"):
        text_sft.main(_args(path, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_text_rejects_source_changed_during_training_before_final_publication(tmp_path, monkeypatch):
    path = _input(tmp_path)
    original_loss = text_sft.text_loss
    def loss(*args):
        result = original_loss(*args)
        _change(path)
        return result
    monkeypatch.setattr(text_sft, "text_loss", loss)
    with pytest.raises((ValueError, RuntimeError), match="changed|fingerprint|hash"):
        text_sft.main(_args(path, tmp_path / "run"))
    assert not (tmp_path / "run").exists()


def test_text_manifest_records_start_fingerprint_before_read(tmp_path, monkeypatch):
    path = _input(tmp_path)
    expected_hash = fingerprint(path)
    events = []
    original_metadata, original_read = text_sft.run_metadata, text_sft._read_samples
    def metadata(*args):
        events.append("metadata")
        return original_metadata(*args)
    def read(*args):
        events.append("read")
        return original_read(*args)
    monkeypatch.setattr(text_sft, "run_metadata", metadata)
    monkeypatch.setattr(text_sft, "_read_samples", read)
    text_sft.main(_args(path, tmp_path / "run"))
    config = json.loads((tmp_path / "run/text_config.json").read_text())
    assert config["run_metadata"]["data_sha256"] == expected_hash
    assert events == ["metadata", "read"]
