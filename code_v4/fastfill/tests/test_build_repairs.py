"""Build failures must never replace a published dataset."""
import argparse
import copy
import json
import sys

import pytest

from fastfill import build


@pytest.fixture
def room():
    return {"uid": "fixture:one", "source": "Fixture", "group": "fixture:one",
            "meta": {"front_known": True}, "boundary_type": "polygon",
            "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]], "height": 3,
            "objects": [{"id": "a", "category": "chair", "size": [1, 1, 1],
                         "pos": [2, 2, 0], "yaw": 0, "anchor": "floor", "parent": None}]}


def invoke(monkeypatch, ir, out, *extra):
    monkeypatch.setattr(sys, "argv", ["build", "--ir", str(ir), "--out", str(out), *extra])
    build.main()


def inputs(tmp_path, room):
    ir = tmp_path / "ir"
    ir.mkdir()
    (ir / "Fixture.jsonl").write_text(json.dumps(room) + "\n")
    return ir


@pytest.mark.parametrize("case", ["missing", "empty", "unknown_source", "rejected"])
def test_bad_input_preserves_existing_dataset(tmp_path, monkeypatch, room, case):
    ir = inputs(tmp_path, room)
    out = tmp_path / "existing"
    out.mkdir()
    target = out / "train.jsonl"
    target.write_bytes(b"existing training data\n")
    extra = []
    if case == "missing":
        ir = tmp_path / "missing"
    elif case == "empty":
        (ir / "Fixture.jsonl").write_text("")
    elif case == "unknown_source":
        extra = ["--sources", "Absent"]
    else:
        room["meta"]["front_known"] = False
        (ir / "Fixture.jsonl").write_text(json.dumps(room) + "\n")
    with pytest.raises((SystemExit, ValueError, FileExistsError)):
        invoke(monkeypatch, ir, out, *extra)
    assert target.read_bytes() == b"existing training data\n"
    assert list(out.iterdir()) == [target]


def test_valid_input_does_not_overwrite_existing_dataset(tmp_path, monkeypatch, room):
    ir = inputs(tmp_path, room)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "sentinel").write_bytes(b"preserve")
    with pytest.raises((SystemExit, ValueError, FileExistsError)):
        invoke(monkeypatch, ir, out)
    assert (out / "sentinel").read_bytes() == b"preserve"


def test_failed_write_leaves_no_partial_output(tmp_path, monkeypatch, room):
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    def fail(*args, **kwargs):
        raise RuntimeError("fixture serialization failure")
    monkeypatch.setattr(build, "messages", fail)
    with pytest.raises(RuntimeError, match="fixture serialization failure"):
        invoke(monkeypatch, ir, out)
    assert not out.exists()
    assert not list(tmp_path.glob(".new-version-*"))


def test_build_writes_complete_manifest_and_does_not_mutate_input(tmp_path, monkeypatch, room):
    ir = inputs(tmp_path, room)
    before = (ir / "Fixture.jsonl").read_bytes()
    out = tmp_path / "new-version"
    invoke(monkeypatch, ir, out)
    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert sum(manifest["files"][f"{s}.jsonl"]["lines"] for s in ["train", "dev", "test"]) == 1
    assert set(manifest["ir_sha256"]) == {"Fixture.jsonl"}
    assert not any("review" in p or "tests/" in p for p in manifest["code_sha256"])
    assert (ir / "Fixture.jsonl").read_bytes() == before


@pytest.mark.parametrize("case", ["empty_selection", "empty_file", "all_rejected", "malformed_json", "invalid_fraction"])
def test_invalid_build_does_not_create_destination(tmp_path, monkeypatch, room, case):
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    extra = []
    if case == "empty_selection":
        extra = ["--sources"]
    elif case == "empty_file":
        (ir / "Fixture.jsonl").write_text("")
    elif case == "all_rejected":
        room["meta"]["front_known"] = False
        (ir / "Fixture.jsonl").write_text(json.dumps(room) + "\n")
    elif case == "malformed_json":
        (ir / "Fixture.jsonl").write_text("{broken\n")
    else:
        extra = ["--dev", ".8", "--test", ".8"]
    with pytest.raises((SystemExit, ValueError)):
        invoke(monkeypatch, ir, out, *extra)
    assert not out.exists()


def test_ir_change_during_build_is_not_published(tmp_path, monkeypatch, room):
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    original_write = build._write_dataset
    def change_input(*args):
        original_write(*args)
        (ir / "Fixture.jsonl").write_text("changed while building\n")
    monkeypatch.setattr(build, "_write_dataset", change_input)
    with pytest.raises(RuntimeError, match="IR changed during build"):
        invoke(monkeypatch, ir, out)
    assert not out.exists()


@pytest.mark.parametrize("module,aborts", [("validate.py", True), ("serve.py", False)])
def test_manifest_distinguishes_build_dependency_edits(tmp_path, monkeypatch, room, module, aborts):
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    original_sha = build._sha256
    calls = 0
    def edited_hash(path):
        nonlocal calls
        if str(path).endswith("/" + module):
            calls += 1
            if calls > 1:
                return "0" * 64
        return original_sha(path)
    monkeypatch.setattr(build, "_sha256", edited_hash)
    if aborts:
        with pytest.raises(RuntimeError, match="implementation changed"):
            invoke(monkeypatch, ir, out)
        assert not out.exists()
    else:
        invoke(monkeypatch, ir, out)
        manifest = json.loads((out / "MANIFEST.json").read_text())
        assert manifest["code_sha256"][module] == "0" * 64
        assert manifest["build_start_code_sha256"][module] != "0" * 64


@pytest.mark.parametrize("split", ["dev", "test"])
def test_eval_sidecars_roundtrip_and_constraints_hold(tmp_path, monkeypatch, room, split):
    from fastfill.scene import messages
    from fastfill.validate import holds
    room["objects"] += [{"id": "b", "category": "table", "size": [.8, .8, .7],
                         "pos": [3, 2, 0], "yaw": 0, "anchor": "floor", "parent": None}]
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    monkeypatch.setattr(build, "assign_splits", lambda rooms, dev, test: [split] * len(rooms))
    invoke(monkeypatch, ir, out)
    row = json.loads((out / f"{split}.jsonl").read_text())
    reference = json.loads((out / f"{split}_rooms.jsonl").read_text())
    constrained = json.loads((out / f"{split}_constrained_rooms.jsonl").read_text())
    assert row["messages"] == messages(reference)
    assert constrained["constraints"]
    assert all(holds(c, reference["objects"], reference["boundary"]) for c in constrained["constraints"])


def test_training_constraints_survive_augmentation_and_serialization(tmp_path, monkeypatch, room):
    from fastfill.scene import parse
    from fastfill.validate import holds
    room["objects"] += [
        {"id": "b", "category": "table", "size": [.8, .8, .7], "pos": [3, 2, 0],
         "yaw": 0, "anchor": "floor", "parent": None},
        {"id": "c", "category": "vase", "size": [.2, .2, .3], "pos": [3, 2, .7],
         "yaw": 0, "anchor": "object", "parent": "b"}]
    ir = inputs(tmp_path, room)
    out = tmp_path / "new-version"
    invoke(monkeypatch, ir, out, "--dev", "0", "--test", "0", "--constraint_frac", "1",
           "--field_dropout", "0", "--desc_dropout", "0")
    row = json.loads((out / "train.jsonl").read_text())
    prompt = json.loads(row["messages"][1]["content"])
    placements, error = parse(row["messages"][2]["content"], [o["id"] for o in prompt["objects"]])
    assert error is None
    objects = [{**o, "pos": placements[o["id"]]["pos"], "yaw": placements[o["id"]]["yaw"],
                "parent": placements[o["id"]].get("on")} for o in prompt["objects"]]
    assert prompt["constraints"]
    assert any(c[0] == "on" for c in prompt["constraints"])
    assert all(holds(c, objects, prompt["boundary"]) for c in prompt["constraints"])


@pytest.mark.parametrize("field,value", [
    ("boundary", [[0, 0], [5, 0], [5, float("nan")], [0, 5]]),
    ("boundary", [[0, 0], [5, 5]]),
    ("boundary", [[0, 0], [0, 0], [0, 0]]),
    ("boundary", [[0, 0], [5, 0, 0], [5, 5], [0, 5]]),
    ("height", float("inf")), ("height", -2), ("height", "3"),
    ("height", 10 ** 400),
    ("boundary", [[0, 0], [10 ** 400, 0], [5, 5], [0, 5]]),
])
def test_malformed_room_geometry_is_rejected(room, field, value):
    args = argparse.Namespace(boundary_types=["polygon", "hull"], source_anchors={},
                              anchors=["floor", "object"], min_objects=1, max_vertices=0,
                              oob_tol=.1, hidden_max=.3, reject_flagged=[])
    original = copy.deepcopy(room)
    result, reason = build.prep({**room, field: value}, args)
    assert result is None
    assert reason in {"boundary_shape", "bad_numbers"}
    assert room == original
