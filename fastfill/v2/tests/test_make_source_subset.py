"""make_source_subset: rows of the chosen sources byte for byte, the manifest's hashes and counts, refusals."""
import hashlib
import json

import pytest

from fastfill.v2.ops import make_source_subset as mss

SAGE = ("A tall minimalist floor lamp with a black brushed metal tripod base, a slender cylindrical shaft and a conical "
        "shade")


def _row(source, *descriptions):
    objects = [{"id": f"obj_{i:04d}", "category": "lamp", "description": d} for i, d in enumerate(descriptions)]
    return json.dumps({"provenance": {"source": source}, "condition": {"objects": objects}}, separators=(", ", ": "))


def _data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for split in mss.SPLITS:
        rows = [_row("SAGE-10k", SAGE, "lamp"), _row("SpatialLM", "lamp"), _row("IL3D_3dfront", "modern white lamp")]
        (data / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
    return data


def test_keeps_the_chosen_sources_byte_for_byte_and_records_hashes_and_counts(tmp_path):
    data, out = _data(tmp_path), tmp_path / "subset"
    mss.main(["--data", str(data), "--output", str(out), "--sources", "SAGE-10k", "IL3D_3dfront"])
    lines = (data / "train.jsonl").read_text().splitlines(keepends=True)
    assert (out / "train.jsonl").read_text() == lines[0] + lines[2]  # SpatialLM dropped, order and bytes kept
    manifest = json.loads((out / "manifest.json").read_text())
    for split in mss.SPLITS:
        assert manifest["outputs"][f"{split}.jsonl"] == hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()
        assert manifest["inputs"][f"{split}.jsonl"] == hashlib.sha256((data / f"{split}.jsonl").read_bytes()).hexdigest()
    assert manifest["splits"]["train"] == {"IL3D_3dfront": {"rooms": 1, "objects": 1, "objects_detailed": 0},
                                           "SAGE-10k": {"rooms": 1, "objects": 2, "objects_detailed": 1}}
    assert not (tmp_path / "subset.tmp").exists()


def test_detailed_needs_length_and_two_attribute_groups_and_not_the_category():
    assert mss.detailed({"category": "lamp", "description": SAGE})
    assert not mss.detailed({"category": "lamp", "description": "modern compact white lamp with a soft shade"})  # 8 words
    assert not mss.detailed({"category": "floor_lamp", "description": "floor lamp"})
    assert not mss.detailed({"category": "lamp", "description": " ".join(["lamp"] * 20)})  # long, no attributes


def test_refuses_an_existing_output_and_an_absent_source(tmp_path):
    data, out = _data(tmp_path), tmp_path / "subset"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        mss.main(["--data", str(data), "--output", str(out)])
    with pytest.raises(SystemExit, match="absent"):
        mss.main(["--data", str(data), "--output", str(tmp_path / "other"), "--sources", "SAGE-10k", "SAGE10k"])
    assert not (tmp_path / "other").exists()
