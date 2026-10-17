"""Tests for source geometry rather than inherited v1 interpretation."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from fastfill.v2.adapters import canonical_obb, multiscan_sample
from fastfill.v2.data import build_dataset, grouped_split
from fastfill.v2.audit import audit_sources, main as audit_main
from fastfill.v2.data import main as build_main
from fastfill.v2.data import _output_path


def row(object_id="1", **overrides):
    return {"object_id": object_id, "scan_id": "scan_00", "category": "chair",
            "obb_center": "[2, 3, 1]", "obb_half_extents": "[0.4, 0.5, 0.2]",
            "obb_axes": "[1,0,0, 0,0,1, 0,-1,0]", "front": "[0,-1,0]",
            "up": "[0,0,1]", "is_architectural": "False", "is_opening": "False",
            **overrides}


def region(**overrides):
    return {"scan_id": "scan_00", "poly_loop": "[[0,0],[4,0],[4,4],[0,4]]",
            "floor_height": "0.25", "ceiling_height": "3.25",
            "height_reliable": "True", "room_type": "office", **overrides}


def test_half_extents_axis_permutation_and_bottom_center():
    box = canonical_obb([2, 3, 1], [.4, .5, .2],
                        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], [0, -1, 0], [0, 0, 1])
    assert box["target_size_local_m"] == pytest.approx([.4, .8, 1])
    assert box["bottom_center_m"] == pytest.approx([2, 3, .5])
    assert box["yaw_rad"] == pytest.approx(-math.pi / 2)


def test_full_lengths_and_millimeters_are_not_halved_twice():
    box = canonical_obb([2000, 3000, 1000], [400, 800, 1000], np.eye(3),
                        [1, 0, 0], [0, 0, 1], half_extents=False, unit_scale=.001)
    assert box["target_size_local_m"] == pytest.approx([.4, .8, 1])
    assert box["bottom_center_m"] == pytest.approx([2, 3, .5])


def test_yaw_does_not_change_local_size_and_corners_round_trip():
    for angle in [0, .7, math.pi - .01, -math.pi]:
        c, s = math.cos(angle), math.sin(angle)
        axes = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        box = canonical_obb([2, 3, 1], [.2, .4, .5], axes, axes[0], axes[2])
        assert box["target_size_local_m"] == pytest.approx([.4, .8, 1])
        source = np.array([[x, y, z] for x in [-.2, .2] for y in [-.4, .4]
                           for z in [-.5, .5]]) @ np.asarray(axes) + [2, 3, 1]
        local = (source - box["bottom_center_m"]) @ np.asarray(axes).T
        assert local.min(0) == pytest.approx([-.2, -.4, 0])
        assert local.max(0) == pytest.approx([.2, .4, 1])


@pytest.mark.parametrize("extents", [[0, .4, .5], [-.2, .4, .5], [float('nan'), .4, .5]])
def test_invalid_size_rejected_without_epsilon_or_absolute_value(extents):
    with pytest.raises(ValueError, match="positive"):
        canonical_obb([0, 0, 1], extents, np.eye(3), [1, 0, 0], [0, 0, 1])


def test_tilt_and_unknown_front_fail_closed():
    with pytest.raises(ValueError, match="upright"):
        canonical_obb([0, 0, 1], [.2, .4, .5], np.eye(3), [1, 0, 0], [0, .1, .99])
    with pytest.raises(ValueError, match="front"):
        canonical_obb([0, 0, 1], [.2, .4, .5], np.eye(3), [0, 0, 1], [0, 0, 1])


def test_condition_never_contains_target_geometry_or_asset_identity():
    sample = multiscan_sample(region(), [row()], {"scene_id": "house", "scan_id": "scan_00"})
    condition = sample["condition"]
    assert condition["objects"] == [{"id": "object_0000", "category": "chair", "description": "a chair"}]
    assert sample["target"]["objects"][0]["bottom_center_m"] == pytest.approx([2, 3, .5])
    assert condition["room"]["floor_z_m"] == .25
    assert condition["room"]["boundary_known"] is False
    assert condition["room"]["floor_known"] is False
    assert sample["validity"]["yaw"] == [True]
    assert sample["provenance"]["source_object_ids"] == ["1"]


def test_missing_height_is_unknown_not_fabricated():
    sample = multiscan_sample(region(height_reliable="False"), [row()], {"scene_id": "house"})
    assert sample["condition"]["room"]["height_m"] is None


def test_fixed_obstacles_preserved_without_becoming_requested_objects():
    sample = multiscan_sample(region(), [row(), row("2", category="pillar", is_architectural="True")],
                              {"scene_id": "house"})
    assert len(sample["condition"]["objects"]) == 1
    assert len(sample["condition"]["room"]["fixed_objects"]) == 1


def test_anonymous_identical_requests_are_exchangeable_without_using_target_sizes():
    sample = multiscan_sample(region(), [row(), row("2", obb_half_extents="[0.2,0.6,0.3]")],
                              {"scene_id": "house"})
    requests, groups = sample["condition"]["objects"], sample["validity"]["exchangeable_group"]
    assert groups[0] == groups[1] is not None
    assert all("exchangeable_group" not in request for request in requests)
    assert "support_parent" not in requests[0]


def test_audit_inventories_all_expected_parents_and_records_missing_sources(tmp_path):
    audit = audit_sources(tmp_path)
    assert len(audit["top_directories"]) == 5
    assert len(audit["sources"]) == 16
    assert all(not source["present"] for source in audit["sources"])
    for source in audit["sources"]:
        assert source["units"] and source["size_semantics"] and source["position_reference"]
        assert source["semantic_front"] and source["eligibility"]


def test_audit_reads_bounded_real_fields_and_measures_csv_rows(tmp_path):
    base = tmp_path / "XXXpilar__multiscan-clean"
    base.mkdir()
    write_csv(base / "objects.csv", [row(), row("2")])
    (base / "README.md").write_text("Actual source fixture: local OBB half extents.")
    exported = tmp_path / "BillLin66__3D_Room_Collections" / "SpatialGen_exported"
    exported.mkdir(parents=True)
    (exported / "spatialgen.jsonl").write_text(json.dumps({"rooms": [{"furniture": [{"position": [0, 0, 1]}]}]}) + "\n")
    report = audit_sources(tmp_path)
    multi = next(source for source in report["sources"] if source["name"] == "MultiScan")
    assert multi["actual_sample"]["measured_records"] == 2
    assert "obb_axes" in multi["actual_sample"]["first_record_keys"]
    assert multi["evidence"][0]["path"] == "README.md"
    spatial = next(source for source in report["sources"] if source["name"] == "SpatialGen")
    assert spatial["actual_sample"]["first_object_keys"] == ["position"]


def test_audit_cli_rejects_source_output_and_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["audit", "--source-root", str(tmp_path), "--output", str(tmp_path / "write.json")])
    with pytest.raises(ValueError, match="read-only"):
        audit_main()
    output = tmp_path.parent / (tmp_path.name + "-audit.json")
    monkeypatch.setattr(sys, "argv", ["audit", "--source-root", str(tmp_path), "--output", str(output)])
    audit_main()
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit_main()
    assert output.read_bytes() == first


def test_builder_cli_bounds_complete_scenes_and_keeps_source_failures_visible(tmp_path, monkeypatch):
    source = tmp_path / "source"
    base = source / "XXXpilar__multiscan-clean"
    base.mkdir(parents=True)
    write_csv(base / "objects.csv", [row(obb_half_extents="[0,1,1]"), row(scan_id="scan_01")])
    write_csv(base / "regions.csv", [region(), region(scan_id="scan_01")])
    write_csv(base / "scans.csv", [{"scan_id": "scan_00", "scene_id": "house"},
                                   {"scan_id": "scan_01", "scene_id": "house"}])
    output = tmp_path / "build"
    monkeypatch.setattr(sys, "argv", ["data", "--source", "multiscan", "--source-root", str(source), "--output", str(output), "--max-scenes", "1"])
    build_main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["samples_written"] == 1
    assert manifest["bounded_build"] is True
    assert manifest["diagnostics"]["scenes_inspected"] == 2
    assert manifest["diagnostics"]["rejections"][0]["scene_id"] == "scan_00"


def test_target_geometry_cannot_expand_source_room_or_change_floor():
    a = multiscan_sample(region(), [row()], {"scene_id": "house"})
    b = multiscan_sample(region(), [row(obb_center="[200,300,20]")], {"scene_id": "house"})
    assert a["condition"] == b["condition"]


def test_related_scans_share_split_and_split_is_stable():
    assert grouped_split("MultiScan", "house", 42) == grouped_split("MultiScan", "house", 42)
    assert grouped_split("MultiScan", "house", 42) in {"train", "validation", "test"}


def write_csv(path, records):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_builder_preserves_sources_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    base = source / "XXXpilar__multiscan-clean"
    base.mkdir(parents=True)
    write_csv(base / "objects.csv", [row()])
    write_csv(base / "regions.csv", [region()])
    write_csv(base / "scans.csv", [{"scan_id": "scan_00", "scene_id": "house"}])
    before = (base / "objects.csv").read_bytes()
    output = tmp_path / "build"
    manifest = build_dataset(source, output)
    assert manifest["samples_written"] == 1
    assert manifest["objects_written"] == 1
    assert (base / "objects.csv").read_bytes() == before
    assert sum(len(p.read_text().splitlines()) for p in output.glob("*.jsonl")) == 1
    with pytest.raises(FileExistsError):
        build_dataset(source, output)
    with pytest.raises(ValueError, match="source"):
        build_dataset(source, source / "new_output")


def test_builder_rejects_source_through_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="source"):
        build_dataset(source, alias / "output")


def test_alternate_source_cannot_bypass_canonical_external_root_guard(tmp_path):
    canonical = Path("/Volumes/harddisk/3D_Room_Collections")
    with pytest.raises(ValueError, match="source"):
        _output_path(tmp_path / "alternate_source", canonical / "guard-test-build")


def test_audit_alternate_source_cannot_write_canonical_external_root(tmp_path, monkeypatch):
    import fastfill.v2.audit as audit_module
    canonical = Path("/Volumes/harddisk/3D_Room_Collections")
    monkeypatch.setattr(sys, "argv", ["audit", "--source-root", str(tmp_path),
                                     "--output", str(canonical / "guard-test-audit.json")])
    # Even during the expected RED run, never allow a protected-source write.
    monkeypatch.setattr(audit_module, "audit_sources", lambda root: {"sources": []})
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    original_open = Path.open
    def read_only_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x")):
            raise AssertionError("protected source write attempted")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read_only_open)
    with pytest.raises(ValueError, match="read-only"):
        audit_main()
