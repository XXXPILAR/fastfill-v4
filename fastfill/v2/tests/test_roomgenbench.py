"""Dynamic mesh handoff: actual RoomGenBench helpers, with explicit proxy status."""
from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import trimesh

from fastfill.v2.direct_layout import export_handoff, layout_to_roomgenbench, request_to_condition


# A checkout without the submodule can point at one: FASTFILL_ROOMGENBENCH_ROOT=/path/to/RoomGenBench
REFERENCE = Path(os.environ.get("FASTFILL_ROOMGENBENCH_ROOT") or Path(__file__).resolve().parents[3] / "RoomGenBench")


def fixture(*, known=False):
    condition = request_to_condition({"room_type": "study", "room_size_m": [5., 4.],
        "furniture_list": [{"id": "desk", "category": "desk"}, {"id": "cup", "category": "cup"}]},
        room_size_semantics="rectangular" if known else "reference_extent")
    condition = {**condition, "room": {**condition["room"], "fixed_objects": [
        {"id": "fixed_shelf", "category": "shelf", "size_local_m": [1., .4, 2.],
         "bottom_center_m": [4., 2., 0.], "yaw_rad": .2}]},
        "objects": [{**condition["objects"][0], "support_parent": "floor"},
                    {**condition["objects"][1], "support_parent": "desk"}],
        "constraints": [{"type": "near", "object_id": "desk", "target_id": "fixed_shelf", "max_distance_m": 2.}]}
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": "desk", "target_size_local_m": [1.4, .7, .75], "bottom_center_m": [2., 1.5, .3], "yaw_rad": .7},
        {"id": "cup", "target_size_local_m": [.1, .08, .15], "bottom_center_m": [2., 1.5, 1.05], "yaw_rad": -.4}]}
    return condition, layout


def write_asset(directory, obj, *, status="ok", method="test_mesh", mismatch=False):
    directory.mkdir(parents=True, exist_ok=True)
    key = obj["asset_key"]
    # GLB Y-up, front +Z; native extents intentionally differ from target.
    mesh = trimesh.creation.box(extents=[.9, .8, .6])
    mesh.apply_translation([.2, .8, -.1])
    mesh.export(directory / f"{key}.glb")
    (directory / f"{key}.json").write_text(json.dumps({
        "asset_key": "another" if mismatch else key, "method": method,
        "status": status, "prompt": obj["description"], "seconds": .1}))


def corners(p, size, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return [[p[0] + c*x - s*y, p[1] + s*x + c*y, p[2] + z]
            for x, y, z in itertools.product((-size[0]/2, size[0]/2), (-size[1]/2, size[1]/2), (0, size[2]))]


def points_for_node(path, node):
    scene = trimesh.load(path, force="scene", process=False)
    transform, geometry = scene.graph[node]
    return trimesh.transform_points(scene.geometry[geometry].vertices, transform)


def test_handoff_preserves_all_canonical_condition_and_declared_support(tmp_path):
    condition, layout = fixture()
    original = deepcopy((condition, layout))
    output = export_handoff(tmp_path / "handoff", condition, layout)
    assert json.loads((output / "condition.json").read_text()) == condition
    assert json.loads((output / "layout.json").read_text()) == layout
    downstream = json.loads((output / "roomgenbench_scene.json").read_text())
    assert downstream["fixed_objects"] == condition["room"]["fixed_objects"]
    assert downstream["constraints"] == condition["constraints"]
    assert [o["place_id"] for o in downstream["objects"]] == ["floor", "desk"]
    registry = [json.loads(line) for line in (output / "assets.jsonl").read_text().splitlines()]
    assert {o["place"] for o in registry} == {"floor", "on_object"}
    assert all(o["placement_eligible"] for o in registry)
    assert (condition, layout) == original


def test_hard_on_constraint_maps_support_but_soft_on_does_not():
    condition, layout = fixture()
    objects = [{key: value for key, value in o.items() if key != "support_parent"} for o in condition["objects"]]
    # A soft on is no declaration: the cup falls back to the geometric candidate (it rests on the desk top).
    for hard, expected in [(True, ("fixed_shelf", "declared")), (False, ("desk", "inferred"))]:
        source = {**condition, "objects": objects, "constraints": [
            {"type": "on", "object_id": "cup", "target_id": "fixed_shelf", "hard": hard}]}
        downstream = layout_to_roomgenbench(source, layout)
        assert (downstream["objects"][1]["place_id"], downstream["objects"][1]["support_status"]) == expected


def test_registry_shares_asset_key_by_type_and_description_and_keeps_first_instance(tmp_path):
    condition, layout = fixture()
    condition["objects"][1] = {**condition["objects"][1], "category": "desk", "description": "desk"}
    output = export_handoff(tmp_path / "handoff", condition, layout)
    downstream = json.loads((output / "roomgenbench_scene.json").read_text())
    assert len({o["asset_key"] for o in downstream["objects"]}) == 1  # benchmark: type+description only
    assert [o["dimensions"]["width"] for o in downstream["objects"]] == [.7, .08]  # per-instance sizes stay
    registry = [json.loads(line) for line in (output / "assets.jsonl").read_text().splitlines()]
    assert len(registry) == 1 and registry[0]["n_instances"] == 2
    assert registry[0]["dimensions"]["width"] == .7 and registry[0]["place"] == "floor"


def test_arbitrary_scene_layout_boxes_preserves_corners_fixed_proxies_and_unknowns(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    assert receipt["scene_key"].startswith("fastfill_")
    assert receipt["assembly_complete"] and not receipt["generated_mesh_success"]
    assert receipt["counts"] == {"box": 2} and receipt["fixed_bbox_proxies"] == 1
    assert receipt["room"]["height_source"] == "display_reference"
    assert receipt["room"]["height"] == 3.
    assert not receipt["room"]["floor_known"] and not receipt["room"]["boundary_known"]
    assert receipt["room"]["source_height_m"] is None
    assert [w["name"] for w in receipt["walls"]] == [f"shell_wall_{i}" for i in range(4)]
    assert receipt["scene"] == receipt["scene_key"]
    glb = trimesh.load(tmp_path / "assembled" / f'{receipt["scene_key"]}.glb', force="scene", process=False)
    assert {"shell_floor", "shell_wall_0_s0", "fixed_bbox_000", "obj_000", "obj_001"} <= set(glb.graph.nodes_geometry)
    # Unknown height: walls rendered at the display reference height, over the floor.
    wall = points_for_node(tmp_path / "assembled" / f'{receipt["scene_key"]}.glb', "shell_wall_0_s0")
    assert wall[:, 1].max() == pytest.approx(3.) and wall[:, 1].min() == pytest.approx(0.)
    assert receipt["validator"] == receipt["physics"] == receipt["host_commit"] == "not_attempted"
    assert receipt["condition"] == condition
    for index, obj in enumerate(layout["objects"]):
        actual = points_for_node(tmp_path / "assembled" / f'{receipt["scene_key"]}.glb', f"obj_{index:03d}")
        expected = [[x, z, -y] for x, y, z in corners(obj["bottom_center_m"], obj["target_size_local_m"], obj["yaw_rad"])]
        rounded = lambda p: sorted(set(tuple(np.round(x, 6)) for x in p))
        assert rounded(actual) == rounded(expected)
    assert json.loads((tmp_path / "assembled" / "receipt.json").read_text()) == receipt


def test_generated_glb_mesh_fit_logs_native_and_fitted_sizes_without_asset_acceptance(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assets = tmp_path / "assets"
    for obj in scene["objects"]:
        write_asset(assets, obj)
    receipt = assemble_handoff(handoff, tmp_path / "assembled", method="test_mesh",
                              assets_dir=assets, roomgenbench_root=REFERENCE)
    assert receipt["generated_mesh_success"] and receipt["counts"] == {"ok": 2}
    assert receipt["fit_policy"] == "roomgenbench_anisotropic_yaw_snap_tip"
    assert receipt["original_asset_geometry_acceptance"] == "not_checked"
    for output, obj in zip(receipt["objects"], scene["objects"]):
        assert output["native_size_gltf_m"] == pytest.approx([.9, .8, .6])
        assert output["fitted_size_sage_local_m"] == pytest.approx([obj["dimensions"][k] for k in ("width", "length", "height")])
        assert len(output["scale"]) == 3
        assert output["support_parent"] == obj["place_id"]


@pytest.mark.parametrize("mode,expected", [("fallback", "fallback"), ("failed", "failed"),
    ("no_sidecar", "missing"), ("bad_json", "invalid_sidecar"), ("identity", "invalid_sidecar"),
    ("bad_status_type", "invalid_sidecar"), ("no_glb", "missing"), ("bad_glb", "failed")])
def test_failures_never_disappear_or_count_as_generated_success(tmp_path, mode, expected):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assets = tmp_path / "assets"
    for obj in scene["objects"]:
        write_asset(assets, obj, status=mode if mode in {"fallback", "failed"} else "ok", mismatch=mode == "identity")
        if mode == "no_sidecar":
            (assets / f'{obj["asset_key"]}.json').unlink()
        elif mode == "bad_json":
            (assets / f'{obj["asset_key"]}.json').write_text("{broken")
        elif mode == "bad_status_type":
            (assets / f'{obj["asset_key"]}.json').write_text(json.dumps({
                "asset_key": obj["asset_key"], "method": "test_mesh", "status": ["ok"]}))
        elif mode == "no_glb":
            (assets / f'{obj["asset_key"]}.glb').unlink()
        elif mode == "bad_glb":
            (assets / f'{obj["asset_key"]}.glb').write_bytes(b"not a glb")
    receipt = assemble_handoff(handoff, tmp_path / "assembled", method="test_mesh",
                              assets_dir=assets, roomgenbench_root=REFERENCE)
    assert not receipt["generated_mesh_success"] and receipt["counts"] == {expected: 2}
    assert {o["id"] for o in receipt["objects"]} == {o["id"] for o in layout["objects"]}
    assert all(o["geometry_kind"] == ("fallback_mesh" if mode == "fallback" else "placeholder_bbox") for o in receipt["objects"])


def test_unknown_placement_is_explicit_and_required_policy_rejects_before_output(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    condition["objects"] = [{key: value for key, value in o.items() if key != "support_parent"} for o in condition["objects"]]
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    registry = {o["type"]: o for o in map(json.loads, (handoff / "assets.jsonl").read_text().splitlines())}
    # The desk floats 0.3 m above the floor: unknown. The cup rests on the desk top: an inferred candidate.
    assert registry["desk"]["place"] == "unknown" and not registry["desk"]["placement_eligible"]
    assert (registry["cup"]["place"], registry["cup"]["support_status"]) == ("on_object", "inferred")
    with pytest.raises(ValueError, match="placement"):
        assemble_handoff(handoff, tmp_path / "invalid", require_placement=True, roomgenbench_root=REFERENCE)
    assert not (tmp_path / "invalid").exists()
    receipt = assemble_handoff(handoff, tmp_path / "display", roomgenbench_root=REFERENCE)
    by_id = {o["id"]: o for o in receipt["objects"]}
    assert by_id["desk"]["place"] == "unknown" and by_id["desk"]["support_parent"] is None and by_id["desk"]["support_status"] == "unknown"
    assert by_id["cup"]["place"] == "on_object" and by_id["cup"]["support_parent"] == "desk" and by_id["cup"]["support_status"] == "inferred"


@pytest.mark.parametrize("method,assets", [("../bad", None), ("sage_gt", None), ("test_mesh", None)])
def test_invalid_method_or_missing_asset_directory_is_rejected_before_output(tmp_path, method, assets):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    with pytest.raises(ValueError):
        assemble_handoff(handoff, tmp_path / "invalid", method=method, assets_dir=assets, roomgenbench_root=REFERENCE)
    assert not (tmp_path / "invalid").exists()


def test_output_must_stay_independent_of_inputs_and_reference_repo(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    reference = tmp_path / "reference"
    (reference / "bench").mkdir(parents=True)
    (reference / "bench" / "assemble.py").write_bytes((REFERENCE / "bench" / "assemble.py").read_bytes())
    for output in [handoff / "new_output", reference / "never_create_test_output"]:
        with pytest.raises(ValueError, match="overlap|reference"):
            assemble_handoff(handoff, output, roomgenbench_root=reference)
        assert not output.exists()


def test_missing_reference_and_invalid_reference_height_fail_before_output(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    with pytest.raises(FileNotFoundError, match="assembler"):
        assemble_handoff(handoff, tmp_path / "invalid", roomgenbench_root=tmp_path / "missing")
    for height in [0, float("nan"), True]:
        with pytest.raises(ValueError):
            assemble_handoff(handoff, tmp_path / "invalid", roomgenbench_root=REFERENCE, display_height_m=height)
    assert not (tmp_path / "invalid").exists()


def test_native_very_thin_mesh_mismatch_remains_visible_and_not_a_success(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assets = tmp_path / "assets"
    for obj in scene["objects"]:
        write_asset(assets, obj)
        trimesh.creation.box(extents=[.9, .8, .000001]).export(assets / f'{obj["asset_key"]}.glb')
    receipt = assemble_handoff(handoff, tmp_path / "assembled", method="test_mesh", assets_dir=assets, roomgenbench_root=REFERENCE)
    assert not receipt["generated_mesh_success"] and receipt["counts"] == {"fit_mismatch": 2}
    assert all(o["geometry_kind"] == "fitted_mesh_with_size_mismatch" for o in receipt["objects"])


def test_handoff_scene_tampering_is_rejected_and_existing_outputs_are_immutable(tmp_path):
    from fastfill.v2.roomgenbench import assemble_handoff
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    snapshots = {p.name: p.read_bytes() for p in (tmp_path / "assembled").iterdir()}
    with pytest.raises(FileExistsError):
        assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE)
    assert {p.name: p.read_bytes() for p in (tmp_path / "assembled").iterdir()} == snapshots
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    scene["objects"][0]["asset_key"] = "../../outside"
    (handoff / "roomgenbench_scene.json").write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="handoff|match"):
        assemble_handoff(handoff, tmp_path / "tampered", roomgenbench_root=REFERENCE)
    assert not (tmp_path / "tampered").exists()


def test_cli_failed_assets_exit_nonzero_but_keep_complete_receipt(tmp_path):
    condition, layout = fixture()
    handoff = export_handoff(tmp_path / "handoff", condition, layout)
    assets = tmp_path / "empty_assets"
    assets.mkdir()
    proc = subprocess.run([sys.executable, "-m", "fastfill.v2.roomgenbench", "--handoff", str(handoff),
        "--output-dir", str(tmp_path / "assembled"), "--assets-dir", str(assets), "--method", "test_mesh",
        "--roomgenbench-root", str(REFERENCE)], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2, proc.stderr
    receipt = json.loads((tmp_path / "assembled" / "receipt.json").read_text())
    assert receipt["counts"] == {"missing": 2} and receipt["assembly_complete"]
