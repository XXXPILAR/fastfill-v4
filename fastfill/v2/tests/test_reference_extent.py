"""Reference room extents retain coordinates without claiming a measured room."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fastfill.v2.direct_layout import bbox_diagnostics, layout_to_roomgenbench, layout_to_scene, request_to_condition
from fastfill.v2.schema import validate_condition, validate_layout


def request(dimensions=(5., 4.)):
    return {"room_type": "unknown", "room_size_m": list(dimensions), "furniture_list": ["chair"]}


def prediction(condition, position=(-1., 2., -.5)):
    return {"schema_version": "fastfill.v2", "objects": [
        {"id": condition["objects"][0]["id"], "target_size_local_m": [1., 1., 1.],
         "bottom_center_m": list(position), "yaw_rad": 0.}]}


@pytest.mark.parametrize("dimensions", [(5., 4.), (5., 4., 3.)])
def test_reference_extent_preserves_request_without_fabricating_boundary_or_floor(dimensions):
    source = request(dimensions)
    saved = deepcopy(source)
    condition = request_to_condition(source, room_size_semantics="reference_extent")
    validate_condition(condition)
    assert source == saved
    assert set(source) == {"room_type", "room_size_m", "furniture_list"}
    assert condition["room"] == {
        "frame": "right_handed_z_up", "room_type": "unknown",
        "floor_polygon_xy_m": [[0., 0.], [5., 0.], [5., 4.], [0., 4.]],
        "floor_z_m": 0., "floor_known": False, "boundary_known": False,
        "boundary_quality": "source_reference_extent",
        "height_m": dimensions[2] if len(dimensions) == 3 else None}
    assert condition["constraints"] == []
    assert condition["objects"] == [{"id": "obj_0000", "category": "chair", "description": "chair"}]
    scene = layout_to_scene(condition, prediction(condition))
    assert scene["room"] == condition["room"]
    scene["room"]["floor_polygon_xy_m"][0][0] = 100.
    assert condition["room"]["floor_polygon_xy_m"][0][0] == 0.


def test_default_and_explicit_rectangular_profiles_remain_exact_and_independent():
    source = request((5., 4., 3.))
    saved = deepcopy(source)
    default = request_to_condition(source)
    explicit = request_to_condition(source, room_size_semantics="rectangular")
    reference = request_to_condition(source, room_size_semantics="reference_extent")
    assert source == saved and default == explicit
    assert default["room"]["floor_known"] is True
    assert default["room"]["boundary_known"] is True
    assert default["room"]["boundary_quality"] == "explicit_rectangular_request"
    assert {key: value for key, value in reference["room"].items()
            if key not in {"floor_known", "boundary_known", "boundary_quality"}} == {
                key: value for key, value in default["room"].items()
                if key not in {"floor_known", "boundary_known", "boundary_quality"}}


@pytest.mark.parametrize("semantics", [None, True, "extent", "", "REFERENCE_EXTENT"])
def test_unknown_room_size_semantics_is_rejected(semantics):
    with pytest.raises(ValueError, match="room_size_semantics"):
        request_to_condition(request(), room_size_semantics=semantics)


@pytest.mark.parametrize("semantics,expected", [("rectangular", "fail"), ("reference_extent", "unknown")])
def test_boundary_floor_and_ceiling_diagnostics_obey_profile(semantics, expected):
    condition = request_to_condition(request((5., 4., 3.)), room_size_semantics=semantics)
    # One outside object below nominal z=0; a second case above nominal ceiling.
    for position in [(-1., 2., -.5), (-1., 2., 4.)]:
        scene = layout_to_scene(condition, prediction(condition, position))
        report = bbox_diagnostics(scene)
        checks = {check["code"]: check["status"] for check in report["checks"]}
        assert checks["boundary"] == expected
        assert checks["floor_lower_bound"] == (expected if position[2] < 0 else (
            "pass" if semantics == "rectangular" else "unknown"))
        assert checks["ceiling"] == (expected if position[2] > 3 else (
            "pass" if semantics == "rectangular" else "unknown"))
        assert report["asset_retrieval"] == "not_attempted"
        assert report["support"] == "unknown" and report["commit"] == "not_attempted"
        if semantics == "reference_extent":
            assert report["room_size_semantics"] == "reference_extent"


def test_reference_extent_prediction_cli_exports_raw_geometry_with_unknown_physical_checks(tmp_path):
    from fastfill.v2.batch import TinyTokenizer
    from fastfill.v2.model import ModelConfig, build_model

    checkpoint = tmp_path / "model"
    model = build_model(ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16,
                        decoder_heads=2, decoder_layers=1, tiny_hidden_size=16))
    with torch.no_grad():
        for head in (model.position_head, model.size_head, model.yaw_logits_head, model.yaw_residual_head):
            head.weight.zero_()
            head.bias.zero_()
        model.position_head.bias.copy_(torch.tensor([-.2, .5, -.5]))
    model.save_pretrained(checkpoint)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    source = request()
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(source))
    output, handoff = tmp_path / "prediction.json", tmp_path / "handoff"
    command = [sys.executable, "-m", "fastfill.v2.predict", "--checkpoint", str(checkpoint),
               "--request", str(request_path), "--room-size-semantics", "reference_extent",
               "--output", str(output), "--export-dir", str(handoff), "--device", "cpu"]
    proc = subprocess.run(command, cwd=Path(__file__).resolve().parents[3],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    condition = request_to_condition(source, room_size_semantics="reference_extent")
    actual = json.loads(output.read_text())
    validate_layout(actual, condition)
    assert actual["objects"][0]["bottom_center_m"][0] < 0.
    assert actual["objects"][0]["bottom_center_m"][2] < 0.
    scene = json.loads((handoff / "scene.json").read_text())
    assert scene == layout_to_scene(condition, actual)
    assert scene["room"]["height_m"] is None
    assert not scene["room"]["boundary_known"] and not scene["room"]["floor_known"]
    diagnostics = json.loads((handoff / "diagnostics.json").read_text())
    assert {check["status"] for check in diagnostics["checks"]} == {"unknown"}
    assert diagnostics["room_size_semantics"] == "reference_extent"
    assert diagnostics["asset_retrieval"] == "not_attempted"
    assert diagnostics["commit"] == "not_attempted"
    assert json.loads(request_path.read_text()) == source
    assert (handoff / "layout.glb").exists() and (handoff / "preview.svg").exists()


def test_roomgenbench_handoff_keeps_reference_range_distinct_from_physical_room():
    condition = request_to_condition(request((5., 4., 3.)), room_size_semantics="reference_extent")
    actual = prediction(condition)
    saved_condition, saved_prediction = deepcopy(condition), deepcopy(actual)
    downstream = layout_to_roomgenbench(condition, actual)
    assert downstream["room_size_semantics"] == "reference_extent"
    assert downstream["room"]["boundary_known"] is False
    assert downstream["room"]["floor_known"] is False
    assert downstream["room"]["boundary_quality"] == "source_reference_extent"
    assert "reference" in downstream["room_interpretation"]
    assert downstream["room"]["dimensions"] == {"width": 5., "length": 4., "height": 3.}
    assert downstream["objects"][0]["position"] == {"x": -1., "y": 2., "z": -.5}
    assert condition == saved_condition and actual == saved_prediction
    strict = request_to_condition(request((5., 4., 3.)))
    assert "room_size_semantics" not in layout_to_roomgenbench(strict, prediction(strict))


def test_prediction_rejects_reference_flag_with_full_condition_before_model_loading(tmp_path, monkeypatch):
    from fastfill.v2 import predict

    def forbidden_load(*args, **kwargs):
        raise AssertionError("inapplicable semantics flags must be rejected before model load")

    monkeypatch.setattr(predict, "load_model", forbidden_load)
    with pytest.raises(ValueError, match="only to --request"):
        predict.main(["--checkpoint", str(tmp_path / "missing-model"),
                      "--condition", str(tmp_path / "missing-condition.json"),
                      "--room-size-semantics", "reference_extent", "--output", str(tmp_path / "prediction.json")])
