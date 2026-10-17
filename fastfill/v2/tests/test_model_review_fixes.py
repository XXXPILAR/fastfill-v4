"""Regression cases from the independent model/objective review."""
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.boxes import bev_giou, bev_iou, intersection_area
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.tests.test_execution import sample
from fastfill.v2.tests.test_supervision import batch, prediction
from fastfill.v2.train import _preflight


@pytest.mark.parametrize("field,value", [
    ("target_size_local_m", [True, 1., 1.]),
    ("target_size_local_m", ["1.0", 1., 1.]),
    ("bottom_center_m", [2., False, 0.]),
    ("bottom_center_m", [2., "1.0", 0.]),
    ("yaw_rad", True), ("yaw_rad", "1.0"),
])
def test_geometry_targets_reject_boolean_and_string_labels(field, value):
    row = sample()
    row["target"] = {**row["target"], "objects": [{**row["target"]["objects"][0], field: value}]}
    with pytest.raises(ValueError):
        collate_samples([row], TinyTokenizer())



@pytest.mark.parametrize("hard,expected_fixed", [(True, True), (False, False)])
def test_hard_on_floor_condition_has_consistent_fixed_height(hard, expected_fixed):
    row = sample()
    obj = {key: value for key, value in row["condition"]["objects"][0].items() if key != "support_parent"}
    row["condition"] = {**row["condition"], "objects": [obj], "constraints": [{
        "type": "on", "object_id": "a", "target_id": "floor", "hard": hard}]}
    original = json.dumps(row["condition"], sort_keys=True)
    b = collate_samples([row], TinyTokenizer())
    assert b["fixed_position_mask"].tolist() == [[[False, False, expected_fixed]]]
    assert json.dumps(row["condition"], sort_keys=True) == original
    assert "support_parent" not in b["objects"][0][0]


def test_preflight_rejects_labels_entirely_supplied_by_condition():
    row = sample()
    row["condition"] = {**row["condition"], "objects": [{
        **row["condition"]["objects"][0], "fixed_size_local_m": [1., 1., 1.]}]}
    row["target"] = {**row["target"], "objects": [{
        **row["target"]["objects"][0], "bottom_center_m": None, "yaw_rad": None}]}
    row["validity"] = {**row["validity"], "position": [[False] * 3], "yaw": [False]}
    with pytest.raises(ValueError, match="no supervised|no learnable"):
        _preflight([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0), {"max_length": 4096})


def test_preflight_keeps_partially_fixed_size_with_learnable_dimensions():
    row = sample()
    row["condition"] = {**row["condition"], "objects": [{
        **row["condition"]["objects"][0], "fixed_size_local_m": [1., None, 1.]}]}
    row["target"] = {**row["target"], "objects": [{
        **row["target"]["objects"][0], "bottom_center_m": None, "yaw_rad": None}]}
    row["validity"] = {**row["validity"], "position": [[False] * 3], "yaw": [False]}
    kept, rejected = _preflight([row], TinyTokenizer(), ModelConfig(backbone="tiny", lora_rank=0), {"max_length": 4096})
    assert len(kept) == 1 and not rejected


def test_graph_connected_zero_does_not_overflow_finite_size_predictions():
    b = batch()
    p = {**prediction(b), "size": torch.full((1, 2, 3), 1e38, requires_grad=True)}
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert all(torch.isfinite(p[key].grad).all() for key in
               ("position_normalized", "size", "yaw_logits", "yaw_residuals"))


def _distributed_uneven_box_worker(rank, rendezvous, result_directory):
    from torch import distributed as dist
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        b = batch()
        b["validity"] = {**b["validity"], "yaw": torch.full((1, 2), bool(rank), dtype=torch.bool)}
        p = prediction(b)
        result = GeometryCriterion(LossConfig(hungarian=False, box=1.))(p, b)
        result["loss"].backward()
        assert result["counts"] == {"position": 4, "size": 4, "yaw": 2, "box": 2}
        assert all(torch.isfinite(p[key].grad).all() for key in
                   ("position_normalized", "size", "yaw_logits", "yaw_residuals"))
        # All ranks also participate when the global box count becomes zero.
        no_yaw = {**b, "validity": {**b["validity"], "yaw": torch.zeros((1, 2), dtype=torch.bool)}}
        empty = GeometryCriterion(LossConfig(hungarian=False, box=1.))(prediction(no_yaw), no_yaw)
        assert empty["counts"]["box"] == 0 and torch.isfinite(empty["loss"])
        (Path(result_directory) / f"rank-{rank}.json").write_text(json.dumps({
            "uneven_counts": result["counts"], "global_zero_box_count": empty["counts"]["box"]}))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="requires Gloo distributed backend")
def test_actual_two_process_uneven_and_global_zero_box_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if __import__("sys").platform == "darwin" else "lo")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CPU Gloo ranks: with one visible GPU, rank 1 would ask for cuda:1
    rendezvous = "file://" + str(tmp_path / "rendezvous")
    torch.multiprocessing.spawn(_distributed_uneven_box_worker,
        args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    assert all((tmp_path / f"rank-{rank}.json").exists() for rank in range(2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal device")
@pytest.mark.parametrize("operator", [bev_giou, bev_iou, intersection_area])
def test_mps_box_operators_match_cpu_and_preserve_gradients(operator):
    position = torch.tensor([.2, .3, 0.], requires_grad=True, device="mps")
    size = torch.tensor([2.3, 1.1, 1.], requires_grad=True, device="mps")
    yaw = torch.tensor(.31, requires_grad=True, device="mps")
    gt_position, gt_size, gt_yaw = torch.zeros(3), torch.tensor([2., 1.3, 1.]), torch.tensor(-.12)
    cpu_inputs = [value.detach().cpu().double().requires_grad_() for value in (position, size, yaw)]
    expected = operator(*cpu_inputs, gt_position.double(), gt_size.double(), gt_yaw.double())
    expected.backward()
    actual = operator(position, size, yaw, gt_position.to("mps"), gt_size.to("mps"), gt_yaw.to("mps"))
    assert actual.device.type == "mps" and actual.dtype == torch.float32
    assert float(actual.detach().cpu()) == pytest.approx(float(expected.detach()), rel=2e-6, abs=1e-7)
    actual.backward()
    for mps, cpu in zip((position, size, yaw), cpu_inputs):
        assert mps.grad.device.type == "mps" and torch.isfinite(mps.grad).all()
        assert torch.allclose(mps.grad.cpu().double(), cpu.grad, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Metal device")
def test_mps_matching_and_all_geometry_loss_branches_receive_gradients():
    from fastfill.v2.io import to_device
    original = batch()
    original["conditions"] = [{"objects": original["objects"][0], "constraints": [], "room": {
        "floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]], "boundary_known": True}}]
    b = to_device(original, "mps")
    p = {key: value.detach().to("mps").requires_grad_(value.is_floating_point())
         for key, value in prediction(original).items()}
    result = GeometryCriterion(LossConfig(box=1., collision=1., boundary=1.))(p, b)
    assert result["assignment"].cpu().tolist() == [[1, 0]]
    assert torch.isfinite(result["loss"]) and result["loss"].device.type == "mps"
    result["loss"].backward()
    assert all(torch.isfinite(p[key].grad).all() for key in
               ("position_normalized", "size", "yaw_logits", "yaw_residuals"))
