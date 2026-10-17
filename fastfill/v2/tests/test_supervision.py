import math

import pytest
import torch

from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch, permute_relations
from fastfill.v2.boxes import bev_giou, bev_iou, intersection_area


def batch():
    pos = torch.tensor([[[.2, .3, 0], [.8, .3, 0]]])
    return {"slot_mask": torch.tensor([[True, True]]),
            "targets": {"position_normalized": pos, "size": torch.ones(1, 2, 3), "yaw": torch.zeros(1, 2)},
            "validity": {"position": torch.ones(1, 2, 3, dtype=torch.bool),
                         "size": torch.ones(1, 2, 3, dtype=torch.bool), "yaw": torch.ones(1, 2, dtype=torch.bool)},
            "objects": [[{"id": "a", "category": "chair", "description": "chair"},
                         {"id": "b", "category": "chair", "description": "chair"}]],
            "exchangeable_group": [["g", "g"]],
            "conditions": [{"constraints": []}], "origin": torch.zeros(1, 3), "scale": torch.ones(1, 3)}


def prediction(b):
    return {"position_normalized": b["targets"]["position_normalized"].flip(1).clone().requires_grad_(),
            "size": torch.full((1, 2, 3), 1.2, requires_grad=True),
            "yaw_logits": torch.zeros(1, 2, 12, requires_grad=True),
            "yaw_residuals": torch.zeros(1, 2, 12, requires_grad=True), "slot_mask": b["slot_mask"]}


def test_exchangeable_swap_detached_assignment_differentiable_loss():
    b = batch()
    p = prediction(b)
    assignment = match_batch(p, b)
    assert assignment.tolist() == [[1, 0]]
    assert not assignment.requires_grad
    result = GeometryCriterion()(p, b)
    result["loss"].backward()
    assert p["size"].grad.abs().sum() > 0
    assert p["yaw_logits"].grad.abs().sum() > 0


def test_same_category_roles_cannot_swap():
    b = batch()
    b["conditions"] = [{"constraints": [{"type": "faces_direction", "object_id": "a", "direction_xy": [1, 0]}]}]
    with pytest.raises(ValueError, match="exchange"):
        match_batch(prediction(b), b)
    assert match_batch(prediction(b), b, enabled=False).tolist() == [[0, 1]]


def test_relations_follow_permutation_external_support_unchanged():
    pi = torch.tensor([1, 0, 2])
    parents = torch.tensor([2, -1, -1])
    relation = torch.arange(9).reshape(3, 3)
    mapped, matrix = permute_relations(pi, parents, relation)
    assert mapped.tolist() == [-1, 2, -1]
    assert torch.equal(matrix, relation[pi][:, pi])


def test_invalid_nan_target_filtered_before_arithmetic_and_padding():
    b = batch()
    b["slot_mask"] = torch.tensor([[True, False]])
    b["objects"] = [b["objects"][0][:1]]
    b["validity"]["size"] = torch.tensor([[[True, True, True], [False, False, False]]])
    b["targets"]["size"] = torch.tensor([[[1., 1., 1.], [float("nan")] * 3]])
    b["targets"]["position_normalized"] = torch.tensor([[[.2, .3, 0.], [float("nan")] * 3]])
    b["targets"]["yaw"] = torch.tensor([[0., float("nan")]])
    p = prediction(b)
    # First prediction valid; padding content deliberately poisonous.
    p["position_normalized"] = torch.tensor([[[.2, .3, 0.], [float("nan")] * 3]], requires_grad=True)
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert torch.isfinite(p["size"].grad).all()


def test_fixed_z_keeps_divisor_three_and_full_fixed_excluded():
    b = batch()
    b["fixed_position_mask"] = torch.tensor([[[False, False, True], [True, True, True]]])
    p = prediction(b)
    p["position_normalized"] = (b["targets"]["position_normalized"] + 1).requires_grad_()
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    # Default L1: two learned coordinates with error 1 each, divided by 3.
    assert result["position"].item() == pytest.approx(2 / 3)
    assert result["counts"]["position"] == 1


def test_yaw_regression_uses_gt_bin_only():
    b = batch()
    b["targets"]["yaw"] = torch.full((1, 2), math.pi / 2 + .1)
    p = prediction(b)
    result = GeometryCriterion()(p, b)
    result["loss"].backward()
    assert p["yaw_residuals"].grad[..., 3].abs().sum() > 0
    assert p["yaw_residuals"].grad[..., :3].abs().sum() == 0
    assert p["yaw_residuals"].grad[..., 4:].abs().sum() == 0


@pytest.mark.parametrize("shift,yaw,expected", [(0., 0., 1.), (2., 0., 0.), (5., 0., None), (0., .4, None)])
def test_box_sanity(shift, yaw, expected):
    p = torch.tensor([0., 0., 0.], dtype=torch.double)
    s = torch.tensor([2., 1., 1.], dtype=torch.double)
    q = p + torch.tensor([shift, 0., 0.], dtype=torch.double)
    value = bev_giou(p, s, torch.tensor(0., dtype=torch.double), q, s, torch.tensor(yaw, dtype=torch.double))
    assert torch.isfinite(value)
    if expected is not None:
        assert value.item() == pytest.approx(expected, abs=1e-8)
    if shift == 5:
        assert value < 0


def test_box_gradients_and_small_boxes():
    p = torch.tensor([.2, .3, 0.], dtype=torch.double, requires_grad=True)
    s = torch.tensor([2.3, 1.1, 1.], dtype=torch.double, requires_grad=True)
    yaw = torch.tensor(.31, dtype=torch.double, requires_grad=True)
    gt = torch.tensor([0., 0., 0.], dtype=torch.double)
    gs = torch.tensor([2., 1.3, 1.], dtype=torch.double)
    gy = torch.tensor(-.12, dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a, b, c: bev_giou(a, b, c, gt, gs, gy), (p, s, yaw))
    tiny = torch.tensor([1e-6, 1e-6, 1e-6], dtype=torch.double)
    assert bev_giou(gt, tiny, gy, gt, tiny, gy).item() == pytest.approx(1., abs=1e-8)


def test_small_float32_boxes_stay_valid_under_translation():
    p = torch.tensor([100., 100., 0.])
    s = torch.tensor([1e-4, 2e-4, 1.])
    yaw = torch.tensor(.2)
    assert bev_giou(p, s, yaw, p, s, yaw).item() == pytest.approx(1., abs=1e-6)
    assert intersection_area(p, s, yaw, p, s, yaw).item() == pytest.approx(2e-8, rel=1e-5)


def test_matching_requires_constraint_graph_and_understands_reference_lists():
    b = batch()
    del b["conditions"]
    with pytest.raises(ValueError, match="condition"):
        match_batch(prediction(b), b)
    b = batch()
    b["conditions"] = [{"constraints": [{"type": "ordered_role", "object_ids": ["a", "b"]}]}]
    with pytest.raises(ValueError, match="exchange"):
        match_batch(prediction(b), b)


def test_nonfinite_loss_weights_fail_at_configuration_boundary():
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            GeometryCriterion(LossConfig(size=value))


def test_relation_permutation_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="permutation"):
        permute_relations(torch.tensor([0, 0]), parents=torch.tensor([-1, -1]))


def test_iou_is_distinct_from_giou_and_global_ddp_reduction(monkeypatch):
    p, q = torch.tensor([0., 0., 0.]), torch.tensor([4., 0., 0.])
    size, yaw = torch.ones(3), torch.tensor(0.)
    assert bev_iou(p, size, yaw, q, size, yaw).item() == 0.
    assert bev_giou(p, size, yaw, q, size, yaw).item() < 0.
    from fastfill.v2.losses import _mean
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda count: count.add_(3))
    local_sum = torch.tensor(4., requires_grad=True)
    normalized, count = _mean(local_sum, 2)
    assert count == 5 and normalized.item() == pytest.approx(8 / 5)
    normalized.backward()
    # DDP subsequently averages two ranks: local coefficient 2/5 -> global 1/5.
    assert local_sum.grad.item() == pytest.approx(2 / 5)
