"""Independent scene-regularizer regressions from the second implementation audit."""
from copy import deepcopy
import json

import pytest
import torch

from fastfill.v2.losses import LossConfig
from fastfill.v2.regularizers import _collision, scene_regularizers


def _scene(*, fixed=False):
    parent = {"id": "desk", "category": "desk"}
    child = {"id": "book", "category": "book"}
    room = {"floor_polygon_xy_m": [[-3, -3], [3, -3], [3, 3], [-3, 3]]}
    if fixed:
        room = {**room, "fixed_objects": [{**parent, "bottom_center_m": [0, 0, 0],
                                           "size_local_m": [2, 2, 1], "yaw_rad": 0}]}
    condition = {"room": room, "objects": [child] if fixed else [parent, child], "constraints": []}
    positions = torch.tensor([[[0., 0., .8]]] if fixed else [[[0., 0., 0.], [0., 0., .8]]], requires_grad=True)
    sizes = torch.tensor([[[.4, .4, .4]]] if fixed else [[[2., 2., 1.], [.4, .4, .4]]], requires_grad=True)
    yaws = torch.zeros((1, len(condition["objects"])), requires_grad=True)
    batch = {"conditions": [condition], "slot_mask": torch.ones_like(yaws, dtype=torch.bool)}
    return positions, sizes, yaws, batch


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("hard", [True, None])
def test_hard_on_exemption_matches_support_parent_without_mutating_condition(fixed, hard):
    positions, sizes, yaws, batch = _scene(fixed=fixed)
    explicit = deepcopy(batch)
    explicit["conditions"][0]["objects"][-1]["support_parent"] = "desk"
    constraint = {"type": "on", "object_id": "book", "target_id": "desk"}
    if hard is not None:
        constraint = {**constraint, "hard": hard}
    constrained = {**batch, "conditions": [{**batch["conditions"][0], "constraints": [constraint]}]}
    original = json.dumps(constrained["conditions"], sort_keys=True)
    config = LossConfig(collision=1.)
    expected = scene_regularizers(positions, sizes, yaws, explicit, config)["collision"]
    actual = scene_regularizers(positions, sizes, yaws, constrained, config)["collision"]
    assert expected.item() == 0.
    assert actual.item() == expected.item()
    assert json.dumps(constrained["conditions"], sort_keys=True) == original


def test_soft_on_does_not_exempt_collision():
    positions, sizes, yaws, batch = _scene()
    condition = {**batch["conditions"][0], "constraints": [{
        "type": "on", "object_id": "book", "target_id": "desk", "hard": False}]}
    result = scene_regularizers(positions, sizes, yaws, {**batch, "conditions": [condition]}, LossConfig(collision=1.))
    assert result["collision"] > 0
    result["collision"].backward()
    assert torch.isfinite(positions.grad).all()


def test_empty_collision_term_zero_does_not_overflow_finite_positions():
    condition = {"room": {}, "objects": [{"id": "chair", "category": "chair"}], "constraints": []}
    positions = torch.full((1, 1, 3), 1.2e38, requires_grad=True)
    sizes, yaws = torch.ones_like(positions), torch.zeros((1, 1))
    batch = {"conditions": [condition], "slot_mask": torch.tensor([[True]])}
    result = scene_regularizers(positions, sizes, yaws, batch, LossConfig(collision=1.))
    assert result["collision"].item() == 0.
    assert result["boundary"].item() == 0.
    result["collision"].backward()
    assert torch.equal(positions.grad, torch.zeros_like(positions))


def test_vertical_disjoint_zero_is_finite_and_connects_both_boxes():
    values = [torch.tensor([1.8e38, 1.8e38, 10.], requires_grad=True),
              torch.ones(3, requires_grad=True), torch.tensor(0., requires_grad=True),
              torch.tensor([0., 0., 0.], requires_grad=True),
              torch.ones(3, requires_grad=True), torch.tensor(0., requires_grad=True)]
    value = _collision(*values)
    assert value.item() == 0.
    value.backward()
    assert all(v.grad is not None and torch.equal(v.grad, torch.zeros_like(v)) for v in values)


@pytest.mark.parametrize("side", [1e-13, 1e13])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_collision_identical_small_and_large_boxes_are_one_with_finite_gradients(side, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal device")
    values = [torch.zeros(3, requires_grad=True, device=device),
              torch.full((3,), side, requires_grad=True, device=device),
              torch.tensor(0., requires_grad=True, device=device)]
    others = [value.detach().clone().requires_grad_() for value in values]
    actual = _collision(*values, *others)
    assert actual.device.type == device
    if device == "mps":
        assert actual.dtype == values[0].dtype
    assert actual.item() == pytest.approx(1., abs=2e-6)
    actual.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in values + others)


def test_collision_non_degenerate_numerical_gradients():
    values = [torch.tensor([.2, .3, .1], dtype=torch.double, requires_grad=True),
              torch.tensor([2.3, 1.1, 1.2], dtype=torch.double, requires_grad=True),
              torch.tensor(.31, dtype=torch.double, requires_grad=True)]
    others = [torch.zeros(3, dtype=torch.double), torch.tensor([2., 1.3, 1.1], dtype=torch.double),
              torch.tensor(-.12, dtype=torch.double)]
    assert torch.autograd.gradcheck(lambda *args: _collision(*args, *others), tuple(values))
