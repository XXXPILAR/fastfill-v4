"""Round 4 (L1): grid_residual matching charges the loss's own per-pair position terms."""
from dataclasses import replace
import itertools
import math

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from fastfill.v2.geometry import decode_grid_position
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch

# Review counterexample on a 2 x 2 grid: both slots argmax cell 0 (same decoded position);
# slot 1 also bets on cell 2, where the second target lies.
COUNTEREXAMPLE = [[5., 0., 0., 0.], [2., 0., 1.5, 0.]]


def group_batch(targets):
    """One request of n identical exchangeable chairs at the given normalized bottom centres."""
    pos = torch.tensor(targets, dtype=torch.float32)[None]
    n = pos.shape[1]
    return {"slot_mask": torch.ones(1, n, dtype=torch.bool),
            "targets": {"position_normalized": pos, "size": torch.ones(1, n, 3), "yaw": torch.zeros(1, n)},
            "validity": {"position": torch.ones(1, n, 3, dtype=torch.bool), "size": torch.ones(1, n, 3, dtype=torch.bool),
                         "yaw": torch.ones(1, n, dtype=torch.bool)},
            "objects": [[{"id": f"chair_{i}", "category": "chair", "description": "chair"} for i in range(n)]],
            "exchangeable_group": [["g"] * n], "conditions": [{"constraints": []}],
            "origin": torch.zeros(1, 3), "scale": torch.ones(1, 3)}


def grid_prediction(logits, residuals=None, z=None):
    """Grid-head outputs; position_normalized XY is the decoded head, as the model returns it."""
    logits = torch.as_tensor(logits, dtype=torch.float32)[None]
    n, cells = logits.shape[1:]
    residuals = torch.zeros(1, n, cells, 2) if residuals is None else residuals
    z = torch.zeros(1, n) if z is None else z
    xy = decode_grid_position(logits, residuals, math.isqrt(cells))
    return {"position_cell_logits": logits, "position_cell_residuals": residuals,
            "position_normalized": torch.cat((xy, z[..., None]), -1), "size": torch.ones(1, n, 3),
            "yaw_logits": torch.zeros(1, n, 12), "yaw_residuals": torch.zeros(1, n, 12),
            "slot_mask": torch.ones(1, n, dtype=torch.bool)}


def flip_slots(p):
    return {k: v if k == "slot_mask" else v.flip(1) for k, v in p.items()}


def permute_targets(b, order):
    return {**b, "targets": {k: v[:, list(order)] for k, v in b["targets"].items()}}


def test_review_counterexample_assignment_minimises_the_charged_loss():
    b = group_batch([[.2, .3, 0.], [.8, .3, 0.]])  # cells 0 and 2
    p = grid_prediction(COUNTEREXAMPLE)
    criterion = GeometryCriterion()
    base = criterion(p, b)["loss"].item()
    renamed = {**b, "objects": [[{**o, "id": f"stool_{9 - i}"} for i, o in enumerate(b["objects"][0])]]}
    for result in (criterion(flip_slots(p), b), criterion(p, permute_targets(b, [1, 0])), criterion(p, renamed)):
        assert result["loss"].item() == pytest.approx(base, abs=1e-6)
    # Tied decoded positions no longer fall back to SciPy's row order; default LossConfig() without loss_config.
    assert match_batch(flip_slots(p), b).tolist() == [[1, 0]]
    forced = [GeometryCriterion(LossConfig(hungarian=False))(p, permute_targets(b, order))["loss"].item()
              for order in ([0, 1], [1, 0])]
    assert base == pytest.approx(min(forced), abs=1e-6) and min(forced) < max(forced) - 1


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("fixed_z", [False, True])
@pytest.mark.parametrize("cfg", [LossConfig(), LossConfig(position_type="smooth_l1", smooth_l1_beta=.05, position_cell=.3,
                                                          position_residual=4., alpha_position=2.)])
def test_grid_assignment_is_the_charged_minimum_over_all_permutations(seed, fixed_z, cfg):
    g = torch.Generator().manual_seed(seed)
    n, grid = 3, 4
    b = group_batch(torch.rand(n, 3, generator=g).tolist())
    if fixed_z:  # the loss charges no z; neither may the matching
        b["fixed_position_mask"] = torch.tensor([[[False, False, True]] * n])
    p = grid_prediction(torch.randn(n, grid * grid, generator=g) * 2,
                        torch.rand(1, n, grid * grid, 2, generator=g) * 2 - 1, torch.rand(1, n, generator=g) * 3)
    charged = lambda order, c: GeometryCriterion(c)(p, permute_targets(b, order))["position"].item()
    forced = [charged(order, replace(cfg, hungarian=False)) for order in itertools.permutations(range(n))]
    assert charged(range(n), cfg) == pytest.approx(min(forced), abs=1e-6)


def test_fixed_z_is_not_charged_by_the_matching():
    # XY prefers identity by 0.2 nats; the z error, uncharged by the loss when fixed, would prefer the swap by 2.
    b = group_batch([[.2, .3, 0.], [.8, .3, 3.]])
    p = grid_prediction([[1., 0., .9, 0.], [.9, 0., 1., 0.]], z=torch.tensor([[3., 0.]]))
    assert match_batch(p, b).tolist() == [[1, 0]]
    b["fixed_position_mask"] = torch.tensor([[[False, False, True]] * 2])
    assert match_batch(p, b).tolist() == [[0, 1]]


def test_identical_predictions_cost_the_same_under_any_assignment():
    b = group_batch([[.2, .3, 0.], [.8, .3, 0.], [.6, .9, .1]])
    p = grid_prediction([[2., 0., 1.5, .5]] * 3, torch.tensor([[.3, -.2]]).expand(1, 3, 4, 2).clone())
    losses = [GeometryCriterion(LossConfig(hungarian=hungarian))(p, permute_targets(b, order))["loss"].item()
              for order in itertools.permutations(range(3)) for hungarian in (True, False)]
    assert max(losses) - min(losses) < 1e-6


def head_assignment(p, b, alpha_size):
    """Frozen pre-round-4 match_batch cost for one fully valid group: decoded L1 + log-size L1 (swap minimum)."""
    pos, gt = p["position_normalized"][0], b["targets"]["position_normalized"][0]
    size, gt_size = p["size"][0].log(), b["targets"]["size"][0].log()
    size_cost = (size[:, None] - gt_size[None]).abs().sum(-1)
    swapped = (size[:, None] - gt_size[None, :, [1, 0, 2]]).abs().sum(-1)
    size_cost = torch.where(b["size_axis_swap_allowed"][0][None], torch.minimum(size_cost, swapped), size_cost)
    _, cols = linear_sum_assignment(((pos[:, None] - gt[None]).abs().sum(-1) + alpha_size * size_cost).double().numpy())
    return [cols.tolist()]


@pytest.mark.parametrize("seed", range(8))
def test_regression_head_assignment_is_unchanged(seed):
    g = torch.Generator().manual_seed(seed)
    n = 4
    b = group_batch(torch.rand(n, 3, generator=g).tolist())
    b["targets"]["size"] = torch.rand(1, n, 3, generator=g) + .2
    b["size_axis_swap_allowed"] = torch.rand(1, n, generator=g) < .5
    p = {"position_normalized": torch.rand(1, n, 3, generator=g), "size": torch.rand(1, n, 3, generator=g) + .2,
         "yaw_logits": torch.zeros(1, n, 12), "yaw_residuals": torch.zeros(1, n, 12), "slot_mask": b["slot_mask"]}
    expected = head_assignment(p, b, .7)
    cfg = LossConfig(alpha_size=.7, position_type="smooth_l1", position_cell=.2)  # grid-only settings are ignored
    assert match_batch(p, b, alpha_size=.7).tolist() == expected
    assert match_batch(p, b, True, 1., .7, loss_config=cfg).tolist() == expected
    assert GeometryCriterion(cfg)(p, b)["assignment"].tolist() == expected
