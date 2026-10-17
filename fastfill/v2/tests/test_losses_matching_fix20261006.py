"""Audit 2026-10-06 fixes: group source/cost (C2), loss types and window sums (C8), yaw bin dtype."""
import math

import pytest
import torch

from fastfill.v2.geometry import decode_yaw, encode_yaw, wrap_yaw
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch
from fastfill.v2.tests.test_supervision import batch, prediction


# --- geometry: bin centres in the input dtype -------------------------------

@pytest.mark.parametrize("degrees", [15, 45, -15, 165, 345])
def test_encode_yaw_accepts_float64_bin_boundaries(degrees):
    y = torch.tensor([math.radians(degrees)], dtype=torch.float64)
    k, r = encode_yaw(y, 12)
    assert r.abs().item() == pytest.approx(1., abs=1e-12)
    logits = torch.nn.functional.one_hot(k, 12).double() * 100
    residuals = torch.zeros(1, 12, dtype=torch.float64).scatter(1, k[:, None], r[:, None])
    assert abs(wrap_yaw(decode_yaw(logits, residuals) - y).item()) < 1e-12


# --- matching: groups from batch["exchangeable_group"], position-first cost ---

def test_groups_come_from_batch_not_from_objects():
    b = batch()
    p = prediction(b)
    b["objects"] = [[{**o, "exchangeable_group": "g"} for o in b["objects"][0]]]
    b["exchangeable_group"] = [[None, None]]
    assert match_batch(p, b).tolist() == [[0, 1]]
    b["objects"] = [[{k: v for k, v in o.items() if k != "exchangeable_group"} for o in b["objects"][0]]]
    b["exchangeable_group"] = [["g", "g"]]
    assert match_batch(p, b).tolist() == [[1, 0]]


def test_matching_requires_group_list_when_enabled():
    b = batch()
    del b["exchangeable_group"]
    with pytest.raises(ValueError, match="exchangeable_group"):
        match_batch(prediction(b), b)
    assert match_batch(prediction(b), b, enabled=False).tolist() == [[0, 1]]
    b["exchangeable_group"] = [["g"]]
    with pytest.raises(ValueError, match="per request slot"):
        match_batch(prediction(b), b)


def test_group_without_complete_size_is_matched_on_position_only():
    b = batch()
    b["validity"]["size"] = torch.tensor([[[True, True, True], [True, False, True]]])
    b["targets"]["size"] = torch.tensor([[[1., 1., 1.], [1., float("nan"), 1.]]])
    p = prediction(b)
    # Size cost alone would prefer identity; position must still swap.
    p["size"] = torch.tensor([[[1., 1., 1.], [8., 8., 8.]]]).requires_grad_()
    assert match_batch(p, b, alpha_position=1., alpha_size=1.).tolist() == [[1, 0]]


def test_group_without_complete_position_keeps_fixed_identity():
    b = batch()
    b["validity"]["position"] = torch.tensor([[[True, True, True], [True, True, False]]])
    b["targets"]["position_normalized"][0, 1, 2] = float("nan")
    assert match_batch(prediction(b), b).tolist() == [[0, 1]]


# --- losses: l1 / smooth_l1 and beta ----------------------------------------

@pytest.mark.parametrize("field", ["position_type", "size_type"])
@pytest.mark.parametrize("invalid", ["l2", "huber", "", None, 1])
def test_loss_types_are_validated(field, invalid):
    with pytest.raises(ValueError, match="position_type/size_type"):
        LossConfig(**{field: invalid})


@pytest.mark.parametrize("invalid", [0, -1., True, "1", None, float("nan"), float("inf")])
def test_smooth_l1_beta_is_validated(invalid):
    with pytest.raises(ValueError, match="smooth_l1_beta"):
        LossConfig(smooth_l1_beta=invalid)


@pytest.mark.parametrize("kind,beta,expected", [("l1", 1., 1.), ("smooth_l1", 1., .5), ("smooth_l1", 4., 1 / 8)])
def test_position_loss_type_and_beta(kind, beta, expected):
    b = batch()
    p = prediction(b)
    p["position_normalized"] = (b["targets"]["position_normalized"] + 1).requires_grad_()
    cfg = LossConfig(hungarian=False, position_type=kind, smooth_l1_beta=beta)
    result = GeometryCriterion(cfg)(p, b)
    assert result["position"].item() == pytest.approx(expected)  # three coords, each error 1, /3


@pytest.mark.parametrize("kind,expected", [("l1", math.log(2)), ("smooth_l1", math.log(2) ** 2 / 2)])
def test_size_loss_type_uses_log_ratio(kind, expected):
    b = batch()
    p = prediction(b)
    p["size"] = torch.full((1, 2, 3), 2., requires_grad=True)
    result = GeometryCriterion(LossConfig(hungarian=False, size_type=kind))(p, b)
    assert result["size"].item() == pytest.approx(expected, rel=1e-5)


# --- losses: window-level sums and counts -----------------------------------

def test_term_sums_and_counts_cover_every_term():
    b = batch()
    b["conditions"] = [{"objects": b["objects"][0], "constraints": [], "room": {
        "floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]], "boundary_known": True}}]
    p = prediction(b)
    result = GeometryCriterion(LossConfig(box=1., collision=1., boundary=1.))(p, b)
    keys = {"position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary"}
    grid_terms = {"position_cell", "position_residual", "position_z"}  # regression head: present, zero count
    assert set(result["term_sums"]) == keys | grid_terms and set(result["term_counts"]) == keys | grid_terms
    assert result["term_counts"] == {"position": 2, "size": 2, "yaw_cls": 2, "yaw_reg": 2, "box": 2, "collision": 1, "boundary": 2,
                                     "position_cell": 0, "position_residual": 0, "position_z": 0}
    for key in keys:
        total = result["term_sums"][key]
        assert not total.requires_grad and total.dtype == torch.float32
        assert total.item() == pytest.approx(result[key].item() * result["term_counts"][key], rel=1e-5, abs=1e-6)


def test_term_sums_are_zero_with_zero_count_when_terms_are_disabled():
    b = batch()
    result = GeometryCriterion(LossConfig(hungarian=False))(p := prediction(b), b)
    assert result["term_counts"]["box"] == 0 and result["term_sums"]["box"].item() == 0
    assert result["term_counts"]["collision"] == 0 and result["term_sums"]["collision"].item() == 0
    assert result["term_sums"]["position"].item() == pytest.approx(result["position"].item() * 2)
    assert result["loss"].requires_grad and p["position_normalized"].requires_grad
