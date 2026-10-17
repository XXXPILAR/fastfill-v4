"""Typed loss and matcher options must preserve the requested experiment."""
import pytest

from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch
from fastfill.v2.tests.test_supervision import batch, prediction


@pytest.mark.parametrize("option", ["position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary"])
@pytest.mark.parametrize("invalid", [True, False, "1", None, float("nan"), float("inf"), -1.])
def test_loss_weights_reject_non_native_invalid_numbers(option, invalid):
    with pytest.raises(ValueError, match="weights|finite|number"):
        GeometryCriterion(LossConfig(**{option: invalid}))


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_hungarian_option_must_be_boolean_at_loss_configuration_boundary(invalid):
    with pytest.raises(ValueError, match="boolean"):
        LossConfig(hungarian=invalid)


@pytest.mark.parametrize("option", ["alpha_position", "alpha_size"])
@pytest.mark.parametrize("invalid", [True, False, "1", None, float("nan"), float("inf"), -1.])
def test_matching_weights_fail_at_loss_configuration_boundary(option, invalid):
    with pytest.raises(ValueError, match="matching weights|finite|number"):
        LossConfig(**{option: invalid})


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1, None])
def test_direct_matcher_rejects_nonboolean_enable(invalid):
    b = batch()
    with pytest.raises(ValueError, match="boolean"):
        match_batch(prediction(b), b, enabled=invalid)


@pytest.mark.parametrize("option", ["alpha_position", "alpha_size"])
@pytest.mark.parametrize("invalid", [True, False, "1", None])
def test_direct_matcher_rejects_nonnumeric_or_boolean_weights(option, invalid):
    b = batch()
    with pytest.raises(ValueError, match="matching weights|number"):
        match_batch(prediction(b), b, **{option: invalid})


def test_zero_matching_weights_fail_but_individual_zero_and_boolean_flags_are_valid():
    with pytest.raises(ValueError, match="both zero"):
        LossConfig(alpha_position=0, alpha_size=0)
    b = batch()
    fixed = GeometryCriterion(LossConfig(hungarian=False, alpha_position=0, alpha_size=1))(prediction(b), b)
    swapped = GeometryCriterion(LossConfig(hungarian=True, alpha_position=1, alpha_size=0))(prediction(b), b)
    assert fixed["assignment"].tolist() == [[0, 1]]
    assert swapped["assignment"].tolist() == [[1, 0]]
