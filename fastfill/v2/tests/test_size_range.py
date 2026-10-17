from copy import deepcopy
import math

import pytest

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.size_range import size_target_conflicts
from fastfill.v2.tests.test_execution import sample
from fastfill.v2.train import _preflight, _training_config


def _batch(size, *, validity=None, fixed=None):
    row = deepcopy(sample())
    row["target"]["objects"][0]["target_size_local_m"] = size
    if validity is not None:
        row["validity"]["size"] = [validity]
    if fixed is not None:
        row["condition"]["objects"][0]["fixed_size_local_m"] = fixed
    return collate_samples([row], TinyTokenizer(), max_length=4096), row


@pytest.mark.parametrize("size", [[3e-9, 1., 1.], [1., 1e-7, 1.], [1., 1., 1e5]])
def test_enabled_size_targets_outside_head_range_are_identified(size):
    batch, _ = _batch(size)
    result = size_target_conflicts(batch, ModelConfig(), LossConfig())
    assert len(result) == 1
    assert result[0]["object_id"] == "a"
    assert result[0]["minimum_m"] == pytest.approx(math.exp(-10))
    assert result[0]["maximum_m"] == pytest.approx(math.exp(10))


def test_reference_and_custom_log_limit_define_per_axis_range():
    batch, _ = _batch([.01, 1., 1.])
    result = size_target_conflicts(batch, ModelConfig(size_reference=(2., .5, 3.), size_log_limit=2), LossConfig())
    assert result[0]["minimum_m"] == pytest.approx(2 * math.exp(-2))


@pytest.mark.parametrize("size", [[math.exp(-10), 1., 1.], [math.exp(10), 1., 1.]])
def test_floating_point_exact_head_boundaries_remain_eligible(size):
    batch, _ = _batch(size)
    assert size_target_conflicts(batch, ModelConfig(), LossConfig()) == []


def test_fixed_small_dimension_bypasses_learned_head_range():
    batch, _ = _batch([3e-9, 1., 1.], fixed=[3e-9, None, None])
    assert size_target_conflicts(batch, ModelConfig(), LossConfig()) == []


def test_incomplete_masked_nan_size_is_not_evaluated_as_complete_supervision():
    batch, _ = _batch([None, 1e-9, 1.], validity=[False, True, True])
    assert size_target_conflicts(batch, ModelConfig(), LossConfig()) == []


def test_size_disabled_and_box_disabled_has_no_size_range_requirement():
    batch, _ = _batch([3e-9, 1., 1.])
    assert size_target_conflicts(batch, ModelConfig(), LossConfig(size=0., box=0.)) == []


def test_enabled_box_also_needs_expressible_size():
    batch, _ = _batch([3e-9, 1., 1.])
    assert size_target_conflicts(batch, ModelConfig(), LossConfig(size=0., box=1.))


def test_real_preflight_reports_range_rejection_without_modifying_targets():
    _, out_of_range = _batch([3e-9, 1., 1.])
    before = deepcopy(out_of_range)
    kept, rejected = _preflight([out_of_range, sample()], TinyTokenizer(), ModelConfig(), _training_config({}))
    assert len(kept) == 1
    assert rejected[0]["reason"] == "size_target_outside_model_range"
    assert rejected[0]["coordinates"][0]["object_id"] == "a"
    assert out_of_range == before
