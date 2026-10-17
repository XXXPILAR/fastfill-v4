"""Audit C2: the uniform-yaw baseline is the model's own yaw metric (``reference_metrics``) applied to a uniformly
random yaw with a perfect size, also for swap-allowed objects (box-equivalent joint minimum), in evaluate and stratify."""
import math

import numpy as np
import pytest

from fastfill.v2.evaluate import baseline_metrics, fit_baselines, reference_metrics
from fastfill.v2.tests.test_evaluate_fix20261006 import row


def _sample(size, swap, size_valid, n=1):
    sample = row([("chair", [2., 2., 0.], size, .3)] * n, orders=[4] * n)
    sample["validity"].update(size_axis_swap_allowed=[swap] * n, size=[[size_valid] * 3] * n)
    return sample


@pytest.mark.parametrize("size, swap, size_valid", [
    ([10., 1., 1.], True, True),  # the audit's box: before pi / 8 = 0.3927, the model's metric gives 0.7677
    ([2., 1., 1.], True, True), ([1., 1., 1.], True, True), ([10., 1., 1.], True, False), ([10., 1., 1.], False, True)])
def test_uniform_yaw_baseline_is_the_model_metric_of_a_uniform_yaw(monkeypatch, size, swap, size_valid):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")  # stratify's import-time setdefault must not leak into later tests
    from fastfill.v2.ops.stratify import object_baselines
    # reference_metrics itself, label copies predicted with their own size on a fixed midpoint grid of yaws; the joint
    # minimum's yaw error jumps where the quarter turn wins, so the grid mean is only O(1 / n) close
    n = 10000
    grid = _sample(size, swap, size_valid, n)
    yaws = .3 + 2 * math.pi * (np.arange(n) + .5) / n
    layout = {"objects": [{**t, "yaw_rad": float(y)} for t, y in zip(grid["target"]["objects"], yaws)]}
    expected = reference_metrics(layout, grid, hungarian=False, include_iou=False)["yaw_error_rad"]["mean"]
    if size == [10., 1., 1.] and swap and size_valid:
        assert expected == pytest.approx(.7677, abs=4e-4)
    sample = _sample(size, swap, size_valid)
    baseline = baseline_metrics(sample, fit_baselines([sample]))["uniform_yaw"]
    assert baseline["yaw_error_rad"]["mean"] == pytest.approx(expected, abs=4e-4)
    per_object, summary = object_baselines(sample, fit_baselines([sample]), None)  # stratify reproduces it exactly
    assert per_object["obj_0000"]["uniform_yaw"] == baseline["yaw_error_rad"]["mean"] and summary["uniform_yaw"] == baseline
