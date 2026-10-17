import math

import pytest
import torch

from fastfill.v2.geometry import bottom_to_center, box_corners, decode_yaw, encode_yaw, wrap_yaw
from fastfill.v2.schema import normalize_room, validate_condition, validate_layout


def condition():
    return {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "floor_z_m": 0., "height_m": 2.8}, "objects": [
        {"id": "a", "category": "chair", "description": "chair"}], "constraints": []}


@pytest.mark.parametrize("angle", [-math.pi, math.pi, 0, 2 * math.pi, math.radians(97), -1e-8])
def test_yaw_roundtrip(angle):
    y = torch.tensor(angle, dtype=torch.float64)
    k, r = encode_yaw(y, 12)
    logits = torch.nn.functional.one_hot(k, 12).double() * 100
    residuals = torch.zeros(12, dtype=torch.float64).scatter(0, k.reshape(1), r.reshape(1))
    actual = decode_yaw(logits, residuals)
    assert abs(wrap_yaw(actual - y).item()) < 1e-10
    assert -1 - 1e-8 <= r.item() <= 1 + 1e-8


def test_local_full_size_corners_and_own_height():
    p = torch.tensor([1., 2., 0.])
    s = torch.tensor([2., 1., 3.])
    c = box_corners(p, s, torch.tensor(math.pi / 2))
    assert torch.allclose(c.max(0).values - c.min(0).values, torch.tensor([1., 2., 3.]))
    assert torch.equal(s, torch.tensor([2., 1., 3.]))
    assert bottom_to_center(p, s)[2].item() == 1.5
    assert bottom_to_center(p, s * 2)[2].item() == 3


def test_protocol_rejects_zero_nan_duplicate_and_unknown_fields():
    c = condition()
    validate_condition(c)
    good = {"schema_version": "fastfill.v2", "objects": [{"id": "a", "target_size_local_m": [1, 1, 1],
              "bottom_center_m": [1, 1, 0], "yaw_rad": 0}]}
    validate_layout(good, c)
    for bad in [dict(good, objects=good["objects"] * 2), dict(good, ignored=True),
                dict(good, objects=[dict(good["objects"][0], target_size_local_m=[0, 1, 1])]),
                dict(good, objects=[dict(good["objects"][0], yaw_rad=float("nan"))])]:
        with pytest.raises(ValueError):
            validate_layout(bad, c)


def test_normalization_is_condition_only():
    c = condition()
    origin, scale = normalize_room(c["room"])
    assert origin == [0., 0., 0.]
    assert scale == [5., 4., 2.8]
    no_height = dict(c["room"], height_m=None)
    assert normalize_room(no_height)[1][2] == 3.
