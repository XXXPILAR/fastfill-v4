"""Round 5 (E5): spread decoding keeps the yaw of objects whose orientation the request constrains."""
import math

import numpy as np
import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.evaluate import serialize_predictions, spread_grid_xy
from fastfill.v2.tests.test_evaluate_fix20261006 import row
from fastfill.v2.tests.test_spread_decode import ROOM, _logits
from fastfill.v2.validation import validate_scene


def _decode(sample, cells, yaw_bins):
    """Spread decode of chairs (.6 x .6 x 1) in the 4 x 4 m room: each takes its 4 x 4 grid cell (cell 1 is
    centred on (.5, 1.5), cell 2 on (.5, 2.5): both 20 cm from the x = 0 wall); its yaw head puts 10 on its bin
    of 12 and 5 on bin 0 (0 rad, facing away from that wall)."""
    n = len(cells)
    logits = torch.full((1, n, 16), -20.)
    yaw_logits = torch.full((1, n, 12), -10.)
    for i, (cell, b) in enumerate(zip(cells, yaw_bins)):
        logits[0, i, cell] = 10.
        yaw_logits[0, i, 0], yaw_logits[0, i, b] = 5., 10.
    predictions = {"position_normalized": torch.tensor([[[.125, .125 + .25 * (c % 4), 0.] for c in cells]]),
                   "size": torch.tensor([[[.6, .6, 1.]] * n]), "yaw_logits": yaw_logits, "yaw_residuals": torch.zeros(1, n, 12),
                   "position_cell_logits": logits, "position_cell_residuals": torch.zeros(1, n, 16, 2)}
    batch = collate_samples([sample], TinyTokenizer(), max_length=10000, max_objects=10)
    objects = serialize_predictions(predictions, batch, grid_decode="spread")[0]["objects"]
    return objects, validate_scene(sample["condition"], objects)["checks"]


def test_review_hard_faces_direction_keeps_the_raw_yaw_and_an_unconstrained_twin_still_turns_from_the_wall():
    # the review's declared_facing case: raw yaw pi satisfies the hard faces_direction [-1, 0]; spread used to turn it to 0
    sample = row([("chair", [.5, 2.5, 0], [.6, .6, 1], -math.pi), ("chair", [.5, 1.5, 0], [.6, .6, 1], -math.pi)])
    sample["condition"]["constraints"] = [{"type": "faces_direction", "object_id": "obj_0000", "direction_xy": [-1, 0], "hard": True}]
    objects, checks = _decode(sample, cells=(2, 1), yaw_bins=(6, 6))
    assert abs(abs(objects[0]["yaw_rad"]) - math.pi) < 1e-6 and objects[0]["bottom_center_m"][:2] == pytest.approx([.5, 2.5])
    # the twin: back to the wall (its float32-decoded pi flipped by pi, so within float32 rounding of 0)
    assert objects[1]["yaw_rad"] == pytest.approx(0., abs=1e-6)
    assert [c["status"] for c in checks if c["code"] == "constraint"] == ["pass"]
    assert not [c for c in checks if c["status"] == "violation"]


def test_a_soft_faces_constraint_also_keeps_the_raw_yaw():
    # obj_0000 faces obj_0001 (2 m along -y, i.e. yaw -pi/2 = bin 9); the wall rule would turn it to 0
    sample = row([("chair", [.5, 2.5, 0], [.6, .6, 1], -math.pi / 2), ("chair", [.5, .5, 0], [.6, .6, 1], 0.)])
    sample["condition"]["constraints"] = [{"type": "faces", "object_id": "obj_0000", "target_id": "obj_0001", "hard": False}]
    objects, checks = _decode(sample, cells=(2, 0), yaw_bins=(9, 0))
    assert objects[0]["yaw_rad"] == pytest.approx(-math.pi / 2, abs=1e-6)
    assert [c["status"] for c in checks if c["code"] == "constraint"] == ["pass"]


def test_keep_yaw_needs_one_entry_per_slot():
    with pytest.raises(ValueError, match="keep_yaw"):
        spread_grid_xy(_logits(0)[None], torch.zeros(1, 16, 2), 4, np.full((1, 3), .4), np.zeros(1), np.zeros((1, 3)), ROOM,
                       keep_yaw=[True, False])
