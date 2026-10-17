"""Round 4 (L3): spread decoding honours declared supports (requests, fixed objects, "floor", "wall") and fixed z."""
import numpy as np
import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.evaluate import serialize_predictions, spread_grid_xy
from fastfill.v2.tests.test_spread_decode import ROOM, _logits
from fastfill.v2.validation import validate_scene

TABLE = {"id": "table", "category": "table", "size_local_m": [1.2, .8, 1.], "bottom_center_m": [1., 1., 0.], "yaw_rad": 0.,
         "support_surfaces": [{"surface_id": "top", "local_polygon_xy_m": [[-.6, -.4], [.6, -.4], [.6, .4], [-.6, .4]],
                               "local_z_m": 1.}]}


def _fixed(ident, xy, size, z=0.):
    return {"id": ident, "size_local_m": list(size), "bottom_center_m": [*xy, z], "yaw_rad": 0.}


def _requests(*parents):
    return [{"id": f"o{i}", "support_parent": parent} for i, parent in enumerate(parents)]


def test_review_cup_on_a_fixed_table_keeps_its_valid_raw_placement():
    # the review's counterexample: raw [1, 1, 1] validates; spread used to drop the cup to [1, 1, 0]
    condition = {"schema_version": "fastfill.v2", "constraints": [],
                 "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
                          "floor_z_m": 0., "floor_known": True, "height_m": 3., "boundary_known": True, "fixed_objects": [TABLE]},
                 "objects": [{"id": "cup", "category": "cup", "description": "a cup", "support_parent": "table"}]}
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=1 << 15)
    grid = 6  # cell (1, 1) of a 6 x 6 grid over the 4 m room is centred on (1, 1)
    logits = torch.full((1, 1, grid * grid), -5.); logits[0, 0, 1 * grid + 1] = 5.
    predictions = {"position_normalized": torch.tensor([[[.25, .25, 1 / 3]]]), "size": torch.full((1, 1, 3), .1),
                   "yaw_logits": torch.tensor([[[3.] + [0.] * 11]]), "yaw_residuals": torch.zeros(1, 1, 12),
                   "position_cell_logits": logits, "position_cell_residuals": torch.zeros(1, 1, grid * grid, 2)}
    for decode in ("spread", "argmax"):
        layout = serialize_predictions(predictions, batch, grid_decode=decode)[0]
        assert layout["objects"][0]["bottom_center_m"] == pytest.approx([1., 1., 1.], abs=1e-6)
        checks = validate_scene(condition, layout["objects"])["checks"]
        assert [c["status"] for c in checks if c["code"] == "support"] == ["pass"]
        assert not [c for c in checks if c["code"] in ("fixed_collision", "collision")]


def test_a_cup_declared_on_a_requested_table_follows_that_table_wherever_it_was_spread():
    logits = torch.stack((_logits(5, 10), _logits(5, 10), _logits(5, 10)))  # two identical tables, the cup prefers cell 5
    size = np.array([[1., 1., 1.], [1., 1., 1.], [.1, .1, .1]])
    position = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]])
    xy, _, z = spread_grid_xy(logits, torch.zeros(3, 16, 2), 4, size, np.zeros(3), position, ROOM,
                              requests=_requests(None, None, "o1"))
    assert np.allclose(xy[0], [1.5, 1.5]) and np.allclose(xy[1], [2.5, 2.5])
    assert np.allclose(xy[2], xy[1]) and np.isclose(z[2], 1.)  # on o1, not on the table under its argmax cell


def test_floor_declared_and_z_fixed_objects_stand_where_the_request_says():
    logits = torch.stack((_logits(5), _logits(5, 10)))  # a 1 m table, then a box predicted 0.9 m up over that table
    size = np.array([[1., 1., 1.], [.4, .4, .4]])
    position = np.array([[0., 0., 0.], [0., 0., .9]])
    xy, _, z = spread_grid_xy(logits, torch.zeros(2, 16, 2), 4, size, np.zeros(2), position, ROOM,
                              requests=_requests(None, "floor"))
    assert np.allclose(z, [0., 0.]) and np.allclose(xy[1], [2.5, 2.5])  # on the floor, beside the table
    _, _, z = spread_grid_xy(_logits(5)[None], torch.zeros(1, 16, 2), 4, np.full((1, 3), .4), np.zeros(1),
                             np.array([[0., 0., .5]]), ROOM, fixed_z=[True])
    assert np.isclose(z[0], .5)


def test_fixed_objects_block_cells_and_floor_standing_ones_hold_undeclared_raised_items():
    room = {**ROOM, "fixed_objects": [_fixed("column", (1.5, 1.5), (.4, .4, 2.5)), _fixed("desk", (3.5, 3.5), (1., 1., .75)),
                                      _fixed("wall_shelf", (3.5, 3.5), (1., 1., .3), z=1.8)]}
    logits = torch.stack((_logits(5, 10), _logits(15)))  # a box whose argmax cell holds the column; a cup over the desk
    size = np.array([[1., 1., .8], [.1, .1, .1]])
    xy, _, z = spread_grid_xy(logits, torch.zeros(2, 16, 2), 4, size, np.zeros(2), np.array([[0., 0., 0.], [0., 0., .7]]), room)
    assert np.allclose(xy[0], [2.5, 2.5]) and np.allclose(xy[1], [3.5, 3.5]) and np.allclose(z, [0., .75])


def test_with_no_top_k_cell_on_the_parent_a_declared_item_takes_the_parent_point_nearest_its_argmax():
    logits = torch.stack((_logits(5), _logits(0, 15)))  # a 1 m table on cell 5; the cup's cells 0 and 15 miss it
    size = np.array([[1., 1., 1.], [.1, .1, .1]])
    xy, _, z = spread_grid_xy(logits, torch.zeros(2, 16, 2), 4, size, np.zeros(2), np.array([[0., 0., 0.], [0., 0., 1.]]),
                              ROOM, requests=_requests(None, "o0"), top_k=2)
    assert np.allclose(xy[1], [1.05, 1.05]) and np.isclose(z[1], 1.)  # the table corner, inset by the cup's half size


def test_an_undeclared_object_no_cell_can_hold_takes_the_least_overlapping_cell_on_the_floor():
    # a 2 x 2 m box fills cell 5; the 1 m box's two cells overlap it fully / by 25%: cell 0 at floor height,
    # as before round 4 (round 4 had kept the raw cell 5 and z 0.02)
    logits = torch.stack((_logits(5), _logits(5, 0)))
    size = np.array([[2., 2., 1.], [1., 1., 1.]])
    xy, _, z = spread_grid_xy(logits, torch.zeros(2, 16, 2), 4, size, np.zeros(2),
                              np.array([[0., 0., 0.], [0., 0., .02]]), ROOM, top_k=2)
    assert np.allclose(xy[1], [.5, .5]) and np.isclose(z[1], 0.)


def test_declared_children_stay_on_a_full_parent_and_spread_over_its_top():
    # the review's counterexample: two 0.3 m cups on a 0.5 x 0.5 m fixed table (1 m top), raw z 0.97; round 4
    # left the second cup at its raw decode, sunk into the table (support violation and fixed_collision)
    table = {**TABLE, "size_local_m": [.5, .5, 1.], "bottom_center_m": [1.5, 1.5, 0.],
             "support_surfaces": [{"surface_id": "top", "local_polygon_xy_m": [[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25]],
                                   "local_z_m": 1.}]}
    condition = {"schema_version": "fastfill.v2", "constraints": [],
                 "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
                          "floor_z_m": 0., "floor_known": True, "height_m": 3., "boundary_known": True, "fixed_objects": [table]},
                 "objects": [{"id": f"cup{i}", "category": "cup", "description": "a cup", "support_parent": "table"} for i in range(2)]}
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=1 << 15)
    logits = torch.full((1, 2, 16), -5.); logits[0, :, 5] = 5.  # both cups bet on the table cell (1.5, 1.5)
    predictions = {"position_normalized": torch.tensor([[[1.5 / 4, 1.5 / 4, .97 / 3]] * 2]), "size": torch.full((1, 2, 3), .3),
                   "yaw_logits": torch.tensor([[[3.] + [0.] * 11] * 2]), "yaw_residuals": torch.zeros(1, 2, 12),
                   "position_cell_logits": logits, "position_cell_residuals": torch.zeros(1, 2, 16, 2)}
    objects = serialize_predictions(predictions, batch)[0]["objects"]
    assert [o["bottom_center_m"][2] for o in objects] == pytest.approx([1., 1.])
    checks = validate_scene(condition, objects)["checks"]
    assert [c["status"] for c in checks if c["code"] == "support"] == ["pass", "pass"]
    assert not [c for c in checks if c["code"] == "fixed_collision"]
    # a larger table holds both cups apart: the second takes the free lattice point nearest its argmax
    xy, _, z = spread_grid_xy(logits[0], torch.zeros(2, 16, 2), 4, np.full((2, 3), .3), np.zeros(2),
                              np.array([[0., 0., .97]] * 2), {**ROOM, "fixed_objects": [{**table, "size_local_m": [1., 1., 1.]}]},
                              requests=[{"id": f"cup{i}", "support_parent": "table"} for i in range(2)])
    assert np.allclose(xy, [[1.5, 1.5], [1.15, 1.5]]) and np.allclose(z, 1.)


def test_a_wall_declared_item_keeps_its_predicted_height():
    # a painting at 1.5 m declared on the wall, above a 0.75 m desk: round 4 put it on the floor (or the desk)
    for fixed in ([], [_fixed("desk", (1.5, .5), (1., .6, .75))]):
        _, _, z = spread_grid_xy(_logits(4)[None], torch.zeros(1, 16, 2), 4, np.array([[.8, .05, .6]]), np.zeros(1),
                                 np.array([[0., 0., 1.5]]), {**ROOM, "fixed_objects": fixed}, requests=_requests("wall"))
        assert np.isclose(z[0], 1.5)


def test_inconsistent_support_inputs_are_rejected():
    args = (_logits(5)[None], torch.zeros(1, 16, 2), 4, np.ones((1, 3)), np.zeros(1), np.zeros((1, 3)), ROOM)
    with pytest.raises(ValueError, match="one entry per slot"):
        spread_grid_xy(*args, fixed_z=[True, False])
    with pytest.raises(ValueError, match="unknown support parent"):
        spread_grid_xy(*args, requests=_requests("ghost"))
    with pytest.raises(ValueError, match="cycle"):
        spread_grid_xy(*args, requests=_requests("o0"))
