"""Expected-geometry loss: per-cell boundary / door-clearance costs and their gradient on the cell logits."""
import numpy as np
import pytest
import torch

from fastfill.v2.batch import AUGMENT_DEFAULTS, augment_sample, collate_samples
from fastfill.v2.cell_costs import cell_cost_maps, cell_positions, door_zones
from fastfill.v2.evaluate import _collapse_counts
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.schema import validate_condition

GRID = 16
L_ROOM = [[0., 0.], [8., 0.], [8., 4.], [4., 4.], [4., 8.], [0., 8.]]  # L-shape: the x > 4, y > 4 quarter is outside
SQUARE = [[0., 0.], [4., 0.], [4., 4.], [0., 4.]]
DOOR = {"type": "door", "wall_index": 0, "position_on_wall": .5, "width_m": 1., "height_m": 2.}


def room(polygon, openings=()):
    return {"frame": "right_handed_z_up", "floor_polygon_xy_m": polygon, "floor_z_m": 0., "floor_known": True,
            "boundary_known": True, "height_m": 2.7, "openings": list(openings)}


def cell(x_index, y_index):
    return x_index * GRID + y_index


def test_cells_outside_an_l_shaped_room_cost_one_and_inside_cost_zero():
    costs, mask = cell_cost_maps(room(L_ROOM), [([.4, .4], 0., True)], {"grid": GRID})
    assert mask.tolist() == [True]
    assert costs[0, cell(14, 14)] == pytest.approx(1.)  # missing quarter of the L
    assert costs[0, cell(4, 4)] == 0. and costs[0, cell(14, 2)] == 0.


def test_a_wall_flush_label_cell_is_free_although_its_centre_would_cross_the_wall():
    # 4 m room, 0.25 m cells: a 2 m deep bed flush with the y=0 wall has its centre at y=1.0, on a cell border;
    # the y index 3 cell centre (0.875) would push it 0.125 m through the wall.
    r = room(SQUARE)
    costs, _ = cell_cost_maps(r, [([1.6, 2.], 0., True)], {"grid": GRID})
    centre_only, _ = cell_cost_maps(r, [([1.6, 2.], 0., True)], {"grid": GRID, "sub": 1})
    assert costs[0, cell(8, 3)] == 0.
    assert centre_only[0, cell(8, 3)] > 0.


def test_door_zone_costs_floor_objects_only():
    r = room(SQUARE, [DOOR])
    zone, = door_zones(r, .8)
    assert sorted(map(tuple, np.round(zone, 6))) == [(1.5, 0.), (1.5, .8), (2.5, 0.), (2.5, .8)]
    costs, _ = cell_cost_maps(r, [([.5, .5], 0., True), ([.5, .5], 0., False)], {"grid": GRID})
    in_front = cell(8, 1)  # x 2.0-2.25, y 0.25-0.5: right in front of the door
    assert costs[0, in_front] > .5 and costs[1, in_front] == 0.
    assert costs[0, cell(2, 12)] == 0.


def test_windows_and_clockwise_rooms():
    window = {**DOOR, "type": "window", "sill_height_m": .9}
    assert door_zones(room(SQUARE, [window]), .8) == []
    clockwise, = door_zones(room(SQUARE[::-1], [{**DOOR, "wall_index": 2}]), .8)  # same y=0 wall, reversed order
    assert clockwise[:, 1].min() == pytest.approx(0.) and clockwise[:, 1].max() == pytest.approx(.8)


def test_cell_positions_follow_the_grid_encoding():
    from fastfill.v2.geometry import encode_grid_position
    r = room(L_ROOM)
    centres = cell_positions(r, {"grid": GRID, "sub": 1})[:, 0]
    normalized = torch.tensor(centres / 8.)
    index, _ = encode_grid_position(normalized, GRID)
    assert index.tolist() == list(range(GRID * GRID))


def sample(r):
    condition = {"schema_version": "fastfill.v2", "room": r, "constraints": [],
                 "objects": [{"id": "bed", "category": "bed", "description": "bed", "support_parent": "floor"},
                             {"id": "art", "category": "painting", "description": "painting", "support_parent": "wall"}]}
    target = {"schema_version": "fastfill.v2",
              "objects": [{"id": "bed", "target_size_local_m": [1.6, 2., .5], "bottom_center_m": [2., 3., 0.], "yaw_rad": 0.},
                          {"id": "art", "target_size_local_m": [.8, .05, .6], "bottom_center_m": [2., 3.97, 1.2],
                           "yaw_rad": 0.}]}
    validity = {"position": [[True] * 3] * 2, "size": [[True] * 3] * 2, "yaw": [True, True], "yaw_symmetry_order": [2, 2],
                "size_axis_swap_allowed": [False, False], "exchangeable_group": [None, None]}
    return {"schema_version": "fastfill.v2", "condition": validate_condition(condition), "target": target, "validity": validity}


def collated(s, **kw):
    from fastfill.v2.batch import TinyTokenizer
    return collate_samples([s], TinyTokenizer(), max_length=10 ** 8, **kw)


def test_collate_adds_cost_rows_only_when_asked():
    s = sample(room(SQUARE, [DOOR]))
    plain = collated(s)
    assert "cell_costs" not in plain
    batch = collated(s, cell_costs={"grid": GRID})
    assert batch["cell_costs"].shape == (1, 2, GRID * GRID) and batch["cell_cost_mask"].tolist() == [[True, True]]
    in_front = cell(8, 1)
    assert batch["cell_costs"][0, 0, in_front] > 0 and batch["cell_costs"][0, 1, in_front] == 0  # the painting is not on the floor


def test_rotated_sample_costs_turn_with_the_room():
    s = sample(room(L_ROOM, [DOOR]))
    base = collated(s, cell_costs={"grid": GRID})["cell_costs"][0, 0].reshape(GRID, GRID)
    options = {**AUGMENT_DEFAULTS, "shuffle_objects": False, "minimal_form_p": 0., "drop_constraints_p": 0.,
               "drop_support_p": 0., "category_only_description_p": 0.}
    seen = set()
    for seed in range(12):
        turned = augment_sample(s, options, torch.Generator().manual_seed(seed))
        costs = collated(turned, cell_costs={"grid": GRID})["cell_costs"][0, 0].reshape(GRID, GRID)
        # raster pixels are aligned to the turned room's corner, so a few border footprint samples may flip (2/16 each)
        matches = [k for k in range(4) if torch.allclose(torch.rot90(base, k), costs, atol=.2)]
        assert matches, "costs must be a quarter turn of the unturned map"
        seen.update(matches)
    assert len(seen) > 1


def predictions_for(batch, logits):
    b, n = batch["slot_mask"].shape
    residuals = torch.zeros((b, n, GRID * GRID, 2))
    k = logits.detach().argmax(-1)
    xy = torch.stack((torch.div(k, GRID, rounding_mode="floor"), k % GRID), -1).float().add(.5).div(GRID)
    return {"position_normalized": torch.cat((xy, torch.zeros((b, n, 1))), -1), "size": batch["targets"]["size"].nan_to_num(1.),
            "yaw_logits": torch.zeros((b, n, 12)), "yaw_residuals": torch.zeros((b, n, 12)), "slot_mask": batch["slot_mask"],
            "position_cell_logits": logits, "position_cell_residuals": residuals}


def test_default_weight_leaves_the_objective_unchanged():
    batch = collated(sample(room(SQUARE, [DOOR])), cell_costs={"grid": GRID})
    logits = torch.randn((1, 2, GRID * GRID), generator=torch.Generator().manual_seed(0))
    off = GeometryCriterion(LossConfig())(predictions_for(batch, logits), batch)
    plain = collated(sample(room(SQUARE, [DOOR])))
    base = GeometryCriterion(LossConfig())(predictions_for(plain, logits), plain)
    assert torch.equal(off["loss"], base["loss"]) and "expected_geometry" not in off["term_sums"]


def test_gradient_lowers_the_probability_of_costly_cells():
    batch = collated(sample(room(L_ROOM, [DOOR])), cell_costs={"grid": GRID})
    logits = torch.zeros((1, 2, GRID * GRID), requires_grad=True)
    result = GeometryCriterion(LossConfig(position=0., size=0., yaw_cls=0., yaw_reg=0., expected_geometry=1.))(
        predictions_for(batch, logits), batch)
    result["loss"].backward()
    grad = logits.grad[0, 0]
    assert grad[cell(14, 14)] > 0 > grad[cell(4, 12)]  # descent lowers the outside cell, raises a free one
    expected = float(batch["cell_costs"][0, :2].mean(-1).mean())  # uniform logits: mean cost per object
    assert float(result["expected_geometry"]) == pytest.approx(expected, rel=1e-5)


def test_expected_geometry_requires_cost_rows():
    batch = collated(sample(room(SQUARE)))
    logits = torch.zeros((1, 2, GRID * GRID))
    with pytest.raises(ValueError, match="cell_costs"):
        GeometryCriterion(LossConfig(expected_geometry=1.))(predictions_for(batch, logits), batch)


def test_collapse_counts_report_footprint_and_door_violations():
    r = room(SQUARE, [DOOR])
    boxes = [([2., .3, 0.], [1., .5, .8], 0.),     # floor, in front of the door
             ([2., 3., 0.], [1., .5, .8], 0.),     # floor, free
             ([3.9, 2., 0.], [1., .5, .8], 0.),    # half its footprint through the x=4 wall
             ([2., .3, 1.5], [.5, .05, .5], 0.)]   # on the wall above the door zone: not a floor object
    counts = _collapse_counts(boxes, ["a", "b", "c", "d"], r, [0., 0., 0.], [4., 4., 2.7])
    assert counts["floor_objects"] == 3 and counts["door_blocking_objects"] == 1
    assert counts["footprint_outside_objects"] == 1


def test_thin_floor_objects_never_block_doors():
    s = sample(room(SQUARE, [DOOR]))
    s["target"]["objects"][0]["target_size_local_m"] = [.5, .5, .02]  # a carpet
    in_front = cell(8, 1)
    assert collated(s, cell_costs={"grid": GRID})["cell_costs"][0, 0, in_front] == 0
    r = room(SQUARE, [DOOR])
    carpet = [([2., .3, 0.], [1., .5, .02], 0.)]
    assert _collapse_counts(carpet, ["carpet"], r, [0., 0., 0.], [4., 4., 2.7])["door_blocking_objects"] == 0


def test_a_label_already_in_a_costly_cell_gets_no_cost_row():
    s = sample(room(SQUARE, [DOOR]))
    s["target"]["objects"][0].update(target_size_local_m=[.6, .6, .8], bottom_center_m=[2., .4, 0.])  # blocks the door
    batch = collated(s, cell_costs={"grid": GRID})
    assert batch["cell_cost_mask"].tolist() == [[False, True]] and batch["cell_costs"][0, 0].abs().sum() == 0
    kept = collated(s, cell_costs={"grid": GRID, "drop_label_conflicts": False})
    assert kept["cell_cost_mask"].tolist() == [[True, True]] and kept["cell_costs"][0, 0, cell(8, 1)] > 0
