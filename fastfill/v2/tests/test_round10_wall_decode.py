"""Round 10 review: wall objects and inner-shelf items survive spread decoding, the hand-off, validate_scene's wall gap,
the structured LLM harness and rotate90 augmentation of rooms with openings."""
import json
import math

import numpy as np
import pytest
import torch

from fastfill.v2.batch import AUGMENT_DEFAULTS, augment_sample
from fastfill.v2.direct_layout import infer_support, layout_to_roomgenbench
from fastfill.v2.evaluate import spread_grid_xy
from fastfill.v2.llm_structured import STRUCTURED, check_answer
from fastfill.v2.validation import validate_scene

ROOM = {"origin": [0., 0., 0.], "scale": [4., 4., 2.7], "bounds": (np.zeros(2), np.full(2, 4.)), "floor_z": 0., "height_m": 2.7}
POLYGON = [[0., 0.], [4., 0.], [4., 4.], [0., 4.]]
GRID = 8


def spread(position, size, yaw, requests):
    """Oracle grid head: one-hot at each object's own cell with the exact residual."""
    position = np.asarray(position, float)
    logits, residuals = torch.full((len(position), GRID * GRID), -20.), torch.zeros(len(position), GRID * GRID, 2)
    for i, (x, y, _) in enumerate(position):
        u = np.array([x, y]) / 4 * GRID
        cell = np.minimum(u.astype(int), GRID - 1)
        logits[i, cell[0] * GRID + cell[1]] = 0.
        residuals[i, cell[0] * GRID + cell[1]] = torch.tensor((u - cell - .5) * 2)
    return spread_grid_xy(logits, residuals, GRID, np.array(size, float), np.array(yaw, float), position, ROOM, requests=requests)


def test_an_undeclared_raised_object_on_a_wall_keeps_its_height():
    # before: the painting went to the floor (z 0) in an empty room and onto the sofa (z .8) above one
    xy, _, z = spread([[.02, 2.25, 1.5]], [[.04, .8, .6]], [0.], [{"id": "painting"}])
    assert z == pytest.approx([1.5]) and xy == pytest.approx(np.array([[.02, 2.25]]))
    _, _, z = spread([[.45, 2., 0.], [.02, 2., 1.4]], [[.9, 2., .8], [.04, .8, .6]], [0., 0.],
                     [{"id": "sofa", "support_parent": "floor"}, {"id": "painting"}])
    assert z == pytest.approx([0., 1.4])


def test_undeclared_raised_objects_away_from_a_wall_or_over_furniture_still_rest_on_it_or_drop():
    _, _, z = spread([[2., 2., 0.], [2., 2., .8]], [[1.2, .6, .75], [.1, .1, .1]], [0., 0.],
                     [{"id": "table", "support_parent": "floor"}, {"id": "cup"}])
    assert z == pytest.approx([0., .75])  # 5 cm over the table top: rests on it
    _, _, z = spread([[2., 2., 1.4]], [[.04, .8, .6]], [0.], [{"id": "painting"}])
    assert z == pytest.approx([0.])  # mid-room, nothing under it: the floor


def test_a_declared_child_inside_its_parent_keeps_its_shelf_height():
    shelf = {"id": "bookcase", "support_parent": "floor"}
    xy, _, z = spread([[1., 3.8, 0.], [1., 3.8, .9], [1.2, 3.8, 1.78]], [[.8, .35, 1.8], [.2, .15, .25], [.1, .1, .2]],
                      [0., 0., 0.], [shelf, {"id": "book", "support_parent": "bookcase"}, {"id": "vase", "support_parent": "bookcase"}])
    assert z == pytest.approx([0., .9, 1.8])  # before: the book on the top (1.8); the vase 2 cm under the top still snaps up
    assert xy[1] == pytest.approx([1., 3.8])  # the bookcase it stands in blocks nothing


def test_a_wall_object_predicted_in_the_swapped_axis_form_keeps_its_thin_side_on_the_wall():
    wall = [{"id": "painting", "support_parent": "wall"}]
    xy, yaw, _ = spread([[.02, 2.25, 1.5]], [[.04, .8, .6]], [0.], wall)
    assert xy == pytest.approx(np.array([[.02, 2.25]])) and yaw == pytest.approx([0.])
    xy, yaw, _ = spread([[.02, 2.25, 1.5]], [[.8, .04, .6]], [math.pi / 2], wall)  # the same box
    assert xy == pytest.approx(np.array([[.02, 2.25]])) and yaw == pytest.approx([math.pi / 2])  # before: x .4, yaw 0


def test_a_turned_wall_projection_measures_overlap_at_its_own_footprint():
    # audit C1: a 2 x .2 m box decoded at 45 degrees, turned flush (yaw 0) onto the x = 0 wall, was compared by its
    # 45-degree bounds (2.42 m^2, not .4): .1 m^2 into a fixed block (25%) read as 4% and beat a free cell
    block = {"id": "block", "bottom_center_m": [1., 2.55, 0.], "size_local_m": [2., 2., 2.7], "yaw_rad": 0.}
    logits = torch.full((1, 16), -100.)
    logits[0, 1], logits[0, 0] = 10., 9.  # cell (0, 1) at (.5, 1.5), then (0, 0)
    xy, yaw, _ = spread_grid_xy(logits, torch.zeros(1, 16, 2), 4, np.array([[2., .2, .2]]), np.array([math.pi / 4]),
                                np.array([[.5, 1.5, .6]]), {**ROOM, "fixed_objects": [block]}, top_k=2,
                                requests=[{"id": "a", "support_parent": "wall"}])
    assert xy[0] == pytest.approx([1., .5]) and yaw == pytest.approx([0.])  # before: [1, 1.5], in the block
    # placed, it keeps that footprint as an obstacle: a 1 m box .1 m^2 into it (25% of .4) moves on (before: stayed)
    logits = torch.full((2, 16), -100.)
    logits[0, 1], logits[1, 2], logits[1, 3] = 10., 10., 9.
    residuals = torch.zeros(2, 16, 2)
    residuals[1, 2, 1] = -1.  # cell (0, 2) shifted to (.5, 2.)
    xy, _, _ = spread_grid_xy(logits, residuals, 4, np.array([[2., .2, .2], [1., 1., .2]]), np.array([math.pi / 4, 0.]),
                              np.array([[.5, 1.5, .6], [.5, 2., .6]]), ROOM, top_k=2,
                              requests=[{"id": "a", "support_parent": "wall"}, {"id": "b", "support_parent": "wall"}])
    assert xy == pytest.approx(np.array([[1., 1.5], [.5, 3.5]]))


def box(ident, position, size, yaw=0.):
    return {"id": ident, "bottom_center_m": position, "target_size_local_m": size, "yaw_rad": yaw}


def condition(*requests):
    return {"schema_version": "fastfill.v2", "constraints": [], "objects": [
        {"id": r[0], "category": r[1], "description": r[1], **({"support_parent": r[2]} if len(r) > 2 else {})} for r in requests],
            "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": POLYGON, "floor_z_m": 0., "floor_known": True,
                     "boundary_known": True, "height_m": 2.7, "room_type": "study"}}


def test_the_handoff_infers_a_wall_for_an_undeclared_hung_box_with_nothing_under_it():
    painting, bookcase = box("p", [.02, 2., 1.4], [.04, .8, .6]), box("bc", [1., 3.8, 0.], [.8, .35, 1.8])
    book = box("b", [1., 3.92, .9], [.2, .15, .25])  # 1 cm off the y = 4 wall, inside the bookcase
    sofa = box("s", [.45, 2., 0.], [.9, 2., .8])
    room = condition()["room"]
    assert infer_support(painting, None, [painting, bookcase, book], 0., room) == ("wall", "inferred")  # before: unknown
    assert infer_support(book, None, [painting, bookcase, book], 0., room) == (None, "unknown")
    assert infer_support(painting, None, [painting, sofa], 0., room) == (None, "unknown")  # something under it
    assert infer_support(painting, None, [painting], 0.) == (None, "unknown")  # no known boundary
    scene = layout_to_roomgenbench(condition(("p", "painting")), {"schema_version": "fastfill.v2", "objects": [painting]})
    assert [(o["place"], o["support_status"]) for o in scene["objects"]] == [("wall", "inferred")]


@pytest.mark.parametrize("gap,status", [(0., "pass"), (.08, "pass"), (.3, "violation")])
def test_validate_scene_accepts_the_source_wall_gap(gap, status):
    result = validate_scene(condition(("p", "painting", "wall")), [box("p", [gap + .02, 2., 1.4], [.04, .8, .6])])
    assert [c["status"] for c in result["checks"] if c["code"] == "wall_support"] == [status]  # before: .08 a violation


def answer(x, z=1.4):
    return json.dumps({"objects": [{"id": "p", "width_m": .8, "depth_m": .04, "height_m": .6, "x": x, "y": 2., "z": z,
                                    "facing_deg": 0}]})


def test_the_structured_harness_lets_wall_objects_hang():
    for request in (("p", "painting"), ("p", "painting", "wall")):
        assert check_answer(answer(.02), condition(request))[1] == []  # before: "floats ... put it on ... the floor"
    assert any(" floats at " in p for p in check_answer(answer(2.), condition(("p", "painting")))[1])
    assert check_answer(answer(2.), condition(("p", "painting", "wall")))[1] == [
        "p (painting) must hang on a wall: one side of its footprint against a wall"]
    assert "hangs on a wall when it says wall" in STRUCTURED


def test_rotate90_augmentation_leaves_rooms_with_openings_unturned():
    room = {**condition()["room"], "openings": [{"type": "door", "wall": 0}]}
    sample = {"schema_version": "fastfill.v2", "condition": {**condition(("p", "painting", "wall")), "room": room},
              "target": {"schema_version": "fastfill.v2", "objects": [box("p", [.02, 2., 1.4], [.04, .8, .6])]},
              "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [True], "yaw_symmetry_order": [2],
                           "size_axis_swap_allowed": [False], "exchangeable_group": [None]}}
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):  # before: 14 of 20 raised "rotate90/mirror cannot transform free-form room openings"
        out = augment_sample(sample, {**AUGMENT_DEFAULTS, "minimal_form_p": 0.}, generator)
        assert out["target"]["objects"][0]["bottom_center_m"] == [.02, 2., 1.4]
