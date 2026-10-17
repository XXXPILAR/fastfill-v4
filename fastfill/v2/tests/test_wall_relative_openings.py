"""Wall-relative room.openings (RoomGenBench shell form, no world coordinates): they turn with the room under
rotate90/mirror augmentation and become the hand-off's doors/windows."""
import math

import pytest
import torch

from fastfill.v2.batch import AUGMENT_DEFAULTS, augment_sample, wall_relative_opening
from fastfill.v2.direct_layout import layout_to_roomgenbench
from fastfill.v2.schema import validate_condition

POLYGON = [[0., 0.], [5., 0.], [5., 3.], [0., 3.]]
OPENINGS = [{"type": "door", "wall_index": 1, "position_on_wall": .25, "width_m": .9, "height_m": 2.},
            {"type": "window", "wall_index": 2, "position_on_wall": .5, "width_m": 1.2, "height_m": 1.,
             "sill_height_m": .9}]


def sample():
    room = {"frame": "right_handed_z_up", "floor_polygon_xy_m": POLYGON, "floor_z_m": 0., "floor_known": True,
            "boundary_known": True, "height_m": 2.7, "openings": OPENINGS}
    condition = {"schema_version": "fastfill.v2", "room": room, "constraints": [],
                 "objects": [{"id": "a", "category": "sofa", "description": "sofa", "support_parent": "floor"}]}
    target = {"schema_version": "fastfill.v2",
              "objects": [{"id": "a", "target_size_local_m": [.9, 2., .8], "bottom_center_m": [1., 1.5, 0.],
                           "yaw_rad": 0.}]}
    validity = {"position": [[True] * 3], "size": [[True] * 3], "yaw": [True], "yaw_symmetry_order": [1],
                "size_axis_swap_allowed": [False], "exchangeable_group": [None]}
    return {"schema_version": "fastfill.v2", "condition": validate_condition(condition), "target": target,
            "validity": validity}


def opening_point(room, o):
    p = room["floor_polygon_xy_m"]
    a, b = p[o["wall_index"]], p[(o["wall_index"] + 1) % len(p)]
    t = o["position_on_wall"]
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def test_predicate_accepts_only_the_wall_relative_form():
    assert all(wall_relative_opening(o) for o in OPENINGS)
    for bad in ({"type": "door", "wall": 0}, {"kind": "door", "position": [2, 0, 0]},
                {**OPENINGS[0], "position_on_wall": 1.5}, {**OPENINGS[0], "wall_index": True},
                {**OPENINGS[0], "width_m": 0}, {**OPENINGS[0], "type": "arch"}):
        assert not wall_relative_opening(bad)


def test_rotation_turns_the_room_and_keeps_each_opening_beside_the_same_object():
    s = sample()
    sofa_to_door = math.dist(opening_point(s["condition"]["room"], OPENINGS[0]),
                             s["target"]["objects"][0]["bottom_center_m"][:2])
    generator, turned = torch.Generator().manual_seed(0), 0
    for _ in range(20):
        out = augment_sample(s, {**AUGMENT_DEFAULTS, "mirror": True, "minimal_form_p": 0.,
                                 "drop_constraints_p": 0., "drop_support_p": 0., "category_only_description_p": 0.},
                             generator)
        room = out["condition"]["room"]
        assert room["openings"] == OPENINGS
        turned += room["floor_polygon_xy_m"] != POLYGON
        sofa = out["target"]["objects"][0]["bottom_center_m"]
        assert math.dist(opening_point(room, OPENINGS[0]), sofa[:2]) == pytest.approx(sofa_to_door)
    assert turned >= 10


def test_handoff_shell_reads_wall_relative_openings():
    s = sample()
    room = layout_to_roomgenbench(s["condition"], s["target"])["room"]
    door, = room["doors"]
    window, = room["windows"]
    assert (door["wall_id"], door["position_on_wall"], door["width"], door["height"]) == ("wall_01", .25, .9, 2.)
    assert (window["wall_id"], window["width"], window["sill_height"]) == ("wall_02", 1.2, .9)
