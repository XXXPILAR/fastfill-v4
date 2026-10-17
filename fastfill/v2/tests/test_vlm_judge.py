"""vlm_judge projection math (plan and z-buffer agree with validation.footprint and local +X front) and anchors."""
import math

import numpy as np
from PIL import Image
import pytest

from fastfill.v2.direct_layout import layout_to_roomgenbench, request_to_condition
from fastfill.v2.ops import vlm_judge as vj
from fastfill.v2.validation import _geometry, footprint, validate_scene


def room(n=1):
    condition = request_to_condition({"room_type": "living room", "room_size_m": [4., 3., 2.6],
                                      "furniture_list": [{"category": "sofa", "count": n}]})
    return {"condition": condition}


def box(yaw, ident="obj_0000", pos=(2.2, 1.3, 0.), size=(1.2, .6, .8)):
    return {"id": ident, "target_size_local_m": list(size), "bottom_center_m": list(pos), "yaw_rad": yaw}


def scene(*objects):  # (row, items) for boxes given as (size, bottom centre, yaw), request order
    row = room(len(objects))
    layout = {"schema_version": "fastfill.v2", "objects": [box(t, f"obj_{i:04d}", p, s) for i, (s, p, t) in enumerate(objects)]}
    return row, vj.items(row, layout)[0]


def test_box_lands_where_validation_footprint_says(tmp_path):
    pytest.importorskip("matplotlib")  # in no fastfill/v2 requirements file
    row, obj = room(), box(.7)
    expected = np.array(footprint(_geometry(obj, "target")).exterior.coords[:4])
    assert np.allclose(vj.corners_xy(obj["target_size_local_m"], obj["bottom_center_m"], obj["yaw_rad"]), expected)
    its, _ = vj.items(row, {"schema_version": "fastfill.v2", "objects": [obj]})
    vj.render_panel(tmp_path / "p.png", row, its)
    image = np.asarray(Image.open(tmp_path / "p.png").convert("RGB"))
    assert image.shape == (vj.PANEL_H, vj.PANEL_W, 3)
    x0, x1, y0, y1 = vj.plan_extent(row)
    c, s = math.cos(obj["yaw_rad"]), math.sin(obj["yaw_rad"])

    def pixel(lx, ly):  # local footprint coordinates -> plan pixel colour
        x, y = obj["bottom_center_m"][0] + c * lx - s * ly, obj["bottom_center_m"][1] + s * lx + c * ly
        return image[int((y1 - y) / (y1 - y0) * vj.PLAN_PX), int((x - x0) / (x1 - x0) * vj.PLAN_PX)]
    blue = vj.PLACE_RGB["floor"]
    assert np.abs(pixel(-.3, -.15).astype(int) - blue).max() <= 2  # inside, away from the label and arrow
    assert np.abs(pixel(.85, 0).astype(int) - blue).max() > 60  # 0.25 m in front: empty floor
    for eye, target in vj.cameras(row):  # z-buffer: the top centre shows the top face
        img, _ = vj.rasterize(vj.view_primitives(row, its, eye), eye, target)
        x, y, _ = vj.project([[*obj["bottom_center_m"][:2], .8]], eye, target)
        top = np.round(vj._shade(blue, (0, 0, 1)) * 255)
        assert np.abs(img[int(y[0]), int(x[0])].astype(int) - top).max() <= 1


def test_front_arrow_points_along_local_x():
    for yaw in (0., .7, math.pi / 2, -2.5, math.pi):
        obj = box(yaw)
        tail, head = vj.front_arrow(obj["target_size_local_m"], obj["bottom_center_m"], yaw)
        corners = vj.corners_xy(obj["target_size_local_m"], obj["bottom_center_m"], yaw)
        assert np.allclose(head - tail, .6 * np.array((math.cos(yaw), math.sin(yaw))))
        assert np.allclose(head, corners[1:3].mean(0))  # the front edge (local +X)


def test_anchors_are_deterministic_and_valid():
    row = room(4)
    gt = {"schema_version": "fastfill.v2", "objects": [box(.3 * i, f"obj_{i:04d}", (.8 + .8 * i, 1. + .2 * i, 0.)) for i in range(4)]}
    first, again, other = vj.anchors(row, 7, gt, 0), vj.anchors(row, 7, gt, 0), vj.anchors(row, 7, gt, 1)
    assert first == again and first["random"] != other["random"] and set(first) == set(vj.ANCHORS)
    assert first["identical"] == gt
    moved = [a["bottom_center_m"] != b["bottom_center_m"] for a, b in zip(first["shuffled"]["objects"], gt["objects"])]
    assert all(moved)
    assert sorted(o["bottom_center_m"] for o in first["shuffled"]["objects"]) == sorted(o["bottom_center_m"] for o in gt["objects"])
    assert "boundary" not in {c["code"] for c in validate_scene(row["condition"], first["random"]["objects"])["checks"]
                              if c["status"] == "violation"}
    mirrored = first["mirrored"]["objects"][1]
    assert np.isclose(mirrored["bottom_center_m"][0], 4. - gt["objects"][1]["bottom_center_m"][0])
    assert np.isclose(math.cos(mirrored["yaw_rad"]), -math.cos(.3)) and np.isclose(math.sin(mirrored["yaw_rad"]), math.sin(.3))
    for layout in first.values():
        layout_to_roomgenbench(row["condition"], layout)  # every anchor is a valid hand-off layout


def test_corner_views_are_unmirrored_whole_and_without_near_walls():
    # the 4 x 3 x 2.6 m room; V1 looks from beyond (0, 0) towards +x+y, V2 from beyond (4, 3) back
    row, its = scene(((.6, .6, .5), (.6, .5, 0.), 0.),  # beside V1's own corner
                     ((1.2, .6, .8), (3.3, .5, 0.), -math.pi))  # beside (4, 0), its front (local +X) towards -x
    for k, (eye, target) in enumerate(vj.cameras(row)):
        prims = vj.view_primitives(row, its, eye)
        walls = [p for p, _ in prims[1:-10]]
        far = (4., 3.) if k == 0 else (0., 0.)
        assert len(walls) == 2 and all(np.allclose(w[:, 0], far[0]) or np.allclose(w[:, 1], far[1]) for w in walls)
        img, ids = vj.rasterize(prims, eye, target)
        assert (ids == 0).sum() > 1000 and not (ids[[0, -1]] == 0).any() and not (ids[:, [0, -1]] == 0).any()  # whole floor
        first = len(prims) - 10  # by top height: the 0.5 m box's 5 faces, then the 0.8 m box's
        y_near, _ = np.nonzero((ids >= first) & (ids < first + 5))
        _, x_side = np.nonzero(ids >= first + 5)
        front = ids == first + 6
        if k == 0:  # (4, 0) to the right of the line of sight, V1's own corner at the bottom; the front seen, darker
            assert x_side.mean() > vj.VIEW_PX / 2 and y_near.mean() > vj.VIEW_PX / 2 and front.sum() > 100
            dark = np.round(vj._shade(vj.PLACE_RGB["floor"], (-1, 0, 0), .55) * 255)
            assert np.abs(np.median(img[front], 0) - dark).max() <= 1
        else:
            assert x_side.mean() < vj.VIEW_PX / 2 and y_near.mean() < vj.VIEW_PX / 2 and not front.any()


def test_view_labels_and_equal_tops():
    row, its = scene(((1.4, .9, .75), (2., 1.5, 0.), 0.),  # 1 table
                     ((.3, .3, .4), (2., 1.5, 0.), 0.),  # 2 inside the table: never seen
                     ((.25, .2, .05), (2.4, 1.5, .75), 0.),  # 3 on the table
                     ((.6, 1., 2.2), (2.05, 2.7, 0.), -math.pi / 2))  # 4 wardrobe: a depth tolerance lost its top in V1
    for eye, target in vj.cameras(row):
        assert [n for n, _, _ in vj.view(row, its, eye, target)[1]] == [1, 3, 4]
    # two overlapping tops at one height: the later box's top is whole (as if alone), not speckled by the other's
    alone, (_, both) = scene(((.8, .8, .5), (2.2, 1.5, 0.), 0.)), scene(((.8, .8, .5), (1.8, 1.5, 0.), 0.),
                                                                          ((.8, .8, .5), (2.2, 1.5, 0.), 0.))
    for eye, target in vj.cameras(row):
        tops = []
        for its in (alone[1], both):  # the last box's top: the fifth face from the end
            prims = vj.view_primitives(row, its, eye)
            tops.append(vj.rasterize(prims, eye, target)[1] == len(prims) - 5)
        assert tops[0].sum() > 1000 and np.array_equal(*tops)


def test_plan_labels_do_not_overlap_and_stay_in_the_plan():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    row, its = scene(*[((.5, .5, .5), (2., 1.5, 0.), 0.)] * 30 + [((.4, .4, .7), (.3, .3, 0.), 0.)])  # a stack, a corner
    fig = plt.figure(figsize=(vj.PLAN_PX / vj.DPI,) * 2, dpi=vj.DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    vj.draw_plan(ax, row, its)
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts if t.get_text()]  # labels, "1 m" and V1/V2
    assert len(boxes) == len(its) + 3
    plt.close(fig)
    for i, a in enumerate(boxes):
        assert 0 <= a.x0 and a.x1 <= vj.PLAN_PX and 0 <= a.y0 and a.y1 <= vj.PLAN_PX
        assert not any(a.overlaps(b) for b in boxes[i + 1:])
