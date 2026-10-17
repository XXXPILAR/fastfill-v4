"""Round 14 math audit: box-preserving wall facing, rotated supports, paired baselines, validator collisions and
clean rooms, the square-grid check and offline re-decoding of saved head outputs."""
import json
import math

import numpy as np
import pytest
import torch

from fastfill.v2 import evaluate
from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.evaluate import run_evaluation, serialize_predictions, spread_grid_xy
from fastfill.v2.model import ModelConfig, build_model
from fastfill.v2.tests.test_evaluate_fix20261006 import evaluate as evaluate_supplied, real_rows, row
from fastfill.v2.tests.test_spread_decode import ROOM, _logits
from fastfill.v2.tests.test_round7_reference import BATHROOM
from fastfill.v2.validation import validate_scene

WALL_CELL = 1  # cell (0, 1) of the 4 x 4 grid: centre (.5, 1.5), within .3 m of the x = 0 wall for the box below


def _wall_yaw(yaw, *, bins=12):
    """Spread yaw of a .8 x .5 m box on WALL_CELL decoded at ``yaw``; its yaw head also gives bin 0 (facing away) mass."""
    yaw_logits = np.full((1, bins), -3.)
    yaw_logits[0, 0], yaw_logits[0, round(yaw / (2 * math.pi / bins)) % bins] = 1., 2.
    xy, out, _ = spread_grid_xy(_logits(WALL_CELL)[None], torch.zeros(1, 16, 2), 4, np.array([[.8, .5, .8]]), np.array([yaw]),
                                np.zeros((1, 3)), ROOM, yaw_logits=yaw_logits, yaw_residuals=np.zeros((1, bins)))
    assert xy[0] == pytest.approx([.5, 1.5])  # it stays on its cell
    return float(out[0])


def _same_box(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2) < 1e-9


def test_the_wall_rule_never_turns_a_box_by_a_quarter():
    # before: a box decoded side-on to the wall (pi/2) took bin 0 (facing away), a 90-degree turn of its box
    assert _wall_yaw(math.pi / 2) == math.pi / 2 and _wall_yaw(-math.pi / 2) == -math.pi / 2
    # facing the wall within 45 degrees: the half turn (before: snapped to bin 0, not the same box)
    assert _wall_yaw(math.pi - .5) == pytest.approx(-.5, abs=1e-12)
    assert _wall_yaw(math.pi) == pytest.approx(0., abs=1e-12)
    assert _wall_yaw(.3) == .3  # already facing away within 45 degrees: kept (before: snapped to bin 0)


def test_the_wall_rule_keeps_every_box_modulo_pi():
    rng = np.random.default_rng(0)
    border = [c for c in range(64) if c // 8 in (0, 7) or c % 8 in (0, 7)]  # 8 x 8 cells of .75 m against the walls
    n = 24
    cells = rng.choice(border, n, replace=False)
    yaw = rng.uniform(-math.pi, math.pi, n)
    logits = torch.from_numpy(rng.normal(size=(n, 64)))
    logits[np.arange(n), cells] = 10.
    size = rng.uniform(.2, .4, (n, 3))
    room = {**ROOM, "bounds": (np.zeros(2), np.full(2, 6.)), "scale": [6., 6., 3.]}
    xy, out, _ = spread_grid_xy(logits, torch.zeros(n, 64, 2), 8, size, yaw, np.zeros((n, 3)), room,
                                yaw_logits=rng.normal(size=(n, 12)), yaw_residuals=np.zeros((n, 12)))
    assert np.allclose(xy, (np.stack((cells // 8, cells % 8), -1) + .5) * .75)  # every box on its own cell
    assert all(_same_box(a, b) for a, b in zip(out, yaw))  # before: all 24 snapped to a bin off their box
    assert 0 < sum(not np.isclose(a, b) for a, b in zip(out, yaw)) < n  # some flipped, some kept


TABLE = {"id": "table", "category": "table", "size_local_m": [2., .4, .75], "bottom_center_m": [2., 2., 0.],
         "yaw_rad": math.pi / 4, "support_surfaces": [{"surface_id": "top", "local_z_m": .75,
                                                       "local_polygon_xy_m": [[-1., -.2], [1., -.2], [1., .2], [-1., .2]]}]}
OFF_TOP = (2.6, 1.4)  # inside the rotated table's axis-aligned bounds (half extents .85), .85 m off its local long axis


def test_a_declared_child_of_a_rotated_table_stands_on_its_rotated_top():
    condition = {"schema_version": "fastfill.v2", "constraints": [],
                 "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
                          "floor_z_m": 0., "floor_known": True, "height_m": 3., "boundary_known": True, "fixed_objects": [TABLE]},
                 "objects": [{"id": "cup", "category": "cup", "description": "a cup", "support_parent": "table"}]}
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=1 << 15)
    grid = 20  # cell (13, 7) of a 20 x 20 grid over the 4 m room is centred on OFF_TOP
    logits = torch.full((1, 1, grid * grid), -5.)
    logits[0, 0, 13 * grid + 7] = 5.
    predictions = {"position_normalized": torch.tensor([[[OFF_TOP[0] / 4, OFF_TOP[1] / 4, .75 / 3]]]),
                   "size": torch.full((1, 1, 3), .1), "yaw_logits": torch.tensor([[[3.] + [0.] * 11]]),
                   "yaw_residuals": torch.zeros(1, 1, 12), "position_cell_logits": logits,
                   "position_cell_residuals": torch.zeros(1, 1, grid * grid, 2)}
    objects = serialize_predictions(predictions, batch)[0]["objects"]
    # before: its argmax cell counted as on the table (axis-aligned bounds), the cup stood beside the rotated top
    checks = validate_scene(condition, objects)["checks"]
    assert [c["status"] for c in checks if c["code"] == "support"] == ["pass"]
    local = np.array(objects[0]["bottom_center_m"][:2]) - [2., 2.]
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    assert abs(c * local[0] + s * local[1]) <= .95 + 1e-9 and abs(c * local[1] - s * local[0]) <= .15 + 1e-9  # the lattice is inset


def test_an_undeclared_raised_item_off_a_rotated_table_is_not_held_by_its_bounds():
    room = {**ROOM, "fixed_objects": [TABLE]}
    cell = int(OFF_TOP[0]) * 4 + int(OFF_TOP[1])  # the 4 x 4 cell (2, 1), centred on (2.5, 1.5): off the rotated top
    _, _, z = spread_grid_xy(_logits(cell)[None], torch.zeros(1, 16, 2), 4, np.full((1, 3), .1), np.zeros(1),
                             np.array([[0., 0., .8]]), room)
    assert z[0] == 0.  # before: .75, resting on the table's axis-aligned bounds
    _, _, z = spread_grid_xy(_logits(10)[None], torch.zeros(1, 16, 2), 4, np.full((1, 3), .1), np.zeros(1),
                             np.array([[0., 0., .8]]), room)  # cell (2, 2), centred on (2.5, 2.5): over the top
    assert z[0] == .75


def test_a_non_square_cell_count_is_refused():
    batch = collate_samples([row([("chair", [1, 1, 0], [.5, .5, .5], 0.)])], TinyTokenizer(), max_length=10000)
    predictions = {"position_normalized": torch.full((1, 1, 3), .25), "size": torch.full((1, 1, 3), .5),
                   "yaw_logits": torch.zeros(1, 1, 12), "yaw_residuals": torch.zeros(1, 1, 12),
                   "position_cell_logits": torch.zeros(1, 1, 15), "position_cell_residuals": torch.zeros(1, 1, 15, 2)}
    with pytest.raises(ValueError, match="not a square grid"):  # before: decoded as a 4 x 4 grid
        serialize_predictions(predictions, batch)


def _two_rooms():
    """Room a: two chairs whose footprints overlap by 1/8 (IoU < 0.3) and a third clear of them; room b: two stacked
    chairs (IoU 1). Labels: the same chairs apart."""
    a = [("chair", [1, 1, 0], [1, 1, 1], 0.), ("chair", [1.75, 1, 0], [1, 1, 1], 0.), ("chair", [3, 3, 0], [1, 1, 1], 0.)]
    b = [("table", [1, 1, 0], [1, 1, 1], 0.), ("sofa", [3, 3, 0], [1, 1, 1], 0.)]
    rows = [row([(c, [p[0] + 0. * i, p[1], 0], s, y) for i, (c, p, s, y) in enumerate(a)], scene="a"), row(b, scene="b")]
    rows[0]["target"]["objects"][1]["bottom_center_m"] = [2.5, 1., 0.]  # the label pair does not overlap
    layouts = [{"schema_version": "fastfill.v2", "objects": [
        {"id": o["id"], "bottom_center_m": o["bottom_center_m"], "target_size_local_m": o["target_size_local_m"], "yaw_rad": 0.}
        for o in rows[0]["target"]["objects"]]}, '{"bad": true}']
    layouts[0]["objects"][1]["bottom_center_m"] = [1.75, 1., 0.]
    return rows, layouts


def test_baselines_are_also_scored_on_the_requests_with_a_layout(tmp_path):
    rows, layouts = _two_rooms()
    report, outcomes = evaluate_supplied(tmp_path, rows, [layouts[0], json.loads(layouts[1])])
    assert report["baselines"]["requests"] == 2 and report["baselines_paired"]["requests"] == 1
    own = outcomes[0]["baselines"]["room_center_position"]["bottom_center_error_m"]
    assert report["baselines_paired"]["room_center_position"]["bottom_center_error_m"] == pytest.approx(own)
    assert report["baselines_paired"]["room_center_position"]["bottom_center_error_m"]["valid_objects"] == \
        report["model"]["reference"]["bottom_center_error_m"]["valid_objects"] == 3  # the same objects
    assert report["by_source"]["fixture"]["baselines_paired"]["requests"] == 1


def test_the_report_counts_validator_collisions_and_clean_rooms_beside_the_labels(tmp_path):
    rows, layouts = _two_rooms()
    stacked = {"schema_version": "fastfill.v2", "objects": [{**o, "bottom_center_m": [1., 1., 0.]} for o in rows[1]["target"]["objects"]]}
    report, outcomes = evaluate_supplied(tmp_path, rows, [layouts[0], stacked])
    model, truth = report["validation"]["model"], report["validation"]["ground_truth"]
    # before: only the IoU > 0.3 overlap rate, which misses room a's collision
    assert (model["rooms"], model["rooms_with_collision"], model["collision_pairs"], model["fixed_collision_pairs"]) == (2, 2, 2, 0)
    assert (truth["rooms"], truth["rooms_with_collision"], truth["collision_pairs"]) == (2, 0, 0)
    assert model["clean_room_rate"] == 0. and truth["clean_room_rate"] == 1. and truth["clean_rooms_with_hard_unknown"] == 2
    # undeclared supports stay unknown: every room fails the strict count, only collisions are hard violations
    assert report["failed_requests"] == report["failed_requests_strict"] == 2 and report["hard_violation_requests"] == 2
    assert outcomes[0]["ground_truth_validation"] == {"collision_pairs": 0, "fixed_collision_pairs": 0,
                                                      "hard_violation": False, "hard_unknown": True}
    pred = report["collapse"]["predicted_matched"]
    assert pred["bev_overlap_rate_iou_gt_0.3"] == pytest.approx(1 / 4)  # pooled: 1 of 3 + 1 pairs
    assert pred["bev_overlap_rate_iou_gt_0.3_room_mean"] == pytest.approx(1 / 2)  # rooms: 0 and 1


def test_hard_violation_requests_leave_out_rooms_failing_only_on_unknown(tmp_path):
    rows, layouts = _two_rooms()
    for r in rows:
        for o in r["condition"]["objects"]:
            o["support_parent"] = "floor"
    report, _ = evaluate_supplied(tmp_path, rows, [rows[0]["target"], rows[1]["target"]])
    assert report["validation"]["model"]["clean_rooms_with_hard_unknown"] == 0 and report["hard_violation_requests"] == 0
    assert report["failed_requests_strict"] == 0 and report["validation"]["model"]["clean_room_rate"] == 1.


def test_the_known_only_clean_room_rate_leaves_out_rooms_with_a_hard_unknown(tmp_path):
    rows, layouts = _two_rooms()
    for o in rows[0]["condition"]["objects"]:
        o["support_parent"] = "floor"  # room a: every hard check decided, and its layout collides
    report, _ = evaluate_supplied(tmp_path, rows, [layouts[0], rows[1]["target"]])  # room b: its labels, supports unknown
    model = report["validation"]["model"]
    assert (model["clean_room_rate"], model["clean_rooms_with_hard_unknown"]) == (.5, 1)  # room b counts as clean
    # new: the rate over the rooms whose hard checks were all decided (room a alone)
    assert (model["rooms_without_hard_unknown"], model["clean_room_rate_known_only"]) == (1, 0.)
    assert report["validation"]["ground_truth"]["clean_room_rate_known_only"] == 1.


def _tiny_grid_checkpoint(directory):
    torch.manual_seed(0)
    build_model(ModelConfig(backbone="tiny", decoder_dim=16, decoder_heads=2, decoder_layers=1, tiny_hidden_size=16,
                            max_objects=8, lora_rank=0, position_head="grid_residual", position_grid=4)).save_pretrained(directory)
    return directory


def _outcomes(directory, projection="full"):
    name = "outcomes.jsonl" if projection == "full" else f"outcomes-{projection}.jsonl"
    volatile = ("evaluation_wall_time_ms", "fastfill_latency_ms", "fastfill_latency_status")
    return [{k: v for k, v in json.loads(line).items() if k not in volatile}
            for line in (directory / name).read_text().splitlines()]


def test_saved_head_outputs_re_decode_offline_to_the_online_result(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in real_rows()))  # 4, 3 and 11 objects: the last is over capacity
    checkpoint, heads = _tiny_grid_checkpoint(tmp_path / "model"), tmp_path / "heads"
    projections = ("full", "minimal")
    online = {decode: run_evaluation(data, tmp_path / f"online-{decode}", checkpoint=checkpoint, projections=projections,
                                     grid_decode=decode, save_head_outputs=heads if decode == "spread" else None)
              for decode in ("spread", "argmax")}
    assert online["spread"]["over_capacity_requests"] == 1 and online["spread"]["saved_head_outputs"] == str(heads.resolve())
    assert sorted(p.name for p in (heads / "full").iterdir()) == ["row-000000.npz", "row-000001.npz", "row-000002.npz"]
    with np.load(heads / "full" / "row-000000.npz") as stored:
        assert {"position_cell_logits", "position_cell_residuals", "position_normalized", "size", "yaw_logits",
                "yaw_residuals", "slot_mask", "ids", "condition"} <= set(stored) and stored["position_cell_logits"].shape == (4, 16)
    for decode in ("spread", "argmax"):
        assert evaluate.main(["--from-head-outputs", str(heads), "--data", str(data), "--grid-decode", decode,
                              "--projection", *projections, "--output", str(tmp_path / f"offline-{decode}")]) is None
        for projection in projections:
            assert _outcomes(tmp_path / f"offline-{decode}", projection) == _outcomes(tmp_path / f"online-{decode}", projection)
        offline = json.loads((tmp_path / f"offline-{decode}" / "report.json").read_text())
        assert offline["baseline"] == "supplied_head_outputs" and offline["grid_decode"] == decode
        for key in ("model", "baselines_paired", "collapse", "validation", "failed_requests", "over_capacity_requests"):
            assert offline[key] == online[decode][key], key
        # before: no manifest, so an offline report named no checkpoint, binding, max_length or code
        saved = json.loads((tmp_path / "online-spread" / "report.json").read_text())
        assert offline["head_outputs_manifest"] == {key: saved[key] for key in evaluate.HEAD_MANIFEST_KEYS}
        assert offline["head_outputs_manifest"]["checkpoint"] == str(checkpoint.resolve()) and offline["checkpoint"] is None


def test_head_outputs_inside_the_output_are_refused_before_evaluating(tmp_path):
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(real_rows()[0]) + "\n")
    checkpoint = _tiny_grid_checkpoint(tmp_path / "model")
    for output, heads in ((tmp_path / "out", tmp_path / "out" / "heads"), (tmp_path / "same", tmp_path / "same"),
                          (tmp_path / "heads" / "out", tmp_path / "heads")):
        # before: the head directory created the output, whose own mkdir failed after the whole evaluation (or, the
        # other way round, the report landed among the head outputs)
        with pytest.raises(ValueError, match="separate directories"):
            run_evaluation(data, output, checkpoint=checkpoint, save_head_outputs=heads)
        assert not output.exists() and not heads.exists()


def test_head_outputs_of_other_rows_abort_the_offline_run(tmp_path):
    rows = real_rows()[:2]
    data, other = tmp_path / "rows.jsonl", tmp_path / "other.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    other.write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))
    run_evaluation(data, tmp_path / "online", checkpoint=_tiny_grid_checkpoint(tmp_path / "model"), save_head_outputs=tmp_path / "heads")
    with pytest.raises(evaluate.HeadOutputMismatch):
        run_evaluation(other, tmp_path / "offline", head_outputs=tmp_path / "heads")
    with pytest.raises(ValueError, match="no head outputs of projection"):
        run_evaluation(data, tmp_path / "offline-minimal", head_outputs=tmp_path / "heads", projections=("minimal",))


@pytest.mark.skipif(not BATHROOM.is_file(), reason="RoomGenBench scenes are not checked out")
def test_reference_check_matches_identical_requests_and_scores_the_box_first(tmp_path):
    from fastfill.v2.direct_layout import export_handoff, request_to_condition
    from fastfill.v2.geometry import wrap_yaw
    from fastfill.v2.roomgenbench import benchmark_request, ground_truth_layout, reference_check
    scene = json.loads(BATHROOM.read_text())
    condition = request_to_condition(benchmark_request(scene))
    layout = ground_truth_layout(condition, scene)
    objects = layout["objects"]
    a, b = objects[10], objects[11]  # two identical shampoo bottles (same category and description), 35 cm apart
    for key in ("bottom_center_m", "target_size_local_m", "yaw_rad"):
        a[key], b[key] = b[key], a[key]
    toilet = objects[0]  # the same box written a quarter turned: sx / sy swapped
    toilet["target_size_local_m"] = [toilet["target_size_local_m"][1], toilet["target_size_local_m"][0], toilet["target_size_local_m"][2]]
    toilet["yaw_rad"] = wrap_yaw(toilet["yaw_rad"] + math.pi / 2)
    export_handoff(tmp_path / "handoff", condition, layout)
    report = reference_check(tmp_path / "handoff", BATHROOM)
    rows = {r["id"]: r for r in report["per_object"]}
    assert report["schema_version"].endswith(".v2")
    # before: the swapped bottles scored their 35 cm distance by id
    assert (rows["obj_0010"]["predicted_id"], rows["obj_0011"]["predicted_id"]) == ("obj_0011", "obj_0010")
    assert rows["obj_0010"]["position_error_m"] == 0 and rows["obj_0010"]["position_error_m_by_id"] > .3
    # before: the quarter-turned toilet's primary yaw error was pi / 2 (sizes and yaw minimised separately)
    toilet = rows["obj_0000"]
    assert toilet["log_size_error"] == toilet["box_equivalent_log_size_error"] == 0
    assert toilet["yaw_error_rad"] == pytest.approx(0., abs=1e-12) == toilet["box_equivalent_yaw_error_rad"]
    assert toilet["log_size_error_marginal_min"] == 0 and toilet["yaw_error_rad_marginal_min"] == pytest.approx(math.pi / 2)
    assert report["errors"]["all"]["position_error_m"] == 0 and report["errors"]["all"]["yaw_error_rad"] == pytest.approx(0., abs=1e-12)
    assert report["errors"]["all"]["position_error_m_by_id"] > 0
    # before: pi / 4 (the marginal expectation) beside the box-equivalent yaw error; now the box-equivalent expectation
    # of a uniform yaw on each truth box, checked against a 0.5-degree quadrature of the metric itself
    from fastfill.v2.evaluate import _box_equivalent_errors
    yaws = (np.arange(720) + .5) * math.pi / 360
    expected = [np.mean([_box_equivalent_errors({**t, "yaw_rad": t["yaw_rad"] + y}, t, True, True)[1] for y in yaws])
                for t in ground_truth_layout(condition, scene)["objects"]]
    assert [r["uniform_yaw_baseline_error_rad"] for r in report["per_object"]] == pytest.approx(expected, abs=5e-3)
    assert report["uniform_yaw_baseline_error_rad"] == pytest.approx(np.mean(expected), abs=1e-3)
    assert report["uniform_yaw_baseline_error_rad"] < .53 and report["uniform_yaw_baseline_error_rad_marginal_min"] == math.pi / 4


@pytest.mark.skipif(not BATHROOM.is_file(), reason="RoomGenBench scenes are not checked out")
def test_reference_check_matches_only_within_evaluates_exchangeable_groups(tmp_path):
    from fastfill.v2.direct_layout import export_handoff, request_to_condition
    from fastfill.v2.roomgenbench import benchmark_request, ground_truth_layout, reference_check
    restaurant = BATHROOM.parent / "restaurant.json"
    scene = json.loads(restaurant.read_text())
    condition = request_to_condition(benchmark_request(scene))
    layout = ground_truth_layout(condition, scene)
    a, b = layout["objects"][64], layout["objects"][65]
    requests = condition["objects"][64:66]  # identical vases, each alone on its own table
    assert [r["id"] for r in requests] == ["obj_0064", "obj_0065"] and requests[0]["description"] == requests[1]["description"]
    assert [r["support_parent"] for r in requests] == ["obj_0001", "obj_0002"]
    for key in ("bottom_center_m", "target_size_local_m", "yaw_rad"):
        a[key], b[key] = b[key], a[key]
    export_handoff(tmp_path / "handoff", condition, layout)
    rows = {r["id"]: r for r in reference_check(tmp_path / "handoff", restaurant)["per_object"]}
    # before: grouped by category and description alone, each vase was matched to the other table's truth vase
    assert (rows["obj_0064"]["predicted_id"], rows["obj_0065"]["predicted_id"]) == ("obj_0064", "obj_0065")
    assert rows["obj_0064"]["position_error_m"] == rows["obj_0064"]["position_error_m_by_id"] > 1
