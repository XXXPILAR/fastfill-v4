"""Round 3 (eval-handoff): exchange-invariant collapse, out-of-room fraction, per-target swap cost, unique door nodes."""
from copy import deepcopy

import pytest
import torch
import trimesh

from fastfill.v2 import evaluate
from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.direct_layout import export_handoff
from fastfill.v2.matching import match_batch
from fastfill.v2.roomgenbench import assemble_handoff
from fastfill.v2.tests.test_evaluate_fix20261006 import row
from fastfill.v2.tests.test_roomgenbench import REFERENCE, points_for_node


def _chairs():
    """Two exchangeable chairs with complete positions; only the first has a size label."""
    sample = row([("chair", (.5, .5, 0.), (.5, .5, 1.), 0.), ("chair", (2., 2., 0.), (.5, .5, 1.), 0.)])
    sample["validity"]["exchangeable_group"] = ["chairs", "chairs"]
    sample["validity"]["size"][1] = [False] * 3
    plain = deepcopy(sample["target"])
    swapped = deepcopy(plain)
    for i in range(2):
        swapped["objects"][i] = {**plain["objects"][1 - i], "id": plain["objects"][i]["id"]}
    return sample, plain, swapped


def _model_outputs(layout, batch, bins=8):
    """Model-shaped outputs (yaw logits, not radians) that decode to the layout; every yaw here is 0."""
    tensors = evaluate._layout_tensors(layout, batch)
    n = tensors["size"].shape[1]
    logits = torch.zeros(1, n, bins)
    logits[..., 0] = 1.
    return {"position_normalized": tensors["position_normalized"], "size": tensors["size"],
            "yaw_logits": logits, "yaw_residuals": torch.zeros(1, n, bins)}


def test_a_legal_id_exchange_changes_neither_reference_error_nor_collapse():
    sample, plain, swapped = _chairs()
    for layout in (plain, swapped):
        assert evaluate.reference_metrics(layout, sample)["bottom_center_error_m"] == {"mean": 0., "valid_objects": 2}
    before, after = evaluate.collapse_metrics(plain, sample), evaluate.collapse_metrics(swapped, sample)
    assert before == after
    assert before["predicted"]["objects"] == 2 and before["predicted"]["central_quarter_objects"] == 1
    assert evaluate.collapse_score(before["predicted"]) == evaluate.collapse_score(after["predicted"]) == .5
    # The ground truth keeps its label-complete objects only (the unsized chair has no labelled box).
    assert before["ground_truth"]["objects"] == 1 and before["ground_truth"]["central_quarter_objects"] == 0
    # The collated batch counts agree: model outputs score every slot, labels (yaw in radians) the label-complete ones.
    batch = collate_samples([sample], TinyTokenizer(), max_length=1 << 15)
    for layout in (plain, swapped):
        observed = evaluate.batch_collapse_counts(_model_outputs(layout, batch), batch)
        for key in evaluate.COLLAPSE_KEYS:
            assert observed[key] == pytest.approx(before["predicted"][key], abs=1e-6)
    labels = {"position_normalized": batch["targets"]["position_normalized"], "size": batch["targets"]["size"],
              "yaw": batch["targets"]["yaw"]}
    anchor = evaluate.batch_collapse_counts(labels, batch)
    for key in evaluate.COLLAPSE_KEYS:
        assert anchor[key] == pytest.approx(before["ground_truth"][key], abs=1e-6)


def test_the_matched_column_holds_the_ground_truth_objects_under_a_legal_exchange():
    sample, plain, swapped = _chairs()
    truth = evaluate.collapse_metrics(plain, sample)["ground_truth"]
    batch = collate_samples([sample], TinyTokenizer(), max_length=1 << 15)
    for layout in (plain, swapped):
        result = evaluate.collapse_metrics(layout, sample)
        # The swapped layout's sized chair box sits under the other ID; the matching follows it there.
        assert result["predicted_matched"] == truth and result["predicted"]["objects"] == 2
        assignment = match_batch(evaluate._layout_tensors(layout, batch), batch)
        observed = evaluate.batch_collapse_counts(_model_outputs(layout, batch), batch, assignment)
        for key in evaluate.COLLAPSE_KEYS:
            assert observed[key] == pytest.approx(truth[key], abs=1e-6)


def test_the_labels_score_no_distribution_penalty_beside_a_size_masked_source(tmp_path):
    from fastfill.v2.autorun import score
    from fastfill.v2.tests.test_evaluate_fix20261006 import evaluate as run
    masked = row([("sofa", (2., 2., 0.), (2., 1., .8), 0.), ("lamp", (.3, .3, 0.), (.3, .3, 1.5), 0.)], source="masked", scene="m")
    masked["validity"]["size"] = [[False] * 3] * 2  # as MansionWorld: positions only
    rows = [masked] + [row([("bed", (2., 3.5, 0.), (2., 1. * k, .5), 0.), ("desk", (.5, 1., 0.), (1., .5 * k, .75), 0.)],
                           source="complete", scene=f"c{k}") for k in (1, 2)]  # two sizes: a nonzero leave-one-out size baseline
    report, _ = run(tmp_path, rows, [r["target"] for r in rows])
    collapse = report["collapse"]
    assert (collapse["predicted"]["objects"], collapse["predicted_matched"]["objects"], collapse["ground_truth"]["objects"]) == (6, 4, 4)
    # Every-object column vs the label-complete ground truth: the central sofa alone makes the labels look "collapsed".
    assert collapse["predicted"]["central_quarter_fraction"] == 1 / 6 and collapse["ground_truth"]["central_quarter_fraction"] == 0.
    assert collapse["predicted_matched"] == collapse["ground_truth"]
    assert score(report) == 0.  # zero errors and no distribution penalty
    assert report["by_source"]["masked"]["collapse"]["predicted_matched"]["objects"] == 0


def test_out_of_room_fraction_flags_objects_beyond_the_floor_polygon_and_leaves_the_score_alone():
    sample = row([("chair", (1., 1., 0.), (.5, .5, 1.), 0.), ("lamp", (3., 3., 0.), (.3, .3, 1.5), 0.),
                  ("sofa", (3.96, 1., 0.), (.5, .5, 1.), 0.)])
    sample["target"]["objects"][2]["bottom_center_m"] = [4.04, 1., 0.]  # 4 cm outside: within the 5 cm tolerance
    pushed = deepcopy(sample["target"])
    pushed["objects"][0]["bottom_center_m"] = [-1., 1., 0.]
    pushed["objects"][1]["bottom_center_m"] = [3., 4.3, 0.]
    result = evaluate.collapse_metrics(pushed, sample)
    assert result["predicted"]["out_of_room_objects"] == 2 and result["ground_truth"]["out_of_room_objects"] == 0
    summary = evaluate._collapse_summary([{"collapse": result}])
    assert summary["predicted"]["out_of_room_fraction"] == pytest.approx(2 / 3)
    assert summary["ground_truth"]["out_of_room_fraction"] == 0.
    # The score's definition is unchanged: the pushed-out layout scores 0, hence out_of_room_fraction beside it.
    assert summary["predicted"]["collapse_score"] == 0.
    assert "out_of_room_objects" in evaluate.COLLAPSE_KEYS


@pytest.mark.parametrize("flags,expected", [((True, False), [0, 1]), ((False, True), [1, 0]), ((True, True), [1, 0])])
def test_swap_cost_follows_each_target_objects_own_flag(flags, expected):
    # Same position; A (1, 1, 1) and B (2, 1, 1). The (1, 2, 1) prediction equals B only with B's axes swapped.
    sample = row([("chair", (2., 2., 0.), (1., 1., 1.), 0.), ("chair", (2., 2., 0.), (2., 1., 1.), 0.)])
    sample["validity"]["exchangeable_group"] = ["chairs", "chairs"]
    sample["validity"]["size_axis_swap_allowed"] = list(flags)
    batch = collate_samples([sample], TinyTokenizer(), max_length=1 << 15)
    predictions = {"position_normalized": batch["targets"]["position_normalized"].clone(),
                   "size": torch.tensor([[[1., 2., 1.], [1.5, 1., 1.]]]), "slot_mask": batch["slot_mask"]}
    # B forbids the swap in the first case, so the square A's permission must not open it for B.
    assert match_batch(predictions, batch)[0].tolist() == expected


def test_two_doors_on_one_wall_and_a_door_on_another_each_keep_a_panel_node(tmp_path):
    condition = {"schema_version": "fastfill.v2", "room": {"frame": "right_handed_z_up",
        "floor_polygon_xy_m": [[0., 0.], [6., 0.], [6., 4.], [0., 4.]], "boundary_known": True, "floor_known": True,
        "floor_z_m": 0., "height_m": 2.6, "room_type": "bedroom", "fixed_objects": [
            {"id": "door_a", "category": "door", "size_local_m": [.8, .1, 2.], "bottom_center_m": [1.5, 0., 0.], "yaw_rad": 0.},
            {"id": "door_b", "category": "door", "size_local_m": [.8, .1, 2.], "bottom_center_m": [4.5, 0., 0.], "yaw_rad": 0.},
            {"id": "door_c", "category": "door", "size_local_m": [.8, .1, 2.], "bottom_center_m": [6., 2., 0.],
             "yaw_rad": 1.5707963267948966}]},
        "objects": [{"id": "obj_0000", "category": "bed", "description": "bed", "support_parent": "floor"}], "constraints": []}
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": "obj_0000", "target_size_local_m": [2., 1.6, .5], "bottom_center_m": [3., 2., 0.], "yaw_rad": 0.}]}
    receipt = assemble_handoff(export_handoff(tmp_path / "handoff", condition, layout), tmp_path / "assembled",
                               roomgenbench_root=REFERENCE)
    glb = tmp_path / "assembled" / f'{receipt["scene_key"]}.glb'
    doors = sorted(n for n in trimesh.load(glb, force="scene", process=False).graph.nodes_geometry if "_door" in n)
    assert doors == ["shell_wall_0_door", "shell_wall_0_door_1", "shell_wall_1_door"]
    # Same-wall panels sit along x at the two doors (glTF x is SAGE x).
    assert sorted(round(float(points_for_node(glb, n)[:, 0].mean()), 4) for n in doors[:2]) == [1.5, 4.5]
