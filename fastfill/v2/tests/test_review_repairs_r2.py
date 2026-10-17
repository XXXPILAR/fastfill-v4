"""Round-2 review repairs (H2): final-step validation/checkpoint, selection-score anchor, minimal-form eligibility,
rectangle tolerance under rotation, plain-convention yaw and pinned-size yaw of swap objects, unknown-floor
hand-off and resolved backbone paths."""
import json
import math

import pytest
import torch
from torch.nn import functional as F

from fastfill.v2 import evaluate, train
from fastfill.v2.batch import _rigid_xy, augment_sample, is_axis_aligned_rectangle, minimal_form_eligible
from fastfill.v2.direct_layout import layout_to_roomgenbench, request_to_condition
from fastfill.v2.io import backbone_provenance, fingerprint
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data
from fastfill.v2.tests.test_batch_handoff_r2 import ONLY_MINIMAL
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows
from fastfill.v2.tests.test_supervision import batch as pair_batch, prediction as pair_prediction
from fastfill.v2.tests.test_train_eval_r2 import _swap_row, _validation_file
from fastfill.v2.tests.test_train_loop_fix20261006 import _config, _rows


# R2C-1 / R2C-2: the final update is scored and saved; the manifest anchors the score with the labels' own

def test_final_update_is_validated_checkpointed_and_the_score_has_a_ground_truth_anchor(tmp_path):
    validation_rows = real_rows()[:2]
    validation = _validation_file(tmp_path, validation_rows)
    config = {**_config(steps=3, checkpoint_every=2, validate_every=2, max_length=1 << 15), "augmentation": NO_AUGMENTATION}
    logs = train.run_training(config, _data(tmp_path, _rows(2)), tmp_path / "run", validation=validation)
    assert [r["step"] for r in logs if r.get("validation")] == [2, 3]
    assert all((tmp_path / "run" / name).is_dir() for name in ("model-step-3", "state-step-3", "model-step-2"))
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    best = min((r["validation"]["selection_metric"], r["step"]) for r in logs if r.get("validation"))
    assert manifest["selection_metric"]["best"] == {"value": best[0], "step": best[1]}
    expected = dict.fromkeys(evaluate.COLLAPSE_KEYS, 0)
    for sample in map(evaluate.project_minimal, validation_rows):
        for key, value in evaluate.collapse_metrics(sample["target"], sample)["ground_truth"].items():
            expected[key] += value
    anchor = manifest["selection_metric"]["ground_truth"]
    assert anchor["objects"] == expected["objects"] == 7 and anchor["pairs"] == expected["pairs"] > 0
    for key in evaluate.COLLAPSE_KEYS:
        assert anchor[key] == pytest.approx(expected[key], abs=1e-4)
    assert anchor["score"] == pytest.approx(evaluate.collapse_score(expected), abs=1e-6)
    assert json.loads((tmp_path / "run/run_manifest_start.json").read_text())["selection_metric"]["ground_truth"] == anchor
    # A disabled interval stays disabled at the final update.
    config = {**_config(steps=3, checkpoint_every=0, validate_every=2), "augmentation": NO_AUGMENTATION}
    logs = train.run_training(config, _data(tmp_path, _rows(2)), tmp_path / "no-checkpoints", validation=validation)
    assert [r["step"] for r in logs if r.get("validation")] == [2, 3]
    assert not any(p.name.startswith(("state-step", "model-step")) for p in (tmp_path / "no-checkpoints").iterdir())


# R2LR-1: the minimal form never rewrites a boundary-unknown rectangle as known

def _reference_extent_row():
    condition = request_to_condition({"room_type": "bedroom", "room_size_m": [4., 3.], "furniture_list": ["bed", "lamp"]},
                                     room_size_semantics="reference_extent")
    targets = [{"id": o["id"], "bottom_center_m": [1. + i, 1., 0.], "target_size_local_m": [1., 1., 1.], "yaw_rad": 0.}
               for i, o in enumerate(condition["objects"])]
    return {"schema_version": "fastfill.v2", "condition": condition, "target": {"objects": targets},
            "validity": {"position": [[True] * 3] * 2, "size": [[True] * 3] * 2, "yaw": [True] * 2,
                         "yaw_symmetry_order": [2] * 2, "size_axis_swap_allowed": [False] * 2},
            "provenance": {"source": "fixture", "house_id": "h", "scene_id": "r", "split": "validation"}}


def test_minimal_form_and_projection_skip_boundary_unknown_rectangles():
    reference = _reference_extent_row()
    assert not minimal_form_eligible(reference["condition"]["room"])
    assert augment_sample(reference, ONLY_MINIMAL)["condition"] == reference["condition"]
    assert evaluate.project_minimal(reference) is None
    hull = real_rows()[0]
    hull["condition"]["room"].update(boundary_known=False, boundary_quality="hull")
    assert evaluate.project_minimal(hull) is None and augment_sample(hull, ONLY_MINIMAL)["condition"] == hull["condition"]
    known = real_rows()[0]
    assert evaluate.project_minimal(known)["condition"]["room"]["boundary_known"] is True
    del known["condition"]["room"]["boundary_known"]  # a missing flag counts as known, as in validate_scene
    assert minimal_form_eligible(known["condition"]["room"])


# R2LR-2: a deviation of exactly the tolerance is a rectangle under every quarter turn

def test_rectangle_at_exactly_one_centimetre_is_stable_under_quarter_turns():
    polygon = [[2.6999999999999997, 0.], [2.69, 2.7600000000000002], [0., 2.7600000000000002], [0., 0.]]
    for turns in range(4):
        condition = {"room": {"floor_polygon_xy_m": [list(p) for p in polygon]}, "constraints": []}
        _rigid_xy(condition, {"objects": []}, {}, turns, False)
        assert is_axis_aligned_rectangle(condition["room"]["floor_polygon_xy_m"]), turns
    assert not is_axis_aligned_rectangle([[0, 0], [4, 0.0101], [4, 3], [0, 3]])


# R2C-3: the plain convention of a swap object still sees a quarter-turned box

def test_plain_convention_yaw_of_a_swap_object_is_modulo_pi():
    sample, layout = _swap_row((2., 1., .8), math.pi / 2)  # same (sx, sy) a quarter turn off: a different box
    metrics = evaluate.reference_metrics(layout, sample, hungarian=False)
    assert metrics["yaw_error_rad_plain_convention"]["mean"] == pytest.approx(math.pi / 2)
    assert metrics["log_size_error_plain_convention"]["mean"] == pytest.approx(0., abs=1e-12)
    assert metrics["yaw_error_rad"]["mean"] == pytest.approx(0., abs=1e-12)  # box-equivalent: swapped sizes instead
    assert metrics["log_size_error"]["mean"] == pytest.approx(2 * math.log(2) / 3)
    sample, layout = _swap_row((2., 1., .8), math.pi)  # half a turn is the same box either way
    assert evaluate.reference_metrics(layout, sample, hungarian=False)["yaw_error_rad_plain_convention"]["mean"] == pytest.approx(0., abs=1e-12)


# R2C-4: a fixed size coordinate pins the yaw candidates of a swap object to the box's pi symmetry

def _quarter_turn_loss(swap, order, fixed):
    b = pair_batch()
    b["targets"]["size"] = torch.tensor([[[2., 1., 1.], [2., 1., 1.]]])
    b["yaw_symmetry_order"] = torch.full((1, 2), order)
    b["size_axis_swap_allowed"] = torch.full((1, 2), swap)
    if fixed:
        b["fixed_size_mask"] = torch.tensor([[[True, False, False]] * 2])
    p = pair_prediction(b)
    p["position_normalized"] = b["targets"]["position_normalized"].clone()
    p["size"] = torch.tensor([[[2., 1., 1.]] * 2])
    p["yaw_logits"] = (F.one_hot(torch.tensor(3), 12).float() * 50).expand(1, 2, 12).clone()  # bin 3 = yaw + pi/2
    return GeometryCriterion(LossConfig(hungarian=False))(p, b)


def test_pinned_swap_object_does_not_accept_a_quarter_turn():
    pinned, plain = _quarter_turn_loss(True, 4, True), _quarter_turn_loss(False, 2, True)
    assert pinned["size"].item() == pytest.approx(0., abs=1e-6) and pinned["yaw_cls"].item() > 10
    assert pinned["yaw_cls"].item() == pytest.approx(plain["yaw_cls"].item())
    assert _quarter_turn_loss(True, 4, False)["yaw_cls"].item() == pytest.approx(0., abs=1e-6)  # unpinned: still free


# R2C-5: a declared-unknown floor gives no floor contact at hand-off

def test_reference_extent_handoff_does_not_infer_floor_contact():
    condition = _reference_extent_row()["condition"]
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": o["id"], "bottom_center_m": [1. + 2 * i, 1.5, 0.], "target_size_local_m": [1., 1., .5], "yaw_rad": 0.}
        for i, o in enumerate(condition["objects"])]}
    scene = layout_to_roomgenbench(condition, layout)
    assert [(o["place"], o["support_status"]) for o in scene["objects"]] == [("unknown", "unknown")] * 2
    assert scene["room"]["position"]["z"] == 0.
    known = request_to_condition({"room_type": "bedroom", "room_size_m": [4., 3.], "furniture_list": ["bed", "lamp"]})
    assert [o["place"] for o in layout_to_roomgenbench(known, layout)["objects"]] == ["floor", "floor"]


# R2C-6: backbone paths are absolute and a symlinked snapshot keeps its revision

def test_backbone_provenance_resolves_relative_and_symlinked_snapshots(tmp_path, monkeypatch):
    snapshot = tmp_path / "hub/models--Qwen--Qwen3-8B/snapshots/0123abcd"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "qwen3"}\n')
    (tmp_path / "link").symlink_to(snapshot, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    expected = {"path": str(snapshot.resolve()), "revision": "0123abcd", "config_sha256": fingerprint(snapshot / "config.json")}
    assert backbone_provenance("link") == backbone_provenance(str(tmp_path / "link")) == expected
    (tmp_path / "local").mkdir()
    (tmp_path / "local/config.json").write_text("{}\n")
    assert backbone_provenance("local")["path"] == str((tmp_path / "local").resolve())
