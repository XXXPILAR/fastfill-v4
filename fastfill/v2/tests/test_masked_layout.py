"""Masked layout training (Stage 1): given-pose sampling, features, model conditioning, supervision and matching."""
import math

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.batch import TinyTokenizer, augment_sample, collate_samples
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.masked_layout import POSE_FEATURES, pose_features
from fastfill.v2.matching import match_batch
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.objective import supervision_masks
from fastfill.v2.tests.test_batch import sample
from fastfill.v2.tests.test_supervision import batch, prediction

ONLY_GIVEN = {"rotate90": False, "shuffle_objects": False, "drop_constraints_p": 0., "drop_support_p": 0.,
              "category_only_description_p": 0., "minimal_form_p": 0., "given_pose_p": 1.}
TINY = dict(backbone="tiny", decoder_dim=32, decoder_heads=4, tiny_hidden_size=32)


def given_sample(n, ids):
    s = sample(n)
    s["_given_pose_ids"] = ids
    return s


def test_features_are_invariant_under_the_labels_own_symmetries():
    position = torch.tensor([.2, .4, 0.])
    plain = pose_features(position, torch.tensor([2., 1., 1.]), torch.tensor(.3), torch.tensor(2), torch.tensor(False))
    turned = pose_features(position, torch.tensor([2., 1., 1.]), torch.tensor(.3 + math.pi), torch.tensor(2), torch.tensor(False))
    assert plain.shape == (POSE_FEATURES,)
    assert torch.allclose(plain, turned, atol=1e-6)
    # A K1 box written (sy, sx) at yaw + pi/2 is the same box.
    swapped = pose_features(position, torch.tensor([1., 2., 1.]), torch.tensor(.3 - math.pi / 2), torch.tensor(4), torch.tensor(True))
    written = pose_features(position, torch.tensor([2., 1., 1.]), torch.tensor(.3), torch.tensor(4), torch.tensor(True))
    assert torch.allclose(swapped, written, atol=1e-6)
    # Without the K1 flag the axis order is semantic.
    assert not torch.allclose(pose_features(position, torch.tensor([1., 2., 1.]), torch.tensor(.3), torch.tensor(2), torch.tensor(False)),
                              plain, atol=1e-3)


def test_given_pose_draws_nothing_at_zero_so_earlier_configs_replay():
    base = {"rotate90": True, "shuffle_objects": True}
    old = augment_sample(sample(4), base, torch.Generator().manual_seed(7))
    new = augment_sample(sample(4), {**base, "given_pose_p": 0.}, torch.Generator().manual_seed(7))
    assert old == new and "_given_pose_ids" not in new


def test_given_ids_are_deterministic_labelled_and_never_the_whole_room():
    s = sample(6)
    s["validity"]["yaw"][0] = False  # no complete pose: never given
    seen = set()
    for seed in range(200):
        first = augment_sample(s, ONLY_GIVEN, torch.Generator().manual_seed(seed))
        again = augment_sample(s, ONLY_GIVEN, torch.Generator().manual_seed(seed))
        assert first.get("_given_pose_ids") == again.get("_given_pose_ids")
        ids = first.get("_given_pose_ids", [])
        assert "chair_0" not in ids and len(ids) <= 5
        seen.add(len(ids))
    assert 0 in seen and max(seen) >= 3  # the schedule covers nothing given through most of the room given


def test_single_request_room_is_never_given():
    assert all("_given_pose_ids" not in augment_sample(sample(1), ONLY_GIVEN, torch.Generator().manual_seed(seed))
               for seed in range(20))


def test_collate_marks_given_slots_with_features_and_rejects_unknown_ids():
    b = collate_samples([given_sample(3, ["chair_1"]), sample(2)], TinyTokenizer())
    assert b["given_pose_mask"].tolist() == [[False, True, False], [False, False, False]]
    assert b["given_pose"].shape == (2, 3, POSE_FEATURES)
    expected = pose_features(b["targets"]["position_normalized"][0, 1], b["targets"]["size"][0, 1], b["targets"]["yaw"][0, 1],
                             b["yaw_symmetry_order"][0, 1], b["size_axis_swap_allowed"][0, 1])
    assert torch.allclose(b["given_pose"][0, 1], expected)
    assert not b["given_pose"][~b["given_pose_mask"]].any()
    assert "given_pose_mask" not in collate_samples([sample(2)], TinyTokenizer())
    with pytest.raises(ValueError, match="name requests"):
        collate_samples([given_sample(2, ["sofa"])], TinyTokenizer())
    unlabelled = given_sample(2, ["chair_0"])
    unlabelled["validity"]["size"][0] = [True, False, True]
    with pytest.raises(ValueError, match="complete position, size and yaw"):
        collate_samples([unlabelled], TinyTokenizer())


def test_given_poses_reach_the_other_slots_only_through_a_conditioned_model():
    torch.manual_seed(0)
    b = collate_samples([given_sample(3, ["chair_0"])], TinyTokenizer())
    with pytest.raises(ValueError, match="pose_conditioning"):
        build_model(ModelConfig(**TINY))(**model_inputs(b))
    model = build_model(ModelConfig(**TINY, pose_conditioning=True)).eval()
    plain = {k: v for k, v in model_inputs(b).items() if not k.startswith("given_")}
    with torch.no_grad():
        # The zero-initialized last layer: a fresh conditioned model ignores given poses.
        assert torch.equal(model(**model_inputs(b))["position_normalized"], model(**plain)["position_normalized"])
        torch.nn.init.normal_(model.given_pose_embedding[-1].weight)
        conditioned, unconditioned = model(**model_inputs(b)), model(**plain)
    # Slot 2 is not given, yet attends to the given slot 0.
    assert not torch.allclose(conditioned["position_normalized"][0, 2], unconditioned["position_normalized"][0, 2])


def test_pose_embedding_joins_the_graph_without_given_poses_for_ddp():
    # A rank whose batch reveals nothing must still give every parameter a gradient (DDP, no find_unused_parameters).
    torch.manual_seed(0)
    model = build_model(ModelConfig(**TINY, pose_conditioning=True))
    b = collate_samples([sample(2)], TinyTokenizer())
    assert "given_pose_mask" not in b
    outputs = model(**model_inputs(b))
    (outputs["position_normalized"].sum() + outputs["size"].sum()).backward()
    assert all(p.grad is not None for p in model.given_pose_embedding.parameters())


def test_given_slots_leave_every_supervised_term():
    b = batch()
    b["exchangeable_group"] = [[None, None]]
    b["given_pose_mask"] = torch.tensor([[True, False]])
    masks = supervision_masks(b)
    assert all(masks[key].tolist() == [[False, True]] for key in ("position", "size", "yaw", "box"))
    p = prediction(b)
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    assert result["counts"]["position"] == 1 and result["counts"]["size"] == 1 and result["counts"]["yaw"] == 1
    result["loss"].backward()
    assert not p["size"].grad[0, 0].any() and p["size"].grad[0, 1].any()
    assert result["active_objective_count_local"] == 3  # position, size, yaw of the one predicted slot


def test_given_member_keeps_fixed_identity_inside_its_exchangeable_group():
    b = batch()
    p = prediction(b)  # predictions swapped: free matching pairs slot 0 with target 1
    assert match_batch(p, b).tolist() == [[1, 0]]
    b["given_pose_mask"] = torch.tensor([[True, False]])
    assert match_batch(p, b).tolist() == [[0, 1]]


def test_scene_regularizers_refuse_given_poses():
    b = batch()
    b["given_pose_mask"] = torch.tensor([[True, False]])
    with pytest.raises(ValueError, match="regularizers"):
        GeometryCriterion(LossConfig(collision=1.))(prediction(b), b)


def test_masked_layout_training_runs_end_to_end_and_needs_a_conditioned_model(tmp_path):
    from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data
    from fastfill.v2.tests.test_train_loop_fix20261006 import _config
    rows = []
    for index in range(3):
        row = sample(4)
        row["provenance"] = {**row["provenance"], "scene_id": f"s{index}"}
        rows.append(row)
    data = _data(tmp_path, rows)
    config = _config(steps=3)
    config["augmentation"] = {**NO_AUGMENTATION, "given_pose_p": 1.}
    with pytest.raises(ValueError, match="pose_conditioning"):
        train.run_training(config, data, tmp_path / "refused")
    config["model"] = {**config["model"], "pose_conditioning": True}
    logs = train.run_training(config, data, tmp_path / "run")
    assert len(logs) == 3 and all(math.isfinite(log["loss"]) for log in logs)
    # Some rooms reveal poses, so fewer than all 4 requests x 3 terms are supervised in at least one step.
    assert min(log["active_objective_count"] for log in logs) < 12


def test_resume_digest_ignores_later_fields_at_their_defaults():
    old = {"model": {"decoder_dim": 128}, "augmentation": {"rotate90": True}, "training": {"seed": 42}}
    new = {"model": {"decoder_dim": 128, "pose_conditioning": False},
           "augmentation": {"rotate90": True, "given_pose_p": 0.}, "training": {"seed": 42}}
    assert train._digest_view(new) == old
    changed = {**new, "augmentation": {"rotate90": True, "given_pose_p": .5}}
    assert train._digest_view(changed)["augmentation"] == {"rotate90": True, "given_pose_p": .5}


def test_anchor_categories_follow_the_design_list_and_exclude_bedside_furniture():
    from fastfill.v2.anchor_probe import is_anchor
    assert all(is_anchor(c) for c in ("bed", "King Size Bed", "three seat sofa", "couch", "desk", "dining table", "tv stand", "table"))
    assert not any(is_anchor(c) for c in ("bedside table", "nightstand", "side table", "coffee table", "chair", "bed_side_cabinet", "desk lamp", "desk chair", "bed pillow", "console table"))
