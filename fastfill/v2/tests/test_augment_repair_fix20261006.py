"""Review repairs 2026-10-06 (H): augmentation float noise, regrouping after drops, loader budget fallback."""
from copy import deepcopy

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.batch import TinyTokenizer, _rigid_xy, augment_sample, collate_samples, tokenize_condition
from fastfill.v2.matching import certify_group, match_batch
from fastfill.v2.tests.test_batch_augment_fix20261006 import NO_AUGMENT, sample


def chairs(descriptions, support=None, constraints=()):
    objects = [{"id": f"obj_{i:04d}", "category": "chair", "description": d, **({"support_parent": s} if s else {})}
               for i, (d, s) in enumerate(zip(descriptions, support or [None] * len(descriptions)))]
    return {"schema_version": "fastfill.v2",
            "condition": {"schema_version": "fastfill.v2", "room": {
                "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [6, 0], [6, 4], [0, 4]], "floor_z_m": 0.,
                "height_m": 3., "floor_known": True, "boundary_known": True}, "objects": objects, "constraints": list(constraints)},
            "target": {"objects": [{"id": o["id"], "target_size_local_m": [.5, .5, .9], "bottom_center_m": [1. + 4. * i, 1., 0.], "yaw_rad": 0.}
                                   for i, o in enumerate(objects)]},
            "validity": {"position": [[True] * 3] * len(objects), "size": [[True] * 3] * len(objects), "yaw": [True] * len(objects),
                         "yaw_symmetry_order": [2] * len(objects), "exchangeable_group": [None] * len(objects)},
            "provenance": {"split": "train", "source": "offline_test", "house_id": "h", "scene_id": "s"}}


def tokens(condition):
    return len(tokenize_condition(condition, TinyTokenizer())[0])


# --- R1 / R-leak-2: float noise and the context budget ------------------------------

def test_rotation_does_not_grow_rendered_numbers_from_float_noise():
    s = chairs(["a", "b"])
    s["condition"]["room"]["floor_polygon_xy_m"] = [[0., 0.], [8.8, 0.], [8.8, 4.4], [0., 4.4]]
    # Source yaws are full-precision floats (k * pi/2 never lengthens this one); positions are mostly short decimals.
    s["condition"]["room"]["fixed_objects"] = [{"id": "fixed_0000", "category": "door", "size_local_m": [.1, .9, 2.1],
                                                "bottom_center_m": [1.1, 2.2, 0.], "yaw_rad": -0.3141592653589793}]
    s["target"]["objects"][0]["bottom_center_m"] = [3.3, 0.1 + 0.2, 0.]  # 0.30000000000000004 in the source
    base = tokens(s["condition"])
    for k in (1, 2, 3):
        a = deepcopy(s)
        _rigid_xy(a["condition"], a["target"], a["validity"], k, False)
        numbers = [v for p in a["condition"]["room"]["floor_polygon_xy_m"] for v in p]
        numbers += a["condition"]["room"]["fixed_objects"][0]["bottom_center_m"][:2] + a["target"]["objects"][0]["bottom_center_m"][:2]
        assert all(len(repr(v)) <= 4 for v in numbers), numbers  # 8.8 - 1.1 renders as 7.7, not 7.699999999999999
        assert tokens(a["condition"]) <= base
        collate_samples([a], TinyTokenizer())
    # Full-precision source coordinates are kept, not shortened to six decimals.
    s["target"]["objects"][1]["bottom_center_m"] = [1.697421321709293, 2.5, 0.]
    a = deepcopy(s)
    _rigid_xy(a["condition"], a["target"], a["validity"], 2, False)
    assert a["target"]["objects"][1]["bottom_center_m"][0] == pytest.approx(8.8 - 1.697421321709293, abs=1e-12)
    assert len(repr(a["target"]["objects"][1]["bottom_center_m"][0])) >= 15


def test_training_rows_fall_back_to_the_preflighted_row_when_augmentation_exceeds_the_budget():
    row = chairs(["red chair", "blue chair"])
    for obj, target in zip(row["condition"]["objects"], row["target"]["objects"]):
        obj["id"] = target["id"] = obj["id"][-1]  # short IDs: shuffling renames them to the longer obj_%04d
    budget = tokens(row["condition"])
    tight = train._Rows([row], {**NO_AUGMENT, "shuffle_objects": True}, 7, TinyTokenizer(), budget)
    assert tight[0] == row and tight.fallbacks == 1
    collate_samples([tight[0]], TinyTokenizer(), max_length=budget)
    roomy = train._Rows([row], {**NO_AUGMENT, "shuffle_objects": True}, 7, TinyTokenizer(), budget + 64)
    assert roomy[0]["condition"]["objects"][0]["id"] == "obj_0000" and roomy.fallbacks == 0
    unchecked = train._Rows([row], {**NO_AUGMENT, "shuffle_objects": True}, 7)
    assert unchecked[0] == roomy[0]  # no tokenizer: legacy behaviour, caller owns the budget


# --- R2 / R-leak-1: identical requests after drops are regrouped ---------------------

def test_category_only_descriptions_regroup_identical_chairs_and_match_them_by_geometry():
    s = chairs(["a red armchair", "a blue dining chair"])
    a = augment_sample(s, {**NO_AUGMENT, "category_only_description_p": 1.}, torch.Generator().manual_seed(0))
    assert a["validity"]["exchangeable_group"] == ["anonymous_0", "anonymous_0"]
    b = collate_samples([a], TinyTokenizer())
    assert b["exchangeable_group"] == [["anonymous_0", "anonymous_0"]]
    certify_group(b["objects"][0], b["conditions"][0]["constraints"], [0, 1])
    swapped = {"position_normalized": b["targets"]["position_normalized"].flip(1), "size": torch.ones(1, 2, 3), "slot_mask": b["slot_mask"]}
    assert match_batch(swapped, b, True)[0].tolist() == [1, 0]
    # Without the drop the descriptions still differ: fixed identity as before.
    plain = collate_samples([augment_sample(s, NO_AUGMENT)], TinyTokenizer())
    assert plain["exchangeable_group"] == [[None, None]] and match_batch({**swapped, "slot_mask": plain["slot_mask"]}, plain, True)[0].tolist() == [0, 1]


def test_drop_support_regroups_nightstands_that_differed_only_by_support_parent():
    s = chairs(["nightstand", "nightstand"], support=(None, "floor"))
    assert augment_sample(s, NO_AUGMENT)["validity"]["exchangeable_group"] == [None, None]
    a = augment_sample(s, {**NO_AUGMENT, "drop_support_p": 1.}, torch.Generator().manual_seed(0))
    assert a["validity"]["exchangeable_group"] == ["anonymous_0", "anonymous_0"]
    assert all("support_parent" not in o for o in a["condition"]["objects"])


def test_regrouping_requires_complete_position_labels_and_follows_target_order_after_shuffle():
    s = chairs(["a", "b", "c"])
    s["validity"]["position"][2] = [True, True, False]
    for seed in range(6):
        a = augment_sample(s, {**NO_AUGMENT, "shuffle_objects": True, "category_only_description_p": 1.}, torch.Generator().manual_seed(seed))
        labels = a["validity"]["exchangeable_group"]
        incomplete = [i for i, row in enumerate(a["validity"]["position"]) if not all(row)]
        assert len(incomplete) == 1 and labels[incomplete[0]] is None
        assert [l for i, l in enumerate(labels) if i != incomplete[0]] == ["anonymous_0", "anonymous_0"]
        b = collate_samples([a], TinyTokenizer())
        members = [i for i, g in enumerate(b["exchangeable_group"][0]) if g]
        certify_group(b["objects"][0], b["conditions"][0]["constraints"], members)
        assert b["validity"]["position"][0, members].all()


def test_build_time_group_survives_when_the_wider_candidate_fails_certification():
    s = chairs(["red", "red", "blue"], constraints=[{"type": "against_wall", "object_id": "obj_0002"}])
    s["validity"]["exchangeable_group"] = ["anonymous_0", "anonymous_0", None]
    a = augment_sample(s, {**NO_AUGMENT, "category_only_description_p": 1.}, torch.Generator().manual_seed(0))
    labels = a["validity"]["exchangeable_group"]
    assert labels[0] == labels[1] and labels[0] is not None and labels[2] is None  # the constrained chair cannot join; the pair stays
    b = collate_samples([a], TinyTokenizer())
    certify_group(b["objects"][0], b["conditions"][0]["constraints"], [0, 1])
    match_batch({"position_normalized": b["targets"]["position_normalized"], "size": torch.ones(1, 3, 3), "slot_mask": b["slot_mask"]}, b, True)


def test_rigid_only_augmentation_keeps_build_time_labels_verbatim():
    s = sample()
    for seed in range(4):
        a = augment_sample(s, {**NO_AUGMENT, "rotate90": True}, torch.Generator().manual_seed(seed))
        assert a["validity"]["exchangeable_group"] == s["validity"]["exchangeable_group"]
