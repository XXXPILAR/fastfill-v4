"""Masked layout Stage 2: encode/decode split, given-pose noise, and remasking decoding."""
import math

import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, augment_sample, collate_samples
from fastfill.v2.cell_costs import cell_cost_maps
from fastfill.v2.evaluate import serialize_predictions
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.remask import open_after, remask_predictions
from fastfill.v2.schema import validate_condition
from fastfill.v2.tests.test_cell_costs import DOOR, L_ROOM, SQUARE, cell, room
from fastfill.v2.tests.test_masked_layout import ONLY_GIVEN, TINY, given_sample

GRID = 16
MODEL = dict(TINY, position_head="grid_residual", position_grid=GRID, pose_conditioning=True)


def floor_sample(r, count=2):
    ids = [f"bed_{k}" for k in range(count)]
    condition = {"schema_version": "fastfill.v2", "room": r, "constraints": [],
                 "objects": [{"id": i, "category": "bed", "description": "bed", "support_parent": "floor"} for i in ids]}
    target = {"schema_version": "fastfill.v2",
              "objects": [{"id": i, "target_size_local_m": [1., 1., .5], "bottom_center_m": [1. + k, 1., 0.], "yaw_rad": 0.}
                          for k, i in enumerate(ids)]}
    validity = {"position": [[True] * 3] * count, "size": [[True] * 3] * count, "yaw": [True] * count,
                "yaw_symmetry_order": [2] * count, "size_axis_swap_allowed": [False] * count, "exchangeable_group": [None] * count}
    return {"schema_version": "fastfill.v2", "condition": validate_condition(condition), "target": target, "validity": validity}


def model_with_logits(logits_for_slot):
    """A tiny conditioned model whose cell logits are replaced; sizes are forced to 1 x 1 x 0.5 m on the floor."""
    torch.manual_seed(0)
    model = build_model(ModelConfig(**MODEL)).eval()
    decode = model.decode

    def forced(encoded, **kwargs):
        out = decode(encoded, **kwargs)
        n = out["position_cell_logits"].shape[1]
        out["position_cell_logits"] = torch.stack([logits_for_slot(i) for i in range(n)])[None]
        out["position_cell_residuals"] = torch.zeros_like(out["position_cell_residuals"])
        out["size"] = torch.tensor([1., 1., .5]).expand_as(out["size"]).clone()
        out["position_normalized"] = out["position_normalized"].clone()
        out["position_normalized"][..., 2] = 0.
        out["yaw_logits"] = torch.zeros_like(out["yaw_logits"])
        out["yaw_logits"][..., 0] = 10.
        out["yaw_residuals"] = torch.zeros_like(out["yaw_residuals"])
        return out
    model.decode = forced
    return model


def peaked(index, second=None):
    logits = torch.zeros(GRID * GRID)
    logits[index] = 10.
    if second is not None:
        logits[second] = 9.
    return logits


def chosen_cells(out, n):
    return [int(out["position_cell_logits"][0, i].argmax()) for i in range(n)]


def test_encode_then_decode_is_the_forward_pass():
    torch.manual_seed(0)
    model = build_model(ModelConfig(**MODEL)).eval()
    torch.nn.init.normal_(model.given_pose_embedding[-1].weight)
    b = collate_samples([given_sample(3, ["chair_0"])], TinyTokenizer())
    inputs = model_inputs(b)
    with torch.no_grad():
        whole = model(**inputs)
        parts = model.decode(model.encode(inputs["input_ids"], inputs["attention_mask"], inputs["object_spans"],
                                          inputs["slot_mask"]),
                             **{k: v for k, v in inputs.items() if k.startswith(("fixed_", "given_"))})
    for key in ("position_normalized", "size", "yaw_logits", "position_cell_logits"):
        assert torch.equal(whole[key], parts[key])


def test_one_round_without_legality_is_the_argmax_decode():
    torch.manual_seed(1)
    model = build_model(ModelConfig(**MODEL)).eval()
    b = collate_samples([floor_sample(room(SQUARE, [DOOR]), 3)], TinyTokenizer())
    with torch.no_grad():
        plain = serialize_predictions(model(**model_inputs(b)), b, grid_decode="argmax")
    remasked = serialize_predictions(remask_predictions(model, b, rounds=1, legal=False), b, grid_decode="argmax")
    assert remasked == plain


def test_rounds_run_the_backbone_once_and_commit_every_slot_on_the_cosine_schedule():
    torch.manual_seed(2)
    model = build_model(ModelConfig(**MODEL)).eval()
    calls = []
    backbone = model.backbone.forward
    model.backbone.forward = lambda *a, **k: calls.append(1) or backbone(*a, **k)
    b = collate_samples([floor_sample(room(SQUARE), 6)], TinyTokenizer())
    stats = {}
    remask_predictions(model, b, rounds=4, legal=False, stats=stats)
    assert len(calls) == 1 and stats["decode_calls"] == 4
    assert stats["commits_per_round"] == [6 - open_after(6, 0, 4), open_after(6, 0, 4) - open_after(6, 1, 4),
                                          open_after(6, 1, 4) - open_after(6, 2, 4), open_after(6, 2, 4)]
    assert sum(stats["commits_per_round"]) == 6
    with pytest.raises(ValueError, match="pose_conditioning"):
        remask_predictions(build_model(ModelConfig(**{**MODEL, "pose_conditioning": False})), b, rounds=2)


def test_legal_decoding_moves_a_slot_out_of_the_missing_quarter_and_away_from_the_door():
    b = collate_samples([floor_sample(room(L_ROOM, [DOOR]), 2)], TinyTokenizer())
    # slot 0 prefers a cell outside the L, slot 1 the cell in front of the door (x 4.0-4.5, y 0-0.5)
    model = model_with_logits(lambda i: peaked(cell(14, 14), cell(3, 10)) if i == 0 else peaked(cell(8, 0), cell(3, 3)))
    raw = chosen_cells(remask_predictions(model, b, rounds=1, legal=False), 2)
    legal = chosen_cells(remask_predictions(model, b, rounds=1, legal=True), 2)
    assert raw == [cell(14, 14), cell(8, 0)]
    assert legal == [cell(3, 10), cell(3, 3)]


def test_committed_floor_footprints_block_later_slots():
    b = collate_samples([floor_sample(room(SQUARE), 2)], TinyTokenizer())
    # both 1 x 1 m beds want the same cell; the second most likely cells are far apart
    model = model_with_logits(lambda i: peaked(cell(8, 8), cell(2, 2) if i == 0 else cell(13, 13)))
    for rounds in (1, 2):  # within one round (confidence order) and across rounds
        cells = chosen_cells(remask_predictions(model, b, rounds=rounds, legal=True), 2)
        assert cell(8, 8) in cells and len(set(cells)) == 2
        assert set(cells) - {cell(8, 8)} <= {cell(2, 2), cell(13, 13)}
    assert chosen_cells(remask_predictions(model, b, rounds=1, legal=False), 2) == [cell(8, 8)] * 2


def revealing_seed(s, options):
    """A seed whose draw reveals at least two requests (the revealed share follows the cosine schedule)."""
    return next(k for k in range(100) if len(augment_sample(s, options, torch.Generator().manual_seed(k))
                                             .get("_given_pose_ids", ())) >= 2)


def test_given_pose_noise_draws_nothing_by_default_and_never_touches_targets():
    s = given_sample(4, [])
    s.pop("_given_pose_ids")
    seed = revealing_seed(s, ONLY_GIVEN)
    plain = augment_sample(s, ONLY_GIVEN, torch.Generator().manual_seed(seed))
    assert "_given_pose_noise" not in plain and plain["_given_pose_ids"]
    noisy_options = {**ONLY_GIVEN, "given_pose_noise_m": .2, "given_pose_yaw_noise_deg": 10., "given_pose_size_noise": .1}
    noisy = augment_sample(s, noisy_options, torch.Generator().manual_seed(seed))
    assert noisy["_given_pose_ids"] == plain["_given_pose_ids"]  # the ids are drawn before the noise
    assert set(noisy["_given_pose_noise"]) == set(noisy["_given_pose_ids"])
    clean_batch, noisy_batch = (collate_samples([x], TinyTokenizer()) for x in (plain, noisy))
    given = clean_batch["given_pose_mask"]
    assert torch.equal(given, noisy_batch["given_pose_mask"])
    assert not torch.allclose(clean_batch["given_pose"][given], noisy_batch["given_pose"][given])
    for key in ("position_normalized", "size", "yaw"):
        assert torch.equal(clean_batch["targets"][key].nan_to_num(), noisy_batch["targets"][key].nan_to_num())
    with pytest.raises(ValueError, match="nonnegative"):
        augment_sample(s, {**ONLY_GIVEN, "given_pose_noise_m": -1.}, torch.Generator())


def test_corrupted_given_poses_land_anywhere_in_the_room_box():
    s = given_sample(4, [])
    s.pop("_given_pose_ids")
    options = {**ONLY_GIVEN, "given_pose_corrupt_p": 1.}
    corrupt = augment_sample(s, options, torch.Generator().manual_seed(revealing_seed(s, options)))
    b = collate_samples([corrupt], TinyTokenizer())
    xy = b["given_pose"][b["given_pose_mask"]][:, :2]
    assert ((xy >= 0) & (xy <= 1)).all()
    assert not torch.allclose(xy, b["targets"]["position_normalized"][b["given_pose_mask"]][:, :2])


def test_cosine_schedule_ends_with_nothing_open():
    assert [open_after(10, r, 4) for r in range(4)] == [math.floor(10 * math.cos(math.pi / 2 * k / 4)) for k in (1, 2, 3, 4)]
    assert open_after(10, 3, 4) == 0


def test_a_residual_that_misses_the_legal_part_of_its_cell_is_snapped_back():
    from shapely.geometry import box
    b = collate_samples([floor_sample(room(SQUARE), 2)], TinyTokenizer())
    # bed 0 at cell (8, 8): x 2.0-2.25; bed 1 prefers cell (12, 8): x 3.0-3.25, legal only near its right border,
    # and its residual pulls it to the left border, 0.125 m into bed 0
    model = model_with_logits(lambda i: peaked(cell(8, 8)) if i == 0 else peaked(cell(12, 8)))
    decode = model.decode

    def pulled(encoded, **kwargs):
        out = decode(encoded, **kwargs)
        out["position_cell_residuals"][0, 1, :, 0] = -1.
        return out
    model.decode = pulled
    layouts = {legal: serialize_predictions(remask_predictions(model, b, rounds=2, legal=legal), b, grid_decode="argmax")[0]
               for legal in (False, True)}
    for legal, layout in layouts.items():
        (x0, y0, _), (x1, y1, _) = (o["bottom_center_m"] for o in layout["objects"])
        overlap = box(x0 - .5, y0 - .5, x0 + .5, y0 + .5).intersection(box(x1 - .5, y1 - .5, x1 + .5, y1 + .5)).area
        assert (overlap < .03 * 1.) if legal else overlap > .1


def test_only_objects_sharing_a_height_interval_block_each_other():
    b = collate_samples([floor_sample(room(SQUARE), 3)], TinyTokenizer())
    for obj in b["conditions"][0]["objects"]:
        obj.pop("support_parent")  # floor contact from the predicted z
    # slot 0: a table (0-0.5 m), slot 1: a lamp on it (0.5-0.9 m), slot 2: a chair (0-0.5 m); all want cell (8, 8)
    model = model_with_logits(lambda i: peaked(cell(8, 8), cell(2, 2)) * (1. if i < 2 else .9))
    decode = model.decode
    height = b["scale"][0, 2]

    def stacked(encoded, **kwargs):
        out = decode(encoded, **kwargs)
        out["position_normalized"][0, 1, 2] = .5 / height
        out["size"][0, 1, 2] = .4
        return out
    model.decode = stacked
    cells = chosen_cells(remask_predictions(model, b, rounds=1, legal=True), 3)
    assert cells == [cell(8, 8), cell(8, 8), cell(2, 2)]


def test_fixed_objects_block_floor_slots_under_their_height_only():
    r = room(SQUARE)
    # a column over cells (8, 8) and a wall cabinet hung at 1.2-2.0 m over cell (2, 2)
    r["fixed_objects"] = [{"id": "column", "category": "column", "size_local_m": [.6, .6, 2.5],
                           "bottom_center_m": [2.125, 2.125, 0.], "yaw_rad": 0.},
                          {"id": "cabinet", "category": "cabinet", "size_local_m": [1., 1., .8],
                           "bottom_center_m": [.625, .625, 1.2], "yaw_rad": 0.}]
    b = collate_samples([floor_sample(r, 1)], TinyTokenizer())
    model = model_with_logits(lambda i: peaked(cell(8, 8), cell(2, 2)))
    assert chosen_cells(remask_predictions(model, b, rounds=1, legal=False), 1) == [cell(8, 8)]
    # the 0.5 m tall bed fits under the cabinet but not through the column
    assert chosen_cells(remask_predictions(model, b, rounds=1, legal=True), 1) == [cell(2, 2)]
    costs, _ = cell_cost_maps(r, [([1., 1.], 0., True, (0., .5))])
    assert costs[0, cell(8, 8)] > 0 and costs[0, cell(2, 2)] == 0
    assert cell_cost_maps(r, [([1., 1.], 0., True, (1.3, 1.9))])[0][0, cell(2, 2)] > 0


def test_retraction_masks_a_low_confidence_commit_again_and_frees_its_footprint():
    b = collate_samples([floor_sample(room(SQUARE), 3)], TinyTokenizer())
    torch.manual_seed(0)
    model = build_model(ModelConfig(**MODEL)).eval()
    decode, calls = model.decode, []

    def by_round(encoded, **kwargs):
        out = decode(encoded, **kwargs)
        r = len(calls)
        calls.append(1)
        # round 0: only slot 0 is sure (of cell A); round 1: slots 1 and 2 are surer of A than slot 0 was
        strength = [[8., 2., 2.], [8., 12., 11.], [8., 12., 11.]][r]
        out["position_cell_logits"] = torch.stack([peaked(cell(8, 8), (cell(2, 2), cell(13, 13), cell(2, 13))[i]) * strength[i] / 10
                                                   for i in range(3)])[None]
        out["position_cell_residuals"] = torch.zeros_like(out["position_cell_residuals"])
        out["size"] = torch.tensor([1., 1., .5]).expand_as(out["size"]).clone()
        out["position_normalized"] = out["position_normalized"].clone()
        out["position_normalized"][..., 2] = 0.
        return out
    model.decode = by_round
    stats = {}
    out = remask_predictions(model, b, rounds=3, legal=True, retract=True, stats=stats)
    assert stats["commits_per_round"] == [1, 2, 1] and stats["retractions_per_round"] == [0, 1, 0]
    # slot 0 held A after round 0; retracted, its footprint no longer blocks: slot 1 takes A, slot 0 moves
    assert chosen_cells(out, 3) == [cell(2, 2), cell(8, 8), cell(2, 13)]
    calls.clear()
    assert remask_predictions(model, b, rounds=3, legal=True, stats=stats) and stats["retractions_per_round"] == [0, 0, 0]
