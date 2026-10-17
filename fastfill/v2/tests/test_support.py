"""Support pointer head: labels survive support drops, the loss follows matched slots, decoding stays acyclic,
layouts carry the predicted parent and spread uses it."""
import pytest
import torch

from fastfill.v2.batch import SUPPORT_IGNORE, TinyTokenizer, augment_sample, collate_samples, support_targets
from fastfill.v2.evaluate import serialize_predictions
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.ops.remask_ablation import strip_support, support_accuracy, support_labels
from fastfill.v2.schema import validate_condition, validate_layout
from fastfill.v2.support import decode_support_parents
from fastfill.v2.tests.test_masked_layout import ONLY_GIVEN, TINY

MODEL = dict(TINY, position_head="grid_residual", position_grid=8, support_head=True)


def room_sample():
    condition = {"schema_version": "fastfill.v2",
                 "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]],
                          "floor_z_m": 0., "height_m": 2.8, "boundary_known": True, "floor_known": True},
                 "constraints": [],
                 "objects": [{"id": "table", "category": "table", "description": "table", "support_parent": "floor"},
                             {"id": "lamp", "category": "lamp", "description": "lamp", "support_parent": "table"},
                             {"id": "clock", "category": "clock", "description": "clock", "support_parent": "wall"},
                             {"id": "plant", "category": "plant", "description": "plant"}]}
    target = {"schema_version": "fastfill.v2", "objects": [
        {"id": "table", "target_size_local_m": [1., 1., .7], "bottom_center_m": [2., 2., 0.], "yaw_rad": 0.},
        {"id": "lamp", "target_size_local_m": [.2, .2, .4], "bottom_center_m": [2., 2., .7], "yaw_rad": 0.},
        {"id": "clock", "target_size_local_m": [.3, .05, .3], "bottom_center_m": [2., 3.97, 1.8], "yaw_rad": 0.},
        {"id": "plant", "target_size_local_m": [.3, .3, .6], "bottom_center_m": [.5, .5, 0.], "yaw_rad": 0.}]}
    validity = {"position": [[True] * 3] * 4, "size": [[True] * 3] * 4, "yaw": [True] * 4,
                "yaw_symmetry_order": [1] * 4, "size_axis_swap_allowed": [False] * 4, "exchangeable_group": [None] * 4}
    return {"schema_version": "fastfill.v2", "condition": validate_condition(condition), "target": target, "validity": validity}


def test_support_labels_are_the_declarations_before_any_drop():
    s = room_sample()
    assert support_targets(s) == [0, 2 + 0, 1, SUPPORT_IGNORE]
    dropped = augment_sample(s, {**ONLY_GIVEN, "given_pose_p": 0., "drop_support_p": 1.}, torch.Generator().manual_seed(0))
    assert not any("support_parent" in o for o in dropped["condition"]["objects"])
    assert collate_samples([dropped], TinyTokenizer())["support_target"][0].tolist() == [0, 2, 1, SUPPORT_IGNORE]


def test_support_loss_learns_and_follows_the_slot_matched_to_the_parent():
    torch.manual_seed(0)
    model = build_model(ModelConfig(**MODEL))
    b = collate_samples([room_sample()], TinyTokenizer())
    criterion = GeometryCriterion(LossConfig(support=1., hungarian=False))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(60):
        result = criterion(model(**model_inputs(b)), b)
        optimizer.zero_grad()
        result["loss"].backward()
        optimizer.step()
    assert result["term_counts"]["support"] == 3 and float(result["term_sums"]["support_accuracy"]) == 3
    logits = model(**model_inputs(b))["support_logits"][0]
    assert logits[1, 2 + 1] <= -1e3  # a slot never stands on itself


def test_decoding_keeps_declarations_and_never_closes_a_cycle():
    requests = [{"id": "a"}, {"id": "b"}, {"id": "c", "support_parent": "floor"}]
    logits = torch.full((3, 5), -10.)
    logits[0, 2 + 1] = 5.  # a on b
    logits[1, 2 + 0] = 4.  # b on a: would close a -> b -> a, takes its next best
    logits[1, 1] = 3.      # ... the wall
    logits[2, 2 + 0] = 9.  # c is declared on the floor
    assert decode_support_parents(logits, requests) == ["b", "wall", "floor"]


def test_layouts_carry_the_predicted_parent_and_the_schema_checks_it():
    torch.manual_seed(1)
    model = build_model(ModelConfig(**MODEL)).eval()
    s = strip_support(room_sample())
    b = collate_samples([s], TinyTokenizer())
    for decode in ("argmax", "spread"):
        with torch.no_grad():
            layout = serialize_predictions(model(**model_inputs(b)), b, grid_decode=decode)[0]
        assert all(o["support_parent"] in {"floor", "wall", "table", "lamp", "clock", "plant"} - {o["id"]}
                   for o in layout["objects"])
    bad = {**layout, "objects": [{**o, "support_parent": "lamp" if o["id"] == "table" else "table"
                                  if o["id"] == "lamp" else "floor"} for o in layout["objects"]]}
    with pytest.raises(ValueError, match="cycle"):
        validate_layout(bad, s["condition"])
    with pytest.raises(ValueError, match="support_parent"):
        validate_layout({**layout, "objects": [{**o, "support_parent": "sofa"} for o in layout["objects"]]}, s["condition"])
    plain = build_model(ModelConfig(**{**MODEL, "support_head": False})).eval()
    with torch.no_grad():
        assert "support_parent" not in serialize_predictions(plain(**model_inputs(b)), b, grid_decode="argmax")[0]["objects"][0]


def test_spread_puts_an_undeclared_object_on_its_predicted_parent():
    torch.manual_seed(2)
    model = build_model(ModelConfig(**MODEL)).eval()
    s = strip_support(room_sample())
    b = collate_samples([s], TinyTokenizer())
    with torch.no_grad():
        predictions = model(**model_inputs(b))
    predictions["support_logits"] = torch.full_like(predictions["support_logits"], -10.)
    predictions["support_logits"][0, :, 0] = 5.          # everything on the floor ...
    predictions["support_logits"][0, 1, 0] = -10.
    predictions["support_logits"][0, 1, 2 + 0] = 5.      # ... but the lamp on the table
    predictions["position_normalized"][0, 1, 2] = 1.2 / float(b["scale"][0, 2])  # predicted above the table top
    layout = {o["id"]: o for o in serialize_predictions(predictions, b, grid_decode="spread")[0]["objects"]}
    table, lamp = layout["table"], layout["lamp"]
    assert lamp["support_parent"] == "table"
    assert abs(lamp["bottom_center_m"][2] - (table["bottom_center_m"][2] + table["target_size_local_m"][2])) < 1e-6


def test_stripped_rows_score_support_against_the_hidden_declarations():
    s = room_sample()
    stripped = strip_support(s)
    assert not any("support_parent" in o for o in stripped["condition"]["objects"])
    labels = support_labels(s)
    assert labels == {"table": "floor", "lamp": "table", "clock": "wall"}
    layout = {"objects": [{"id": "table", "support_parent": "floor"}, {"id": "lamp", "support_parent": "floor"},
                          {"id": "clock", "support_parent": "wall"}, {"id": "plant", "support_parent": "floor"}]}
    acc = support_accuracy([{"layout": layout}], [labels])
    assert acc["all"] == 2 / 3 and acc["floor"] == 1 and acc["wall"] == 1 and acc["object"] == 0 and acc["objects"] == 3
    assert support_accuracy([{"layout": {"objects": [{"id": "table"}]}}], [labels]) is None
    # declarations the decoded condition still shows are copies, never scored
    copied = support_accuracy([{"layout": layout}], [labels], [support_labels(s)])
    assert copied["all"] is None and copied["objects"] == 0 and copied["declared_in_input"] == 3
    hidden = support_accuracy([{"layout": layout}], [labels], [support_labels(stripped)])
    assert hidden["all"] == 2 / 3 and hidden["declared_in_input"] == 0


def test_trainer_refuses_the_support_loss_without_the_head(tmp_path):
    import json
    from fastfill.v2 import train
    from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({**room_sample(), "provenance": {"source": "test", "scene_id": "s", "split": "train"}}) + "\n")
    config = {"model": {"backbone": "tiny", "decoder_dim": 16, "decoder_heads": 2, "decoder_layers": 1,
                        "tiny_hidden_size": 16, "lora_rank": 0},
              "loss": {"support": 1.}, "augmentation": NO_AUGMENTATION,
              "training": {"cpu": True, "steps": 1, "batch_size": 1, "max_length": 4096, "checkpoint_every": 0,
                           "validate_every": 0, "cpu_threads": 1}}
    with pytest.raises(ValueError, match="support_head"):
        train.run_training(config, data, tmp_path / "run")
    config["model"]["support_head"] = True
    logs = train.run_training(config, data, tmp_path / "run2")
    assert logs[-1]["unweighted"]["support"] is not None


def test_every_parameter_gets_a_gradient_without_any_support_label():
    """DDP (find_unused_parameters off) needs a gradient for every parameter on every rank, every step."""
    torch.manual_seed(3)
    model = build_model(ModelConfig(**{**MODEL, "pose_conditioning": True}))
    s = room_sample()
    for obj in s["condition"]["objects"]:
        obj.pop("support_parent", None)
    b = collate_samples([s], TinyTokenizer())
    assert (b["support_target"] < 0).all()
    for weights in (LossConfig(support=1.), LossConfig()):
        model.zero_grad()
        GeometryCriterion(weights)(model(**model_inputs(b)), b)["loss"].backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, missing
