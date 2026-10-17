"""Round 2 (C2): box-symmetric size/yaw candidates (K1), grid-residual position head (K7),
checkpoint-bound predict settings (K8) and the formal training configs (K9)."""
import json
import math
from pathlib import Path

import pytest
import torch

from fastfill.v2 import predict, train
from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.geometry import decode_grid_position, encode_grid_position
from fastfill.v2.io import CHECKPOINT_MANIFEST
from fastfill.v2.losses import GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch
from fastfill.v2.model import ModelConfig, build_model, load_model, model_inputs
from fastfill.v2.schema import validate_layout
from fastfill.v2.tests.test_batch import sample as batch_sample
from fastfill.v2.tests.test_execution import sample as floor_sample
from fastfill.v2.tests.test_supervision import batch, prediction

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# --- K1: box-symmetric objects ------------------------------------------------

def box_batch(swap, yaw_valid=True):
    b = batch()
    b["targets"]["size"] = torch.tensor([[[2., 1., 1.], [2., 1., 1.]]])
    b["validity"]["yaw"] = torch.full((1, 2), yaw_valid)
    b["yaw_symmetry_order"] = torch.full((1, 2), 4)
    b["size_axis_swap_allowed"] = torch.full((1, 2), swap)
    return b


def box_prediction(b, size, yaw_bin=3):
    p = prediction(b)
    p["position_normalized"] = b["targets"]["position_normalized"].clone().requires_grad_()
    p["size"] = torch.tensor(size).expand(1, 2, 3).clone().requires_grad_()
    p["yaw_logits"] = (torch.nn.functional.one_hot(torch.tensor(yaw_bin), 12).float() * 50).expand(1, 2, 12).clone().requires_grad_()
    return p


@pytest.mark.parametrize("yaw_valid", [True, False])
def test_swap_allowed_box_in_the_other_axis_order_costs_nothing(yaw_valid):
    # (sy, sx) at yaw + pi/2 is the same box; order 4 alone (no swap tier) still charges the size.
    for swap, expected in ((True, 0.), (False, 2 * math.log(2) / 3)):
        b = box_batch(swap, yaw_valid)
        result = GeometryCriterion(LossConfig(hungarian=False))(box_prediction(b, [1., 2., 1.]), b)
        assert result["size"].item() == pytest.approx(expected, abs=1e-6)
        assert result["yaw_cls"].item() == pytest.approx(0., abs=1e-6)


def test_swap_candidate_is_chosen_jointly_with_yaw():
    # The plain order fits the size better, but only the odd candidate agrees with the yaw head.
    b = box_batch(True)
    p = box_prediction(b, [1.9, 1.1, 1.])
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    swapped = (abs(math.log(1.9)) + abs(math.log(1.1 / 2))) / 3
    assert result["size"].item() == pytest.approx(swapped, rel=1e-5)
    assert result["yaw_cls"].item() == pytest.approx(0., abs=1e-6)
    result["loss"].backward()
    assert torch.isfinite(p["size"].grad).all() and p["size"].grad.abs().sum() > 0
    # Yaw head tied between the plain (bin 0) and odd (bin 3) candidates: the size order decides.
    p = box_prediction(b, [1.1, 1.9, 1.])
    p["yaw_logits"] = (p["yaw_logits"] + p["yaw_logits"].roll(-3, -1)).detach().requires_grad_()
    result = GeometryCriterion(LossConfig(hungarian=False))(p, b)
    assert result["size"].item() == pytest.approx((abs(math.log(1.1)) + abs(math.log(1.9 / 2))) / 3, rel=1e-5)


def test_fixed_size_coordinate_pins_the_axis_order():
    b = box_batch(True, yaw_valid=False)
    b["fixed_size_mask"] = torch.tensor([[[True, False, False]] * 2])
    result = GeometryCriterion(LossConfig(hungarian=False))(box_prediction(b, [2., 2., 1.]), b)
    assert result["size"].item() == pytest.approx(math.log(2) / 3, rel=1e-5)  # a 2 x 2 box is not a 2 x 1 box


def test_fixed_height_alone_keeps_the_box_equivalence():
    # Audit C5: (1, 2, 1) at yaw + pi/2 is the label's (2, 1, 1) box; a fixed sz never moves under the swap.
    b = box_batch(True)
    b["fixed_size_mask"] = torch.tensor([[[False, False, True]] * 2])
    result = GeometryCriterion(LossConfig(hungarian=False))(box_prediction(b, [1., 2., 1.]), b)
    assert result["loss"].item() == pytest.approx(0., abs=1e-6)  # was 50.46 when any fixed axis pinned the swap


def test_matching_size_cost_takes_the_swap_minimum():
    b = batch()
    b["targets"]["size"] = torch.tensor([[[2., 1., 1.], [1., 1., 1.]]])
    p = prediction(b)
    p["position_normalized"] = b["targets"]["position_normalized"].clone()
    p["size"] = torch.tensor([[[1., 1., 1.], [1., 2., 1.]]])
    assert match_batch(p, b, alpha_position=.5).tolist() == [[0, 1]]
    b["size_axis_swap_allowed"] = torch.tensor([[False, False]])
    assert match_batch(p, b, alpha_position=.5).tolist() == [[0, 1]]
    b["size_axis_swap_allowed"] = torch.tensor([[True, False]])
    assert match_batch(p, b, alpha_position=.5).tolist() == [[1, 0]]


# --- K7: grid-residual position head -----------------------------------------

def test_grid_position_encoding_round_trip_and_border_clamp():
    xy = torch.rand(64, 2, dtype=torch.float64)
    cells, residual = encode_grid_position(xy, 16)
    logits = torch.nn.functional.one_hot(cells, 256).double()
    residuals = torch.zeros(64, 256, 2, dtype=torch.float64).scatter(1, cells[:, None, None].expand(-1, 1, 2), residual[:, None])
    assert torch.allclose(decode_grid_position(logits, residuals, 16), xy, atol=1e-12)
    assert residual.abs().max() <= 1
    cells, residual = encode_grid_position(torch.tensor([[1., -.1]]), 16)
    assert cells.tolist() == [15 * 16] and residual.tolist() == [[1., -1.]]
    with pytest.raises(ValueError):
        decode_grid_position(torch.zeros(2, 16), torch.zeros(2, 16, 2), 3)


def grid_config(**kwargs):
    return ModelConfig(backbone="tiny", decoder_dim=16, decoder_heads=2, decoder_layers=1, tiny_hidden_size=16,
                       max_objects=8, lora_rank=0, position_head="grid_residual", position_grid=4, **kwargs)


@pytest.mark.parametrize("kwargs", [{"position_head": "grid"}, {"position_grid": 0}, {"position_grid": True}])
def test_position_head_options_are_validated(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_grid_model_always_decodes_position_and_round_trips(tmp_path):
    torch.manual_seed(0)
    model = build_model(grid_config()).eval()
    b = collate_samples([batch_sample(3), floor_sample()], TinyTokenizer())
    with torch.no_grad():
        p = model(**model_inputs(b))
    assert p["position_cell_logits"].shape == (2, 3, 16) and p["position_cell_residuals"].shape == (2, 3, 16, 2)
    active = b["slot_mask"]
    decoded = decode_grid_position(p["position_cell_logits"], p["position_cell_residuals"], 4)
    assert torch.allclose(p["position_normalized"][..., :2][active], decoded[active])
    assert p["position_normalized"][1, 0, 2] == b["fixed_position_normalized"][1, 0, 2]  # declared floor z still fixed
    model.save_pretrained(tmp_path / "model")
    loaded = load_model(tmp_path / "model").eval()
    assert loaded.config.position_head == "grid_residual" and loaded.config.position_grid == 4
    with torch.no_grad():
        assert torch.equal(loaded(**model_inputs(b))["position_normalized"], p["position_normalized"])


def test_grid_loss_terms_supervise_gt_cell_only():
    torch.manual_seed(0)
    b = collate_samples([batch_sample(2), floor_sample()], TinyTokenizer())
    p = build_model(grid_config())(**model_inputs(b))
    p["position_cell_logits"].retain_grad()
    p["position_cell_residuals"].retain_grad()
    cfg = LossConfig(hungarian=False, position_cell=.5, position_residual=2.)
    result = GeometryCriterion(cfg)(p, b)
    sums, counts = result["term_sums"], result["term_counts"]
    # Three valid slots; the floor-declared chair has a fixed z, so only two z terms.
    assert (counts["position"], counts["position_cell"], counts["position_residual"], counts["position_z"]) == (3, 3, 3, 2)
    combined = (.5 * sums["position_cell"] + 2. * sums["position_residual"] + sums["position_z"]) / 3
    assert result["position"].item() == pytest.approx(combined.item(), rel=1e-5)
    regression = GeometryCriterion(cfg)({k: v for k, v in p.items() if not k.startswith("position_cell")}, b)
    assert sums["position"].item() == pytest.approx(regression["term_sums"]["position"].item(), rel=1e-6)  # decoded diagnostic
    result["loss"].backward()
    cells, _ = encode_grid_position(b["targets"]["position_normalized"][..., :2].nan_to_num(), 4)
    gradient = p["position_cell_residuals"].grad.abs().sum(-1)
    gt_cell = torch.nn.functional.one_hot(cells, 16).bool() & b["slot_mask"][..., None]
    assert gradient[gt_cell].gt(0).all() and gradient[~gt_cell].eq(0).all()
    assert p["position_cell_logits"].grad[b["slot_mask"]].abs().sum() > 0


def toy_row(x):
    return {"schema_version": "fastfill.v2", "condition": {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z_m": 0., "height_m": 3.},
        "objects": [{"id": "lamp_0", "category": "lamp", "description": "floor lamp"}], "constraints": []},
        "target": {"objects": [{"id": "lamp_0", "target_size_local_m": [.4, .4, 1.5], "bottom_center_m": [x, 2.2, 0.], "yaw_rad": 0.}]},
        "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [False]},
        "provenance": {"source": "offline_test", "house_id": "toy", "split": "train"}}


def fit_bimodal(head):
    # One condition, mirror-image layouts 50/50: normalized x 0.2 or 0.8 (y 0.55, z 0).
    torch.manual_seed(0)
    b = collate_samples([toy_row(.8), toy_row(3.2)], TinyTokenizer())
    model = build_model(ModelConfig(backbone="tiny", decoder_dim=16, decoder_heads=2, decoder_layers=1, tiny_hidden_size=16,
                                    max_objects=2, lora_rank=0, position_head=head, position_grid=8))
    # smooth_l1: the squared regime makes the regression optimum the conditional mean (plain L1's
    # optimum is the whole interval between the modes, so regression cannot pick a mode either).
    criterion = GeometryCriterion(LossConfig(hungarian=False, position_type="smooth_l1", size=0., yaw_cls=0., yaw_reg=0.))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(100):
        optimizer.zero_grad()
        criterion(model(**model_inputs(b)), b)["loss"].backward()
        optimizer.step()
    with torch.no_grad():
        return model.eval()(**model_inputs(b))["position_normalized"][:, 0, 0]


def test_grid_head_picks_a_mode_where_regression_averages_the_modes():
    regression, grid = fit_bimodal("regression"), fit_bimodal("grid_residual")
    assert (regression - .5).abs().max() < .05
    assert torch.allclose(grid[0], grid[1])  # identical inputs, one answer
    assert min(abs(float(grid[0]) - .2), abs(float(grid[0]) - .8)) < .02


# --- K8: predict binds max_length to the checkpoint ---------------------------

@pytest.mark.parametrize("head", ["regression", "grid_residual"])
def test_predict_reads_max_length_from_the_checkpoint(tmp_path, monkeypatch, head):
    model_dir = tmp_path / "model"
    build_model(grid_config() if head == "grid_residual" else
                ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16, decoder_heads=2, decoder_layers=1,
                            tiny_hidden_size=16)).save_pretrained(model_dir)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    condition = floor_sample()["condition"]
    request = tmp_path / "condition.json"
    request.write_text(json.dumps(condition))
    seen, real = [], predict.predict_layout
    monkeypatch.setattr(predict, "predict_layout", lambda *a, **k: seen.append(k["max_length"]) or real(*a, **k))
    run = lambda name, *extra: predict.main(["--checkpoint", str(model_dir), "--condition", str(request),
                                             "--output", str(tmp_path / name), *extra])
    run("bare.json")
    (model_dir / CHECKPOINT_MANIFEST).write_text(json.dumps({"max_length": 8192}))
    run("bound.json")
    run("explicit.json", "--max-length", "2048")
    assert seen == [4096, 8192, 2048]
    validate_layout(json.loads((tmp_path / "bound.json").read_text()), condition)


# --- K9: configs ----------------------------------------------------------------

def resolve(path):
    """train.run_training's strict config validation, without data or a backbone."""
    config = json.loads(path.read_text())
    assert not set(config) - set(train.CONFIG_SECTIONS)
    training = train._training_config(config.get("training", {}))
    return (config, ModelConfig(**config.get("model", {})), LossConfig(**config.get("loss", {})), training,
            train._optimizer_config(config, training), train._augmentation_config(config), train._validation_config(config))


TRAINING_CONFIGS = sorted(p for p in CONFIGS.glob("*.json") if p.name != "direct_request.json")
QWEN_CONFIGS = [p for p in TRAINING_CONFIGS if "qwen" in json.loads(p.read_text())["model"]["backbone"].lower()]
FORMAL = {"qwen3_8b_main_4gpu_regression.json": (4, "regression"), "qwen3_8b_main_3gpu_grid.json": (3, "grid_residual")}


@pytest.mark.parametrize("path", QWEN_CONFIGS, ids=[p.name for p in QWEN_CONFIGS])
def test_every_qwen_config_loads_with_8192_context(path):
    assert resolve(path)[3]["max_length"] == 8192


@pytest.mark.parametrize("name", sorted(FORMAL))
def test_formal_configs_share_global_batch_and_inherit_the_main_config(name):
    world, head = FORMAL[name]
    config, model, loss, training, optimizer, augmentation, _ = resolve(CONFIGS / name)
    assert model.position_head == head and model.position_grid == 16
    assert world * training["batch_size"] * training["gradient_accumulation_steps"] == 96
    assert training["steps"] == math.ceil(3 * 124375 / 96) == 3887
    assert training["validate_every"] == training["checkpoint_every"] == 500
    assert optimizer["warmup_steps"] == round(.03 * training["steps"])
    assert loss.yaw_reg <= 2. and augmentation["minimal_form_p"] == .5
    main = json.loads((CONFIGS / "qwen3_8b_main_world7.json").read_text())
    differs = {("model", "position_head"), ("training", "steps"), ("training", "gradient_accumulation_steps"),
               ("training", "checkpoint_every"), ("training", "validate_every"), ("optimizer", "warmup_steps")}
    assert set(config) == set(main)
    for section in main:
        assert set(config[section]) == set(main[section])
        assert {key for key in main[section] if config[section][key] != main[section][key]} <= {k for s, k in differs if s == section}


def test_main_and_pilot_configs_cap_yaw_reg_and_stay_in_sync():
    main = resolve(CONFIGS / "qwen3_8b_main_world7.json")
    pilot = resolve(CONFIGS / "qwen3_8b_pilot_1gpu.json")
    assert main[2].yaw_reg <= 2.
    for index in (1, 2, 5, 6):  # model, loss, augmentation, validation
        assert pilot[index] == main[index]
    assert pilot[3]["max_length"] == main[3]["max_length"] == 8192
