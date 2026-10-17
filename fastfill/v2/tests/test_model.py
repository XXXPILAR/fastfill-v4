import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.model import ModelConfig, build_model, load_model, model_inputs
from fastfill.v2.tests.test_batch import sample


def config():
    return ModelConfig(backbone="tiny", decoder_dim=32, decoder_heads=4,
                       decoder_layers=2, tiny_hidden_size=32, max_objects=16,
                       tiny_layers=1, dropout=0.0)


@pytest.mark.parametrize("kwargs", [
    {"decoder_dim": True}, {"decoder_layers": 1.5}, {"decoder_ffn_multiplier": 0},
    {"dropout": float("nan")}, {"lora_dropout": float("inf")}, {"lora_rank": -1},
    {"train_backbone": "false", "lora_rank": 0}, {"local_files_only": "false"},
    {"size_reference": (True, 1., 1.)}, {"size_reference": (1e38, 1., 1.)},
    {"size_reference": (1e-42, 1., 1.)}, {"backbone": ""}, {"lora_target_modules": [None]},
])
def test_model_config_rejects_ambiguous_types_and_nonfinite_geometry(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_positive_size_bounds_survive_fp16_autocast():
    cfg = ModelConfig(backbone="tiny", decoder_dim=32, decoder_heads=4, decoder_layers=1,
                      tiny_hidden_size=32, size_log_limit=30.)
    model = build_model(cfg)
    with torch.no_grad():
        model.size_head.weight.zero_()
        model.size_head.bias.copy_(torch.tensor([30., -30., 0.]))
    batch = collate_samples([sample(1)], TinyTokenizer())
    with torch.autocast("cpu", dtype=torch.float16):
        sizes = model(**model_inputs(batch))["size"]
    assert sizes.dtype == torch.float32
    assert torch.isfinite(sizes).all() and (sizes > 0).all()
    sizes.log().sum().backward()
    assert torch.isfinite(model.size_head.bias.grad).all()


def test_exact_slots_positive_size_and_gradients_to_condition_backbone():
    model = build_model(config())
    b = collate_samples([sample(2), sample(1)], TinyTokenizer())
    out = model(**model_inputs(b))
    assert out["position_normalized"].shape == (2, 2, 3)
    assert out["yaw_logits"].shape == (2, 2, 12)
    assert out["slot_mask"].tolist() == [[True, True], [True, False]]
    assert (out["size"][out["slot_mask"]] > 0).all()
    assert torch.equal(out["position_normalized"][1, 1], torch.zeros(3))
    loss = sum(out[k].square().sum() for k in ("position_normalized", "size", "yaw_logits", "yaw_residuals"))
    loss.backward()
    assert model.backbone.embedding.weight.grad.abs().sum() > 0
    assert model.position_head.weight.grad.abs().sum() > 0


def test_batch_padding_does_not_change_valid_object_predictions():
    torch.manual_seed(2)
    model = build_model(config()).eval()
    one = collate_samples([sample(1)], TinyTokenizer())
    padded = collate_samples([sample(1), sample(3)], TinyTokenizer())
    with torch.no_grad():
        a = model(**model_inputs(one))
        b = model(**model_inputs(padded))
    for key in ("position_normalized", "size", "yaw_logits", "yaw_residuals"):
        assert torch.allclose(a[key][0, 0], b[key][0, 0], atol=2e-6), key


def test_full_condition_after_object_changes_predictions():
    model = build_model(config()).eval()
    a = sample(1)
    b = sample(1)
    b["condition"]["constraints"] = [{"type": "faces_direction", "object_id": "chair_0", "direction_xy": [0, -1]}]
    with torch.no_grad():
        x = model(**model_inputs(collate_samples([a], TinyTokenizer())))
        y = model(**model_inputs(collate_samples([b], TinyTokenizer())))
    assert not torch.allclose(x["position_normalized"], y["position_normalized"])


def test_fixed_size_and_ground_height_override_head():
    s = sample(1)
    s["condition"]["objects"][0].update(fixed_size_local_m=[1.1, .7, 1.2], support_parent="floor")
    s["target"]["objects"][0]["target_size_local_m"] = [1.1, .7, 1.2]
    model = build_model(config())
    b = collate_samples([s], TinyTokenizer())
    out = model(**model_inputs(b))
    assert torch.equal(out["size"][0, 0], torch.tensor([1.1, .7, 1.2]))
    assert out["position_normalized"][0, 0, 2] == 0


def test_checkpoint_round_trip(tmp_path):
    model = build_model(config()).eval()
    b = collate_samples([sample(1)], TinyTokenizer())
    model.save_pretrained(tmp_path)
    loaded = load_model(tmp_path).eval()
    with torch.no_grad():
        assert torch.equal(model(**model_inputs(b))["size"], loaded(**model_inputs(b))["size"])


def test_empty_scene_and_mixed_empty_batch_do_not_create_nan():
    model = build_model(config())
    empty = collate_samples([sample(0)], TinyTokenizer())
    out = model(**model_inputs(empty))
    assert out["size"].shape == (1, 0, 3)
    mixed = collate_samples([sample(0), sample(2)], TinyTokenizer())
    out = model(**model_inputs(mixed))
    assert torch.isfinite(out["size"]).all()
    assert torch.equal(out["size"][0], torch.zeros(2, 3))


def test_bidirectional_object_interaction_changes_first_slot():
    torch.manual_seed(7)
    model = build_model(config()).eval()
    a = sample(2)
    b = sample(2)
    b["condition"]["objects"][1]["description"] = "large reading chair with an attached lamp"
    with torch.no_grad():
        x = model(**model_inputs(collate_samples([a], TinyTokenizer())))
        y = model(**model_inputs(collate_samples([b], TinyTokenizer())))
    assert not torch.allclose(x["size"][0, 0], y["size"][0, 0])


@pytest.mark.parametrize("training", ["lora", "full", "frozen"])
def test_local_random_qwen_api_gradients_and_checkpoint(tmp_path, training):
    transformers = pytest.importorskip("transformers")
    if training == "lora":
        pytest.importorskip("peft")
    source = tmp_path / "qwen"
    transformers.Qwen2Model(transformers.Qwen2Config(vocab_size=258, hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=4096)).save_pretrained(source)
    cfg = ModelConfig(backbone=str(source), decoder_dim=32, decoder_heads=4, decoder_layers=1,
        lora_rank=2 if training == "lora" else 0, lora_alpha=4,
        train_backbone=training == "full", local_files_only=True)
    model = build_model(cfg)
    b = collate_samples([sample(1)], TinyTokenizer())
    model(**model_inputs(b))["size"].sum().backward()
    gradients = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert bool(gradients) == (training != "frozen")
    if gradients:
        assert sum(float(grad.abs().sum()) for grad in gradients) > 0
    checkpoint = tmp_path / "checkpoint"
    model.eval().save_pretrained(checkpoint)
    loaded = load_model(checkpoint, local_files_only=True).eval()
    with torch.no_grad():
        assert torch.equal(model(**model_inputs(b))["size"], loaded(**model_inputs(b))["size"])
