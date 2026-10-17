"""Condition span pooling must retain means under real mixed-precision projection."""
import pytest
import torch

from fastfill.v2.model import ModelConfig, build_model


@pytest.mark.parametrize("dtype,span,constant", [
    (torch.bfloat16, [2000, 2020], 1.),
    (torch.float16, [3000, 3020], 32.),
])
def test_autocast_condition_span_pooling_preserves_means_and_gradients(dtype, span, constant):
    model = build_model(ModelConfig(backbone="tiny", decoder_dim=16, decoder_heads=2,
        decoder_layers=1, tiny_hidden_size=16, lora_rank=0))
    with torch.no_grad():
        model.memory_projection.weight.copy_(torch.eye(16))
        model.memory_projection.bias.zero_()
        model.slot_seed.weight.zero_()
    memory = torch.full((1, 4096, 16), constant, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        projected = model.memory_projection(memory)
        assert projected.dtype == dtype
        slots = model._slots(projected, torch.tensor([[span]]), torch.tensor([[True]]))
    assert torch.equal(slots, torch.full_like(slots, constant))
    slots.float().sum().backward()
    expected = torch.zeros_like(memory)
    expected[:, span[0]:span[1]] = 1 / (span[1] - span[0])
    assert torch.allclose(memory.grad, expected, rtol=.005, atol=1e-7)


def test_pooling_retains_float64_input_precision():
    model = build_model(ModelConfig(backbone="tiny", decoder_dim=16, decoder_heads=2,
        decoder_layers=1, tiny_hidden_size=16, lora_rank=0))
    with torch.no_grad():
        model.slot_seed.weight.zero_()
    value = 1. + 1e-10
    memory = torch.full((1, 4096, 16), value, dtype=torch.double)
    slots = model._slots(memory, torch.tensor([[[2000, 2020]]]), torch.tensor([[True]]))
    assert slots.dtype == torch.double
    assert torch.allclose(slots, torch.full_like(slots, value), rtol=0, atol=1e-11)
