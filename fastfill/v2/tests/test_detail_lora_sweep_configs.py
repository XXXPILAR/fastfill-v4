"""The v4detail LoRA rank/alpha arms: each is the every-room LoRA arm (r8, alpha 16) with only lora_rank and lora_alpha
changed, and resolves to that LoRA."""
import json
from pathlib import Path

import pytest

from fastfill.v2.model import ModelConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SWEEP = {"r16a32": (16, 32), "r64a128": (64, 128), "r8a8": (8, 8), "r8a32": (8, 32)}


@pytest.mark.parametrize("tag", ["1.7b", "4b"])
@pytest.mark.parametrize("arm", sorted(SWEEP))
def test_sweep_arm_changes_only_rank_and_alpha(tag, arm):
    base = json.loads((CONFIGS / f"main7-cell05-detail-{tag}-all.json").read_text())
    config = json.loads((CONFIGS / f"main7-cell05-detail-{tag}-all-{arm}.json").read_text())
    rank, alpha = SWEEP[arm]
    assert (base["model"]["lora_rank"], base["model"]["lora_alpha"]) == (8, 16)
    assert (config["model"]["lora_rank"], config["model"]["lora_alpha"]) == (rank, alpha)
    for key in ("lora_rank", "lora_alpha"):
        del base["model"][key], config["model"][key]
    assert config == base
    model = ModelConfig(**json.loads((CONFIGS / f"main7-cell05-detail-{tag}-all-{arm}.json").read_text())["model"])
    assert (model.lora_rank, model.lora_alpha, model.train_backbone) == (rank, alpha, False)
