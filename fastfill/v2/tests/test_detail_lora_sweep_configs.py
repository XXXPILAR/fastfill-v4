"""The v4detail every-room configurations: every room and every description in training; the LoRA rank/alpha arms are
the every-room LoRA arm (r8, alpha 16) with only lora_rank and lora_alpha changed."""
import json
from pathlib import Path

import pytest
import torch

from fastfill.v2.batch import augment_sample
from fastfill.v2.model import ModelConfig
from fastfill.v2.tests.test_execution import sample

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


EVERY_ROOM = sorted(CONFIGS.glob("main7-cell05-detail-*-all*.json"))


@pytest.mark.parametrize("path", EVERY_ROOM, ids=[p.name for p in EVERY_ROOM])
def test_every_room_configs_keep_every_room_and_every_description(path):
    """Real requests always carry descriptions: no training row has them replaced by the category name."""
    config = json.loads(path.read_text())
    assert config["augmentation"]["category_only_description_p"] == 0
    assert (config["training"]["max_length"], config["model"]["max_objects"], config["training"]["skip_rejected_rows"]) == (32768, 512, False)


def test_augmentation_of_the_every_room_configs_never_replaces_a_description():
    augmentation = json.loads((CONFIGS / "main7-cell05-detail-1.7b-all-full.json").read_text())["augmentation"]
    row = sample()
    row["condition"]["objects"][0]["description"] = "a tall oak chair with a curved back and woven seat"
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):  # minimal form (p 0.5) keeps id/category/description
        augmented = augment_sample(row, augmentation, generator)
        assert [o["description"] for o in augmented["condition"]["objects"]] == ["a tall oak chair with a curved back and woven seat"]
