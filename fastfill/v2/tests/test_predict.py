import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from fastfill.v2.batch import TinyTokenizer
from fastfill.v2.model import ModelConfig, build_model
from fastfill.v2.predict import main
from fastfill.v2.schema import validate_layout
from fastfill.v2.tests.test_execution import sample


def test_condition_only_prediction_cli_no_targets(tmp_path):
    model_dir = tmp_path / "model"
    model = build_model(ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16,
                        decoder_heads=2, decoder_layers=1, tiny_hidden_size=16))
    model.save_pretrained(model_dir)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    condition = sample()["condition"]
    request = tmp_path / "request.json"
    request.write_text(json.dumps(condition))
    output = tmp_path / "prediction.json"
    main(["--checkpoint", str(model_dir), "--condition", str(request), "--output", str(output)])
    validate_layout(json.loads(output.read_text()), condition)
    with pytest.raises(FileExistsError):
        main(["--checkpoint", str(model_dir), "--condition", str(request), "--output", str(request)])
    with pytest.raises(ValueError):
        main(["--checkpoint", str(model_dir), "--condition", str(request), "--output",
              "/Volumes/harddisk/3D_Room_Collections/generated.json"])


@pytest.mark.parametrize("catalog_kind,expected_exit", [(None, 0), ("empty", 2), ("valid", 0)])
def test_actual_cli_exit_distinguishes_prediction_and_runtime_failure(tmp_path, catalog_kind, expected_exit):
    model_dir = tmp_path / "model"
    model = build_model(ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16,
                        decoder_heads=2, decoder_layers=1, tiny_hidden_size=16))
    with torch.no_grad():
        for head in (model.position_head, model.size_head, model.yaw_logits_head, model.yaw_residual_head):
            head.weight.zero_()
            head.bias.zero_()
        model.position_head.bias.copy_(torch.tensor([.4, .25, 0.]))
    model.save_pretrained(model_dir)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    condition = sample()["condition"]
    request, output = tmp_path / "request.json", tmp_path / "result.json"
    request.write_text(json.dumps(condition))
    args = [sys.executable, "-m", "fastfill.v2.predict", "--checkpoint", str(model_dir),
            "--condition", str(request), "--output", str(output)]
    if catalog_kind:
        assets = [] if catalog_kind == "empty" else [{"ref": "fixture-chair", "category": "chair",
                                                       "actual_size_local_m": [1., 1., 1.]}]
        catalog = tmp_path / "catalog.json"
        catalog.write_text(json.dumps(assets))
        args += ["--catalog", str(catalog), "--commit-in-memory"]
    proc = subprocess.run(args, cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=30)
    result = json.loads(output.read_text())
    assert proc.returncode == expected_exit, proc.stderr
    if catalog_kind:
        assert result["ok"] is (expected_exit == 0)
        assert result["committed"] is (expected_exit == 0)
        summary = json.loads(proc.stdout.splitlines()[-1])
        assert summary["runtime_ok"] is result["ok"]
    else:
        validate_layout(result, condition)
