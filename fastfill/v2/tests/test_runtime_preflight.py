"""Configuration mistakes must fail before reading requests or model weights."""
from unittest.mock import patch

import pytest

from fastfill.v2.evaluate import run_evaluation


def test_unknown_validation_level_fails_evaluation_preflight(tmp_path):
    output = tmp_path / "evaluation"
    with patch("fastfill.v2.evaluate.read_samples", side_effect=AssertionError("data must not be read")), \
            pytest.raises(ValueError, match="required_levels"):
        run_evaluation("unused-data.jsonl", output, predictions="unused-predictions.jsonl",
                       required_levels=("bbox", "mesh_typo"))
    assert not output.exists()
