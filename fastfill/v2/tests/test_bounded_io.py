import json
import pytest

from fastfill.v2.io import read_samples
from fastfill.v2.tests.test_execution import sample


def test_bounded_read_does_not_parse_unrequested_tail(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(sample()) + "\nnot-json-outside-selected-prefix\n")
    rows = read_samples(path, training=True, max_samples=1)
    assert len(rows) == 1
    with pytest.raises(json.JSONDecodeError):
        read_samples(path)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_bounded_read_requires_positive_integer(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        read_samples(tmp_path / "absent", max_samples=limit)
