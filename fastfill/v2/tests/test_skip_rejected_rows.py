"""training.skip_rejected_rows: false stops the run on a row the preflight would leave out; the default keeps skipping."""
import pytest

from fastfill.v2 import train
from fastfill.v2.tests.test_active_objective_training import _data, _row, _yaw_config


def _strict():
    cfg = _yaw_config()
    return {**cfg, "training": {**cfg["training"], "skip_rejected_rows": False}}


def test_strict_run_refuses_a_row_it_would_skip_before_building_the_model(tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(train, "build_model", lambda cfg: builds.append(cfg))
    with pytest.raises(ValueError, match=r"skip_rejected_rows is false.*1 training and 0 validation rows.*no_active_objective"):
        train.run_training(_strict(), _data(tmp_path, [_row("empty", False), _row("valid")]), tmp_path / "run")
    assert not builds and not (tmp_path / "run").exists()


def test_strict_run_trains_when_every_row_is_kept(tmp_path):
    train.run_training(_strict(), _data(tmp_path, [_row("valid")]), tmp_path / "run")
    assert (tmp_path / "run/run_manifest.json").is_file()


def test_the_default_is_left_out_of_the_resume_digest():
    view = train._digest_view({"training": {"skip_rejected_rows": True, "steps": 1}})
    assert view == {"training": {"steps": 1}}
    assert train._digest_view({"training": {"skip_rejected_rows": False}}) == {"training": {"skip_rejected_rows": False}}
