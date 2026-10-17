"""Orchestrator follow-ups to round 2: optimizer-state retention, read-only downstream import, swap categories."""
from fastfill.v2 import roomgenbench, train
from fastfill.v2.legacy_bridge import size_axis_swap_allowed
from fastfill.v2.tests.test_active_objective_training import NO_AUGMENTATION, _data
from fastfill.v2.tests.test_train_loop_fix20261006 import _config, _rows


def test_only_the_newest_optimizer_states_are_kept_but_every_model_export_stays(tmp_path):
    config = {**_config(steps=4, checkpoint_every=1, validate_every=0), "augmentation": NO_AUGMENTATION}
    train.run_training(config, _data(tmp_path, _rows(2)), tmp_path / "run")
    assert sorted(p.name for p in (tmp_path / "run").glob("state-step-*")) == ["state-step-3", "state-step-4"]
    assert sorted(p.name for p in (tmp_path / "run").glob("model-step-*")) == [f"model-step-{i}" for i in range(1, 5)]


def test_loading_the_downstream_assembler_writes_no_bytecode_into_its_checkout(tmp_path):
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "assemble.py").write_text("VALUE = 1\n")
    module, _ = roomgenbench._reference(tmp_path)
    assert module.VALUE == 1 and not list(tmp_path.rglob("__pycache__"))


def test_swap_categories_cover_hssd_seat_template_and_only_true_multiscan_beds():
    assert size_axis_swap_allowed("HSSD200", "seat") and size_axis_swap_allowed("HSSD200", "semi chair")
    assert not size_axis_swap_allowed("HSSD200", "table")
    assert size_axis_swap_allowed("MultiScan", "bed") and not size_axis_swap_allowed("MultiScan", "bed net")
