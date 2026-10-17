"""Round 3: resume binds the world size and validation data hash (--allow-resume-change accepts and records a change)."""
import json

import accelerate
import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.io import fingerprint
from fastfill.v2.tests.test_active_objective_training import _data
from fastfill.v2.tests.test_train_loop_fix20261006 import _Killed, _config, _killing_accelerator, _rows


def _validation(path, scene_id):
    row = _rows(1)[0]
    path.write_text(json.dumps({**row, "provenance": {**row["provenance"], "split": "validation", "house_id": "h2",
                                                      "scene_id": scene_id}}) + "\n")
    return path


def _killed_state(tmp_path, monkeypatch, config, data, validation=None):
    original = accelerate.Accelerator
    monkeypatch.setattr(accelerate, "Accelerator", _killing_accelerator(1))
    with pytest.raises(_Killed):
        train.run_training(config, data, tmp_path / "killed", validation=validation)
    monkeypatch.setattr(accelerate, "Accelerator", original)
    return tmp_path / "killed/state-step-1"


def _edit_saved(state, drop=(), **changes):
    path = state / "custom_checkpoint_0.pkl"
    values = {key: value for key, value in torch.load(path, weights_only=False).items() if key not in drop}
    torch.save({**values, **changes}, path)


def _resume(config, state):
    return {**config, "training": {**config["training"], "resume": str(state)}}


def _manifests(run):
    return [json.loads((run / name).read_text()) for name in ("run_manifest_start.json", "run_manifest.json")]


def test_world_size_change_is_refused_unless_allowed_and_then_recorded(tmp_path, monkeypatch):
    data = _data(tmp_path, _rows(2))
    config = _config(steps=2, checkpoint_every=1)
    state = _killed_state(tmp_path, monkeypatch, config, data)
    saved = torch.load(state / "custom_checkpoint_0.pkl", weights_only=False)
    assert saved["world_size"] == 1 and saved["validation_data_sha256"] is None
    _edit_saved(state, world_size=2)  # as if saved by a two-rank run
    with pytest.raises(ValueError, match="world_size.*--allow-resume-change"):
        train.run_training(_resume(config, state), data, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    logs = train.run_training(_resume(config, state), data, tmp_path / "accepted", allow_resume_change=True)
    assert [r["step"] for r in logs] == [1, 2]
    for manifest in _manifests(tmp_path / "accepted"):
        assert manifest["resumed_with_changes"] == {"world_size": {"saved": 2, "current": 1}}
    # The new run's states bind its own world size, so resuming them needs no flag.
    resumed = torch.load(tmp_path / "accepted/state-step-2/custom_checkpoint_0.pkl", weights_only=False)
    assert resumed["world_size"] == 1


def test_validation_change_is_refused_unless_allowed_and_then_recorded(tmp_path, monkeypatch):
    data = _data(tmp_path, _rows(2))
    first, second = _validation(tmp_path / "a.jsonl", "v0"), _validation(tmp_path / "b.jsonl", "v1")
    config = _config(steps=2, checkpoint_every=1)
    state = _killed_state(tmp_path, monkeypatch, config, data, validation=first)
    for validation, name in ((second, "other"), (None, "dropped")):
        with pytest.raises(ValueError, match="validation_data_sha256"):
            train.run_training(_resume(config, state), data, tmp_path / name, validation=validation)
    train.run_training(_resume(config, state), data, tmp_path / "same", validation=first)
    assert _manifests(tmp_path / "same")[1]["resumed_with_changes"] == {}
    train.run_training(_resume(config, state), data, tmp_path / "accepted", validation=second, allow_resume_change=True)
    change = {"validation_data_sha256": {"saved": fingerprint(first), "current": fingerprint(second)}}
    assert all(manifest["resumed_with_changes"] == change for manifest in _manifests(tmp_path / "accepted"))
    # The restored step-1 score was taken on a.jsonl (lower than step 2 on b here); selection ignores it.
    assert _manifests(tmp_path / "accepted")[1]["selection_metric"]["best"]["step"] == 2
    assert _manifests(tmp_path / "same")[1]["selection_metric"]["best"]["step"] == 1


def test_state_without_bound_fields_resumes_with_a_warning(tmp_path, monkeypatch, capsys):
    data = _data(tmp_path, _rows(2))
    config = _config(steps=2, checkpoint_every=1)
    state = _killed_state(tmp_path, monkeypatch, config, data)
    _edit_saved(state, drop=("world_size", "validation_data_sha256"))  # a state saved before this check
    capsys.readouterr()
    assert [r["step"] for r in train.run_training(_resume(config, state), data, tmp_path / "old")] == [1, 2]
    assert "predates the resume check of ['world_size', 'validation_data_sha256']" in capsys.readouterr().err
    for manifest in _manifests(tmp_path / "old"):  # unchecked, not "checked and unchanged"
        assert manifest["resumed_with_changes"] == {}
        assert manifest["resume_unverified"] == ["world_size", "validation_data_sha256"]
    assert _manifests(tmp_path / "old")[1]["selection_metric"]["best_after_step"] == 0  # no selection_after_step saved


def test_a_validation_change_keeps_old_scores_out_of_selection_across_later_plain_resumes(tmp_path, monkeypatch):
    data = _data(tmp_path, _rows(2))
    first, second = _validation(tmp_path / "a.jsonl", "v0"), _validation(tmp_path / "b.jsonl", "v1")
    config = _config(steps=3, checkpoint_every=1)
    state = _killed_state(tmp_path, monkeypatch, config, data, validation=first)  # step 1 scored on a.jsonl
    original = accelerate.Accelerator
    monkeypatch.setattr(accelerate, "Accelerator", _killing_accelerator(2))
    with pytest.raises(_Killed):  # accepted change to b.jsonl, killed after step 2
        train.run_training(_resume(config, state), data, tmp_path / "changed", validation=second, allow_resume_change=True)
    monkeypatch.setattr(accelerate, "Accelerator", original)
    assert torch.load(tmp_path / "changed/state-step-2/custom_checkpoint_0.pkl", weights_only=False)["selection_after_step"] == 1
    logs = train.run_training(_resume(config, tmp_path / "changed/state-step-2"), data, tmp_path / "plain", validation=second)
    assert [r["step"] for r in logs if r.get("validation")] == [1, 2, 3]
    start, final = _manifests(tmp_path / "plain")
    assert start["resumed_with_changes"] == final["resumed_with_changes"] == {} and final["resume_unverified"] == []
    selection = final["selection_metric"]
    assert selection["best_after_step"] == 1 and selection["best"]["step"] in (2, 3)
    assert selection["best"]["value"] == min(r["validation"]["selection_metric"] for r in logs if r["step"] > 1)


def test_cli_passes_allow_resume_change(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(train, "run_training", lambda *args, **kwargs: seen.append(kwargs["allow_resume_change"]))
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    argv = ["--config", str(tmp_path / "config.json"), "--data", "d", "--output", "o", "--resume", "s"]
    train.main(argv)
    train.main([*argv, "--allow-resume-change"])
    assert seen == [False, True]
