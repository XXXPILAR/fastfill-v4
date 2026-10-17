"""ops/isambard_inputs.py: arm refusal and choice, the baseline's backbone-path copy, the rebuilt old cohorts."""
import importlib.util
import json
from pathlib import Path
import random

import pytest

from fastfill.v2.autorun import sha256

_spec = importlib.util.spec_from_file_location("isambard_inputs", Path(__file__).parents[1] / "ops/isambard_inputs.py")
I = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(I)


def make_arm(runs, name, score, phase="done", cohort_text="a\nb\n", code="c0de"):
    d = runs / name
    run = runs / f"run-{name}"
    (run / "model-step-4").mkdir(parents=True)
    d.mkdir(parents=True)
    cohort = d / "validation-0123456789ab-all.jsonl"
    cohort.write_text(cohort_text)
    report = runs / f"select2-{name}" / "report.json"
    report.parent.mkdir()
    report.write_text(json.dumps({"data_path": str(cohort.resolve()), "data_sha256": sha256(cohort), "implementation_sha256": code}))
    best = {"checkpoint": str(run / "model-step-4"), "report": str(report), "score": score, "implementation_sha256": code}
    config = d / "config.json"
    config.write_text(json.dumps({"loss": {"yaw_cls": .5}, "model": {"max_objects": 256},
                                  "training": {"steps": 10, "gradient_accumulation_steps": 24, "batch_size": 1}}))
    (run / "run_manifest_start.json").write_text(json.dumps({"world_size": 4, "supervised_samples": 960, "rejected_samples": 2}))
    (d / "select2-selection.json").write_text(json.dumps([best, {**best, "score": score + 1}]))
    (d / "STATUS.json").write_text(json.dumps({"phase": phase, "best": best, "run": f"run-{name}", "config": str(config)}))


def test_arms_refused_and_chosen(tmp_path):
    make_arm(tmp_path, "a", 2.5)
    make_arm(tmp_path, "b", 2.4)
    a, b = I.arm(tmp_path, "a"), I.arm(tmp_path, "b")
    assert a["global_batch"] == 96 and a["epochs_supervised"] == 1. and (a["best_step"], a["best_epochs"]) == (4, .4)
    final, control = I.choose([a, b])
    assert (final["dir"], control["dir"]) == ("b", "a")
    with pytest.raises(SystemExit, match="same select2 score"):
        I.choose([a, {**b, "best_score": a["best_score"]}])
    make_arm(tmp_path, "c", 2.0, cohort_text="b\na\n")
    with pytest.raises(SystemExit, match="cohorts differ"):
        I.choose([a, I.arm(tmp_path, "c")])
    make_arm(tmp_path, "e", 2.0, code="other")
    with pytest.raises(SystemExit, match="different fastfill/v2 code"):
        I.choose([a, I.arm(tmp_path, "e")])
    make_arm(tmp_path, "d", 1.0, phase="E-train")
    with pytest.raises(SystemExit, match="not done"):
        I.arm(tmp_path, "d")


def test_baseline_copy_rewrites_only_the_copy(tmp_path):
    backbone = tmp_path / "models/Qwen3-8B"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text('{"model_type": "qwen3"}')
    source = tmp_path / "download" / I.BASELINE
    (source / I.STEP).mkdir(parents=True)
    original = json.dumps({"backbone": I.OLD_BACKBONE, "max_objects": 128})
    (source / I.STEP / "model_config.json").write_text(original)
    (source / I.STEP / "checkpoint_manifest.json").write_text(json.dumps({"backbone": {"config_sha256": sha256(backbone / "config.json")}}))
    (source / "run_manifest.json").write_text("{}")
    target = tmp_path / "copy" / I.BASELINE
    target.parent.mkdir()
    record = I.baseline_copy(source, target, backbone)
    assert json.loads((target / I.STEP / "model_config.json").read_text()) == {"backbone": str(backbone), "max_objects": 128}
    assert (source / I.STEP / "model_config.json").read_text() == original  # the download is never edited
    assert (target / "run_manifest.json").read_text() == "{}"
    assert json.loads((target / "BACKBONE_REWRITE.json").read_text()) == record
    assert record["backbone_before"] == I.OLD_BACKBONE and record["backbone_config_sha256"] == sha256(backbone / "config.json")
    (backbone / "config.json").write_text('{"model_type": "other"}')
    with pytest.raises(SystemExit, match="not the baseline's backbone"):
        I.baseline_copy(source, target, backbone)


def test_pins_dry_run_reports_except_always(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    dry = I.Pins(enforce=False)
    dry(f, "0" * 64)
    dry(tmp_path / "missing", "0" * 64)
    assert [m["sha256"] for m in dry.mismatches] == [sha256(f), None]
    with pytest.raises(SystemExit, match="not the pinned"):
        dry(f, "0" * 64, always=True)
    with pytest.raises(SystemExit, match="not the pinned"):
        I.Pins(enforce=True)(f, "0" * 64)
    assert I.Pins(enforce=True)(f, sha256(f)) == sha256(f)


def test_old_cohorts_are_autopilot_eval_data(tmp_path):
    old = tmp_path / "data/old"
    old.mkdir(parents=True)
    lines = [json.dumps({"i": i}) + "\n" for i in range(3500)]
    (old / "validation.jsonl").write_text("".join(lines))
    every, first = I.cohorts(tmp_path, tmp_path / "runs/analysis", old)
    random.Random(0).shuffle(lines)
    assert every.read_text() == "".join(lines) and first.read_text() == "".join(lines[:3000])
    assert every.name == f"validation-{sha256(old / 'validation.jsonl')[:12]}-all.jsonl"
