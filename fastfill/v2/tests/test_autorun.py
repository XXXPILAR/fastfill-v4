"""Autopilot decisions: selection score, the rescaled next-run configuration and the final acceptance."""
import json
from pathlib import Path

import pytest

from fastfill.v2.autorun import next_config, score


def _baselines(room_centre=2.5):
    return {"category_median_size": {"log_size_error": {"mean": .4}}, "uniform_yaw": {"yaw_error_rad": {"mean": .7}},
            "room_center_position": {"bottom_center_error_m": {"mean": room_centre}}}


def _report(size, yaw, pos, central=.17, wall=.66, overlap=.01, out=0.):
    collapse = lambda c, w, o, r: {"central_quarter_fraction": c, "mean_nearest_wall_distance_m": w,
                                   "bev_overlap_rate_iou_gt_0.3": o, "out_of_room_fraction": r}
    return {"model": {"reference": {"log_size_error": {"mean": size}, "yaw_error_rad": {"mean": yaw},
                                    "bottom_center_error_m": {"mean": pos}}},
            "baselines": _baselines(), "baselines_paired": _baselines(),
            "collapse": {"predicted_matched": collapse(central, wall, overlap, out), "ground_truth": collapse(.17, .66, .01, 0.)}}


def test_score_divides_by_the_baselines_of_the_same_requests():
    # before: the all-row baselines, whose room-centre error includes requests the model never answered
    report = {**_report(.3, .6, 2.), "baselines": _baselines(room_centre=4.), "baselines_paired": _baselines(room_centre=2.)}
    assert score(report) == pytest.approx(score(_report(.3, .6, 2.)) + 1. - .8)  # 2 / 2 instead of 2 / 2.5
    with pytest.raises(KeyError):  # a report without the paired baselines is not scorable (evaluated again)
        score({key: value for key, value in report.items() if key != "baselines_paired"})


def test_summary_shows_validator_collisions_clean_rooms_and_their_ground_truth(tmp_path):
    import argparse
    from fastfill.v2.autorun import Autopilot
    pilot = Autopilot(argparse.Namespace(repo=str(tmp_path), model_repo="mock/model"))
    checked = lambda rooms, pairs, clean, unknown: {"rooms": 10, "rooms_with_collision": rooms, "collision_pairs": pairs,
                                                    "fixed_collision_pairs": 1, "clean_rooms": clean, "clean_room_rate": clean / 10,
                                                    "clean_rooms_with_hard_unknown": unknown}
    report = {**_report(.3, .6, 2.), "requests": 10, "inference_failed_requests": 0, "hard_violation_requests": 6,
              "failed_requests_strict": 9, "validation": {"model": checked(5, 12, 4, 3), "ground_truth": checked(2, 3, 8, 6)}}
    report["collapse"]["predicted_matched"]["bev_overlap_rate_iou_gt_0.3_room_mean"] = .05
    report["baselines_paired"]["room_center_position"]["bottom_center_error_m"]["mean"] = 1.25
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "report.json").write_text(json.dumps(report))
    pilot.summary({"checkpoint": "c", "score": 1., "report": str(tmp_path / "test" / "report.json")}, tmp_path / "test", None)
    text = (pilot.dir / "SUMMARY.md").read_text()
    # before: only the IoU > 0.3 overlap rate (0.010), which hid these collisions
    assert "| 0.010 / 0.050 | 5/10 rooms, 12 pairs (+1 fixed) | 0.400 (3 with unknown) | 6 / 9 |" in text
    assert "2/10 rooms, 3 pairs (+1 fixed); clean 0.800 (6 with unknown)" in text  # the labels under the same validator
    assert "| 1.250 / 0.400 / 0.700 |" in text  # the paired baselines
    table = [line for line in text.splitlines() if line.startswith("|")]
    # before: a 16-cell delimiter row under 15 header cells, which GFM does not render as a table
    assert len(table) == 4 and {len(line.strip("|").split("|")) for line in table} == {15}


def test_a_cached_report_without_paired_baselines_is_evaluated_again(tmp_path, monkeypatch):
    from fastfill.v2 import autorun
    pilot, data, runs, calls, report = _candidates(tmp_path, monkeypatch, alive=False)
    report(pilot.runs / "select-a-model-step-400", tmp_path / "a/model-step-400", projection="minimal", grid_decode="spread",
           implementation_sha256=autorun.implementation_sha256(tmp_path), baselines_paired=None)
    pilot.evaluate_candidates(runs[:1], "select", newest=1, data=data)
    assert len(calls) == len(list(pilot.runs.glob("select-a-model-step-400.stale-*"))) == 1


def test_score_prefers_accurate_layouts_and_punishes_collapse_or_leaving_the_room():
    good, worse = score(_report(.3, .6, 2.0)), score(_report(.35, .6, 2.0))
    assert good < worse
    assert score(_report(.3, .6, 2.0, central=.9, wall=2.0)) > good + .7          # everything in the centre
    assert score(_report(.3, .6, 2.0, overlap=.2)) > good + .85                    # stacked
    assert score(_report(.3, .6, 2.0, out=.5)) > good + 2                          # pushed out of the room


def test_next_config_keeps_the_model_and_loss_and_rescales_to_seven_or_four_gpus():
    config = {"model": {"position_head": "grid_residual"}, "loss": {"position_cell": .5},
              "training": {"steps": 3887, "batch_size": 1, "gradient_accumulation_steps": 32, "resume": "x",
                           "checkpoint_every": 500, "validate_every": 500},
              "optimizer": {"warmup_steps": 117}}
    out = next_config(config, world_size=7, train_rows=124584, epochs=5)
    assert out["model"] == config["model"] and out["loss"] == config["loss"]
    t = out["training"]
    assert t["gradient_accumulation_steps"] == 14 and t["batch_size"] == 1 and t["resume"] is None
    assert t["steps"] == -(-5 * 124584 // (7 * 14)) and out["optimizer"]["warmup_steps"] == round(.03 * t["steps"])
    assert json.loads(json.dumps(config))["training"]["steps"] == 3887  # input untouched
    four = next_config(config, world_size=4, train_rows=124584, epochs=5)["training"]
    assert four["gradient_accumulation_steps"] == 24 and four["steps"] == 6489  # global batch 96


def test_adopt_restores_the_resume_chain(tmp_path):
    import argparse
    from fastfill.v2.autorun import Autopilot
    pilot = Autopilot(argparse.Namespace(repo=str(tmp_path)))
    base = tmp_path / "runs" / "job"
    for suffix in ("", "-resume1", "-resume2"):
        (tmp_path / "runs" / f"job{suffix}").mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / f"job{suffix}.log").write_text("")
    job = pilot.adopt({"outputs": [str(base)], "output": str(base)})
    assert job["resumes"] == 2 and job["output"].endswith("job-resume2") and len(job["outputs"]) == 3


def _phase_f(tmp_path, *, failed=0, over=0, upload_error=False, rows_late=False, **extra):
    """Autopilot.run end to end with stubbed processes: a finished prior run, data unchanged, phase F checks."""
    import argparse
    from unittest.mock import patch
    from fastfill.v2 import autorun

    class Proc:
        def __init__(self, code=0): self.code = code
        def wait(self): return self.code
        def poll(self): return self.code

    class Thread:  # the LLM thread: never finishes when rows arrive late, else nothing to do
        def __init__(self, *a, **k): pass
        def start(self): pass
        def join(self, timeout): pass

    old = tmp_path / "old"
    old.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (old / name).write_text("{}\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "NEXT_DATA.json").write_text(json.dumps({"name": "../old", "repo": "r", "revision": "v", "folder": "f", "commit": "c",
        "sha256": {n: autorun.sha256(old / n) for n in ("train.jsonl", "validation.jsonl", "test.jsonl")}}))
    a = argparse.Namespace(repo=str(tmp_path), current_data=str(old), data_root=str(tmp_path / "data"), data_wait_h=0., epochs=5,
                           eval_rows="3000", model_repo="mock/model", llm_env=str(tmp_path / "noenv"), llm_rows="300",
                           current=[("prior", "c", "1,2,3,4")], **extra)
    pilot = autorun.Autopilot(a)
    checkpoint = tmp_path / "runs/prior/model-step-10"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint.parent / "run_manifest_start.json").write_text(json.dumps({"config": {"training": {}, "optimizer": {}, "loss": {"position_cell": .5}}}))
    pilot.supervise = lambda *a: None
    pilot.evaluate_candidates = lambda *a, **k: {"checkpoint": str(checkpoint), "report": "x", "score": 1.}
    pilot.launch = lambda *a: Proc()
    if upload_error:
        def upload(*a): raise OSError("403 Forbidden")
        pilot.upload = upload
    else:
        pilot.upload = lambda *a: None
    commands, pilot.gpus_used = [], set()
    def sh(args, **kw):
        args = list(map(str, args))
        commands.append(args)
        pilot.gpus_used.add(kw.get("gpus"))
        if args[0] == "bash":
            for i in range(5):
                room = pilot.runs / "roomgenbench" / args[-1] / f"r{i}.layout_boxes"
                room.mkdir(parents=True, exist_ok=True)
                (room / "receipt.json").write_text("{}")
        if "fastfill.v2.evaluate" in args:
            if rows_late:  # the LLM rows land while the test set is evaluated
                rows = pilot.runs / "llm-prompt-300/rows.jsonl"
                rows.parent.mkdir(parents=True, exist_ok=True)
                rows.write_text("{}\n")
            out = Path(args[args.index("--output") + 1])
            out.mkdir(parents=True)  # evaluate refuses an existing output
            (out / "report.json").write_text(json.dumps({**_report(.3, .6, 2.), "requests": 10, "inference_failed_requests": failed,
                                                         "over_capacity_requests": over}))
        return Proc()
    pilot.sh = sh
    with patch.object(autorun.threading, "Thread", Thread):
        pilot.run()
    return pilot, commands



def test_phase_f_reports_over_capacity_requests_without_failing_on_them(tmp_path):
    pilot, commands = _phase_f(tmp_path, failed=5, over=5)
    assert pilot.status["phase"] == "done" and pilot.status["accepted"]
    decodes = {a[a.index("--grid-decode") + 1] for a in commands if "--grid-decode" in a}
    assert decodes == {"spread", "argmax"}  # the model alone is evaluated even without LLM rows
    assert "5 (5 over capacity)" in (pilot.dir / "SUMMARY.md").read_text()
    pilot, _ = _phase_f(tmp_path / "real", failed=6, over=5)
    assert pilot.status["failures"] == ["test: 1 of 10 requests without a layout"]


def test_phase_f_survives_an_upload_error_and_late_llm_rows(tmp_path):
    pilot, commands = _phase_f(tmp_path, upload_error=True, rows_late=True)
    assert pilot.status["phase"] == "done-with-failures" and pilot.status["failures"] == ["upload failed: OSError: 403 Forbidden"]
    assert "upload failed" in (pilot.dir / "SUMMARY.md").read_text()
    assert any("test-" in " ".join(a) for a in commands)  # the test set was still evaluated after the failed upload


def test_phase_f_can_run_again_after_a_failed_upload(tmp_path):
    _phase_f(tmp_path, upload_error=True)
    pilot, commands = _phase_f(tmp_path)  # e.g. after fixing the credentials
    assert pilot.status["phase"] == "done" and pilot.status["uploaded_model"] == "mock/model/main7-cell05-old-e5"
    assert not any(a[0] == "bash" for a in commands)  # the same checkpoint's five rooms are not exported twice


def test_a_start_config_skips_phases_a_to_d_and_uses_the_given_gpus(tmp_path):
    (tmp_path / "start.json").write_text(json.dumps({"training": {}, "optimizer": {}, "loss": {"position_cell": .5}}))
    pilot, commands = _phase_f(tmp_path, start_config=str(tmp_path / "start.json"), gpus="0,1,2,3", autorun_dir="autorun-a")
    assert pilot.status["phase"] == "done" and pilot.status["run"] == "main4-cell05-old-e5"
    assert not any(a[0] == "git" for a in commands)  # no runs to finish, no data to wait for
    assert pilot.gpus_used <= {None, "0", "1", "2", "3"} and (tmp_path / "runs/autorun-a/SUMMARY.md").is_file()


def test_no_gpu_evaluation_starts_beside_a_running_training(tmp_path, monkeypatch):
    import argparse
    import pytest
    from fastfill.v2 import autorun
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), current_data=str(tmp_path), eval_rows="3000"))
    (tmp_path / "run/model-step-5").mkdir(parents=True)
    monkeypatch.setattr(autorun, "any_training_alive", lambda: True)
    monkeypatch.setattr(pilot, "sh", lambda *a, **k: pytest.fail("a GPU job was started"))
    with pytest.raises(RuntimeError, match="while a fastfill.v2.train process runs"):
        pilot.evaluate_candidates([[str(tmp_path / "run")]], "select1", data=tmp_path / "sample.jsonl")


def _candidates(tmp_path, monkeypatch, *, alive, implementation=None):
    """An autopilot over runs a and b (steps 100-400, the newest three compete) whose evaluate writes a scorable
    spread/minimal report of ``implementation`` (default: the code on disk under its repo)."""
    import argparse
    from fastfill.v2 import autorun
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), eval_rows="3000"))
    data = tmp_path / "sample.jsonl"
    data.write_text("{}\n")
    for run in "ab":
        for step in (100, 200, 300, 400):
            (tmp_path / run / f"model-step-{step}").mkdir(parents=True)
    calls = []

    class Proc:
        def wait(self): return 0

    def sh(args, **kw):
        args = list(map(str, args))
        calls.append(args)
        out = Path(args[args.index("--output") + 1])
        report(out, args[args.index("--checkpoint") + 1], projection=args[args.index("--projection") + 1],
               grid_decode=args[args.index("--grid-decode") + 1], implementation_sha256=implementation or autorun.implementation_sha256(tmp_path))
        return Proc()

    def report(out, checkpoint, pos=2., **fields):
        out.mkdir(parents=True)
        (out / "report.json").write_text(json.dumps({**_report(.3, .6, pos), "requests": 10, "inference_failed_requests": 0,
                                                     "data_sha256": autorun.sha256(data), "checkpoint": str(Path(checkpoint).resolve()), **fields}))
    pilot.sh = sh
    monkeypatch.setattr(autorun, "any_training_alive", lambda: alive)
    runs = [[str(tmp_path / "a")], [str(tmp_path / "b")]]
    return pilot, data, runs, calls, report


def test_a_restart_beside_the_formal_training_reuses_the_six_select1_reports_without_evaluating(tmp_path, monkeypatch):
    """The live server's shape: six cached select1 reports of one older implementation, the data sample unchanged."""
    pilot, data, runs, calls, report = _candidates(tmp_path, monkeypatch, alive=True)  # any evaluation would raise
    old = "1c97878e" + "0" * 56
    for i, (run, step) in enumerate((r, s) for r in "ab" for s in (200, 300, 400)):
        report(pilot.runs / f"select1-{run}-model-step-{step}", tmp_path / run / f"model-step-{step}", pos=2. - i / 10,
               projection="minimal", grid_decode="spread", implementation_sha256=old)
    winner = pilot.evaluate_candidates(runs, "select1", data=data)
    assert calls == [] and winner["checkpoint"] == str(tmp_path / "b/model-step-400")
    selection = json.loads((pilot.dir / "select1-selection.json").read_text())
    assert len(selection) == 6 and {r["implementation_sha256"] for r in selection} == {old}
    assert not list(pilot.runs.glob("*.stale-*"))


def test_cached_reports_of_another_command_or_implementation_are_evaluated_again(tmp_path, monkeypatch):
    import pytest
    from fastfill.v2 import autorun
    # the audit's case: a cached argmax / full-projection report of old code is not what this method runs
    pilot, data, runs, calls, report = _candidates(tmp_path / "decode", monkeypatch, alive=False)
    report(pilot.runs / "select-a-model-step-400", tmp_path / "decode/a/model-step-400", projection="full", grid_decode="argmax",
           implementation_sha256="WRONG_OLD_CODE")
    winner = pilot.evaluate_candidates(runs[:1], "select", newest=1, data=data)
    assert len(calls) == 1 and calls[0][calls[0].index("--grid-decode") + 1] == "spread" and "minimal" in calls[0]
    assert json.loads(Path(winner["report"]).read_text())["grid_decode"] == "spread"
    assert len(list(pilot.runs.glob("select-a-model-step-400.stale-*"))) == 1
    # one report missing: fresh ones carry the code on disk, so a cached report of other code is evaluated again too
    pilot, data, runs, calls, report = _candidates(tmp_path / "mixed", monkeypatch, alive=False)
    report(pilot.runs / "select-a-model-step-400", tmp_path / "mixed/a/model-step-400", projection="minimal", grid_decode="spread",
           implementation_sha256="1c97878e")
    pilot.evaluate_candidates(runs[:1], "select", newest=2, data=data)
    assert len(calls) == 2 and {r["implementation_sha256"] for r in json.loads((pilot.dir / "select-selection.json").read_text())} \
        == {autorun.implementation_sha256(tmp_path / "mixed")}
    # cached reports that disagree: only those of other code than the one on disk are evaluated again
    pilot, data, runs, calls, report = _candidates(tmp_path / "split", monkeypatch, alive=False)
    for step, implementation in ((300, "1c97878e"), (400, autorun.implementation_sha256(tmp_path / "split"))):
        report(pilot.runs / f"select-a-model-step-{step}", tmp_path / f"split/a/model-step-{step}", projection="minimal",
               grid_decode="spread", implementation_sha256=implementation)
    pilot.evaluate_candidates(runs[:1], "select", newest=2, data=data)
    assert [c[c.index("--checkpoint") + 1] for c in calls] == [str(tmp_path / "split/a/model-step-300")]
    # the same beside a running training: refused before any GPU job
    pilot, data, runs, calls, report = _candidates(tmp_path / "alive", monkeypatch, alive=True)
    report(pilot.runs / "select-a-model-step-400", tmp_path / "alive/a/model-step-400", projection="minimal", grid_decode="spread",
           implementation_sha256="1c97878e")
    with pytest.raises(RuntimeError, match="while a fastfill.v2.train process runs"):
        pilot.evaluate_candidates(runs[:1], "select", newest=2, data=data)
    assert calls == []
    # evaluations that come back from other code than the one on disk never enter one selection
    pilot, data, runs, calls, report = _candidates(tmp_path / "moved", monkeypatch, alive=False, implementation="changed-meanwhile")
    with pytest.raises(RuntimeError, match="were not evaluated by implementation"):
        pilot.evaluate_candidates(runs[:1], "select", newest=1, data=data)


@pytest.mark.parametrize("field,value", [(None, None), ("projection", "full"), ("grid_decode", "argmax"),
                                         ("data_sha256", "0" * 64)])
def test_a_cached_report_is_evaluated_again_when_any_single_command_field_differs(tmp_path, monkeypatch, field, value):
    from fastfill.v2 import autorun
    pilot, data, runs, calls, report = _candidates(tmp_path, monkeypatch, alive=False)
    fields = {"projection": "minimal", "grid_decode": "spread", "implementation_sha256": autorun.implementation_sha256(tmp_path)}
    report(pilot.runs / "select-a-model-step-400", tmp_path / "a/model-step-400", **{**fields, **({field: value} if field else {})})
    pilot.evaluate_candidates(runs[:1], "select", newest=1, data=data)
    assert len(calls) == len(list(pilot.runs.glob("select-a-model-step-400.stale-*"))) == (field is not None)


def test_the_implementation_hash_is_the_one_evaluate_records():
    from fastfill.v2 import autorun, io
    repo = Path(autorun.__file__).resolve().parents[2]
    assert autorun.implementation_sha256(repo) == io.run_metadata(autorun.__file__)["implementation_sha256"]


def test_llm_baselines_rerun_unless_rows_code_parameters_and_predictions_match(tmp_path):
    """Real llm_baseline.run / run_evaluation outputs with a stubbed LLM; the API key never reaches the log."""
    import argparse
    import shutil
    from fastfill.v2 import autorun, llm_baseline
    from fastfill.v2.evaluate import project_minimal, run_evaluation
    from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows
    from fastfill.v2.tests.test_llm_baseline import _answer_from_targets
    (tmp_path / "fastfill/v2").mkdir(parents=True)
    for code in Path(llm_baseline.__file__).parent.glob("*.py"):  # the code the subprocesses would run
        shutil.copy(code, tmp_path / "fastfill/v2" / code.name)
    rows = [p for p in map(project_minimal, real_rows()) if p is not None]
    answers = {llm_baseline.prompt(r["condition"]): _answer_from_targets(r) for r in rows}
    sample, env = tmp_path / "sample.jsonl", tmp_path / "api.env"
    sample.write_text("".join(json.dumps(r) + "\n" for r in rows))
    env.write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=secret-test-key\nOPENAI_MODEL=m\n")
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path), llm_env=str(env), llm_rows="300"))
    calls = []

    def sh(args, **kw):
        args = list(map(str, args))
        get = lambda flag: args[args.index(flag) + 1]
        calls.append(args[2])
        if args[2] == "fastfill.v2.llm_baseline":
            llm_baseline.run(get("--data"), get("--output"), get("--env"), mode=get("--mode"), max_samples=int(get("--max-samples")),
                             ask=lambda env, model, messages: answers[messages[0]["content"]])
        else:
            run_evaluation(get("--data"), get("--output"), predictions=get("--predictions"))
    pilot.sh = sh
    pilot.llm_baselines(sample)
    assert calls == ["fastfill.v2.llm_baseline", "fastfill.v2.evaluate"] * 2 and len(pilot.llm_reports) == 2
    calls.clear()
    pilot.llm_baselines(sample)  # a restart with the same inputs
    assert calls == []
    for change in ("rows", "parameters", "predictions", "evaluator", "old summary", "code"):
        calls.clear()
        if change == "rows":
            sample.write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))
        elif change == "parameters":
            env.write_text(env.read_text() + "FASTFILL_LLM_REASONING_EFFORT=high\n")
        elif change == "predictions":  # e.g. an earlier autopilot's scored report next to regenerated predictions
            predictions = pilot.runs / "llm-prompt-300/predictions.jsonl"
            predictions.write_text(predictions.read_text() + "\n")
        elif change == "evaluator":  # scored by another evaluate implementation (same rows and predictions)
            report = pilot.runs / "llm-prompt-300-eval/report.json"
            report.write_text(json.dumps({**json.loads(report.read_text()), "implementation_sha256": "OLD_EVALUATOR"}))
        elif change == "old summary":  # the c750c31 summary: no identity recorded
            summary = pilot.runs / "llm-harness-300/summary.json"
            summary.write_text(json.dumps({k: v for k, v in json.loads(summary.read_text()).items()
                                           if k not in ("data_sha256", "implementation_sha256")}))
        else:  # llm_baseline.py on disk changed (the stub still records the imported module's hash: no convergence check)
            code = tmp_path / "fastfill/v2/llm_baseline.py"
            code.write_text(code.read_text() + "\n")
        pilot.llm_baselines(sample)
        both = ["fastfill.v2.llm_baseline", "fastfill.v2.evaluate"]
        assert calls == {"predictions": ["fastfill.v2.evaluate"], "evaluator": ["fastfill.v2.evaluate"],
                         "old summary": both}.get(change, both * 2), change
        if change != "code":
            calls.clear()
            pilot.llm_baselines(sample)  # converged: nothing left to redo
            assert calls == [], change
    assert "secret-test-key" not in (pilot.dir / "autorun.log").read_text()


def test_set_overrides_an_existing_config_key_only():
    import pytest
    from fastfill.v2.autorun import override
    config = {"model": {"max_objects": 128}, "training": {"steps": 10}}
    override(config, "model.max_objects=256")
    assert config["model"]["max_objects"] == 256
    with pytest.raises(ValueError, match="no existing config key"):
        override(config, "model.max_object=256")


def test_a_roomgenbench_hand_off_is_reused_only_for_the_same_checkpoint_and_code(tmp_path, monkeypatch):
    import argparse
    from fastfill.v2 import autorun
    pilot = autorun.Autopilot(argparse.Namespace(repo=str(tmp_path)))
    launched = []
    def sh(args, **kw):
        launched.append(args)
        tag = list(map(str, args))[-1]
        for i in range(5):
            room = pilot.runs / "roomgenbench" / tag / f"r{i}.layout_boxes"
            room.mkdir(parents=True, exist_ok=True)
            (room / "receipt.json").write_text("{}")
        return autorun.subprocess.Popen(["true"])
    monkeypatch.setattr(pilot, "sh", sh)
    monkeypatch.setattr(autorun, "implementation_sha256", lambda root: "code-A")
    pilot.roomgenbench("/ck/step-1", "best").wait()
    pilot.roomgenbench("/ck/step-1", "best").wait()
    assert len(launched) == 1  # same checkpoint, same code: reused
    monkeypatch.setattr(autorun, "implementation_sha256", lambda root: "code-B")
    pilot.roomgenbench("/ck/step-1", "best").wait()
    assert len(launched) == 2  # new code: exported again, the old directory set aside
    assert any(p.name.startswith("best.stale-") for p in (pilot.runs / "roomgenbench").iterdir())
