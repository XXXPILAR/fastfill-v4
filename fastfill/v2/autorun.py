"""Server autopilot: finish the running pair, select, retrain on the newest data, evaluate, publish.

Phases (state in <runs>/autorun/STATUS.json, log in <runs>/autorun/autorun.log):
  A  wait for the running training runs; a run that dies without run_manifest.json is resumed from its
     newest state-step into <run>-resume<k> (at most twice per run)
  B  evaluate the newest three model-step checkpoints of every run on the validation set (three-field
     projection, collision-aware spread decoding), one GPU each
  C  select the winner (``score``), export its RoomGenBench hand-off for the five benchmark rooms (CPU)
  D  wait for <data root>/NEXT_DATA.json (written after a rebuild is uploaded to Hugging Face) at most
     ``--data-wait-h`` hours; download and verify it and pull the matching code, else stop (phase failed)
  E  train the winner's configuration on all GPUs for ``--epochs`` epochs (same global batch), with the
     same crash-resume rule as A
  F  evaluate its newest checkpoints on the whole validation set, select, upload the best model directory to
     a private Hugging Face model repository (an upload error is a recorded failure), evaluate the best on the
     test set with spread and raw argmax decoding, export RoomGenBench; accepted (phase ``done``) only when
     every check passes, else ``done-with-failures``. Requests over the model capacity are reported, not failed
  LLM (background from C) once the API env file exists: prompt-only and harness LLM baselines on the
     first ``--llm-rows`` three-field rows of the validation sample, scored by evaluate --predictions; in F
     the best model is scored on the same rows. ``done`` writes <runs>/autorun/SUMMARY.md.
  With --start-config (a new machine) A-D are skipped: that configuration trains on --current-data.

Every decision and command is logged; a failing step stops the autopilot with STATUS phase "failed".
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import threading
import time

ENV = {"PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
       "HF_HOME": os.environ.get("HF_HOME", "/home/jovyan/shanliantian/.huggingface")}


def score(report):
    """Lower is better. Errors relative to the report's own trivial baselines, plus distribution penalties.

    The baselines are ``baselines_paired``: scored on the requests the model reference scores (those with a
    layout), so model and baseline errors cover the same objects. size: log-size error / per-category median
    size; yaw: yaw error / the expected error of a uniformly random yaw scored like the model's
    (``evaluate._uniform_yaw_error``: pi / (2 * symmetry order) per object, box-equivalent with the label's own size
    for swap-allowed objects; not a constant yaw); position: bottom-centre error / room-centre placement;
    |central-quarter fraction - GT|; |mean wall distance - GT| / GT; overlap above GT + 1 point and objects outside
    the room above GT count fivefold; the share of requests without a layout counts tenfold. The distribution terms use
    ``predicted_matched`` (the predictions matched to the label-complete objects the ground-truth column
    counts), so the labels themselves score 0 there.
    """
    ref, base = report["model"]["reference"], report["baselines_paired"]
    pred, gt = report["collapse"]["predicted_matched"], report["collapse"]["ground_truth"]
    ratio = lambda a, b: a / b if a is not None and b else 1.
    s = (ratio(ref["log_size_error"]["mean"], base["category_median_size"]["log_size_error"]["mean"])
         + ratio(ref["yaw_error_rad"].get("mean"), base["uniform_yaw"]["yaw_error_rad"].get("mean"))
         + ratio(ref["bottom_center_error_m"]["mean"], base["room_center_position"]["bottom_center_error_m"]["mean"])
         + abs(pred["central_quarter_fraction"] - gt["central_quarter_fraction"])
         + abs(pred["mean_nearest_wall_distance_m"] - gt["mean_nearest_wall_distance_m"]) / max(gt["mean_nearest_wall_distance_m"], 1e-6)
         + 5 * max(0., pred["bev_overlap_rate_iou_gt_0.3"] - gt["bev_overlap_rate_iou_gt_0.3"] - .01))
    if "out_of_room_fraction" in pred:
        s += 5 * max(0., pred["out_of_room_fraction"] - gt.get("out_of_room_fraction", 0.))
    # requests without a layout count tenfold: surviving-only means must not hide failures
    return s + 10 * (report.get("inference_failed_requests") or 0) / max(1, report.get("requests") or 0)


def next_config(config, *, world_size, train_rows, epochs, global_batch=96):
    """The winner's configuration rescaled to ``world_size`` GPUs at batch 1 and about ``global_batch``."""
    config = json.loads(json.dumps(config))
    training = config["training"]
    accumulation = max(1, round(global_batch / world_size))
    steps = math.ceil(epochs * train_rows / (world_size * accumulation))
    training.update(batch_size=1, gradient_accumulation_steps=accumulation, steps=steps, resume=None,
                    checkpoint_every=500, validate_every=500)
    config["optimizer"]["warmup_steps"] = round(.03 * steps)
    return config


def step_of(path):
    return int(str(path).rsplit("-", 1)[1])


def complete_states(outputs):
    """Saved states of every output of one run, oldest first. state-step-N counts as complete once model-step-N
    exists (train.py exports it after save_state and the barrier), so a state torn by a crash is skipped."""
    states = [p for o in outputs for p in glob.glob(f"{o}/state-step-*")
              if Path(p).with_name(f"model-step-{step_of(p)}").is_dir()]
    return sorted(states, key=step_of)


def training_alive(output):
    """Any fastfill.v2.train process whose command line names this output directory (relative or absolute)."""
    name = Path(output).name
    for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            args = Path(cmdline).read_bytes().split(b"\0")
        except OSError:
            continue
        if b"fastfill.v2.train" in args and any(a.rstrip(b"/").endswith(name.encode()) for a in args):
            return True
    return False


def override(config, assignment):
    """config[\"a\"][\"b\"] = json value for 'a.b=value'; the key must already exist (no silent typos)."""
    key, _, value = assignment.partition("=")
    *parents, leaf = key.split(".")
    node = config
    for name in parents:
        node = node[name]
    if not _ or leaf not in node:
        raise ValueError(f"--set {assignment!r}: no existing config key {key!r}")
    node[leaf] = json.loads(value)


def any_training_alive():
    """Any fastfill.v2.train process on this machine: GPU jobs must never start beside one."""
    return training_alive("")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256(root):
    """What an evaluate report records as ``implementation_sha256`` (io.run_metadata) for the fastfill/v2 code on disk
    under the repository ``root``: the evaluate subprocesses run from there."""
    root = Path(root).resolve()
    hashes = {str(path.relative_to(root)): sha256(path) for path in sorted((root / "fastfill/v2").glob("*.py"))}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def recorded(path, **fields):
    """``path`` is a readable JSON file recording exactly these field values (not missing, torn or from other inputs)."""
    try:
        old = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return all(old.get(key) == value for key, value in fields.items())


class Autopilot:
    def __init__(self, args):
        self.a, self.root = args, Path(args.repo)
        self.runs = self.root / "runs"
        self.dir = self.runs / getattr(args, "autorun_dir", "autorun")  # one per concurrent autopilot
        self.gpus = getattr(args, "gpus", "1,2,3,4,5,6,7").split(",")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.status = {"phase": "start", "history": []}
        self.python = str(self.root / "env/bin/python")
        self.lock, self.llm_reports = threading.Lock(), {}

    def log(self, message, **fields):
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message, **fields}
        with self.lock:  # the LLM baseline thread logs too
            with (self.dir / "autorun.log").open("a") as stream:
                stream.write(json.dumps(line, ensure_ascii=False) + "\n")
            self.status["history"].append(line)
            (self.dir / "STATUS.json").write_text(json.dumps(self.status, indent=2, ensure_ascii=False) + "\n")

    def phase(self, name, **fields):
        self.status.update(phase=name, **fields)
        self.log(f"phase {name}", **fields)

    def sh(self, args, *, gpus=None, log=None, wait=True):
        env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": gpus if gpus is not None else ""}
        self.log("run", args=" ".join(map(str, args)), gpus=gpus)
        out = open(self.root / log, "w") if log else subprocess.DEVNULL  # relative logs live under the repo
        process = subprocess.Popen(list(map(str, args)), cwd=self.root, env=env, stdout=out, stderr=subprocess.STDOUT)
        if not wait:
            return process
        if process.wait():
            raise RuntimeError(f"command failed ({process.returncode}): {' '.join(map(str, args))}")
        return process

    # -- training ---------------------------------------------------------------------------------
    def launch(self, config, data, output, gpus):
        world = len(gpus.split(","))
        return self.sh([self.python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={world}",
                        "--module", "fastfill.v2.train", "--config", config, "--data", f"{data}/train.jsonl",
                        "--validation", f"{data}/validation.jsonl", "--expect-data-sha256", sha256(f"{data}/train.jsonl"),
                        "--expect-validation-sha256", sha256(f"{data}/validation.jsonl"), "--output", output]
                       + list(self.extra_resume), gpus=gpus, log=f"{output}.log", wait=False)

    def out(self, name):
        return str(self.root / "runs" / name)

    def adopt(self, job):
        """Restore a run's resume chain after an autopilot restart: outputs, the resume count and the newest output."""
        resumes = sorted((p for p in glob.glob(f"{job['outputs'][0]}-resume*") if Path(p).is_dir()),  # not the .log files
                         key=lambda p: int(p.rsplit("resume", 1)[1]))
        job.update(outputs=[job["outputs"][0], *resumes], resumes=len(resumes))
        job["output"] = job["outputs"][-1]
        self.log("adopted training output", output=job["output"], resumes=job["resumes"])
        return job

    def supervise(self, jobs):
        """jobs: name -> dict(process|None, config, data, gpus, output). Resume crashed runs twice."""
        while True:
            pending = 0
            for name, job in jobs.items():
                done = (Path(job["output"]) / "run_manifest.json").is_file()
                alive = job["process"] is not None and job["process"].poll() is None
                if job["process"] is None and not done:  # adopted from an earlier launcher: find its trainer processes
                    alive = training_alive(job["output"])
                    if alive and time.time() - Path(f"{job['output']}.log").stat().st_mtime > 3600:
                        self.log(f"{name} alive but silent for over an hour", log=f"{job['output']}.log")
                if done or alive:
                    pending += not done
                    continue
                states = complete_states(job["outputs"])
                if job.get("resumes", 0) >= 2 or not states:
                    raise RuntimeError(f"{name} stopped without a final manifest and cannot be resumed")
                job["resumes"] = job.get("resumes", 0) + 1
                job["outputs"].append(f"{job['outputs'][0]}-resume{job['resumes']}")
                self.extra_resume = ("--resume", states[-1])
                self.log(f"{name} died; resuming", state=states[-1], output=job["outputs"][-1])
                job["process"] = self.launch(job["config"], job["data"], job["outputs"][-1], job["gpus"])
                self.extra_resume = ()
                job["output"] = job["outputs"][-1]
                pending += 1
            if not pending:
                return
            time.sleep(120)

    # -- evaluation -------------------------------------------------------------------------------
    def eval_data(self, rows=None):
        """A seeded random sample of --eval-rows (or `rows`; "all" = the whole file) validation rows: the file is
        grouped by source, so its head (what --max-samples reads) leaves whole sources out."""
        rows = rows or self.a.eval_rows
        source = f"{self.data}/validation.jsonl"
        path = self.dir / f"validation-{sha256(source)[:12]}-{rows}.jsonl"  # bound to the file's content
        if not path.is_file():
            with open(source) as stream:
                lines = [line.rstrip("\n") + "\n" for line in stream if line.strip()]
            random.Random(0).shuffle(lines)
            path.write_text("".join(lines if rows == "all" else lines[:int(rows)]))
        return path

    def evaluate_candidates(self, runs, tag, *, newest=3, gpus=None, data=None):
        """runs: one list of output directories (the run and its resumes) per run; its newest checkpoints compete.

        A restarted autopilot reuses a finished report of exactly the command below (same data and checkpoint, minimal
        projection, spread decoding, scorable). The reports of one selection all carry one ``implementation_sha256``
        (recorded per entry in <tag>-selection.json): the cached reports' when every candidate is cached under the same
        one, else that of the code on disk; other reports are set aside and evaluated again (refused beside a training)."""
        candidates, data, gpus = [], data or self.eval_data(), gpus or self.gpus
        for outputs in runs:
            candidates += sorted((p for o in outputs for p in glob.glob(f"{o}/model-step-*")), key=step_of)[-newest:]
        outs = {checkpoint: self.runs / f"{tag}-{Path(checkpoint).parent.name}-{Path(checkpoint).name}" for checkpoint in candidates}

        def set_aside(out):
            out.rename(out.with_name(f"{out.name}.stale-{time.strftime('%Y%m%d%H%M%S')}"))
            self.log("stale evaluation set aside", output=str(out))
        cached = {}  # checkpoint -> implementation_sha256 of its reusable report
        for checkpoint, out in outs.items():
            if (out / "report.json").is_file():  # a restarted autopilot reuses finished evaluations of the same command
                old = json.loads((out / "report.json").read_text())
                if (old.get("data_sha256") == sha256(data) and old.get("checkpoint") == str(Path(checkpoint).resolve())
                        and old.get("projection") == "minimal" and old.get("grid_decode") == "spread" and old.get("implementation_sha256")
                        and "predicted_matched" in (old.get("collapse") or {}) and old.get("baselines_paired")):  # older reports cannot be scored
                    cached[checkpoint] = old["implementation_sha256"]
                    continue
                set_aside(out)
        found = set(cached.values())
        implementation = found.pop() if len(found) == 1 and len(cached) == len(candidates) else implementation_sha256(self.root)
        for checkpoint in [c for c, digest in cached.items() if digest != implementation]:
            del cached[checkpoint]
            set_aside(outs[checkpoint])
        results = []
        for start in range(0, len(candidates), len(gpus)):
            batch = candidates[start:start + len(gpus)]
            jobs = []
            for gpu, checkpoint in zip(gpus, batch):
                out = outs[checkpoint]
                if checkpoint in cached:
                    jobs.append((checkpoint, out, None))
                    continue
                if any_training_alive():  # e.g. a restarted autopilot whose cached evaluation went missing
                    raise RuntimeError(f"{tag}: {checkpoint} needs a GPU evaluation while a fastfill.v2.train process runs")
                jobs.append((checkpoint, out, self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", checkpoint,
                    "--data", data, "--projection", "minimal",
                    "--device", "cuda", "--grid-decode", "spread", "--output", out], gpus=gpu, log=f"{out}.log", wait=False)))
            for checkpoint, out, process in jobs:
                if process is not None and process.wait():
                    self.log("evaluation failed", checkpoint=checkpoint)
                    continue
                try:
                    report = json.loads((out / "report.json").read_text())
                    value = score(report)
                except (KeyError, TypeError, ValueError) as error:
                    self.log("evaluation unscorable", checkpoint=checkpoint, error=repr(error))
                    continue
                results.append({"checkpoint": checkpoint, "report": str(out / "report.json"), "score": value,
                                "implementation_sha256": report.get("implementation_sha256")})
        if not results:
            raise RuntimeError("no checkpoint could be evaluated")
        mixed = [r["report"] for r in results if r["implementation_sha256"] != implementation]
        if mixed:  # the code on disk changed during the evaluations; a restart evaluates the reports of other code again
            raise RuntimeError(f"{tag}: {mixed} were not evaluated by implementation {implementation}")
        results.sort(key=lambda r: r["score"])
        (self.dir / f"{tag}-selection.json").write_text(json.dumps(results, indent=2) + "\n")
        self.log(f"{tag} selection", winner=results[0], ranking=[(r["checkpoint"], round(r["score"], 4)) for r in results])
        return results[0]

    def roomgenbench(self, checkpoint, tag):
        """The five rooms' hand-off of ``checkpoint``; reused only for the same checkpoint and the same fastfill/v2 code
        (prediction, decoding and export all run from it), else the old directory is set aside and exported again."""
        out, bound = self.runs / "roomgenbench" / tag, self.runs / "roomgenbench" / tag / "checkpoint.txt"
        identity = f"{checkpoint}\n{implementation_sha256(self.root)}"
        if (len(glob.glob(str(out / "*.layout_boxes" / "receipt.json"))) == 5 and bound.is_file()
                and bound.read_text().strip() == identity):
            self.log("RoomGenBench hand-off already complete", tag=tag)  # a restarted autopilot
            return subprocess.Popen(["true"])
        if out.exists():  # predict refuses existing outputs: a different checkpoint or code exports afresh
            stale = out.with_name(f"{out.name}.stale-{time.strftime('%Y%m%d%H%M%S')}")
            out.rename(stale)
            self.log("stale RoomGenBench hand-off set aside", tag=tag, output=str(stale))
        out.mkdir(parents=True)
        bound.write_text(identity + "\n")
        return self.sh(["bash", "fastfill/v2/ops/run_checkpoint.sh", checkpoint, tag],
                       log=self.runs / "roomgenbench" / f"{tag}.log", wait=False)

    # -- data ---------------------------------------------------------------------------------------
    def next_data(self):
        marker, deadline = Path(self.a.data_root) / "NEXT_DATA.json", time.time() + 3600 * self.a.data_wait_h
        while not marker.is_file() and time.time() < deadline:
            time.sleep(120)
        if not marker.is_file():  # never train the formal run on data that was meant to be replaced
            raise RuntimeError(f"no {marker} within {self.a.data_wait_h} h; publish the rebuilt data and restart the autopilot")
        spec = json.loads(marker.read_text())
        missing = {"train.jsonl", "validation.jsonl", "test.jsonl"} - set(spec["sha256"])
        if missing:  # phase F reads test.jsonl only ~10 h later
            raise RuntimeError(f"NEXT_DATA.json does not pin {sorted(missing)}")
        target = Path(self.a.data_root) / spec["name"]
        if not target.is_dir():
            staging = Path(self.a.data_root) / f".download-{spec['name']}"
            self.sh([str(self.root / "env/bin/hf"), "download", spec["repo"], "--repo-type", "dataset", "--revision",
                     spec["revision"], "--include", f"{spec['folder']}/*", "--local-dir", staging], log=self.dir / "hf-download.log")
            (staging / spec["folder"]).rename(target)
        for name, digest in spec["sha256"].items():
            if sha256(target / name) != digest:
                raise RuntimeError(f"{target / name} does not match NEXT_DATA.json")
        self.sh(["git", "pull", "--ff-only", "origin", "main"], log=self.dir / "git-pull-data.log")
        self.sh(["git", "merge-base", "--is-ancestor", spec["commit"], "HEAD"])
        self.data = str(target)
        self.log("switched to new data", data=self.data, commit=spec["commit"])

    def upload(self, checkpoint, name):
        """<name>/model-step-N plus <name>/tokenizer and the run manifests: the layout predict/evaluate load
        (they read the tokenizer from the checkpoint's parent)."""
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(self.a.model_repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(folder_path=Path(checkpoint).parent, path_in_repo=name, repo_id=self.a.model_repo, repo_type="model",
                          allow_patterns=[f"{Path(checkpoint).name}/*", "tokenizer/*", "run_manifest*.json"],
                          commit_message=f"FastFill v2 {name}")
        self.log("uploaded model", repo=self.a.model_repo, path=name)

    def llm_baselines(self, rows_file):
        """Prompt-only and harness LLM baselines once the API env file exists, each scored by evaluate --predictions.

        A restarted autopilot reuses a baseline only when its summary.json records these rows, the llm_baseline.py on disk
        and the request parameters of the env file's model, and its scored report only when that records these rows,
        predictions and the evaluate code on disk; anything else (interrupted, older or other inputs) is removed and run again."""
        from fastfill.v2.llm_baseline import load_env, request_parameters
        env = Path(self.a.llm_env)
        while not env.is_file():
            if self.status["phase"] in ("F-evaluate-final", "done", "failed"):
                self.log("LLM baselines skipped: no API env file", env=str(env))
                return
            time.sleep(300)
        for mode in ("prompt", "harness"):
            out, scored = self.runs / f"llm-{mode}-{self.a.llm_rows}", self.runs / f"llm-{mode}-{self.a.llm_rows}-eval"
            try:
                values = load_env(env)  # holds the API key: never logged
                if not recorded(out / "summary.json", data_sha256=sha256(rows_file),
                                implementation_sha256=sha256(self.root / "fastfill/v2/llm_baseline.py"),
                                request_parameters=request_parameters(values, values.get("OPENAI_MODEL"))):
                    shutil.rmtree(out, ignore_errors=True)
                    self.sh([self.python, "-m", "fastfill.v2.llm_baseline", "--data", rows_file, "--env", env, "--mode", mode,
                             "--max-samples", self.a.llm_rows, "--output", out], log=f"{out}.log")
                if not recorded(scored / "report.json", data_sha256=sha256(out / "rows.jsonl"),
                                predictions_sha256=sha256(out / "predictions.jsonl"),
                                implementation_sha256=implementation_sha256(self.root)):
                    shutil.rmtree(scored, ignore_errors=True)
                    self.sh([self.python, "-m", "fastfill.v2.evaluate", "--data", out / "rows.jsonl", "--predictions",
                             out / "predictions.jsonl", "--output", scored], log=f"{scored}.log")
                self.llm_reports[f"LLM {mode}"] = scored / "report.json"
                self.log(f"LLM {mode} baseline scored", summary=json.loads((out / "summary.json").read_text()))
            except Exception as error:
                self.log(f"LLM {mode} baseline failed", error=f"{type(error).__name__}: {error}")

    def summary(self, best, test_dir, rows_report, failures=(), raw_test=None):
        """SUMMARY.md: selection, the best model on validation and test, and the model next to the LLM baselines."""
        def metrics(path):
            r = json.loads(Path(path).read_text())
            ref, pred, gt, base = (r["model"]["reference"], r["collapse"]["predicted_matched"], r["collapse"]["ground_truth"],
                                   r.get("baselines_paired"))
            def get(d, *keys):
                for key in keys:
                    d = (d or {}).get(key)
                return f"{d:.3f}" if isinstance(d, float) else "-"
            def checked(v):  # validator collisions and clean rooms of one layout set
                if not v:
                    return "-", "-"
                return (f"{v['rooms_with_collision']}/{v['rooms']} rooms, {v['collision_pairs']} pairs (+{v['fixed_collision_pairs']} fixed)",
                        f"{v['clean_room_rate']:.3f} ({v['clean_rooms_with_hard_unknown']} with unknown)")
            validation = r.get("validation") or {}
            model_collisions, model_clean = checked(validation.get("model"))
            gt_collisions, gt_clean = checked(validation.get("ground_truth"))
            walls = (r.get("target_validation_checks") or {}).get("boundary") or {}
            inside = walls.get("pass", 0) / max(1, walls.get("pass", 0) + walls.get("violation", 0))
            failed = f"{r.get('inference_failed_requests', '-')}" + (f" ({r['over_capacity_requests']} over capacity)"
                                                                       if r.get("over_capacity_requests") else "")
            return [f"{r.get('requests', '-')} / {failed}", f"{inside:.3f}",
                    get(ref, "bottom_center_error_m", "mean"),
                    get(ref, "log_size_error", "mean"), get(ref, "yaw_error_rad", "mean"),
                    f"{get(pred, 'bev_overlap_rate_iou_gt_0.3')} / {get(pred, 'bev_overlap_rate_iou_gt_0.3_room_mean')}",
                    model_collisions, model_clean,
                    f"{r.get('hard_violation_requests', '-')} / {r.get('failed_requests_strict', r.get('failed_requests', '-'))}",
                    get(pred, "central_quarter_fraction"), get(pred, "mean_nearest_wall_distance_m"), get(pred, "out_of_room_fraction"),
                    f"GT {get(gt, 'bev_overlap_rate_iou_gt_0.3')} / {get(gt, 'bev_overlap_rate_iou_gt_0.3_room_mean')}; "
                    f"{gt_collisions}; clean {gt_clean}; {get(gt, 'central_quarter_fraction')} / {get(gt, 'mean_nearest_wall_distance_m')}",
                    f"{get(base, 'room_center_position', 'bottom_center_error_m', 'mean')} / "
                    f"{get(base, 'category_median_size', 'log_size_error', 'mean')} / {get(base, 'uniform_yaw', 'yaw_error_rad', 'mean')}"]
        head = ("| | requests / no layout | objects inside walls | position err (m) | log-size err | yaw err (rad) "
                "| overlap pairs / room mean | validator collisions | clean rooms (no hard violation) | hard-violation / strict-failed requests "
                "| central | wall dist (m) | out of room "
                "| ground truth overlap pairs / room mean; collisions; clean rooms; central / wall "
                "| paired baselines (same requests) room centre / category median size (eval-set LOO) / uniform-random yaw |")
        head += "\n|" + "---|" * (head.count("|") - 1)  # GFM: as many delimiter cells as header cells
        def table(rows):
            lines = [head]
            for name, path in rows:
                try:
                    lines.append(f"| {name} | " + " | ".join(metrics(path)) + " |")
                except Exception as error:
                    lines.append(f"| {name} | unreadable: {type(error).__name__} |")
            return "\n".join(lines)
        passed = ("PASSED (test report with a layout for every request within the model capacity, raw argmax test report, "
                  "five RoomGenBench rooms assembled as bbox scenes; their validator/physics are not run)")
        text = [f"# FastFill v2 autopilot summary ({time.strftime('%Y-%m-%d %H:%M')})", "",
                f"Best model: `{best['checkpoint']}` (selection score {best['score']:.4f}; lower is better)", "",
                f"Acceptance: {passed if not failures else 'FAILED: ' + '; '.join(failures)}. "
                + (f"The model was uploaded to {self.a.model_repo} before these checks; treat it as accepted only if they passed."
                 if not any(f.startswith("upload failed") for f in failures) else "The model upload failed; the checkpoint stays on the server."), "",
                "Requests over the model capacity (more objects than max_objects) count as no layout and are listed separately; "
                "'objects inside walls' is the validator's bbox check, 'out of room' only the bottom-centre. 'overlap' counts "
                "BEV IoU > 0.3 pairs (pooled over pairs / mean over rooms) and misses smaller overlaps; 'validator collisions' "
                "are validate_scene's collision checks (any footprint overlap sharing height; rooms with one, requested pairs, "
                "pairs with fixed objects). A clean room has no hard violation; unknown checks are not violations, and the clean "
                "rooms with one are counted beside the rate; strict-failed requests also fail on unknown. The ground-truth "
                "column is the same rooms' labels under the same validator. Baselines are scored on the requests with a "
                "layout (the model's own objects).", "",
                "## Best model", "", table([("validation (three-field, select2 cohort)", best["report"]),
                                             ("test (three-field), spread decoding", test_dir / "report.json")]
                                            + ([("test (three-field), raw argmax: the model alone", raw_test)] if raw_test else [])), "",
                "The full-condition test results are in the test report under projections.full.", ""]
        if rows_report is not None:
            text += ["## Same rooms: trained model vs LLM", "", "FastFill best uses collision-aware spread decoding (post-processing); "
                     "the raw argmax row is the model alone.", "",
                     table([("FastFill best, spread decoding", rows_report)] + sorted(self.llm_reports.items())), ""]
        for tag in ("select1", "select2"):
            path = self.dir / f"{tag}-selection.json"
            if path.is_file():
                ranking = json.loads(path.read_text())
                text += [f"## {tag} ranking (its own validation cohort; scores of different tags are not comparable)", ""] + \
                        [f"- {r['score']:.4f} `{r['checkpoint']}`" for r in ranking] + [""]
        (self.dir / "SUMMARY.md").write_text("\n".join(text) + "\n")
        self.log("wrote SUMMARY.md")

    def run(self):
        self.extra_resume, self.data = (), str(Path(self.a.current_data).resolve())
        if getattr(self.a, "start_config", None):  # a new machine: no earlier runs to finish, the data is in place
            config = json.loads(Path(self.a.start_config).read_text())
            self.log("start configuration", config=self.a.start_config, data=self.data)
            llm = threading.Thread(target=self.llm_baselines, args=(self.eval_data(),), daemon=True)
            llm.start()
        else:
            self.phase("A-wait-current-runs")
            jobs = {name: self.adopt({"process": None, "config": config, "data": self.data, "gpus": gpus, "output": self.out(name),
                                      "outputs": [self.out(name)]}) for name, config, gpus in self.a.current}
            self.supervise(jobs)
            self.sh(["git", "pull", "--ff-only", "origin", "main"], log=self.dir / "git-pull-a.log")
            self.phase("B-evaluate-current")
            winner = self.evaluate_candidates([job["outputs"] for job in jobs.values()], "select1")
            self.phase("C-selected", winner=winner)
            llm = threading.Thread(target=self.llm_baselines, args=(self.eval_data(),), daemon=True)
            llm.start()
            handoff = self.roomgenbench(winner["checkpoint"], "best-round1")
            # its five rooms each start a new process from the repo; finish them before phase D pulls new code
            self.log("round-1 RoomGenBench hand-off finished", returncode=handoff.wait())
            self.phase("D-wait-data")
            self.next_data()
            run_dir = Path(winner["checkpoint"]).parent
            config = json.loads((run_dir / "run_manifest_start.json").read_text())["config"]
        rows, world = sum(1 for _ in open(f"{self.data}/train.jsonl")), len(self.gpus)
        config = next_config(config, world_size=world, train_rows=rows, epochs=self.a.epochs)
        for assignment in getattr(self.a, "set", None) or ():  # explicit, logged departures from the winner's configuration
            override(config, assignment)
            self.log("config override", set=assignment)
        name = f"main{world}-cell{config['loss']['position_cell']:g}-{Path(self.data).name}-e{self.a.epochs}".replace(".", "")
        name += getattr(self.a, "name_suffix", None) or ""  # distinct output for a run with --set overrides
        config_path = self.dir / f"{name}.json"
        if not config_path.is_file():
            config_path.write_text(json.dumps(config, indent=2) + "\n")
        self.phase("E-train", run=name, config=str(config_path), data=self.data, steps=config["training"]["steps"])
        job = {"config": str(config_path), "data": self.data, "gpus": ",".join(self.gpus), "output": self.out(name),
               "outputs": [self.out(name)], "process": None}
        if Path(job["output"]).exists():  # a restarted autopilot adopts its own earlier launch
            self.adopt(job)
        else:
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            if any_training_alive():
                raise RuntimeError(f"{name}: would launch beside a running fastfill.v2.train process")
            job["process"] = self.launch(job["config"], self.data, job["output"], job["gpus"])
        self.supervise({name: job})
        self.phase("F-evaluate-final")
        best = self.evaluate_candidates([job["outputs"]], "select2", newest=4, data=self.eval_data("all"))
        failures, uploaded, g = [], True, self.gpus  # F uses four GPUs
        try:
            self.upload(best["checkpoint"], name)  # an UNACCEPTED backup until the checks below pass (see final phase)
        except Exception as error:  # the evaluations below still run
            uploaded = False
            failures.append(f"upload failed: {type(error).__name__}: {error}")
            self.log(failures[-1])
        export = self.roomgenbench(best["checkpoint"], f"best-{name}")
        llm.join(timeout=4 * 3600)
        rows_report, rows_job, raw_job = None, None, None
        rows = self.runs / f"llm-prompt-{self.a.llm_rows}" / "rows.jsonl"
        if rows.is_file():  # the trained model on exactly the rooms the LLM baselines answered
            rows_out = self.runs / f"llmrows-{name}"
            shutil.rmtree(rows_out, ignore_errors=True)
            rows_job = self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", best["checkpoint"], "--data", rows,
                                "--projection", "full", "--device", "cuda", "--grid-decode", "spread", "--output", rows_out],
                               gpus=g[1], log=f"{rows_out}.log", wait=False)
            raw_out = self.runs / f"llmrows-{name}-argmax"
            shutil.rmtree(raw_out, ignore_errors=True)
            raw_job = self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", best["checkpoint"], "--data", rows,
                               "--projection", "full", "--device", "cuda", "--grid-decode", "argmax", "--output", raw_out],
                              gpus=g[2], log=f"{raw_out}.log", wait=False)
        out, raw_test = self.runs / f"test-{name}", self.runs / f"test-{name}-argmax"
        shutil.rmtree(raw_test, ignore_errors=True)
        raw_test_job = self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", best["checkpoint"], "--data",
                                f"{self.data}/test.jsonl", "--projection", "minimal", "--device", "cuda", "--grid-decode", "argmax",
                                "--output", raw_test], gpus=g[3], log=f"{raw_test}.log", wait=False)  # the model alone
        shutil.rmtree(out, ignore_errors=True)  # evaluate refuses an existing output; a rerun of F starts over
        if self.sh([self.python, "-m", "fastfill.v2.evaluate", "--checkpoint", best["checkpoint"], "--data", f"{self.data}/test.jsonl",
                    "--projection", "minimal", "full", "--device", "cuda", "--grid-decode", "spread", "--output", out],
                   gpus=g[0], log=f"{out}.log", wait=False).wait():
            failures.append("spread test evaluation failed")
        if rows_job is not None and rows_job.wait() == 0:
            rows_report = rows_out / "report.json"
        export_code = export.wait()  # wait first, then count the five rooms' receipts
        rooms = sorted(glob.glob(str(self.runs / "roomgenbench" / f"best-{name}" / "*.layout_boxes" / "receipt.json")))
        self.log("final RoomGenBench export", returncode=export_code, rooms_with_receipt=len(rooms))
        if raw_job is not None and raw_job.wait() == 0:
            self.llm_reports["FastFill best, raw argmax (no post-processing)"] = raw_out / "report.json"
        failures += [f"RoomGenBench export exit {export_code}"] if export_code else []
        failures += [f"RoomGenBench receipts {len(rooms)}/5"] if len(rooms) != 5 else []
        if raw_test_job.wait():
            failures.append("raw argmax test evaluation failed")
        if not (out / "report.json").is_file():
            failures.append("test report missing")
        else:
            test = json.loads((out / "report.json").read_text())
            failed = test["inference_failed_requests"] - test.get("over_capacity_requests", 0)  # capacity is reported, not failed
            failures += [f"test: {failed} of {test['requests']} requests without a layout"] if failed else []
        self.summary(best, out, rows_report, failures, raw_test / "report.json")
        self.phase("done" if not failures else "done-with-failures", best=best, failures=failures,
                   uploaded_model=f"{self.a.model_repo}/{name}" if uploaded else None, accepted=not failures,
                   test_report=str(out / "report.json"), summary=str(self.dir / "SUMMARY.md"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="/home/jovyan/shanliantian/fastfill")
    p.add_argument("--current", nargs=3, action="append", metavar=("RUN", "CONFIG", "GPUS"))
    p.add_argument("--start-config", help="skip A-D: train this configuration on --current-data (a new machine)")
    p.add_argument("--gpus", default="1,2,3,4,5,6,7", help="training uses all, phase F the first four")
    p.add_argument("--autorun-dir", default="autorun", help="status directory under runs/, one per concurrent autopilot")
    p.add_argument("--current-data", required=True)
    p.add_argument("--data-root", default="/home/jovyan/shanliantian/fastfill/data")
    p.add_argument("--data-wait-h", type=float, default=2.)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--name-suffix", default="", help="appended to the next run's name, e.g. -yawcls05")
    p.add_argument("--set", action="append", metavar="KEY=JSON",
                   help="override a key of the next run's config, e.g. model.max_objects=256 (repeatable)")
    p.add_argument("--eval-rows", default="3000")
    p.add_argument("--model-repo", default="liantian/fastfill-v2-models")
    p.add_argument("--llm-env", default="/home/jovyan/shanliantian/.fastfill_api.env")
    p.add_argument("--llm-rows", default="300")
    os.environ.setdefault("HF_HOME", ENV["HF_HOME"])  # the in-process Hugging Face upload uses the logged-in home
    args = p.parse_args(argv)
    if not (args.current or args.start_config):
        p.error("give --current runs to finish or a --start-config")
    pilot = Autopilot(args)
    try:
        pilot.run()
    except Exception as error:
        pilot.phase("failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
