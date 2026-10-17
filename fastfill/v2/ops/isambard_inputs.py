"""Inputs of the Isambard final analysis (run_plan.sh inputs): the two autopilot arms, the old baseline, the saved LLM
answers and the old data, every downloaded file sha256-pinned. Writes OUT/inputs.json (read by report.py) and
OUT/inputs.env (sourced by run_plan.sh).

    python isambard_inputs.py --repo . --out runs/analysis [--dry-run]

Arms (runs/autorun-yawcls05, runs/autorun-yawcls008): refused unless both STATUS.json phases are done or
done-with-failures, both select2 cohorts (the data of each STATUS best report) hold the same bytes and both select2
scores were computed by the same fastfill/v2 code (the reports' implementation_sha256). final = the arm
whose STATUS best.score is lower (select2: the whole new validation set, minimal projection, spread decoding; lower is
better); control = the other; equal scores are refused.
Downloads (huggingface_hub with the submitting shell's HF_HOME and login) go to staging directories that are never
edited: runs/.download-baseline and runs/.download-llm (liantian/fastfill-v2-models@MODEL_REV), and
data/.download-main-20261007b (dataset liantian/fastfill-v2@DATA_REV), moved to data/main-20261007b. The baseline is used
from a copy, runs/baseline-hf/<run>, whose model-step-6357/model_config.json names this checkout's models/Qwen3-8B
instead of the old server's path (the same weights: its config.json fingerprint is the one the checkpoint manifest
records), recorded in BACKBONE_REWRITE.json there. The LLM answers are installed into runs/llm-{prompt,harness}-300/ (an
existing file must be identical); the structured modes count only if runs/llm-structured{,-harness}-300 exist. The
baseline's old selection cohorts are rebuilt by Autopilot.eval_data itself.
--dry-run (the CPU dry run only): no downloads (the staging directories are filled by hand) and the pins of the baseline,
data and cohort files are reported instead of enforced; the LLM pins stay enforced.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil

from fastfill.v2.autorun import Autopilot, sha256, step_of

MODEL_REPO, MODEL_REV = "liantian/fastfill-v2-models", "6134404806239db0865ec59ab85a2e1cfa8c63fb"
DATA_REPO, DATA_REV = "liantian/fastfill-v2", "eb5fbce19060cbecafaf53236e73ac47179e5803"
BASELINE, STEP = "main7-cell05-main-20261007b-e5", "model-step-6357"
OLD_BACKBONE = "/home/jovyan/shanliantian/models/Qwen3-8B"
BASELINE_FILES = {
    "model-step-6357/backbone/README.md": "0d6bb7b105e804fcb41d27263a3210ce3a1a85bb4b1dc6d5d2768dc1a58adcdc",
    "model-step-6357/backbone/adapter_config.json": "89f8a13343c1bcb5ba556b26fe0b2dafccf9001773accc1a78ac16fe4fbb480c",
    "model-step-6357/backbone/adapter_model.safetensors": "47f4db050cd38b252af749bc37275f2757f0bf06e4f4d75dad618c56970a9a9f",
    "model-step-6357/checkpoint_manifest.json": "17811551473e615b6818ce1616c33dd8011e153b9555f68143f78b7de24e364c",
    "model-step-6357/geometry_model.pt": "d863ed8e8a9cb266efeefd8761c467bab49fb205dd4258ae54ac0f293f17e2f1",
    "model-step-6357/model_config.json": "ae89bd799ac596290b11ae68f8ad59570fb9b0ff5c1ab7b016871d2a4e3cad72",
    "run_manifest.json": "dbaae6c18b081789c10880b17a4b6680513c37d78b01b640a7052374b4daac2b",
    "run_manifest_start.json": "ef1d29bb719df312724dfecf3b8f1e164ac06997fec136211d946c859f15e73a",
    "tokenizer/chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "tokenizer/tokenizer.json": "be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506",
    "tokenizer/tokenizer_config.json": "1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95"}
LLM_ROWS = "e02da65e6844630eba506952ee1f059655552a2d14e1af763eb787cc2d0152ea"
LLM_FILES = {"prompt": {"rows.jsonl": LLM_ROWS, "summary.json": "635350e15959be4487cc83604bb92712746a91f5877c8e2492f59449469cf359",
                        "predictions.jsonl": "a59d886c05ebc776460338a04623e4792f1ffbe545c5e6539915b612f8804f0c"},
             "harness": {"rows.jsonl": LLM_ROWS, "summary.json": "0f6889aca52ee31654df3d1ca549f69c3b85d60bf3fd3edd4391ed3ee766447f",
                         "predictions.jsonl": "fd27a5bb57fcd6f08d8161e7d7ca2e7090c39eb55c55f7ea0887b18cb398575d"}}
STRUCTURED = ("structured", "structured-harness")  # their answers stayed on the old server
NEW_DATA = {"train.jsonl": "5d9439f63b32bdfbf8200b52ef1c2681435053a0dc8b88a84234f24f0d8160eb",
            "validation.jsonl": "93cca6cb1abe80ac6eaa5713143d0b6b44774fa0d874a7c37842f7d514f10a39",
            "test.jsonl": "212fc6bfa2302173757a110a51a5f3ac34ae66111bd5d96860de6997ab82f107"}
OLD_DATA = {"train.jsonl": "609ae3cb46bd897a8942455a9bff251962f14c466fd51a0beb9785f914328226",
            "validation.jsonl": "e83cd12417deb1df6c4a6462cb69cd085728bdfcbbd8183fb5772eee1eae4236",
            "test.jsonl": "3374ca9160973ba6c44e22ddb28298a1f5a5cc51a336d7b080aba2a853a3e704"}
OLD_COHORT_ALL = "06c0bdcc1b909ab69e8f5811da17fcbfdc26bac4942877528a32b0a6dfe30299"  # data_sha256 of the baseline's select2 report
ARMS = ("autorun-yawcls05", "autorun-yawcls008")
RULE = ("最终臂 = STATUS.json best.score 更低的一臂（autopilot 的 select2 分数：新验证集全集、三字段投影、spread 解码，"
        "越低越好）；另一臂为对照。两臂的 select2 队列逐字节相同，且由同一评测代码（implementation_sha256）评分。下面的配对比较是这一选择的证据，测试集只报告。")


class Pins:
    """sha256 checks; ``enforce`` False (dry run) records a mismatch instead of refusing, except for ``always`` pins."""

    def __init__(self, enforce):
        self.enforce, self.checked, self.mismatches = enforce, 0, []

    def __call__(self, path, expected, always=False):
        actual = sha256(path) if Path(path).is_file() else None
        self.checked += 1
        if actual != expected:
            if always or self.enforce:
                raise SystemExit(f"refused: {path} sha256 {actual} is not the pinned {expected}")
            self.mismatches.append({"file": str(path), "sha256": actual, "pinned": expected})
        return actual


def download(repo, revision, pattern, local_dir, repo_type="model"):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type=repo_type, revision=revision, allow_patterns=[pattern], local_dir=str(local_dir))


def arm(runs, name):
    d = runs / name
    status = json.loads((d / "STATUS.json").read_text())
    if status.get("phase") not in ("done", "done-with-failures"):
        raise SystemExit(f"refused: {d}/STATUS.json phase is {status.get('phase')!r}, not done or done-with-failures")
    best = status["best"]
    report = json.loads(Path(best["report"]).read_text())
    cohort = Path(report["data_path"])
    if cohort.parent.resolve() != d.resolve() or not cohort.name.endswith("-all.jsonl"):
        raise SystemExit(f"refused: {name}'s best report scored {cohort}, not its select2 cohort")
    if sha256(cohort) != report["data_sha256"]:
        raise SystemExit(f"refused: {cohort} changed after {name}'s select2 evaluation")
    if best.get("implementation_sha256") != report["implementation_sha256"]:
        raise SystemExit(f"refused: {name}: STATUS best implementation_sha256 is not its report's")
    ranking = json.loads((d / "select2-selection.json").read_text())
    if (ranking[0]["checkpoint"], ranking[0]["score"]) != (best["checkpoint"], best["score"]):
        raise SystemExit(f"refused: {name}: STATUS best is not the top of select2-selection.json")
    config = json.loads(Path(status["config"]).read_text())
    start = json.loads((Path(best["checkpoint"]).parent / "run_manifest_start.json").read_text())
    training = config["training"]
    batch = start["world_size"] * training["gradient_accumulation_steps"] * training["batch_size"]
    best_step = step_of(best["checkpoint"])
    return {"dir": name, "run": status["run"], "phase": status["phase"], "failures": status.get("failures") or [],
            "accepted": status.get("accepted"), "uploaded_model": status.get("uploaded_model"), "config": status["config"],
            "yaw_cls": config["loss"]["yaw_cls"], "max_objects": config["model"]["max_objects"],
            "steps": training["steps"], "global_batch": batch, "supervised_samples": start["supervised_samples"],
            "rejected_samples": start["rejected_samples"], "epochs_supervised": training["steps"] * batch / start["supervised_samples"],
            "best_checkpoint": best["checkpoint"], "best_step": best_step, "best_epochs": best_step * batch / start["supervised_samples"],
            "best_score": best["score"], "best_report": best["report"], "select2_cohort": str(cohort),
            "select2_cohort_sha256": report["data_sha256"], "select2_implementation_sha256": report["implementation_sha256"],
            "select2_ranking": [[r["checkpoint"], r["score"]] for r in ranking], "test_report": status.get("test_report"),
            "summary": str(d / "SUMMARY.md")}


def choose(arms):
    """(final, control): the lower select2 score; refused unless both cohorts are the same bytes, scored by the same code."""
    a, b = arms
    if a["select2_cohort_sha256"] != b["select2_cohort_sha256"]:
        raise SystemExit(f"refused: the arms' select2 cohorts differ ({a['select2_cohort']} vs {b['select2_cohort']})")
    if a["select2_implementation_sha256"] != b["select2_implementation_sha256"]:
        raise SystemExit(f"refused: the arms' select2 scores come from different fastfill/v2 code "
                         f"({a['select2_implementation_sha256'][:12]} vs {b['select2_implementation_sha256'][:12]})")
    if a["best_score"] == b["best_score"]:
        raise SystemExit("refused: both arms have the same select2 score; choose the final arm by hand")
    return (a, b) if a["best_score"] < b["best_score"] else (b, a)


def baseline_copy(source, target, backbone):
    """``target``: a copy of the downloaded run directory ``source`` whose model_config.json names ``backbone`` (only
    when it names the old server's backbone; the dry run's tiny stand-in keeps "tiny"). The download stays untouched."""
    config_file = source / STEP / "model_config.json"
    config = json.loads(config_file.read_text())
    record = {"source": str(source), "file": f"{STEP}/model_config.json", "sha256_before": sha256(config_file),
              "backbone_before": config["backbone"]}
    if config["backbone"] == OLD_BACKBONE:
        manifest = json.loads((source / STEP / "checkpoint_manifest.json").read_text())
        found = sha256(Path(backbone) / "config.json")
        if found != manifest["backbone"]["config_sha256"]:
            raise SystemExit(f"refused: {backbone}/config.json {found[:12]} is not the baseline's backbone "
                             f"{manifest['backbone']['config_sha256'][:12]}")
        config["backbone"], record["backbone_config_sha256"] = str(backbone), found
    record["backbone_after"] = config["backbone"]
    staging = target.with_name(target.name + ".tmp")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source, staging)  # real copies: the write below never reaches the download
    (staging / STEP / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    record["sha256_after"] = sha256(staging / STEP / "model_config.json")
    (staging / "BACKBONE_REWRITE.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    return record


def install(source, target):
    """Copy ``source`` to ``target`` unless an identical file is there; a different one is refused."""
    if target.exists():
        if sha256(target) != sha256(source):
            raise SystemExit(f"refused: {target} exists and differs from the downloaded {source}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def cohorts(repo, out, old):
    """The baseline's selection cohorts (select2: all rows; select1: 3000) of the old validation file, written by
    Autopilot.eval_data itself into OUT/cohorts."""
    pilot = Autopilot(argparse.Namespace(repo=str(repo), autorun_dir=str((out / "cohorts").resolve())))
    pilot.data = str(old)
    every, first = pilot.eval_data("all"), pilot.eval_data("3000")
    if not every.read_bytes().startswith(first.read_bytes()):
        raise SystemExit(f"refused: {first} is not the first 3000 rows of {every}")
    return every, first


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    repo, runs = a.repo.resolve(), a.repo.resolve() / "runs"
    a.out.mkdir(parents=True, exist_ok=True)
    pins = Pins(enforce=not a.dry_run)
    new, old = repo / "data/main-20261008a", repo / "data/main-20261007b"
    final, control = choose([arm(runs, name) for name in ARMS])  # refuse before any download
    for name, digest in NEW_DATA.items():
        pins(new / name, digest)
    if Path(final["select2_cohort"]).name != f"validation-{sha256(new / 'validation.jsonl')[:12]}-all.jsonl":
        raise SystemExit(f"refused: {final['select2_cohort']} is not a cohort of {new}/validation.jsonl")

    stage = runs / ".download-llm"
    if not a.dry_run:
        download(MODEL_REPO, MODEL_REV, "llm-comparison/*/*", stage)
    llm = {"repo": MODEL_REPO, "revision": MODEL_REV, "modes": {}, "missing": []}
    for mode, files in LLM_FILES.items():  # always enforced: the dry run uses the real answers
        for name, digest in files.items():
            pins(stage / "llm-comparison" / mode / name, digest, always=True)
            install(stage / "llm-comparison" / mode / name, runs / f"llm-{mode}-300" / name)
        llm["modes"][mode] = {"dir": str(runs / f"llm-{mode}-300"), "sha256": files}
    rows = runs / "llm-prompt-300/rows.jsonl"
    for mode in STRUCTURED:
        d = runs / f"llm-{mode}-300"
        if all((d / name).is_file() for name in ("rows.jsonl", "predictions.jsonl", "summary.json")):
            if sha256(d / "rows.jsonl") != LLM_ROWS:
                raise SystemExit(f"refused: {d}/rows.jsonl are not the 300 LLM rows")
            llm["modes"][mode] = {"dir": str(d), "sha256": {n: sha256(d / n) for n in ("predictions.jsonl", "summary.json")}}
        else:
            llm["missing"].append(mode)

    if not a.dry_run and not old.is_dir():
        staging = repo / "data/.download-main-20261007b"
        download(DATA_REPO, DATA_REV, "rebuild-main-20261007b/main/*", staging, repo_type="dataset")
        (staging / "rebuild-main-20261007b/main").rename(old)
    elif a.dry_run and not old.is_dir():
        (repo / "data/.download-main-20261007b/rebuild-main-20261007b/main").rename(old)
    for name, digest in OLD_DATA.items():
        pins(old / name, digest)

    source = runs / ".download-baseline" / BASELINE
    if not a.dry_run:
        download(MODEL_REPO, MODEL_REV, f"{BASELINE}/*", runs / ".download-baseline")
    for name, digest in BASELINE_FILES.items():
        pins(source / name, digest)
    rewrite = baseline_copy(source, runs / "baseline-hf" / BASELINE, repo / "models/Qwen3-8B")
    base = runs / "baseline-hf" / BASELINE / STEP
    start = json.loads((base.parent / "run_manifest_start.json").read_text())
    training = start["resolved_config"]["training"]
    base_batch = start["world_size"] * training["gradient_accumulation_steps"] * training["batch_size"]

    every, first = cohorts(repo, a.out, old)
    pins(every, OLD_COHORT_ALL)
    train_rows = sum(1 for _ in open(new / "train.jsonl"))
    for x in (final, control):
        x["train_rows_in_file"] = train_rows

    role = {"final": "最终", "control": "对照"}
    provenance = [
        ["新数据 data/main-20261008a", "；".join(f"{n} {d[:12]}" for n, d in NEW_DATA.items()) + f"（两臂训练与本报告的验证/测试集；train.jsonl {train_rows} 行）"],
        ["旧数据 data/main-20261007b", f"{DATA_REPO}（dataset）@{DATA_REV[:12]} rebuild-main-20261007b/main；"
         + "；".join(f"{n} {d[:12]}" for n, d in OLD_DATA.items()) + "（基线的训练数据；消融的旧行与基线先验）"],
        ["基线", f"{MODEL_REPO}@{MODEL_REV[:12]} {BASELINE}/{STEP}（{len(BASELINE_FILES)} 个文件 sha256 固定）；{start['world_size']} 卡、"
         f"max_objects {start['resolved_config']['model']['max_objects']}、yaw_cls {start['resolved_config']['loss']['yaw_cls']}、全局批量 {base_batch}；训练 {training['steps']} 步 × {base_batch} = "
         f"{training['steps'] * base_batch / start['supervised_samples']:.3f} 轮（以过滤后 {start['supervised_samples']} 个有效样本计）"],
        ["基线 backbone 路径", f"使用副本 {base}：model_config.json 的 backbone 由 {rewrite['backbone_before']} 改为 {rewrite['backbone_after']}"
         + (f"（同一权重：config.json sha256 {rewrite['backbone_config_sha256'][:12]} = 检查点清单记录值）" if "backbone_config_sha256" in rewrite else "（未改写）")
         + f"；下载文件未改动，记录见 {base.parent}/BACKBONE_REWRITE.json"],
        ["基线的选择队列", f"按 Autopilot.eval_data 重建（旧验证集 random.Random(0) 打乱）：{every.name}（sha256 {sha256(every)[:12]}，"
         f"应为基线 select2 报告记录的 {OLD_COHORT_ALL[:12]}）与 {first.name}（前 3000 行）"],
        ["LLM 答案", f"{MODEL_REPO}@{MODEL_REV[:12]} llm-comparison/{{prompt,harness}}：gpt-6.1-sol（reasoning medium）在 300 行上的已有答案，"
         f"predictions sha256 prompt {LLM_FILES['prompt']['predictions.jsonl'][:12]}、harness {LLM_FILES['harness']['predictions.jsonl'][:12]}；"
         "本作业只用当前评测器重新评分，不调用任何 API"
         + (f"；缺失：{'、'.join(llm['missing'])}（其输出在无法访问的旧服务器上）" if llm["missing"] else "")],
        ["Hugging Face", f"HF_HOME = {os.environ.get('HF_HOME') or '未设置（默认 ~/.cache/huggingface）'}" + ("；演练：未下载" if a.dry_run else "")]]
    for x in (final, control):
        provenance.append([f"{role['final' if x is final else 'control']}臂 {x['dir']}",
                           f"运行 {x['run']}；yaw_cls {x['yaw_cls']}、max_objects {x['max_objects']}、全局批量 {x['global_batch']}；"
                           f"整个运行（配置）{x['steps']} 步 × {x['global_batch']} = {x['epochs_supervised']:.3f} 轮；select2 所选检查点 "
                           f"model-step-{x['best_step']} = {x['best_epochs']:.3f} 轮（均以过滤后 {x['supervised_samples']} 个有效样本计；"
                           f"train.jsonl {train_rows} 行，训练拒收 {x['rejected_samples']}）"])
    record = {"dry_run": a.dry_run, "repo": str(repo), "rule": RULE,
              "arms": [{**final, "role": "final"}, {**control, "role": "control"}],
              "baseline": {"repo": MODEL_REPO, "revision": MODEL_REV, "path": f"{BASELINE}/{STEP}", "checkpoint": str(base),
                           "download": str(source), "files": BASELINE_FILES, "backbone_rewrite": rewrite,
                           "steps": training["steps"], "global_batch": base_batch, "supervised_samples": start["supervised_samples"]},
              "data": {"new": {"path": str(new), "sha256": NEW_DATA}, "old": {"path": str(old), "sha256": OLD_DATA, "repo": DATA_REPO,
                                                                              "revision": DATA_REV}},
              "cohorts": {"new_select2": final["select2_cohort"], "old_select2": str(every), "old_select1": str(first)},
              "llm": llm, "pins": {"checked": pins.checked, "mismatches": pins.mismatches, "enforced": pins.enforce},
              "provenance": provenance}
    (a.out / "inputs.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    env = {"FINAL": final["best_checkpoint"], "FINAL_NAME": final["run"], "FINAL_DIR": final["dir"],
           "CTRL": control["best_checkpoint"], "CTRL_NAME": control["run"], "CTRL_DIR": control["dir"], "BASE": str(base),
           "COHORT_NEW": final["select2_cohort"], "COHORT_OLD": str(every), "COHORT_OLD1": str(first),
           "LLM_MODES": " ".join(llm["modes"])}
    (a.out / "inputs.env").write_text("".join(f"{k}={shlex.quote(v)}\n" for k, v in env.items()))
    print(json.dumps({"final": [final["dir"], final["best_score"]], "control": [control["dir"], control["best_score"]],
                      "llm_missing": llm["missing"], "pins_checked": pins.checked, "pin_mismatches": len(pins.mismatches)}))


if __name__ == "__main__":
    main()
