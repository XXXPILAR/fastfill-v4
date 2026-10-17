# fastfill/v2/ops — operations and final-analysis scripts (2026-10-08)

Server-side scripts used to rebuild data, run the autopilot and analyse the final model. The isambard_* scripts and
run_plan.sh run from any checkout; the others assume the old server checkout `/home/jovyan/shanliantian/fastfill`.
Nothing here holds credentials — the LLM key lived only in `/home/jovyan/shanliantian/.fastfill_api.env` on that server.

Operations
- `rebuild_and_publish.sh <name>` (Mac): full data chain (legacy_build → legacy_verify → review_data → qualified_data
  → multisource_verify) from a pinned worktree (`SRC=`), HF upload, `NEXT_DATA.json`; `NO_HANDOFF=1` holds the server
  handoff until the reports are checked. Stops at the first failing verifier (set -euo pipefail).
- `start_autorun_retrain.sh`: retrain on new data with logged `--set` overrides (`model.max_objects=256`,
  `loss.yaw_cls=0.5`), `--name-suffix`, LLM step disabled (no paid calls).
- `run_checkpoint.sh <model-step> <tag>`: five-room RoomGenBench hand-off on CPU with `--require-placement`
  (the autopilot calls it; runs from any checkout). It rebuilds `runs/roomgenbench/requests` from the scenes and builder
  on disk each time (a differing old directory is set aside as `requests.stale-*`) and records them in `<tag>/requests.sha256`.
- Isambard-AI (aarch64 GH200, Slurm): from a checkout under `$PROJECTDIR`, `bash fastfill/v2/ops/isambard_submit.sh`
  submits `isambard_setup.sbatch` (conda env with the baseline's package versions, pinned data, backbone and LLM rows,
  tests) and, once it succeeds, two `isambard_autorun.sbatch` jobs (one node each): yaw_cls 0.5 and the 0.08 control,
  both from `configs/main7-cell05-main-20261007b-e5.json` (the baseline's configuration) with `max_objects` 256.
  Status in `runs/autorun-yawcls05/` and `runs/autorun-yawcls008/`, job logs in `runs/isambard/`. Resubmitting an
  autorun job resumes it.
- `run_structured.sh`, `rescore_llm.sh`: the OptiScene-style LLM modes on the frozen 300 rows and re-scoring of saved
  LLM predictions with the current evaluator (no API calls); `run_reference.sh`: reference check of a 5-room export.

Final analysis on Isambard-AI (after both autopilot arms; one node, 4 GPUs, at most 12 h)
- First `sacct -j <setup>,<yawcls05>,<yawcls008> -o JobID,JobName,State,ExitCode` and `git log -1`: the setup of
  2cd4f47 fails its test step on Isambard (fixed in 3806325), and arms queued with afterok on a failed setup never
  start; then pull main and run `isambard_submit.sh` again. Pull only while no arm's STATUS.json phase is F-* (a change
  of a fastfill/v2/*.py file during phase F stops it). Submit from the checkout with the job ids of the arms that will actually run (once
  both `runs/autorun-yawcls05/STATUS.json` and `runs/autorun-yawcls008/STATUS.json` say done / done-with-failures the
  dependency can be dropped):
  `sbatch --dependency=afterok:<yawcls05 job>:<yawcls008 job> fastfill/v2/ops/isambard_analysis.sbatch`
  Unattended only if both arm jobs exit 0 within their 24 h: an arm that fails or times out (no automatic resubmission;
  resubmitting resumes it) leaves the analysis job pending; after resubmitting that arm run
  `scontrol update JobId=<ff-analysis job> Dependency=afterok:<new ids>` (or scancel it and submit again).
  It runs `run_plan.sh all` (inputs → gpu → cpu → report) from the checkout; the report is
  `runs/analysis/final/report.html`, the job log `runs/isambard/ff-analysis-<job>.out`, step times
  `runs/analysis/walltime.tsv`. Resubmitting resumes (finished outputs are kept; all of them are redone if the chosen
  checkpoints changed). Single steps by hand only inside an allocation (`srun --nodes=1 --gpus=4 ...`), not on a login node.
- `isambard_inputs.py` (step inputs): refuses unless both arms are done / done-with-failures with the same select2 cohort
  bytes scored by the same fastfill/v2 code (implementation_sha256); final = the arm with the lower select2 score
  (STATUS best.score), control = the other. Downloads with the submitting shell's HF_HOME / login, every file sha256-pinned: the old baseline (`liantian/fastfill-v2-models`, used from
  `runs/baseline-hf/` whose model_config.json names this checkout's `models/Qwen3-8B`, recorded in BACKBONE_REWRITE.json),
  the saved LLM answers (prompt, harness → `runs/llm-{prompt,harness}-300/`; the structured modes only if
  `runs/llm-structured*-300` exist, else the report says they are missing) and the old data `data/main-20261007b`.
  Rebuilds the baseline's old selection cohorts with `Autopilot.eval_data`. Writes `runs/analysis/inputs.json`.
- The report compares final vs baseline (≤ 128-object rooms), final vs control (the evidence for the choice), spread vs
  argmax per model, final vs the LLM answers re-scored by the evaluator on disk (no API calls), the ablations and the
  RoomGenBench reference checks; validation decides, test is report-only.
- `isambard_dryrun.sh WORK PYTHON NEW_DATA OLD_DATA LLM_DIR ROOMGENBENCH`: CPU end-to-end dry run of the whole job in a
  throwaway checkout (tiny-backbone stand-ins, both arms made by the real autopilot, `DRYRUN=1`: no downloads).

Analysis scripts (`make_sample.sh` is the old server's CPU sample test of 2026-10-08, kept as a record)
- `stratify.py`: per-object errors split by target selection rule (old vs added targets), place, source and object
  count, reproducing `evaluate`'s report exactly; validator collision / clean-room rates per stratum.
- `compare.py`: paired room-level A−B (bootstrap CI of the mean per-room difference as the primary criterion, Holm
  sign test as robustness); refuses different rows or different decoding code.
- `llm_compare.py`: FastFill (raw / spread) vs the four LLM modes on the frozen 300 rows; transport failures,
  latency and token columns.
- `sample_validation.py`, `diag_ablation.py`, `selftest_shift.py`, `crosscheck.py`, `report.py` (self-contained HTML).
Decisions (spread vs argmax, replacing the baseline) read validation comparisons only; test sections are report-only.
