#!/bin/bash
# FastFill v2 final analysis on one Isambard-AI node (4 GH200) once both autopilot arms have finished: final arm (lower
# select2 score) vs control arm vs the old baseline, the saved LLM answers re-scored, one HTML report:
# runs/analysis/final/report.html. From the checkout, with the job ids of the arms that will actually run (check
# `sacct -j <setup>,<yawcls05>,<yawcls008> -o JobID,JobName,State,ExitCode` first: arms behind a failed setup never start):
#   sbatch --dependency=afterok:<yawcls05 job>:<yawcls008 job> fastfill/v2/ops/isambard_analysis.sbatch
# afterok holds only for an arm job that exits 0 (phase done / done-with-failures); an arm that fails or hits its 24 h
# limit leaves this job pending: after resubmitting that arm, `scontrol update JobId=<this job> Dependency=afterok:<ids>`.
# Steps and outputs: fastfill/v2/ops/run_plan.sh (it refuses unless both STATUS.json phases are done or
# done-with-failures). Resubmitting resumes: finished steps are kept.
#SBATCH --job-name=ff-analysis
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
exec bash fastfill/v2/ops/run_plan.sh all
