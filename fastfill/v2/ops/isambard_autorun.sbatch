#!/bin/bash
# One FastFill v2 autopilot on a whole Isambard-AI node (4 GH200): train the baseline configuration on the round-10 data
# (max_objects 256, global batch 96), select on validation, upload, test, RoomGenBench export. No LLM calls.
# Usage: sbatch fastfill/v2/ops/isambard_autorun.sbatch <status dir under runs/> [autorun options, e.g. --set/--name-suffix]
# Submitting the same command again adopts the run and resumes it from its newest complete state.
#SBATCH --job-name=ff-autorun
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
dir=$1; shift
exec env/bin/python -m fastfill.v2.autorun --repo "$PWD" --data-root "$PWD/data" --autorun-dir "$dir" \
  --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --start-config fastfill/v2/configs/main7-cell05-main-20261007b-e5.json \
  --current-data data/main-20261008a --llm-env /nonexistent-no-llm-calls \
  --set "model.backbone=\"$PWD/models/Qwen3-8B\"" --set model.max_objects=256 "$@"
