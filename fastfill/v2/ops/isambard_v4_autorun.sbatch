#!/bin/bash
# One FastFill v4 arm on a whole Isambard-AI node (4 GH200): the autopilot trains CONFIG on data/v4fixed for 5 epochs at
# global batch 96, selects on validation, uploads the best model (private, FASTFILL_MODEL_REPO), tests, exports RoomGenBench.
# No LLM calls. Usage: sbatch fastfill/v2/ops/isambard_v4_autorun.sbatch <status dir under runs/> <CONFIG> [autorun options]
# Submitting the same command again adopts the run and resumes it from its newest complete state.
#SBATCH --job-name=ff4-autorun
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout's code, also with the env shared with v3
# V4's own Hugging Face login (XXXpilar), apart from V3's in ~/.cache/huggingface:
#   HF_HOME=$HOME/.cache/huggingface-v4 hf auth login
export HF_HOME=${FASTFILL_HF_HOME:-$HOME/.cache/huggingface-v4}; unset HF_TOKEN
dir=$1; config=$2; shift 2
exec env/bin/python -m fastfill.v2.autorun --repo "$PWD" --data-root "$PWD/data" --autorun-dir "$dir" \
  --gpus "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" --start-config "$config" --current-data data/v4fixed \
  --llm-env /nonexistent-no-llm-calls --model-repo "${FASTFILL_MODEL_REPO:-XXXpilar/fastfill-v4-models}" \
  --set "model.backbone=\"$PWD/models/Qwen3-8B\"" "$@"
