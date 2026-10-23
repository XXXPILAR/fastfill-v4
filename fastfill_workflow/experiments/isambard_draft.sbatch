#!/bin/bash
# Draft workflow on Isambard-AI, from the v4 checkout root (this folder is <checkout>/fastfill_workflow/):
# ① GPT-5.6 plan, ② FastFill size (base arm, in-process on the GPU), ③ GPT-5.6 draft, ④ local repair, then
# fastfill.v2.evaluate on the rooms run. LLM calls go to the official OpenAI API only, with the key in
# fastfill_workflow/.env (create it on the server, never commit it):  OPENAI_API_KEY=sk-...
# A free preflight checks that this node reaches api.openai.com before the checkpoint loads. If compute nodes have no
# internet access, make the sizes here and run the workflow on a login node instead (it needs no GPU):
#   sbatch:  env/bin/python -m fastfill.v2.ops.remask_ablation --checkpoint $CKPT --data $ROWS --output runs/draft-sizes \
#              --modes argmax --device cuda
#   login:   env/bin/python fastfill_workflow/experiments/run_draft.py --rows $ROWS \
#              --sizes-file runs/draft-sizes/predictions_argmax.jsonl --out $OUT --budget-usd 55 --evaluate
# Resumable: resubmit and finished rooms are skipped; the budget is shared through $OUT/ledger.json.
# Usage: sbatch fastfill_workflow/experiments/isambard_draft.sbatch
#   env: ROWS (default the test300 rows), OUT, BUDGET_USD (default 55), WORKERS (rooms in parallel, default 8),
#        CKPT (default: base arm's selected checkpoint),
#        EXTRA (more run_draft.py flags, e.g. "--per-bin 6" or "--indices 8 13")
#SBATCH --job-name=ff4-draft
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
P=env/bin/python
[ -f fastfill_workflow/.env ] || { echo "missing fastfill_workflow/.env (OPENAI_API_KEY=...)"; exit 1; }
checkpoint() {  # the arm's selected checkpoint, else its newest model-step (as isambard_v4_ablation.sbatch)
  $P - "$1" <<'EOF'
import glob, json, sys
arm = sys.argv[1]
try:
    print(json.load(open(f"runs/autorun-v4-{arm}/STATUS.json"))["best"]["checkpoint"]); sys.exit()
except (OSError, KeyError, TypeError, ValueError):
    pass
steps = sorted(glob.glob(f"runs/main4-*-e5-{arm}*/model-step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
if steps:
    print(steps[-1])
EOF
}
CKPT=${CKPT:-$(checkpoint base)}
ROWS=${ROWS:-fastfill/v2/ops/test300/rows.jsonl}
OUT=${OUT:-runs/draft-workflow/test300}
[ -n "$CKPT" ] || { echo "no base checkpoint found (set CKPT=RUN/model-step-N)"; exit 1; }
mkdir -p "$OUT" runs/isambard
echo "checkpoint $CKPT rows $ROWS out $OUT budget \$${BUDGET_USD:-55} workers ${WORKERS:-8}"
nvidia-smi --query-gpu=name,memory.total --format=csv
$P fastfill_workflow/experiments/run_draft.py --rows "$ROWS" --checkpoint "$CKPT" --device cuda --out "$OUT" \
  --budget-usd "${BUDGET_USD:-55}" --workers "${WORKERS:-8}" --evaluate ${EXTRA:-}
