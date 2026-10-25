#!/bin/bash
# Placement-order comparison on Isambard-AI (experiments/run_orders.py), from the v4 checkout root: per room one
# GPT plan (shared), FastFill sizes (base arm, on the GPU), then A all at once / B one by one / C group by group
# (critic loop: doors hard, sizes fixed, up to 2 GPT repairs per loop; critic-judged polish), plus B_final / C_final
# (the same B / C placement followed by a whole-room repair with A's budget); at the end per-variant predictions,
# summary.json, quality.md (incremental and end-to-end cost and seconds, steps per room; layout_quality with the odd
# half's templates) and fastfill.v2.evaluate. Official OpenAI API only, key in fastfill_workflow/.env (never commit it).
# Resumable: resubmit and finished rooms are skipped and plans reused; a rerun whose settings or workflow code differ
# from $OUT/config.json stops (use another OUT). Budget in $OUT/ledger.json.
# Usage: sbatch fastfill_workflow/experiments/isambard_orders.sbatch
#   env: OUT, BUDGET_USD (default 60), WORKERS (default 8), CKPT (default: base arm's selected checkpoint),
#        ROOMS (default the test50 rooms file), EXTRA (more run_orders.py flags, e.g. "--limit-rooms 6" for a pilot,
#        "--batch 3", "--modes A C", "--no-final-repair")
# Pilot first:  EXTRA="--indices 15 22 29 170" OUT=runs/draft-workflow/orders-pilot sbatch fastfill_workflow/experiments/isambard_orders.sbatch
#SBATCH --job-name=ff4-orders
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
P=env/bin/python
[ -f fastfill_workflow/.env ] || { echo "missing fastfill_workflow/.env (OPENAI_API_KEY=...)"; exit 1; }
checkpoint() {  # the arm's selected checkpoint, else its newest model-step (as isambard_draft.sbatch)
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
ROOMS=${ROOMS:-fastfill_workflow/experiments/test50_rooms.json}
OUT=${OUT:-runs/draft-workflow/orders-test50}
[ -n "$CKPT" ] || { echo "no base checkpoint found (set CKPT=RUN/model-step-N)"; exit 1; }
mkdir -p "$OUT" runs/isambard
echo "checkpoint $CKPT rows $ROWS rooms $ROOMS out $OUT budget \$${BUDGET_USD:-60} workers ${WORKERS:-8} extra ${EXTRA:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv
$P fastfill_workflow/experiments/run_orders.py --rows "$ROWS" --rooms-file "$ROOMS" --checkpoint "$CKPT" --device cuda \
  --out "$OUT" --budget-usd "${BUDGET_USD:-60}" --workers "${WORKERS:-8}" --quality --evaluate ${EXTRA:-}
