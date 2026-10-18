#!/bin/bash
# The V3 models (../fastfill/runs/main4-cell05-main-20261008a-e5-{yawcls05,yawcls008}, read only) on the same 300 test
# rooms as V4 and GPT-5.6, decoded and scored by this checkout's code exactly like isambard_v4_test300.sbatch:
#   GPU0 V3 yawcls05: argmax spread remask1L      GPU1 V3 yawcls008: argmax spread remask1L
# (no pose conditioning in V3: one remask round only, as V4's base arm). Rooms over V3's capacity (128 objects) or
# context are failed requests. Each arm uses the checkpoint its autorun selected (STATUS.json) when recorded, else its
# newest model-step. Output runs/test300-v3/ (nothing under ../fastfill is written).
#   Usage: sbatch fastfill/v2/ops/isambard_v3_on_test300.sbatch
#SBATCH --job-name=ff3-test300
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --time=02:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V3=${FASTFILL_V3_RUNS:-$PWD/../fastfill/runs}
O=runs/test300-v3; mkdir -p $O
P=env/bin/python
T=fastfill/v2/ops/test300
pick() {  # the arm's selected checkpoint, else its newest model-step
  $P - "$V3" "$1" <<'EOF'
import glob, json, sys
root, arm = sys.argv[1:]
for status in glob.glob(f"{root}/autorun*/STATUS.json"):
    best = (json.load(open(status)).get("best") or {}).get("checkpoint") or ""
    if arm in best:
        print(best); sys.exit()
steps = sorted(glob.glob(f"{root}/main4-cell05-main-20261008a-e5-{arm}/model-step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
print(steps[-1] if steps else "")
EOF
}
pids=()
gpu=0
for arm in yawcls05 yawcls008; do
  ckpt=$(pick $arm)
  [ -n "$ckpt" ] || { echo "$arm: no checkpoint"; continue; }
  echo "V3 $arm: $ckpt"
  CUDA_VISIBLE_DEVICES=$gpu $P -m fastfill.v2.ops.remask_ablation --device cuda --checkpoint "$ckpt" --data $T/rows.jsonl \
    --output $O/test300_V3-$arm --modes argmax spread remask1L > $O/V3-$arm.log 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
failed=0
for j in "${pids[@]}"; do wait $j || failed=1; done
tail -n 5 $O/*.log
exit $failed
