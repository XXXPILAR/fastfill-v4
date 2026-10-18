#!/bin/bash
# FastFill v4 on the 300 test rooms GPT-5.6 answered (fastfill/v2/ops/test300), quickly, one GPU per arm:
#   GPU0 base: argmax spread remask1L        GPU1 D: argmax spread remask4L remask4LX
#   GPU2 E:    argmax spread remask4L remask4LX
# with GPT-5.6's layouts scored by the same evaluator, then runs/test300-v4/REPORT.md (v4_report.py). Independent of
# isambard_v4_ablation.sbatch (own output directory). The predictions_<mode>.jsonl files are the input of the hybrid
# workflow (ops/llm_hybrid.py), run where the API key is.  Usage: sbatch fastfill/v2/ops/isambard_v4_test300.sbatch
#SBATCH --job-name=ff4-test300
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=04:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=runs/test300-v4; mkdir -p $O
P=env/bin/python
T=fastfill/v2/ops/test300
best() { $P -c "import json,sys; print(json.load(open(f'runs/autorun-v4-{sys.argv[1]}/STATUS.json'))['best']['checkpoint'])" "$1"; }
A="$P -m fastfill.v2.ops.remask_ablation --device cuda --data $T/rows.jsonl"
pids=()
for spec in "0 base argmax spread remask1L" "1 D argmax spread remask4L remask4LX" "2 E argmax spread remask4L remask4LX"; do
  set -- $spec; gpu=$1 arm=$2; shift 2
  ckpt=$(best $arm) || { echo "$arm: no selected checkpoint"; continue; }
  echo "$arm: $ckpt"
  CUDA_VISIBLE_DEVICES=$gpu $A --checkpoint "$ckpt" --output $O/test300_$arm --modes "$@" > $O/$arm.log 2>&1 &
  pids+=($!)
done
rm -rf $O/test300_gpt56  # evaluate refuses an existing output
$P -m fastfill.v2.evaluate --data $T/rows.jsonl --predictions $T/gpt56_predictions.jsonl --output $O/test300_gpt56 > $O/gpt56.log 2>&1
failed=0
for j in "${pids[@]}"; do wait $j || failed=1; done
tail -n 5 $O/*.log
$P fastfill/v2/ops/v4_report.py $O && echo "report: $O/REPORT.md"
exit $failed
