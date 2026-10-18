#!/bin/bash
# FastFill v4 ablations on one node (4 GH200), after the arms: each arm's best checkpoint (STATUS.json), or, when an arm
# did not finish, its newest exported model-step (said in the log). Missing arms are skipped, not fatal.
# Decoded several ways on the same seeded sample of ROWS validation rows (default 1000), one GPU per arm:
#   GPU0  D: argmax spread remask1L remask4 remask4L remask4LX
#   GPU1  D: anchor_probe; then base: argmax spread remask1L (no pose conditioning: one round only)
#   GPU2  E: argmax spread remask4L remask4LX
#   GPU3  E without any support declaration (--strip-support): the support head's own accuracy
# and on the rooms GPT-5.6 answered, its layouts scored by the same evaluator: test300/ (300 test rooms, stratified by
# source, make_test300.py) and probe20/ (the first 20-room probe).
# Then runs/ablation-v4/REPORT.md (v4_report.py). Usage: sbatch fastfill/v2/ops/isambard_v4_ablation.sbatch
#SBATCH --job-name=ff4-ablation
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout's code, also with the env shared with v3
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=runs/ablation-v4; mkdir -p $O
P=env/bin/python
G=fastfill/v2/ops/probe20
T=fastfill/v2/ops/test300
checkpoint() {  # the arm's selected checkpoint, else its newest model-step, else nothing
  $P - "$1" <<'EOF'
import glob, json, sys
arm = sys.argv[1]
try:
    print(json.load(open(f"runs/autorun-v4-{arm}/STATUS.json"))["best"]["checkpoint"]); sys.exit()
except (OSError, KeyError, TypeError, ValueError):
    pass
steps = sorted(glob.glob(f"runs/main4-*-e5-{arm}*/model-step-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
if steps:
    print(f"{arm}: no selected checkpoint, using the newest {steps[-1]}", file=sys.stderr)
    print(steps[-1])
EOF
}
BASE=$(checkpoint base); D=$(checkpoint D); E=$(checkpoint E)
echo "base=${BASE:-missing} D=${D:-missing} E=${E:-missing}"
V=$O/validation-${ROWS:=1000}.jsonl  # seeded like the autopilot's samples: the file is grouped by source
$P - data/v4fixed/validation.jsonl $V $ROWS <<'EOF'
import random, sys
lines = [l.rstrip("\n") + "\n" for l in open(sys.argv[1]) if l.strip()]
random.Random(0).shuffle(lines)
open(sys.argv[2], "w").writelines(lines[:int(sys.argv[3])])
EOF
$P -m fastfill.v2.evaluate --data $G/rows.jsonl --predictions $G/gpt56_predictions.jsonl --output $O/probe20_gpt56 > $O/probe20_gpt56.log 2>&1
[ -f $T/gpt56_predictions.jsonl ] && $P -m fastfill.v2.evaluate --data $T/rows.jsonl --predictions $T/gpt56_predictions.jsonl \
  --output $O/test300_gpt56 > $O/test300_gpt56.log 2>&1
A="$P -m fastfill.v2.ops.remask_ablation --device cuda"
pids=()
if [ -n "$D" ]; then
  (CUDA_VISIBLE_DEVICES=0 $A --checkpoint $D --data $V --output $O/D --modes argmax spread remask1L remask4 remask4L remask4LX
   CUDA_VISIBLE_DEVICES=0 $A --checkpoint $D --data $T/rows.jsonl --output $O/test300_D --modes argmax spread remask4L remask4LX
   CUDA_VISIBLE_DEVICES=0 $A --checkpoint $D --data $G/rows.jsonl --output $O/probe20_D --modes argmax spread remask4L remask4LX) > $O/D.log 2>&1 &
  pids+=($!)
fi
(if [ -n "$D" ]; then CUDA_VISIBLE_DEVICES=1 $P -m fastfill.v2.anchor_probe --checkpoint $D --data $V --device cuda --output $O/anchor_probe_D.json; fi
 if [ -n "$BASE" ]; then
   CUDA_VISIBLE_DEVICES=1 $A --checkpoint $BASE --data $V --output $O/base --modes argmax spread remask1L
   CUDA_VISIBLE_DEVICES=1 $A --checkpoint $BASE --data $T/rows.jsonl --output $O/test300_base --modes argmax spread remask1L
   CUDA_VISIBLE_DEVICES=1 $A --checkpoint $BASE --data $G/rows.jsonl --output $O/probe20_base --modes argmax spread remask1L
 fi) > $O/probe_base.log 2>&1 &
pids+=($!)
if [ -n "$E" ]; then
  (CUDA_VISIBLE_DEVICES=2 $A --checkpoint $E --data $V --output $O/E --modes argmax spread remask4L remask4LX
   CUDA_VISIBLE_DEVICES=2 $A --checkpoint $E --data $T/rows.jsonl --output $O/test300_E --modes argmax spread remask4L remask4LX
   CUDA_VISIBLE_DEVICES=2 $A --checkpoint $E --data $G/rows.jsonl --output $O/probe20_E --modes argmax spread remask4L remask4LX) > $O/E.log 2>&1 &
  pids+=($!)
  CUDA_VISIBLE_DEVICES=3 $A --checkpoint $E --data $V --output $O/E_nosupport --modes argmax spread remask4L remask4LX --strip-support \
    > $O/E_nosupport.log 2>&1 &
  pids+=($!)
fi
failed=0
for j in "${pids[@]}"; do wait $j || failed=1; done
tail -n 12 $O/*.log
$P fastfill/v2/ops/v4_report.py $O && echo "report: $O/REPORT.md"
# figures (training curves, metric bars vs GPT-5.6, per-source, rooms): optional, the env may lack matplotlib;
# otherwise copy runs/main4-*/training_log.json and runs/ablation-v4 to a machine with it and run the same command
$P -c "import matplotlib" 2>/dev/null && $P -m fastfill.v2.ops.v4_plots all --out $O/plots > $O/plots.log 2>&1 \
  && echo "figures: $O/plots" || echo "figures skipped (see $O/plots.log, or no matplotlib)"
exit $failed
