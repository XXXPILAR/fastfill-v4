#!/bin/bash
# FastFill v4 ablations on one node (4 GH200), after the arms are done: the best checkpoint of each arm (its STATUS.json)
# decoded several ways on the same seeded sample of ROWS validation rows (default 1000), one GPU per job:
#   GPU0  D: argmax spread remask1L remask4 remask4L remask4LX
#   GPU1  D: anchor_probe; then base: argmax spread remask1L (no pose conditioning: one round only)
#   GPU2  E: argmax spread remask4L remask4LX
#   GPU3  E without any support declaration (--strip-support): the support head's own accuracy
# Outputs in runs/ablation-v4/. Usage: sbatch fastfill/v2/ops/isambard_v4_ablation.sbatch
#SBATCH --job-name=ff4-ablation
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout's code, also with the env shared with v3
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=runs/ablation-v4; mkdir -p $O
P=env/bin/python
best() { $P -c 'import json, sys; s = json.load(open(sys.argv[1])); print(s["best"]["checkpoint"])' runs/autorun-v4-$1/STATUS.json; }
BASE=$(best base); D=$(best D); E=$(best E)
echo "base=$BASE D=$D E=$E"
V=$O/validation-${ROWS:=1000}.jsonl  # seeded like the autopilot's samples: the file is grouped by source
$P - data/v4fixed/validation.jsonl $V $ROWS <<'EOF'
import random, sys
lines = [l.rstrip("\n") + "\n" for l in open(sys.argv[1]) if l.strip()]
random.Random(0).shuffle(lines)
open(sys.argv[2], "w").writelines(lines[:int(sys.argv[3])])
EOF
A="$P -m fastfill.v2.ops.remask_ablation --data $V --device cuda"
CUDA_VISIBLE_DEVICES=0 $A --checkpoint $D --output $O/D --modes argmax spread remask1L remask4 remask4L remask4LX > $O/D.log 2>&1 & j0=$!
(CUDA_VISIBLE_DEVICES=1 $P -m fastfill.v2.anchor_probe --checkpoint $D --data $V --device cuda --output $O/anchor_probe_D.json \
  && CUDA_VISIBLE_DEVICES=1 $A --checkpoint $BASE --output $O/base --modes argmax spread remask1L) > $O/probe_base.log 2>&1 & j1=$!
CUDA_VISIBLE_DEVICES=2 $A --checkpoint $E --output $O/E --modes argmax spread remask4L remask4LX > $O/E.log 2>&1 & j2=$!
CUDA_VISIBLE_DEVICES=3 $A --checkpoint $E --output $O/E_nosupport --modes argmax spread remask4L remask4LX --strip-support \
  > $O/E_nosupport.log 2>&1 & j3=$!
failed=0
for j in $j0 $j1 $j2 $j3; do wait $j || failed=1; done
tail -n 12 $O/*.log
exit $failed
