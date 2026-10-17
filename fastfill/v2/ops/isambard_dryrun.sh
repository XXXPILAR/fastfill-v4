#!/bin/bash
# CPU end-to-end dry run of the Isambard final analysis (run_plan.sh all) in a throwaway checkout layout, no GPU,
# no downloads, no API calls:
#   fastfill/v2/ops/isambard_dryrun.sh WORK PYTHON NEW_DATA_DIR OLD_DATA_DIR LLM_DIR ROOMGENBENCH_DIR
# NEW/OLD_DATA_DIR: rebuild-main-20261008a/main and rebuild-main-20261007b/main; LLM_DIR: llm-comparison/ as on Hugging
# Face (prompt/, harness/); ROOMGENBENCH_DIR: a RoomGenBench checkout (read only).
# WORK/checkout gets: a copy of this repository's fastfill, env/bin/python -> PYTHON, RoomGenBench -> ROOMGENBENCH_DIR;
# seeded small row subsets of the real data (new validation and old validation: the same scenes; two rooms over 128
# objects in new validation and test); the real 300 LLM rows and answers (staged where the download would put them);
# both arms produced by the real Autopilot (phases E-F, --gpus 0,1,2,3 as on Isambard) from a copy of the baseline
# configuration with a tiny backbone, patched only: evaluate --device cuda -> cpu, training in one process instead of
# torch.distributed.run (same command and config; its rendezvous needs the network), no Hugging Face upload, shorter
# sleeps; a tiny baseline stand-in (max_objects 128) trained on the old rows, its model-step renamed to 6357,
# staged as the baseline download. Then DRYRUN=1 DEVICE=cpu ABLATION_ROWS=12 run_plan.sh all.
set -euo pipefail
[ $# = 6 ] || { sed -n '2,14p' "$0"; exit 2; }
abs() { case $1 in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }  # the script cds below: every path made absolute
WORK=$(abs "$1"); PY=$(abs "$(command -v "$2")"); NEWD=$(abs "$3"); OLDD=$(abs "$4"); LLM=$(abs "$5"); RGB=$(abs "$6")
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
CK=$WORK/checkout
[ ! -e $WORK ] || [ -d $CK ] || { echo "refused: $WORK exists and is not an earlier dry run (it is replaced)"; exit 2; }
rm -rf $WORK; mkdir -p $CK/env/bin $CK/data/main-20261008a $CK/data/.download-main-20261007b/rebuild-main-20261007b/main \
  $CK/runs/llm-prompt-300 $CK/runs/.download-llm/llm-comparison $CK/runs/.download-baseline $CK/runs/isambard
cp -R $REPO/fastfill $CK/fastfill  # a copy: io.run_metadata needs the code under the checkout root
ln -s $PY $CK/env/bin/python; ln -s $RGB $CK/RoomGenBench
cd $CK
export PYTHONPATH=$CK PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
P=$CK/env/bin/python

# (1) seeded subsets: small rooms (tiny-tokenizer condition <= 30000 tokens, <= 40 objects) plus two > 128-object rooms
$P - $NEWD $OLDD <<'EOF'
import json, random, sys
from fastfill.v2.batch import TinyTokenizer, tokenize_condition
new, old = sys.argv[1:]
tok = TinyTokenizer()
def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]
def small(r):
    return len(r["condition"]["objects"]) <= 40 and len(tokenize_condition(r["condition"], tok)[0]) <= 30000
def pick(items, k, rng, ok):
    order = list(range(len(items))); rng.shuffle(order)
    return [i for i in order if ok(items[i])][:k]
def write(path, items):
    open(path, "w").writelines(json.dumps(r) + "\n" for r in items)
rng = random.Random(0)
nv, ov = rows(f"{new}/validation.jsonl"), rows(f"{old}/validation.jsonl")
twin = {r["provenance"]["scene_id"]: r for r in ov}
both = [r for r in nv if r["provenance"]["scene_id"] in twin]
chosen = sorted(pick(both, 46, rng, lambda r: small(r) and small(twin[r["provenance"]["scene_id"]]))
                + pick(both, 2, rng, lambda r: len(r["condition"]["objects"]) > 128))
write("data/main-20261008a/validation.jsonl", [both[i] for i in chosen])
write("data/.download-main-20261007b/rebuild-main-20261007b/main/validation.jsonl", [twin[both[i]["provenance"]["scene_id"]] for i in chosen])
nt = rows(f"{new}/test.jsonl")
write("data/main-20261008a/test.jsonl", [nt[i] for i in sorted(pick(nt, 40, rng, small) + pick(nt, 2, rng, lambda r: len(r["condition"]["objects"]) > 128))])
for src, dst, k in ((f"{new}/train.jsonl", "data/main-20261008a/train.jsonl", 60),
                    (f"{old}/train.jsonl", "data/.download-main-20261007b/rebuild-main-20261007b/main/train.jsonl", 60),
                    (f"{old}/test.jsonl", "data/.download-main-20261007b/rebuild-main-20261007b/main/test.jsonl", 20)):
    head = []
    for line in open(src):  # the first 3000 rows are enough for a seeded pick
        head.append(json.loads(line))
        if len(head) == 3000:
            break
    write(dst, [head[i] for i in sorted(pick(head, k, rng, small))])
print("subsets written")
EOF

# (2) LLM rows (as setup leaves them) and answers (where the download would put them); RoomGenBench requests (setup)
cp $LLM/prompt/rows.jsonl runs/llm-prompt-300/rows.jsonl
cp -R $LLM/prompt $LLM/harness runs/.download-llm/llm-comparison/
$P -m fastfill.v2.roomgenbench --requests-from RoomGenBench/bench/inputs/scenes --requests-out runs/roomgenbench/requests > /dev/null

# (3) the baseline configuration with a tiny backbone (max_length 32768: the byte tokenizer needs it for the 300 LLM rows
#     and the five RoomGenBench rooms)
$P - <<'EOF'
import json
c = json.load(open("fastfill/v2/configs/main7-cell05-main-20261007b-e5.json"))
c["model"]["backbone"] = "tiny"
c["training"].update(cpu=True, mixed_precision="no", max_length=32768, cpu_threads=2)
json.dump(c, open("../tiny-start.json", "w"), indent=2)
b = json.loads(json.dumps(c))
b["training"].update(steps=4, gradient_accumulation_steps=4, checkpoint_every=0, validate_every=0)
b["optimizer"]["warmup_steps"] = 0
json.dump(b, open("../tiny-baseline.json", "w"), indent=2)
EOF

# (4) both arms: the real Autopilot, flags as isambard_autorun.sbatch (no backbone path), checkpoints every 2 steps
cat > ../autopilot_cpu.py <<'EOF'
import sys, time
from fastfill.v2 import autorun
sleep = time.sleep
time.sleep = lambda s: sleep(min(s, 2))
run = autorun.Autopilot.sh
def sh(self, args, **kw):
    args = ["cpu" if str(a) == "cuda" else str(a) for a in args]
    if "torch.distributed.run" in args:  # the same training command in one process (no rendezvous needed)
        i = args.index("torch.distributed.run")
        args = args[:i] + args[i + 4:]  # drops torch.distributed.run --standalone --nproc_per_node=N --module
    return run(self, args, **kw)
autorun.Autopilot.sh = sh
autorun.Autopilot.upload = lambda self, checkpoint, name: self.log("dry run: no Hugging Face upload", checkpoint=checkpoint)
autorun.main(sys.argv[1:])
EOF
arm() {
  $P ../autopilot_cpu.py --repo $CK --data-root $CK/data --autorun-dir $1 --gpus 0,1,2,3 --start-config ../tiny-start.json \
    --current-data data/main-20261008a --llm-env /nonexistent-no-llm-calls --set model.max_objects=256 \
    --set training.checkpoint_every=2 "${@:2}" > runs/isambard/$1.out 2>&1
}
arm autorun-yawcls05 --name-suffix=-yawcls05 --set loss.yaw_cls=0.5 & a=$!
arm autorun-yawcls008 --name-suffix=-yawcls008 & b=$!
wait $a; wait $b
for d in autorun-yawcls05 autorun-yawcls008; do $P -c "import json; s=json.load(open('runs/$d/STATUS.json')); print('$d', s['phase'], s['best']['score'], s.get('failures'))"; done

# (5) the baseline stand-in, staged as the baseline download
B=runs/.download-baseline/main7-cell05-main-20261007b-e5
$P -m fastfill.v2.train --config ../tiny-baseline.json --data data/.download-main-20261007b/rebuild-main-20261007b/main/train.jsonl \
  --output $B > runs/isambard/baseline-standin.log 2>&1
mv $B/model $B/model-step-6357  # the final export (checkpoint_every 0)

# (6) the analysis
export DRYRUN=1 DEVICE=cpu ABLATION_ROWS=12
unset CUDA_VISIBLE_DEVICES
bash fastfill/v2/ops/run_plan.sh all
ls -la runs/analysis/final/report.html
