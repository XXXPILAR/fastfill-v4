#!/bin/bash
# GPU smoke of the every-room Qwen3-1.7B configurations on data/v4detail, one GH200, before their 4-GPU arms: for each
# CONFIG, 2 updates on the 4 longest training rooms (most tokens; the room with the most objects among them), then evaluate
# on the 2 longest test rooms (the longest, 19,399 tokens and 342 objects, included); every room must get a layout. Prints
# each configuration's peak GPU memory. Usage: sbatch fastfill/v2/ops/isambard_v4_detail_smoke.sbatch CONFIG...
#SBATCH --job-name=ff4-smoke-detail-all
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD HF_HOME=${FASTFILL_HF_HOME:-$HOME/.cache/huggingface-v4}; unset HF_TOKEN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
S=runs/isambard/v4-smoke-detail-all; rm -rf $S; mkdir -p $S
env/bin/python - "$PWD" "$S" "$@" <<'EOF'
import json, sys
from pathlib import Path
from fastfill.v2.autorun import next_config
from fastfill.v2.batch import load_tokenizer, tokenize_condition
root, out, *configs = sys.argv[1:]
tokenizer = load_tokenizer(f"{root}/models/Qwen3-1.7B", local_files_only=True)
def sized(path):
    rows = []
    for line in open(path):
        condition = json.loads(line)["condition"]
        rows.append((len(tokenize_condition(condition, tokenizer)[0]), len(condition["objects"]), line))
    return rows
train, test = sized("data/v4detail/train.jsonl"), sized("data/v4detail/test.jsonl")
picked = sorted(train, key=lambda t: t[0])[-3:]
most = max(train, key=lambda t: t[1])
picked += [most] if most not in picked else [sorted(train, key=lambda t: t[0])[-4]]
longest_test = sorted(test, key=lambda t: t[0])[-2:]
open(f"{out}/train-longest.jsonl", "w").writelines(line for *_, line in picked)
open(f"{out}/test-longest.jsonl", "w").writelines(line for *_, line in longest_test)
print("train rows (tokens, objects):", [t[:2] for t in picked], "test rows:", [t[:2] for t in longest_test], flush=True)
for path in configs:
    config = next_config(json.load(open(path)), world_size=1, train_rows=4, epochs=1, global_batch=2)
    config["model"].update(backbone=f"{root}/models/Qwen3-1.7B")
    config["training"].update(steps=2, checkpoint_every=2)  # 2 updates x 2 microbatches = the 4 rows once
    config["augmentation"].update(drop_constraints_p=0, drop_support_p=0, category_only_description_p=0, minimal_form_p=0)
    json.dump(config, open(f"{out}/{Path(path).stem}.json", "w"), indent=2)
EOF
for config in "$@"; do
  name=$(basename "$config" .json)
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > $S/$name-memory.csv & monitor=$!
  env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 --module fastfill.v2.train \
    --config $S/$name.json --data $S/train-longest.jsonl --output $S/$name
  env/bin/python -m fastfill.v2.evaluate --checkpoint $S/$name/model-step-2 --data $S/test-longest.jsonl \
    --projection full --device cuda --grid-decode spread --output $S/$name-eval
  kill $monitor
  env/bin/python -c "import json, sys; r = json.load(open(sys.argv[1])); print(sys.argv[2], 'test rooms', r['requests'], 'without a layout', r['inference_failed_requests']); sys.exit(r['inference_failed_requests'] > 0)" \
    $S/$name-eval/report.json $name
  sort -n $S/$name-memory.csv | tail -1 | awk -F', ' -v n=$name '{print n, "smoke peak GPU memory", $1, "of", $2, "MiB"}'
  rm -rf $S/$name/state-step-*  # fp32 weights and optimizer state, tens of GB
done
echo SMOKE OK
