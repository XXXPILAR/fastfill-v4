#!/bin/bash
# FastFill v4, one-time Isambard-AI setup on a compute node (the login nodes have no GPU), from the v4 checkout root:
# kept apart from the v3 checkout: only the conda env is shared (linked from FASTFILL_ENV, default the earlier checkout's
# env/; made here when missing); Qwen3-8B is copied from that checkout when it has it, else downloaded; the
# fixed-object data from Hugging Face (sha256-pinned), RoomGenBench requests, the CPU test suite, then a GPU smoke of the
# E configuration (pose conditioning + support head, max_objects 256): train, resume, evaluate, remasking decode.
# Writes runs/isambard/v4-setup.ok on success. Rerunning skips what is already in place.
# FASTFILL_BACKBONE=Qwen3.5-27B (default Qwen3-8B) sets up that backbone instead: downloads it, installs
# flash-linear-attention (its Gated DeltaNet kernels) into the env, smokes the 27B E config, writes v4-setup-27b.ok.
#SBATCH --job-name=ff4-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=runs/isambard/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PIP_NO_CACHE_DIR=1 PYTHONPATH=$PWD
# V4's own Hugging Face login (XXXpilar), apart from V3's in ~/.cache/huggingface:
#   HF_HOME=$HOME/.cache/huggingface-v4 hf auth login
export HF_HOME=${FASTFILL_HF_HOME:-$HOME/.cache/huggingface-v4}; unset HF_TOKEN
OLD=${FASTFILL_OLD:-$PWD/../fastfill}
SHARED_ENV=${FASTFILL_ENV:-$OLD/env}
BACKBONE=${FASTFILL_BACKBONE:-Qwen3-8B}
case $BACKBONE in
  Qwen3-8B) REV=b968826d9c46dd6066d109eabc6255188de91218 CONFIG_SHA=f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
            TAG= SMOKE_CONFIG=main7-cell05-E-support-e5.json;;
  Qwen3.5-27B) REV=fc05daec18b0a78c049392ed2e771dde82bdf654 CONFIG_SHA=f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c
               TAG=-27b SMOKE_CONFIG=main7-cell05-E-support-e5-27b.json;;
  *) echo "unknown FASTFILL_BACKBONE $BACKBONE (Qwen3-8B or Qwen3.5-27B)"; exit 1;;
esac
rm -f runs/isambard/v4-setup$TAG.ok
uname -m; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
git submodule update --init RoomGenBench
mkdir -p models data runs/isambard

if [ ! -e env ]; then
  if [ -x "$SHARED_ENV/bin/python" ]; then
    ln -s "$SHARED_ENV" env && echo "env -> $SHARED_ENV"
  else
    ~/miniforge3/bin/conda create -y -p "$PWD/env" python=3.12 pip
    env/bin/pip install 'torch==2.13.0+cu126' --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
    env/bin/pip install -r fastfill/v2/requirements.txt -r fastfill/v2/requirements-roomgenbench.txt \
      transformers==5.14.1 peft==0.21.0 accelerate==1.15.0 scipy==1.15.2 pytest
  fi
fi
env/bin/pip freeze > runs/isambard/v4-env-freeze.txt
# the shared env must run this checkout's code, never the v3 checkout's (e.g. an editable install of it)
env/bin/python -c "import fastfill, os, sys; f = os.path.realpath(fastfill.__file__); print('fastfill from', f); sys.exit(not f.startswith(os.path.realpath(os.getcwd()) + '/'))"
env/bin/python -c "import platform, torch, transformers, peft, accelerate; assert torch.cuda.is_available(); print(platform.machine(), torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

H=env/bin/hf
$H auth whoami  # must be XXXpilar (the private dataset); fails here when V4 has no login
if [ ! -f models/$BACKBONE/model.safetensors.index.json ]; then
  if [ -f "$OLD/models/$BACKBONE/model.safetensors.index.json" ]; then
    cp -rL "$OLD/models/$BACKBONE" models/.copy-$BACKBONE && mv models/.copy-$BACKBONE models/$BACKBONE && echo "copied $OLD/models/$BACKBONE"
  else
    $H download Qwen/$BACKBONE --revision $REV --local-dir models/$BACKBONE
  fi
fi
echo "$CONFIG_SHA  models/$BACKBONE/config.json" | sha256sum -c
if [ "$BACKBONE" = Qwen3.5-27B ]; then
  # Triton kernels for the 48 Gated DeltaNet layers; without them transformers falls back to a slow torch loop
  env/bin/python -c "import fla" 2>/dev/null || env/bin/pip install flash-linear-attention==0.5.2 fla-core==0.5.2 ||
    echo "WARNING: flash-linear-attention did not install; Qwen3.5 runs on the torch fallback (slower)"
  env/bin/python -c "from transformers.utils.import_utils import is_flash_linear_attention_available as a; print('flash-linear-attention:', a())"
fi

D=data/v4fixed  # private dataset: needs V4's login above
if [ ! -f $D/train.jsonl ]; then
  $H download XXXpilar/fastfill-v4-v2-fixed --repo-type dataset --local-dir data/.download-v4fixed
  mv data/.download-v4fixed $D
fi
(cd $D && sha256sum -c) <<EOF
11d978325d91cd59a19e2a7a02f78b1afc923b96c40344eeaff413373498e60e  train.jsonl
ba8c26010f7cfe2b424821a977afff4e22c44b2b514b827ef80c1f4ed891eca3  validation.jsonl
bfaef332e40226660fd6663e9ad05b3454035e825d22236ed821aae3409e865b  test.jsonl
EOF
# made once here, so three autopilots exporting at the same time do not race on it
[ -d runs/roomgenbench/requests ] || env/bin/python -m fastfill.v2.roomgenbench \
  --requests-from RoomGenBench/bench/inputs/scenes --requests-out runs/roomgenbench/requests
# CPU suite, GPUs hidden (two-rank Gloo tests); any failure stops the setup, and the arms queued afterok never start
CUDA_VISIBLE_DEVICES= env/bin/python -m pytest fastfill/v2/tests -q

# GPU smoke: the E configuration on the 8 longest rows training accepts (most tokens within 8192, at most 256 objects),
# full length (no dropped constraints/supports, no minimal or category-only form); save, resume, evaluate, remask decode
S=runs/isambard/v4-smoke$TAG; rm -rf $S $S-resume $S-eval $S-remask
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
env/bin/python - "$PWD" "$BACKBONE" "$SMOKE_CONFIG" "$TAG" <<'EOF'
import json, sys
from fastfill.v2.autorun import next_config
from fastfill.v2.batch import load_tokenizer, tokenize_condition
root, backbone, smoke_config, tag = sys.argv[1:]
tokenizer, sized = load_tokenizer(f"models/{backbone}", local_files_only=True), []
for line in open("data/v4fixed/train.jsonl"):
    condition = json.loads(line)["condition"]
    if len(condition["objects"]) <= 256:
        tokens = len(tokenize_condition(condition, tokenizer)[0])
        if tokens <= 8192:
            sized.append((tokens, len(condition["objects"]), line))
sized.sort(key=lambda t: t[0])
open(f"runs/isambard/v4-longest{tag}.jsonl", "w").writelines(line for *_, line in sized[-8:])
print("smoke rows (tokens, objects):", [t[:2] for t in sized[-8:]], "of", len(sized), "eligible")
config = next_config(json.load(open(f"fastfill/v2/configs/{smoke_config}")), world_size=1, train_rows=8, epochs=1, global_batch=2)
config["model"].update(backbone=f"{root}/models/{backbone}")
config["training"].update(steps=4, checkpoint_every=2)  # 4 updates x 2 microbatches = the 8 rows once
config["augmentation"].update(drop_constraints_p=0, drop_support_p=0, category_only_description_p=0, minimal_form_p=0)
json.dump(config, open(f"runs/isambard/v4-smoke{tag}.json", "w"), indent=2)
EOF
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 1000 > $S-memory.csv & monitor=$!
T="env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 --module fastfill.v2.train --config runs/isambard/v4-smoke$TAG.json --data runs/isambard/v4-longest$TAG.jsonl"
$T --output $S
$T --output $S-resume --resume $S/state-step-2
env/bin/python -m fastfill.v2.evaluate --checkpoint $S-resume/model-step-4 --data $D/validation.jsonl \
  --max-samples 20 --projection minimal --device cuda --grid-decode spread --output $S-eval
env/bin/python -m fastfill.v2.ops.remask_ablation --checkpoint $S-resume/model-step-4 --data $D/validation.jsonl \
  --output $S-remask --modes argmax remask4LX --max-samples 5 --strip-support --device cuda
kill $monitor
grep -h '"step": 4,' $S/train.log $S-resume/train.log | cut -c1-90  # uninterrupted vs resumed step 4: about the same loss
sort -n $S-memory.csv | tail -1 | awk -F', ' '{print "smoke peak GPU memory", $1, "of", $2, "MiB"}'
rm -rf $S/state-step-* $S-resume/state-step-*  # ~29 GB each (they hold the frozen backbone)
touch runs/isambard/v4-setup$TAG.ok
echo SETUP OK
