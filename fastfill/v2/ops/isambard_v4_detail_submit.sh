#!/bin/bash
# Isambard-AI, from the v4 checkout root on a login node: Qwen3-1.7B on the detailed-description subset data/v4detail
# (SAGE-10k, IL3D_3dfront, IL3D_synthetic: 29,611 train rooms), full fine-tuning against LoRA, one node each, same data,
# 5 epochs at global batch 32 (~4,600 updates; batch 96 would give only ~1,500), category-only descriptions 0.2:
#   full  main7-cell05-detail-1.7b-full   whole backbone, lr 1e-5 on fp32 master weights; decoder lr 1e-4
#   lora  main7-cell05-detail-1.7b        LoRA r8 / alpha 16 on q/k/v/o, lr 1e-4
# Without runs/isambard/v4-setup-1.7b.ok the Qwen3-1.7B setup job runs first (its GPU smoke trains the full fine-tuning
# configuration). Without data/v4detail/manifest.json a one-GPU job cuts it from data/v4fixed (make_source_subset.py) and
# checks it against fastfill/v2/ops/v4detail.sha256 (the local build); the arms wait for both (afterok) and get one
# continuation job each (afternotok, and afterok on the same prerequisite: Slurm counts a job cancelled for a failed
# prerequisite as not ok, so afternotok alone would start the continuation then).
#   fastfill/v2/ops/isambard_v4_detail_submit.sh                  # both arms
#   FASTFILL_ARMS=full fastfill/v2/ops/isambard_v4_detail_submit.sh  # one arm
# Status: runs/autorun-v4-detail-1.7b-<arm>/STATUS.json; the run is runs/main4-cell05-v4detail-e5-1.7b-<arm>.
# Those configurations leave out the rooms over 8192 tokens or 256 objects (363 train, 21 validation, 19 test rooms, all
# SAGE-10k) and 8 train rooms with a size below the size head's range. FASTFILL_VARIANT=all trains and evaluates every room:
# configurations *-all*.json (max_length 32768 = Qwen3's context window, max_objects 512, size_log_limit 12, and
# training.skip_rejected_rows false, so a room the preflight would leave out stops the run); a one-GPU smoke on the longest
# rooms (isambard_v4_detail_smoke.sbatch) runs before the arms. Status runs/autorun-v4-detail-1.7b-all-<arm>:
#   FASTFILL_VARIANT=all fastfill/v2/ops/isambard_v4_detail_submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
export FASTFILL_BACKBONE=Qwen3-1.7B FASTFILL_DATA=data/v4detail
case ${FASTFILL_VARIANT:-} in
  "") v=;; all) v=-all;;
  *) echo "unknown FASTFILL_VARIANT $FASTFILL_VARIANT (empty or all)"; exit 1;;
esac
configs=()
for arm in ${FASTFILL_ARMS:-full lora}; do
  case $arm in
    full) configs+=(fastfill/v2/configs/main7-cell05-detail-1.7b$v-full.json);; lora) configs+=(fastfill/v2/configs/main7-cell05-detail-1.7b$v.json);;
    *) echo "unknown arm $arm (full or lora)"; exit 1;;
  esac
done
dep=  # afterok:<job> the arms wait for, when setup or the subset still has to run
if [ ! -f runs/isambard/v4-setup-1.7b.ok ]; then
  setup=$(sbatch --parsable --time=03:00:00 --job-name=ff4-setup-1.7b fastfill/v2/ops/isambard_v4_setup.sbatch)
  dep=afterok:$setup
  echo "setup=$setup ($FASTFILL_BACKBONE)"
fi
if [ ! -f data/v4detail/manifest.json ]; then
  rm -rf data/v4detail.tmp
  subset=$(sbatch --parsable --job-name=ff4-subset-detail --nodes=1 --gpus=1 --time=00:30:00 \
    --output=runs/isambard/%x-%j.out ${dep:+--dependency=$dep} --wrap "set -e; export PYTHONPATH=\$PWD
env/bin/python fastfill/v2/ops/make_source_subset.py --data data/v4fixed --output data/v4detail
cd data/v4detail && sha256sum -c ../../fastfill/v2/ops/v4detail.sha256")
  dep=afterok:$subset
  echo "subset=$subset (data/v4detail)"
fi
if [ -n "$v" ]; then
  smoke=$(sbatch --parsable ${dep:+--dependency=$dep} fastfill/v2/ops/isambard_v4_detail_smoke.sbatch "${configs[@]}")
  dep=afterok:$smoke
  echo "smoke=$smoke (the longest rooms)"
fi
arms=(${FASTFILL_ARMS:-full lora})
for i in "${!arms[@]}"; do
  arm=${arms[$i]}
  args=(fastfill/v2/ops/isambard_v4_autorun.sbatch autorun-v4-detail-1.7b$v-$arm ${configs[$i]}
        --name-suffix=-1.7b$v-$arm --global-batch=32)
  first=$(sbatch --parsable --job-name=ff4-detail$v-$arm ${dep:+--dependency=$dep} "${args[@]}")
  resume=$(sbatch --parsable --job-name=ff4-detail$v-$arm-resume --dependency=${dep:+$dep,}afternotok:$first "${args[@]}")
  echo "$arm: train=$first resume=$resume  status: runs/autorun-v4-detail-1.7b$v-$arm/STATUS.json"
done
squeue --me
