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
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
export FASTFILL_BACKBONE=Qwen3-1.7B FASTFILL_DATA=data/v4detail
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
for arm in ${FASTFILL_ARMS:-full lora}; do
  case $arm in
    full) config=main7-cell05-detail-1.7b-full;; lora) config=main7-cell05-detail-1.7b;;
    *) echo "unknown arm $arm (full or lora)"; exit 1;;
  esac
  args=(fastfill/v2/ops/isambard_v4_autorun.sbatch autorun-v4-detail-1.7b-$arm fastfill/v2/configs/$config.json
        --name-suffix=-1.7b-$arm --global-batch=32)
  first=$(sbatch --parsable --job-name=ff4-detail-$arm ${dep:+--dependency=$dep} "${args[@]}")
  resume=$(sbatch --parsable --job-name=ff4-detail-$arm-resume --dependency=${dep:+$dep,}afternotok:$first "${args[@]}")
  echo "$arm: train=$first resume=$resume  status: runs/autorun-v4-detail-1.7b-$arm/STATUS.json"
done
squeue --me
