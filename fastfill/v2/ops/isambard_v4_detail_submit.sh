#!/bin/bash
# Isambard-AI, from the v4 checkout root on a login node: Qwen3-1.7B (or FASTFILL_BACKBONE=Qwen3-4B) on the
# detailed-description subset data/v4detail (SAGE-10k, IL3D_3dfront, IL3D_synthetic: 29,611 train rooms), full fine-tuning
# against LoRA, one node each, same data, 5 epochs at global batch 32 (~4,600 updates; batch 96 would give only ~1,500),
# category-only descriptions 0.2:
#   full  main7-cell05-detail-<tag>-full   whole backbone, lr 1e-5 on fp32 master weights; decoder lr 1e-4
#   lora  main7-cell05-detail-<tag>        LoRA r8 / alpha 16 on q/k/v/o, lr 1e-4
# <tag> is 1.7b or 4b. Without runs/isambard/v4-setup-<tag>.ok that backbone's setup job runs first. Without
# data/v4detail/manifest.json a one-GPU job cuts it from data/v4fixed (make_source_subset.py) and checks it against
# fastfill/v2/ops/v4detail.sha256 (the local build); the arms wait for both (afterok) and get one continuation job each
# (afternotok, and afterok on the same prerequisite: Slurm counts a job cancelled for a failed prerequisite as not ok, so
# afternotok alone would start the continuation then).
#   fastfill/v2/ops/isambard_v4_detail_submit.sh                  # both arms
#   FASTFILL_ARMS=full fastfill/v2/ops/isambard_v4_detail_submit.sh  # one arm
# Status: runs/autorun-v4-detail-<tag>-<arm>/STATUS.json; the run is runs/main4-cell05-v4detail-e5-<tag>-<arm>.
# Those 1.7B configurations leave out the rooms over 8192 tokens or 256 objects (363 train, 21 validation, 19 test rooms,
# all SAGE-10k) and 8 train rooms with a size below the size head's range. FASTFILL_VARIANT=all trains and evaluates every
# room: configurations *-all*.json (max_length 32768 = Qwen3's context window, max_objects 512, size_log_limit 12, and
# training.skip_rejected_rows false, so a room the preflight would leave out stops the run); each arm first gets a two-GPU
# smoke on the longest rooms (isambard_v4_detail_smoke.sbatch), so an arm that does not fit does not hold up the other.
# Status runs/autorun-v4-detail-<tag>-all-<arm>; Qwen3-4B has only the every-room configurations:
#   FASTFILL_VARIANT=all fastfill/v2/ops/isambard_v4_detail_submit.sh
#   FASTFILL_BACKBONE=Qwen3-4B FASTFILL_VARIANT=all fastfill/v2/ops/isambard_v4_detail_submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
export FASTFILL_BACKBONE=${FASTFILL_BACKBONE:-Qwen3-1.7B} FASTFILL_DATA=data/v4detail
case $FASTFILL_BACKBONE in
  Qwen3-1.7B) tag=1.7b;; Qwen3-4B) tag=4b;;
  *) echo "unknown FASTFILL_BACKBONE $FASTFILL_BACKBONE (Qwen3-1.7B or Qwen3-4B)"; exit 1;;
esac
case ${FASTFILL_VARIANT:-} in
  "") v=;; all) v=-all;;
  *) echo "unknown FASTFILL_VARIANT $FASTFILL_VARIANT (empty or all)"; exit 1;;
esac
arms=(${FASTFILL_ARMS:-full lora})
configs=()
for arm in "${arms[@]}"; do
  case $arm in
    full) config=fastfill/v2/configs/main7-cell05-detail-$tag$v-full.json;; lora) config=fastfill/v2/configs/main7-cell05-detail-$tag$v.json;;
    *) echo "unknown arm $arm (full or lora)"; exit 1;;
  esac
  [ -f "$config" ] || { echo "no configuration $config for this backbone, arm and FASTFILL_VARIANT"; exit 1; }
  configs+=("$config")
done
dep=  # afterok:<job> the arms wait for, when setup or the subset still has to run
if [ ! -f runs/isambard/v4-setup-$tag.ok ]; then
  setup=$(sbatch --parsable --time=03:00:00 --job-name=ff4-setup-$tag fastfill/v2/ops/isambard_v4_setup.sbatch)
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
for i in "${!arms[@]}"; do
  arm=${arms[$i]} armdep=$dep
  if [ -n "$v" ]; then
    smoke=$(sbatch --parsable --job-name=ff4-smoke-detail-$tag$v-$arm ${dep:+--dependency=$dep} \
      fastfill/v2/ops/isambard_v4_detail_smoke.sbatch "${configs[$i]}")
    armdep=afterok:$smoke
  fi
  args=(fastfill/v2/ops/isambard_v4_autorun.sbatch autorun-v4-detail-$tag$v-$arm "${configs[$i]}"
        --name-suffix=-$tag$v-$arm --global-batch=32)
  first=$(sbatch --parsable --job-name=ff4-detail-$tag$v-$arm ${armdep:+--dependency=$armdep} "${args[@]}")
  resume=$(sbatch --parsable --job-name=ff4-detail-$tag$v-$arm-resume --dependency=${armdep:+$armdep,}afternotok:$first "${args[@]}")
  echo "$arm: ${smoke:+smoke=$smoke }train=$first resume=$resume  status: runs/autorun-v4-detail-$tag$v-$arm/STATUS.json"
done
squeue --me
