#!/bin/bash
# Isambard-AI, from the v4 checkout root on a login node: three arms on three nodes, same data and global batch:
#   base  main7-cell05-main-o256-e5       the old baseline with max_objects 256
#   D     main7-cell05-D-fixed-noise-e5   + masked training with given poses and pose noise
#   E     main7-cell05-E-support-e5       D + support head
# Without runs/isambard/v4-setup.ok the setup job is submitted first and the arms wait for it (afterok). Each arm gets a
# continuation job (afternotok) that resumes it once if the first job fails or reaches the 24 h limit.
#
# Same three arms on Qwen3.5-27B (configs *-27b.json, status runs/autorun-v4-27b-<arm>, own setup marker):
#   FASTFILL_BACKBONE=Qwen3.5-27B fastfill/v2/ops/isambard_v4_submit.sh
#   FASTFILL_BACKBONE=Qwen3.5-27B FASTFILL_ARMS=E fastfill/v2/ops/isambard_v4_submit.sh     # only the E arm
# The 8B arms took ~17 h; 27B is expected at roughly 4-5x, so each 27B arm gets 5 chained continuations (6 x 24 h).
#
# Backbone-size sweep, base arm only (configs *-1.7b.json, *-3b.json, *-4b.json; the 8B base arm is the sweep's 8B point):
#   FASTFILL_BACKBONE=Qwen3-1.7B FASTFILL_ARMS=base fastfill/v2/ops/isambard_v4_submit.sh
# FASTFILL_FT=full trains the whole backbone instead of LoRA (config *-full.json, status runs/autorun-v4-<tag>-full-base):
#   FASTFILL_BACKBONE=Qwen3-1.7B FASTFILL_ARMS=base FASTFILL_FT=full fastfill/v2/ops/isambard_v4_submit.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
export FASTFILL_BACKBONE=${FASTFILL_BACKBONE:-Qwen3-8B}
case $FASTFILL_BACKBONE in
  Qwen3-8B) tag= resumes=1 setup_time=03:00:00;;
  Qwen3.5-27B) tag=-27b resumes=5 setup_time=06:00:00;;
  Qwen3-1.7B) tag=-1.7b resumes=1 setup_time=03:00:00;;
  Qwen2.5-3B) tag=-3b resumes=1 setup_time=03:00:00;;
  Qwen3-4B) tag=-4b resumes=1 setup_time=03:00:00;;
  *) echo "unknown FASTFILL_BACKBONE $FASTFILL_BACKBONE (Qwen3-8B, Qwen3.5-27B, Qwen3-1.7B, Qwen2.5-3B or Qwen3-4B)"; exit 1;;
esac
case ${FASTFILL_FT:-lora} in
  lora) ft=;; full) ft=-full;;
  *) echo "unknown FASTFILL_FT $FASTFILL_FT (lora or full)"; exit 1;;
esac
after=()
if [ ! -f runs/isambard/v4-setup$tag.ok ]; then
  setup=$(sbatch --parsable --time=$setup_time --job-name=ff4-setup$tag fastfill/v2/ops/isambard_v4_setup.sbatch)
  after=(--dependency=afterok:$setup)
  echo "setup=$setup ($FASTFILL_BACKBONE)"
fi
for name in ${FASTFILL_ARMS:-base D E}; do
  case $name in
    base) config=main7-cell05-main-o256-e5;; D) config=main7-cell05-D-fixed-noise-e5;; E) config=main7-cell05-E-support-e5;;
    *) echo "unknown arm $name (base, D or E)"; exit 1;;
  esac
  config=fastfill/v2/configs/$config$tag$ft.json
  [ -f "$config" ] || { echo "no configuration $config for this backbone, arm and FASTFILL_FT"; exit 1; }
  args=(fastfill/v2/ops/isambard_v4_autorun.sbatch autorun-v4$tag$ft-$name $config --name-suffix=$tag$ft-$name)
  first=$(sbatch --parsable --job-name=ff4$tag$ft-$name "${after[@]}" "${args[@]}")
  chain=$first
  for _ in $(seq $resumes); do
    chain=$(sbatch --parsable --job-name=ff4$tag$ft-$name-resume --dependency=afternotok:$chain "${args[@]}")
  done
  echo "$name: train=$first resumes up to $chain  status: runs/autorun-v4$tag$ft-$name/STATUS.json"
done
squeue --me
