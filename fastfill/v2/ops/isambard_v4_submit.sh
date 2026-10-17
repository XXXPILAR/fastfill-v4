#!/bin/bash
# Isambard-AI, from the v4 checkout root on a login node: three arms on three nodes, same data and global batch:
#   base  main7-cell05-main-o256-e5       the old baseline with max_objects 256
#   D     main7-cell05-D-fixed-noise-e5   + masked training with given poses and pose noise
#   E     main7-cell05-E-support-e5       D + support head
# Without runs/isambard/v4-setup.ok the setup job is submitted first and the arms wait for it (afterok). Each arm gets a
# continuation job (afternotok) that resumes it once if the first job fails or reaches the 24 h limit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
after=()
if [ ! -f runs/isambard/v4-setup.ok ]; then
  setup=$(sbatch --parsable fastfill/v2/ops/isambard_v4_setup.sbatch)
  after=(--dependency=afterok:$setup)
  echo "setup=$setup"
fi
for arm in base:main7-cell05-main-o256-e5 D:main7-cell05-D-fixed-noise-e5 E:main7-cell05-E-support-e5; do
  name=${arm%%:*}; config=fastfill/v2/configs/${arm#*:}.json
  args=(fastfill/v2/ops/isambard_v4_autorun.sbatch autorun-v4-$name $config --name-suffix=-$name)
  first=$(sbatch --parsable --job-name=ff4-$name "${after[@]}" "${args[@]}")
  again=$(sbatch --parsable --job-name=ff4-$name-resume --dependency=afternotok:$first "${args[@]}")
  echo "$name: train=$first resume=$again  status: runs/autorun-v4-$name/STATUS.json"
done
squeue --me
