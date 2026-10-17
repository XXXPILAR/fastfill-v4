#!/bin/bash
# Isambard-AI, from the repository root on a login node: the setup job, then two autopilots on two nodes once it
# succeeds: the final candidate (yaw_cls 0.5) and its control (the baseline's yaw_cls 0.08), same data and batch.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/isambard
setup=$(sbatch --parsable fastfill/v2/ops/isambard_setup.sbatch)
sbatch --dependency=afterok:$setup fastfill/v2/ops/isambard_autorun.sbatch autorun-yawcls05 --name-suffix=-yawcls05 --set loss.yaw_cls=0.5
sbatch --dependency=afterok:$setup fastfill/v2/ops/isambard_autorun.sbatch autorun-yawcls008 --name-suffix=-yawcls008
squeue --me
