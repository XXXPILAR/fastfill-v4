#!/bin/bash
# Usage: fastfill/v2/ops/run_checkpoint.sh <model-step dir> <tag>; CPU inference (GPUs are training). The autopilot calls it.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/home/jovyan/shanliantian/.huggingface}
PY=./env/bin/python; CK=$1; OUT=runs/roomgenbench/$2; mkdir -p $OUT; export OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES=
# The requests are what the builder on disk makes of the scenes on disk, whoever made runs/roomgenbench/requests (the
# Isambard setup, older code or scenes): built afresh (~1 s) into a new directory, the old one is kept when byte-identical,
# else set aside and replaced by a rename (never rewritten in place). The hand-off records the request bytes it used.
# ponytail: two exports that both find the directory outdated at the same moment race on the rename: one may exit non-zero
# (a recorded export failure, rerunnable) or nest a harmless requests/requests that the next export clears.
REQ=runs/roomgenbench/requests; NEW=$(mktemp -d $REQ.new-XXXXXX)
$PY -m fastfill.v2.roomgenbench --requests-from RoomGenBench/bench/inputs/scenes --requests-out $NEW/requests
if ! diff -r -q $REQ $NEW/requests > /dev/null 2>&1; then
  [ ! -d $REQ ] || mv $REQ $REQ.stale-${NEW##*-}
  mv $NEW/requests $REQ
fi
rm -r $NEW; sha256sum $REQ/*.json > $OUT/requests.sha256
for req in $REQ/*.json; do
  key=$(basename $req .json)
  $PY -m fastfill.v2.predict --checkpoint $CK --request $req --output $OUT/$key.prediction.json --export-dir $OUT/$key.handoff --device cpu
  $PY -m fastfill.v2.roomgenbench --handoff $OUT/$key.handoff --output-dir $OUT/$key.layout_boxes --roomgenbench-root RoomGenBench --method layout_boxes --require-placement
done
echo DONE $OUT
