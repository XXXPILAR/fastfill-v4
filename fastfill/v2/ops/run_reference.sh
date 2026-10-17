#!/bin/bash
set -euo pipefail
cd /home/jovyan/shanliantian/fastfill
OUT=runs/roomgenbench/reference-best-main7; mkdir -p $OUT
for room in bathroom bedroom gym livingroom restaurant; do
  PYTHONDONTWRITEBYTECODE=1 ./env/bin/python -m fastfill.v2.roomgenbench --reference-check runs/roomgenbench/best-main7-cell05-main-20261007b-e5/$room.handoff \
    --scene RoomGenBench/bench/inputs/scenes/$room.json --output $OUT/$room.json
done
echo REFERENCE-DONE
