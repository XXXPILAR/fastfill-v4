#!/bin/bash
# Retrain on the round-10 data: adopt the finished baseline run, select its best checkpoint, wait for NEXT_DATA
# (main-20261008a), then train data fix + max_objects 256 + yaw_cls 0.5. No LLM calls (the env path does not exist).
cd /home/jovyan/shanliantian/fastfill
mkdir -p runs/autorun
nohup ./env/bin/python -m fastfill.v2.autorun \
  --current main7-cell05-main-20261007b-e5 runs/autorun/main7-cell05-main-20261007b-e5.json 1,2,3,4,5,6,7 \
  --current-data data/main-20261007b \
  --llm-env /home/jovyan/shanliantian/.no-llm-for-retrain \
  --data-wait-h 8 \
  --set model.max_objects=256 --set loss.yaw_cls=0.5 --name-suffix=-yawcls05 >> runs/autorun/nohup.log 2>&1 &
echo "autopilot pid $!"
