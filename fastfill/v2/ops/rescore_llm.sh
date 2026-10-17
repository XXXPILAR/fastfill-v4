#!/bin/bash
# Re-score the four saved LLM predictions with the evaluator now on disk (no API calls); keep the old reports.
set -uo pipefail
cd /home/jovyan/shanliantian/fastfill
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
TAG=$(git rev-parse --short HEAD)
for m in prompt harness structured structured-harness; do
  ./env/bin/python -m fastfill.v2.evaluate --data runs/llm-$m-300/rows.jsonl --predictions runs/llm-$m-300/predictions.jsonl \
    --output runs/llm-$m-300-eval-$TAG > runs/llm-$m-300-eval-$TAG.log 2>&1 && echo "$m rescored -> llm-$m-300-eval-$TAG" || echo "$m FAILED"
done
echo RESCORE-DONE $TAG
