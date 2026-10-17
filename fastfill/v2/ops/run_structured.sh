#!/bin/bash
# Two OptiScene-style LLM agent modes on the frozen 300 rows, then the same scorer as every other method.
set -uo pipefail
cd /home/jovyan/shanliantian/fastfill
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
ROWS=runs/llm-prompt-300/rows.jsonl; ENV=/home/jovyan/shanliantian/.fastfill_api.env
for mode in structured structured-harness; do
  ( ./env/bin/python -m fastfill.v2.llm_structured --data $ROWS --env $ENV --mode $mode --repairs 2 --workers 4 --max-samples 300 \
      --output runs/llm-$mode-300 > runs/llm-$mode-300.log 2>&1
    cmp -s runs/llm-$mode-300/rows.jsonl $ROWS && echo "$mode rows identical" || echo "$mode ROWS DIFFER"
    ./env/bin/python -m fastfill.v2.evaluate --data runs/llm-$mode-300/rows.jsonl --predictions runs/llm-$mode-300/predictions.jsonl \
      --output runs/llm-$mode-300-eval > runs/llm-$mode-300-eval.log 2>&1 && echo "$mode scored" ) &
done
wait
echo ALL-DONE
