#!/bin/bash
# FastFill v2 final analysis on Isambard-AI (one node, 4 GH200), after both autopilot arms (runs/autorun-yawcls05: yaw_cls
# 0.5; runs/autorun-yawcls008: the 0.08 control) have finished. Submitted as
#   sbatch --dependency=afterok:<yawcls05 job>:<yawcls008 job> fastfill/v2/ops/isambard_analysis.sbatch   (runs "all")
# or one step by hand inside an allocation (Isambard login nodes have no GPU and forbid heavy work), from the checkout:
#   srun --nodes=1 --gpus=4 --time=12:00:00 bash fastfill/v2/ops/run_plan.sh inputs | gpu | cpu | report | all
# (only report is light enough for a login node).
# Everything is written under runs/analysis (inputs also: runs/.download-*, runs/baseline-hf, runs/llm-*-300,
# data/main-20261007b). Every step keeps what a previous run of it finished; resubmitting the job resumes. If the
# chosen checkpoints (FINAL, CTRL, BASE) differ from the previous run's, the inputs step removes the earlier outputs.
#
# inputs  (CPU, ~10 min) isambard_inputs.py: refuses unless both arms' STATUS.json phase is done / done-with-failures and
#         their select2 cohorts are the same bytes scored by the same code; final = the arm with the lower select2
#         score, control = the other.
#         Downloads, sha256-pinned: the old baseline (used from a copy whose backbone path is models/Qwen3-8B), the saved
#         LLM answers (prompt, harness; the structured modes only if runs/llm-structured*-300 exist) and the old data
#         main-20261007b; rebuilds the baseline's old selection cohorts (Autopilot.eval_data). Then the 500-row ablation
#         sample (sample_validation.py: the same scenes in new and old validation, eligible for all three checkpoints).
# gpu     one lane per GPU of the job (round robin; GPU 0 is ours here): every forward pass saves its head outputs:
#         final, control and baseline on NEW validation and NEW test (minimal + full; the baseline cannot answer rooms
#         over 128 objects), final on the 300 LLM rows (full); diag_ablation on the sample: final and control (new rows,
#         new train prior), baseline (new rows and old rows, old train prior). Beside, on the CPU: RoomGenBench reference
#         checks of both arms' phase F exports and of the baseline exported again like phase F (not fatal).
# cpu     every head output decoded twice by the same code (spread, argmax); cross-check against phase F's GPU-decoded
#         reports; stratify x24; compare: final vs baseline (<= 128 objects), final vs control, spread vs argmax per
#         model; the saved LLM answers re-scored by the evaluator on disk (no API calls) and llm_compare (final vs LLM).
# report  runs/analysis/final/report.html. Decisions read validation only; the test, LLM, ablation and RoomGenBench
#         sections are report-only.
# Wall time per step: runs/analysis/walltime.tsv (step, start, end, seconds, exit code, card).
# DRYRUN=1 (the CPU dry run, isambard_dryrun.sh, only): no downloads, the baseline/data pins are reported instead of
# enforced, and DEVICE (cpu) and ABLATION_ROWS may be overridden.
set -euo pipefail
R=${R:-$PWD}
cd "$R"
A=$R/runs/analysis; G=$A/gpu; F=$A/final; CACHE=$A/cache; OPS=$R/fastfill/v2/ops
mkdir -p $G $F
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # HF_HOME: the submitting shell's (sbatch exports it)
P=$R/env/bin/python
NEW=$R/data/main-20261008a
OLD=$R/data/main-20261007b
LLMROWS=$R/runs/llm-prompt-300/rows.jsonl
GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3}   # the job's GPUs (Slurm sets CUDA_VISIBLE_DEVICES)
DEVICE=${DEVICE:-cuda}
ABL=${ABLATION_ROWS:-500}
if [ "${DRYRUN:-0}" != 1 ] && { [ "$DEVICE" != cuda ] || [ "$ABL" != 500 ]; }; then
  echo "refused: DEVICE / ABLATION_ROWS overrides are for the dry run (DRYRUN=1) only"; exit 2
fi
SAMPLE=$A/ablation-sample-$ABL
CPUS=$($P -c "import os; print(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count())")
W=$(( CPUS < 32 ? CPUS : 32 ))   # stratify / compare workers and parallel CPU decodes

load_inputs() {  # FINAL CTRL BASE (checkpoints), FINAL_NAME CTRL_NAME (runs), *_DIR, COHORT_*, LLM_MODES
  [ -f $A/inputs.env ] || { echo "refused: run the inputs step first"; exit 1; }
  . $A/inputs.env
}

label() { case $1 in base) echo 基线 ;; final) echo 最终 ;; ctrl) echo 对照 ;; esac; }

finished() {  # finished DIR: a complete evaluate output. evaluate writes report.json before the outcomes files, so the
              # mark is every outcomes file holding as many complete lines as its projection's requests
  $P -c 'import json, sys
from pathlib import Path
d = Path(sys.argv[1])
try:
    r = json.loads((d / "report.json").read_text())
    parts = [("outcomes.jsonl", r)] + [(f"outcomes-{p}.jsonl", x) for p, x in r["projections"].items()]
    ok = all(sum(line.endswith(b"\n") for line in open(d / f, "rb")) == x["requests"] for f, x in parts)
except (OSError, ValueError, KeyError):
    ok = False
sys.exit(0 if ok else 1)' "$1"
}

# Functions run by timed (or after ||) run with errexit ignored (bash rule): each propagates failures with || return 1.
timed() {  # timed STEP CARD LOG CMD...: runs CMD > LOG, appends its wall time to walltime.tsv, returns its exit code
  local step=$1 card=$2 log=$3 start t0 rc=0; shift 3; start=$(date -u +%FT%TZ); t0=$(date +%s)
  "$@" > "$log" 2>&1 || rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$step" "$start" "$(date -u +%FT%TZ)" $(( $(date +%s) - t0 )) $rc "$card" >> $A/walltime.tsv
  echo "$step (card $card): exit $rc after $(( $(date +%s) - t0 )) s"
  return $rc
}

parallel() {  # parallel N < COMMANDS: each line run by bash, at most N at a time; fails if any fails
  local n=$1 line fail=0 p pids=()
  while IFS= read -r line; do
    bash -c "$line" & pids+=($!)
    if [ ${#pids[@]} -ge $n ]; then wait ${pids[0]} || fail=1; pids=(${pids[@]+"${pids[@]:1}"}); fi
  done
  for p in ${pids[@]+"${pids[@]}"}; do wait $p || fail=1; done
  return $fail
}

step_inputs() {
  local dry= before; [ "${DRYRUN:-0}" != 1 ] || dry=--dry-run
  before=$(grep -E '^(FINAL|CTRL|BASE)=' $A/inputs.env 2>/dev/null || true)
  $P $OPS/isambard_inputs.py --repo $R --out $A $dry
  load_inputs
  if [ -n "$before" ] && [ "$before" != "$(grep -E '^(FINAL|CTRL|BASE)=' $A/inputs.env)" ]; then
    echo "the chosen checkpoints changed since the previous run: its outputs are removed"
    rm -rf $G $F $A/ablation-sample-* $A/reference-final $A/reference-ctrl; mkdir -p $G $F
  fi
  if [ ! -f $SAMPLE/sample.json ]; then  # drawn on the CPU; eligible for all three checkpoints (min max_objects 128)
    rm -rf $SAMPLE
    $P $OPS/sample_validation.py --new $NEW/validation.jsonl --old $OLD/validation.jsonl --checkpoint $FINAL \
      --checkpoint $CTRL --checkpoint $BASE --rows $ABL --seed 0 --out-dir $SAMPLE
  fi
  echo INPUTS-DONE
}

gpu_step() {  # gpu_step STEP CARD LOG CMD...: CMD on GPU CARD (no GPU in the CPU dry run)
  local step=$1 card=$2 log=$3 visible=$2; shift 3
  [ $DEVICE = cuda ] || visible=
  timed $step $card $log env CUDA_VISIBLE_DEVICES=$visible "$@"
}

heads() {  # heads NAME CHECKPOINT DATA CARD PROJECTION...: one forward pass per request, head outputs saved
  local name=$1 ckpt=$2 data=$3 card=$4; shift 4
  if finished $G/$name-online; then echo "kept: heads-$name"; return 0; fi
  rm -rf $G/$name-online $G/$name-heads $G/$name-spread $G/$name-argmax   # the decodes of the old head outputs too
  gpu_step heads-$name $card $G/$name-online.log $P -m fastfill.v2.evaluate --checkpoint $ckpt --data $data \
    --projection "$@" --device $DEVICE --grid-decode spread --output $G/$name-online --save-head-outputs $G/$name-heads
}

ablation() {  # ablation NAME CHECKPOINT ROWS TRAIN CARD; the sample holds exactly $ABL eligible rows in a seeded order
  local check="import json,sys; r=json.load(open('$G/$1/report.json')); sys.exit(0 if r['rows'] == $ABL else f'$1 scored {r[\"rows\"]} rows')"
  if [ ! -f $G/$1/report.json ]; then
    rm -rf $G/$1
    gpu_step $1 $5 $G/$1.log $P $OPS/diag_ablation.py --checkpoint $2 --validation $3 --train $4 --rows $ABL \
      --device $DEVICE --out $G/$1 || return 1
  fi
  $P -c "$check"
}

TASKS=(final-test ctrl-test final-val ctrl-val base-val base-test final-llm abl-ctrl-new abl-final-new abl-base-new abl-base-old)
task() {  # task NAME CARD
  case $1 in
    final-test) heads final-test $FINAL $NEW/test.jsonl $2 minimal full ;;
    ctrl-test) heads ctrl-test $CTRL $NEW/test.jsonl $2 minimal full ;;
    final-val) heads final-val $FINAL $NEW/validation.jsonl $2 minimal full ;;
    ctrl-val) heads ctrl-val $CTRL $NEW/validation.jsonl $2 minimal full ;;
    base-val) heads base-val $BASE $NEW/validation.jsonl $2 minimal full ;;
    base-test) heads base-test $BASE $NEW/test.jsonl $2 minimal full ;;
    final-llm) heads final-llm $FINAL $LLMROWS $2 full ;;
    abl-final-new) ablation ablation-final-new $FINAL $SAMPLE/new.jsonl $NEW/train.jsonl $2 ;;   # each checkpoint's own train prior
    abl-ctrl-new) ablation ablation-ctrl-new $CTRL $SAMPLE/new.jsonl $NEW/train.jsonl $2 ;;
    abl-base-new) ablation ablation-base-new $BASE $SAMPLE/new.jsonl $OLD/train.jsonl $2 ;;
    abl-base-old) ablation ablation-base-old $BASE $SAMPLE/old.jsonl $OLD/train.jsonl $2 ;;
  esac
}

rgb_export() {  # the baseline exported on the CPU like phase F's run_checkpoint.sh; every usable export reference-checked
  local req key side
  mkdir -p $A/rgb-base
  for req in $R/runs/roomgenbench/requests/*.json; do
    key=$(basename $req .json)
    if [ ! -f $A/rgb-base/$key.prediction.json ] || [ ! -d $A/rgb-base/$key.handoff ]; then
      rm -rf $A/rgb-base/$key.prediction.json $A/rgb-base/$key.handoff
      OMP_NUM_THREADS=48 CUDA_VISIBLE_DEVICES= $P -m fastfill.v2.predict --checkpoint $BASE --request $req --device cpu \
        --output $A/rgb-base/$key.prediction.json --export-dir $A/rgb-base/$key.handoff || return 1
    fi
    for side in $RGB_SIDES base:$A/rgb-base; do
      mkdir -p $A/reference-${side%%:*}
      [ -f $A/reference-${side%%:*}/$key.json ] || CUDA_VISIBLE_DEVICES= $P -m fastfill.v2.roomgenbench --reference-check \
        ${side#*:}/$key.handoff --scene $R/RoomGenBench/bench/inputs/scenes/$key.json --output $A/reference-${side%%:*}/$key.json || return 1
    done
  done
}

step_rgb() {  # an arm's phase F export (run_checkpoint.sh) counts only when complete and made by the code on disk
  local code m ckpt export
  code=$($P -c "from fastfill.v2.autorun import implementation_sha256; print(implementation_sha256('$R'))")
  RGB_SIDES=
  for m in final ctrl; do
    if [ $m = final ]; then ckpt=$FINAL; export=$R/runs/roomgenbench/best-$FINAL_NAME; else ckpt=$CTRL; export=$R/runs/roomgenbench/best-$CTRL_NAME; fi
    if [ $(find $export -path '*.layout_boxes/receipt.json' 2>/dev/null | wc -l) -ne 5 ]; then
      echo "skipped $m: $export lacks 5 receipts"
    elif [ "$(sed -n 1p $export/checkpoint.txt)" != "$ckpt" ] || [ "$(sed -n 2p $export/checkpoint.txt)" != "$code" ]; then
      echo "skipped $m: $export holds another checkpoint or was made by other fastfill/v2 code than is on disk now"
    else
      RGB_SIDES="$RGB_SIDES $m:$export"
    fi
  done
  timed rgb-export-and-check cpu $A/rgb.log rgb_export
}

step_gpu() {
  load_inputs
  local cards pids=() n k i fail=0
  IFS=, read -r -a cards <<< "$GPUS"
  n=${#cards[@]}
  echo "lanes on $DEVICE: ${cards[*]}"
  step_rgb > $A/rgb-step.log 2>&1 & local rgb=$!
  for k in $(seq 0 $((n - 1))); do
    ( rc=0; for i in "${!TASKS[@]}"; do if [ $((i % n)) = $k ]; then task ${TASKS[i]} ${cards[k]} || rc=1; fi; done; exit $rc ) &
    pids+=($!)
  done
  for i in "${pids[@]}"; do wait $i || fail=1; done   # every lane's own exit code
  wait $rgb || echo "RoomGenBench step failed (report-only, not fatal): see $A/rgb-step.log and $A/rgb.log"
  [ $fail = 0 ] || { echo "GPU-STEP FAILURE: see $A/walltime.tsv and the logs in $G"; exit 1; }
  echo GPU-DONE
}

crosscheck() {  # identical layouts expected: phase F's GPU-decoded reports and our online runs vs the CPU decodes
  local pairs=() m n o
  for m in final ctrl; do
    if [ $m = final ]; then n=$FINAL_NAME; else n=$CTRL_NAME; fi
    pairs+=(--pair "phase F $(label $m) 测试·三字段·spread=$R/runs/test-$n/outcomes.jsonl,$G/$m-test-spread/outcomes.jsonl")
    pairs+=(--pair "phase F $(label $m) 测试·完整条件·spread=$R/runs/test-$n/outcomes-full.jsonl,$G/$m-test-spread/outcomes-full.jsonl")
    pairs+=(--pair "phase F $(label $m) 测试·三字段·argmax=$R/runs/test-$n-argmax/outcomes.jsonl,$G/$m-test-argmax/outcomes.jsonl")
  done
  pairs+=(--pair "phase F 最终 LLM 行·spread=$R/runs/llmrows-$FINAL_NAME/outcomes.jsonl,$G/final-llm-spread/outcomes.jsonl")
  pairs+=(--pair "phase F 最终 LLM 行·argmax=$R/runs/llmrows-$FINAL_NAME-argmax/outcomes.jsonl,$G/final-llm-argmax/outcomes.jsonl")
  for m in base-val final-val ctrl-val base-test final-test ctrl-test final-llm; do for o in outcomes.jsonl outcomes-full.jsonl; do
    if [ -f $G/$m-online/$o ]; then pairs+=(--pair "本计划在线运行 ${m}（${o}）=$G/$m-online/$o,$G/$m-spread/$o"); fi
  done; done
  $P $OPS/crosscheck.py "${pairs[@]}" > $F/crosscheck.md
}

S() { $P $OPS/stratify.py --workers $W --cache $CACHE "$@" > /dev/null; }
C() { $P $OPS/compare.py --workers $W --cache $CACHE "$@" > /dev/null; }

stratify_all() {
  local m s d
  for m in base final ctrl; do for s in val test; do for d in spread argmax; do
    S --eval-dir $G/$m-$s-$d --out $F/$m-$s-minimal-$d.json || return 1
    S --eval-dir $G/$m-$s-$d --outcomes outcomes-full.jsonl --out $F/$m-$s-full-$d.json || return 1
  done; done; done
}

compare_all() {  # final minus baseline on rooms both can answer (<= 128 objects); final minus control; spread minus argmax
  local s proj o d sel m
  for s in val test; do for proj in minimal full; do
    o=outcomes.jsonl; [ $proj = minimal ] || o=outcomes-full.jsonl
    for d in spread argmax; do
      for sel in all "old rule=frozen_prep" "new rule=wall_anchor,support_inside_parent,support_on_added_parent"; do
        set -- $sel
        C --a $G/final-$s-$d --a-outcomes $o --b $G/base-$s-$d --b-outcomes $o --label-a 最终 --label-b 基线 --max-objects 128 \
          ${2:+--where $2} --out $F/compare-$s-$proj-$d-$1.json || return 1
        C --a $G/final-$s-$d --a-outcomes $o --b $G/ctrl-$s-$d --b-outcomes $o --label-a 最终 --label-b 对照 \
          ${2:+--where $2} --out $F/control-$s-$proj-$d-$1.json || return 1
      done
    done
    for m in final ctrl base; do
      C --a $G/$m-$s-spread --a-outcomes $o --b $G/$m-$s-argmax --b-outcomes $o --label-a spread --label-b argmax \
        --out $F/decode-$s-$m-$proj.json || return 1
    done
  done; done
}

llm_all() {  # the saved LLM answers re-scored by the evaluator on disk; FastFill final (CPU decodes; latency from the online run)
  local methods=() cohorts=() m out
  mkdir -p $A/llm
  for m in $LLM_MODES; do
    out=$A/llm/llm-$m-300-eval
    if ! finished $out; then
      rm -rf $out
      $P -m fastfill.v2.evaluate --data $R/runs/llm-$m-300/rows.jsonl --predictions $R/runs/llm-$m-300/predictions.jsonl \
        --output $out > $out.log 2>&1 || return 1
    fi
    methods+=(--method "LLM $m=$out,$R/runs/llm-$m-300/summary.json")
  done
  for m in $COHORT_NEW $COHORT_OLD $COHORT_OLD1; do cohorts+=(--selection-cohort $m); done
  $P $OPS/llm_compare.py --method "FastFill 最终·spread（防碰撞后处理）=$G/final-llm-spread,$G/final-llm-online" \
    --method "FastFill 最终·argmax（模型原始输出）=$G/final-llm-argmax,$G/final-llm-online" "${methods[@]}" "${cohorts[@]}" \
    --out $F/llm-compare.json --workers $W
}

step_cpu() {
  load_inputs
  export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
  local x name data proj d
  # every saved head output decoded twice by the same code
  for x in base-val:$NEW/validation.jsonl final-val:$NEW/validation.jsonl ctrl-val:$NEW/validation.jsonl \
           base-test:$NEW/test.jsonl final-test:$NEW/test.jsonl ctrl-test:$NEW/test.jsonl final-llm:$LLMROWS; do
    name=${x%%:*}; data=${x#*:}; proj="minimal full"; [ $name != final-llm ] || proj=full
    for d in spread argmax; do
      finished $G/$name-$d || echo "rm -rf $G/$name-$d && $P -m fastfill.v2.evaluate --from-head-outputs $G/$name-heads --data $data --projection $proj --grid-decode $d --output $G/$name-$d > $G/$name-$d.log 2>&1"
    done
  done > $G/decode-commands.txt
  timed decode cpu $G/decode.log parallel $(( W < 14 ? W : 14 )) < $G/decode-commands.txt
  timed crosscheck cpu $F/crosscheck.log crosscheck
  timed stratify cpu $F/stratify.log stratify_all
  timed compare cpu $F/compare.log compare_all
  timed llm-compare cpu $F/llm-compare.log llm_all
  echo CPU-DONE
}

pj() { case $1 in minimal) echo 三字段 ;; full) echo 完整条件 ;; esac; }
sel() { case $1 in all) echo 全部对象 ;; old) echo "旧对象 frozen_prep" ;; new) echo 新对象 ;; esac; }

step_report() {
  load_inputs
  local args=() m s p d x side
  for s in val test; do
    for m in final ctrl base; do for p in minimal full; do for d in spread argmax; do
      args+=(--$s-run "$(label $m)·$(pj $p)·$d=$F/$m-$s-$p-$d.json"); done; done; done
    for p in minimal full; do for d in spread argmax; do for x in all old new; do
      args+=(--$s-compare "最终 vs 基线·$(pj $p)·${d}·$(sel $x)（≤128 物体）=$F/compare-$s-$p-$d-$x.json")
      args+=(--$s-control "最终 vs 对照·$(pj $p)·${d}·$(sel $x)=$F/control-$s-$p-$d-$x.json"); done; done
      for m in final ctrl base; do args+=(--$s-decode "$(label $m)·$(pj $p)：spread vs argmax=$F/decode-$s-$m-$p.json"); done; done
  done
  for side in final:"最终（phase F 的 CPU 导出）" ctrl:"对照（phase F 的 CPU 导出）" base:"基线（当前代码 CPU 重新导出）"; do
    if [ $(ls $A/reference-${side%%:*}/*.json 2>/dev/null | wc -l) -gt 0 ]; then args+=(--reference "${side#*:}=$A/reference-${side%%:*}"); fi
  done
  $P $OPS/report.py --out $F/report.html --inputs $A/inputs.json \
    --title "FastFill v2 评测报告（Isambard）：最终 $FINAL_NAME vs 对照 $CTRL_NAME vs 基线" \
    --summary "最终臂 $FINAL_DIR=$R/runs/$FINAL_DIR/SUMMARY.md" --summary "对照臂 $CTRL_DIR=$R/runs/$CTRL_DIR/SUMMARY.md" \
    "${args[@]}" --llm $F/llm-compare.json \
    --selection-cohort "最终（select2，新验证集全集；对照的相同）=$COHORT_NEW" \
    --selection-cohort "基线（select2，旧验证集全集）=$COHORT_OLD" --selection-cohort "基线（select1，旧验证集 3000 行）=$COHORT_OLD1" \
    --ablation "最终·新行（$ABL 行分层样本）=$G/ablation-final-new/report.json" \
    --ablation "对照·新行（同一样本）=$G/ablation-ctrl-new/report.json" \
    --ablation "基线·新行（同一样本，相同输入）=$G/ablation-base-new/report.json" \
    --ablation "基线·旧行（同一批场景）=$G/ablation-base-old/report.json" --ablation-sample $SAMPLE/sample.json \
    --walltime $A/walltime.tsv --notes $F/crosscheck.md
}

case "${1:-}" in
  inputs) step_inputs ;; gpu) step_gpu ;; cpu) step_cpu ;; report) step_report ;;
  all) step_inputs; step_gpu; step_cpu; step_report ;;
  *) sed -n '2,31p' "$0"; exit 2 ;;
esac
