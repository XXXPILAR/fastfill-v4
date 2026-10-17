#!/bin/bash
# CPU-only test of the analysis scripts on the BASELINE artifacts (final = baseline placeholder); writes test/* and
# test/sample_report.html under runs/analysis. Usage: bash runs/analysis/make_sample.sh   (~10 minutes, <= 8 threads)
#   validation section: the head-output path of run_plan on 24 NEW validation rooms (16 small, 6 mid, 2 > 128 objects),
#     baseline checkpoint, forward pass on the CPU (8 threads), spread and argmax decoded from the saved heads
#   test section: the baseline's 10-07 GPU test reports (old data; spread and argmax by the same code)
set -euo pipefail
R=/home/jovyan/shanliantian/fastfill
cd $R/runs/analysis
export PYTHONPATH=$R PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false \
       HF_HOME=/home/jovyan/shanliantian/.huggingface
P=$R/env/bin/python; W=7; T=test; K=$T/cache; mkdir -p $T
BASE=$R/runs/main7-cell05-main-20261007b-e5/model-step-6357
B=$R/runs/test-main7-cell05-main-20261007b-e5          # baseline test: outcomes.jsonl = minimal, outcomes-full.jsonl = full
L=$R/runs/llmrows-main7-cell05-main-20261007b-e5       # baseline on the 300 LLM rows (full condition)
OLD=$R/data/main-20261007b
COHORT_OLD=$R/runs/autorun/validation-$(sha256sum $OLD/validation.jsonl | cut -c1-12)-all.jsonl   # baseline select2
COHORT_OLD1=${COHORT_OLD%-all.jsonl}-3000.jsonl                                                     # baseline select1

# (0) run_plan's preflight on the live cards (read-only nvidia-smi queries): GPU 0 and every training card must be refused
eval "$(sed -n '/^idle() {/,/^}/p' run_plan.sh)"
for c in 0 1 2 3 4 5 6 7; do if idle $c; then echo "card $c idle"; else echo "card $c refused"; fi; done | tee $T/preflight.txt

# (1) 24 fixed NEW validation rooms (seeded), head outputs saved once (CPU forward, 8 threads), decoded twice by the
#     same code; crosscheck of the online decode vs the CPU decode
pick_rows() {  # $1 = output file: 8 small rooms with added objects + 8 small without (seed 0), 2 > 128 objects + 6 of 9-40 (seed 1)
  $P - "$1" <<'PY'
import json, random, sys
from fastfill.v2.evaluate import project_minimal
mixed, plain, big, mid = [], [], [], []
for line in open("/home/jovyan/shanliantian/fastfill/data/main-20261008a/validation.jsonl"):
    r = json.loads(line); k = len(r["condition"]["objects"])
    if project_minimal(r) is None:
        continue
    rules = {(e or {}).get("selection_rule") or "frozen_prep" for e in r["provenance"].get("field_evidence") or []}
    if k <= 8:
        (mixed if len(rules) > 1 else plain).append(line)
    if k > 128:
        big.append(line)
    elif 9 <= k <= 40:
        mid.append(line)
rng = random.Random(0); rng.shuffle(mixed); rng.shuffle(plain)
small = mixed[:8] + plain[:8]; rng.shuffle(small)
rng = random.Random(1); rng.shuffle(big); rng.shuffle(mid)
rows = small + big[:2] + mid[:6]; rng.shuffle(rows)
open(sys.argv[1], "w").writelines(rows)
PY
}
[ -f $T/hv-rows.jsonl ] || pick_rows $T/hv-rows.jsonl
[ -e $T/hv-heads ] || OMP_NUM_THREADS=8 $P -m fastfill.v2.evaluate --checkpoint $BASE --data $T/hv-rows.jsonl --projection minimal full \
  --device cpu --grid-decode spread --output $T/hv-online --save-head-outputs $T/hv-heads > $T/hv-online.log 2>&1
for d in spread argmax; do [ -e $T/hv-$d ] || echo "$P -m fastfill.v2.evaluate --from-head-outputs $T/hv-heads --data $T/hv-rows.jsonl --projection minimal full --grid-decode $d --output $T/hv-$d > $T/hv-$d.log 2>&1"; done \
  | xargs -d '\n' -P 2 -I CMD bash -c CMD
$P crosscheck.py --pair "在线 spread（三字段）=$T/hv-online/outcomes.jsonl,$T/hv-spread/outcomes.jsonl" \
  --pair "在线 spread（完整条件）=$T/hv-online/outcomes-full.jsonl,$T/hv-spread/outcomes-full.jsonl" \
  --pair "缺文件示例=$T/none/outcomes.jsonl,$T/hv-spread/outcomes.jsonl" > $T/crosscheck.md
grep -q "| 在线 spread（三字段） | 24 | 24 |" $T/crosscheck.md && grep -q "| 在线 spread（完整条件） | 24 | 24 |" $T/crosscheck.md
# the code check refuses an online run against a head-output decode
if $P compare.py --a $T/hv-online --b $T/hv-argmax --out $T/refused.json 2> $T/refused.txt; then echo "NOT REFUSED"; exit 1; fi
grep -q "refused: the runs were not decoded by the same code" $T/refused.txt

# (2) stratify (every overall number must reproduce report.json exactly) — validation (heads) and test (10-07 reports)
S() { local out; out=$($P stratify.py --workers $W --cache $K "$@"); grep -q '"reproduces_report_exactly": true' <<< "$out" || { echo "NOT REPRODUCED: $*"; return 1; }; }
for d in spread argmax; do
  S --eval-dir $T/hv-$d --out $T/val-minimal-$d.json
  S --eval-dir $T/hv-$d --outcomes outcomes-full.jsonl --out $T/val-full-$d.json
done
S --eval-dir $B --rows $OLD/test.jsonl --out $T/baseline-test-minimal-spread.json
S --eval-dir $B-argmax --rows $OLD/test.jsonl --out $T/baseline-test-minimal-argmax.json
S --eval-dir $B --outcomes outcomes-full.jsonl --rows $OLD/test.jsonl --out $T/baseline-test-full-spread.json
[ -d $T/shift ] || $P selftest_shift.py --rows ablation-sample-500/new.jsonl --work $T/shift   # rule attribution (asserts inside)

# (3) compare: placeholder final-vs-baseline (<= 128 objects) and spread-vs-argmax, as run_plan does
C() { $P compare.py --workers $W --cache $K "$@" > /dev/null; }
for proj in minimal full; do
  o=outcomes.jsonl; [ $proj = minimal ] || o=outcomes-full.jsonl
  for d in spread argmax; do
    for sel in all "old rule=frozen_prep" "new rule=wall_anchor,support_inside_parent,support_on_added_parent"; do
      set -- $sel
      C --a $T/hv-$d --a-outcomes $o --b $T/hv-$d --b-outcomes $o --label-a "最终（占位=基线）" --label-b 基线 --max-objects 128 \
        ${2:+--where $2} --out $T/compare-val-$proj-$d-$1.json
    done
  done
  C --a $T/hv-spread --a-outcomes $o --b $T/hv-argmax --b-outcomes $o --label-a spread --label-b argmax --out $T/decode-val-$proj.json
done
$P -c "import json; r=json.load(open('$T/compare-val-minimal-spread-all.json'))['rooms']; assert r['excluded_by_room_filter'] == 2, r"
C --a $B --b $B-argmax --label-a spread --label-b argmax --out $T/decode-test-minimal.json
for sel in "old rule=frozen_prep" "new rule=wall_anchor,support_inside_parent,support_on_added_parent"; do
  set -- $sel
  C --a $B --b $B --a-outcomes outcomes-full.jsonl --b-outcomes outcomes-full.jsonl --label-a "最终（占位=基线）" --label-b 基线 \
    --max-objects 128 --where $2 --out $T/compare-test-full-spread-$1.json
done

# (4) FastFill vs the four LLM modes on the frozen 300 rows (failure causes from predictions.jsonl)
M=(); for m in prompt harness structured structured-harness; do
  M+=(--method "LLM $m=$R/runs/llm-$m-300-eval-4125b82,$R/runs/llm-$m-300/summary.json"); done
#     argmax latency from the online spread run, as run_plan does (a head-output decode observes none)
$P llm_compare.py --method "FastFill spread=$L" --method "FastFill argmax=$L-argmax,$L" "${M[@]}" \
  --selection-cohort $COHORT_OLD --selection-cohort $COHORT_OLD1 --out $T/llm-compare-baseline.json --workers $W > /dev/null
$P - $T/llm-compare-baseline.json <<'PY'
import json, sys
c = {x["method"]: x for x in json.load(open(sys.argv[1]))["cost"]}
assert c["LLM prompt"]["failure_causes"] == {"IncompleteRead": 7}, c["LLM prompt"]
assert c["FastFill argmax"]["latency_grid_decode"] == "spread" and c["FastFill spread"]["latency_source"] == "this run"
for m, x in c.items():
    assert x["kind"] != "llm" or abs(x["mean_latency_s"] - x["summary_mean_latency_s"]) < 1e-6, m
    f = x["failed_rooms"]
    print(m, "failed (objects, rank):", f and [(r["objects"], r["rank"]) for r in f["rooms"]], f and f["p_all_in_largest_if_size_independent"])
PY
# the ablation sample overlaps the selection cohort of the same scenes (report.py --ablation-sample flag)
$P -c "import json, report; s=json.load(open('ablation-sample-500/sample.json')); print('ablation sample in old cohort:', len(set(s['scene_ids']) & report.scene_ids('$COHORT_OLD')), '/', len(s['scene_ids']))"
# the selection-bias note's three cases; the final's select2 rows do not exist yet: stand-ins named like a cohort file
ln -sf hv-rows.jsonl $T/validation-$(sha256sum $R/data/main-20261008a/validation.jsonl | cut -c1-12)-standin.jsonl
[ -s $T/one-test-row.jsonl ] || head -1 $OLD/test.jsonl > $T/one-test-row.jsonl
$P - $T $COHORT_OLD <<'PY'
import glob, sys, report
t, old = sys.argv[1:]
ids, base = report.scene_ids(f"{t}/hv-rows.jsonl"), [("基线（select2）", old)]
both = report.bias_note("x", ids, base + [("最终（替身）", glob.glob(f"{t}/validation-*-standin.jsonl")[0])])
assert "方向无法确定" in both and "不同数据版本" in both and "只有最终的选择见过新对象" in both and "偏向" not in both, both
one = report.bias_note("x", ids, base + [("最终（替身：一行测试集）", f"{t}/one-test-row.jsonl")])
assert "只有基线的选择队列含这些场景" in one and "可能偏向基线" in one, one
assert "未提供最终的选择队列" in report.bias_note("x", ids, base)
print("bias note:", both, one, sep="\n")
PY

# (5) RoomGenBench reference check with the code on disk of the baseline's CPU export (best-round1, 2026-10-08 03:00)
mkdir -p $T/reference-base-round1
for room in bathroom bedroom gym livingroom restaurant; do
  [ -f $T/reference-base-round1/$room.json ] || $P -m fastfill.v2.roomgenbench --reference-check $R/runs/roomgenbench/best-round1/$room.handoff \
    --scene $R/RoomGenBench/bench/inputs/scenes/$room.json --output $T/reference-base-round1/$room.json > /dev/null
done

# (6) the sample report
V=(); for d in spread argmax; do for p in minimal full; do
  n=三字段; [ $p = minimal ] || n=完整条件
  V+=(--val-run "基线·$n·$d（24 房间，CPU 头输出）=$T/val-$p-$d.json")
  for x in all:全部对象 old:旧对象 new:新对象; do V+=(--val-compare "最终 vs 基线·$n·$d·${x#*:}（≤128 物体，占位）=$T/compare-val-$p-$d-${x%%:*}.json"); done
done; done
$P report.py --out $T/sample_report.html --title "FastFill v2 评测报告（样例：最终=基线占位；验证集=24 个新验证房间的 CPU 头输出）" \
  --summary $R/runs/autorun-baseline-20261007b/SUMMARY.md "${V[@]}" \
  --val-decode "基线·三字段：spread vs argmax=$T/decode-val-minimal.json" --val-decode "基线·完整条件：spread vs argmax=$T/decode-val-full.json" \
  --test-run "基线·三字段·spread=$T/baseline-test-minimal-spread.json" --test-run "基线·三字段·argmax=$T/baseline-test-minimal-argmax.json" \
  --test-run "基线·完整条件·spread=$T/baseline-test-full-spread.json" \
  --test-compare "最终 vs 基线·完整条件·spread·旧对象（占位）=$T/compare-test-full-spread-old.json" \
  --test-compare "最终 vs 基线·完整条件·spread·新对象（占位，旧数据无新对象）=$T/compare-test-full-spread-new.json" \
  --test-decode "基线·三字段：spread vs argmax=$T/decode-test-minimal.json" \
  --llm $T/llm-compare-baseline.json \
  --ablation "基线（diag/ablation500：旧验证集文件顺序前 500 行）=$R/runs/diag/ablation500/report.json" \
  --selection-cohort "基线（select2，旧验证集全集）=$COHORT_OLD" --selection-cohort "基线（select1，旧验证集 3000 行）=$COHORT_OLD1" \
  --reference "基线 best-round1（CPU 导出；当前代码检查）=$T/reference-base-round1" \
  --walltime $T/walltime.tsv --notes $T/crosscheck.md
echo SAMPLE-DONE
