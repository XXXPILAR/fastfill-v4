"""One self-contained HTML report (Simplified Chinese, tables only, no external resources) from the analysis JSONs.

    python report.py --out report.html [--inputs inputs.json] --summary "最终臂=runs/autorun-X/SUMMARY.md" ... \
        --val-run "基线·三字段·spread=S.json" ... --val-compare "最终 vs 基线·三字段·spread·全部对象=C.json" ... \
        [--val-control "最终 vs 对照·三字段·spread·全部对象=C.json" ...] --val-decode "最终·三字段：spread vs argmax=C.json" ... \
        [--test-run ... --test-compare ... --test-control ... --test-decode ...] [--llm L.json] \
        [--ablation "最终=ablation/report.json" ... --ablation-sample sample.json] \
        [--selection-cohort "最终（select2）=validation-X-all.jsonl" ...] [--reference "最终=DIR" ...] \
        [--walltime walltime.tsv] [--notes NOTES.md]

Inputs: stratify.py / compare.py / llm_compare.py outputs, diag_ablation.py report.json, roomgenbench --reference-check
JSONs (one directory per checkpoint), the autopilot SUMMARY.md files (``[LABEL=]PATH``), run_plan's walltime.tsv and
isambard_inputs.py's inputs.json (the two arms, which one is final and why, data / model provenance, missing LLM modes).
Every number and every "all / none" statement shown is read from these files.

Decisions come ONLY from the validation comparisons (``--val-compare``: final vs baseline; ``--val-control``: final vs
control, the evidence for the choice of the final arm; ``--val-decode``: spread vs argmax). Test, LLM, ablation and
RoomGenBench sections are labelled report-only and never feed a decision. Significance:
primary = paired room-bootstrap 95% CI of the mean per-room difference (direction from it); robustness = Holm-corrected
sign test; disagreements are flagged; pooled object-weighted differences are descriptive only. A final-vs-baseline
comparison whose two sides carry the same checkpoint is flagged as a placeholder (and "differences are 0" when both
sides are also the same evaluate output). A ``--selection-cohort`` label starts with 最终 or 基线, the checkpoint that
cohort selected; the selection-bias note names a direction only when one checkpoint's cohorts hold the validation
scenes and the other's were given and do not.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
import math
from pathlib import Path
import re

NAMES = {"bottom_center_error_m": "位置误差 (m) ↓", "log_size_error": "log 尺寸误差 ↓", "yaw_error_rad": "朝向误差 (rad) ↓",
         "bev_iou": "BEV IoU ↑", "log_size_error_plain_convention": "log 尺寸误差·原约定 ↓",
         "yaw_error_rad_plain_convention": "朝向误差·原约定 (rad) ↓", "object_collision_rate": "物体碰撞率 ↓",
         "object_hard_violation_rate": "物体硬约束违例率 ↓", "gt_object_collision_rate": "真值物体碰撞率",
         "room_clean_rate": "洁净房间率（有布局）↑", "room_collision_rate": "有碰撞房间率 ↓", "gt_room_clean_rate": "真值洁净房间率",
         "gt_room_collision_rate": "真值有碰撞房间率", "interface_pass_rate": "接口通过率 ↑", "strict_ok_rate": "严格几何通过率 ↑",
         "clean_room_rate_incl_failures": "洁净房间率（失败计不洁净）↑",
         "baseline:room_center_position": "基线·房间中心 (m)", "baseline:category_mean_position": "基线·类别均值位置 (m)",
         "baseline:category_median_size": "基线·类别中位尺寸", "baseline:uniform_yaw": "基线·均匀随机朝向 (rad)",
         "cell_ce": "格子交叉熵 ↓", "cell_entropy": "预测格子熵", "p_gt": "p(真值格子) ↑", "argmax_hit": "argmax 命中率 ↑",
         "xy_err_m": "argmax XY 误差 (m) ↓", "size_err": "尺寸误差 ↓", "yaw_ce": "朝向分类交叉熵 ↓", "yaw_err_rad": "朝向误差 (rad) ↓",
         "gt_rank": "真值格子名次 ↓", "cell_tv_vs_a": "格子分布 TV 距离（相对 a）", "yaw_tv_vs_a": "朝向分布 TV 距离（相对 a）"}
KINDS = {"rule": "目标选择规则", "place": "声明位置", "family": "来源族", "objects": "物体数分箱", "all": "全部"}
RULE_NAMES = {"frozen_prep": "frozen_prep（旧对象）", "wall_anchor": "wall_anchor（新·墙面）",
              "support_inside_parent": "support_inside_parent（新·内层）", "support_on_added_parent": "support_on_added_parent（新·新增父物体上）"}
ABLATION = {"b": "b 房间尺寸换成另一房间的", "c": "c 房间类型换成另一房间的", "d": "d 描述换成类别名", "e": "e 请求顺序打乱"}
MODES = {"prompt": "单次询问", "harness": "修复循环（校验后带问题清单重问）", "structured": "单次询问（结构化提示）",
         "structured-harness": "修复循环（结构化提示，校验后重问）"}
CAUSES = {"IncompleteRead": "中继传输错误（HTTP 响应体读取不完整，IncompleteRead）：传输层失败，不是模型给出的无效布局",
          "over_capacity": "超出模型容量 max_objects", "other": "其他推理失败"}
VERDICT = {"a_better": ("A 更好", "good"), "a_worse": ("A 更差", "bad"), None: ("无显著差异", "")}
CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1f;--muted:#6b6b70;--line:#e2e2e0;--card:#fff;--accent:#2f5d9e;--good:#1f7a4d;--bad:#a8322d;--warn:#8a5a00;--warnbg:#fff6e0}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151517;--fg:#ececee;--muted:#a0a0a8;--line:#333338;--card:#1d1d20;--accent:#8fb3ec;--good:#6cc79a;--bad:#ef8b85;--warn:#e7c06a;--warnbg:#2b2416}}
:root[data-theme="dark"]{--bg:#151517;--fg:#ececee;--muted:#a0a0a8;--line:#333338;--card:#1d1d20;--accent:#8fb3ec;--good:#6cc79a;--bad:#ef8b85;--warn:#e7c06a;--warnbg:#2b2416}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Noto Sans CJK SC",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:1.6em;margin:.2em 0}h2{font-size:1.25em;margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{font-size:1.05em;margin-top:1.6em}p,li{max-width:80ch}.muted{color:var(--muted)}code{font-size:.9em;background:var(--card);border:1px solid var(--line);border-radius:4px;padding:0 .3em;word-break:break-all}
.wrap{overflow-x:auto;margin:.6em 0 1.2em;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.88em;font-variant-numeric:tabular-nums}th,td{padding:.45em .7em;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{background:var(--bg);font-weight:600}tr:last-child td{border-bottom:0}
.ci{color:var(--muted);font-size:.85em}.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
.flag{background:var(--warnbg);color:var(--warn);border-left:3px solid var(--warn);padding:.6em .9em;border-radius:4px;margin:1em 0}
.tag{display:inline-block;font-size:.75em;font-weight:600;padding:.05em .5em;border-radius:4px;border:1px solid currentColor;margin-left:.5em;vertical-align:middle}
details{margin:.8em 0}summary{cursor:pointer;color:var(--accent)}nav a{color:var(--accent);margin-right:1em;text-decoration:none}
"""
REPORT_ONLY = "<span class='tag muted'>仅报告，不参与决策</span>"
DECISION = "<span class='tag good'>决策依据：验证集</span>"


def esc(value):
    return html.escape(str(value))


def num(x, digits=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{digits}g}" if abs(x) and abs(x) < 1e-3 else f"{x:.{digits}f}"


def cell(stat, digits=3):
    """mean [lo, hi] from {"mean", "ci95"} or [mean, lo, hi, ...]."""
    if stat is None:
        return "—"
    mean, lo, hi = (stat["mean"], *stat["ci95"]) if isinstance(stat, dict) else stat[:3]
    return f'{num(mean, digits)} <span class="ci">[{num(lo, digits)}, {num(hi, digits)}]</span>'


def pval(p):
    return "—" if p is None else "&lt;1e-300" if p == 0 else f"{p:.2g}"


def table(head, rows):
    out = "<div class='wrap'><table><thead><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>"
    return out + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def markdown(text):
    """The subset SUMMARY.md uses: headings, pipe tables, bullets, paragraphs, `code`, **bold**."""
    inline = lambda s: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(s)))
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")) + 2, 6)
            out.append(f"<h{level}>{inline(line.lstrip('#').strip())}</h{level}>")
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in block[1:] if not all(set(c) <= set("-: ") for c in r)]
            out.append(table([inline(c) for c in block[0]], [[inline(c) for c in r] for r in body]))
            continue
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


def load(spec):
    label, _, path = spec.rpartition("=")  # labels may hold "=", paths do not
    return label, json.loads(Path(path).read_text()), path


def short_ckpt(path):
    return f"{Path(path).parent.name}/{Path(path).name}" if path else "—"


def rows_info(path, _cache={}):
    """(scene ids, objects whose selection_rule is not frozen_prep) of a rows JSONL file."""
    if path not in _cache:
        ids, new = set(), 0
        with open(path) as stream:
            for line in stream:
                if line.strip():
                    prov = json.loads(line)["provenance"]
                    ids.add(prov.get("scene_id"))
                    new += sum(((e or {}).get("selection_rule") or "frozen_prep") != "frozen_prep" for e in prov.get("field_evidence") or [])
        _cache[path] = ids, new
    return _cache[path]


def scene_ids(path):
    return rows_info(path)[0]


def overlaps(ids, cohorts):
    """(label, path, scenes of `ids` in the cohort) for every cohort holding any of them."""
    return [(label, path, len(ids & scene_ids(path))) for label, path in cohorts if ids & scene_ids(path)]


def cohort_flags(what, ids, cohorts):
    return "".join(f"<div class='flag'><b>选择队列重叠：</b>{esc(what)} {len(ids)} 个场景中有 {k} 个在「{esc(label)}」的检查点选择队列 "
                   f"<code>{esc(Path(path).name)}</code>（{len(scene_ids(path))} 个场景，非 frozen_prep 规则物体 {rows_info(path)[1]} 个）内："
                   "这些行参与了挑选该检查点，该检查点在这些行上不是留出评测。</div>" for label, path, k in overlaps(ids, cohorts))


ROLES = ("最终", "基线")  # a --selection-cohort label starts with the checkpoint it selected


def bias_note(what, ids, cohorts):
    """Selection-bias statement for the scenes `ids`, from each checkpoint's cohorts: a direction only when one
    checkpoint's cohorts hold these scenes and the other's were given and do not."""
    given = {r: [(l, p) for l, p in cohorts if l.startswith(r)] for r in ROLES}
    hit = {r: overlaps(ids, given[r]) for r in ROLES}
    cover = lambda r: "、".join(f"「{esc(l)}」{k}/{len(ids)}" for l, _, k in hit[r])
    if all(hit.values()):
        version = {r: {m.group(1) for _, p, _ in hit[r] if (m := re.match(r"validation-([0-9a-f]{12})-", Path(p).name))} for r in ROLES}
        saw = [r for r in ROLES if any(rows_info(p)[1] for _, p, _ in hit[r])]
        return (f"{esc(what)}：两个检查点都在这些场景上被选择过（最终 {cover('最终')}；基线 {cover('基线')}"
                + (f"；不同数据版本：源验证文件 sha256 前缀 最终 {'/'.join(sorted(version['最终']))}，基线 {'/'.join(sorted(version['基线']))}"
                   if all(version.values()) and not version["最终"] & version["基线"] else "")
                + "），选择偏差的方向无法确定。选择队列中的新对象（非 frozen_prep 规则）："
                + "；".join(f"「{esc(l)}」{rows_info(p)[1]} 个" for r in ROLES for l, p, _ in hit[r])
                + (f"：只有{saw[0]}的选择见过新对象。" if len(saw) == 1 else "。"))
    for r, o in (ROLES, ROLES[::-1]):
        if hit[r]:
            return (f"{esc(what)}：只有{r}的选择队列含这些场景（{cover(r)}；{o}的选择队列 0/{len(ids)}）：这些行对{r}不是留出评测，比较可能偏向{r}。"
                    if given[o] else f"{esc(what)}：{r}的选择队列含这些场景（{cover(r)}）；未提供{o}的选择队列，选择偏差的方向无法判断。")
    return ""


def run_rows(runs):
    rows = []
    for split, label, s in runs:
        m, c = s["meta"], s["checks"]
        rows.append([split, esc(label), f"<code>{esc(short_ckpt(m['checkpoint']))}</code>",
                     f"<code>{esc(Path(m['rows']).parent.name)}/{esc(Path(m['rows']).name)}</code>", esc(m["projection"]),
                     esc(m.get("grid_decode") or "—"), "头输出重解码" if m.get("head_outputs") else esc(m.get("baseline") or "—"),
                     f"<code>{esc((m.get('data_sha256') or '')[:12])}</code>",
                     f"<code>{esc((m.get('forward_implementation_sha256') or '—')[:12])}</code>",
                     f"<code>{esc((m.get('implementation_sha256') or '—')[:12])}</code>",
                     f"{c['requests_with_layout']}/{c['requests']}",
                     '<span class="good">是</span>' if c["reproduces_report_exactly"] else '<span class="bad">否</span>',
                     str(c["rooms_validation_differs_from_stored"])])
    return table(["集合", "运行", "检查点", "数据", "投影", "解码", "来源", "数据 sha256", "前向代码", "解码/评分代码",
                  "有布局/请求", "逐位复现 report.json", "校验器漂移房间"], rows)


def overall(s, metric):
    return s["strata"]["all"]["all"]["metrics"].get(metric)


def placeholder(c):
    """A final-vs-baseline comparison whose two sides carry the same checkpoint."""
    return c["meta"]["a"].get("checkpoint") == c["meta"]["b"].get("checkpoint")


def same_output(c):
    """Both sides are the same evaluate output: every difference is 0."""
    return all(c["meta"]["a"].get(k) == c["meta"]["b"].get(k) for k in ("eval_dir", "outcomes"))


def verdict_cell(v, a="A"):
    text, cls = VERDICT[v["verdict"]["primary"]]
    return f'<span class="{cls}">{text.replace("A", esc(a))}</span>' if cls else text


def compare_table(c, final_vs_baseline=False):
    a, b = c["meta"]["a"]["label"], c["meta"]["b"]["label"]
    rows = []
    for metric, v in c["metrics"].items():
        st, lo, hi = v["sign_test"], *v["room_mean_diff_ci95"]
        robust = VERDICT[v["verdict"]["sign_test"]][0]
        rows.append([NAMES.get(metric, metric), num(v["a"]), num(v["b"]), num(v.get("a_room_mean")), num(v.get("b_room_mean")),
                     f"{num(v['room_mean_diff'], 4)} <span class='ci'>[{num(lo, 4)}, {num(hi, 4)}]</span>", verdict_cell(v),
                     f"{st['a_greater']}/{st['b_greater']}/{st['ties']}", pval(st["p_holm"]),
                     robust if v["verdict"]["agree"] else f"<span class='bad'>⚠ 不一致：{robust}</span>",
                     f"{num(v['pooled_diff'], 4)} <span class='ci'>[{num(v['pooled_diff_ci95'][0], 4)}, {num(v['pooled_diff_ci95'][1], 4)}]</span>",
                     str(v["rooms"])])
    r, where = c["rooms"], c["meta"].get("where") or {}
    note = (f"<p class='muted'>A = {esc(a)}，B = {esc(b)}；差值 = A − B；合并 = 物体加权均值，房间均值 = 先在房间内平均再对房间平均"
            f"（A、B 房间均值之差即主判据的房间均值差）。房间 {r['compared']}"
            + (f"（另有 {r['excluded_by_room_filter']} 个房间因物体数 &gt; {c['meta']['max_objects']} 被排除）" if c["meta"].get("max_objects") else "")
            + f"：两者都有布局 {r['layout_both']}，仅 A 有布局 {r['layout_only_a']}，仅 B 有布局 {r['layout_only_b']}，都没有 {r['layout_neither']}；"
            "配对指标只在两者都有布局的房间上计算" + (f"；筛选 {esc(where)}" if where else "") + f"；bootstrap {c['meta']['boot']} 次（房间为单位，配对）。</p>")
    if final_vs_baseline and placeholder(c):
        note += (f"<p class='bad'>占位：A、B 是同一个检查点 <code>{esc(short_ckpt(c['meta']['a'].get('checkpoint')))}</code>，差值不说明最终模型"
                 + ("；A 与 B 也是同一个评测输出：差值必为 0" if same_output(c) else "") + "。</p>")
    elif same_output(c):
        note += "<p class='muted'>A 与 B 是同一个评测输出：差值必为 0。</p>"
    if not rows:
        return note + "<p class='muted'>没有符合筛选条件的物体（例如旧数据没有新增对象），无可比较的指标。</p>"
    return note + table(["指标", "A（合并，物体加权）", "B（合并，物体加权）", "A 房间均值", "B 房间均值", "房间均值差 [95% CI]（主判据）",
                         "结论（主判据）", "符号检验 A&gt;B/A&lt;B/平", "p (Holm)", "稳健性：符号检验", "合并差值 [95% CI]（物体加权，仅描述）",
                         "房间"], rows)


def strata_tables(s, metrics):
    out = []
    for kind in ("rule", "place", "family", "objects"):
        entries = s["strata"].get(kind, {})
        rows = [[esc(RULE_NAMES.get(v, v)), str(e["rooms"]), str(e.get("rooms_with_layout", "—")), str(e["objects"])]
                + [cell(e["metrics"].get(m)) for m in metrics] for v, e in entries.items()]
        out.append(f"<h3>按{KINDS[kind]}</h3>" + table([KINDS[kind], "房间", "有布局房间", "物体"] + [NAMES[m] for m in metrics], rows))
    return "".join(out)


STRATA_METRICS = ("bottom_center_error_m", "baseline:room_center_position", "log_size_error", "yaw_error_rad",
                  "object_collision_rate", "room_clean_rate")


def results_table(runs):
    """Model metrics per run next to the trivial baselines of the first run's objects."""
    pairs = (("bottom_center_error_m", ("baseline:room_center_position", "baseline:category_mean_position")),
             ("log_size_error", ("baseline:category_median_size",)), ("yaw_error_rad", ("baseline:uniform_yaw",)), ("bev_iou", ()),
             ("object_collision_rate", ()), ("room_clean_rate", ()))
    first = runs[0]
    rows = [[NAMES[metric]] + [cell(overall(s, metric)) for _, s in runs]
            + ["<br>".join(f"{NAMES[b]}：{cell(overall(first[1], b))}" for b in baselines) or "—"] for metric, baselines in pairs]
    return table(["指标"] + [esc(label) for label, _ in runs] + [f"平凡基线（{esc(first[0])} 的同一批物体）"], rows)


def full_set_table(runs):
    """Every room of each run, including those beyond the baseline's max_objects (not paired; room sets differ)."""
    rows = []
    for label, s in runs:
        big = s["strata"].get("objects", {}).get("129+", {})
        rows.append([esc(label), f"{s['checks']['requests_with_layout']}/{s['checks']['requests']}", str(s["checks"]["over_capacity_requests"]),
                     f"{big.get('rooms_with_layout', 0)}/{big.get('rooms', 0)}"]
                    + [cell(overall(s, m)) for m in ("bottom_center_error_m", "log_size_error", "yaw_error_rad", "object_collision_rate", "room_clean_rate")])
    return table(["运行", "有布局/请求", "超出容量", "&gt;128 物体房间：有布局/总数", NAMES["bottom_center_error_m"], NAMES["log_size_error"],
                  NAMES["yaw_error_rad"], NAMES["object_collision_rate"], NAMES["room_clean_rate"]], rows)


def decision_table(compares, final_vs_baseline):
    rows = []
    for label, c, _ in compares:
        by = {key: [NAMES.get(m, m) for m, v in c["metrics"].items() if v["verdict"]["primary"] == key] for key in ("a_better", "a_worse", None)}
        disagree = [f"{NAMES.get(m, m)}（符号检验：{VERDICT[v['verdict']['sign_test']][0]}）" for m, v in c["metrics"].items() if not v["verdict"]["agree"]]
        rows.append([esc(label) + ("<br><span class='bad'>占位：A、B 同一检查点</span>" if final_vs_baseline and placeholder(c) else ""),
                     f"{esc(c['meta']['a']['label'])} / {esc(c['meta']['b']['label'])}", str(c["rooms"]["layout_both"]),
                     f"<span class='good'>{'、'.join(by['a_better'])}</span>" if by["a_better"] else "—",
                     f"<span class='bad'>{'、'.join(by['a_worse'])}</span>" if by["a_worse"] else "—", str(len(by[None])),
                     f"<span class='bad'>⚠ {'；'.join(disagree)}</span>" if disagree else "无"])
    return table(["验证集比较", "A / B", "配对房间", "A 显著更好（主判据）", "A 显著更差（主判据）", "无显著差异的指标数", "与符号检验不一致"], rows)


def section_reference(refs):
    out = []
    for label, directory in refs:
        rows, check_rows, versions = [], [], set()
        for path in sorted(Path(directory).glob("*.json")):
            r = json.loads(path.read_text())
            versions.add(r["schema_version"])
            for place in ("all", "floor", "wall", "on_object"):
                e = r["errors"].get(place)
                if e:  # box_equivalent_* exist in schema v1 and v2; v2 adds matching (by_id = unmatched) and a per-place yaw baseline
                    rows.append([esc(r["scene_key"]) if place == "all" else "", {"all": "全部", "floor": "地面", "wall": "墙面", "on_object": "物体上"}[place],
                                 str(e["objects"]), num(e["position_error_m"]), num(e.get("position_error_m_by_id")),
                                 num(e["room_center_position_error_m"]), num(e["box_equivalent_log_size_error"]),
                                 num(e["box_equivalent_yaw_error_rad"]), num(e.get("uniform_yaw_baseline_error_rad", r["uniform_yaw_baseline_error_rad"]))])
            f, g = r["checks"]["fastfill"], r["checks"]["ground_truth"]
            coll = lambda x: x["by_code"].get("collision", {}).get("violation", 0)
            check_rows.append([esc(r["scene_key"]), str(r["objects"]), "通过" if f["ok"] else "未通过", str(f["counts"]["violation"]), str(coll(f)),
                               "通过" if g["ok"] else "未通过", str(g["counts"]["violation"]), str(coll(g)), str(r.get("ground_truth_tilted_objects", "—"))])
        out.append(f"<h3>{esc(label)} <span class='muted'>（{esc(directory)}；{esc(', '.join(sorted(versions)))}）</span></h3>"
                   + table(["房间", "真值位置", "物体", "位置误差 (m)", "位置误差·按 ID (m)", "房间中心基线 (m)", "log 尺寸（盒等价）",
                            "朝向（盒等价, rad）", "均匀随机朝向基线 (rad)"], rows)
                   + table(["房间", "物体", "FastFill 校验", "违例", "碰撞对", "真值校验", "真值违例", "真值碰撞对", "真值倾斜物体"], check_rows))
    return "".join(out)


def sample_name(path, s):
    return f"消融样本（{Path(path).parent.name}，抽自 {Path(s['new']).parent.name}/{Path(s['new']).name}）"


def section_ablation(ablations, sample, s, cohorts):
    out = []
    if s and cohorts:
        out.append(cohort_flags(sample_name(sample, s), set(s["scene_ids"]), cohorts))
    keys = ("cell_ce", "p_gt", "argmax_hit", "xy_err_m", "size_err", "yaw_err_rad")
    for label, a, path in ablations:
        summary, diffs, prior = a["summary_mean_ci95_count"], a["diff_vs_a_mean_ci95"], a["model_a_minus_prior_cell_ce"]
        best = max(prior.items(), key=lambda kv: kv[1][0])  # the prior closest to the model: the largest difference
        rows = [["a 原始输入"] + [cell(summary["a"].get(k)) for k in keys]]
        rows += [[ABLATION[v]] + [cell(summary[v].get(k)) for k in keys] for v in "bcde"]
        drows = [[ABLATION[v[0]]] + [cell(diffs[v].get(k), 4) for k in keys] for v in diffs]
        prior_a = summary["a"]
        out.append(f"<h3>{esc(label)}</h3><p class='muted'>检查点 <code>{esc(a['checkpoint'])}</code>；{a['rows']} 个验证房间"
                   f"（{prior_a['cell_ce'][3]} 个位置可评物体）；改变的房间 b={a['changed_rooms']['b']} c={a['changed_rooms']['c']}；"
                   f"判据交叉核对最大相对误差 {num(a['criterion_crosscheck_max_rel_err'], 2)}；均匀分布交叉熵 {num(a['prior']['uniform_ce'])}。</p>"
                   + table(["变体"] + [NAMES[k] for k in keys], rows)
                   + "<p>变体减 a（95% CI，房间 bootstrap）：</p>" + table(["变体"] + [NAMES[k] for k in keys], drows)
                   + f"<p>模型格子交叉熵减输入无关先验（训练集频率，{a['prior']['train_rows']} 行）：最接近模型的先验 <code>{esc(best[0])}</code>"
                   f" 差值 {cell(best[1], 4)}；负值 = 模型优于该先验。{len(prior)} 种先验中差值的 CI 上界最大为 "
                   f"{num(max(v[2] for v in prior.values()), 4)}。</p>")
    return "".join(out)


def section_llm(llm):
    out = []
    for c in llm["selection_cohort"]:
        out.append(f"<div class='flag'><b>选择队列重叠：</b>这 {c['rows']} 行中有 {c['rows_in_cohort']} 行的场景出现在检查点选择队列"
                   f" <code>{esc(Path(c['file']).name)}</code>（{c['cohort_rows']} 个场景）里：用该队列选出的 FastFill 检查点在这些行上不是留出评测，LLM 智能体不受影响。</div>")
    cost = {c["method"]: c for c in llm["cost"]}
    shown = {}
    for m, c in cost.items():
        loop = c.get("mode") and "harness" in c["mode"]
        shown[m] = m + (f"（修复循环，{c['calls_per_room']:.2f} 次调用/房间）" if loop else "")
    failures = []
    for m, c in cost.items():
        fr = c.get("failed_rooms")
        if not fr:
            continue
        p = fr["p_all_in_largest_if_size_independent"]
        sizes = "、".join(f"{x['objects']}（第 {x['rank']} 名）" for x in sorted(fr["rooms"], key=lambda x: x["rank"]))
        failures.append(f"{esc(m)}：{len(fr['rooms'])} 个房间无布局——" + "；".join(f"{esc(CAUSES.get(k, k))} ×{n}" for k, n in c["failure_causes"].items())
                        + f"。这些房间的物体数（{fr['of_rows']} 行中按物体数从多到少的名次）：{sizes}；全部落在物体数最多的 {fr['largest_rooms_holding_all']} 个房间内，"
                        f"若失败与房间大小无关，{len(fr['rooms'])} 个失败全落在其中的概率为 {pval(p)}"
                        + ("：失败集中在最大的房间" + ("，可能是与房间规模相关的中继超时（传输被截断，而非模型给出无效布局）" if "IncompleteRead" in c["failure_causes"] else "")
                           if p < .05 else "") + "。")
    if failures:
        out.append("<p><b>无布局的房间及原因</b>（LLM 取自 predictions.jsonl 的 error 字段；名次 1 = 物体最多，并列取同一名次）：</p><ul>"
                   + "".join(f"<li>{x}</li>" for x in failures) + "</ul>")
    rows = []
    for m, c in cost.items():
        tokens = c.get("tokens")
        rows.append([esc(m), esc(c.get("model") or "本地 Qwen3-8B + 几何头"), esc(MODES.get(c.get("mode"), "一次前向 + 解码")),
                     esc(c.get("reasoning_effort") or "—"), str(llm["layouts"][m]), num(c["calls_per_room"], 2), num(c.get("mean_latency_s"), 2),
                     num(c.get("p95_latency_s"), 2), num(c.get("mean_latency_s_layout_rooms"), 2) + f" <span class='ci'>（{c.get('latency_rooms_layout', '—')} 房间）</span>",
                     "—" if not tokens else f"{tokens['total_tokens']:,}", "—" if not tokens else f"{c['tokens_per_room']:,.0f}"])
    no_tokens = [m for m, c in cost.items() if c["kind"] == "llm" and not c.get("tokens")]
    source = lambda c: ("本运行" if c["latency_source"] == "this run" else f"在线运行 <code>{esc(Path(c['latency_source']).name)}</code>") \
        + f"（{esc(c.get('latency_grid_decode') or '—')} 解码）"
    out.append("<h3>成本</h3>" + table(["方法", "模型", "模式", "推理强度", "有布局房间", "调用/房间", "平均延迟 (s)", "p95 延迟 (s)",
                                       "平均延迟·仅有布局房间 (s)", "总 tokens", "tokens/房间"], rows)
               + "<p class='muted'>LLM 延迟取自 predictions.jsonl 的逐房间 latency_s（全部房间的均值即 summary.json 的 mean_latency_s）："
               "每房间全部 API 调用的墙钟时间，含客户端限速等待、重试与退避（修复循环还含修复调用）。FastFill 延迟为在线评测记录的 fastfill_latency_ms"
               "（一次 GPU 前向 + 解码）：" + "；".join(f"{esc(m)} 取自{source(c)}" for m, c in cost.items() if c["kind"] == "fastfill")
               + "。“仅有布局房间”只平均给出布局的房间。" + (f"summary 未记录 token 用量的方法：{esc('、'.join(no_tokens))}。" if no_tokens else "") + "</p>")
    meta = llm["meta"]["methods"]
    sha = lambda x: f"<code>{esc((x or '—')[:12])}</code>"
    out.append("<details><summary>复现检查与代码哈希（llm_compare 拒绝数据或 FastFill 解码/评分代码不同的方法）</summary>"
               + table(["方法", "评测目录", "检查点", "解码", "来源", "数据 sha256", "前向代码", "解码/评分代码", "逐位复现 report.json"],
                       [[esc(m), f"<code>{esc(Path(x['eval_dir']).name)}</code>", f"<code>{esc(short_ckpt(x.get('checkpoint')))}</code>",
                         esc(x.get("grid_decode") or "—"), esc(x.get("baseline") or "—"), sha(x.get("data_sha256")),
                         sha(x.get("forward_implementation_sha256")), sha(x.get("implementation_sha256")),
                         '<span class="good">是</span>' if llm["checks"][m] else '<span class="bad">否</span>'] for m, x in meta.items()])
               + "</details>")
    views = (("common", f"口径一·共同房间：只看每种方法都给出布局的房间（{llm['rows'] - llm['common_rooms']} 个房间因任一方法无布局而对所有方法排除）"),
             ("failures_as_failures", "口径二·失败计入：全部房间，无布局的房间按下方退化规则计分"))
    for name, title in views:
        sc = llm["scenarios"][name]
        metrics = list(next(iter(sc["methods"].values())))
        out.append(f"<h3>{esc(title)}：{sc['rooms']} 个房间</h3>"
                   + table(["方法"] + [NAMES[m] for m in metrics], [[esc(shown[k])] + [cell(v.get(m)) for m in metrics] for k, v in sc["methods"].items()]))
        rows = []
        for pair, stats in sc["pairs"].items():
            for metric, v in stats.items():
                lo, hi = v["room_mean_diff_ci95"]
                robust = VERDICT[v["verdict"]["sign_test"]][0].replace("A", "LLM")
                rows.append([esc(pair), NAMES[metric], f"{num(v['room_mean_diff'], 4)} <span class='ci'>[{num(lo, 4)}, {num(hi, 4)}]</span>",
                             verdict_cell(v, "LLM"), f"{v['sign_test']['a_greater']}/{v['sign_test']['b_greater']}", pval(v["sign_test"]["p_holm"]),
                             robust if v["verdict"]["agree"] else f"<span class='bad'>⚠ 不一致：{robust}</span>"])
        out.append("<details><summary>配对比较（差值 = LLM − FastFill；主判据 = 房间均值差的配对 bootstrap CI；Holm 校正在每一对内的指标间）</summary>"
                   + table(["配对", "指标", "房间均值差 [95% CI]", "结论（主判据）", "符号检验 LLM&gt;/&lt;", "p (Holm)", "稳健性：符号检验"], rows) + "</details>")
    out.append(f"<p class='muted'>退化规则（口径二）：{esc(llm['meta']['fallback'])}。</p>")
    return "".join(out)


def section_arms(inp):
    """The two autopilot arms, which one is final and why, and where data and models come from (isambard_inputs.py)."""
    out = ("<div class='flag'><b>演练（DRYRUN）：</b>未下载；基线、数据与队列文件的固定 sha256 只记录未强制"
           f"（{len(inp['pins']['mismatches'])} 处不符），本报告的数字不说明任何真实模型。</div>" if inp["dry_run"] else "")
    rows = [["<b>最终</b>" if x["role"] == "final" else "对照", f"<code>{esc(x['dir'])}</code>", f"<code>{esc(x['run'])}</code>",
             num(x["yaw_cls"], 2), str(x["max_objects"]), str(x["global_batch"]), f"{x['steps']}（{num(x['epochs_supervised'])} 轮）",
             esc(x["phase"]), esc("；".join(x["failures"]) or "无"), f"<code>{esc(short_ckpt(x['best_checkpoint']))}</code>",
             f"{x['best_step']}（{num(x['best_epochs'])} 轮）", num(x["best_score"], 4), f"<code>{esc(x['select2_cohort_sha256'][:12])}</code>",
             f"<code>{esc(x['select2_implementation_sha256'][:12])}</code>", esc(x.get("uploaded_model") or "—")]
            for x in inp["arms"]]
    ranking = "".join(f"<li>{'最终' if x['role'] == 'final' else '对照'}：" + "；".join(f"<code>{esc(short_ckpt(c))}</code> {num(s, 4)}"
                                                                                for c, s in x["select2_ranking"]) + "</li>" for x in inp["arms"])
    return (out + "<h3>1.1 两臂与最终臂的选择</h3>"
            + table(["角色", "autopilot 目录", "运行", "yaw_cls", "max_objects", "全局批量", "整个运行的步数（配置；按有效样本的轮数）", "阶段",
                     "失败项", "select2 最优检查点", "该检查点的步数（轮数）", "select2 分数 ↓", "select2 队列 sha256",
                     "select2 评测代码 sha256", "上传"], rows)
            + f"<p><b>规则：</b>{esc(inp['rule'])}</p><details><summary>两臂的 select2 排名（分数越低越好）</summary><ul>{ranking}</ul></details>"
            + "<h3>1.2 数据与模型来源</h3>" + table(["项目", "来源与校验"], [[esc(k), esc(v)] for k, v in inp["provenance"]]))


def missing_llm(inp):
    missing = (inp or {}).get("llm", {}).get("missing") or []
    return "、".join(f"{m}（{MODES.get(m, m)}）" for m in missing)


def walltime_table(path):
    rows = [line.rstrip("\n").split("\t") for line in Path(path).read_text().splitlines() if line.strip()]
    return table(["步骤", "卡", "开始 (UTC)", "结束 (UTC)", "墙钟 (min)", "退出码"],
                 [[esc(r[0]), esc(r[5] if len(r) > 5 else "—"), esc(r[1]), esc(r[2]), num(int(r[3]) / 60, 1), esc(r[4])] for r in rows])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="FastFill v2 评测报告")
    p.add_argument("--summary", action="append", default=[], metavar="[LABEL=]SUMMARY.md")
    p.add_argument("--inputs", type=Path, help="isambard_inputs.py inputs.json: arms, final-arm rule, provenance")
    for split in ("val", "test"):
        for kind in ("run", "compare", "control", "decode"):
            p.add_argument(f"--{split}-{kind}", action="append", default=[], metavar="LABEL=JSON")
    p.add_argument("--llm")
    p.add_argument("--ablation", action="append", default=[])
    p.add_argument("--ablation-sample", type=Path, help="sample_validation.py sample.json of the ablation rows")
    p.add_argument("--selection-cohort", action="append", default=[], metavar="LABEL=ROWS_JSONL")
    p.add_argument("--reference", action="append", default=[])
    p.add_argument("--walltime", type=Path)
    p.add_argument("--notes", type=Path)
    a = p.parse_args(argv)
    val_runs, test_runs = [load(x) for x in a.val_run], [load(x) for x in a.test_run]
    val_compare, val_decode = [load(x) for x in a.val_compare], [load(x) for x in a.val_decode]
    test_compare, test_decode = [load(x) for x in a.test_compare], [load(x) for x in a.test_decode]
    val_control, test_control = [load(x) for x in a.val_control], [load(x) for x in a.test_control]
    inp = json.loads(a.inputs.read_text()) if a.inputs else None
    ablations, refs = [load(x) for x in a.ablation], [tuple(x.rsplit("=", 1)) for x in a.reference]
    cohorts = [tuple(x.rsplit("=", 1)) for x in a.selection_cohort]
    llm = json.loads(Path(a.llm).read_text()) if a.llm else None
    runs = [("验证集", l, s) for l, s, _ in val_runs] + [("测试集", l, s) for l, s, _ in test_runs]
    sample = json.loads(a.ablation_sample.read_text()) if a.ablation_sample else None
    val_ph, test_ph = [c for _, c, _ in val_compare if placeholder(c)], [c for _, c, _ in test_compare if placeholder(c)]
    parts = []
    if val_ph or test_ph:
        where = (["3.1 节比较表", "8.1 节决策表"] if val_ph else []) + (["第 4 节比较表"] if test_ph else [])
        parts.append(f"<div class='flag'><b>占位：</b>{len(val_ph) + len(test_ph)} 个“最终 vs 基线”比较（验证集 {len(val_ph)}，测试集 {len(test_ph)}）"
                     f"的两侧是同一个检查点（{'、'.join(where)}逐个标出），其中 {sum(map(same_output, val_ph + test_ph))} 个两侧还是同一个评测输出"
                     "（差值必为 0）：这些差值不说明最终模型，只用于检验流程。</div>")

    # 1 setup
    rules = {split: Counter() for split in ("验证集", "测试集")}
    for split, _, s in runs:
        if not rules[split]:
            rules[split].update({r: e["objects"] for r, e in s["strata"].get("rule", {}).items()})
    fits = sorted({s["meta"]["baseline_fit"] for _, _, s in runs})
    online = [f"{split}·{l}" for split, l, s in runs if not s["meta"].get("head_outputs")]
    heads_note = (f"{len(runs) - len(online)}/{len(runs)} 个运行由保存的头输出在 CPU 上用同一份代码重解码（evaluate --from-head-outputs，spread 与 argmax 各一次）"
                  + (f"；其余为在线解码的报告：{esc('、'.join(online))}。" if online else "。"))
    val_rows = sorted({s["meta"]["rows"] for _, s, _ in val_runs})
    rows_name = lambda r: f"验证集行（{Path(r).parent.name}/{Path(r).name}）"
    bias = "".join(f"<p>{x}</p>" for r in val_rows if (x := bias_note(rows_name(r), scene_ids(r), cohorts)))
    boots = sorted({s["meta"]["boot"] for _, _, s in runs})
    parts.append("<h2 id='s1'>1 设置与协议</h2>" + (section_arms(inp) + "<h3>1.3 各运行</h3>" if inp else "") + run_rows(runs) + f"""
<ul><li><b>两种解码：</b>argmax = 模型原始输出（每物体取最高分格子）；spread = 防碰撞后处理解码（每个物体取留在房间内、不与已放物体重叠的最可能格子，并服从支撑声明）。两者用同一次前向的头输出，只差解码；第 3、5、8 节的 spread / argmax 均指此义。</li>
<li><b>决策只读验证集：</b>“是否以最终替代基线”和“默认解码 spread / argmax”两项决策只读第 3 节的验证集配对比较；第 4–7 节（测试集、LLM、消融、RoomGenBench）仅报告，不参与任何决策。</li>
<li><b>相同解码代码：</b>{heads_note}compare.py 拒绝解码/评分代码（implementation_sha256）、来源或前向代码不同的两侧；上表“前向代码”“解码/评分代码”两列即这两个哈希。</li>
<li><b>显著性：</b>主判据为逐房间差值均值的配对房间 bootstrap 95% CI（不含 0 即显著，方向取其符号），<b>按指标逐个判断，未做多重比较校正</b>；
稳健性列为逐房间差值的精确双侧符号检验（同一比较内跨指标 Holm 校正）。两者不一致时显式标出：不一致可能来自多重比较（主判据未校正、符号检验已校正），
也可能来自偏态分布（符号检验只看差值的方向/中位数，主判据看的是均值）。合并（物体加权）差值只作描述。</li>
<li><b>最终 vs 基线的主比较</b>只在两模型都能回答的房间上（物体数 ≤ 128，基线的 max_objects），配对指标只用两者都给出布局的房间；仅一方有布局的房间数在每张表上方列出。最终模型在全部房间（含 &gt; 128 物体）上的结果单列于 3.{4 if val_control else 3}。</li>
<li>误差定义与 <code>evaluate.reference_metrics</code> 相同（合法对应匹配、盒等价尺寸/朝向）；逐物体数值由 stratify.py 用评测自身的函数重算，并逐请求、逐位核对 outcomes 与 report.json（上表“逐位复现”）。
平凡基线拟合来源：{esc('、'.join(fits))}。分层置信区间：房间为单位的百分位 bootstrap（{esc('/'.join(map(str, boots)))} 次）。</li>
<li>归属：误差与基线归于标签物体（匹配目标），校验器标记归于检查点名的预测物体；声明位置取完整行的支撑声明（三字段投影不向模型显示）；
选择规则取 <code>provenance.field_evidence[i].selection_rule</code>（缺失记为 frozen_prep）。各集合的规则分布（物体数）："""
                 + "；".join(f"{split}：" + "、".join(f"{esc(r)} {n}" for r, n in c.items()) for split, c in rules.items() if c) + """。</li>
<li>校验器：<code>validation.validate_scene</code>（当前代码），模型布局与标签各跑一次；“洁净房间”= 无硬约束违例（unknown 不计违例）。</li></ul>""")
    if cohorts and val_rows:
        parts.append("".join(cohort_flags(rows_name(r), scene_ids(r), cohorts) for r in val_rows)
                     + (f"<div class='flag'><b>选择偏差：</b>{bias}</div>" if bias else ""))
    for label, _, path in (x.rpartition("=") for x in a.summary):
        if Path(path).is_file():
            parts.append(f"<details><summary>autopilot SUMMARY.md（原文{'：' + esc(label) if label else ''}）</summary>"
                         + markdown(Path(path).read_text()) + "</details>")
    if a.walltime and a.walltime.is_file():
        parts.append("<details><summary>GPU / CPU 步骤墙钟时间（run_plan 记录）</summary>" + walltime_table(a.walltime) + "</details>")

    # 2 interface vs geometry
    metrics = ("interface_pass_rate", "strict_ok_rate", "room_clean_rate", "clean_room_rate_incl_failures", "gt_room_clean_rate",
               "object_collision_rate", "gt_object_collision_rate", "room_collision_rate")
    parts.append("<h2 id='s2'>2 接口通过率 vs 几何通过率</h2><p>接口通过 = 返回了每个请求 ID 恰好一次、尺寸为正的布局；几何通过分两级："
                 "严格（所有硬检查 pass，unknown 也算不通过）与洁净（无硬违例）。真值列是同一校验器对标签本身的结果：是同一校验器下的参照，"
                 "不是上限（模型可能比标签更洁净）。</p>"
                 + table(["集合", "运行"] + [NAMES[m] for m in metrics], [[split, esc(l)] + [cell(overall(s, m)) for m in metrics] for split, l, s in runs]))

    # 3 validation: the decisions' basis
    sec = f"<h2 id='s3'>3 验证集 {DECISION}</h2>"
    if not (val_compare or val_decode):
        sec += "<p class='muted'>（未提供验证集比较：本报告不作决策）</p>"
    sec += "<h3>3.1 最终 vs 基线（两模型都能回答的房间）</h3>" + ("".join(f"<h3>{esc(l)}</h3>" + compare_table(c, True) for l, c, _ in val_compare) or "<p class='muted'>（未提供）</p>")
    sec += "<h3>3.2 默认解码：spread vs argmax</h3>" + ("".join(f"<h3>{esc(l)}</h3>" + compare_table(c) for l, c, _ in val_decode) or "<p class='muted'>（未提供）</p>")
    if val_control:
        sec += ("<h3>3.3 最终 vs 对照（同一数据与配置，只差 yaw_cls；选择最终臂的配对证据）</h3>"
                + "".join(f"<h3>{esc(l)}</h3>" + compare_table(c) for l, c, _ in val_control))
    if val_runs:
        sec += (f"<h3>3.{4 if val_control else 3} 各运行在全部房间上的结果（含 &gt; 128 物体房间；房间集合不同，非配对，仅描述）</h3>" + full_set_table([(l, s) for l, s, _ in val_runs])
                + results_table([(l, s) for l, s, _ in val_runs])
                + "".join(f"<details><summary>分层结果（{esc(l)}）</summary>" + strata_tables(s, STRATA_METRICS) + "</details>" for l, s, _ in val_runs))
    parts.append(sec)

    # 4 test: report only
    sec = f"<h2 id='s4'>4 测试集结果 {REPORT_ONLY}</h2><p class='muted'>平凡基线在同一批有布局的房间、同一批物体上计算。argmax = 模型原始输出；spread = 防碰撞后处理解码。</p>"
    if test_runs:
        sec += (full_set_table([(l, s) for l, s, _ in test_runs]) + results_table([(l, s) for l, s, _ in test_runs])
                + "".join(f"<details><summary>分层结果（{esc(l)}）</summary>" + strata_tables(s, STRATA_METRICS) + "</details>" for l, s, _ in test_runs))
    sec += "".join(f"<h3>{esc(l)}</h3>" + compare_table(c, True) for l, c, _ in test_compare)
    sec += "".join(f"<h3>{esc(l)}</h3>" + compare_table(c) for l, c, _ in test_control)
    sec += "".join(f"<h3>{esc(l)}</h3>" + compare_table(c) for l, c, _ in test_decode)
    parts.append(sec if test_runs or test_compare or test_control or test_decode else sec + "<p class='muted'>（未提供）</p>")

    # 5 LLM, 6 ablation, 7 RoomGenBench: report only
    n_llm = sum(c["kind"] == "llm" for c in llm["cost"]) if llm else 0
    missing = ("<div class='flag'><b>缺失的 LLM 模式：</b>" + esc(missing_llm(inp)) + "：其答案在无法访问的旧服务器上，本节不含它们。"
               "其余 LLM 答案是已有的预测，由当前评测器重新评分，未调用任何 API。</div>" if missing_llm(inp) else "")
    parts.append(f"<h2 id='s5'>5 FastFill vs {n_llm} 种 LLM 智能体（冻结的 LLM 行）{REPORT_ONLY}</h2>" + missing
                 + (section_llm(llm) if llm else "<p class='muted'>（未提供）</p>"))
    parts.append(f"<h2 id='s6'>6 模型是否使用输入：消融 + 输入无关先验 {REPORT_ONLY}</h2><p>同一批验证房间（三字段投影）上改动一个输入、其余不变，看格子分布与误差是否随之变化；"
                 "再与只看类别（及房间类型）的训练集频率先验比较格子交叉熵。</p>"
                 + (section_ablation(ablations, a.ablation_sample, sample, cohorts) or "<p class='muted'>（未提供）</p>"))
    parts.append(f"<h2 id='s7'>7 RoomGenBench 五个房间 vs 真值 {REPORT_ONLY}</h2><p class='muted'>五个房间、单次预测：没有统计功效，只看量级与失败模式；"
                 "同一校验器也跑在真值上作为参照。</p>" + (section_reference(refs) or "<p class='muted'>（未提供）</p>"))

    # 8 limitations and decisions
    lim, seen = [], set()
    for split, label, s in runs:
        c, key = s["checks"], (s["meta"]["eval_dir"], s["meta"]["outcomes"])
        if key in seen:
            continue
        seen.add(key)
        if not c["reproduces_report_exactly"]:
            lim.append(f"{split}·{esc(label)}：逐物体重算未逐位复现 report.json，该运行的数字不可直接与报告对照。")
        if c["rooms_validation_differs_from_stored"]:
            lim.append(f"{split}·{esc(label)}：{c['rooms_validation_differs_from_stored']} 个房间的当前校验器结果与评测时存储的不同（校验器代码已变）；本报告用当前校验器。")
        if c["requests"] - c["requests_with_layout"]:
            lim.append(f"{split}·{esc(label)}：{c['requests'] - c['requests_with_layout']} 个请求没有布局（其中 {c['over_capacity_requests']} 个超出模型容量 "
                       "max_objects），误差只在有布局的房间上计算；接口通过率中计为失败。")
    if any(s["meta"]["projection"] == "minimal" for _, _, s in runs):
        lim.append("三字段投影不向模型显示支撑声明；“声明位置”分层用的是完整行的声明，作为真值位置标签。")
    if missing_llm(inp):
        lim.append(f"第 5 节缺少 LLM 模式 {esc(missing_llm(inp))}（答案不可得）：LLM 对比只含已有答案的模式。")
    if llm:
        lim += [f"LLM 行与检查点选择队列 {esc(Path(c['file']).name)} 重叠 {c['rows_in_cohort']}/{c['rows']}：用该队列选出的 FastFill 检查点在这些行上不是留出评测。"
                for c in llm["selection_cohort"] if c["rows_in_cohort"]]
        lim += [f"第 5 节 {esc(m)}：逐物体重算未逐位复现 report.json，该方法的数字不可直接与其报告对照。" for m, ok in llm["checks"].items() if not ok]
        lim += [f"第 5 节 {esc(c['method'])}：predictions.jsonl 逐房间延迟的均值 {num(c['mean_latency_s'], 4)} s 与 summary.json 的 mean_latency_s "
                f"{num(c['summary_mean_latency_s'], 4)} s 不一致。" for c in llm["cost"]
                if c["kind"] == "llm" and abs(c["mean_latency_s"] - c["summary_mean_latency_s"]) > 1e-6]
    if ablations and sample and (hits := overlaps(set(sample["scene_ids"]), cohorts)):
        lim.append(f"{esc(sample_name(a.ablation_sample, sample))}与检查点选择队列重叠："
                   + "、".join(f"「{esc(l)}」{k}/{len(sample['scene_ids'])}" for l, _, k in hits) + "：这些场景对相应检查点不是留出数据。")
    family_rows = []
    for split, label, s in runs:
        fam = s["strata"].get("family", {})
        worse = [f for f, e in fam.items() if (e["metrics"].get("bottom_center_error_m") or {}).get("mean") is not None
                 and e["metrics"]["bottom_center_error_m"]["mean"] > (e["metrics"].get("baseline:room_center_position") or {}).get("mean", math.inf)]
        family_rows.append([split, esc(label), f"{len(worse)}/{len(fam)}", esc("、".join(worse)) or "—"])
    parts.append("<h2 id='s8'>8 已知局限与决策清单</h2><h3>已知局限</h3><ul>" + "".join(f"<li>{x}</li>" for x in lim) + "</ul>"
                 + "<p>位置误差（点估计）高于房间中心平凡基线的来源族：</p>" + table(["集合", "运行", "来源族数", "这些来源族"], family_rows)
                 + f"<h3>决策 {DECISION}</h3><p>以下两项只读第 3 节验证集比较（主判据 = 房间均值差的配对 bootstrap 95% CI 不含 0，按指标逐个判断，"
                   "未做多重比较校正；稳健性 = Holm 校正的符号检验，只看差值方向/中位数而非均值，与主判据不一致可能来自多重比较或偏态分布）。"
                   "测试集与其它各节不进入此处。</p>" + (f"<div class='flag'><b>选择偏差：</b>{bias}</div>" if bias else "")
                 + "<h3>8.1 是否以最终替代基线（判据：旧对象 frozen_prep 不应显著变差；新对象应显著变好）</h3>"
                 + (decision_table(val_compare, True) if val_compare else "<p class='muted'>（未提供验证集比较：不作决策）</p>")
                 + "<h3>8.2 默认解码 spread 还是 argmax（spread 以碰撞率换取可能的位置/朝向代价）</h3>"
                 + (decision_table(val_decode, False) if val_decode else "<p class='muted'>（未提供验证集比较：不作决策）</p>")
                 + ("<h3>8.3 最终臂 vs 对照臂（最终臂按 select2 分数选出；此处为验证集配对证据）</h3>"
                    + (f"<p>{esc(inp['rule'])}</p>" if inp else "") + decision_table(val_control, False) if val_control else "")
                 + (f"<h3>附注</h3>" + markdown(a.notes.read_text()) if a.notes and a.notes.is_file() else ""))
    nav = "".join(f"<a href='#s{i}'>{i}</a>" for i in range(1, 9))
    page = (f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{esc(a.title)}</title><style>{CSS}</style></head><body><main><h1>{esc(a.title)}</h1>"
            f"<nav>{nav}</nav>" + "\n".join(parts) + "</main></body></html>")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    print(a.out, len(page))


if __name__ == "__main__":
    main()
