"""One Markdown page from the v4 ablation outputs: every arm x decoding mode on the validation sample, the same on
the 300 test rooms and the first 20 rooms GPT-5.6 answered (with GPT-5.6's own layouts), the support head's accuracy and the anchor probe.
Whatever is missing (an arm that did not finish) is left out, not an error.

    python fastfill/v2/ops/v4_report.py runs/ablation-v4      # writes runs/ablation-v4/REPORT.md
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fastfill.v2.ops.remask_ablation import summarize  # noqa: E402

COLUMNS = [("position_m", "位置误差 m", 2), ("log_size", "尺寸 log", 3), ("yaw_rad", "朝向 rad", 3),
           ("footprint_outside", "出界", "%"), ("door_blocking", "挡门", "%"), ("overlap_iou_0.3", "重叠", "%"),
           ("collision_pairs_per_room", "碰撞/房间", 2), ("constraint_pass", "约束通过", "%"),
           ("clean_room_rate", "干净房间", "%"), ("decode_ms", "解码 ms", 0)]


def cell(value, digits):
    if value is None:
        return "–"
    return f"{100 * value:.1f}%" if digits == "%" else f"{value:.{digits}f}"


def table(rows):
    head = "| 模型 | 解码 | " + " | ".join(name for _, name, _ in COLUMNS) + " |"
    lines = [head, "|" + "---|" * (len(COLUMNS) + 2)]
    lines += ["| " + " | ".join([arm, mode] + [cell(metrics.get(key), d) for key, _, d in COLUMNS]) + " |"
              for arm, mode, metrics in rows]
    return "\n".join(lines)


def ablation_rows(root, prefix=""):
    rows = []
    for arm in ("base", "D", "E", "E_nosupport"):
        path = root / f"{prefix}{arm}" / "ablation.json"
        if path.is_file():
            rows += [(arm, mode, m) for mode, m in json.loads(path.read_text()).items() if mode != "ground_truth"]
    return rows


def main(root):
    root, out = Path(root), []
    out.append("# FastFill v4 结果汇总\n")
    rows = ablation_rows(root)
    if rows:
        out += ["## 验证集抽样（同一批房间）\n", table(rows), ""]
    for prefix, title in (("test300_", "测试集 300 个房间（按来源分层抽样），和 GPT-5.6 同一批房间、同一评测代码"),):
        held = ablation_rows(root, prefix)
        gpt = root / f"{prefix}gpt56" / "report.json"
        if gpt.is_file():
            held.append(("GPT-5.6", "单次调用", summarize(json.loads(gpt.read_text()))))
        if held:
            out += [f"## {title}\n", table(held), ""]
    probe = ablation_rows(root, "probe20_")
    gpt = root / "probe20_gpt56" / "report.json"
    if gpt.is_file():
        probe.append(("GPT-5.6", "单次调用", summarize(json.loads(gpt.read_text()))))
    if probe:
        out += ["## GPT-5.6 回答过的 20 个房间（同一评测代码）\n", table(probe), ""]
    support = [(arm, mode, m["support_accuracy"]) for arm, mode, m in rows if (m.get("support_accuracy") or {}).get("all") is not None]
    if support:
        out += ["## 支撑预测准确率（去掉全部支撑声明）\n", "| 模型 | 解码 | 全部 | 地面 | 墙 | 放在家具上 | 个数 |", "|---|---|---|---|---|---|---|"]
        out += [f"| {arm} | {mode} | " + " | ".join(cell(s.get(k), "%") for k in ("all", "floor", "wall", "object"))
                + f" | {s.get('objects')} |" for arm, mode, s in support]
        out.append("")
    anchor = root / "anchor_probe_D.json"
    if anchor.is_file():
        out += ["## anchor_probe（D）\n", "```", json.dumps(json.loads(anchor.read_text()), indent=1, ensure_ascii=False)[:4000], "```", ""]
    (root / "REPORT.md").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/ablation-v4")
