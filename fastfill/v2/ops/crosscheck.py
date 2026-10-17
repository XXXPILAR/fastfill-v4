"""Do two evaluate outcome files of the same rows hold identical layouts? E.g. a GPU-decoded report (phase F, or the
online run that saved the head outputs) vs the CPU decode of the saved head outputs. Prints a markdown table (for
report.py --notes); a missing file is listed, not an error.

    python crosscheck.py --pair "LABEL=A/outcomes.jsonl,B/outcomes.jsonl" ... > crosscheck.md
"""
import argparse
import json
import math
from pathlib import Path


def layouts(path):
    with open(path) as stream:
        return [(o["row"], None if "error" in o else o["raw_prediction"]) for o in map(json.loads, stream)]


def gap(a, b):
    """Largest absolute difference of any position, size or yaw value of two layouts of the same ids."""
    if a is None or b is None:
        return 0. if a is b else math.inf
    p = {o["id"]: o for o in a["objects"]}
    return max([abs(x - y) for o in b["objects"] for k in ("bottom_center_m", "target_size_local_m") for x, y in zip(p[o["id"]][k], o[k])]
               + [abs(p[o["id"]]["yaw_rad"] - o["yaw_rad"]) for o in b["objects"]] + [0.])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True, metavar="LABEL=A,B")
    a = p.parse_args(argv)
    print("## 交叉核对：在线解码的布局（phase F 自己的报告 / 保存头输出的那次运行）vs 头输出的 CPU 重解码（期望逐房间相同）\n")
    print("| 对照 | 行 | 布局完全相同 | 不同房间的最大绝对差 |\n|---|---|---|---|")
    for item in a.pair:
        label, _, paths = item.rpartition("=")
        x, y = paths.split(",")
        if not (Path(x).is_file() and Path(y).is_file()):
            print(f"| {label} | 缺文件 | — | — |")
            continue
        u, v = layouts(x), layouts(y)
        if [r for r, _ in u] != [r for r, _ in v]:
            print(f"| {label} | 行不同 | — | — |")
            continue
        gaps = [gap(s, t) for (_, s), (_, t) in zip(u, v) if s != t]
        print(f"| {label} | {len(u)} | {len(u) - len(gaps)} | {max(gaps) if gaps else 0:.3g} |")


if __name__ == "__main__":
    assert gap({"objects": [{"id": "a", "bottom_center_m": [0, 0, 0], "target_size_local_m": [1, 1, 1], "yaw_rad": 0}]},
               {"objects": [{"id": "a", "bottom_center_m": [0, .5, 0], "target_size_local_m": [1, 1, 1], "yaw_rad": .1}]}) == .5
    main()
