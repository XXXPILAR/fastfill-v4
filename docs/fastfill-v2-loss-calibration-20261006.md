# FastFill v2 损失权重标定（2026-10-06）

本页记录 `fastfill/v2/configs/qwen3_8b_main_world7.json`（及派生的 `qwen3_8b_pilot_1gpu.json`、两份正式配置 `qwen3_8b_main_4gpu_regression.json` / `qwen3_8b_main_3gpu_grid.json`）中几何损失权重的来源；2026-10-07 第二轮的改动（yaw_reg 封顶、grid_residual 位置头权重、正式配置）见文末。`LossConfig` 是严格 dataclass、`run_training` 只接受已知顶层段，配置文件不能携带 `_calibration` 注释键，所以数值和规则写在这里。

## 规则

1. 在未训练基线下计算每一项的逐对象平均值 `B_k`（定义见下）。
2. 以 position 为参照（`λ_position = 1`），`λ_k = round_1sig(B_position / B_k)`，取一位有效数字。
3. 检查每项基线贡献 `λ_k · B_k` 两两之比不超过 2。

## 基线定义（全部由 `GeometryCriterion` 本身在新 L1 定义下计算）

| 项 | 基线预测 | 损失定义（`losses.py`） |
|---|---|---|
| position | 地面多边形形心、z = `floor_z_m`（房间归一化坐标） | 可学习坐标 L1 之和 / 3；`support_parent == floor` 的 z 固定、贡献 0 |
| size | 按 category 取 20,000 个训练房间的逐轴中位数；类别未见则用全局中位数 | log 尺寸 L1 之和 / 3 |
| yaw_cls | 均匀 logits（12 bin） | 对称感知联合最小的交叉熵 = ln 12 |
| yaw_reg | 零残差 | GT bin 残差 Huber（β = 1） |

yaw 有效性按契约 C4 模拟：`front_known` 且非 tilted 且 yaw 有限；对称阶 MultiScan 为 1、其余为 2。磁盘上的行仍是 strict 策略（抽样 2,000 房间中 yaw 有效 0 个），不模拟则 yaw 两项没有样本。

## 数据与命令

- 验证样本：`/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local/validation.jsonl`（8,125 行）中 `random.Random(20261006)` 抽 2,000 房间，24,279 个请求对象；14 个来源，SpatialLM 551 / MansionWorld 276 / IL3D_3dfront 230 / InternScenes_gen 214 / Structured3D 200 / SAGE-10k 121 / 其余 408。
- 训练中位数：同一 RNG 从 `train.jsonl`（124,375 行）抽 20,000 房间，170,875 个完整尺寸标签，5,288 个类别；全局中位数 `[0.381, 0.475, 0.594]` m。验证集 5,028 / 24,279 个对象（20.7%）落到全局中位数回退。
- 匹配关闭（`hungarian=false`）：基线预测在组内完全相同，匹配不影响数值。
- 脚本：会话 scratchpad 的 `calibrate_loss_weights.py`（见附录），运行：
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python calibrate_loss_weights.py`（约 15 s，CPU）。

## 结果

| 项 | 有效对象数 | 基线 `B_k` | `λ_k` | 贡献 `λ_k·B_k` |
|---|---:|---:|---:|---:|
| position | 23,471 | 0.2026 | 1 | 0.203 |
| size | 17,222 | 0.3222 | 0.6 | 0.193 |
| yaw_cls | 23,471 | 2.4849 (= ln 12) | 0.08 | 0.199 |
| yaw_reg | 23,471 | 0.0276 | 7 | 0.193 |

贡献最大/最小 = 1.05，满足因子 2 要求。

说明：
- size 有效数少于 position：抽样中 7,057 个对象（29%）没有尺寸标签（SpatialLM 等来源），与 C2 "按位置匹配、尺寸可缺" 一致。
- yaw_reg 基线很小，因为 81% 的 GT yaw 落在 bin 中心（轴对齐家具，|残差| < 0.01），残差 90/95/99 分位为 0.37 / 0.70 / 0.97。按规则得到 `λ_yaw_reg = 7`；tanh 残差头使单对象 Huber 上界为 0.5，加权后 ≤ 3.5。第二轮已把它封顶到 2（偏离规则，原因见文末）。
- 旧配置 SmoothL1(β = 1) 在归一化坐标上几乎全在二次区（位置误差 < 1），等价于 L2 且随房间尺度变化；本次改为 L1，默认 `position_type = size_type = "l1"`，`smooth_l1_beta` 仅在 `smooth_l1` 下生效。
- 其它配置（`structured*.json`、`qwen3_8b_pilot.json`、`qwen3_8b_bbox_*.json`、`smoke.json`）只补齐新键，权重保持 1.0，不改既有实验定义。

## 2026-10-07 第二轮

### yaw_reg 封顶 2.0（偏离规则）

`qwen3_8b_main_world7.json`、`qwen3_8b_pilot_1gpu.json` 和两份正式配置的 `loss.yaw_reg` 由 7 改为 2.0。规则给出的 7 来自 `1 / B_yaw_reg`，而 `B_yaw_reg = 0.0276` 很小只是因为 81% 的 GT yaw 恰好落在 bin 中心；它度量的是"多数标签残差为 0"，不是残差项的难度。对其余 19% 非轴对齐标签（残差 90/95/99 分位 0.37 / 0.70 / 0.97），单对象加权 Huber 可达 7 × 0.5 = 3.5，约为每项目标贡献 0.2 的 17 倍，单个斜放物体的残差梯度会压过同一窗口里位置、尺寸、yaw 分类三项。第二轮数据又把 InternScenes / InteriorGS 等改为 4 阶盒对称（K1），yaw 有效对象和候选数都变多，这一失衡只会放大。封顶 2 后单对象上界 1.0（5 倍），基线贡献 2 × 0.0276 = 0.055，低于其它项；这是有意的：残差只细化 bin 内角度，不应主导训练。

### grid_residual 位置头的两项权重

`model.position_head = "grid_residual"` 时（`qwen3_8b_main_3gpu_grid.json`），`position` 项改为
`position_cell · CE(GT 格) + position_residual · L(GT 格 XY 残差，半格单位，两轴平均) + L(z)/3`；
z 保持回归头下完全相同的 `/3` 份额，所以 `loss.position` 仍是整个位置项的权重和资格开关。两个新 `LossConfig` 字段 `position_cell` / `position_residual` 只在 grid 头下生效，回归头忽略它们。

按同一规则、同一 2,000 房间样本（RNG 先抽训练样本再抽验证样本，与附录脚本一致）计算未训练基线：均匀 logits（16 × 16 = 256 格）、零残差、z = `floor_z_m`。复算的回归 `B_position = 0.2026`（23,471 个对象）与上表一致，说明样本相同。

| 项 | 有效对象数 | 基线 `B_k` | `λ_k = round_1sig(B_position / B_k)` | 贡献 |
|---|---:|---:|---:|---:|
| position_cell | 23,471 | 5.5452 (= ln 256) | 0.04 | 0.222 |
| position_residual | 23,471 | 0.4980 | 0.4 | 0.199 |
| position_z（z 可学习的对象） | 14,951 | 0.0445 | 跟随 `position` = 1 | 0.028（按全部位置对象折算） |

与 size 0.193、yaw_cls 0.199 的最大/最小比为 1.15。两份正式配置和 main/pilot 都写入 `position_cell = 0.04`、`position_residual = 0.4`，使两份正式配置只在位置头和梯度累积上不同。脚本见附录二，在工作树根目录运行 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python <脚本>`（约 25 s，CPU，只读 `/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local/validation.jsonl`）。

窗口日志键（`term_sums` / `term_counts`，两种头都总是存在）：`position`（解码后位置的 L1/3，两种头同一定义，可直接对比；grid 头下只作诊断、不回传梯度）、`position_cell`、`position_residual`、`position_z`（回归头下和为 0、计数为 0）。

### 盒对称（K1）对已有标定的影响

`validity.size_axis_swap_allowed` 为真的对象在损失里取 (sx, sy, θ) 与 (sy, sx, θ + π/2) 四个候选的联合最小，只会降低 size / yaw 两项的基线；上表的 size 和 yaw 数值来自尚无该字段的数据，按"不放大权重"的方向保守，不重新标定。

### 正式配置与上下文长度

所有 Qwen 主干配置（`qwen3_8b_*`、`structured*`）的 `training.max_length` 统一为 8192，评估/预测从 checkpoint 读同一值。两份正式配置除下表外完全继承 `qwen3_8b_main_world7.json`；全局 batch 都是 96，124,375 个训练场景三轮 = ⌈3 × 124,375 / 96⌉ = 3,887 次更新，warmup = round(3% × 3,887) = 117，`validate_every = checkpoint_every = 500`。

| 配置 | 进程数 | batch | 累积 | 全局 batch | 位置头 |
|---|---:|---:|---:|---:|---|
| `qwen3_8b_main_4gpu_regression.json` | 4 | 1 | 24 | 96 | regression |
| `qwen3_8b_main_3gpu_grid.json` | 3 | 1 | 32 | 96 | grid_residual（16 × 16） |

进程数不在配置里（训练配置只接受六个段），启动时用 `--nproc_per_node=4` / `--nproc_per_node=3`；进程数不同会改变全局 batch，与配置不符。

## 附录：标定脚本

```python
"""Untrained-baseline value of each FastFill v2 loss term under the new L1 definitions."""
import json, math, random, statistics, sys, time
from collections import defaultdict

import torch
from shapely.geometry import Polygon

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.losses import GeometryCriterion, LossConfig

ROOT = "/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local"
SEED, N_VAL, N_TRAIN = 20261006, 2000, 20000


def sampled_rows(path, total, n, rng):
    chosen = set(rng.sample(range(total), n))
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index in chosen:
                yield json.loads(line)


def emulate_c4_yaw(row):
    source = row["provenance"]["source"]
    front_known = bool(row["provenance"].get("source_meta", {}).get("front_known", False))
    evidence = row["provenance"]["field_evidence"]
    yaw = [o.get("yaw_rad") for o in row["target"]["objects"]]
    valid = [front_known and not fe["tilted"] and isinstance(y, (int, float)) and math.isfinite(y)
             for fe, y in zip(evidence, yaw)]
    return {**row, "validity": {**row["validity"], "yaw": valid,
                                "yaw_symmetry_order": [1 if source == "MultiScan" else 2] * len(valid)}}


def main():
    rng = random.Random(SEED)
    train = list(sampled_rows(f"{ROOT}/train.jsonl", 124375, N_TRAIN, rng))
    per_category, everything = defaultdict(list), []
    for row in train:
        by_id = {o["id"]: o for o in row["condition"]["objects"]}
        for target, size_ok in zip(row["target"]["objects"], row["validity"]["size"]):
            if all(size_ok):
                per_category[by_id[target["id"]]["category"]].append(target["target_size_local_m"])
                everything.append(target["target_size_local_m"])
    median = lambda rows: [statistics.median(r[q] for r in rows) for q in range(3)]
    category_median = {c: median(v) for c, v in per_category.items()}
    global_median = median(everything)

    validation = [emulate_c4_yaw(r) for r in sampled_rows(f"{ROOT}/validation.jsonl", 8125, N_VAL, rng)]
    criterion = GeometryCriterion(LossConfig(hungarian=False))
    sums, counts = defaultdict(float), defaultdict(int)
    for row in validation:
        batch = collate_samples([row], TinyTokenizer(), max_length=10**8, max_objects=10**6)
        room = row["condition"]["room"]
        centroid = Polygon(room["floor_polygon_xy_m"]).centroid
        centre = torch.tensor([centroid.x, centroid.y, room.get("floor_z_m") or 0.])
        n = batch["slot_mask"].shape[1]
        sizes = [category_median.get(obj["category"], global_median) for obj in row["condition"]["objects"]]
        predictions = {
            "position_normalized": ((centre - batch["origin"][0]) / batch["scale"][0]).expand(1, n, 3).clone(),
            "size": torch.tensor(sizes).reshape(1, n, 3),
            "yaw_logits": torch.zeros(1, n, 12), "yaw_residuals": torch.zeros(1, n, 12),
            "slot_mask": batch["slot_mask"]}
        result = criterion(predictions, batch)
        for key in ("position", "size", "yaw_cls", "yaw_reg"):
            sums[key] += float(result["term_sums"][key])
            counts[key] += result["term_counts"][key]
    baseline = {k: sums[k] / counts[k] for k in sums}
    one_sig = lambda x: float(f"{x:.1g}")
    lambdas = {k: one_sig(baseline["position"] / baseline[k]) for k in baseline}
    contributions = {k: lambdas[k] * baseline[k] for k in baseline}
    print(json.dumps({"baseline": baseline, "counts": dict(counts), "lambda": lambdas,
                      "contribution": contributions,
                      "max_over_min_contribution": max(contributions.values()) / min(contributions.values())}, indent=2))


if __name__ == "__main__":
    main()
```

## 附录二：grid 项标定脚本

```python
"""Untrained-baseline value of the grid_residual position terms (same sample and rule as the 2026-10-06 calibration)."""
import json, math, random, sys
from collections import defaultdict

import torch
from shapely.geometry import Polygon

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.losses import GeometryCriterion, LossConfig

ROOT = "/Volumes/harddisk/FastFill_v2_multisource_20261006/eligible-local"
SEED, N_VAL, N_TRAIN, GRID = 20261006, 2000, 20000, 16


def sampled_rows(path, total, n, rng):
    chosen = set(rng.sample(range(total), n))
    with open(path) as handle:
        for index, line in enumerate(handle):
            if index in chosen:
                yield json.loads(line)


def main():
    rng = random.Random(SEED)
    rng.sample(range(124375), N_TRAIN)  # the original script drew the training-median sample first
    criterion = GeometryCriterion(LossConfig(hungarian=False))
    sums, counts = defaultdict(float), defaultdict(int)
    for row in sampled_rows(f"{ROOT}/validation.jsonl", 8125, N_VAL, rng):
        batch = collate_samples([row], TinyTokenizer(), max_length=10**8, max_objects=10**6)
        room = row["condition"]["room"]
        centroid = Polygon(room["floor_polygon_xy_m"]).centroid
        centre = torch.tensor([centroid.x, centroid.y, room.get("floor_z_m") or 0.])
        n = batch["slot_mask"].shape[1]
        position = ((centre - batch["origin"][0]) / batch["scale"][0]).expand(1, n, 3).clone()
        common = {"size": torch.ones(1, n, 3), "yaw_logits": torch.zeros(1, n, 12), "yaw_residuals": torch.zeros(1, n, 12),
                  "slot_mask": batch["slot_mask"]}
        regression = criterion({**common, "position_normalized": position}, batch)
        grid = criterion({**common, "position_normalized": position, "position_cell_logits": torch.zeros(1, n, GRID * GRID),
                          "position_cell_residuals": torch.zeros(1, n, GRID * GRID, 2)}, batch)
        sums["position"] += float(regression["term_sums"]["position"])
        counts["position"] += regression["term_counts"]["position"]
        for key in ("position_cell", "position_residual", "position_z"):
            sums[key] += float(grid["term_sums"][key])
            counts[key] += grid["term_counts"][key]
    baseline = {k: sums[k] / counts[k] for k in sums}
    one_sig = lambda x: float(f"{x:.1g}")
    lambdas = {k: one_sig(baseline["position"] / baseline[k]) for k in ("position_cell", "position_residual")}
    print(json.dumps({"baseline": baseline, "counts": dict(counts), "lambda": lambdas,
                      "contribution": {k: lambdas[k] * baseline[k] for k in lambdas}}, indent=2))


if __name__ == "__main__":
    main()
```
