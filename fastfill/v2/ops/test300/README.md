# test300：FastFill v4 与 LLM 对比用的固定测试子集

从 `data_v4_v2_fixed/test.jsonl`（8,700 个房间）中抽出的 300 个房间，用于在**同一批房间、同一套评测代码**上
对比 FastFill 各组模型与 GPT-5.6（以及后续的"FastFill 草稿 + GPT 纠正"等方法）。

## 抽样方式

- 生成脚本：`fastfill/v2/ops/make_test300.py --test test.jsonl --output test300 --rows 300 --seed 0`，可完全复现（`rows_sha256` 一致）。
- 先排除家具超过 256 件的房间（模型容量上限，共 3 个），保证每种方法都能回答每个房间。
- 按数据来源分层，各来源的份额与测试集中的比例一致（最大余数法）；MultiScan（6 个）和 Scan2CAD（1 个）占比太小，分不到名额。
- 每个来源内部用同一个随机数生成器（seed 0）不放回抽取；`rows.jsonl` 按原测试集顺序排列。
- `MANIFEST.json` 记录了种子、各来源数量、原测试集 sha256、每个房间在 test.jsonl 中的行号和 scene_id。

## 文件

| 文件 | 内容 |
|---|---|
| `rows.jsonl` | 300 个房间，格式与 test.jsonl 相同（condition / target / validity / provenance） |
| `MANIFEST.json` | 抽样记录 |
| `gpt56/predictions.jsonl` | 纯 GPT-5.6（gpt-5.6-sol）的布局，与 `rows.jsonl` 逐行对应，可直接用于 `evaluate --predictions` |
| `gpt56/room_*.json` | 每个房间的原始请求、原始回答、耗时和 token 用量 |
| `gpt56_eval/report.json` | 上面这份答案的评测报告（`fastfill.v2.evaluate`） |

GPT-5.6 的协议：完整条件（房间多边形、门窗、固定障碍物、家具清单、约束），每个房间调用一次，不做任何修补
（`fastfill/v2/ops/llm_probe.py`，提示词来自 `llm_workflow.instructions` / `request_text`）。

## 评测

```bash
python -m fastfill.v2.evaluate --data test300/rows.jsonl --predictions test300/gpt56/predictions.jsonl --output <目录>
python -m fastfill.v2.ops.remask_ablation --checkpoint <run>/model-step-N --data test300/rows.jsonl --output <目录> \
    --modes argmax spread remask4L remask4LX --device cuda
```

## GPT-5.6 运行记录（2026-10-17）

- 模型 `gpt-5.6-sol`，8 个并行请求（重试时 3 个），限速 30 次/分钟；共约 116 万 token。
- 300 个房间中 284 个得到布局；**16 个失败**，都是 67–189 件家具的大房间（SAGE-10k 12 个、MansionWorld 2 个、InteriorGS 1 个、SceneSmith 1 个），
  原因是 API 网关超时（HTTP 524）、连接中断或返回内容缺少答案。失败房间重试过一次，仍失败的记为 GPT-5.6 失败，
  `predictions.jsonl` 里 `layout` 为 null，评测时计入失败，不从样本中去掉。
- 成功房间的耗时中位数 44 秒，平均 63 秒。
