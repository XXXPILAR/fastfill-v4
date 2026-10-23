# Fastfill workflow

独立的家具布局工作流原型。输入是现有 `fastfill.v2` 条件协议：房间地面多边形、层高、门窗、固定障碍物、确定的家具 ID/类别/文字描述和约束。**待生成家具的尺寸、位置和 bbox 不作为输入**。输出每件家具的 `target_size_local_m`、`bottom_center_m`、`yaw_rad`、`support_parent`。

## 已实现的链路

1. `workflow.contract`：复用兄弟目录 `fastfill-main` 的协议校验，并拒绝待生成家具尺寸输入及无法解释的门窗。
2. `workflow.planning`：**默认一次 LLM 调用**（`workflow.llm`，Responses API，带重试，记录 token 与耗时）生成支撑图、软关系、功能分组（anchor + members）、摆放优先级、尺寸建议（`width/depth/height` 命名字段）和位姿草稿。输入的 `support_parent` 与硬 `on` 约束锁定不变；LLM 每个字段都经过校验（未知 id、自支撑、成环、非数值一律丢弃并计数到 `diagnostics.llm.dropped`）。LLM 调用失败或回复无法解析时退回规则规划，`plan_source` 记为 `rules (llm failed)`。`--no-llm` 只用规则（消融）。规则按类别的中心词匹配（`bedside table` 是 table 不是 bed，`table lamp` 是 lamp）。功能分组以软 `near`（成员→anchor，1.5 m）进入求解器；优先级决定已满足支撑与硬关系依赖的家具之间的放置顺序。
3. `workflow.catalog`：仅从 `data_v4_v2_fixed/train.jsonl` 的有效尺寸标签统计类别尺寸，低频类退回通用先验。当前 `size_catalog.json` 含 2,221 个出现至少 20 次的类别。
4. `workflow.solver`：从墙边、房间网格、相关家具附近、支撑面和可选 LLM 草稿生成候选；采用多个布局模式的 beam search，搜索中拒绝出界、碰撞、门窗净空、固定物体冲突、支撑和显式硬关系不成立的候选。
5. `workflow.geometry`：完整布局按旋转矩形和高度区间复核。对软关系错误选择关联物体及其支撑子树做有限次局部重求解，只在改进且仍满足硬约束时提交。
6. `workflow.export`：可选输出 RoomGenBench 风格的房间与物体 JSON；不在本流程里生成网格或渲染。

这里的 beam search 是**有限候选的近似求解**，不是 CP-SAT 的全局最优性证明。搜索预算内未找到可行布局时返回 `layout: null`，而不是放松硬约束。

## 运行

从此目录执行，复用上层现有虚拟环境：

```bash
../.venv/bin/python -m workflow run \
  --input examples/bedroom.json \
  --catalog size_catalog.json \
  --output /tmp/bedroom_workflow.json \
  --roomgenbench-output /tmp/bedroom_roomgenbench.json
```

重新从训练集生成尺寸统计：

```bash
../.venv/bin/python -m workflow catalog \
  --train ../data_v4_v2_fixed/train.jsonl \
  --output size_catalog.json
```

语义规划默认调用一次 LLM（OpenAI 兼容的 `POST <base>/responses`；请求总带 `instructions`，避免中转站自行加长提示）：

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
export OPENAI_BASE_URL=...            # 可选，或传 --base-url；推理强度 WORKFLOW_REASONING_EFFORT，默认 medium
../.venv/bin/python -m workflow run \
  --input examples/bedroom.json --catalog size_catalog.json \
  --output /tmp/bedroom_llm.json
```

不想调用 API（规则规划消融、离线测试）时加 `--no-llm`；没有设置模型或 key 又未加 `--no-llm` 时直接报错，不会悄悄退回规则。LLM 路径的解析与校验由 `tests/test_planning.py` 用假回复测试，**尚未用真实 API 跑过**。

外部模型已经生成初稿时，可加 `--draft-layout /path/to/layout.json`。草稿必须包含同一家具清单的完整布局；流程只把其尺寸与位姿当成**可修改候选**，仍需经过同一套硬约束检查。它不改变 Harness 的无家具 bbox 输入定义，也不应传入 GT 布局做正式评测。

批量生成和现有评测器对接：

```bash
../.venv/bin/python -m workflow batch \
  --data ../data_v4_v2_fixed/validation.jsonl \
  --catalog size_catalog.json --limit 20 \
  --output /tmp/workflow_predictions20.jsonl

PYTHONPATH=../fastfill-main ../.venv/bin/python -m fastfill.v2.evaluate \
  --data ../data_v4_v2_fixed/validation.jsonl \
  --predictions /tmp/workflow_predictions20.jsonl \
  --max-samples 20 --output /tmp/workflow_eval20
```

`batch` 输出与数据行一一对应；失败行写 `layout: null`，不删掉失败样本。输入也可使用单行 `{ "condition": ... }`；CLI 只把其中的 `condition` 传进流程，绝不读取同行 GT 作为推理输入。

测试：

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

## 坐标与边界

- 本流程输出沿用 FastFill 本地 `+X` 为正面、局部 `[x, y, z]` 尺寸、世界 `z` 向上，位置是底面中心。
- 导出到 SAGE/RoomGenBench 时交换局部 `x/y` 尺寸，并把 yaw 减 90 度，以适配其 `+Y` 正面约定；相同世界 OBB 应保持不变。**资产本身的正面定义仍需下游核验**。
- 门内侧保留 0.8 m 净空；窗内侧保留 0.12 m 的窄条，只有与窗高范围相交的物体才被阻止。这是保守 bbox 代理，不是采光或通行性证明。
- 默认采用与现有 V2 evaluator 一致的严格 bbox 不重叠；如做视觉摆放探索，可显式加 `--allow-thin-overlap` 允许高度不超过 0.05 m 的地毯/地垫在家具下。**两种设置不能混在同一指标表里**。
- 对没有已验证 `support_surfaces` 的家具，支撑检查使用父物体 bbox 顶面代理。沙发靠背、桌腿等真实网格拓扑无法在这里证明；输出中的 `bbox_level_only` 标明这一点。
- 未实现整屋连通路径、真实 mesh 碰撞、物理稳定性、视觉 critic、资产生成或 CP-SAT 证明。这些需要独立验证，不能因 bbox 检查通过就宣称已解决。

## 当前验证

- 18 项单元测试通过（`tests/test_planning.py` 9 项覆盖规划：中心词匹配、`on` 与 `support_parent` 冲突、LLM 字段校验与映射、成环、失败回退、优先级生效）；5 件家具样例成功生成且全部显式硬约束满足。
- 从 `validation.jsonl` 选出的前 20 个 4-8 物体房间，严格 bbox 模式、beam 16 / 候选 36 下，18/20 完整输出，2 个明确失败（规则规划，`--no-llm`；2026-10-22 修正规划后重跑仍为 18/20，失败的是同两个房间）。现有 V2 evaluator 对 18 个输出的 `clean_rooms` 为 18/18、碰撞对 0；`target_geometry_valid` 仍为 0/20，主要受 evaluator 的 `support_unknown` / `openings_unchecked` 严格口径影响。位置 GT 误差均值约 1.72 m；这也是**烟测，不是论文性能结论**。
- [experiments/smoke20](experiments/smoke20/README.md) 保留该批的输入行、预测、原始 evaluator 报告和复现命令。剩余失败包括一个高密度多地毯房间，以及一个类别 `board` 对应的真实物体尺度与类别中位数相差巨大的 L 形房间。

相关源码的来源、版本和哪些部分适合借鉴见 [references/README.md](references/README.md)。这些仓库没有被作为此工作流的运行时依赖，也没有把其源码复制进 `workflow/`。

## Draft workflow on the server (official OpenAI API)

`run_draft_workflow` (`workflow/pipeline.py`): ① GPT semantic plan → ② FastFill size, one per object
(`workflow/fastfill_sizes.py`, the checkpoint runs in-process) → ③ GPT draft at those sizes → hard checks → ④ local
repair of the objects with issues (`workflow/local_repair.py`). LLM calls use the official OpenAI API only: the key is
in `.env` in this folder (`OPENAI_API_KEY=...`, git-ignored; create it by hand on each machine), any other host is
refused, and a rejected key or exhausted quota stops the run.

In the v4 checkout this folder is `fastfill_workflow/`. From the checkout root:

```bash
sbatch fastfill_workflow/experiments/isambard_draft.sbatch                       # test300, budget $55
BUDGET_USD=8 EXTRA="--per-bin 6" sbatch fastfill_workflow/experiments/isambard_draft.sbatch   # 30-room sample
WORKERS=16 sbatch fastfill_workflow/experiments/isambard_draft.sbatch     # rooms in parallel (default 8)
```

Outputs in `runs/draft-workflow/test300/`: `room_XXX.json` (layout, diagnostics, timing, tokens, cost),
`ledger.json` (spend), `summary.json`, `rows.jsonl` + `predictions.jsonl` and `eval/` (`fastfill.v2.evaluate`).
The run is resumable and stops before passing the budget. If compute nodes cannot reach api.openai.com the
preflight says so in seconds; then make the sizes on a GPU node and run `experiments/run_draft.py --sizes-file ...`
on a login node (see the sbatch header).
