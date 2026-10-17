给定房间、指定对象和可获得的空间条件，一次前向预测各请求对象的目标局部尺寸、底面中心位置与朝向，并把结果导出为带 bbox 的交接包，由下游 RoomGen/RoomGenBench 生成或拟合 mesh、装配场景。
主方案采用 Qwen 条件编码、请求绑定的双向对象解码器和网格/连续几何监督；当前与主方案并列比较的是输入字段相同的外部 LLM 智能体，文本 SFT 仍是未完成的独立对照。
修订日期：2026-10-09。本文保留 2026-10-05 定稿中仍然成立的部分，改正已过时或错误的部分，补上定稿没有写到、但系统已有的部分（三字段投影、spread 解码、K1 盒等价规则、RoomGenBench 交接、LLM 智能体对照），并写入已商定的下一步。全文不把任何 [计划] 项写成已实现或已验证；布局质量、下游成功率和算法收益以注明来源的实测为准。状态标记、引用约定与实测来源代号见第 1.4 节末。

1. 任务与系统职责

1.1 任务定义

令 $$R$$ 表示房间语义及几何，$$O_{\mathrm{req}}$$ 表示上游指定的对象实例，$$C$$ 表示用户要求和空间约束。FastFill v2 先把条件投影成模型实际读取的形式，再一次预测：

$$\boxed{
F_\theta\bigl(\Pi(R,O_{\mathrm{req}},C)\bigr)
\longrightarrow
\widehat L=
\{(id_i,\hat s_i,\hat p_i,\hat\theta_i)\}_{i=1}^{N},
\qquad
\Pi\in\{\Pi_{\mathrm{full}},\Pi_3\}
}$$

其中 $$\Pi_{\mathrm{full}}$$ 是完整条件（恒等投影），$$\Pi_3$$ 是三字段投影（第 2.6 节）。

| 变量 | 定义 |
|---|---|
| $$id_i$$ | 请求对象的稳定标识，输出必须与请求一一对应 |
| $$\hat s_i=(\hat w_i,\hat d_i,\hat h_i)$$ | 对象局部坐标中的目标全尺寸，单位米 |
| $$\hat p_i=(\hat x_i,\hat y_i,\hat z_i)$$ | 目标包络底面中心在房间坐标中的位置 |
| $$\hat\theta_i$$ | 绕房间 $$+Z$$ 轴的 yaw，单位弧度，取值 $$[-\pi,\pi)$$ |

模型同时回答“这个对象需要多大”和“摆在哪里、朝向哪里”。[现状] 输出接口只有 `id`、`target_size_local_m`、`bottom_center_m`、`yaw_rad` 四个字段；`validate_layout` 拒绝未知字段，要求尺寸为正有限数、yaw 在 $$[-\pi,\pi)$$、ID 不重复，并要求输出 ID 集合等于请求 ID 集合（`schema.py:295-313`）。对象类别、数量和身份由请求决定，模型没有删除对象或替换类别的输出通道；默认的 spread 解码只改 XY、yaw 和 z，不改尺寸（`evaluate.py:121-124`）。

[现状] 完整条件下，$$R$$ 包含房型、地板多边形、地面高度、房高，以及 `room.fixed_objects` 中必须保留的已有物体；门窗也以固定盒子的形式写在这里（第 2.5 节）。`room.openings` 字段被 schema 接受（`schema.py:53-54, 285-288`），但 train、validation、test 中都没有出现。已有物体只作条件，不占输出 slot（`legacy_bridge.py:198-217`）。边界、地面和房高未提供时带未知标记；门窗没有已知/未知标记（第 2.4、2.5 节）。

[现状] 当前数据中，$$O_{\mathrm{req}}$$ 的每个对象只有 `id`、`category`、`description` 和可选的 `support_parent`。schema 和模型还支持 `fixed_size_local_m`、`size_bounds_local_m`、能力要求等字段（`schema.py:56-59, 158-182`；`model.py:217-220`），但数据中没有任何对象带这些字段。$$C$$ 是从参考布局中抽取的稀疏关系约束（每房 1–4 条，`fastfill/build.py:347`），类型为 `against_wall`、`between`、`near`、`on`、`faces`。它们只出现在 train（共 63,665 条，其中 `faces` 12 条），validation 为 0 条，test 为 3 条（manifest `constraint_counts`）。所以验证集上没有约束，测试集只有 3 条（autorun F 阶段的完整条件测试会对它们运行 `validate_scene` 的约束检查，`autorun.py:558-559`），约束遵守没有可用的评测。

[现状] 当前有三种入口：
- `predict --condition` 读取完整条件；
- 训练中以 0.5 概率使用三字段投影，选模、主测试表和 LLM 对照全部使用三字段投影（第 2.6 节）；
- `predict --request` 只接受房型、房间尺寸和物品清单（可带 `support_parent`）。它先渲染三字段投影，再把条目声明的 `support_parent` 加回对象（`direct_layout.py:97-109`）：条目都不带 `support_parent`、且用默认的 rectangular 语义时，条件文本与三字段投影逐字节相同；声明了 `support_parent` 时对象上多出该字段（声明 floor 的对象 z 由条件固定，`batch.py:251-261`），这种形式在三字段训练样本中没有，只在完整条件训练样本中出现；reference_extent 语义另把 `boundary_known` 写成 false。RoomGenBench 基准请求走这个入口，不含门窗；它为每个对象声明 `support_parent`，所以文本是三字段投影加 `support_parent`（`roomgenbench.py:253-265`）。

[计划] Stage 0 起，选模、主表和 LLM 对照改用含门窗的完整条件；三字段投影降为附加结果，报告时必须同时写出被跳过的非矩形房间数。

1.2 target geometry envelope 的语义

目标尺寸描述供下游生成或检索资产时使用的期望几何规格。例如，桌子的目标尺寸为 $$(1.4,0.7,0.75)$$ 米，表示希望得到几何尺度接近该规格、且满足功能要求的桌子。

它不是尚未选择的 mesh 的唯一真实尺寸，也不自动表示严格的最大容纳盒。接口分别表达：

| 信息 | 含义 | 状态 |
|---|---|---|
| target size | 模型提出的期望尺寸 | [现状] 唯一的尺寸输出 |
| size bounds / fixed size | 用户或任务明确要求的尺寸范围、固定尺寸 | [现状] schema 与模型支持，固定维度直接替换预测（`model.py:217-220`）；当前数据中 0 个对象 |
| retrieval tolerance | 资产解析时允许的尺寸偏差 | [现状] 只有离线参考 runtime 使用（`runtime.py:109-113`）；RoomGenBench 不设容差，直接把 mesh 拟合到 target box |
| actual size | 选定资产经标准化变换后的实际尺寸 | [现状] RoomGenBench 装配回执记录 `native_size_*` 与 `fitted_size_sage_local_m`，拟合结果与 target 不一致时标 `fit_mismatch`（`roomgenbench.py:89-103`）；离线 runtime 记录 `actual_size_local_m`（`runtime.py:157`） |

[现状] 在 RoomGenBench 路径上，下游 `fit_asset` 对 mesh 做非等比缩放、yaw snap 和必要时的 tip，把它拟合到 target box（`docs/fastfill-v2-roomgenbench-interface-20261006.md`）。所以这条路径上的 actual 几何按构造接近 target：mesh 拟合成功不等于原始资产满足尺寸、房间可行性或物理稳定。装配回执中的 `original_asset_geometry_acceptance`、`support_verification` 写为 `not_checked`，`validator`、`physics`、`host_commit` 写为 `not_attempted`（`roomgenbench.py:239-242`）。“保留 target 与 actual 两份记录、按实际几何验收”的要求只在离线参考 runtime 中实现，从未在真实资产库上运行（第 1.3 节）。

训练场景中的实测对象尺寸是目标包络的一份监督样本：它说明该条件下曾存在一个可用选择，不代表这个条件只有一个正确尺寸或布局。[现状] 尺寸标签只对有完整局部尺寸证据的对象有效：全部 split 中有效尺寸标签 1,533,512 个（manifest `validity_counts.size`），train 中完整尺寸标签对象 1,351,470/1,998,842（67.6%，[D] §6.6-b）。MansionWorld 与 OptiScene_holodeck 的尺寸是标注足迹或加垫的代理框，整源屏蔽（`legacy_bridge.py:111-112`）。

[现状] 尺寸是 FastFill 目前明确强于 LLM 智能体的一项：300 条三字段验证行上 log-size 误差 FastFill 0.237，LLM 智能体 0.388–0.412，类别中位尺寸查表 0.385（[M2]；全部结果见第 9.10 节）。

1.3 适用范围与职责分工

主协议覆盖直立、仅绕竖直轴旋转的家具及物体。[现状] 当前没有显式分流：倾斜的源对象仍作为请求 slot 保留，位置、尺寸、朝向监督全部屏蔽（`legacy_bridge.py:97-113`），推理时照常输出直立框。train 中有 84,377 个倾斜对象（占 1,998,842 个请求对象的 4.2%），validation 5,105 个，test 6,154 个（`provenance.field_evidence[].tilted`；validation、test 于 2026-10-09 重新计数）。倾斜的固定几何会让整行被拒收（`legacy_bridge.py:205-206`）。RoomGenBench 真值中绕 X/Y 的倾斜在 `ground_truth_layout` 中被丢弃，并计入 `ground_truth_tilted_objects`（`roomgenbench.py:297-320, 394-395`）。非刚性或装配状态没有任何字段。显式分流没有列入 Stage 0/1，这里只记为缺口。

| 模块 | 负责 | 状态 |
|---|---|---|
| 上游请求方 | 房间条件（完整条件，或房型 + 尺寸）、所需对象、实例身份、可选支撑声明 | [现状] `predict --condition` / `--request`（`predict.py:29-31, 60-66`）；RoomGenBench 场景由 `roomgenbench --requests-from` 转成请求（`roomgenbench.py:253-294`）；WorldEdge Harness 未接入 |
| FastFill v2 | 目标尺寸、底面中心、朝向的联合预测；spread 解码；schema 校验；导出交接包（`place` / `support_status`、房间 shell、门窗开口、bbox 诊断） | [现状] `predict.py`；`evaluate.py:340-381`；`direct_layout.py:235-373` |
| RoomGen / RoomGenBench（下游） | 按 `asset_key` 生成或检索 mesh、拟合到 target box；渲染与评分只针对基准自身的 GT 布局 | [现状] 下游仓库独立。FastFill 侧提供 `roomgenbench.assemble_handoff`（调用下游参考 assembler；`layout_boxes` 方法用 target 盒代替 mesh，`roomgenbench.py:206-250`）、基准请求构造（`--requests-from`，`roomgenbench.py:253-294`）与参考比对（`--reference-check`，`roomgenbench.py:297-399`）。生成方法不在 FastFill 内运行；下游评分不读 FastFill 的交接或装配输出（第 8.1 节） |
| 离线参考 runtime | Resolver 契约、actual 几何对齐、bbox 级 Validator、有限次资产重选与可选平移修复、内存原子提交 | [现状] `runtime.py:338-448`。仓库中没有资产库，autorun 与 ops 从不传 `--catalog`，从未在真实资产上运行 |
| Host / Solver | 持久世界的原子提交；提交后的任务执行 | [现状] 未实现；只有内存 `AtomicMemoryHost`，`solver_success` 恒为 None（`runtime.py:333`） |

[现状] 交接边界如下，完整字段见第 8 节。
- FastFill 保证：输出 ID 与请求一一对应、尺寸为正、yaw 已 wrap。导出时按 `width` $$=s_y$$、`length` $$=s_x$$、`height` $$=s_z$$ 与 `rotation.z` $$=\deg(\operatorname{wrap}(\theta-\pi/2))$$ 转到 SAGE 的局部 $$+Y$$ 前向约定，底面中心原样作为 `position`（`direct_layout.py:272-279`）。`place_id` 取请求声明的 `support_parent` 或 hard `on` 约束（`support_status = declared`）；否则由预测框推断 floor / on_object / wall 候选（`inferred`）；都不成立时为 `unknown`（`direct_layout.py:192-232`）。墙由地板多边形各边生成，门窗由 `fixed_objects` 中类别含 door/window 的盒子挂到最近的墙上（`direct_layout.py:140-174`）。
- FastFill 不提供：资产 ID（`asset_key` 只是下游生成用的键，标为 `downstream_generation_key_only`）、经过验证的支撑面、语义前向、mesh 或物理检查（`direct_layout.py:1-6, 236-250`）。
- 下游负责：资产生成或检索、mesh 拟合（下游 `fit_asset`）。FastFill 侧适配器 `python -m fastfill.v2.roomgenbench --require-placement` 代下游执行“每个对象都要有 `place_id`”的策略（`roomgenbench.py:52-53`）；autorun 流程中 RoomGenBench 的 validator 与 physics 不运行（`autorun.py:450-451`）。下游的渲染与评分目前只评基准自身的 GT 布局（第 8.1 节）。

FastFill 不承担开放式任务规划，也不重新决定家具数量。[现状] 请求对象数受 `max_objects` 限制，超出时整条请求拒收而不截断（`batch.py:443-444`；`direct_layout.py:53-54`）。需要由 Solver 完成的操作，不能通过预先生成操作完成后的世界状态来规避；当前没有 Solver。

1.4 文献定位与证据口径

本文以 [原文]、[代码] 和 [设计] 区分文献来源，以 [现状] 和 [计划] 区分系统状态。未带文献来源标签的实施规范均属于 FastFill 设计，不能写成论文已经验证的结论。

| 来源 | 支持的判断 | 在本方案中的用途 |
|---|---|---|
| OptiScene | 语言条件布局生成、合成数据及多阶段偏好优化可构成室内布局训练路线 | 条件语言建模与数据组织参考；之后的偏好优化参考 |
| ATISS / DiffuScene | 对象尺寸、位置、朝向可以共同成为场景生成变量 | 联合几何输出与对象集合表示的依据 |
| DETR / 3DETR | 用离散二分匹配确定预测与监督的对应，再计算可微损失 | 集合监督机制参考 |
| V-DETR | 3D box 的位置、尺度、角度监督及检测集合匹配 | 几何参数化与训练实现参考 |
| CasLayout | 家具属性、关系、OBB 可分阶段生成 | 相关工作，不决定本方案的网络结构 |

这些工作不证明 FastFill 的具体架构一定有效，也不能据此宣称“联合生成尺寸与位姿”本身是新的贡献。

[现状] 已有实测的要点（完整数字与区间见第 9.10 节）：
- 关系：用从训练集 GT 挖掘的 32 种成对关系模板（如床头柜在床侧、椅子在桌旁、沙发与电视柜相对）计算关系成立率，GT 0.71，LLM harness 0.76，LLM prompt 0.73，FastFill spread 0.43，FastFill argmax 0.25；GT 位置在房间内打乱后为 0.23。spread 减 harness 为 −0.33 [−0.41, −0.25]（[M1]，293 间三字段验证房间，不含门窗）。
- 强项：尺寸（log-size 误差 0.237，LLM 智能体 0.388–0.412）和延迟（单房间中位 56 ms，LLM harness 16.8 s）（[M2]）。
- 单参考位置误差区分不了方法：四种方法为 2.62–2.75 m，都比把所有对象放在房间中心（2.35 m）差（`outputs/fastfill_v2/comparison-20261007b/autorun/SUMMARY.md`）。

研究问题因此改为：[计划] 在相同输入字段（含门窗）下，单次前向的结构化预测能否在布局关系上不输于 LLM 智能体，同时保持尺寸精度与毫秒级延迟。判定标准见第 7.8 节，冻结评测集见第 9.7 节，评审方式见第 9.5 节；这些实验完成之前，不声称 FastFill 的布局质量达到或超过 LLM 智能体。文献细节见第 11 节。

本文的状态标记、引用约定、实测来源代号与统一用语如下，全文通用。

修订依据：2026-10-05 定稿的 FastFill v2 设计规范；逐节对照文档 `docs/fastfill-v2-spec-vs-implementation-20261009.md`（63 条，判定与证据以它为准）；仓库 `main` @ `2e86188`（代码 `fastfill/v2/`，主数据 `outputs/fastfill_v2/rebuild-main-20261008a/main/`，运行配置 `fastfill/v2/configs/main7-cell05-main-20261007b-e5.json`；Isambard 运行另加 `--set model.max_objects=256`，两组 `loss.yaw_cls` 分别为 0.5 与 0.08，见 `ops/isambard_autorun.sbatch:17`、`ops/isambard_submit.sh:8-9`）。

状态标记：

| 标记 | 含义 |
|---|---|
| [现状] | main 2e86188 上已实现的行为或已有数据的统计，附 `文件:行` 或数据文件证据；“没有实现”的缺口也作为现状记录 |
| [计划] | 已商定、尚未实现也未验证的下一步。Stage 0（不重训）：含门窗的评测集；含门前净空的布局指标；带对照样本校验的 Qwen3-VL 成对评审；输入含门窗的 LLM 智能体对照；用训练集挖掘的关系先验做关系解码。Stage 1：条件（masked）布局训练；loss 中的整房对称；锚点优先的迭代解码；以含门窗的完整条件为主训练。之后：在来源能证实真实前向的对象上学习整圈 yaw；偏好优化 |
| [计划]（本修订提出，待确认） | 本修订为落实已商定计划而补出的细则（如选模规则、聚类 bootstrap、第二种子），尚未商定 |
| [原文] / [代码] / [设计] | 沿用定稿的文献来源标签：论文原文、论文官方代码、本方案自己的设计选择 |

引用约定（全文通用，各节不再重复）：
- 不带目录的 `xxx.py:N` 指 `fastfill/v2/xxx.py` 第 N 行（main 2e86188）；`ops/…`、`tests/…`、`configs/…`、`README.md` 都在 `fastfill/v2/` 下；`fastfill/build.py`、`fastfill/split.py` 是 `fastfill/` 下父数据的构建与划分代码；`docs/…`、`outputs/…` 相对仓库根目录。
- `config:N` 指上述运行配置文件第 N 行。
- [D] §x 指对照文档 `docs/fastfill-v2-spec-vs-implementation-20261009.md` 的条目。
- train、validation、test 指主数据目录中的 jsonl，manifest 指同目录的 `manifest.json`。
- `RoomGenBench/…` 指仓库内的下游检出（HEAD `30f2e05`，FastFill 只读、不改）；`V-DETR/…` 指本地 V-DETR 检出（HEAD `9062d75`）。
- `scratchpad/…`、`tasks/…` 指 2026-10-09 分析会话目录 `/private/tmp/claude-501/-Users-slian-Desktop-3D-Worldedge-OptiScene/6ff4e6a3-42ea-486c-8da5-935ed540b0fe/` 下的文件。它们是临时文件，本文定稿发布前应复制进 `docs/` 或 `outputs/`。

实测来源代号（日期均为 2026-10-09；实测数字一律注明来源）：

| 代号 | 文件 | 内容 |
|---|---|---|
| [M1] | `scratchpad/arrange/report.md`、`summary.json`（及同目录 `relations.json`、`relation_probe.out`） | 布局关系指标。模型 main7-cell05-main-20261007b-e5 step 6357；300 条三字段验证行（不含门窗），取所有方法都有输出的 293 间；FastFill 输出由 commit a21a67fdb 解码；95% CI 为按房间 bootstrap |
| [M2] | `tasks/wfk1supiw.output` 的 `result.head` / `result.metrics` / `result.plan`，及其引用的 `scratchpad/arrange/analysis.json`、`extra.json`、`prior_ce.json` | 根因诊断与分阶段计划；尺寸误差与延迟数字取自 `outputs/fastfill_v2/comparison-20261007b/` 下的 `autorun/SUMMARY.md`、`llmrows-main7-cell05-main-20261007b-e5/outcomes.jsonl`、`llm-harness-300/predictions.jsonl` |
| [M3] | `scratchpad/stage0q/quick.py`，读取 `eval-yawcls05__outcomes{,-full}.jsonl`、`eval-yawcls008__outcomes{,-full}.jsonl` | Isambard 两臂（yaw_cls 0.5 / 0.08）在 299 间上的三字段与含门窗对比；本次合并时重新运行，数字一致 |
| [D] | `docs/fastfill-v2-spec-vs-implementation-20261009.md` | 2026-10-05 版逐条对照；数据计数基于 main-20261008a |

统一用语：“关系成立率”指 [M1] 的挖掘关系指标（第 9.4 节）；“含门窗”指完整条件（第 2.5 节）；“Isambard 两臂”指 `runs/autorun-yawcls05` 与 `runs/autorun-yawcls008`；“背墙翻转”指 spread 的近墙 π 翻转（第 4.5 节）；“整圈 yaw”指学习 $$[-\pi,\pi)$$ 上的语义朝向。“spread”指默认的贪心避碰解码（第 4.5 节）；“argmax”指每个对象各取自己最可能的格，即模型原始输出；“LLM prompt”指外部 LLM 问一次；“LLM harness”指带出界、重叠、悬空检查、最多修 2 轮的 LLM 智能体（第 9.6 节）。K1–K10 是 `README.md:83-92` 中数据与交接契约条目的编号，本文用到 K1 盒等价、K2 房高冲突只标记、K3 地面吸附、K5 三字段投影、K6 交接放置推断。“锚点对象”指关系解码中先放置的床、沙发、书桌、餐桌、电视柜（第 4.8、4.9 节）；VLM 评审中已知好坏的对照（shuffled、mirrored 等，`ops/vlm_judge.py` 中称 anchors）称“对照样本”；第 3.2 节的地面、墙面锚点是来源数据的支撑标注。

2. 坐标、尺寸与数据协议

2.1 统一几何约定

所有数据源进入训练前转换到同一个协议：

| 项目 | 约定 | 状态与证据 |
|---|---|---|
| 房间坐标 | 右手系，$$+Z$$ 向上；预处理把地板多边形外包框的最小角放在 $$(0,0)$$ | [现状] `schema.py:236-237` 只接受 `right_handed_z_up`；test 8,620/8,620 行锚定在原点（[D] §3.4-b） |
| 对象局部坐标 | $$+X$$ 为协议前向轴，$$+Y$$ 为横向，$$+Z$$ 向上 | [现状] $$+X$$ 是轴约定；对绝大多数对象，$$\pm X$$ 哪一端是语义前向没有真值（见下文） |
| 尺寸 | $$s=(w,d,h)$$ 分别为局部 X、Y、Z 方向的全长度；均为正数、单位米 | [现状] `schema.py:304`；`geometry.py:1`；K1 对象例外见下文 |
| 位置 | $$p$$ 为局部 bbox 底面中心映射到房间坐标后的点 | [现状] 第 2.2 节 |
| 旋转 | yaw 绕 $$+Z$$ 按右手规则，弧度；统一到 $$[-\pi,\pi)$$ | [现状] `geometry.py:9-10`；`schema.py:306-308` |

这里的 $$w,d$$ 是协议字段名；不能直接假定某个资产库的 width/depth 与本协议轴序相同。资产适配器必须显式记录轴置换、单位、缩放、pivot 与规范前向。[现状] 目前唯一实现的适配器是 RoomGenBench/SAGE：局部前向为 $$+Y$$，`width` $$=s_y$$、`length` $$=s_x$$、`height` $$=s_z$$，`rotation.z` $$=\deg(\operatorname{wrap}(\theta-\pi/2))$$，世界角点不变（`direct_layout.py:235-279`）。固定物体的 bbox 代理用同一换算（`roomgenbench.py:188-192`），`ground_truth_layout` 是它的精确逆变换（`roomgenbench.py:297-320`）。

局部尺寸不随 yaw 变化。旋转后的世界轴对齐包围盒会随 yaw 变化，不能将它当作局部尺寸标签。几何长轴也不必等于语义前向：椅背、显示屏朝向等需要单独核验。

[现状] 朝向的监督周期由来源策略决定，不由真实的旋转对称决定（manifest `parent_front_policy = axis`；`legacy_bridge.py:14-21, 97-113`）。train 的 `yaw_symmetry_order` 分布（[D] §2.1）：
- 1,872,154 个对象（93.7%）为 2，只学 yaw mod $$\pi$$；
- 126,178 个（6.3%）为 4，几乎全部来自 K1 盒等价规则，只有 Scan2CAD 少量 `__SYM_ROTATE_UP_4` 是真实的对称标注；
- 只有 MultiScan 的 510 个对象（0.03%）为 1，保留语义前向。

数据中没有 `faces_direction` 约束，train 只有 12 条 `faces`。因此“背墙还是朝墙”不是模型学到的能力。[现状] 实测（[M1]，293 间三字段房间）：argmax 解码下，有前向的家具 49% 正对墙；默认 spread 解码的近墙朝向规则使正对墙降到 2.3%。[M1] 用的是 commit a21a67fdb 的旧版规则：对离墙 0.3 m 以内的地面对象，在背离最近墙 45° 内取概率最大的 yaw bin；当前 main 只做 π 翻转，即翻转后朝向落在“背离最近墙” ±45° 内时把 yaw 翻转 $$\pi$$（`evaluate.py:289-293`），尚未在同一批房间上测量（头输出没有保存，无法重新解码）。同一类规则也会把书桌前的椅子转成背对书桌：spread 下书桌椅朝向书桌的比例为 0.10（GT 0.66，LLM harness 0.85），46% 恰好背对（同为旧规则下的 [M1] 测量）。

[现状] K1 盒等价规则（`legacy_bridge.py:22-35`；`losses.py:10-19, 171-206`）。`validity.size_axis_swap_allowed` 为真的对象，其标注框可以等价写成 $$(s_y,s_x,s_z,\theta+\pi/2)$$。适用范围：InternScenes arkit/3rscan/mp3d/scannet 与 InteriorGS 的全部对象、HSSD200 中类别含 chair 或等于 seat 的对象、MultiScan 的床；全部 split 共 147,673 个（manifest `source_size_axis_swap_allowed_objects`），train 中 126,173 个（6.3%，[D] §6.4-b；2026-10-09 逐行复核 `validity.size_axis_swap_allowed`）。对这些对象，“尺寸轴固定”不成立：
- loss 在 $$k\in\{0,1,2,3\}$$ 四个候选 $$\theta+k\pi/2$$（奇数 $$k$$ 交换 $$s_x,s_y$$）上取 detached 联合最小（公式见第 6.4 节）；
- 匹配取两种轴序中较小的尺寸代价（第 5.2 节）；
- 评测同时报告盒等价误差和 `*_plain_convention` 误差（第 9.3 节）。

只有请求固定了 $$s_x$$ 或 $$s_y$$ 时才锁定轴序，而当前数据没有固定尺寸。这条规则取代定稿 §6.4 的“不能交换 $$w,d$$”。

[计划]（之后）对来源 GT yaw 能证实一致前向的对象学习整圈 yaw，其余对象保持 mod $$\pi$$。学习之前先审计：按来源统计靠墙类家具的背墙比例。在此之前，语义朝向只能由解码规则或约束给出，不能写成模型能力。

2.2 bottom-center 与 geometric center

FastFill 的 position head、matching cost 和 position loss 均使用底面中心 $$p$$（网格位置头下的代价形式见第 5 节）。若框算子要求几何中心，则分别构造：

$$\hat c=\hat p+(0,0,\hat h/2),
\qquad
c^*=p^*+(0,0,h^*/2).$$

预测框使用预测高度，监督框使用监督高度。不能两边共用一个高度。纯位置监督和 BEV 中心坐标不需要此转换。

用以下定义构造直立 OBB：

$$B(p,s,\theta)=
\left\{
p+R_z(\theta)q:
q_x\in[-w/2,w/2],\
q_y\in[-d/2,d/2],\
q_z\in[0,h]
\right\}.$$

该定义同时固定了尺寸、底面中心、角度与角点的关系，是训练算子、校验器和导出的共同几何契约。[现状] `geometry.py:61-62` 的 `bottom_to_center` 加 $$h/2$$；`geometry.py:74-78` 的直立框从底面 z 向上延伸 $$h$$；导出的 `scene.json` 同时给出 `center_m` 与 8 个角点（`direct_layout.py:130-132`）；RoomGenBench 的 `position` 是底面中心（`direct_layout.py:275`）。

2.3 输入与输出示例

[现状] 以下示例的字段与主数据一致：对象 ID 是样本内的 `obj_%04d`，固定物体是 `fixed_%04d`；门窗是 `fixed_objects` 里的盒子；`description` 沿用来源文本，没有时用类别名；`constraints` 在 validation 和 test 中几乎总是空表。完整条件：

```json
{
  "schema_version": "fastfill.v2",
  "room": {
    "frame": "right_handed_z_up",
    "room_type": "bedroom",
    "floor_polygon_xy_m": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.5], [0.0, 3.5]],
    "floor_z_m": 0.0,
    "floor_known": true,
    "boundary_known": true,
    "height_m": 2.7,
    "fixed_objects": [
      {"id": "fixed_0000", "category": "door", "size_local_m": [0.9, 0.1, 2.05],
       "bottom_center_m": [3.3, 0.0, 0.0], "yaw_rad": 0.0},
      {"id": "fixed_0001", "category": "window", "size_local_m": [1.2, 0.1, 1.2],
       "bottom_center_m": [0.0, 1.5, 0.9], "yaw_rad": 1.5707963}
    ]
  },
  "objects": [
    {"id": "obj_0000", "category": "bed", "description": "double bed with upholstered headboard",
     "support_parent": "floor"},
    {"id": "obj_0001", "category": "nightstand", "description": "wooden nightstand", "support_parent": "floor"},
    {"id": "obj_0002", "category": "nightstand", "description": "wooden nightstand", "support_parent": "floor"}
  ],
  "constraints": []
}
```

同一房间的三字段投影（`batch.py:137-156`）去掉门窗、支撑声明和约束，只留外包矩形：

```json
{
  "schema_version": "fastfill.v2",
  "room": {
    "frame": "right_handed_z_up",
    "floor_polygon_xy_m": [[0.0, 0.0], [4.0, 0.0], [4.0, 3.5], [0.0, 3.5]],
    "floor_z_m": 0.0, "floor_known": true, "boundary_known": true,
    "height_m": 2.7, "room_type": "bedroom"
  },
  "constraints": [],
  "objects": [
    {"id": "obj_0000", "category": "bed", "description": "double bed with upholstered headboard"},
    {"id": "obj_0001", "category": "nightstand", "description": "wooden nightstand"},
    {"id": "obj_0002", "category": "nightstand", "description": "wooden nightstand"}
  ]
}
```

`predict --request` 的直接请求（`direct_layout.py:36-109`）渲染成上面的三字段文本，再把每个条目声明的 `support_parent` 加回对象：

```json
{
  "room_type": "bedroom",
  "room_size_m": [4.0, 3.5, 2.7],
  "furniture_list": [
    {"id": "obj_0000", "category": "bed", "description": "double bed with upholstered headboard",
     "support_parent": "floor"},
    {"category": "nightstand", "description": "wooden nightstand", "count": 2, "support_parent": "floor"}
  ]
}
```

训练行另带与目标顺序对齐的 validity。两个相同的床头柜构成可交换组（`matching.py:48-77`）；`yaw_symmetry_order` 为 2 表示只监督 yaw mod $$\pi$$：

```json
{
  "position": [[true, true, true], [true, true, true], [true, true, true]],
  "size": [[true, true, true], [true, true, true], [true, true, true]],
  "yaw": [true, true, true],
  "yaw_symmetry_order": [2, 2, 2],
  "size_axis_swap_allowed": [false, false, false],
  "exchangeable_group": [null, "anonymous_1", "anonymous_1"]
}
```

输出：

```json
{
  "schema_version": "fastfill.v2",
  "objects": [
    {"id": "obj_0000", "target_size_local_m": [2.0, 1.6, 0.9], "bottom_center_m": [2.0, 2.5, 0.0], "yaw_rad": -1.5707963},
    {"id": "obj_0001", "target_size_local_m": [0.45, 0.4, 0.55], "bottom_center_m": [0.95, 3.27, 0.0], "yaw_rad": -1.5707963},
    {"id": "obj_0002", "target_size_local_m": [0.45, 0.4, 0.55], "bottom_center_m": [3.05, 3.27, 0.0], "yaw_rad": -1.5707963}
  ]
}
```

[现状] 正式 schema 已实现：每层拒绝未知字段（`schema.py:10-16`）；数值必须有限；ID 为至多 256 字符的非空字符串、全局唯一、不得占用保留字 floor/wall（`schema.py:33-36, 274-276`）；地板与 keepout 多边形必须有效（`schema.py:39-50`）；支撑引用必须存在且无环（`schema.py:130-138, 277-281`）；校验从不修复输入（`schema.py:1`）。JSON 是预测的确定性序列化结果，不是几何训练的梯度来源。

2.4 归一化与未知字段

位置使用仅由输入确定的房间原点 $$o_R$$ 和正尺度 $$d_R$$：

$$\tilde p=(p-o_R)\oslash d_R.$$

[现状] `normalize_room`（`schema.py:316-324`）取：

$$o_R=(x_{\min},\,y_{\min},\,z_{\mathrm{floor}}),
\qquad
d_R=(x_{\max}-x_{\min},\,y_{\max}-y_{\min},\,H).$$

$$z_{\mathrm{floor}}$$ 缺失时取 0；$$H$$ 取已知房高，缺失时取固定的 3 m，不用监督对象高度补齐（`batch.py:178-180`）。归一化参数随 batch 保存，并用于输出反归一化。网格位置头在这个外包矩形归一化的 XY 上划分 16×16 格（第 4 节）；底面中心落在 $$[0,1)$$ 之外的目标被夹到边界格，residual 取 ±1（`geometry.py:37-47`）。

未知值不等于零；错误的零值会把“不知道”变成错误监督。[现状] 已有的未知标记：

| 字段 | 未知的表示 | 数据中的数量 |
|---|---|---|
| 房间边界 | `boundary_known: false` | train 6,751 行（[D] §2.4-b）；validation 492；test 409 |
| 地面 | `floor_known: false` | MultiScan 房间（`floor_z_m` 为 null，`legacy_bridge.py:86-91`）；资格处理改写的 12 个 Scan2CAD 房间（估计地面，`floor_z_m` 保留 0.0；manifest `qualification_change_counts["room.floor_known"]`，train 11、test 1） |
| 房高 | `height_m: null` | validation 2,160 行；test 1,883 |
| 目标字段 | NaN + validity 掩码（`batch.py:193-200`） | 全部 split 2,260,510 个对象中，有效位置 2,164,741、尺寸 1,533,512、yaw 2,162,611（manifest `validity_counts`） |
| 支撑 | 缺少 `support_parent` 一律按未知处理，不当作地面（`validation.py:216-217`） | train 中 68.5% 的对象有声明（[D] §4.3-c） |
| 门窗 | 没有标记 | 见第 2.5 节 |

validation、test 的行数于 2026-10-09 重新计数。

[现状] 只有 `support_parent` 为 floor 且 `floor_known` 时，地面对象的 $$p_z$$ 才由条件固定为地面高度，并屏蔽该坐标的监督（`batch.py:251-261`）。三字段投影去掉 `support_parent`，这时地面 z 由模型自己学。

2.5 门窗与已有物体：`room.fixed_objects`（新增）

[现状] 表示。每个固定物体是一个直立盒子：`id`（`fixed_%04d`）、`category`、`size_local_m`、`bottom_center_m`、`yaw_rad`（`schema.py:251-258`；`legacy_bridge.py:213-215`）。schema 还接受 `support_surfaces`、`capabilities`、`semantic_front_local`（`schema.py:252-269`），但数据中没有固定物体带支撑面。门窗没有单独的类型，靠 `category` 字符串识别。`room.openings` 从未使用；`_rigid_xy` 遇到 openings 会拒绝旋转（`batch.py:342-343`），Validator 只在 openings 非空时报告 `openings_unchecked`（`validation.py:393-394`）。

[现状] 数量：

| | train | validation | test |
|---|---|---|---|
| 行数 | 124,584 | 8,137 | 8,620 |
| 固定物体（manifest `fixed_objects_by_split`） | 342,976 | 24,304 | 24,819 |
| 至少有一个固定物体的行 | 103,128 | 6,906 | 7,039 |
| 有门盒的行 | 96,476（77.4%） | 6,422 | 6,538 |
| 有门或窗盒的行 | 101,659（81.6%） | 6,804（83.6%） | 6,926（80.3%） |
| 没有任何门窗盒的行 | 22,925（18.4%） | 1,333 | 1,694 |

三列都按类别（小写）包含 door / window 计数，与 `direct_layout._shell` 同规则（`direct_layout.py:155-156`），于 2026-10-09 全量计数。train 门盒 179,679 个、窗盒 134,046 个；按类别精确等于 door / window 计数时为 178,022 / 133,767，有门或窗的行为 101,486（[D] §1.1-b）。train 全量 124,584 行中 96,476 行（77.4%）有门盒，与 [M2] `result.plan` 的抽样（每 20 行抽 1 行，4,826/6,230）一致。

[现状] 使用方式：
- 条件文本：完整条件下，固定物体作为 room 段的 JSON 进入 Qwen 输入（`batch.py:80-103`）；三字段投影把它们全部去掉。
- 解码：spread 把固定物体当作不可移动的障碍，也可以承托高处物体（`evaluate.py:126-128, 204-212`）。
- 校验：请求对象与固定物体的重叠记为 `fixed_collision`（`validation.py:325-332`）；碰撞正则也包括固定物体，但权重为 0（config:45）。
- 交接：类别含 door/window 的固定物体挂到最近的墙，给出 `position_on_wall`、沿墙投影宽度和高度，窗另带 `sill_height`，写进 `roomgenbench_scene.json` 的 `room.doors` / `room.windows`（`direct_layout.py:140-174`）；装配时只有门在墙上开洞，窗和其余固定物体都画成 bbox 代理（`roomgenbench.py:166-203`）。
- 数据审计：IL3D_3dfront 斜向窗户必须有源 mesh 修正，否则整行拒收；倾斜的固定物体也让整行拒收（`legacy_bridge.py:205-212`）。

[现状] 缺口：
- 没有门前净空、通行或开门扫掠区域的检查。用户显式给出的 `clearance`、`keepout` 约束受支持（`validation.py:287-297`），但数据中没有。
- 门窗没有已知/未知标记：没有门窗盒的行（train 18.4%）无法区分“确实没有门窗”和“门窗未标注”。三字段投影去掉门窗时也不留未知标记，`boundary_known` 照样写成 true（`batch.py:153`）。标记方式尚未商定，不在 Stage 0/1 之内。在补上之前，含门窗的评测只能把“没有门窗盒”报告为“门窗未知或不存在”。
- RoomGenBench 请求不含门窗（`roomgenbench.py:253-265`），墙面物体可能挡住门洞（`README.md:227`）。

[现状] 推理时给出门窗对当前模型的影响（[M3]；数字与区间见第 9.10 节）：Isambard 两臂在 299 间共同房间上，含门窗与三字段的关系成立率差都不显著（−0.003 [−0.051, 0.046]，−0.012 [−0.063, 0.041]，n = 125），碰撞率由 0.040 / 0.058 降到 0.021 / 0.024。给出门窗降低了碰撞，但没有改善对象间关系，所以关系差距不是由缺少门窗输入造成的。这两组模型训练时有一半合格样本是三字段投影（config:81）。

[计划] Stage 0：
- 冻结含门窗的评测集：现有 300 个比较房间在 `validation.jsonl` 中都有完整条件版本，其中 220 个有门；另取一组不相交的 300 房间留作最终结论，不参与选模。
- 布局指标加门前净空；净空区定义见第 9.4 节（草案为门内侧 0.8 m 深、与门同宽，冻结前确定）。
- 关系解码把门当作带净空区的障碍。
- LLM 智能体拿到同样的 `fixed_objects`。

[计划] Stage 1：以含门窗的完整条件为主训练（第 6.8 节）。

2.6 三字段投影（新增）

[现状] 三字段投影（契约 K5）只保留房型、房间外包矩形和物品清单：

$$\Pi_3(R,O_{\mathrm{req}},C)=
\Bigl(\text{room\_type},\ \operatorname{AABB}(R),\ z_{\mathrm{floor}},\ H,\ \{(id_i,\text{category}_i,\text{description}_i)\}_{i=1}^{N}\Bigr).$$

实现为 `render_minimal_condition`（`batch.py:137-156`）：保留 `schema_version`；room 只剩 frame、4 点外包矩形、`floor_z_m`、`floor_known`、`boundary_known = true`、`height_m`、`room_type`；`constraints` 为空表；对象只保留 id、category、description。固定物体（含门窗）、openings、多边形细节、约束和 `support_parent` 全部去掉。

[现状] 适用房间与数量。只有边界已知、4 个顶点在 1 cm 内贴合外包框角点的轴对齐矩形才能投影（`batch.py:127-134`），其余房间计为 `skipped_non_rectangular_rooms`（`evaluate.py:68-74`）：
- train 65,987/124,584（53.0%）；
- validation 4,056/8,137；
- test 4,455/8,620，跳过 4,165 间（[D] §9-a）。

[现状] 用途：
- 训练：`minimal_form_p 0.5`（config:81）；
- 选模：`train.py:92` 的 `selection_metric` 是三字段投影验证行上的加权几何 loss（不经解码）；autorun 实际按 `autorun.score` 选模，用三字段投影和 spread 解码（`autorun.py:288-292`）；
- LLM 对照：前 300 条三字段验证行（`llm_baseline.py:189`）；
- 主测试表：三字段；完整条件的测试数字只出现在报告的 `projections.full`（`autorun.py:559`）；
- 直接请求：条目不带 `support_parent`、用 rectangular 语义时，`predict --request` 与投影逐字节等价；带 `support_parent` 时多出该字段（`direct_layout.py:97-109`；第 1.1 节）。

[现状] 三字段投影下房间朝向无法辨识。矩形在 180° 旋转和两种镜像下不变（正方形还有 90° 旋转），这些变换后的布局对应同一个输入；训练又做 rotate90 增强。所以每个对象的网格分布有并列的对称峰，各对象独立地挑一个峰。实测（[M2] `extra.json`，127 对同类对象，取自同房型、同尺寸但家具清单不同的房间，每间房该类别只有一个对象）：在这些房间之间，argmax 位置差的中位数为 0.43（房间归一化单位），允许整房镜像与 180° 旋转后降到 0.049；GT 在同样允许下仍为 0.20。对非正方形房间，至少有 $$\ln 2$$ nats 的网格 CE 无法消除（推导见第 4.7 节）。

[计划] Stage 0：三字段投影只作附加结果，主结果用含门窗的完整条件。Stage 1：
- loss 对无门窗的房间，在矩形对称群上对整个参考布局取一次最小（所有对象一致地选同一个副本）；有门窗的房间通常只剩恒等变换，因为门窗打破对称（第 6.7 节）；
- 训练以完整条件为主，例如把 `minimal_form_p` 降到 0.2，具体值未定。

3. 数据构造与审计

3.1 统一中间表示

每个训练样本分为四部分：

| 部分 | 内容 | 状态与证据 |
|---|---|---|
| condition | 房间、对象需求、明确约束；模型允许读取的信息 | [现状] 见第 2.3 节；`legacy_bridge.py:268-269` |
| target | 局部尺寸、底面中心、yaw | [现状] 没有关系标签或支撑监督目标；`permute_relations` 只在测试中调用（[D] §5.1） |
| validity | 各字段是否可信、yaw 对称阶、K1 轴交换许可、可交换组 | [现状] `position`/`size`（每坐标）、`yaw`、`yaw_symmetry_order`、`size_axis_swap_allowed`、`exchangeable_group`，按目标顺序对齐（`legacy_bridge.py:218-267`） |
| provenance | 原场景与资产来源、变换记录、派生版本、数据划分 | [现状] `source`、`scene_id`、`house_id`、`group`、`split`、逐对象 `field_evidence`（tilted、front_policy、size_semantics、selection_rule、z 吸附等）、`height_conflict`、`descriptions = source_desc_or_category`（`legacy_bridge.py:246-282`） |

训练标签来自有几何依据的场景与资产；功能需求、任务文本和关系描述只有在来源明确时才进入条件。可以从已有布局提取部分可验证关系，但不能把全部目标位置关系重新写入输入，再将重建结果当作无泄漏的布局生成能力。[现状] 现有约束是从参考布局抽取的 1–4 条成立关系（`fastfill/build.py:347-349`），只给部分关系，不给完整关系图；训练中以 0.3 概率整体丢弃（第 3.4 节）。

3.2 几何审计

每个数据源在接入时完成以下核验：

1. 尺寸与变换：单位、全长/半长、局部框/世界 AABB、缩放是否重复应用。[现状] `legacy_verify.py`、`multisource_verify.py`、`review_data.py`、`qualified_data.py` 自动复算。任一轴小于 3 mm 时屏蔽整个尺寸向量（manifest `degenerate_axis_m = 0.003`，`qualification_change_counts.size` 9,742 次）；log 尺寸超出 $$\pm10$$ 时屏蔽（manifest `size_output_policy`）；MansionWorld、OptiScene_holodeck 的代理尺寸整源屏蔽。
2. 位置与方向：center/pivot/bottom-center、坐标手性、轴序、yaw 单位和语义前向。[现状] 来源地面锚点距已知地面 2 cm 内才声明 floor 并吸附 z，更远的保持自由 z（K3，`legacy_bridge.py:26-27, 233-236`）；前向策略为 `axis`，语义前向只对 MultiScan 成立（第 2.1 节）。
3. 标签有效性：有限数、正尺寸、旋转范围、缺失字段及异常离群值。[现状] 目标超出房高 0.05 m 时只标记 `provenance.height_conflict`，不删标签也不删房高（K2，`legacy_bridge.py:28-47`）。
4. 场景关系：房间轮廓、门窗、楼层、支撑、悬挂及所需能力是否有可靠证据。[现状] 边界只有来源给出多边形时才记为已知（`legacy_bridge.py:86-87`）。支撑声明只来自来源标注：地面锚点、round 10 的墙面锚点（SAGE-10k、MansionWorld、SceneSmith，且几何上贴墙），以及来源声明了父物体（非推断）、且父物体在选中对象中的子物体；round 10 额外加入的子物体还须中心在父物体足迹内、底面在父物体盒内 ±5 cm（`legacy_bridge.py:127-160, 225-236`）。门窗只核验作为固定盒的几何，没有开口语义审计；悬挂（天花板）物体沿用旧处理，没有字段。
5. 视觉复核：将规范化 OBB 与原资产/原场景叠加检查，覆盖不同来源和异常样本。[现状] 未执行（`docs/fastfill-v2-data-audit.md:175`）。`ops/vlm_judge.py` 和 `bbox_visualization.py` 能渲染盒子，但只用于预测评审，不是数据复核。

不能用绝对值或大 epsilon 把负尺寸、零尺寸修成“有效标签”。轻微数值误差与错误数据来源应分开处理；变换规则和剔除原因都应可追溯。[现状] 资格处理只改掩码、不改目标数值（manifest `target_geometry_modified: false`）；逐成员改动写入 `changes.jsonl`，拒收原因写入 `rejections.jsonl`。

部分字段缺失时保留明确掩码。例如缺失高度会影响完整 size 和 3D box 监督，但不必否定独立可信的底面位置。[现状] 可交换组只要求成员位置完整（`matching.py:48-77`）；组内任一成员尺寸不完整时，匹配只用位置代价（`matching.py:156`），train 中 69,404/324,456 个组（21.4%）属于这种情况（[D] §5.2-a）；失去完整位置的组退回固定身份（manifest `qualification_change_counts.exchangeable_group` 98 次）。

[计划]（之后）学习整圈 yaw 之前，按来源审计 GT yaw 的 $$+X$$ 是否是真实前向（第 2.1 节）。

3.3 条件构造与泄漏控制

请求对象由训练场景对象派生，但条件只保留测试时确实可获得的信息：类别、普通描述、功能要求、已知支撑要求和真实尺寸限制。

[现状] 已满足的部分：
- 请求对象只有 `id`、`category`、`description`、`support_parent`；资产 ID 和支撑面不进入条件。
- 对象 ID 是样本内标识 `obj_%04d`，顺序由 SHA256(seed, uid, source_id) 决定，与目标几何无关（`legacy_bridge.py:197-200`；`provenance.request_order`）。
- 可交换组只写在 validity，从不渲染进条件文本（`batch.py:68-77`）。
- 模型输入只有条件 token、对象 token 区间和条件给定的固定值（`model.py:30-33`）。

[现状] 缺口：对象描述直接沿用来源文本，没有过滤尺寸（`legacy_bridge.py:163-165`）。用米制尺寸正则匹配，约 1.5k 个训练对象（[D] §3.3-a）、104 个验证对象、91 个测试对象（2026-10-09 重新计数）的描述含尺寸，例如 “modern 90cm wide range cooker…”。这些尺寸要么过滤，要么作为真实尺寸要求单独标注和统计；两种处理都未实现，也不在 Stage 0/1 之内。

真实用户给出的固定尺寸或范围保留，并标注为带尺寸约束的条件子集。训练与评测分别统计该子集，避免它与完全自由尺寸任务混淆。[现状] 数据中没有这样的对象，所以这个子集为空，对应的尺寸满足率指标也为空（[D] §3.3-b、§9-b）。

训练中的固定尺寸和范围条件必须与示范几何相容。若人为改变尺寸要求，应重新生成并验收对应布局；不能只覆盖 size 字段，再沿用可能发生碰撞或支撑失效的原始位姿和框标签。[现状] `_geometry_rows` 遇到固定尺寸、尺寸范围或地面声明与标签不一致时直接报错（`batch.py:244-253`）。

[现状] 支撑声明是条件的一部分，但条件之间不对等：train 中 68.5% 的对象带 `support_parent`；`drop_support_p 0.2` 会整体丢弃，三字段投影也会丢弃；RoomGenBench 请求则为每个对象声明 `place_id`（`roomgenbench.py:262-265`）。

[计划] Stage 1 的条件（masked）布局训练会把随机一部分对象的真值位姿写入其 slot，预测其余对象；锚点优先的迭代解码在推理时把已放置对象的预测位姿写回（第 6.8、4.9 节）。规则：
- 测试时进入条件的位姿只能来自已放置的预测或用户给定的位姿，不能来自待评测的目标；
- 评测报告注明使用了几轮解码；
- 不能把“给出部分真值位姿”的结果当作无条件生成的结果报告。

3.4 划分、增强与样本完整性

先按底层房屋、扫描或场景划分训练、验证、测试，再生成文本或几何增强版本。同一场景的裁剪、不同视角与改写不得跨集合。

[现状] 划分沿用 v3.2 父数据，不重新划分（manifest `split_rule = inherit parent without re-splitting`）。父划分把共享同一房屋/扫描 `group`、`group_aliases`、家具内容键或布局键的房间连成连通分量，每个分量按稳定哈希整体进入一个集合；含仅评测来源的分量整体进入 test（`fastfill/split.py:1-9`）。`legacy_verify` 检查同一底层身份不跨集合（`legacy_verify.py:183-200`）。manifest `split_integrity`：重复 UID 0，跨集合组冲突 0，检查别名 60,656 个。

| | train | validation | test |
|---|---|---|---|
| 房间行 | 124,584 | 8,137 | 8,620 |
| 请求对象 | 1,998,842 | 126,572 | 135,096 |
| 底层房屋 | — | 3,146（1,294 个多房间房屋，最多 193 间） | 3,319（1,300 个多房间房屋，最多 252 间） |
| 三字段合格行 | 65,987 | 4,056 | 4,455 |

房屋数按 `provenance.house_id` 于 2026-10-09 计数。同一房屋的多个房间相关，置信区间应按房屋聚类（第 9.8 节）。

[现状] 来源。16 个训练来源标签：HSSD200、IL3D_3dfront、IL3D_synthetic、InteriorGS、InternScenes 的 3rscan/arkit/gen/mp3d/scannet、MansionWorld、MultiScan、OptiScene_holodeck、SAGE-10k、Scan2CAD、SpatialLM、Structured3D。SceneSmith 与 SpatialGen 只在 test（manifest `evaluation_only_sources`）。各来源的行数与有效标签数见 manifest `source_split_samples`、`source_split_validity`。

[现状] RoomGenBench 留出：5 个 SAGE 基准房间（manifest `roomgenbench_holdout_groups`）从 train 移到 test，带 `provenance.holdout_reason = roomgenbench_benchmark_room`（test 中 5 行）。`io.read_samples(training=True)` 拒绝读入它们（`io.py:16, 52`）。round 10 起，这 5 个房间带有全部 313 个对象及 RoomGenBench 的 `place_id`（`README.md:221, 236`）。

[现状] 评测子集：
- 资产不重叠子集未建立，只有上面的来源级留出（[D] §3.4-a）；
- 真实尺寸要求子集为空（第 3.3 节）；
- 选模用的验证行先剔除带 `oob_objects`、`fixed_collision`、`overlapping_furniture` 标记的行（config:84-88）；
- 300 条 LLM 对照行是前 300 条三字段验证行，属于选模队列，不是留出集（`README.md:849`）。

[计划] Stage 0 冻结两组含门窗的评测房间（第 9.7 节），其中最终结论用的一组不参与任何选模或调参。

增强包括整个房间与对象的联合旋转、对象输入顺序变化和匿名 ID 重命名；旋转时同步变换门窗、面向、关系和全部目标。随机改变真实家具尺度不是默认增强。[现状] 实际增强（`batch.py:283-285, 388-421`；config:74-82；按 (seed, epoch, row) 播种，`train.py:353`）：

| 增强 | 主配置 | 行为 |
|---|---|---|
| `rotate90` | true | 房间、固定物体（含门窗）、约束方向、keepout 多边形和目标统一绕 $$+Z$$ 旋转 $$k\cdot90^\circ$$（$$k$$ 在 0–3 中均匀取），再平移回原外包最小角；position 掩码随轴交换（`batch.py:334-370, 404-407`） |
| `mirror` | false | 不启用（config:76） |
| `shuffle_objects` | true | 打乱请求顺序，重编号为 `obj_%04d`，同步改写全部引用与 validity（`batch.py:301-321`）；顺序打乱与 ID 重命名是同一步 |
| `drop_constraints_p` | 0.3 | 清空约束（`batch.py:408-410`） |
| `drop_support_p` | 0.2 | 去掉全部 `support_parent`（`batch.py:411-413`） |
| `category_only_description_p` | 0.5 | 把描述整体替换为类别名（`batch.py:414-415`）；这是降级增强，不是语义改写 |
| `minimal_form_p` | 0.5 | 对合格房间改用三字段投影（`batch.py:416-418`） |

任一丢弃或投影发生后，按建库规则重算可交换组（`batch.py:372-385, 419-420`）。没有平移增强，因为预处理把外包最小角固定在原点；没有任意角度旋转，没有语义改写，没有尺度增强。评测从不增强；三字段评测用同一投影函数，固定种子（`evaluate.py:64, 68-74`）。

长度裁剪应保持对象及其约束、支撑引用的完整性；删去支撑桌子而保留其上的物体，会制造错误场景。[现状] 不做任何裁剪：对象数超过 `max_objects` 或条件超过 `max_length` 时整条样本拒收，不截断（`batch.py:443-447`）。主配置 `max_objects` 为 128（config:10）、`max_length` 为 8192（config:58）；Isambard 两组运行改为 256。完整条件下超过 128 个对象的行：validation 81 行，test 73 行（2026-10-09 计数）；三字段合格行中只有 validation 3/4,056、test 5/4,455（`autorun/SUMMARY.md`）。增强后超长的样本退回预检过的原行，并计入 `augmentation_fallbacks`（`train.py:759`）。

训练集尺寸统计、类别参考尺寸、词表映射和过滤阈值只由训练数据确定。[现状]
- 模型输入符合：归一化只依赖各自房间（第 2.4 节），`size_reference` 是常数 $$[1,1,1]$$（config:11-15）；训练器日志中的基线用前 20,000 个训练样本拟合（`train.py:649`）。
- 评测不符合：`evaluate.py` 在不传 `--baseline-fit` 时，直接在评测集上逐行留一拟合平凡基线（`evaluate.py:927-935`）；autorun 从不传这个参数，而 `autorun.score` 以这些基线为分母（`autorun.py:44-69`）。本规范要求报告平凡基线时用 `--baseline-fit` 指向训练集；这一项尚未执行，第 7.7 节的选模规则草案把它列入。

[计划] Stage 0 的关系先验只从训练集挖掘。现有的 32 个模板来自 train 每 20 行抽 1 行得到的 6,230 个房间（`scratchpad/arrange/relations.json`，不在 main 上）；同一组模板用于关系指标，所以关系解码的结果必须与原始模型输出分开报告，标为“FastFill + 挖掘先验（后处理）”。

4. 模型：条件编码与结构化对象解码

4.1 主架构

[现状] 当前主模型如下（`model.py:125-227`；`config:2-34`）：

```
房间 + 对象清单 + 约束（完整条件；或三字段投影，见 4.2）
            │
            ▼
Qwen3-8B 条件编码器（bf16 冻结权重 + LoRA r=8，作用于 q/k/v/o）
            │ 全部条件 token 的 hidden states → Linear 投影到 128 维
            ▼
N 个绑定请求对象的 slots = 该对象 token 区间均值 + slot seed[序号]
            │
            ▼
对象解码器：2 层 TransformerDecoder（宽 128，4 头，FFN 512，pre-norm）
  双向 self-attention（只屏蔽 padding）+ 对全部条件 token 的 cross-attention
            │
            ├── position head：16×16 格 logits + 每格 XY tanh residual + 回归 z
            ├── size head：ŝ = 1 m · exp(clip(u, −10, 10))
            └── yaw head：12 个 bin logits + 12 个 tanh residual
            │
            ▼
解码：spread（默认交付）或 argmax（原始头输出），见 4.5
            │
            ▼
validate_layout 做 schema 校验 → JSON 布局
```

[现状] 骨干必须是 Qwen 系列，否则直接报错（`model.py:125-129`）。运行配置固定本地 Qwen3-8B checkpoint、bf16 权重，用 PEFT LoRA 包装：rank 8、alpha 16、dropout 0，作用于 q/k/v/o（`config:3,21-30`；`model.py:133-137`）。条件 hidden states 经 `memory_projection` 投影到 128 维（`model.py:148,193`）。对象解码器为 2 层 `nn.TransformerDecoder`：宽 128、4 头、FFN 倍数 4、pre-norm、dropout 0（`config:4-8`；`model.py:150-153`）。

[现状] 运行配置中 `max_objects` 为 128（`config:10`）；Isambard 上的两组训练用 `--set model.max_objects=256` 覆盖（`ops/isambard_autorun.sbatch:17`）。超出预算的请求整条拒绝，评测计为超容量（`batch.py:443-444`）。

[现状] 主模型直接按第 6 节的几何目标训练，不需要先训练文本布局模型。文本 SFT 对照只跑过一次冒烟（第 7.9 节）。

[设计] 这是条件语言模型上的结构化几何生成器：保留语言条件能力，位置格、尺寸和角度直接接受监督。

[现状] 所有 slot 在一次并行前向中各自输出自己的头。对象之间的依赖只能经 self-attention 进入每个 slot 自己的分布，输出端没有联合分布；4.7 说明这一形式为什么表达不了对象间关系。

4.2 条件编码、条件形式与 causal mask

[现状] 模型输入只有条件 token、对象 token 区间、slot 掩码和条件给定的固定值（`model.py:30-33`），目标几何不进入输入。条件按固定字段顺序 schema_version → room → constraints → objects 序列化，每个对象的 JSON 片段单独编码，以得到精确的 token 区间（`batch.py:67,80-103,159-169`）。对象片段在最后，所以 Qwen 的因果注意力让每个对象 token 都能看到房间和约束。超过 `max_length` 8192（`config:58`）的样本整条拒绝，不截断（`batch.py:446-447`）。

[现状] Qwen 是 causal decoder：把 object token 放到条件前面，它看不到后续条件；放到末尾，也不会自动获得对象间的双向交互。所以对象间交互由外部双向解码器提供，不依赖改变文本排列就能得到集合解码的假设。

[现状] 条件有两种形式：

| 形式 | 内容 | 使用位置 |
|---|---|---|
| 完整条件 | 地板多边形、房高、房型、`room.fixed_objects`（门窗是 category 为 door/window 的盒子）、约束、对象的 support_parent | 训练中未被投影的样本；测试报告的 `projections.full` |
| 三字段投影 `render_minimal_condition` | 外包矩形、房型、房高，以及对象的 id/category/description；去掉 fixed_objects（含门窗）、约束和 support_parent | 训练中以 `minimal_form_p` 0.5 用于边界已知的轴对齐矩形房间；选模、主表和 LLM 对照 |

两种形式的定义、适用房间、数量和用途见第 2.5、2.6 节；条目不带 `support_parent`、用 rectangular 语义时，直接请求 `predict --request` 的条件文本与三字段投影逐字节相同（`batch.py:145-146`），带 `support_parent` 时多出该字段（第 1.1 节）。

[现状] 三字段投影后，矩形房间转 180° 文本完全不变，正方形房间转 90° 也不变。训练的 rotate90 增强（`config:75`；`batch.py:404-407`）会把同一份目标布局以旋转后的形式配给这同一段文本。于是对这类条件，同一房间的几个对称副本都是合法监督。这正是 4.7 所述朝向歧义的来源。

[计划] Stage 0：评测集改为含门窗的完整条件。Stage 1：训练主要使用含门窗的完整条件，例如把 `minimal_form_p` 降到 0.2。三字段结果降为附加结果，并写明被跳过的房间数（第 9.1 节）。

4.3 对象 slots 与身份

[现状] 每个请求实例一个 slot。slot 的初始特征是该对象 JSON 片段的投影 hidden states 均值，加上按 slot 序号学习的 seed（`model.py:163-181`）：

$$h_i^{(0)}=\frac{1}{|T_i|}\sum_{t\in T_i}W_m\,m_t+e^{\mathrm{seed}}_{i}.$$

其中 $$T_i$$ 为对象 $$i$$ 的 token 区间，$$m_t$$ 为 Qwen 最后一层 hidden state，均值至少按 float32 累加（`model.py:170-179`）。

[现状] 没有单独的类别嵌入或约束角色特征。类别、描述和 support_parent 只以文本形式出现在 $$T_i$$ 中；约束只能经编码器的因果注意力和解码器的 cross-attention 间接进入（[D] §4.2）。schema 允许 `constraint_role` 字段，但数据中没有一个对象带它。

[现状] 训练时 `shuffle_objects` 会打乱请求顺序，并重编号为 `obj_%04d`（`batch.py:301-321`）。因此 slot seed 实际是随机的对称性破缺量，不携带目标位置。

[现状] padding slot 在解码器 self-attention（`tgt_key_padding_mask`）、全部输出、匹配和 loss 中一致屏蔽（`model.py:196-199,224-227`；`matching.py:119-120`；`objective.py:20-25`）。对象类别、ID 和数量都由请求继承，模型没有 objectness 或 background 头（`model.py:156-160`）。

[现状] 不可交换对象固定对应，匿名重复对象在第 5 节的已认证组内匹配。只用 Hungarian 不保证网络严格 permutation-equivariant。输入顺序与 ID 重命名的稳定性还没有实测：只有诊断脚本 `ops/diag_ablation.py` 的变体 e，它把顺序打乱和重命名合在一起，结果尚未产出（[D] §4.3-b）。

[现状] 模型没有支撑预测头。上游声明 `support_parent: floor` 且地面已知时，z 由条件直接给定（`batch.py:251-261`；`model.py:215-216`）。其余支撑声明只作为对象文本输入，由运行时 Validator 核验（`validation.py:213-244`），并指导 spread 放置（`evaluate.py:138-145`）。三字段投影去掉 support_parent，这时 z 全由模型回归。

4.4 数值输出头

[现状] 位置归一化只依赖输入（`schema.py:316-324`）：

$$o_R=(x_{\min},\,y_{\min},\,z_{\mathrm{floor}}),\qquad d_R=(x_{\max}-x_{\min},\ y_{\max}-y_{\min},\ H),\qquad \tilde p=(p-o_R)\oslash d_R.$$

XY 取地板多边形的外包范围；$$H$$ 为房高，缺失时取固定的 3 m；地面高度缺失时 $$z_{\mathrm{floor}}$$ 取 0。不用监督对象补齐任何尺度。

[现状] 位置头。主配置为 `position_head = "grid_residual"`、`position_grid` $$G=16$$（`config:18-19`）。位置头是一个 `Linear(128, 3G^2+1)`，输出拆成三部分（`model.py:154-157,203-209`）：

- $$G^2=256$$ 个格 logits $$\ell_i$$；
- 每个格一对经 tanh 的 XY residual $$\varrho_{i,k}\in[-1,1]^2$$，以半格为单位；
- 一个回归的 $$\hat{\tilde z}_i$$。

格按行优先编号，$$k=G\,a+b$$。对监督位置（`geometry.py:37-47`）：

$$a^*=\operatorname{clip}\big(\lfloor G\tilde x^*\rfloor,0,G-1\big),\qquad b^*=\operatorname{clip}\big(\lfloor G\tilde y^*\rfloor,0,G-1\big),\qquad k^*=G\,a^*+b^*,$$

$$\varrho^*=\operatorname{clip}\!\left(2G\Big(\tilde p^*_{xy}-\frac{(a^*,b^*)+\tfrac12}{G}\Big),\,-1,\,1\right).$$

外包矩形以外的底面中心被夹到边界格，residual 取 ±1。原始解码取 argmax 格的中心，加上该格的 residual（`geometry.py:50-58`），再按上式反归一化：

$$\hat k_i=\arg\max_k\ell_{ik},\qquad \hat{\tilde p}_{i,xy}=\frac{(\hat a_i,\hat b_i)+\tfrac12}{G}+\frac{\varrho_{i,\hat k_i}}{2G}.$$

[现状] 纯连续回归的 `regression` 头仍保留，而且是 `ModelConfig` 的默认值（`model.py:53,157`），但不是主配置。regression 头的正式配置（`configs/qwen3_8b_main_4gpu_regression.json`）与 `position_cell` 0.04 / 0.2 / 0.5 / 1.0 的配置都存在；`README.md:606-612` 记录了最终取 0.5 / 0.4，但没有记录两种头之间的比较结果。select1 只用 `autorun.score` 比较了 0.5 与 1.0（`autorun/SUMMARY.md`）。

[现状] z 的处理：`fixed_position_mask` 只覆盖 z，XY 从不由条件固定（`batch.py:260-261`）。被固定的 z 在输出中直接替换为地面高度（`model.py:215-216`）。

[现状] 尺寸头（`model.py:210-213`；`config:11-16`）：

$$\hat s=s_{\mathrm{ref}}\odot\exp\big(\operatorname{clip}(u,-10,10)\big),\qquad s_{\mathrm{ref}}=(1,1,1)\ \mathrm m.$$

指数在 float32 中计算，参数化保证正值。合法范围和功能尺度仍靠数据监督和验证器。请求给出 `fixed_size_local_m` 的维度直接替换为给定值，并从尺寸监督中屏蔽（`model.py:217-220`；`losses.py:146,152-155`）。`size_bounds_local_m` 只作为对象文本输入，由 Validator 检查，没有有界参数化或投影。当前训练、验证、测试数据中没有任何对象带这两个字段（[D] §3.3-b）。

[现状] 角度头：`yaw_bins` 为 12，输出 12 个 logits 和 12 个经 tanh 的归一化 residual（`config:9,17`；`model.py:159-160,221-223`）。编码、解码和损失见 6.3、6.4。

4.5 解码：argmax 与 spread

[现状] 同一次前向的头输出有两种确定性解码（`evaluate.py:117`），之后都经 `validate_layout` 做 schema 校验，再序列化为 JSON（`evaluate.py:373-381`；`schema.py:295`）：

| 解码 | 定义 | 使用位置 |
|---|---|---|
| argmax（原始头输出） | 每个 slot 独立取 argmax 格加 residual、回归 z、argmax yaw bin 加 residual | `--grid-decode argmax` |
| spread（贪心后处理） | `spread_grid_xy`，规则见下 | `predict`、`evaluate` 以及 autorun 选模和测试的默认值（`predict.py:39`；`evaluate.py:1067`；`README.md:613-617`） |

[现状] spread 的规则（`evaluate.py:121-328`）：

1. 放置顺序：先放地面物体，再放高处物体；父物体先于其声明的子物体；同一层按占地面积从大到小（`evaluate.py:238`）。
2. 每个物体按自己的格 logits，在 top-64 格中取第一个满足条件的位置。条件是：占地留在地板多边形的外包矩形内（`evaluate.py:229, 336`；非矩形房间可能越出多边形；越界不超过 0.02 m 时先推回）；与已放置物体或固定物体在高度区间重叠时，重叠面积除以较小占地不超过 0.15，碰撞按旋转占地的轴对齐外包比较（`evaluate.py:179-180`）。都不满足时取重叠最小的格（`evaluate.py:122,294-326`）。
3. 支撑声明优先于预测高度。声明 floor 的贴地；声明 wall 的投影到最近的墙，背面贴墙、面向室内；声明在请求对象或固定物体上的，放在父物体顶面或内层上；未声明的高处物体，按规则悬挂在近墙处或落在下方的落地物体上（`evaluate.py:138-177`）。
4. 朝向：模型只学 yaw mod π（见 6.4）。对离墙 0.3 m 以内的落地物体，若翻转 π 后的朝向与“背向最近墙”相差不超过 45°，就把 yaw 翻转 π；不做 90° 转动。被 faces 或 faces_direction 约束点名的对象不翻转（`evaluate.py:158-167,289-293,358-369`）。docstring 写明，这一规则按设计也会把面向桌子的椅子翻成背对墙（`evaluate.py:163-164`）。

[现状] spread 是不可微的推理端后处理，不参与训练。它逐个贪心放置物体，只按各自的格分数避让碰撞，不建立物体间关系。报告必须分别给出 argmax 与 spread 的结果。

[M1] 中的对比：
- 碰撞（重叠超过 20%）：argmax .406，spread .103；
- 关系成立率：argmax .253，spread .433；
- 书桌椅面向书桌：GT .66，spread .10，argmax .20；spread 输出中 46% 的书桌椅正好背对书桌（[M1] `relation_probe.out`）。

注意：[M1] 的 FastFill 输出由 commit a21a67fdb 解码，该版本的近墙规则是“取背向墙 45° 内概率最大的 yaw bin”，与当前 main 只翻转 π 的规则不同（[M2] result.metrics）。

4.6 梯度边界

[现状] 尺寸、z 和 yaw 头的输出直接进入各自的 loss。XY 通过格 CE 和 GT 格的 residual 受监督（`losses.py:101-114,164-170`）。argmax 解码出的 XY 不进入任何有梯度的项：它只用于记录 `term_sums["position"]`，以及 box、collision、boundary 三项，而这三项在运行配置中权重都是 0（`losses.py:160-161,214-235`；`config:40,45-46`）。

[现状] Hungarian 在 `torch.no_grad()` 下求索引（`matching.py:89`），对称候选的选择也是 detached argmin（`losses.py:201`）。JSON 序列化和 spread 都发生在训练损失之外。

[现状] 对生成的 JSON 做字符串解析再算几何距离，不能把梯度传回自回归 token 采样。文本模型的 token CE 是另一条训练路径（第 7.9 节），不是这里的连续几何监督。

4.7 逐对象边缘分布与朝向歧义：现有输出为什么表达不了关系

[现状] 现有模型对一个房间输出逐对象独立的分布，解码也逐对象独立：

$$q_\theta(L\mid c)=\prod_{i=1}^{N}q_\theta\big(k_i,\varrho_i,z_i,s_i,\theta_i\mid c\big).$$

训练目标（第 6 节）是每个对象对单份参考布局的逐项损失之和。对位置格而言，$$\sum_i\mathrm{CE}$$ 的总体最优解就是每个对象各自的条件边缘分布 $$P(k_i\mid c)$$。对象之间的联合依赖，例如“沙发在左则电视在右”，不在这个目标的表达范围内；argmax 又为每个对象单独选峰。

[现状] 朝向歧义叠加在这之上。若条件 $$c$$ 在房间变换 $$g$$ 下不变，则 $$P(k_i\mid c)=P(g\,k_i\mid c)$$。三字段投影加 rotate90 时，$$g$$ 是 180° 旋转；正方形房间还有 90° 旋转。$$G=16$$ 为偶数，这些旋转在格上没有不动格。所以对这类条件，每个对象的格分布至少有 $$\ln|\mathcal H_c|$$ 的熵：非正方形房间为 $$\ln 2$$，正方形为 $$\ln 4$$。这部分 CE 无法消除。各对象再各自从对称副本中挑一个峰，同一房间的对象就落在不同副本上。

[现状] 证据：
- 关系成立率（[M1]；定义见第 9.4 节，全部数字见第 9.10 节）：FastFill argmax .253，与同房间 GT 位置打乱后的 .225 相差约 +0.03，95% CI 跨 0；spread .433，LLM harness .764，GT .714。
- 127 对同类对象（取自同房型、同尺寸但家具清单不同的房间，每间房该类别只有一个对象）：argmax 位置的中位差为 0.432（房间归一化单位）；允许 180° 旋转和两个镜像后降到 0.049。GT 在同样允许下仍为 0.20。来源：[M2] `extra.json`。
- argmax 输出中 25.8% 的落地家具与同房间另一物体共用一个格，GT 为 0.8%。来源：[M2] `analysis.json`。
- 格 CE：均匀分布 5.545 nats；按 rotate90 对称化的“类别 × 房间尺寸”查表先验在平滑参数 β=10、30 时为 5.255、5.116 nats（β=3 时为 5.453；3,416 个对象，[M2] `prior_ce.json`；查表用 2026-10-06 的数据构建；[M2] `result.head` 引用为 5.12–5.20）。模型验证集格 CE 约 4.8 nats，这个数出自诊断任务简报，本地没有日志，未核实。
- 门窗信息本身没有带来可测的关系改善：Isambard 两臂含门窗减三字段的关系成立率差为 −0.003 [−0.051, 0.046] 和 −0.012 [−0.063, 0.041]，碰撞率则约减半（[M3]；第 9.10 节）。
- 选模分数 `autorun.score` 只包含三项单参考误差与平凡基线之比和几项分布偏差（`autorun.py:44-69`），从未度量对象间关系。

[现状] FastFill 在不依赖关系的量上仍有优势，Stage 1 必须保住：log-size 误差 .237 对 LLM 智能体 .388–.412，每房间延迟中位数 56 ms 对 LLM harness 16.8 s（[M2]；第 9.10 节）。

[现状] 结论：加大解码器、调 yaw_cls 或 max_objects 都不改变上述分解形式。[M2] result.plan 第 3 点曾预测，Isambard 两臂（只改了 yaw_cls、max_objects 和第 10 轮数据）的 spread 关系成立率会停在 0.4–0.5 左右。[M3] 测得两组为 .390 / .387，与 [M1] 中旧模型 spread 的 .433 属同一水平，未见改善；房间集合和解码版本不同，两者不做显著性比较。

[计划] Stage 1 的两处改动：
1. 让同一房间的所有对象在同一个对称副本上受监督（6.7）；
2. 让后放置的对象以已放置对象为条件（训练见 6.8，解码见 4.9）。

4.8 Stage 0 关系解码：挖掘先验的后处理

[计划] 这一步不重新训练，在保存的头输出上进行；`evaluate --save-head-outputs` 已在 main 上（`evaluate.py:23-24`）。步骤：
1. 先按 FastFill 的格分布放置锚点：床、沙发、书桌、餐桌、电视柜。
2. 每个依附对象 $$j$$（其锚点为 $$a(j)$$）取

$$\hat k_j=\arg\max_k\Big[\log q_\theta(k\mid c)+\lambda\,\log p_{\mathrm{rel}}\big(T_{a(j)}^{-1}(x_k),\ \theta\ \big|\ \mathrm{cat}_{a(j)},\mathrm{cat}_j\big)\Big].$$

其中 $$x_k$$ 为格中心；$$T_a^{-1}$$ 把它变换到锚点的局部坐标；$$p_{\mathrm{rel}}$$ 从训练房间挖掘（[M1] 的关系指标已从 6,230 间训练房间挖出 32 个模板，见第 9.4 节）；$$\lambda$$ 在验证集上选定。
3. 依附对象朝向锚点，不做背墙翻转；门作为带净空区的障碍。

[计划] 结果单列为“FastFill + 挖掘先验（后处理）”，不与原始模型混报。它的上限是：无法修正放错的锚点，而且是规则，不是学习。

4.9 Stage 1 迭代式锚点优先解码

[计划] 这一解码与 6.8 的条件训练配套。Qwen 只前向一次，对象解码器和各头运行少数几轮，记为 $$R$$ 轮（例如 2–3，在配置中固定）：

$$M=\mathrm{Enc}(c),\qquad h^{(0)}=\mathrm{Pool}(M)+e^{\mathrm{seed}},$$

$$\hat Y^{(r)}=\mathrm{Heads}\Big(\mathrm{Dec}\big(h^{(0)}+\mathbb 1[i\in\mathcal K_{r-1}]\;E_\psi(\phi(\hat y_i)),\ M\big)\Big),\qquad \mathcal K_r=\mathcal K_{r-1}\cup\mathcal S_r,\qquad \mathcal K_0=\varnothing.$$

- $$\mathcal S_1$$ 是锚点类别的请求（床、沙发、书桌、餐桌、电视柜；类别表在配置中固定）。
- 剩余对象在之后各轮放置，最后一轮放完全部；分组规则在配置中固定。
- 已放置对象的位姿 $$\hat y_i$$ 冻结，作为后续轮次的给定位姿输入。编码 $$\phi$$ 和嵌入 $$E_\psi$$ 与训练一致（见 6.8）。
- 没有锚点的房间退化为一轮，即现有解码。

[计划] 这相当于用链式分解近似联合分布：

$$P(L\mid c)\approx P(Y_{\mathcal S_1}\mid c)\prod_{r\ge 2}P\big(Y_{\mathcal S_r}\mid c,\ Y_{\mathcal S_{<r}}\big).$$

同一轮内的对象仍按逐对象边缘分布独立解码，所以锚点之间的相互关系不由这一机制保证。

[计划] 每轮新放置对象的解码规则与 4.5 相同（argmax 或 spread）。用 spread 时，已放置对象和固定物体都作为障碍；原始结果与后处理结果分开报告。在整圈 yaw（6.9）落地之前，模型仍只学到 yaw mod π，两种朝向之间的选择沿用 4.5 的规则；若同时启用 4.8 的关系解码，则依附对象朝向其锚点。

[计划] 代价：Qwen 前向占延迟的主要部分，额外开销只是 $$R-1$$ 次 2 层解码器。验收要求 p50 延迟不超过 150 ms、log-size 误差不超过 0.26（成功判据见第 7.8 节）。

5. 实例对应与受约束 Hungarian matching

5.1 先定义允许的对应

[现状] 匹配只解决“几个真正相同的请求实例可以用任意顺序表示”的问题，不修改用户定义的身份或角色。规则如下：

| 对象条件 | 对应规则 |
|---|---|
| 有固定 ID 语义或唯一角色 | 固定对应 |
| 同类但描述、尺寸要求、面向要求或支撑角色不同 | 固定对应 |
| 除 id 外请求字段完全相同、交换后约束图和支撑引用不变、位置标签完整的匿名实例 | 组内 Hungarian |
| 交换需要其他组同步置换才能保留关系 | 保留固定身份 |

[现状] 实现（`matching.py:30-77`）：
- `group_labels` 只把除 id 外字段完全相同、而且位置标签完整的请求归为候选组。
- `certify_group` 检查组内交换不会改变约束图和支撑引用。不满足时，退到组内没被任何约束或 support_parent 引用的成员；仍不满足就保留固定身份。
- 增强中丢弃约束、丢弃支撑、把描述降为类别名，或改用三字段投影之后，分组会重新计算（`batch.py:372-385,419-420`）。
- 分组标签记录在 `validity.exchangeable_group`，不渲染进条件文本（`batch.py:68`）。

[设计] 例如，两把无差别的餐椅可以属于同一可交换组；“面向屏幕的椅子”和“面向门的椅子”不能因为类别相同就交换。

5.2 匹配成本

[现状] grid_residual 头下，对每个已认证的可交换组 $$\mathcal G$$，配对成本为（`matching.py:139-164`）：

$$C_{ij}=\alpha_p\Big(\mathbb 1^{xy}_i\big[w_{\mathrm{cell}}\,\mathrm{CE}(\ell_i,k^*_j)+w_{\mathrm{res}}\tfrac12\|\varrho_{i,k^*_j}-\varrho^*_j\|_1\big]+\mathbb 1^{z}_i\,\tfrac13\big|\hat{\tilde z}_i-\tilde z^*_j\big|\Big)+\alpha_s\,\mathbb 1\big[\mathcal G\ \text{尺寸标签全有效}\big]\min_{\sigma\in\Sigma_j}\big\|\log\hat s_i-\log\sigma(s^*_j)\big\|_1,$$

$$\pi_{\mathcal G}^*=\arg\min_{\pi\in\operatorname{Perm}(\mathcal G)}\sum_{i\in\mathcal G}C_{i,\pi(i)}.$$

各符号的含义：
- $$\mathbb 1^{xy}_i,\ \mathbb 1^{z}_i$$ 表示预测 slot $$i$$ 的相应坐标需要学习，即未被条件固定。
- $$w_{\mathrm{cell}}=0.5$$，$$w_{\mathrm{res}}=0.4$$（`config:50-51`）。所以位置部分正好等于该配对下第 6.2 节的格位置 loss。
- $$\alpha_p=\alpha_s=1$$（`config:43-44`）。
- 若目标 $$j$$ 的 `size_axis_swap_allowed` 为真，$$\Sigma_j=\{\mathrm{id},\ \mathrm{swap}_{wd}\}$$，否则 $$\Sigma_j=\{\mathrm{id}\}$$。

[现状] 实现细节：
- 成本使用底面中心，不混用几何中心。
- 尺寸项是三维 L1 之和，不除以 3，也不乘 loss 权重：匹配权重与训练 loss 权重是独立的。
- 组内任一成员的尺寸标签不完整时，去掉尺寸项；train 中有 69,404 / 324,456 个组（21.4%）属于这种情况（[D] §5.2-a）。任一成员的位置标签不完整时，整组保留固定身份（`matching.py:134`）。
- 与 loss 不同，尺寸成本不考虑请求固定了 $$w$$ 或 $$d$$ 的情况。当前数据没有固定尺寸，所以目前没有差别。

[现状] regression 头和评测时的匹配用解码位置的 L1：$$\alpha_p\|\hat{\tilde p}_i-\tilde p^*_j\|_1$$ 加同样的尺寸项（`matching.py:155`）。评测的布局不带格输出，所以走这条路径（`evaluate.py:440-447,495-499`）。

[现状] yaw 和 box 都不在匹配成本中，相关消融未实现（`matching.py:80-90`）。位置成本完全相同而 yaw 不同的平局，由 SciPy 的行列顺序决定，可能改变 yaw loss（`README.md:212`）。旋转框重叠成本即使没有显式 yaw 项也会隐式依赖角度，所以在没有实验之前，不声称省略它们更稳定。

5.3 实现与退化情况

[现状] 流程如下：

1. 每个样本的 slot 必须连续，并与请求一一绑定（`matching.py:119-120`）。固定身份对象直接取 $$\pi(i)=i$$。
2. 每组再认证一次；位置标签不完整的组和单元素组跳过（`matching.py:132-135`）。
3. 在 `no_grad` 下构造成本，检查有限性：grid 成本以 float64 累加，Metal 设备上改在 CPU 计算（`matching.py:136-166`）。然后用 `linear_sum_assignment` 求一对一分配（`matching.py:168`）。
4. criterion 按分配索引，把目标和有效性掩码 gather 到预测 slot 的顺序，再与原始预测计算 loss（`losses.py:135-138`）。不把 detached 的预测用于 loss。
5. 平局和训练中被跳过的组都不记录。只有评测报告 `incomplete_groups`、`position_only_groups` 和 `matching_scope`（`evaluate.py:544-557`），数据构建的 manifest 记录被降级的组（[D] §5.3-b）。

[现状] Hungarian 的离散索引不参与反向传播；经索引选中的位置、尺寸和角度损失仍有梯度。请求数由上游确定，所以没有 unmatched 或 background 标签，也不需要检测器的大量候选查询或重复 GT 配对。

5.4 关系标签随置换变换

[现状] 若监督包含对象间关系，匹配后必须同时变换引用。若原监督对象 $$j$$ 的 parent 为 $$r(j)$$，则预测 slot $$i$$ 的 parent 目标是：

$$\tilde r_i=\pi^{-1}\bigl(r(\pi(i))\bigr).$$

floor、wall 等外部支撑以负索引表示，保持不变。若关系矩阵为 $$A$$，则：

$$\tilde A_{ik}=A_{\pi(i),\pi(k)}.$$

[现状] 这一变换已实现为 `permute_relations`（`matching.py:175-193`），但只在测试中调用（`tests/test_supervision.py:54`），因为训练中没有关系或支撑监督目标。简单的组内 Hungarian 解决不了带任意关系耦合的全局图匹配；一个组若不能在其他组不变的情况下独立交换，就保留固定身份。

5.5 Stage 1 对匹配的改动

[计划] 两处改动，公式见 6.7、6.8：
- 条件训练中给定的 slot 固定身份，并移出可交换组；
- 整房对称：对每个候选变换 $$g$$ 重新算一次组内匹配，再在各 $$g$$ 之间取最小。

6. 训练目标

6.1 主训练目标

[现状] 主模型从训练开始就同时监督位置、尺寸和 yaw（`losses.py:236`）：

$$\boxed{
L=\lambda_pL_p+\lambda_sL_s+\lambda_{ac}L_{\mathrm{yaw\mbox{-}cls}}+\lambda_{ar}L_{\mathrm{yaw\mbox{-}reg}}+\lambda_bL_{\mathrm{box}}+\lambda_{\mathrm{col}}L_{\mathrm{collision}}+\lambda_{\mathrm{bd}}L_{\mathrm{boundary}}
}$$

| 权重 | 值 | 依据 |
|---|---|---|
| $$\lambda_p$$ | 1.0 | `config:36` |
| $$\lambda_s$$ | 0.6 | `config:37` |
| $$\lambda_{ac}$$ | 0.08；Isambard 候选组用 `--set loss.yaw_cls=0.5` 改为 0.5，对照组保持 0.08 | `config:38`；`ops/isambard_submit.sh:8-9` |
| $$\lambda_{ar}$$ | 2.0 | `config:39` |
| $$\lambda_b$$ | 0；为 0 时这一项不计算 | `config:40-41`；`losses.py:218` |
| $$\lambda_{\mathrm{col}},\ \lambda_{\mathrm{bd}}$$ | 0 | `config:45-46` |
| $$w_{\mathrm{cell}},\ w_{\mathrm{res}}$$（$$L_p$$ 内部） | 0.5、0.4 | `config:50-51` |

[现状] `LossConfig` 默认 position、size、yaw_cls、yaw_reg 为 1，box、collision、boundary 为 0，格内两项 position_cell、position_residual 以及匹配系数 alpha_position、alpha_size 也为 1（`losses.py:40-58`）。主配置的 $$\lambda_s,\lambda_{ac},\lambda_{ar}$$ 是标定值：规则是让每项在未训练基线下的加权贡献约为 0.2，其中 yaw_reg 封顶为 2.0（`docs/fastfill-v2-loss-calibration-20261006.md`）。

[现状] 逐元素算子：位置项（包括格 residual 和 z）以及 log 尺寸项都用 L1（`config:47-48`；`losses.py:53-54,73-76`）；`smooth_l1` 只是可选项，β 由 `smooth_l1_beta` 给定。yaw residual 固定用 β=1 的 SmoothL1（`losses.py:197`），下文记为 $$\rho_1$$。2026-10-05 版规范写的是 SmoothL1(β=1)，2026-10-06 已改为 L1：在归一化坐标上，误差几乎都小于 1，SmoothL1 实际等价于随房间尺度变化的 L2（标定文档第 44 行）。

[现状] reduction：每一项先求和，再除以该项在全局（所有 rank）的有效实例数；分布式下对计数做 all_reduce，并乘 world size，抵消 DDP 的梯度平均（`losses.py:79-88`）。

6.2 位置与尺寸

[现状] 有效集合（`objective.py:14-25`）的定义：$$\mathcal M_p$$ 是 slot 有效、匹配后位置标签三坐标完整、且至少一个坐标需要学习的实例；$$\mathcal M_s$$ 对尺寸同理。grid_residual 头下（`losses.py:101-114,164-170`）：

$$L_p=\frac{1}{\max(1,|\mathcal M_p|)}\sum_{i\in\mathcal M_p}\Big(\mathbb 1^{xy}_i\Big[w_{\mathrm{cell}}\,\mathrm{CE}\big(\ell_i,k^*_{\pi(i)}\big)+w_{\mathrm{res}}\,\frac12\sum_{q\in\{x,y\}}\big|\varrho_{i,k^*_{\pi(i)},q}-\varrho^*_{\pi(i),q}\big|\Big]+\mathbb 1^{z}_i\,\frac13\big|\hat{\tilde z}_i-\tilde z^*_{\pi(i)}\big|\Big).$$

对尺寸，令

$$e_i(\sigma)=\frac13\sum_{q\in\{w,d,h\}}m^s_{iq}\,\Big|\log\hat s_{iq}-\log\sigma\big(s^*_{\pi(i)}\big)_q\Big|,\qquad L_s=\frac{1}{\max(1,|\mathcal M_s|)}\sum_{i\in\mathcal M_s}e_i(\sigma_i^\star).$$

除 K1 对象外 $$\sigma_i^\star=\mathrm{id}$$；K1 对象的 $$\sigma_i^\star$$ 与 yaw 联合选择（见 6.4）（`losses.py:145-158,171-208`）。

[现状] 说明：
- XY 从不由条件固定，所以 $$\mathbb 1^{xy}_i=1$$。z 被条件固定时 z 项为 0，但实例仍留在分母中（`losses.py:107-113`）。
- 尺寸中被条件固定的坐标取 $$m^s=0$$，仍除以固定的 3（`losses.py:151-155`）。
- 缺少完整标签不同于条件固定：缺标签的实例不进入该项。完整尺寸标签覆盖 train 中 1,351,470 / 1,998,842 个对象（67.6%，[D] §6.6-b）。
- `term_sums["position"]` 是解码位置的 L1/3，没有梯度，只作诊断（`losses.py:160-161`）。
- 回归头下，$$L_p$$ 是三坐标 L1 的掩码和除以 3（`losses.py:154-155`）。

[现状] 格 CE 能保留多峰的位置分布，不会把两个合理位置平均成一个中间点。格内 residual 和 z 的 L1 仍是单点回归。log 比值对相同的相对偏差给相同惩罚：预测为目标尺寸两倍时，各类别受到一致的对数尺度惩罚。它仍受监督样本限制，不自动保证人体工学或资产可检索性。

6.3 yaw：bin classification 与 residual

[现状] 设 bin 数 $$K$$，宽度 $$\Delta=2\pi/K$$，第 $$k$$ 个 bin 的中心为 $$\phi_k=k\Delta$$。对角度 $$\theta$$（`geometry.py:13-23`）：

$$k(\theta)=\left\lfloor\frac{\operatorname{wrap}_{[0,2\pi)}(\theta+\Delta/2)}{\Delta}\right\rfloor\bmod K,\qquad r(\theta)=\frac{\operatorname{wrap}_{[-\pi,\pi)}\big(\theta-\phi_{k(\theta)}\big)}{\Delta/2}.$$

整数索引再对 $$K$$ 取模，防止浮点除法在周期边界舍入为 $$K$$。residual 超出 $$[-1,1]$$ 的容差为 $$32\,\varepsilon_{\mathrm{dtype}}\max(1,K)$$，在容差内裁剪，超出则报错（`geometry.py:20-23`）。

[现状] 令 $$\mathcal M_a$$ 为 slot 有效且 yaw 标签有效的实例（`objective.py:21`）。yaw 有效要求对象直立、来源 front_known、yaw 有限，且不是 Scan2CAD 的 `__SYM_ROTATE_UP_INF`（`legacy_bridge.py:97-110`）。没有角度标签的样本不按默认零角训练。对 6.4 选出的候选角 $$\theta^{\star}_i$$：

$$L_{\mathrm{yaw\mbox{-}cls}}=\frac1{\max(1,|\mathcal M_a|)}\sum_{i\in\mathcal M_a}\mathrm{CE}\big(\hat a_i,\,k(\theta^\star_i)\big),\qquad L_{\mathrm{yaw\mbox{-}reg}}=\frac1{\max(1,|\mathcal M_a|)}\sum_{i\in\mathcal M_a}\rho_1\big(\hat r_{i,k(\theta^\star_i)}-r(\theta^\star_i)\big).$$

训练只取 GT bin 对应的预测 residual（`losses.py:194-197`）。推理取预测 bin（`geometry.py:26-34`）：

$$\hat k_i=\arg\max_k\hat a_{ik},\qquad\hat\theta_i=\operatorname{wrap}_{[-\pi,\pi)}\Big(\phi_{\hat k_i}+\frac{\Delta}{2}\hat r_{i,\hat k_i}\Big).$$

解码不再裁剪，residual 由 tanh 限定在 $$[-1,1]$$ 内。

[现状] 运行配置取 $$K=12$$（`config:9`），每个 bin 宽 30°。例如 97° 的目标对应中心为 90° 的 bin，归一化 residual 为 $$7/15\approx0.467$$。这是 FastFill 的选择，不是 V-DETR 跨数据集的固定设置。

6.4 对称阶、K1 盒等价与语义朝向

[现状] 每个有 yaw 标签的实例取一组等价候选角，对每个候选计算一整套“分类 + residual”代价，再选同一候选的联合最小值；分类项和 residual 项不能各挑各的角度。候选数 $$n_i$$ 的规则如下（`losses.py:185-193`）：

$$n_i=\begin{cases}4,&\text{K1 对象，}w,d\text{ 都未被请求固定}\\ 2,&\text{K1 对象，}w\text{ 或 }d\text{ 被请求固定}\\ \mathrm{order}_i,&\text{其他对象}\end{cases}\qquad\theta^{c}_i=\theta^*_{\pi(i)}+c\,\frac{2\pi}{n_i},\quad c=0,\dots,n_i-1.$$

$$J_{ic}=\lambda_{ac}\,\mathrm{CE}\big(\hat a_i,k(\theta^c_i)\big)+\lambda_{ar}\,\rho_1\big(\hat r_{i,k(\theta^c_i)}-r(\theta^c_i)\big)+\mathbb 1\big[\mathrm{K1}_i,\ i\in\mathcal M_s\big]\,\lambda_s\,e_i(\sigma_c),\qquad\sigma_c=\begin{cases}\mathrm{swap}_{wd},&c\ \text{为奇数}\\ \mathrm{id},&c\ \text{为偶数}\end{cases}$$

$$c^\star_i=\arg\min_c\operatorname{sg}(J_{ic}),\qquad\theta^\star_i=\theta^{c^\star_i}_i,\qquad\sigma^\star_i=\sigma_{c^\star_i}.$$

依据：`losses.py:171-206`。没有有效 yaw 的 K1 对象只在两种尺寸轴序中取较小者（`losses.py:179`）。选择依赖当前的 $$\lambda$$，所以改 yaw_cls 也会改变候选的选择。

[现状] 当前数据中对称阶的来源（`legacy_bridge.py:14-36,97-110`；`multisource_data.py:288-289`；分布见 [D] §2.1、§6.4-b）：

| 对称阶 | 来源 | train 对象数 |
|---|---|---|
| 2（只学 yaw mod π） | front_policy 为 `axis` 时，除 MultiScan 外的全部来源 | 1,872,154（93.7%） |
| 4 | K1 盒等价对象，以及少量 Scan2CAD `__SYM_ROTATE_UP_4` | 126,178（6.3%） |
| 1（保留语义前向） | MultiScan | 510（0.03%） |

[现状] K1 盒等价规则：某些来源中，对象局部轴的配对本身不可靠。这类对象的标签 $$(w,d,h,\theta)$$ 与 $$(d,w,h,\theta+\pi/2)$$ 视为同一个盒子。适用范围按来源和类别固定：InternScenes 的 arkit / 3rscan / mp3d / scannet 子集、InteriorGS，HSSD200 的 chair 和 seat，MultiScan 的 bed（`legacy_bridge.py:23-36`）。train 中这类对象有 126,173 个（6.3%）。评测同时报告盒等价误差和按原写法计算的 `*_plain_convention` 误差（`evaluate.py:470-480,502-557`）。

[现状] 2026-10-05 版规范说“尺寸轴必须固定，不能通过随意交换 $$w,d$$ 隐式制造等价标签”，现改为：尺寸轴原则上固定；只有在上述声明的数据策略范围内，才允许 K1 交换；请求固定了 $$w$$ 或 $$d$$ 时，轴序锁定（`losses.py:174-175,193`）。

[现状] 后果：几何上转 180° 后包围盒相同，不意味着语义朝向相同。但当前数据中 99.97% 的对象没有语义前向监督（93.7% 只学 yaw mod π，6.3% 只学 yaw mod π/2），模型学不到前向。最终朝向由 spread 的背墙翻转决定（4.5）。[M1] 中，“有前向的物体正对墙”的比例：argmax .491，spread .023，GT .035（spread 数字为 a21a67fdb 旧规则下的 [M1] 测量）。

6.5 框重叠监督

[现状] 框监督是独立的增强项。只实现了显式命名的 BEV 有向框凸包 GIoU 算子（`losses.py:120-121,214-225`）：

$$L_{\mathrm{box}}=\frac1{\max(1,|\mathcal M_B|)}\sum_{i\in\mathcal M_B}\Big[1-\operatorname{GIoU}_{\mathrm{BEV}}\big(\hat B_i,B^*_{\pi(i)}\big)\Big].$$

$$\mathcal M_B$$ 只包含位置、尺寸、yaw 标签都有效的实例（`objective.py:25`）。主配置 $$\lambda_b=0$$，这一项不计算（`config:40`；`losses.py:218`）。`configs/structured_bev_box.json` 设为 0.1，但没有跑过对比（[D] §6.6-b）。

[现状] 预测框使用 argmax 解码的 XY 和 argmax yaw。即使启用，box loss 也不会穿过 argmax 给格 logits 或 yaw bin logits 提供普通梯度；只有 argmax 格的 XY residual、argmax bin 的 yaw residual 以及局部尺寸 $$s_x,s_y$$ 能接收梯度，z 和 $$s_z$$ 不在 BEV 算子中（`geometry.py:65-71`；`boxes.py:69-75, 102-123`）；分类 CE 仍然必要。

[设计] BEV 有向框与 3D 有向框是不同的实验选项：前者不能监督竖直位置或高度，后者需要更完整的几何标签和可微实现。GIoU 的包围区域定义、退化框处理和数值精度必须随具体算子写明。对有旋转对称性的框，不直接用依赖任意角点编号的逐角点距离。

6.6 场景约束与训练正则

三种几何比较分别回答不同的问题：

| 项目 | 比较对象 | 回答的问题 |
|---|---|---|
| $$L_{\mathrm{box}}$$ | 预测 vs 匹配的 GT | 预测框是否接近参考框 |
| $$L_{\mathrm{collision}}$$ | 预测 vs 预测 / 固定物体 | 场景中的物体是否发生不允许的穿插 |
| $$L_{\mathrm{boundary}}$$ | 预测 vs 房间 | 对象是否越出房间边界 |

[现状] 已实现的两项正则（`regularizers.py:17-100`）：

$$L_{\mathrm{collision}}=\frac{1}{\max(1,|\mathcal P|)}\sum_{(i,j)\in\mathcal P}\frac{\operatorname{vol}\big(\hat B_i\cap\hat B_j\big)}{\tfrac12\big(\operatorname{vol}\hat B_i+\operatorname{vol}\hat B_j\big)},$$

$$L_{\mathrm{boundary}}=\frac{1}{\max(1,|\mathcal O|)}\sum_{i\in\mathcal O}\frac{1}{4|E|}\sum_{c=1}^{4}\sum_{e\in E}\max\big(0,\,-\delta_e(\hat x_{i,c})\big).$$

- $$\mathcal P$$ 包含两类配对：所有请求对象对，但排除声明的支撑对；以及请求对象与全部固定物体（包括门窗）的配对，但排除对象声明的父物体（`regularizers.py:64-79`）。
- $$\hat x_{i,c}$$ 为 BEV 角点，$$\delta_e$$ 为到房间边 $$e$$ 的有向距离。
- boundary 只支持边界已知的凸多边形，否则报错。

两项权重都是 0（`config:45-46`）。`L_requirements` 没有实现。

[现状] 后果：
- 训练中没有任何成对或整房项。
- 碰撞只靠推理端 spread 缓解；[M3] 中含门窗条件下碰撞率降到约 .02，但关系成立率不变。
- 没有防止“缩小家具换取低碰撞”的专门机制，尺寸只靠 67.6% 有完整标签的对象上的 log-size 监督约束。
- 若启用正则，必须单独消融，并配合功能尺度约束。接触和支撑不能一律当作碰撞惩罚：桌上物体、部分位于桌面下方的椅子等，需要结合高度和接触语义判断。

[现状] 运行时 Validator 目前只在 bbox 层级检查边界、天花板、碰撞（包括与固定物体）和支撑（`validation.py:335`；[D] §8-d、§9-c）。是否加入训练正则只影响学习方式，不改变系统的验收责任；即使某个 loss 为零，也不能宣称场景必然有效。

6.7 Stage 1：整房对称最小化

[计划] 目的是去掉 4.7 中对称条件下 $$\ln|\mathcal H_c|$$ 的不可消除下限，并让同一房间的所有对象在同一个参考副本上受监督。

候选变换取地板外包矩形关于其中心 $$\bar x$$ 的对称：

$$\mathcal D(R)=\begin{cases}\{e,\ r_{180},\ m_x,\ m_y\},&\text{非正方形}\\ \{e,\ r_{90},\ r_{180},\ r_{270},\ m_x,\ m_y,\ t,\ t'\},&\text{正方形}\end{cases}\qquad g(x)=\bar x+A_g(x-\bar x).$$

其中 $$m_x$$ 把 $$x$$ 变为 $$-x$$，$$m_y$$ 把 $$y$$ 变为 $$-y$$，$$t$$ 与 $$t'$$ 是关于两条对角线的反射。$$A_g$$ 是旋转 $$\varphi$$ 时，$$\theta\mapsto\theta+\varphi$$；$$A_g$$ 是关于方向角 $$\psi$$ 的轴的反射时，$$\theta\mapsto2\psi-\theta$$。例如 $$m_x$$ 给出 $$\pi-\theta$$，$$m_y$$ 给出 $$-\theta$$。

[计划] 只保留不改变条件的变换：

$$\mathcal H(c)=\big\{g\in\mathcal D(R):\ g(\text{地板多边形})=\text{地板多边形},\ \ g(\text{fixed\_objects})=\text{fixed\_objects}\big\}.$$

对 fixed_objects，“相等”的意思是：每个固定物体变换后，都有一个同类别的固定物体与之对应，底面中心、尺寸和 yaw 在容差内一致；yaw 保守地按整圈比较。另有三种情况只保留恒等：条件中含 `room.openings`；约束引用世界方向或世界坐标（如 `direction_xy`、`polygon_xy_m`）；6.8 中有给定位姿。

所以门窗会打破对称：含门窗的房间通常只剩恒等，不取最小。三字段投影中，所有矩形房间都有完整的 $$\mathcal D(R)$$。

[计划] 参考布局整体变换：

$$g\cdot y^*_j=\big(g(p^*_{j,xy}),\ z^*_j,\ s^*_j,\ \theta_g(\theta^*_j)\big).$$

局部尺寸、z、对称阶和 K1 标记都不变。交换坐标轴的变换（$$r_{90},r_{270},t,t'$$）同时交换位置有效性掩码的 x、y。反射后盒子作为集合不变，前向 $$+X$$ 映射到反射后的方向，只有局部 $$+Y$$ 的手性翻转；对于带手性的资产（如 L 形沙发），这一点在资产解析时需要注意。

[计划] 对每个房间，在全部候选上取 detached 最小，一个房间只选一个变换，所有对象共同使用：

$$g^\star_c=\arg\min_{g\in\mathcal H(c)}\operatorname{sg}\Big(L_{\mathrm{room}}\big(\hat Y,\ g\cdot Y^*\big)\Big),\qquad\mathcal L=L\Big(\hat Y,\ \big\{g^\star_{c_b}\cdot Y^*_b\big\}_{b=1}^{B}\Big).$$

$$L_{\mathrm{room}}$$ 是第 6.1 节的总目标在单个房间上的值，包括在 $$g\cdot Y^*$$ 上重新计算的组内匹配。$$\mathcal L$$ 是把每个房间的参考换成所选副本之后的普通批损失，reduction 与 6.1 相同。

不能逐对象各取最小：那样每个对象又各自挑副本，等于恢复了原来的问题。整房取最小与 6.4 的 yaw 候选最小是同一形式，只是作用在房间层级。最优解不再是对称的边缘分布：模型可以学到一个确定的规范副本约定，所有对象按同一约定输出。

[计划] 评测配套：
- 关系、贴墙、碰撞等布局指标在 $$\mathcal H(c)$$ 下本来不变；[M1] 中关系、贴墙、碰撞等指标在镜像下与 GT 相同（第 9.4 节）。
- 单参考误差应对同一房间取同一个 $$g$$ 的最小值后再报告，否则会惩罚一个正确的副本。
- 这一项能起作用的房间比例和实际收益都未测量。

6.8 Stage 1：条件（masked）布局训练

[计划] 目的是让一个对象的分布可以以已放置对象为条件，即 $$P(k_j\mid c,\,y_{\mathcal K})$$ 而不只是 $$P(k_j\mid c)$$，并与 4.9 的迭代解码配套。

给定集合的抽样：对每个训练房间抽 $$q\sim\mathcal U(0,q_{\max})$$；在位置、尺寸、yaw 标签都完整的 slot 集合 $$\mathcal V$$ 上独立抽取

$$\mathcal K=\{\,i\in\mathcal V:\ \xi_i<q\,\},\qquad\xi_i\sim\mathcal U(0,1).$$

$$q_{\max}$$ 在配置中固定。$$\mathcal K=\varnothing$$ 的概率为 $$\mathbb E\big[(1-q)^{|\mathcal V|}\big]>0$$，所以无条件的情形仍在训练分布内。

[计划] 给定位姿嵌入：给定 slot 的初始特征加上位姿嵌入，解码器和各头不变：

$$h_i^{(0)}\leftarrow h_i^{(0)}+\mathbb 1[i\in\mathcal K]\;E_\psi(\phi_i),\qquad\phi_i=\big[\tilde p^*_i,\ \log\bar s_i,\ \cos(\bar n_i\bar\theta_i),\ \sin(\bar n_i\bar\theta_i)\big]\in\mathbb R^8.$$

- $$E_\psi$$ 是映射到 128 维的小 MLP。
- $$(\bar s_i,\bar\theta_i,\bar n_i)$$ 是标签等价类的代表元。K1 对象先改写为 $$w\ge d$$ 的写法，即 $$w<d$$ 时取 $$(d,w,h,\theta+\pi/2)$$，并取 $$\bar n_i=2$$；其他对象取 $$\bar n_i=n_i$$。这样同一等价类的不同写法给出同一特征，不把标签约定中的任意性当作输入。
- 推理时（4.9），已放置对象的解码位姿经同一个 $$\phi$$ 输入。

[计划] 损失只计非给定 slot，所有有效集合都去掉 $$\mathcal K$$，分母也只计非给定实例：

$$\mathcal M_\bullet\leftarrow\mathcal M_\bullet\setminus\mathcal K,\qquad\bullet\in\{p,s,a,B\}.$$

[计划] 匹配：给定 slot 固定身份，即对 $$i\in\mathcal K$$ 取 $$\pi(i)=i$$；每个可交换组 $$\mathcal G$$ 的 Hungarian 只在 $$\mathcal G\setminus\mathcal K$$ 上进行。给定 slot 用的是它自己的标签，所以只能取恒等。

[计划] 与整房对称的关系：给定位姿属于条件。$$\mathcal K\neq\varnothing$$ 时取 $$\mathcal H(c)=\{e\}$$，因为给定位姿已经固定了房间的参考系。

[计划] 训练条件：主要用含门窗的完整条件，例如 `minimal_form_p` 取 0.2（见 4.2）。场景正则权重仍为 0。

[计划] 开训前的 go/no-go（依附对象相对锚点的格条件熵是否比无条件先验低约 0.5 nats 以上；不满足则不做本节的条件训练）和验收判据见第 7.8 节。

6.9 之后的计划：整圈 yaw 与偏好优化

[计划] 整圈 yaw：先按来源审计 GT 的 $$+X$$ 是否是一致的前向。方法是统计每个来源中贴墙类家具背靠墙的比例；当前所有对象都标为 front_policy `axis`（[M2] result.plan）。审计通过的来源取对称阶 1，其余仍为 2。这样朝向由模型学习，不再由 spread 的背墙翻转决定。6.4 的候选公式不变，只是 $$n_i=1$$ 时候选只有 $$\theta^*$$ 本身。

[计划] 偏好优化：在 Stage 1 试点有结论后再设计，本文不预先规定其目标形式（OptiScene 的参照见第 11.1 节）。

7. 训练配置与执行

7.1 实际运行

[现状] 到目前为止只训练过一种结构：Qwen3-8B + LoRA 条件编码器、2 层宽 128 的双向 slot 解码器、grid_residual 位置头和 $$K=12$$ 的 yaw 头（第 4、6 节）。进入选模或分析的运行如下：

| 运行 | 数据 | 硬件与全局 batch | 步数 | 角色 |
|---|---|---|---|---|
| `grid-cell05-3gpu`、`grid-cell10-4gpu` | 上一版数据 | yxd-dev，3 / 4 GPU | 最多 3,887 | [现状] 第一轮候选（select1），比较 position_cell 0.5 与 1.0；cell05 的 step 3500 以 score 2.8295 胜出 |
| `main7-cell05-main-20261007b-e5` | main-20261007b | yxd-dev 7 GPU，micro-batch 1 × 累积 14 = 98 | 6,357（5 epoch） | [现状] 当前基线；select2 选中 step 6357（score 2.8554），已上传私有 HF `liantian/fastfill-v2-models` |
| `runs/autorun-yawcls05`、`runs/autorun-yawcls008` | main-20261008a | Isambard-AI，每臂 1 节点 = 4×GH200，micro-batch 1 × 累积 24 = 96 | 6,489（5 epoch） | [现状] yaw_cls 0.5 及其对照 0.08，其余同基线，另加 `max_objects 256`；已训练完毕，2026-10-09 用于含门窗检查（第 9.10 节） |

来源：select1/select2 排名见 `outputs/fastfill_v2/comparison-20261007b/autorun/SUMMARY.md`（2026-10-07）；两臂的提交方式见 `ops/isambard_submit.sh:7-9` 和 `ops/isambard_autorun.sbatch:6-17`。两臂各自的 select2 排名和验收状态本文没有核对。

[现状] Isambard 实测 9.0 s/step（2026-10-08 作业记录，经 [M2] `result.plan` 引用）：6,489 步约 16.2 h，加上验证约 18 h，每臂约 18 节点·时（72 GPU·时）。每个 rank 都受 CPU 限制，一个核占满，GPU 利用率 29–78%。

7.2 运行配置

| 项目 | 取值 | 依据 |
|---|---|---|
| 条件骨干 | Qwen3-8B 本地权重，bf16，主干冻结；LoRA rank 8、α 16、dropout 0，作用于 q/k/v/o | config:3, 20-30 |
| 对象解码器 | TransformerDecoder，2 层，宽 128，4 头，FFN ×4，dropout 0 | config:4-8 |
| 位置头 | grid_residual，16×16 格；z 单独回归 | config:18-19 |
| 尺寸头 | $$s_{\mathrm{ref}}=(1,1,1)$$，log 尺度上限 10 | config:11-16 |
| yaw 头 | $$K=12$$，residual 经 tanh | config:9, 17 |
| 对象上限 | 128；Isambard 两臂运行时加 `--set model.max_objects=256` | config:10；`ops/isambard_autorun.sbatch:17` |
| loss 权重 | position 1.0（内部 cell CE 0.5、格内 residual 0.4、z 1/3）、size 0.6、yaw_cls 0.08（yawcls05 臂 `--set loss.yaw_cls=0.5`）、yaw_reg 2.0；box、collision、boundary 均为 0 | config:36-51；`ops/isambard_submit.sh:8` |
| 逐元素损失 | position 与 size 用 L1；yaw residual 固定为 β=1 的 SmoothL1 | config:47-49；`losses.py:197` |
| 匹配 | 组内 Hungarian，$$\alpha_p=\alpha_s=1$$ | config:42-44 |
| 优化 | lr 1e-4（解码器相同），cosine，warmup 为总步数的 3%，梯度裁剪 1.0，bf16 混合精度，seed 42 | config:55-72；`autorun.py:80` |
| 序列与保存 | max_length 8192；每 500 步保存 checkpoint 并做一次验证 | config:58, 64-65 |
| 增强 | rotate90；打乱顺序并重编号；drop_constraints 0.3、drop_support 0.2、category_only_description 0.5、minimal_form 0.5；不镜像 | config:74-82 |
| 验证过滤 | 剔除带 oob_objects、fixed_collision、overlapping_furniture 标记的验证行 | config:84-88 |

[现状] 全局 batch 和步数由 autorun 按 GPU 数换算（`autorun.py:72-81`）：

$$A=\max\!\left(1,\operatorname{round}\frac{96}{n_{\mathrm{GPU}}}\right),\qquad
T=\left\lceil\frac{E\,N_{\mathrm{train}}}{n_{\mathrm{GPU}}\,A}\right\rceil,\qquad
T_{\mathrm{warm}}=\operatorname{round}(0.03\,T).$$

Isambard 上 $$n_{\mathrm{GPU}}=4$$，于是 $$A=24$$，全局 batch 96；$$N_{\mathrm{train}}=124{,}584$$，$$E=5$$，所以 $$T=6{,}489$$。7 卡基线的 $$A=14$$，全局 batch 98。

[现状] resume 拒绝任何配置变更（`train.py:599`），一次运行内不能途中改 loss 权重或开启 box loss；要改只能新开运行，并在它的配置记录里体现。学习率、LoRA 参数和解码器宽度都没有做过系统搜索；增强参数写死在配置里，也从未在验证集上选择。

7.3 执行流程

[现状] 服务器 autopilot（`autorun.py:1-19`）。各阶段：A 等待正在运行的训练（中断时最多 resume 两次）；B 在每个运行最新 3 个 checkpoint 上做验证集评测；C 按 `autorun.score` 选出第一轮胜者（select1）并导出 RoomGenBench 交接；D 等待并下载新数据；E 训练；F 选模（select2）、上传、测试与导出。在新机器上用 `--start-config` 启动时跳过 A–D 阶段：

```
--start-config（新机器跳过 A–D）
  │
  ▼
E 训练：单一配置，增强固定；每 500 步保存，train.py 记录 selection_metric
  │
  ▼
F 选模：最新 4 个 checkpoint × 全部验证行，三字段投影 + spread 解码 → autorun.score 最小者
  │
  ├── 上传私有 HF（验收之前的备份）
  ├── 测试：三字段与完整条件（spread）；三字段 argmax（模型本身）
  ├── LLM 行：300 条已投影的三字段行原样评测（--projection full 即不再投影），spread 与 argmax 各一次（仅当有 llm-prompt-300/rows.jsonl）
  └── RoomGenBench：5 个基准房间，layout_boxes + --require-placement
  │
  ▼
SUMMARY.md；全部检查通过 → done，否则 done-with-failures
```

依据：B 阶段在最新 3 个 checkpoint 中选（`autorun.py:249`），F 阶段在最新 4 个中选（`autorun.py:529`）；评测命令固定为 `--projection minimal --grid-decode spread`（`autorun.py:290-292`）；上传发生在验收之前（`autorun.py:532`）；测试与 argmax 测试见 `autorun.py:553-559`；RoomGenBench 导出见 `ops/run_checkpoint.sh:22`。

[现状] 两臂结束后，Isambard 上的分析作业（`ops/run_plan.sh`）保存每个前向的头输出，用同一代码分别做 spread 和 argmax 解码，再跑分层统计、房间级配对比较、LLM 对照和输入消融。它规定决策只读验证集，测试、LLM、消融和 RoomGenBench 部分只报告（`ops/run_plan.sh:27-28`）。

7.4 当前选模分数及其缺陷

[现状] 训练器在每次验证时记录 `selection_metric = validation.minimal.geometry_objective`（`train.py:92`），即三字段投影验证行上的加权单参考 loss。autorun 实际选模只用 `autorun.score`（`autorun.py:44-69`），数值越小越好：

$$S=
\frac{e_{\mathrm{size}}}{e_{\mathrm{size}}^{\mathrm{med}}}
+\frac{e_{\mathrm{yaw}}}{e_{\mathrm{yaw}}^{\mathrm{unif}}}
+\frac{e_{\mathrm{pos}}}{e_{\mathrm{pos}}^{\mathrm{ctr}}}
+\bigl|f_{\mathrm{cq}}-f_{\mathrm{cq}}^{*}\bigr|
+\frac{\bigl|\bar d_{\mathrm{wall}}-\bar d_{\mathrm{wall}}^{*}\bigr|}{\bar d_{\mathrm{wall}}^{*}}
+5\,[\,o-o^{*}-0.01\,]_+
+5\,[\,u-u^{*}\,]_+
+10\,\frac{n_{\mathrm{fail}}}{n_{\mathrm{req}}}.$$

其中：

| 记号 | 含义 |
|---|---|
| $$e_{\mathrm{size}},e_{\mathrm{yaw}},e_{\mathrm{pos}}$$ | 单参考的 log-size 误差与 yaw 误差（按对称阶；K1 对象取盒等价最小值），以及底面中心米制误差 |
| 三个分母 | 同一批请求上的平凡基线：类别中位尺寸、均匀随机 yaw 的期望误差（`evaluate._uniform_yaw_error`，`evaluate.py:483-492`）、房间中心 |
| $$f_{\mathrm{cq}},\bar d_{\mathrm{wall}}$$ | 位于中央 1/4 区域的对象比例、到最近墙的平均距离 |
| $$o,u$$ | BEV IoU > 0.3 的对象对比例、出界对象比例 |
| 带 $$*$$ 的量 | 同一批对象的 GT |
| $$n_{\mathrm{fail}}/n_{\mathrm{req}}$$ | 没有布局的请求比例 |

这个分数存在下列缺陷：

| 缺陷 | 证据 | 后果 |
|---|---|---|
| 单参考误差排不出好坏 | GT 的镜像布局位置误差 2.586 m，比房间中心的 2.347 m 还差（300 行，2026-10-08 LLM 评测审计，`tasks/w4fssd3cf.output`）；基线在验证集上的位置误差为 2.896 m，房间中心为 2.472 m，比值大于 1 仍被选中（`SUMMARY.md`） | 位置项主要奖励靠中央的摆法 |
| 没有物体之间的关系项 | 被选中的 step 6357：关系成立率 spread .43，LLM harness .76；argmax .25，与随机打乱的 GT（.23）相当（`scratchpad/arrange/report.md`，2026-10-09） | 选模和训练都没有测量过布置是否成立 |
| 只用三字段投影 | 去掉门窗、固定物体、约束和支撑声明，并跳过非矩形房间：验证集只剩 4,056 间；测试集 8,620 间中跳过 4,165 间（`autorun.py:290-292`；测试报告顶层 `skipped_non_rectangular_rooms`） | 选出的模型从未按门窗处理被评价过 |
| 基线在评测集上拟合 | 不传 `--baseline-fit` 时，类别基线在评测集上逐行留一拟合（`evaluate.py:929-935`），autorun 从不传这个参数 | 违反“统计量只由训练数据确定” |
| 打分对象混入后处理 | 只在 spread 解码上打分（`autorun.py:291-292`） | 模型本身与贪心后处理的贡献分不开 |
| 候选少，没有统计判断 | 只看最新 3 或 4 个 checkpoint；select2 中 step 6357 与 6000 只差 0.036（2.8554 vs 2.8913），没有置信区间 | 排名可能只是噪声 |
| 选模集与 LLM 对照行重叠 | 300 行 LLM 对照取自同一验证样本的前 300 条三字段行（`README.md:790-795, 849`） | 这些行对 FastFill 不是留出集 |
| 不含资产与约束 | 没有资产解析率和约束通过率；验证集本身也没有约束 | 2026-10-05 版 §7.3 要求的选模依据缺失 |

[计划] 在 Stage 1 之前，`autorun.score` 降为记录项，不再用于选模或宣称进展（第 7.7 节）。

7.5 训练诊断

[现状] 单元测试覆盖各 head（含 grid 头）的梯度可达性、yaw 和 grid 编码与 checkpoint 的 round-trip、单个 fixture 上 loss 下降，以及一个玩具双峰样本的拟合（`tests/test_model.py:43, 92`；`tests/test_model_loss_r2.py:101-193`；`tests/test_execution.py:64`）。在真实可信小样本集上过拟合的运行记录没有找到。

[现状] `evaluate --save-head-outputs` 保存每个请求的头输出，可以在 CPU 上用不同解码规则重放（`evaluate.py:23, 399-437, 1069-1072`）。`ops/diag_ablation.py` 在验证行上做输入消融：b 换房间外包尺寸，c 换房型，d 描述降为类别，e 打乱请求顺序并重编号（`ops/diag_ablation.py:4-12`）。

7.6 Stage 0：不训练

[计划] Stage 0 不做新的训练，Isambard 两臂作为 Stage 1 的对照保留。工作项：含门窗评测集（第 9.7 节）、布置指标与门前净空（第 9.4 节）、Qwen3-VL 成对评审（第 9.5 节）、含门窗的 LLM 智能体对照（第 9.6 节）、关系解码（第 4.8 节）。下列做法停止：

1. 用单参考位置误差或 `autorun.score` 选模、宣称进展。
2. 只在三字段投影上评测。train 全量 124,584 行中 96,476 行（77.4%）有门盒（类别包含 door，2026-10-09 计数，见第 2.5 节；与 [M2] 的抽样 4,826/6,230 一致），产品场景也有门。
3. 为了修布置问题继续排队当前结构的超参变体（yaw_cls、max_objects、epoch 数、宽度）。
4. 对有搭档的对象（椅子对书桌、扶手椅对茶几等）使用 spread 的背墙翻转规则；它自己的 docstring 承认会把面向书桌的椅子转成背对书桌（`evaluate.py:121-178`）。
5. 把 argmax 渲染当作“模型”展示，却不说明那是每个对象各自最可能的格子。

7.7 Stage 1 的选模与决策规则

[计划] Stage 1 不再用 `autorun.score` 或单参考误差选模（已商定，第 7.6 节）。[计划]（本修订提出，待确认）具体规则如下：

1. 决策只读验证集。选模集是验证集完整条件（含门窗）的冻结子集，用 sha256 绑定。`test-doors-300` 和整个测试集只报告（第 9.7 节）。
2. 排名的主指标是关系成立率（第 9.4 节）。对候选 checkpoint 做配对、按房屋聚类的 bootstrap（第 9.8 节）。差的 95% CI 包含 0 时记为持平；仍取点估计最好的一个，但不得据此宣称改进。
3. 护栏：碰撞、出界、悬空、正对墙和门前净空不劣于对照（同一解码口径），log-size 误差不超过 0.26。不满足护栏的 checkpoint 不参加排名。
4. 排名用模型自身的解码结果，即 Stage 1 的迭代、锚点优先解码，不经 spread 或关系后处理。后处理结果单独报告。
5. 单参考误差和 `autorun.score` 照常记录，但不参与排名。
6. 平凡基线用训练集拟合（`evaluate --baseline-fit` 指向训练行）。
7. Qwen3-VL 评审只有在通过对照样本校验之后才计入决策（第 9.5 节）。

7.8 试点（pilot）的 go/no-go 与成功判据

[计划] Stage 1 试点保留 Qwen3-8B + LoRA 和现有各 head，只改四处：条件（masked）布局训练（第 6.8 节）；loss 中的整房对称（第 6.7 节）；锚点优先的迭代解码（第 4.9 节）；训练样本以含门窗的完整条件为主（例如 minimal_form_p 0.2）。整圈 yaw 与偏好优化放在试点之后（第 6.9 节）。四处改动在一次运行里同时生效，试点本身不能区分各自的贡献；需要归因时另做消融。

[计划] 训练前的 go/no-go（CPU，数小时）：在训练房间上挖掘“给定锚点后，依附对象所在格”的条件熵，与对称化的“类别 × 房间尺寸”无条件先验比较。该先验在 300 行的 3,416 个对象上，β=10、30 时为 5.255、5.116 nats（β=3 时为 5.453；[M2] `prior_ce.json`）。下降不足约 0.5 nats 时，试点去掉条件（masked）训练这一项。

$$\Delta H=H\bigl(c_{\mathrm{dep}}\mid \mathrm{cat},\mathrm{size}\bigr)-H\bigl(c_{\mathrm{dep}}\mid \mathrm{cat},\mathrm{size},\mathrm{anchor}\bigr)\ \ge\ 0.5\ \mathrm{nats}.$$

[现状] 2026-10-09 已有一份未复核的预检草稿（`scratchpad/stage1-r2/conditional_entropy_5000.json` / `.out`，训练房间 5,000 间，其中 1,000 间留出；脚本不在 main 上）：16 个挖掘出的关系加权下降 1.34 nats；全部 71 个锚点–依附对加权只下降 0.33 nats；脚本给出 go。较早一版（`scratchpad/stage1/`，9 个关系 1.54 nats、76 对 0.33 nats）已被取代。这说明条件信息主要对已挖掘的配对有用。复核之前，它不作为 go 的依据。

[计划] 运行顺序与预算（Isambard，1 节点 = 4×GH200）：

| 步骤 | 内容 | 预算 |
|---|---|---|
| 冒烟 | 1 epoch，1,298 步：确认新输入与新解码端到端可跑，并实测步长 | 约 3.3 h，约 4 节点·时 |
| 正式 | 5 epoch，6,489 步 | 约 18 节点·时 |
| 分析 | 保存头输出、Qwen3-VL 评审、布置指标与统计 | 约 3 节点·时（评审部分为估计） |
| 合计 | | 约 25 节点·时，约 100 GPU·时 |

这些预算按现有结构的 9.0 s/step 估算（[M2]，引用 2026-10-08 的作业记录）。迭代解码会让解码器多跑几轮，实际步长以冒烟实测为准。节点·时记在 brics.u6uv 分配上，没有记录 £ 单价。唯一的现金开销是含门窗的 LLM 智能体运行；上一次 harness 在 300 行上发出 371 次 API 调用，费用没有记录。对照组就是 Isambard 两臂。

[计划] 成功判据：在留出选模的含门窗冻结集上，与拿到完全相同字段的 LLM harness 比较，下列各项必须同时成立。

| 判据 | 阈值 | 当前值（三字段；关系类为 [M1] 293 间，尺寸与延迟为 [M2] 300 行） |
|---|---|---|
| 关系成立率：FastFill − harness | > 0，且 95% CI 下界 > 0 | −0.33 [−0.41, −0.25] |
| 碰撞（> 20%） | 不超过 harness + 0.02 | spread .103，harness .070 |
| 悬空、门前净空 | 不劣于 harness | 悬空 .064 vs .044；门前净空尚无 |
| 正对墙 | 不劣于 harness，且 ≤ 0.03 | spread .023，harness .015 |
| 出界（> 10%） | 不劣于 harness，且 ≤ 0.01 | spread .017，harness .000 |
| Qwen3-VL 成对评审 vs harness | 胜率 > 50%，CI 下界 > 50%（平局记 0.5，两种顺序都评） | 尚无 |
| 评审本身有效 | GT 胜 shuffled ≥ 90%；GT 对 mirrored 在 45–55%；两种顺序结论一致 ≥ 80% | 尚无 |
| 保住尺寸优势 | log-size 误差 ≤ 0.26 | .237 |
| 保住速度 | p50 延迟 ≤ 150 ms | 56 ms |
| 报告口径 | 模型原始输出与后处理结果分开 | — |

当前值来源：[M1]、[M2]。“不劣于”的具体检验方式在评测集冻结前写定。达不到判据时，计划中提到的候选方向是对 slot 集合做联合生成（扩散）或把 LLM 教师蒸馏进 FastFill（[M2]），在试点之后再定，不属于本文的已商定计划。

7.9 文本 SFT 对照

[现状] `text_sft.py` 使用与主模型相同的条件渲染和目标字段，以 Qwen/LoRA 学习输出 JSON，只在 assistant token 上计算 CE：

$$L_{\mathrm{text}}
=
-\sum_t
\log P_\theta(y_t\mid x,y_{<t}).$$

[现状] 它要求每一行标签完整，遇到不完整行直接报错（`text_sft.py:33, 110-112`），所以当前训练集必须先过滤成 82,991/124,584 行的完整子集。它没有增强，没有三字段投影，也没有验证集选模；默认骨干是 Qwen2.5-0.5B-Instruct。只跑过一次 2026-10-05 的冒烟，那次单请求生成的 JSON/schema 解析失败（`README.md:769-773`）。

[计划] Stage 0 和 Stage 1 都不包含文本 SFT。“连续结构化 vs 文本 SFT”这一对照尚未进行，在正式运行之前不列为已交付的对照。

8. WorldEdge 下游对接：RoomGen / RoomGenBench 交接契约

[现状] FastFill v2 当前实际对接的下游是 RoomGen / RoomGenBench。FastFill 给出每个请求对象的目标包络（局部全尺寸、底面中心、yaw）和放置标签；下游按 `asset_key` 生成或检索 mesh，再由共享装配器把 mesh 拟合进目标框（`README.md:25-32`；`docs/fastfill-v2-roomgenbench-interface-20261006.md`）。下游的渲染与评分目前只针对基准自身的 GT 布局，尚未接入 FastFill 布局（下游，未排期；第 8.1 节）。

[现状] 2026-10-05 定稿 §8 说 Harness → Resolver → Reconciliation → Validator → Host → Solver 闭环“作为主系统整体交付”，这一点不成立。仓库只有该闭环的离线参考实现，从未在真实资产库上运行过，也没有持久世界和 Solver（`runtime.py:1-10`；`README.md:8-10, 715-716`；`runtime.py:333` 中 `solver_success` 恒为 None）。本节先写交接契约（8.1–8.7），再写参考闭环缩成的、由下游实现的接口（8.8）。

8.1 实际 pipeline 与职责边界

```
上游请求                                         [上游：任务分析 / 数据行 / RoomGenBench 场景]
  │  --request  ：room_type + room_size_m + furniture_list
  │               （每项 category，可选 id / description / count / support_parent）
  │  --condition：完整条件（地板多边形、房高、fixed_objects 含门窗、constraints、support_parent）
  ▼
FastFill v2 推理  predict.py                                        [FastFill]
  │  Qwen 条件编码 + slot 解码 → grid / size / yaw 头输出
  │  --grid-decode spread（默认）| argmax → layout.json
  ▼
交接导出  direct_layout.export_handoff（predict --export-dir）        [FastFill]
  │  condition.json  layout.json  scene.json  roomgenbench_scene.json
  │  assets.jsonl  diagnostics.json  preview.svg  layout.glb
  ╞═══════════════════════ FastFill 输出边界 ═══════════════════════
  ▼
RoomGen 资产方法  RoomGenBench/methods/*        [下游；当前只读 bench/inputs/assets.jsonl]
  │  按 asset_key 读 description + dimensions + place
  │  → <asset_key>.glb（glTF Y-up，front +Z）+ sidecar（status ok / fallback / failed）
  ▼
装配  python -m fastfill.v2.roomgenbench                   [FastFill 适配器，几何规则来自下游]
  │  交接一致性检查 →（--require-placement）放置检查
  │  → 下游 build_shell / fit_asset / place_matrix → <scene_key>.glb + receipt.json
  ┊  （未接入）
  ▼
渲染与评分  RoomGenBench/bench/render_eval · score_vlm · score_geom · score_aggregate
            [下游；现状只评基准自身的 GT 布局与 184 个资产，不读 FastFill 交接或装配输出]
```

[现状] 下游的渲染与评分从不读取 FastFill 的交接包或 `fastfill.v2.roomgenbench` 的装配结果：`render_eval.py:40` 只读 `results/<M>/assets`（资产按 `bench/inputs/assets.jsonl` 拟合到 GT bbox）；`score_vlm.py:112, 146-147` 读 `bench/inputs/assets.jsonl` 和 `results/<M>/renders/<scene>_persp.png`，后者由 `render.py:132` 从下游 `bench/assemble.py` 写出的 `results/<M>/scenes` 渲染；`score_geom.py:81, 95-117` 的支撑对用 `bench/inputs/scenes/<s>.json` 里的 GT 位置和 `place_id`；`score_aggregate.py:51, 98` 同样只读 `bench/inputs`（路径均在 `RoomGenBench/bench/` 下）。所以 FastFill 布局目前不进入任何下游分数。

[现状] 各环节的负责方与实现位置：

| 环节 | 负责方 | 实现与证据 |
|---|---|---|
| 房型、尺寸、物品清单、可选支撑声明 | 上游。RoomGenBench 五间基准房由 FastFill 的请求构造器转换 | `roomgenbench.py:253-294` |
| 每个请求 ID 一个目标尺寸、底面中心、yaw | FastFill | `predict.py:73-85`；`schema.py:295-311` |
| 坐标转换、place / place_id、墙门窗 shell、`asset_key` 与 registry | FastFill | `direct_layout.py:140-296, 346-373` |
| 目标框代理诊断（只报告，不修复） | FastFill | `direct_layout.py:299-334` |
| 按 `asset_key` 生成、检索或程序化构造 mesh，写 sidecar 状态 | 下游 RoomGen 方法（Infinigen、Holodeck 检索、Shap-E、TRELLIS 等）；输入只读基准自己的 `bench/inputs/assets.jsonl`，不读 FastFill 的 `assets.jsonl` | `RoomGenBench/README.md`；`RoomGenBench/bench/CONTRACT.md:10, 21-60` |
| mesh 拟合（非等比缩放、yaw snap、必要时 tip）、放置矩阵、墙体与门洞几何 | 下游 `assemble.py`；FastFill 适配器只调用并记录 | `RoomGenBench/bench/assemble.py:37-38, 58-60, 83-129, 177-227`；`roomgenbench.py:68-203` |
| 多视角 VQA、SigLIP-2、检索与支撑射线评分 | 下游；只评基准 GT 布局 | `RoomGenBench/bench/score_vlm.py:112`；`score_geom.py:95-117` |
| 实际资产验收、物理稳定、原子提交、Solver | 当前无人执行；receipt 写 `not_checked` / `not_attempted` | `roomgenbench.py:240-241` |

8.2 交接目录

[现状] `predict --export-dir DIR` 在推理后调用 `export_handoff`（`predict.py:102-104`）。所有内容先计算并校验，再创建目录；目录必须不存在，文件以独占方式写入，校验失败不会修改预测（`direct_layout.py:346-373`）。同时给 `--catalog` 会被拒绝，因为交接只导出原始预测（`predict.py:54-55`）。目录内容：

| 文件 | 内容 |
|---|---|
| `condition.json` | 规范条件原样。`--request` 时是 `request_to_condition` 的三字段投影，加上请求中声明的 `support_parent`（`direct_layout.py:68-109`） |
| `layout.json` | 解码后的预测布局原样：`{id, target_size_local_m, bottom_center_m, yaw_rad}` |
| `scene.json` | `fastfill.bbox-scene.v1`，`geometry_kind = predicted_local_obb_envelope`。每个对象附 category、description 和 bbox（几何中心、局部尺寸、yaw、8 个角点）（`direct_layout.py:120-137`） |
| `roomgenbench_scene.json` | RoomGenBench SceneSpec，见 8.3–8.5 |
| `assets.jsonl` | 每个 `asset_key` 一行：type、description、首个实例的 dimensions、place、support_status、`placement_eligible`（place_id 非空）、scenes、n_instances、`asset_key_kind = downstream_generation_key_only`（`direct_layout.py:353-360`） |
| `diagnostics.json` | `bbox_diagnostics`：boundary、floor_lower_bound、ceiling、obb_overlap；资产检索为 `not_attempted`，mesh 与 physics 为 `not_checked`，support 为 `unknown`，commit 为 `not_attempted`（`direct_layout.py:299-334`） |
| `preview.svg`、`layout.glb` | 不含资产的彩色框可视化 |

[现状] 下游方法的输入契约只读 `RoomGenBench/bench/inputs/assets.jsonl`（`CONTRACT.md:10, 21`）；把 FastFill 的 `assets.jsonl` 交给生成方法的环节不存在（下游，未排期）。两份文件的字段也不同：FastFill 的 `scenes` 写的是 `fastfill_…` scene_key，`dimensions` 取首个实例的预测尺寸，另外多出 `support_status`、`placement_eligible`、`asset_key_kind`。5 间基准房的 type 与 description 原样传入（`roomgenbench.py:262-265`），asset_key 与基准相同（同一公式，`direct_layout.py:30-33`、`RoomGenBench/bench/prepare_inputs.py:23-25`），可以用 `--assets-dir RoomGenBench/results/<M>/assets` 复用已生成的资产；这时生成输入的尺寸是基准 GT，不是 FastFill 的预测尺寸，装配时再非等比拟合到预测框。

[现状] 交接目录本身不记录 checkpoint 和解码方式（spread 或 argmax）：`layout.json` 只允许四个键（`schema.py:295-302`），`export_handoff` 也不写运行元数据。autorun 在 `runs/roomgenbench/<tag>/checkpoint.txt` 记录 checkpoint 路径与 `implementation_sha256`，并据此决定是否复用导出（`autorun.py:318-329`）；`ops/run_checkpoint.sh:18` 另存请求的 sha256。解码方式在任何地方都没有显式记录（`run_checkpoint.sh:21` 不传 `--grid-decode`，用的是 predict 默认的 spread）。

8.3 对象与房间字段映射

[现状] `layout_to_roomgenbench` 把 FastFill 的规范约定转成 SAGE / RoomGenBench 约定（`direct_layout.py:235-296`）。FastFill 的对象局部 $$+X$$ 是规范前向；RoomGenBench 的局部前向是 $$+Y$$，dimensions 依次是局部 X、Y、Z 方向的全长，SAGE 的 length 即前后深度（`RoomGenBench/bench/CONTRACT.md:29`；`RoomGenBench/bench/assemble.py:9-10`）。

| FastFill | RoomGenBench SceneSpec | 规则 |
|---|---|---|
| `id` | `id` | 不变 |
| `category` | `type` | 不变 |
| `description` | `description` | 不变 |
| (category, description) | `asset_key` | `slug(category)[:24] + "_" + sha1(description)[:8]`；slug 为小写、非字母数字字符换成 `_`、去掉首尾 `_` 后取前 24 个字符；sha1 对 UTF-8 编码的 description 计算，取前 8 位十六进制（`direct_layout.py:30-33`）。与 RoomGenBench `prepare_inputs.asset_key` 相同（`RoomGenBench/bench/prepare_inputs.py:23-25`）。type 与 description 都相同的实例共用一个 key；`asset_key_kind` 标为 `downstream_generation_key_only`，不是真实资产 ID |
| `bottom_center_m` | `position {x, y, z}` | 米制底面中心，不变 |
| `yaw_rad` $$\theta$$ | `rotation {x: 0, y: 0, z: φ}` | φ 单位为度，见下式 |
| `target_size_local_m` $$(s_x,s_y,s_z)$$ | `dimensions {width, length, height}` | $$(s_y,s_x,s_z)$$ |
| 声明或推断的支撑 | `place_id`、`place` | 见 8.4 |
| 请求的 `support_surface_id` | `support_surface_id` | 请求对象或 hard `on` 约束的 `surface_id` 声明时才有，否则为 null（`validation.py:68-75`） |
| — | `support_status` | `declared` / `inferred` / `unknown` |

$$\mathrm{width}=s_y,\quad \mathrm{length}=s_x,\quad \mathrm{height}=s_z,
\qquad
\phi=\frac{180}{\pi}\operatorname{wrap}_{[-\pi,\pi)}\!\left(\theta-\frac{\pi}{2}\right).$$

按第 2.2 节的直立框定义 $$B$$（第一个尺寸分量沿局部 X），有：

$$B\bigl(p,(s_x,s_y,s_z),\theta\bigr)=B\bigl(p,(s_y,s_x,s_z),\theta-\pi/2\bigr).$$

世界角点不变；FastFill 的 $$+X$$ 前向映到 SAGE 的 $$+Y$$ 前向，朝向角仍为 $$\theta$$。`tests/test_direct_layout.py:243` 对多个 yaw 检查角点集合相同。`roomgenbench.ground_truth_layout` 是精确逆变换：size = [length, width, height]，yaw = radians(rotation.z) + π/2 后 wrap；基准里的 rotation.x、rotation.y 倾斜在逆变换中丢弃（`roomgenbench.py:297-320`）。

[现状] 这一转换保持的是“模型当作前向的轴”，不证明它就是语义正面（`direct_layout.py:239-240`）。训练数据中 93.7% 的对象 yaw 对称阶为 2，只学 yaw mod π（第 2.1、6.4 节）。导出的正反方向要么是模型对两个峰的选择，要么由 spread 的背墙翻转决定（第 4.5 节）。下游 GLB 约定 front 为 +Z，装配时经 `GLTF_TO_SAGE` 映到 SAGE 的 +Y（`RoomGenBench/bench/assemble.py:37-38`）。因此 yaw 差 π 时，生成的资产会整体背对应朝的方向。[计划]（之后）在来源能确认真实前向的对象上改学整圈 yaw，先审计来源（第 6.9 节）。

[现状] 房间与场景级字段（`direct_layout.py:255-296`）：

| 字段 | 规则 |
|---|---|
| `scene_key` | `"fastfill_" + sha256(scene.json 的规范 JSON)[:16]` |
| `room_type`、`geometry_only` | 房型原样；`geometry_only = true` |
| `fixed_objects`、`constraints` | 条件原样深拷贝 |
| `room.dimensions` | width、length 为地板多边形外包框的 X、Y 范围；height 为 `height_m`，未知时为 null |
| `room.position`、`ceiling_height` | 外包框最小角 (x, y) 与 `floor_z_m`；`ceiling_height = height_m` |
| `room.walls / doors / windows` | 见 8.5 |
| reference_extent 请求 | 另带 `room_size_semantics`、`room_interpretation`，以及 room 内的 `boundary_known`、`floor_known`、`boundary_quality`（`direct_layout.py:291-295`） |

8.4 place、place_id 与 support_status

[现状] 先取声明：`effective_support_requests` 把请求的 `support_parent` 与硬 `on` 约束合并（软 `on` 只是偏好，不算），冲突、悬空引用和环都在这里报错（`validation.py:38-83`）。有声明的对象 `support_status = declared`。没有声明的对象，按下列顺序取第一条成立的规则推断，状态为 `inferred`；都不成立时为 `unknown`（`direct_layout.py:192-225, 253-271`）：

$$\mathrm{place\_id}_i=
\begin{cases}
\mathrm{floor}, & z_{\mathrm{floor}}\ \text{已知且}\ |z_i-z_{\mathrm{floor}}|\le 0.02\ \mathrm{m}\\[2pt]
\arg\max_{j\in\mathcal S_i}(z_j+h_j), & \mathcal S_i\neq\varnothing\\[2pt]
\mathrm{wall}, & \text{边界与地面已知},\ z_i-z_{\mathrm{floor}}>0.15\ \mathrm{m},\ \text{某条 footprint 边距边界}\le 0.1\ \mathrm{m},\ \text{中心下方无框}\\[2pt]
\varnothing, & \text{其他}
\end{cases}$$

$$\mathcal S_i=\left\{j\neq i:\ z_j<z_i,\ \bigl|z_i-(z_j+h_j)\bigr|\le 0.03\ \mathrm{m},\ c^{xy}_i\in F_j,\ j\ \text{的支撑链不回到}\ i\right\}.$$

其中 $$F_j$$ 是旋转后的 footprint，$$c^{xy}_i$$ 是对象 $$i$$ 的底面中心。$$j$$ 只取请求对象的预测框，固定物体不作为推断出的支撑（`direct_layout.py:269` 只把 `scene["objects"]` 当作候选框）；声明的父对象可以是固定物体（`validation.py:47-48`），此时 place 为 on_object。阈值依次是 `FLOOR_CONTACT_M`、`ON_OBJECT_CONTACT_M`、`RAISED_M`（`direct_layout.py:25-27`）和 `validation.WALL_GAP_M`（`validation.py:19`）。`floor_known = false` 时不做地面推断（`direct_layout.py:262`）；`boundary_known = false` 时不做墙面推断（`direct_layout.py:270`）。声明的父对象和已经推断出的父对象都记入同一张父表，所以推断不会闭合支撑环（`direct_layout.py:264-271`）。

[现状] `place_of` 把 place_id 映到 RoomGenBench 的词表：None → `unknown`，`floor` / `wall` 不变，其他对象 ID → `on_object`（`direct_layout.py:228-232`）。下游自己的输入准备把一切非 floor / wall 的值都当成 `on_object`（`RoomGenBench/bench/prepare_inputs.py:28-29`）；Holodeck 检索方法的阈值表只有 floor、wall、on_object 三个键（`RoomGenBench/methods/holodeck_retrieval/retrieve.py:49, 96`）。所以 `unknown` 必须在交给下游方法之前被发现或拒绝（8.6）。

[现状] 推断出的放置只是几何候选，不是经过验证的接触（`direct_layout.py:1-6`）。Validator 中，放在物体上的支撑因为没有 `support_surfaces` 一律是 `support_surface_unknown`（`validation.py:232-236`）。

[现状] 三字段请求不带支撑声明，此时 place 全部来自推断或为 unknown。RoomGenBench 基准请求则为每个对象声明场景中的 `place_id`：`benchmark_request` 把对象 ID 改写为按场景顺序的 `obj_%04d`，把 place_id 同样改写后作为 `support_parent`，`room_size_m = [width, length, height]`（`roomgenbench.py:253-265`）。五间基准房共 313 个对象（`README.md:221`）。五间房的 `room.position` 都是 (0, 0, 0)，与 `request_to_condition` 把外包最小角放在原点的约定一致。因此 ops 的 5 间房检查中 `--require-placement` 按构造必然通过（place 全部是 declared，`README.md:227`），检验不到预测框的放置推断。

8.5 房间 shell：墙、门、窗与固定物

[现状] `_shell` 由条件构造 shell（`direct_layout.py:140-174`）：
- 墙：地板多边形的每条非退化边生成一面墙 `wall_%02d`，起止点 z 取地面高度；墙高取 `height_m`，未知时用 2.7 m；墙厚 0.1 m（`direct_layout.py:23-24, 147-152`）。
- 门、窗：来自 `room.fixed_objects` 中 category（小写）含 `door` 或 `window` 的对象，所以 doorframe 也算门。每个开口挂到点到线段距离最近的墙上，没有距离上限；`position_on_wall` 为沿墙参数 $$t\in[0,1]$$；width 是 footprint 在墙方向上的投影长度；height 取 $$s_z$$；窗另带 `sill_height` = 底面高度 − 地面高度（`direct_layout.py:153-173`）。

[现状] 装配时，FastFill 适配器逐面墙调用下游 `build_shell`。墙的外法向由地板多边形的有向面积决定，不用下游按外包框中心判断的做法，这样非凸房间的墙不会被挤进室内（`roomgenbench.py:135-163`；`tests/test_leak_guards_fix20261006.py:66`）。门在 shell 上开洞；下游 `build_shell` 只读 doors（`RoomGenBench/bench/assemble.py:93-108`），所以窗和其他固定物都画成 bbox 代理（`roomgenbench.py:166-203`）。房高未知时，`roomgenbench_scene.json` 里 `dimensions.height` 为 null、墙高为 2.7 m；装配显示用 `--display-height-m`（默认 3 m），并在 receipt 中记为 `height_source = display_reference`（`roomgenbench.py:177-178, 197-203`）。

[现状] 只有 `--condition` 输入（完整条件）带 fixed_objects，因此只有它导出的 shell 有门窗。`--request` 不假设任何门窗（`direct_layout.py:73`）。RoomGenBench 的五个场景各有 1 扇门（`RoomGenBench/bench/inputs/scenes/*.json` 的 `room.doors`），但 `benchmark_request` 只传房型、尺寸和清单，门被丢掉。所以当前基准交接只有墙、没有门，墙面物体可能挡住门口（`README.md:227`）。

[现状] 门窗作为完整条件输入对布局的影响见第 2.5 节与第 9.10 节（[M3]）：碰撞减少，关系成立率没有显著变化。

8.6 装配、`--require-placement` 与参考比对

[现状] `python -m fastfill.v2.roomgenbench --handoff DIR --output-dir OUT --method M [--assets-dir A] [--require-placement]`（`roomgenbench.py:402-440`）：
1. 一致性：用 `condition.json` 和 `layout.json` 重算 `layout_to_roomgenbench`，必须与 `roomgenbench_scene.json` 逐字段相等，否则拒绝（`roomgenbench.py:46-51`；`tests/test_roomgenbench.py:246`）。
2. 放置：加 `--require-placement` 时，任一对象 `place_id` 为空就在写任何输出之前报错（`roomgenbench.py:52-53`；`tests/test_roomgenbench.py:179`）。
3. 方法：`layout_boxes` 用目标框代理；其他 snake_case 方法名表示已生成的 mesh，必须给 `--assets-dir`；`sage_gt` 被拒绝（`roomgenbench.py:214-217`）。
4. 资产：sidecar 的 `asset_key`、`method` 必须匹配，status 只能是 ok / fallback / failed。缺失、无效或失败时保留该实例并画 placeholder 框。GLB 经 `GLTF_TO_SAGE` 转换后，用下游 `fit_asset` 拟合到本实例自己的 dimensions；拟合后尺寸与目标的相对偏差超过 1e-5 时，状态改记 `fit_mismatch`（`roomgenbench.py:68-105`）。registry 用首个实例的尺寸作为生成输入，装配时每个实例仍拟合到自己的框（`direct_layout.py:354`；`roomgenbench.py:95`）。
5. receipt：分母是全部请求对象。只有每个对象都为 ok 时 `generated_mesh_success` 才为真。receipt 记录原生尺寸与拟合后尺寸、GLB 与 sidecar 的 hash、`fit_policy`（生成方法为 `roomgenbench_anisotropic_yaw_snap_tip`，`layout_boxes` 为 `target_bbox_proxy`，`roomgenbench.py:239`）；`original_asset_geometry_acceptance` 与 `support_verification` 为 `not_checked`，`validator`、`physics`、`host_commit` 为 `not_attempted`（`roomgenbench.py:206-250`）。输出目录必须不存在，且不能与输入或下游仓库重叠（`roomgenbench.py:219-226`）。
6. 退出码：`layout_boxes`，或生成方法全部成功时为 0，否则为 2（`roomgenbench.py:440`）。

[现状] 自动运维只跑框代理：`ops/run_checkpoint.sh:12-22` 从 RoomGenBench 场景重建请求，对五间房依次执行 `predict --request --export-dir --device cpu` 和 `roomgenbench --method layout_boxes --require-placement`。生成 mesh 的装配没有进入自动流程。五个 SAGE 基准房在训练中留出，只出现在 test（`io.py:16-18, 52`；`cohort.py:108`）。

[现状] `--reference-check HANDOFF --scene S --output R` 用交接里的条件，对 FastFill 布局和基准真值各跑一次 `validate_scene`，再按 place 汇总误差：可交换组内按底面中心距离做 Hungarian，主误差为盒等价的 (size, yaw)，基线为房间中心和均匀随机 yaw（`roomgenbench.py:323-399`）。

[现状] 两种策略必须分开报告：
- RoomGenBench 把 mesh 非等比缩放到目标框（`RoomGenBench/bench/assemble.py:177`）；FastFill 的资产闭环契约不缩放、不删除资产（`runtime.py:3-6`）。拟合成功不等于实际资产验收。
- 下游 `score_geom` 只在子物体和支撑物都生成成功时评估支撑对（`RoomGenBench/bench/score_geom.py:13-14`），它的 n 不是全部请求支撑对。FastFill 侧目前没有另报“全部请求支撑对、缺失资产数、可评数”的汇总。

8.7 已实现的验证层级

[现状] `validate_scene` 对每项检查报告 pass / violation / unknown。`ok` 当且仅当没有未通过的 hard 检查，hard 的 unknown 也算未通过（`validation.py:335-403`，`:400`）。默认 tolerance 为 1e-4 m。

| 层级 | 实现 | 检查内容 | 状态 |
|---|---|---|---|
| Schema | `schema.validate_layout`；`validate_scene` 的 schema 项 | 键集合、有限数、正尺寸、yaw ∈ [−π, π)、输出 ID 与请求 ID 一一对应 | [现状] |
| 目标几何（bbox） | `validate_scene(stage="target")` | floor_lower_bound、多边形 boundary、ceiling；fixed_size / size_bounds；支撑：floor_support、wall_support（footprint 边距边界 ≤ 0.1 m）、物体支撑（需 support_surfaces，否则 unknown）；collision 与 fixed_collision（直立 OBB，高度区间重叠且 footprint 交面积 > tol²，没有按类别豁免）；约束 faces_direction、faces、near、clearance、keepout、on、against_wall、between | [现状] |
| 开口与通行 | `validation.py:393-394` | 只有 `room.openings` 非空时才报 `openings_unchecked`（unknown），而当前数据不用 openings。门窗作为 fixed_objects 只参与 fixed_collision。没有门前净空检查，也没有通行检查 | [现状] 缺失；[计划] Stage 0 门前净空 |
| 交接代理诊断 | `direct_layout.bbox_diagnostics` | boundary、floor、ceiling、obb_overlap；只报告 | [现状] |
| 实际资产（bbox） | `validate_scene(stage="actual")` + `runtime.py` | 用资产实际尺寸构造直立 OBB；能力、语义前向、`attributes_unverified` | [现状] 仅离线参考，未在真实资产库运行 |
| mesh / physics / solver | `validation.py:398-399` | 恒为 unknown；列入 `required_levels` 就会阻止提交 | [现状] 未接入任何检查器 |
| 装配 receipt | `roomgenbench.assemble_handoff` | 资产状态、拟合偏差；validator、physics、host 为 not_attempted | [现状] |

[现状] 三字段请求没有支撑声明，所以每个对象都是 hard 的 `support_unknown`，严格 `ok` 不可能成立（`validation.py:213-217`）。评测因此分开报告 pass / violation / unknown 计数，不只报 ok。

8.8 离线资产闭环：缩为下游实现的接口

[现状] `runtime.py` 实现了旧闭环的参考版本：
- 资产：`Asset` 记录 ref、category、actual_size_local_m、`canonical_transform`（原始坐标 → 米制、Z-up、bbox 底面中心）、semantic_front_local（必须已是规范 +X）、capabilities、经过验证的 support_surfaces、provenance（`runtime.py:45-87`）。
- 硬过滤：带 typed attributes 的请求一律拒绝；类别做忽略大小写的精确匹配；还检查必需能力、semantic_front_required、固定尺寸、尺寸范围和 log 尺寸容差（`runtime.py:90-117`）。
- `CatalogResolver` 在合格资产中按下式排序，相等时按 ref 排（`runtime.py:141-144`）：

$$d_{\mathrm{size}}(a,i)=\sum_{q\in\{x,y,z\}}\left|\log\frac{s^{\mathrm{actual}}_{a,q}}{\hat s_{i,q}}\right|.$$

- `reconcile` 构造世界变换（`runtime.py:147-150`）：

$$T^{\mathrm{world}}_i=T(\hat p_i)\,R_z(\hat\theta_i)\,T^{\mathrm{canon}}_{a(i)}.$$

  只有父对象恰好有一个匹配的已验证支撑面时，子对象高度才改为支撑面高度；否则保持 unknown（`runtime.py:153-186`）。
- `run_pipeline` 的顺序：schema → 目标几何校验 → 逐对象解析资产 → reconcile → 实际几何校验 → 有界资产重选（默认 2 次）→ 可选的 `BoundedTranslationRepair`（默认 0 次；每次最多平移 0.25 m，连带支撑后代一起移动，且只允许改位姿）→ 最终校验通过才调用 `host.commit`（`runtime.py:189-249, 338-448`）。
- `AtomicMemoryHost` 检查幂等键与世界版本，要求对象 ID 唯一，全部提交或全部不提交（`runtime.py:264-297`）。
- 入口：`serve.py`（`--commit-in-memory` 只写内存）、`predict --catalog`、`evaluate --catalog`。仓库里没有资产库文件，autorun 和 ops 从不传 `--catalog`。没有“重新调用 FastFill”的重规划。v1 的 WorldEdge resolver 契约 `resolve(FloorObjectSpec)` 不能代替 v2 的接口（`README.md:717-718`）。

[现状] 定稿中由 FastFill “整体交付”的环节，现在都是下游拥有的接口。FastFill 只提供协议与参考实现：

| 接口 | FastFill 提供 | 下游需要实现 |
|---|---|---|
| Resolver | `Resolver` 协议 `resolve(request, prediction, excluded_refs) → Asset \| None`；`CatalogResolver` 参考实现 | 语义检索或生成，以及 Asset 元数据（规范变换、前向、能力、支撑面） |
| Reconciliation | `reconcile` 参考实现 | 真实资产的 pivot、轴序、单位，转成 `canonical_transform` |
| Validator mesh / physics | `required_levels` 接口，结果恒为 unknown | mesh、physics 检查器与证据导入 |
| Host | `Host` 协议 `commit(objects, idempotency_key, expected_world_version)`；`AtomicMemoryHost` | 持久世界的原子适配器 |
| Solver | 无 | 全部 |

[现状] Harness、持久 Host、Solver、mesh / physics 检查和重规划都不在已约定的 Stage 0 / Stage 1 计划内。它们留作下游集成，本规范不为它们设定交付日期。

8.9 与下游相关的计划

[计划] Stage 0 不重新训练，也不改交接字段。与下游相关的有两点，其余 Stage 0 工作项（布置指标与门前净空、Qwen3-VL 评审、含门窗的 LLM 对照）见第 9.4–9.6 节：
- 含门窗评测集（第 9.7 节）由 evaluate 直接读数据行评测；若另经 `predict --condition --export-dir` 导出，shell 会带门窗开口（第 8.5 节）。RoomGenBench 基准请求仍不含门，修正它不在已商定的计划内。
- 关系解码（第 4.8 节）是可选的推理端后处理，其布局同样经 `export_handoff` 导出。交接目录本身不记录 checkpoint 与解码方式（第 8.2 节）。[计划]（本修订提出，待确认）要求另行记录解码方式，手工导出还要记录 checkpoint；autorun 已记录 checkpoint，解码方式尚未记录。

[计划] Stage 1 改的是模型、训练和解码（条件（masked）布局训练、loss 中的整房对称、锚点优先的迭代解码、以含门窗的完整条件为主训练），不涉及交接字段。

[计划] 之后：在前向可信的来源上改学整圈 yaw，使导出的 $$+X$$ 在这些对象上成为语义前向（第 6.9 节）；偏好优化。

9. 评测与消融

9.1 评测原则

[现状] 在被评测的投影内部，所有请求都计入分母，包括推理失败和超出容量的请求（`evaluate.py:846-866` 的 `stage_denominator`）。三字段投影跳过非矩形或边界未知的房间（`evaluate.py:68-74, 1017`），所以报告必须同时写出跳过的房间数。例如基线测试报告的三字段部分为 4,455 间，跳过 4,165 间；`projections.full` 为 8,620 间，跳过 0 间。

[现状] 模型原始输出（`--grid-decode argmax`）和 spread 后处理分开报告（`autorun.py:545-559`）。决策只读验证集，测试只报告（`ops/run_plan.sh:27-28`）。

单份 GT 只是众多合理布局之一。GT 的镜像同样合理，位置误差却是 2.586 m，比把所有对象放在房间中心还差（第 7.4 节）。所以参考相似度只能作为补充层，不能单独判断布局好坏。

[计划] 主表改用含门窗的完整条件；三字段结果降为附加，并写明跳过的房间数。任何后处理（spread、关系解码）的结果都标为“FastFill + 后处理”，与模型本身分开。

9.2 三层评测

| 层 | 回答的问题 | 内容 | 状态 |
|---|---|---|---|
| 参考相似度与校验计数 | 单个对象与参考是否接近；bbox 级是否出界、碰撞 | 底面中心误差、log-size、按对称阶的 yaw、BEV IoU、盒等价与 plain 两种口径；平凡基线；Validator 的 pass / violation / unknown 计数 | [现状] |
| 布置指标 | 对象之间的关系是否成立；门口是否通畅 | 关系成立率、贴墙、正对墙、悬空、碰撞、出界、通行、门前净空 | [计划]；2026-10-09 已有离线测量（[M1]、[M3]），未合入 main |
| VLM 成对评审 | 人看来哪个房间更合理 | Qwen3-VL 盲评，两种顺序都评，用对照样本校验评审 | [计划] |

9.3 第一层：参考相似度与校验计数

[现状] 参考指标为 `bottom_center_error_m`、`log_size_error`、`yaw_error_rad`、`bev_iou` 以及两个 `*_plain_convention`（`evaluate.py:52-53`）。对应关系采用固定身份加可交换组内 Hungarian，评测代价用解码后位置的 L1（`evaluate.py:495-557`）。yaw 误差按对象记录的对称阶 $$n$$ 计算（`evaluate.py:461-463`）：

$$e_{\mathrm{yaw}}=\left|\operatorname{wrap}_{[-P/2,\,P/2)}\bigl(\hat\theta-\theta^*\bigr)\right|,\qquad P=\frac{2\pi}{n}.$$

[现状] K1 允许交换轴序的对象报告盒等价误差：在 $$k=0..3$$ 四个候选（yaw 为 $$\theta^*+k\pi/2$$；$$k$$ 为奇数时尺寸交换为 $$(s_y,s_x,s_z)$$，偶数时保持 $$(s_x,s_y,s_z)$$）中，取 log-size 误差（三轴平均）与整圈 yaw 误差之和最小的一个（`evaluate.py:466-480`），同时保留按书写轴序计算的 `*_plain_convention`（`evaluate.py:470-480`；`README.md:87`）。

[现状] 平凡基线包括房间中心、类别均值位置、类别中位尺寸和均匀随机 yaw。均匀随机 yaw 的期望误差为 $$\pi/(2n)$$，交换对象按闭式计算（`evaluate.py:483-492, 619-690`）。默认在评测集上逐行留一拟合，不符合第 3.4 节；[计划]（本修订提出，待确认）改为一律用 `--baseline-fit` 指向训练行（第 7.7 节第 6 条）。

[现状] 分布统计包括中央 1/4 占比、平均离墙距离、BEV 重叠率和出界比例（`evaluate.py:781-797`）。Validator 按检查码计数边界、天花板、碰撞（含与固定物）和支撑，并标注 bbox 层级；mesh、physics、solver 三层始终为 unknown（`evaluate.py:800-812`；`validation.py:398-399`）。

[现状] 这一层的已知局限：

| 局限 | 证据 |
|---|---|
| 位置误差不能排序布局 | 四种方法在 300 行上为 2.62–2.75 m，都差于房间中心 2.347 m；GT 镜像为 2.586 m |
| yaw 看不到朝向 | 训练集 93.7% 的对象为对称阶 2，只按 mod π 计分 |
| 完整率不是独立指标 | `requested_ids_exactly_once` 和 `positive_valid_size` 与 `schema_success` 同时赋值（`evaluate.py:700, 995`） |
| 尺寸/能力满足率为空 | 数据中没有尺寸范围或能力要求；资产阶段从未运行，没有资产库 |
| 没有通行指标 | 门窗只作为固定盒子参与碰撞检查；`openings_unchecked` 只在 `room.openings` 非空时出现，而当前数据没有这个字段（`validation.py:394`） |

[现状] 下游接口层的替代检查：用 `direct_layout.layout_to_roomgenbench` 转换后，若每个对象都能得到支撑位置，即通过 `--require-placement`。300 行上的通过数为：harness 296/300，FastFill spread 268/300，GT 262/300，prompt 255/293，FastFill argmax 25/300（2026-10-08 LLM 评测审计，`tasks/w4fssd3cf.output`）。这只是接口检查，不代表物理或任务成功。

9.4 第二层：布置指标

[计划] 把关系成立率、贴墙、朝向、悬空、碰撞、出界、通行和门前净空做成仓库内的评测工具；实现在未合并的工作树中进行（`ops/arrange_metrics.py`，不在 main 上）。[现状] 2026-10-09 已用离线脚本 `scratchpad/arrange/arrange.py` 完成一次测量（[M1]），定义如下。

1. 挖掘。 取训练 GT 中每 20 行一行（6,230 间），按关键词规则把 652 个以上的原始类别名映射到 33 类。对每个有序类别对 $$(A,B)$$，每个 $$A$$ 取最近的 $$B^\dagger$$，生成模板：间隙 $$g\le g_{\max}=\max(q_{0.8},0.3\,\mathrm m)$$，$$B^\dagger$$ 位于允许的方位，GT 中至少 60% 面向 $$A$$ 时还要求面向。一个模板保留的条件是：在 GT 中至少对 40% 的 $$A$$ 成立，且成立频率至少是同一批房间打乱后的 2 倍。625 个候选对中保留 32 个。墙类（GT 中至少 60% 背靠墙）和有前向的类别也从同一批训练行确定。
2. 关系成立率。 对每个适用的 $$A$$ 实例：

$$h(A)=\mathbb 1\!\left[g(A,B^\dagger)\le g_{\max}\right]\cdot
\mathbb 1\!\left[\mathrm{side}(B^\dagger\mid A)\in\mathcal S_{AB}\right]\cdot
\mathbb 1\!\left[\text{面向要求满足}\right],$$

房间得分是适用实例的均值，方法得分是房间均值。
3. 其他逐房间指标： 搭档在挖掘距离内、靠近时方位与朝向正确、搭档间隙中位数、墙类背面距墙 ≤ 0.1 m 和 ≤ 0.3 m、贴墙、yaw 偏离墙轴超过 5°、有前向的对象正对墙、悬空于房间中部、碰撞超过 20%、出界超过 10%、最大可走区域占比、可走区域连通性、对象可达性。
4. 门前净空（[计划]，只评有门的房间；以下为实现草案，阈值在评测集冻结前确定）： 门前、房间内侧取 0.8 m 深、与门同宽的区域。单个地面家具的占地覆盖该区域超过设定比例时，记为该门被堵；同时检查每个门的区域是否连到最大可走连通区（5 cm 网格，人半径 0.3 m）。构造对照样本（打乱、镜像等）时门不移动。
5. 校验。 每个对象用请求和 GT 确定它是否属于地面家具及其类别，各方法用各自的框打分。离线测量中，镜像 GT 在关系、贴墙、正对墙、碰撞、出界等 13 项指标上与 GT 的差为 0.000；yaw 偏离墙轴（−0.002 [−0.006, 0.001]）和可达性（+0.001 [0.000, 0.003]）两项有栅格化带来的微小差异（[M1] `report.md` 配对差表）；打乱、随机和房间中心的得分都低；yaw 翻转使墙类背靠墙降到 .02、正对墙升到 .98。

[计划] 关系模板只从训练集挖掘，冻结为带 sha256 的 `relations.json`；阈值写在配置中，评测时不再调整。

局限：类别映射是关键词规则；阈值是固定选择；这 300 间验证房间的 GT 本身并不完美，例如 TV 柜→沙发在其中只成立 .21，在训练中成立 .53（[M2] 原文误称“test GT”）。部分关系样本很少（n = 13–24）。293 间公共房间中只有 123 间至少含一个关系，浴室和大多数厨房没有，因此还需要贴墙、朝向和门前净空等指标补充。

9.5 第三层：Qwen3-VL 成对评审

[现状] `ops/vlm_judge.py` 提供渲染、对照样本、RoomGen 交接和对照页，本身不调用任何模型（`ops/vlm_judge.py:1-21`）。每个（房间, 方法）渲染为一张 1536×1024 的面板：1024 px 俯视平面图加两张对角视图（numpy z-buffer），图上不出现方法名。对照样本（代码中称 anchors）有 room-centre、random、shuffled、yaw-flipped、mirrored、identical 六种（`ops/vlm_judge.py:51`）。main 上的渲染还不画门窗。

[计划]
1. 渲染画出门窗。在 Isambard 上运行 Qwen3-VL 评审；候选权重 Qwen3-VL-235B-A22B-Instruct-FP8 已于 2026-10-09 下载到项目目录（`tasks/bka0objtj.output`），评审尚未运行。
2. 成对盲评：同一房间的两张面板按 A/B 和 B/A 两种顺序各评一次，平局记 0.5。评分细则包括关系与朝向、门前净空和通行。胜率为

$$\hat w=\frac{1}{2R}\sum_{r=1}^{R}\sum_{o\in\{AB,\,BA\}}v_{r,o},\qquad v_{r,o}\in\{0,\tfrac12,1\},$$

置信区间按房屋聚类做 bootstrap（第 9.8 节）。
3. 评审只有在通过对照样本校验后才计入结论：GT 胜 shuffled ≥ 90%；GT 对 mirrored 在 45–55%；identical 应判平局；两种顺序结论一致 ≥ 80%。另外请用户在约 30 个房间上亲自判断，检查评审是否与人一致。
4. Stage 0/1 只把评审用于评测和决策，不进入训练。用评审结果构造偏好数据属于后续的偏好优化阶段。

9.6 LLM 智能体对照

[现状] 未微调语言模型的对照，实际做法是让外部 LLM 智能体坐在 FastFill 的位置上回答同样的请求。模型为 gpt-6.1-sol（reasoning effort medium），共四种模式：

| 模式 | 实现 | 做法 |
|---|---|---|
| prompt | `llm_baseline.py` | 问一次 |
| harness | `llm_baseline.py` | 检查出界、重叠和悬空，带着具体问题最多修 2 轮 |
| structured | `llm_structured.py` | OptiScene 式固定指令加一条训练示例，问一次 |
| structured-harness | `llm_structured.py` | structured 加只用请求和回答本身的检查与修复 |

[现状] 协议（`README.md:786-821`）：
- 行：autorun 验证样本中的前 300 条三字段行（`llm_baseline.py:189`；`llm_structured.py:242`）。
- 输入：与 FastFill 相同的字段，即房型、房间尺寸与地板多边形、家具清单（id、category、description），不给尺寸先验、统计量或位置。
- 输出与打分：输出同样的四个字段，用同一个 `evaluate --predictions` 打分。
- 运行次数：每种模式只跑一次，付费模式不重跑。
- 留出关系：这 300 行属于 FastFill 的选模队列，对 FastFill 不是留出集。
- 本地数据：本地 `outputs/fastfill_v2/comparison-20261007b/` 只有 prompt 与 harness 的结果；四种模式的尺寸误差都在 0.39–0.41（`README.md:845-849`）。

[现状] 三字段投影没有门窗，所以这组对照不涉及门窗。

[计划] 含门窗的对照：
- 输入：给 LLM 智能体与 FastFill 完整条件完全相同的字段，即房型、地板多边形、房高、`room.fixed_objects`（门、窗和其他固定盒子）、家具清单（含 `support_parent`）。
- 输出：与 FastFill 相同。
- harness 检查：只能使用请求和回答本身，因此可以检查门口区域。
- 运行：在 `llm300-full`（配对，非留出）和 `test-doors-300`（留出，只报告）上各付费运行一次。

9.7 冻结评测集

| 集合 | 来源与筛选 | 用途 | 状态 |
|---|---|---|---|
| 选模样本 | 验证集以 `random.Random(0)` 打乱后取前 3,000 行（B 阶段），或全部验证行（F 阶段），三字段投影 | 当前选模 | [现状] `autorun.py:236-247` |
| LLM 300 行（三字段） | 上述样本中的前 300 条三字段行 | 当前 LLM 对照 | [现状] `README.md:790-795` |
| `llm300-full` | 同样 300 个场景在 main-20261008a 验证集中的完整行：220 行含门对象，229 行的请求与目标和旧三字段行一致（两者数据版本不同） | 含门窗的配对对照；非留出 | [计划]；2026-10-09 草案 `outputs/fastfill_v2/eval-sets-20261009/llm300-full/manifest.json` |
| `test-doors-300` | 测试集中至少 1 个门区、至少 3 个地面家具、矩形、不超过 256 个对象；按来源 × 对象数分层，seed 0；8,620 行中 2,076 行合格，抽 300 行，共 366 扇门 | 留出选模，只报告；最终结论用 | [计划]；草案 `…/test-doors-300/manifest.json`（`held_out_from_selection: true`） |
| Stage 1 选模集 | 验证集完整条件（含门窗）的冻结子集 | Stage 1 决策 | [计划] |
| RoomGenBench 5 间 | SAGE-10k 基准房间，已留出到测试集 | 接口检查 | [现状] manifest `roomgenbench_holdout_groups`；`ops/run_checkpoint.sh:22` |

[现状] 两份草案的 manifest 已记录数据、rows 与 builder 的 sha256（`data_sha256`、`rows_sha256`、`builder_sha256`）。[计划] 构建代码尚未合入 main（同一目录 `outputs/fastfill_v2/eval-sets-20261009/` 下还有一份 `relations.json` 副本），冻结前需要复核并把构建代码合入 main。

9.8 统计

[现状]
- 主判定：`ops/compare.py` 对两个评测输出逐房间配对求差，房间级 bootstrap 默认 10,000 次，差的 95% CI 不含 0 时判为显著（`ops/compare.py:14-19, 145`）。
- 稳健性：精确符号检验加 Holm 校正。
- 其他：`ops/stratify.py:324` 做房间级比值 bootstrap；离线布置指标按房间 bootstrap 2,000 次。
- 种子：所有运行都只有一个种子（seed 42，config:59）。

[现状] 当前只按房间重采样，没有按房屋聚类。测试集 8,620 间房来自 3,319 个房屋，其中 1,300 个房屋有不止一间房，最多的有 252 间（[D] §9-i），房间级区间会偏窄。数据行已带 `provenance.house_id`（`legacy_bridge.py:274`）。

[计划]（本修订提出，待确认）最终结论改用按房屋聚类的配对 bootstrap。设 $$\mathcal U^{(b)}$$ 为第 $$b$$ 次有放回抽到的房屋集合，$$m_r$$ 为房间 $$r$$ 上的指标：

$$\hat\Delta^{(b)}=
\frac{\sum_{u\in\mathcal U^{(b)}}\sum_{r\in u}\bigl(m^{A}_r-m^{B}_r\bigr)}
{\sum_{u\in\mathcal U^{(b)}}|u|}.$$

[计划] 试点本身是单种子，以 Isambard 两臂为对照。[计划]（本修订提出，待确认）通过判据后、宣称结论前，再用第二个种子复现一次。

9.9 对照与消融：已做与计划

| 对照 / 消融 | 状态 | 结果或缺口 |
|---|---|---|
| spread vs argmax 解码 | [现状] 已做（[M1]） | 关系成立率 spread 比 argmax 高 .18 [.11, .25]；碰撞 .10 vs .41；正对墙 .02 vs .49；argmax 与打乱的 GT 相差 +.03 [−.05, +.11] |
| 三字段 vs 含门窗输入（Isambard 两臂） | [现状] 2026-10-09 已做（[M3]） | 关系不变，碰撞约减半，见第 9.10 节 |
| yaw_cls 0.5 vs 0.08 | [现状] 已训练 | 三字段关系成立率 .390 vs .387（[M3]），没有可见差别；其他 yaw 指标的分析作业结果本文未核对 |
| 位置头：regression vs grid_residual；position_cell 0.04 / 0.2 / 0.5 / 1.0 | [现状] 已有配置（`configs/qwen3_8b_main_4gpu_*`）；`README.md:606-613` 记有 position_cell 的比较 | 只按 `autorun.score` 选出 0.5；没有用布置指标评价 |
| 平凡基线 | [现状] 已做 | 默认在评测集上留一拟合；改为训练集拟合列入第 7.7 节的选模规则草案 |
| LLM 智能体（三字段） | [现状] 已做 | 见第 9.6、9.10 节 |
| 输入消融 b–e | [现状] 代码已有，在 Isambard 分析作业中运行 | 结果本文未核对；顺序打乱与 ID 重命名合在一个变体里，没有分开 |
| 固定对应 vs 组内 Hungarian | [现状] 只有冒烟（`outputs/fastfill_v2/hungarian-smoke-*`） | 没有正式对比 |
| 连续结构化 vs 文本 SFT | [现状] 只有冒烟，JSON 解析失败 | 没有做 |
| 匹配代价加 yaw / box | [现状] 未实现 | `match_batch` 只接受位置与尺寸系数 |
| log-size vs raw-size；yaw bin vs sin/cos | [现状] 未实现 | 代码中没有这两个选项 |
| box loss 与场景正则 | [现状] 未运行 | 正式配置中权重全为 0；`configs/structured_bev_box.json` 从未运行 |
| 类别尺寸先验 + 布局模型；ATISS / DiffuScene | [现状] 未实现 | 没有适配和运行 |
| 资产闭环、修复预算、目标框 vs 实际资产 | [现状] 未运行 | 没有资产库；`solver_success` 恒为 None |
| FastFill + 挖掘先验的关系解码 | [计划] Stage 0 | 定义见第 4.8 节；与模型原始输出、spread 分开报告 |
| 含门窗的 LLM 智能体 | [计划] Stage 0 | 见第 9.6 节 |
| Stage 1 试点 vs Isambard 两臂 | [计划] Stage 1 | 四处改动捆绑在一次运行中；要归因各自的贡献，需要另做消融 |
| 整圈 yaw（仅前向可信的来源）；偏好优化 | [计划] 之后 | 先按来源审计 GT 的 +X 是否真是前向 |

9.10 当前结果

[现状] 截至 2026-10-09 的实测结果。除非另注，数字都来自基线 step 6357，在 300 行（公共 293 间）三字段验证行上，与 LLM 对照相同。FastFill 的输出由提交 a21a67fdb 的代码解码；该版本的 spread 对靠墙对象取 45° 以内最可能的 yaw bin，而当前 main 只做 π 翻转。头输出 logits 没有保存，无法用当前代码重新解码。

1. 布置明显差于 LLM harness。 关系成立率：GT .714，LLM harness .764，LLM prompt .730，FastFill spread .433，FastFill argmax .253，打乱的 GT .225。spread 减 harness 为 −.331 [−.413, −.246]；argmax 减打乱的 GT 为 +.028 [−.050, +.107]；harness 减 GT 为 +.051 [−.021, +.123]（[M1]；argmax 减打乱 GT 由 `per_room.jsonl` 重算，见 [M2] `result.plan`；[M2] `result.metrics` 给出的区间为 [−.046, +.109]）。部分具体关系（GT / spread / harness）：

| 关系 | GT | spread | harness |
|---|---|---|---|
| 床头柜→床 | .79 | .35 | .97 |
| 餐椅→餐桌 | .95 | .37 | 1.0 |
| 书桌椅→书桌 | .60 | .00 | .87 |
| 沙发→茶几 | .84 | .53 | .74 |
| TV 柜→沙发 | .21 | .00 | .71 |
| 床→衣柜 | .60 | .56 | .10 |

在列出的关系中，只有床→衣柜是 FastFill 明显好于 LLM 的一项。spread 还在以下几项上差于 harness：搭档间隙中位数 0.77 vs 0.44 m；悬空 .064 vs .044；碰撞 .103 vs .070（+.033 [.004, .065]）；yaw 偏离墙轴 .064 vs .002，这一项来自旧解码规则。

2. 真实的优势。 尺寸：log-size 误差 .237，harness .388，prompt .412，类别中位尺寸基线 .385。速度：每间房延迟中位数 56 ms，p90 77 ms；harness 中位数 16.8 s，p90 75 s。墙类对象背面距墙 ≤ 0.1 m 的比例 .69，harness .62（+.08 [.03, .13]），LLM 常留 10–30 cm 的空隙。FastFill 300 行都返回了布局，prompt 缺 7 行。来源：[M2]（尺寸 `autorun/SUMMARY.md`；FastFill 延迟 `outcomes.jsonl` 的 `fastfill_latency_ms`，中位 56.4 ms；harness 延迟 `predictions.jsonl` 的 `latency_s`）；墙类贴墙为 [M1]。
3. 单参考指标分不出好坏。 四种方法的位置误差为 2.62–2.75 m，都差于房间中心 2.347 m。
4. 测试集（三字段 4,455 间，`SUMMARY.md` 与 `README.md:839-844`）。 log-size 误差 .296（类别中位尺寸 .394）；yaw .673（argmax）/ .700（spread），均匀随机为 .783；位置 2.643 / 2.663 m，房间中心为 2.264 m。延迟 p50 51.6 ms，p95 133 ms（测试报告 `latency_ms`）。
5. 只加门窗不能修复布置。 2026-10-09 对 Isambard 两臂在同一批 300 个比较场景的 main-20261008a 完整行上检查（[M3]；公共 299 间，关系成立率在其中 125 间有关系的房间上计算）：

| 指标 | GT | yaw_cls 0.5 三字段 | yaw_cls 0.5 含门窗 | yaw_cls 0.08 三字段 | yaw_cls 0.08 含门窗 |
|---|---|---|---|---|---|
| 关系成立率 | .718 | .390 | .387 | .387 | .375 |
| 碰撞 > 20% | .080 | .040 | .021 | .058 | .024 |
| 正对墙 | .035 | .094 | .071 | .065 | .066 |

含门窗减三字段的关系成立率差：yaw_cls 0.5 为 −.003 [−.051, .046]，yaw_cls 0.08 为 −.012 [−.063, .041]（n = 125，按房间 bootstrap），没有显著变化；碰撞约减半。两臂在三字段上的关系成立率（约 .39）与基线 spread 的 .43 相近，后者是另一批数据和解码代码下的测量，两者不做显著性比较。[M3] 的输出文件没有记录解码方式；`evaluate` 的默认是 spread。

6. 尚未测量的部分。 场景协调性只有第二层的离线测量，Qwen3-VL 评审还没有运行。含门窗的 LLM 对照、门前净空、按房屋聚类的区间都还没有。第 7.8 节的判据规定在含门窗冻结集上与含门窗的 LLM harness 比较，该评测尚未进行，所以尚无结论。按三字段 300 行的现有数字（[M1]、[M2]），尺寸（.237 ≤ .26）和延迟（56 ms ≤ 150 ms）两项已满足；关系成立率、碰撞、悬空、出界和正对墙（.023，劣于 harness .015）几项不满足；门前净空和评审两项还没有数据。Stage 0 和 Stage 1 都是计划，不是结果。

10. 实现交付与验收

10.1 已交付模块

[现状] main `2e86188` 的 `fastfill/v2/` 有 38 个 Python 模块（不含 `__init__.py`）和 `ops/` 运维脚本。按职责分组：

| 模块组 | 文件 | 已实现 | 未交付或限制 |
|---|---|---|---|
| 数据与协议 | `schema.py`、`geometry.py`、`io.py`、`adapters.py`、`audit.py`、`data.py`、`legacy_build.py`、`legacy_bridge.py`、`legacy_evidence.py`、`legacy_verify.py`、`qualified_data.py`、`review_data.py`、`multisource_data.py`、`multisource_verify.py`、`minimal_data.py`、`cohort.py`、`size_range.py` | 严格协议与有限数校验；yaw 与网格编解码；v3.2 选定语料迁移与独立流式校验；合格化修订；三字段派生；RoomGenBench 基准房留出 | 资产不重叠子集、真实尺寸子集、门窗已知/未知标记、数据视觉复核都没有 |
| 模型与训练 | `model.py`、`batch.py`、`matching.py`、`losses.py`、`objective.py`、`boxes.py`、`regularizers.py`、`train.py`、`configs/` | Qwen + LoRA 条件编码；双向 slot 解码；grid_residual 位置头、log 尺寸头、12 bin yaw + residual；认证组内 Hungarian；L1 位置与尺寸；K1 盒等价；增强；resume | box、碰撞、边界正则在正式配置中权重为 0；没有 requirements 项 |
| 推理与交接 | `evaluate.py`（`predict_layout`、`spread_grid_xy`）、`predict.py`、`direct_layout.py`、`bbox_visualization.py`、`roomgenbench.py` | spread / argmax 解码；`--save-head-outputs`；交接导出；RoomGenBench 装配适配器；基准请求构造；参考比对 | 交接不记录 checkpoint 与解码方式；基准请求不含门 |
| 离线资产闭环参考 | `runtime.py`、`validation.py`、`serve.py` | 见 8.7、8.8 | 未在真实资产库运行；mesh / physics / Solver / 持久 Host 都没有 |
| 评测与对照 | `evaluate.py`、`autorun.py`（`score`）、`llm_baseline.py`、`llm_structured.py`、`text_sft.py`、`ops/compare.py`、`ops/stratify.py`、`ops/llm_compare.py`、`ops/diag_ablation.py`、`ops/crosscheck.py`、`ops/report.py`、`ops/vlm_judge.py` | 全请求分母；参考误差与平凡基线；坍缩诊断；LLM 智能体四种模式；房间级配对 bootstrap；渲染与对照样本 | 文本 SFT 只跑过冒烟；平凡基线默认在评测集上逐行留一拟合；布局关系指标未入 main；VLM 评审未运行 |
| 运维与复现 | `autorun.py`、`ops/isambard_*.sbatch`、`ops/isambard_submit.sh`、`ops/run_plan.sh`、`ops/run_checkpoint.sh`、`ops/rebuild_and_publish.sh`、`verify_bundle.py` | 自动选模与发布；私有 HF 模型仓库（`README.md:841`）；Isambard 上 yaw_cls 0.5 与 0.08 两臂（`ops/isambard_submit.sh:8-9`） | 单种子；置信区间按房间而非房屋聚类 |

10.2 已覆盖的正确性检查

[现状] `fastfill/v2/tests/` 有 83 个测试文件、717 个 `def test_` 函数（2e86188 上的静态计数；参数化展开后用例更多）。最近一次有记录的全量运行是 2026-10-07：`pytest fastfill/tests fastfill/v2/tests` 1236 passed、114 subtests passed（`README.md:214`）。本次修订没有在 2e86188 上重跑。

| 模块 | 已有测试（`tests/…`） | 未覆盖 |
|---|---|---|
| 几何协议 | yaw round-trip（`test_geometry_core.py:18`）；局部全尺寸角点使用各自高度（`:28`）；拒绝零、NaN、重复与未知字段（`:38`）；归一化只依赖条件（`:51`）；bf16 下 yaw bin 中心保持 float32（`test_leak_guards_fix20261006.py:98`） | — |
| 数据与匹配 | 同类不同角色不可交换（`test_supervision.py:42`）；子物体支撑引用阻止交换（`test_matching.py:76`）；约束引用阻止换角色（`test_matching.py:85`）；关系随置换变换（`test_supervision.py:50`）；grid 分配是所有置换中收费最小者（`test_round4_matching.py:72`）；基准房不得进入训练（`test_leak_guards_fix20261006.py:38`）；留一基线排除整行标签（`:112`） | 平局与被跳过的匹配组没有计数 |
| 角度与梯度 | yaw residual 只取 GT bin（`test_supervision.py:87`）；梯度到达条件骨干（`test_model.py:43`）；可交换交换下 detached 分配与可微 loss（`test_supervision.py:30`）；双向对象交互（`test_model.py:112`） | 真实可信小样本集上的过拟合运行 |
| loss 与批处理 | NaN 目标在运算前被过滤（`test_supervision.py:59`）；固定 z 保留除数 3（`:76`）；padding 不改变有效预测（`test_model.py:58`）；两进程目标窗口（`test_active_objective_distributed.py:98`）；box 梯度与小框（`test_supervision.py:111`）；IoU 与 GIoU 区分及全局 DDP reduction（`:153`） | — |
| 交接与装配 | +X → +Y 转换保持世界框（`test_direct_layout.py:243`）；不编造房高与支撑（`:275`）；无 catalog 导出并保留原始预测（`:306`）；无效交接在建目录前被拒（`:329`）；registry key 约定（`:406`）；硬 / 软 on 的映射（`test_roomgenbench.py:78`）；角点、固定物代理与 unknown（`:101`）；失败实例不消失、不计成功（`:153`）；未知放置在 require 策略下被拒（`:179`）；极薄 mesh 的拟合偏差可见（`:232`）；篡改被拒、输出不可覆盖（`:246`） | 真实生成资产上的装配；全请求支撑对汇总 |
| 离线运行与提交 | target / actual 尺寸分开（`test_runtime.py:31`）；pivot 变换顺序（`:43`）；重选有界（`:64`）；预算为 0 时失败且无部分提交（`:73`）；已验证支撑面带动子物体（`:92`）；修复不能缩小对象或改 ID（`:132`）；平移修复连带子物体（`:154`）；幂等提交与世界版本（`:169`）；重复世界 ID 原子拒绝（`:205`）；朝向约束需要实际语义前向（`:185`）；未转换的 openings 阻止提交（`test_validation.py:82`） | 真实资产库与持久 Host |

[现状] 定稿要求的四个核心例子都有对应测试：两把可交换椅子（`test_supervision.py:30`）、同类不同约束的固定角色（`:42`）、带支撑的桌面物体（`test_runtime.py:92`）、需要语义前向的朝向约束（`test_runtime.py:185`）。

10.3 未交付项

[现状] 以下内容在 2e86188 上没有交付。“状态”列说明它在已约定计划中的位置：

| 项目 | 现状 | 状态 |
|---|---|---|
| 含门窗评测集；选模与主表使用含门窗的完整条件 | 选模、主表和 LLM 对照都用三字段投影 | [计划] Stage 0 |
| 布置指标（含关系成立率）与门前净空 | 只有 2026-10-09 分析会话的离线脚本（[M1]），不在 main | [计划] Stage 0 |
| Qwen3-VL 成对评审 | 渲染器与对照样本已有，未调用任何模型 | [计划] Stage 0 |
| 含门窗的 LLM 智能体对照 | 只有三字段对照 | [计划] Stage 0 |
| 关系解码（挖掘先验） | 不在 main | [计划] Stage 0 |
| 条件（masked）布局训练、loss 中的整房对称、锚点优先的迭代解码、以完整条件为主训练 | 不在 main | [计划] Stage 1 |
| 整圈 yaw；偏好优化 | 不在 main | [计划] 之后 |
| 平凡基线用训练集拟合 | `evaluate.py:1064` 默认在评测集上逐行留一；autorun 不传 `--baseline-fit` | [计划]（本修订提出，待确认），第 7.7 节 |
| 文本 SFT 正式对照；结构化与文本 SFT 消融 | 只跑过一次冒烟，生成 JSON 解析失败（`README.md:769-773`） | 未排期，不在 Stage 0/1 内 |
| 资产不重叠子集、真实尺寸子集；门窗已知/未知标记；倾斜对象显式分流；描述中尺寸的过滤 | 没有 | 未排期，不在 Stage 0/1 内 |
| 多种子；按房屋聚类的置信区间 | 单种子 42；按房间 bootstrap | [计划]（本修订提出，待确认），第 9.8 节 |
| 真实资产库评测、mesh / physics / Solver、持久 Host、重规划 | 只有离线参考实现 | 下游，不在计划内 |

10.4 验收门槛

[计划] Stage 1 的训练前 go/no-go、试点成功判据（关系成立率对含门窗 LLM harness 的 95% CI 下界大于 0；碰撞、悬空、正对墙、出界、门前净空不劣于 harness；Qwen3-VL 评审通过对照样本校验后胜率 CI 下界高于 50%；log-size 误差 ≤ 0.26，p50 延迟 ≤ 150 ms；原始输出与后处理分开报告）及各项当前值见第 7.8 节；停止的做法见第 7.6 节；选模规则草案见第 7.7 节。截至 2026-10-09，判据规定的含门窗冻结集评测尚未进行，所以尚无结论；按三字段 300 行的现有数字，尺寸与延迟两项已满足，关系成立率、碰撞、悬空、出界、正对墙不满足，门前净空与评审尚无数据（第 7.8、9.10 节）。

10.5 定稿边界

本修订以 main `2e86188` 的代码和 2026-10-09 的测量为准。[计划] 各项都没有实现或验证，结果出来后按证据修改相应章节。本文不把任何未测量的收益写成结论。

11. 文献与 V-DETR 实现依据

11.1 场景生成文献与后续阶段的动机

[原文] OptiScene 以检索到的资产及其 bbox 为条件，输出布局的位置和旋转（§3.1、§3.3）。训练先做带推理（布局摘要）的 SFT，再做两轮 DPO（§3.4）：
- 第一轮：以专家精选的人类共识布局为正例，以 SFT 模型生成的次优变体为负例；
- 第二轮：向正例注入碰撞、越界等空间违规，构造更难的负例。

[原文] 官方 SFT 代码是 causal LM 加 token 训练（固定版本 `409f131` 的 `scripts/sft_train.py:35-41, 58-65`），没有连续几何头。OptiScene 证明了语言条件布局生成与偏好优化这条路线可行，但没有解决“实际尺寸未知时联合预测尺寸”。

[现状] 仓库中的 `llm_structured.py` 是用外部 API 模型、仿照 OptiScene 格式写的提示基线，不是 OptiScene 训练出的模型。

[计划]（之后）偏好优化以 OptiScene 第二轮“注入几何违规的负例”为直接参照。如何在 FastFill 的连续头上构造偏好目标，还没有设计。

[原文] ATISS 把场景当作无序集合，自回归地逐个生成对象（§3.1–3.2，图 2）：
- 每一步用 transformer 编码地板布局和已放置对象，预测下一个对象；
- 属性按类别 → 位置 → 朝向 → 尺寸的顺序依次条件化，连续属性用混合 logistic 分布建模（式 8–11）；
- 同一模型支持场景补全、固定用户指定对象后的部分重排、物体建议和失败检测与修正（图 1、§4）。

集合表示不等于 Hungarian。

[现状] FastFill 一次前向同时给出所有 slot 的结果。每个 slot 有自己的 256 格分布，loss 是逐对象对单份参考的交叉熵，没有任何 slot 看到其他对象已经定下的位置（`model.py:143-226`；`losses.py:101-114`）。[M1] 测得 FastFill argmax 的关系成立率（.25）与把 GT 位置在房内对象间打乱后的 .23 没有显著差别（第 4.7、9.10 节）。

[计划] Stage 1 的条件（masked）布局训练与锚点优先迭代解码（第 6.8、4.9 节）借鉴 ATISS 的“以已放置对象为条件”：训练时把随机一部分对象的位姿写进它们的 slot token，预测其余对象；推理时分几轮，先锚点，再把已放置对象回填。Qwen 只跑一次，解码器多跑几轮。它不是逐对象自回归采样，效果要等试点验证。

[原文] DiffuScene 用扩散联合建模对象集合的位置、尺寸、朝向（sin / cos）、类别和形状编码。训练目标是噪声预测 MSE，另加对象之间的 IoU 正则（§3.1，式 9–10）。这个正则属于预测对预测的碰撞项，不是对 GT 的 box loss；它的训练 objective 也不应统称为 DETR set loss。

[原文] CasLayout 分四个阶段生成：类别与数量、尺寸特征、关系 latent、位姿。关系从场景几何中提取；§3.8 的重排接受给定类型和尺寸，也允许部分尺寸未知。固定对象清单或缺少人工关系图，并不使它不适用（官方 arXiv HTML `2604.27361v1`，§3.3、§3.8、附录 A）。

[计划] Stage 0 的关系解码同样从训练几何中挖掘成对模板，但它是规则化的后处理，不是 CasLayout 学到的关系 latent，结果单独报告。

[设计] 本方案选单个条件对象解码器，是工程范围和待验证假设的选择，不是文献结论。

11.2 Hungarian、matching cost 与 training loss

[原文 / 代码] DETR 式二分匹配先求一一对应，再对匹配结果计算训练损失。Hungarian 是分配算法，不是一种几何 loss（DETR matcher、3DETR criterion）。

V-DETR 把这一机制用于 3D 检测：从点云预测带语义和几何的候选框。FastFill 的请求对象是已知的，所以采用受约束的组内分配，不照搬检测任务的全部匹配项。

[代码] V-DETR 的 matcher 在 `@torch.no_grad()` 下构造 detached 代价（`V-DETR/criterion.py:119-120`），用 `linear_sum_assignment` 求解（`:215`）。其中 GIoU 匹配项为 $$-\operatorname{GIoU}$$（`:189`），对应的 training loss 为 $$1-\operatorname{GIoU}$$（`:458-459`）。有效项和分配数量固定时两者只差常数，负号本身不是错误。

[代码] 重叠张量都叫 `gious`，但实际算子取决于配置（`V-DETR/main.py:110`；`criterion.py:619-635`）：
- 默认 `iou_type = giou` 时走 `generalized_box3d_iou`；
- 选 `diou` 或 `iou` 时走旋转框可微 DIoU 或 IoU。

focal 分类匹配使用正、负类别成本之差（`criterion.py:138-139`）：

$$C_{\mathrm{cls}}(p)
=
\alpha(1-p)^\gamma[-\log p]
-
(1-\alpha)p^\gamma[-\log(1-p)].$$

这不是把非负的 focal training loss 直接乘以负号；训练分类另有自己的 target 与归一化。

[现状] FastFill 不预测对象是否存在，也不用检测式语义 focal loss 去猜输入已给定的类别。只有 `certify_group` 认证的可交换组才进入 Hungarian（`matching.py:30-77`）。配对代价是该配对下的 grid 位置 loss 加 swap-min 的 log 尺寸 L1，$$\alpha_p=\alpha_s=1$$，没有 yaw、box 或语义项；完整公式见第 5.2 节。它借用的只是 DETR 式“detached 代价 + 线性分配 + 按索引计算可微 loss”的机制。

11.3 object-normalized regression 与 FastFill 的实际回归

[原文 / 代码] V-DETR 用参考 proposal 的尺寸规范化 center residual，size target 是相对 proposal 的对数比。略去数值 epsilon，令参考框为 $$(c^0,s^0)$$：

$$t_c^*=(c^*-c^0)\oslash s^0,
\qquad
t_s^*=\log(s^*\oslash s^0).$$

解码为：

$$\hat c=c^0+s^0\odot\hat t_c,
\qquad
\hat s=s^0\odot\exp(\hat t_s).$$

[代码] center 与 size 的匹配距离都是 L1（`V-DETR/criterion.py:640-644`）；size 训练损失也是 L1（`:518`）。attention 的 reference 更新与几何回归的基准要分别读取（`models/vdetr_transformer.py`），不能笼统说所有 head 都相对上一层预测框回归。

[现状] 旧定稿 §11.3 说 FastFill 用“SmoothL1 log-size”，这不对；`docs/fastfill-v2-reference-code-audit-20261006.md:26` 也写着 SmoothL1，同样过时。实际是（公式见第 4.4、6.2 节）：
- 位置：在房间外包矩形归一化坐标上做 16×16 网格分类，加格内 XY residual（tanh，以半格为单位），z 单独回归；
- 损失：XY residual、z 和 log 尺寸都用 L1（`config:47-48`；`losses.py:53-54, 73-75, 101-114`）；
- 只有 yaw residual 固定用 β = 1 的 SmoothL1（`losses.py:197`）。

用第 6.2 节的记号，单个实例的位置与尺寸项为：

$$\ell^{p}_i=0.5\,\mathrm{CE}\big(\ell_i,k^*\big)+0.4\cdot\frac12\sum_{q\in\{x,y\}}\big|\varrho_{i,k^*,q}-\varrho^*_q\big|+\frac13\big|\hat{\tilde z}_i-\tilde z^*\big|,
\qquad
\ell^{s}_i=\frac13\sum_{q\in\{w,d,h\}}\left|\log\frac{\hat s_{iq}}{s^*_{q}}\right|.$$

[设计] 房间归一化坐标与 log 尺寸比只借鉴了“相对尺度”思想。FastFill 没有点云 proposal，网格分类也不是 V-DETR 的做法，这些都不是对 V-DETR object-normalized regression 的复现。

11.4 angle bin + residual 与配置范围

[代码] V-DETR 训练 angle class，GT residual 除以 $$\pi/K$$，只监督 GT bin 对应的预测 residual（`V-DETR/criterion.py:379-412`）。residual 损失是 δ = 1 的 Huber（`utils/misc.py:25-36`），与 β = 1 的 SmoothL1 相同。

[现状] FastFill 的 residual 归一化同为 $$\Delta/2=\pi/K$$，K = 12（`geometry.py:13-34`；`config:9`），GT-bin residual 也用 β = 1 的 SmoothL1（`losses.py:197`）。在此之上，FastFill 多了一套对称候选（`losses.py:12-19, 182-206`），V-DETR 没有对应机制：
- 对称阶 n 下取 $$\theta^*+k\cdot2\pi/n$$ 各候选“分类 + residual”的联合最小；
- 允许轴交换的对象取 n = 4，并在奇数 k 时交换 $$(s_x,s_y)$$（K1 盒等价规则）。

[现状] 训练数据中 93.7% 的对象阶为 2，只学 yaw mod π。yaw 权重：基线运行 yaw_cls 0.08、yaw_reg 2.0（`config:38-39`）；Isambard 两臂对比 yaw_cls 0.5 与 0.08（`ops/isambard_submit.sh:8-9`）。[计划]（之后）在前向可信的来源上改学整圈 yaw（第 6.9 节）。

[代码] 官方 ScanNet 配置的 angle bin 数为 1（`V-DETR/datasets/scannet.py:41`），走轴对齐框路径，所以不能用 ScanNet 结果证明 K = 12 的家具 yaw 模型有效。

[代码] 官方参数解析器的默认值（`V-DETR/main.py:118-137`）：
- matching：GIoU 2、class 3、center 1、objectness 0、size 0.5，angle cls / reg 均为 0；
- 训练：GIoU 2、semantic class 3（`focalloss_0.25`）、angle cls 0.1、angle reg 0.5、center 1、size 0.5、point cls 0.05。

这些是查阅版本 parser 的默认值，实际运行可以覆盖。FastFill 不把这些数字当作自己的理论权重。FastFill 的权重是标定值：position 1.0、size 0.6、yaw_reg 2.0、yaw_cls 0.08 或 0.5（`config:36-39`），没有被文献验证。

11.5 3DV-RPE 与 one-to-many 分别解决什么

[原文] 3DV-RPE 利用点与预测框八个顶点的相对位置，在对象规范坐标中编码几何关系并形成 attention bias，帮助点云检测器聚焦与当前框有关的点（V-DETR §3.2；实现见 `V-DETR/models/vdetr_transformer.py:672-684`）。这是点到框的几何注意力，与 Hungarian 求对应无关。FastFill 没有观测点云，本方案不移植它。

[原文 / 代码] one-to-many 通过重复 GT，让多个候选 query 都对同一对象获得正样本监督；在扩展后的目标集合上仍做分配。它解决的是检测中正样本太少的问题，不能作为允许输出任意数量对象的理由。parser 默认 `repeat_num = 5`、`nqueries = 1024`（`V-DETR/main.py:86, 105`），criterion 按 replica 计数执行重复监督。

[现状] FastFill 每个请求一个有效 slot，只有位置、尺寸、yaw 三个头，没有 objectness（`model.py:156-160`）。one-to-many、大 query bank、检测分类与 NMS 都不在本方案中。重复请求的两个对象是两个应保留的实例。V-DETR README 要求的 MinkowskiEngine（`V-DETR/README.md:47-51`）不是本系统的依赖。

11.6 引用与复核范围

[现状] 核对过的材料（`outputs/fastfill_v2/review-20261005/papers/paper-code-verification.md:10-17`）：
- OptiScene 论文与官方 SFT 代码（commit `409f131`）；
- V-DETR arXiv 2308.04409v1 PDF：本项目从官方地址获取，不是用户提供，SHA256 `e6d7cbdf…`；官方代码 commit `9062d75`，与本地 `V-DETR/` 检出的 HEAD 一致；
- ATISS NeurIPS 2021 原文与代码 SHA `0909ce0`；
- DiffuScene CVPR 2024 原文；
- CasLayout arXiv HTML `2604.27361v1`；
- DETR / 3DETR 的固定 SHA。

旧定稿写“用户提供的 V-DETR PDF”，应以本条为准。链接到 main 的源码是可变版本，正式实验必须保存实际 commit、配置、数据与算子版本。查阅代码不等于已运行或复现该仓库。

可用于论文方法定位的表述：

FastFill v2 predicts, in a single forward pass, a target local size, a bottom-centre position and a yaw for every object of a given inventory, conditioned on a textual room description (room type and extent, optionally the floor polygon, fixed objects including doors and windows, constraints and declared supports). A LoRA-adapted Qwen3-8B encodes the condition; a two-layer bidirectional slot decoder emits per-object heads: a 16×16 room-grid classification with in-cell residuals and a regressed height, a positive log-size, and 12 yaw bins with residuals. Each object is supervised against one reference layout, with Hungarian assignment restricted to certified exchangeable groups and a box-equivalence rule for objects whose horizontal axes are ambiguous; a deterministic greedy decoder then reduces collisions. The layout is handed to RoomGen/RoomGenBench as oriented boxes with placement labels for asset generation and assembly. On 300 validation rooms given without doors or windows, it predicts object sizes more accurately than LLM agents at tens of milliseconds per room, but it captures pairwise arrangement far less well; conditional layout training with anchor-first decoding is planned to address this and has not been validated.

[设计] 各部分的出处：
- 联合对象属性的场景表示有室内场景生成文献作依据（ATISS、DiffuScene、CasLayout）；
- 二分匹配机制来自 DETR 系列；
- V-DETR 提供 bin + residual 角度监督与相对尺度回归的实现参考；
- 以下是本设计自己的选择，没有被所引论文验证：Qwen 条件编码加外部对象解码器、16×16 网格位置头、K = 12、各项权重、可交换组认证、K1 盒等价规则、spread 解码、RoomGenBench 交接契约，以及 WorldEdge 资产闭环的离线参考协议。

[现状] 英文段落中的比较结论来自 2026-10-09 的测量（尺寸误差与延迟 [M2]；关系成立率 [M1]），测的是 step-6357 基线模型、三字段输入、300 间（关系为 293 间）属于选模队列的房间。[计划] 正式投稿前换成含门窗、未参与选模的冻结集上的结果（Stage 0，第 9.7 节）。

附录 A：相对 2026-10-05 版的主要变更

下表只列实质性变更；措辞调整和补充的 `文件:行` 证据不单列。“依据”列中的 [D] §x 指对照文档条目，[M1]–[M3] 见第 1.4 节末的实测来源代号。

| 节 | 2026-10-05 版的表述 | 本版的表述 | 原因与证据 |
|---|---|---|---|
| 开篇 | 预测后“通过真实资产解析、几何校验和有限修复形成可提交场景” | 一次前向预测后导出带 bbox 的交接包，由下游 RoomGen/RoomGenBench 生成或拟合 mesh、装配场景 | 资产闭环只有离线参考实现，从未在真实资产上运行；[D] §8-a；第 1.3、8 节 |
| 1.1 | $$F_\theta(R,O_{\mathrm{req}},C)$$；$$R$$ 含门窗开口 | $$F_\theta(\Pi(R,O_{\mathrm{req}},C))$$，$$\Pi\in\{\Pi_{\mathrm{full}},\Pi_3\}$$；门窗是 `room.fixed_objects` 盒子，`room.openings` 未使用；约束只出现在 train | [D] §1.1-b；`batch.py:137-156`；manifest `constraint_counts` |
| 1.2 | 系统保留 target 与 actual 两份记录，按实际几何验收 | 只有离线参考 runtime 这样做；RoomGenBench 把 mesh 非等比拟合到 target，receipt 写 `not_checked` / `not_attempted` | `roomgenbench.py:206-250`；第 8.6 节 |
| 1.3 | 有显著 pitch/roll 的对象“应显式分流” | 没有分流：倾斜对象保留 slot、屏蔽全部监督（train 4.2%），记为缺口 | [D] §1.3；`legacy_bridge.py:97-113` |
| 1.3 | 职责表：Harness、FastFill、Asset Resolver、Reconciliation/Validator、Host/Solver | 上游请求方、FastFill、RoomGen/RoomGenBench、离线参考 runtime、Host/Solver（未实现），另写交接边界 | [D] §8-a；`direct_layout.py:192-296` |
| 1.4 | 研究目标：条件生成与资产闭环能否提高场景可用率 | 研究问题：相同字段（含门窗）下，单次前向能否在布局关系上不输于 LLM 智能体，同时保住尺寸精度与毫秒级延迟；附当前实测 | [M1]、[M2] |
| 2.1 | 对象局部 $$+X$$ 为规范前向 | $$+X$$ 只是轴约定：93.7% 的对象只学 yaw mod π，只有 MultiScan 510 个对象保留语义前向；最终朝向由 spread 的背墙翻转给出 | [D] §2.1；[M1] |
| 2.1、6.4 | “尺寸轴也必须固定，不能通过随意交换 $$w,d$$ 隐式制造等价标签” | K1 盒等价：`size_axis_swap_allowed` 对象可写成 $$(s_y,s_x,s_z,\theta+\pi/2)$$，loss 在四个候选上取 detached 联合最小，匹配取较小轴序，评测报告盒等价与 plain 两种误差 | [D] §6.4-b；`legacy_bridge.py:22-35`；`losses.py:171-206` |
| 2.3 | 示例为单张桌子，带 `size_bounds_local_m` 与 `faces_direction` 约束 | 示例按主数据字段：门窗 `fixed_objects`、`obj_%04d`、三字段投影、直接请求、validity | 数据中没有尺寸要求，也没有 `faces_direction`；[D] §3.3-b、§2.1 |
| 2.4 | 未知边界、门窗、支撑、角度都带有效性标记 | 边界、地面、房高、目标字段、支撑有未知标记；门窗没有 | [D] §2.4-b |
| 2.5（新增） | — | 门窗与已有物体的表示、数量、用途、缺口，以及含门窗输入的检查结果 | [D] §1.1-b、§8-d；[M3] |
| 2.6（新增） | — | 三字段投影的定义、适用房间、用途，以及该投影下房间朝向无法辨识 | [D]“规范里没有、但当前已有的部分”第 1 条；[M2] `extra.json` |
| 3.2 | 接入核验包括视觉复核 | 视觉复核未执行，只有自动审计 | `docs/fastfill-v2-data-audit.md:175`；[D] §3.2 |
| 3.3 | 条件中不得含精确尺寸摘要 | 对象描述沿用来源文本，约 1.5k 个训练对象的描述含米制尺寸，未过滤，记为缺口 | [D] §3.3-a |
| 3.4 | 增强含整体平移/旋转与语义改写；另设资产不重叠子集；统计量只由训练数据确定 | 实际 7 项增强设置（rotate90、打乱并重编号、三种丢弃/降级、三字段投影；不镜像、不平移、无语义改写）；资产不重叠子集未建；评测基线默认在评测集上留一拟合，不符合要求 | [D] §3.4-a、§3.4-b、§3.4-c、§3.4-d；config:74-82 |
| 4.1 | 连续几何张量 → 确定性校验与序列化 | 头输出 → spread（默认）或 argmax → `validate_layout`；两种结果分开报告 | [D] §4.1-b；`evaluate.py:121-381` |
| 4.2 | 对象特征由 token 区间、类别特征、约束角色和 slot seed 构造 | slot 只是对象 token 区间均值加 slot seed，没有类别嵌入或约束角色特征 | [D] §4.2；`model.py:163-181` |
| 4.3 | 必须实测输入顺序打乱与 ID 重命名的稳定性 | 未实测；诊断脚本变体 e 把两者合在一起，结果未核对 | [D] §4.3-b |
| 4.4 | 位置头输出归一化底面中心（连续回归） | grid_residual：16×16 格 logits + 格内 tanh residual + 回归 z；regression 头只是可选项 | [D] §4.4-b；config:18-19 |
| 4.5–4.9（新增） | — | spread 规则；逐对象边缘分布为什么表达不了关系；Stage 0 关系解码与 Stage 1 迭代解码（[计划]） | [M1]、[M2] |
| 5.2 | $$C_{ij}=\alpha_p\|\hat{\tilde p}_i-\tilde p^*_j\|_1+\alpha_s\|\log(\hat s_i\oslash s^*_j)\|_1$$ | grid 头下，位置代价就是该配对的格位置 loss；尺寸取两种轴序的较小者；组内尺寸不完整时只用位置；评测匹配仍用解码位置的 L1 | [D] §5.2-a；`matching.py:139-166` |
| 5.3 | 保存 tie-breaking 约定与无效样本计数 | 训练中不记录平局和被跳过的组；只有评测报告和数据 manifest 有计数 | [D] §5.3-b |
| 5.5（新增） | — | Stage 1 对匹配的改动（[计划]） | 第 6.7、6.8 节 |
| 6.1 | $$\rho$$ 为 β=1 的 SmoothL1；各 $$\lambda$$ 从 1 起步 | 位置与尺寸用 L1，只有 yaw residual 用 β=1 的 SmoothL1；权重 1.0 / 0.6 / 0.08（或 0.5）/ 2.0，box、collision、boundary 为 0 | [D] §6.1、§7.2；config:36-51；`ops/isambard_submit.sh:8-9` |
| 6.2 | $$L_p$$ 是三坐标 $$\rho$$ 的平均 | $$L_p=0.5\,\mathrm{CE}+0.4\cdot$$格内 residual L1 + z 的 L1/3 | [D] §6.2-a；`losses.py:101-114, 164-170` |
| 6.6 | $$L_{\mathrm{total}}$$ 含 $$L_{\mathrm{requirements}}$$；正则需消融并防“缩小换低碰撞” | 只有 collision、boundary 两项，权重为 0，没有 requirements 项；碰撞只靠 spread 缓解 | [D] §6.6-a、§6.6-b |
| 6.7–6.9（新增） | — | loss 中的整房对称、条件（masked）布局训练（Stage 1）；整圈 yaw 与偏好优化（之后），均为 [计划] | [M2] `result.plan` |
| 7.1–7.3 | 五步流程：审计 → 联合训练 → 选模 → 真实资产闭环评估 → 冻结部署包 | 实际运行与 autorun 流程：单配置训练 → `autorun.score` 选模（三字段 + spread）→ 验收前上传私有 HF → 测试 → RoomGenBench `layout_boxes` | [D] §7.1；`autorun.py:1-19, 529-559` |
| 7.4（原 7.3） | 选模依据包括对象完整性、几何可用性、资产解析率和约束通过率 | 实际选模分数 `autorun.score`（公式已写出）及其缺陷；Stage 1 不再用它选模 | [D] §7.3-a；`autorun.py:44-69`；[M1] |
| 7.6–7.8（新增） | — | Stage 0 不训练与停止的做法；选模规则草案；试点 go/no-go、预算与成功判据 | [M2] `result.plan` |
| 7.9（原 7.4） | 文本 SFT 对照可与主模型并行开发和训练 | 只跑过一次冒烟且 JSON 解析失败；要求完整标签；不在 Stage 0/1 内 | [D] §7.4-a、§7.4-b |
| 8 | Harness → Resolver → Reconciliation → Validator → Host → Solver “作为主系统整体交付” | 实际下游是 RoomGen/RoomGenBench 交接契约（字段映射、place/place_id、shell、`--require-placement`）；旧闭环缩为离线参考实现和下游接口 | [D] §8-a、§8-c；`direct_layout.py:140-373`；`roomgenbench.py:44-440` |
| 8.7 | Validator 检查开口与通行、实际资产几何、仿真与任务 | 只有 bbox 级检查；开口与通行没有检查；mesh、physics、solver 恒为 unknown | [D] §8-d、§8-e |
| 9.1 | 以全部输入请求为分母 | 在被评测的投影内以全部请求为分母；三字段投影跳过非矩形房间（test 4,165/8,620），必须写出跳过数 | [D] §9-a |
| 9.2–9.5 | 指标组：输出与需求、几何可用性、参考相似度、资产与任务闭环、成本 | 三层：参考相似度与校验计数（[现状]）、布置指标（[计划]，已有离线测量）、Qwen3-VL 成对评审（[计划]） | [D] §9-b、§9-c、§9-d；[M1] |
| 9.6 | 对照：相同条件下的未微调语言模型 | 外部 LLM 智能体（gpt-6.1-sol，四种模式）在 300 条三字段验证行上对照；含门窗的对照为 [计划] | [D] §9-e；`README.md:786-849` |
| 9.8 | 报告多种子或置信区间，按房屋聚类 | 单种子、房间级 bootstrap；按房屋聚类和第二种子为本修订提出、待确认的计划 | [D] §9-i |
| 9.9 | 核心消融表 | 逐项写明已做、只有冒烟、未实现或计划 | [D] §5.2-b、§6.6-b、§7.4-b、§9-f、§9-g、§9-h |
| 9.10（新增） | — | 当前结果：关系成立率明显低于 LLM harness，尺寸与延迟占优，单独加门窗不改善关系 | [M1]、[M2]、[M3] |
| 10.1、10.3 | 五个交付模块“共同构成 FastFill v2 的完整交付范围” | 按 main 2e86188 的 38 个模块列出已实现与未交付，未交付项标明属于 Stage 0、Stage 1、之后、未排期或下游 | [D] §10-a、§10-b、§10-c |
| 10.2 | 必须通过的正确性检查（只列类别） | 对应到具体测试文件与行；最近一次有记录的全量运行为 2026-10-07 | `README.md:214`；`tests/` |
| 11.3 | FastFill 用 room-normalized bottom-center 与 SmoothL1 log-size | 网格分类 + 格内 residual + z 回归；log-size 用 L1；只有 yaw residual 是 SmoothL1 | [D] §11.3；config:47-48 |
| 11.6 | 核对了“用户提供的” OptiScene PDF 与 V-DETR PDF；方法定位段描述完整资产闭环 | V-DETR PDF 由本项目从官方获取；方法定位段按现状重写，写明当前关系差距，并说明 Stage 1 的修正尚未验证 | [D] §11.6；`outputs/fastfill_v2/review-20261005/papers/paper-code-verification.md:10-17` |
