# FastFill v2：设计、全量源码与实际交付审核

审核日期：2026-10-06。用户提供的 2026-10-05 设计是规范，不是已训练模型或已交付系统的报告。本次审核以它为主线，核对当前 FastFill v2、固定 V-DETR / MinkowskiEngine 源码与论文，并修订现行文档。当前主条件仍为房间、指定实例、固定几何和空间要求；最终下游已明确为 RoomGenBench。

## 1. 设计成立的部分

以下定义与当前连续几何路径一致：请求数量和稳定身份由输入决定；局部全尺寸不随 yaw 变化；position 为 bottom-center；转换 geometric center 时预测与 GT 各用自己的高度；正 size、room-normalized position、周期 yaw-bin + GT-bin residual；只在合法独立可交换组做 detached assignment；从原始预测张量计算可微 loss；JSON 序列化不参与反向传播。文本 SFT 可独立开展，不是主模型的前置。

“OptiScene + V-DETR，把 bbox 从输入移到输出”正确描述任务变化与监督参考。但架构也改变了：OptiScene 是 causal LM token 训练，当前是 Qwen 条件 hidden states 加外部双向对象解码器及连续 heads。FastFill 使用 room-normalized bottom-center SmoothL1 与 log-size ratio SmoothL1；V-DETR 使用 proposal-relative center / log-size L1。不能写成原模型原样复现。

## 2. 原文需要改的具体位置

| 位置与原说法 | 问题 | 可直接替换的表述 |
|---|---|---|
| §1.2 “实测尺寸说明该条件下曾存在一个可用选择” | 来源有物理位姿、框代理及缺失支撑证据，数值参考不自动证明可用 | “经字段资格审核的场景尺寸是一份监督样本；是否为可检索／可生成且通过实际几何验收的选择，须由资产与场景证据另行确认。” |
| §6.1 “从训练开始同时监督完整 size、position 与 yaw” | 容易误读为每个样本四项都有有效标签；主集只有 581 个有效 semantic yaw | “从训练开始联合启用三类输出及四项基础 loss；每项仅使用相应有效且需学习的标签，并记录各窗口实际曝光。” |
| §6.2 “分布式全局有效计数” | 当前保证每个 global microbatch 的有效数等价；累积窗仍平均 microbatch 均值 | “各 rank 在每个 microbatch 按全局有效实例数归一化；累积为这些 microbatch 均值的平均，不等于全窗全对象均值。尾窗按实际长度修正。” |
| §6.5 box 有效性与 3D 选项 | 当前可选算子只有 BEV oriented convex-hull GIoU | “BEV 只需要可信 XY、局部水平尺寸与 yaw，不监督 z/h；3D box loss 是待实现的单独选项，不计入当前交付。” |
| §4.4 正尺寸参数化 | exp 的实现有 clamp 和有限输出范围 | “预测 size = s_ref × exp(clamp(u,−limit,limit))；记录 reference、limit、float32 指数运算以及有效 GT 的范围预检。固定维度直接采用条件值。” |
| §8.2 Asset Resolver 只写既有资产检索 | 最终 RoomGenBench 是 per-object text+bbox → 生成 mesh／装配 | “资产层支持给定 text+bbox 的 mesh 生成或既有资产选择，返回状态、原生 actual geometry、规范化变换和证据。具体生成器按固定版本与 sidecar 契约接入。” |
| §8.2 “非均匀缩放不是默认补救”与实际 RoomGenBench | 原装配器默认将 mesh AABB 非均匀拟合到 bbox，属于真实协议冲突 | “RoomGenBench benchmark fit 单列为装配实验政策，保存拟合前后 size 与变换；不将其计作原生尺寸适配成功。严格实际资产模式需独立禁止或限定缩放。” |
| §8.4、§10 “整体交付” | 当前无真实 mesh/physics/Solver checker、持久 Host；非空 openings 仍 unknown | “完整闭环是设计验收目标；当前交付为模型／数据／bbox 下游契约及离线参考验证。已实现与待接入能力按模块和版本列明。” |
| §7.4 / §9.3 相同数据 Text SFT 比较 | 文本入口要求全部对象完整几何；主集含部分标签 | “受控 Text / Structured 比较在同一冻结完整标签子集上进行，或另定义缺失字段文本协议；全量部分监督 structured 是另一个覆盖实验。” |
| §9.3 未微调条件语言模型 baseline | 新外部 decoder 未训练时是随机 heads，不能笼统叫未微调 LM | “分别定义未微调 Qwen 的文本输出 baseline，以及冻结 Qwen、训练对象 decoder/heads 的 structured baseline，报告可训练参数和曝光预算。” |
| §11.6 “核对用户提供的 … V-DETR PDF” | 早期官方 arXiv v1 是下载件，当前另有用户本地 details PDF | “分别列出本地 PDF 与官方论文版本、SHA256、代码 commit 和查阅范围；不把官方获取件写成用户提供件。” |

这些修订不要求先做文本 SFT，也不改变联合预测 size/position/yaw 的主任务。参考框贴合、对象间不穿插、房间边界、真实支撑、任务能力分别有不同责任；GIoU 不能代替其他验收。

## 3. 当前数据与历史审核的关系

主数据全量构建和逐行资格核验已完成：141,341 场景、1,787,052 请求对象、392,099 固定对象，train / validation / test 为 124,589 / 8,137 / 8,615。真实 tokenizer 资格为 124,375 / 8,125 / 8,602。19,307 个旧训练 flags 过滤、316 个 v2 拒收和 239 个当前 token/object 拒收都有独立台账，不能把原 144,140 行解释为所有历史 run 的实际入训量。

review2 C1–C4 已在当前代码修复；D1/D2 已在新主版本保守处理。原数据不被抬高、缩小或补成有效零值；全部数值保留。详见 [review2 当前处置](fastfill-v2-full-review-followup.md) 与 [多源完成记录](fastfill-v2-multisource-20261006.md)。

“构建完成”指当前定义的数据、split、masks、provenance、hash 和 tokenizer 资格产物完成。它不包括全部原生 mesh/front/支撑重新认证、不包括恢复旧 flags 候选、不包括正式训练或完整资产可提交率验证。

## 4. RoomGenBench 的实际接法

上游完整 condition → FastFill 原始 layout → RoomGenBench scene/asset registry → 指定 text-to-mesh 方法 → sidecar `ok / fallback / failed` → 实际 mesh 装配与验证。RoomGenBench `layout_boxes` 只显示输入框；灰色 placeholder 和 fallback 不能计作真实生成成功。它的 benchmark 装配不实现 WorldEdge 原子提交或任务验收。

FastFill local [w,d,h] 对应 RoomGenBench [length,width,height]，yaw_export = wrap(yaw−π/2)，position 保留 bottom-center。本轮新增独立动态装配入口，不改下游共享源码；导出完整 condition/layout，保存 fixed/support/constraints 和未知标记，消费任意场景的已生成 GLB＋sidecar，记录原生与拟合尺寸，并保留失败实例。未知高度只使用明确 display reference。实际生成器运行、支撑 mesh 核验与 Host 仍不在该入口内。详细命令和原源码边界见 [RoomGenBench 接口审核](fastfill-v2-roomgenbench-interface-20261006.md)。

## 5. 审核与证据边界

固定参考仓库为 V-DETR `9062d75fe2c91e5d4a771b5325483fc330a3e827`、MinkowskiEngine `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`；RoomGenBench `30f2e059d046ccd4aa928d0109a09a52f11693b5`。逐文件阅读范围、独立反例与论文证据保存于 `outputs/fastfill_v2/audit-four-20261006/`。

源码阅读、哈希完整性、CPU 反例、GPU 编译运行和论文模型复现是不同证据。当前 FastFill 不引用 MinkowskiEngine；V-DETR 的稀疏点云骨干、检测语义/objectness、大 query bank、one-to-many、NMS 不自动进入已知对象清单的生成任务。克隆参考库不能当成当前 Python3.12 / torch2.13 / CUDA12.6 下已完成兼容性编译。

本轮全文范围为 V-DETR 53 源文件／10,488 行、MinkowskiEngine 232 文本文件／52,848 行；当前 FastFill v2 36 个生产模块也全文覆盖（最终 hash 清单随发布收据）。参考代码有默认入口、旋转／薄轴 GIoU、Python API、native 生命周期和索引等确证缺陷，详见 [参考审核](fastfill-v2-reference-code-audit-20261006.md)。没有修改参考仓库或冒称原 detector／CUDA 复现。

## 6. 本轮新增 FastFill 修复

| 问题 | 修复及证据 |
|---|---|
| FP16 overflow 没有真实 optimizer update 却计入 step | 只有 Accelerate 确认真正更新后才增加步数、写日志、验证和保存；overflow 单独计数。真实训练入口 CPU GradScaler 的 RED 为零 optimizer 调用但旧 step=1，GREEN 不再记录成功更新。 |
| 一个不完整交换组关闭整个场景 Hungarian 指标 | 只将该组回退固定身份，其他完整组继续经合法交换认证及组内 assignment。反例误差 1.0 m → 0.0 m；当前冻结主数据已资格降级，不改变其数值指标。 |

这些修复与历史 C1–C4 的修复分别记录。四项基础 loss 方向、GT-bin 选择、DDP 有效计数和不足额尾窗重标未发现新的确证公式缺陷。这里没有声称“不存在其他 bug”。完整审查见 [训练审查证据](evidence/fastfill-v2-audit-20261006/fastfill-training/README.md)。

最新服务器安装 metadata 的依赖 advisory 检查已执行，公开版本解析的 `pip-audit` 没有已知漏洞；这不是源码安全证明或 native 编译验收。当前修改另由独立代理检查输入、输出路径、状态码与秘密，发布时保存实际结果。

## 7. 2026-10-07 第二轮修正

第二轮在提交 7ea1a0d 之上实现了 K1–K10 契约（改动未提交）。其中三项修正了本审核及第一轮的结论：

| 项目 | 第一轮做法 | 第二轮修正 |
|---|---|---|
| 目标高于声明房高（K2） | 把 `room.height_m` 置为 null（原 C7(b)） | 撤销。标签和 condition 都不改；只在 `provenance.height_conflict = {objects, max_excess_m}` 记录，并在 changes.jsonl 写一条 `provenance.height_conflict`，原因 `target_exceeds_declared_height_flag_only`。容差仍是 0.05 m＋1e-6。评测中这类行的 ceiling 检查单独计为 `ceiling_on_height_conflict_rows`，不算模型违规。第三轮（2026-10-07）：bridge 也不再因目标把冻结 prep 的房高置空（只记 `provenance.legacy_height_dropped`），并直接写入该标记；qualified 按自己的掩码重算，只在结果与父行不同时写 journal，原因 `height_conflict_follows_qualified_masks`。 |
| 盒对称层级（K1） | 局部轴配对不确定的对象也按写出的轴序监督尺寸，yaw 按 π 周期（order 2，Scan2CAD 标注的对象除外） | 新增逐对象 `validity.size_axis_swap_allowed`：InternScenes arkit／3rscan／mp3d／scannet、InteriorGS 全部对象，HSSD200 类别含 chair、MultiScan 类别含 bed 的对象为 true，其 `yaw_symmetry_order` 为 4。loss 在四个候选（yaw＋kπ/2，奇数 k 交换 xy 尺寸）中按联合代价取最小；匹配和评测用同一组候选，评测另报写出轴序下的误差。这只表示盒子本身对称，不认证语义正面。 |
| 三字段输入契约（K5） | 直接请求和训练行各自生成条件，边界标志和多边形写法可能不同 | `batch.render_minimal_condition` 是唯一的三字段投影（房型、外包矩形、物品清单，`boundary_known=true`，不含约束和支撑）。默认 rectangular 请求经 `direct_layout.request_to_condition` 得到的条件文本与之逐字节相同；`boundary_quality` 从不渲染。训练以 `minimal_form_p` 概率在边界已知的矩形房间上使用该投影（hull／`reference_extent` 矩形不投影），验证与评测同时报告完整条件和三字段投影，选模用三字段投影上的加权几何目标 `validation.minimal.geometry_objective`（`train.SELECTION_METRIC`）；collapse 只作诊断，与真值锚点比较时用覆盖同一批对象的 `collapse_matched`／`predicted_matched` 列。`reference_extent` 消融保留自己的 `boundary_known=false` 设定和独立数据。 |

同轮其余变化：地面支撑只取源标注（K3，2 cm 内吸附 target z，更远的不声明）；交付时 `place`／`support_status` 为 declared／inferred／unknown 三态（K6）；新增可选 grid_residual 位置头（K7）；训练启动即写 `run_manifest_start.json`，evaluate／predict 按 checkpoint 绑定 max_length（K8）；正式配置改为 4 卡回归和 3 卡 grid 两组、全局 batch 96、3,887 次更新（K9）；校验状态 fail 改名为 violation，并统计 pass／violation／unknown 三态（K10）。MansionWorld 的 IR 没有区分标注尺寸与 footprint 代理的字段，尺寸继续整源无效（K4），是否以 `anchor` 间接区分有待决定。细节和本地验证记录见 [v2 README 的 round 2 一节](../fastfill/v2/README.md#2026-10-07-round-2)。这些是代码与数据契约修正，不是模型质量结论。
