# FastFill v2 独立复核：数据、实现与文献

**历史审核范围。** 本页保留 2026-10-05 / 初版实施的代码、数据和反例事实。当前主数据、服务器、C1–C4 / D1–D2 处置及本轮参考库全审见 [当前处置](fastfill-v2-full-review-followup.md)、[多源记录](fastfill-v2-multisource-20261006.md) 和 [设计审核](fastfill-v2-design-audit-20261006.md)。下文“尚未接入／未运行”和旧命令只解释当时快照。

日期：2026-10-05。复核对象为本地 `OptiScene/fastfill/v2` 当前未提交实现，以及 `/Volumes/harddisk/3D_Room_Collections` 中的源文件。初次复核保持被测代码不变；后续修正只写入独立 `fastfill/v2` 实验路径。外盘源数据、MinkowskiEngine、v1 入口与 v3/v3.1/v3.2 冻结实验保持不变。复现脚本与下载的参考文献保存在 `outputs/fastfill_v2/review-20261005/`。

**结论：沿用我们已经筛选的语料，迁移为新的 v2 监督格式；不重新选库，也不把它替换成 208 份 MultiScan。设计主线成立，基础结构化训练路径有证据支持，初次复核发现的语料接入缺口和代码错误已在后续本地实施中修正。需要生成新的 v2 派生数据，不需要重新下载或覆盖原始数据。**

本文件的实施记录描述第一版历史快照。后续外部审核发现的 P1/P2、修复与新数据修订见 [review2 审核](fastfill-v2-review-20261006.md)；当前实验使用最新多源资格版本，不使用 review2 旧包替代。

## 实施更新：2026-10-06

以下各节保留初次复核的反例和证据，不能将“已复现的错误”误读为当前仍未修复。当前独立 v2 实现已修正 DDP count dtype、Metal 框算子梯度路径、统一 hard-on 支撑解释、严格 target 类型、无学习标签过滤、验收层级与 capability 边界，以及数值安全的图连接零。最终运行时复核还堵住属性要求被忽略的提交路径：当前非空 typed attributes 在缺少可信资产属性契约时拒收，实际 Validator 返回硬性 `attributes_unverified`；外部 Resolver 或未验证 metadata 不能绕过。完整自然语言检索与属性证据校验仍需真实资产适配器。

已接入我们的冻结 v3.2：校验 26 个输入哈希和原预处理代码，继承 UID/split 与请求对象选择，用未经旧版 cm/degree/z-snap 舍入的 IR 连续标签构建新的监督记录。匿名 ID/顺序采用独立散列；bbox 推断的 `on` 不进入可信支撑输入，按对象记录原始与旧版推断证据；矛盾房高保持未知。已记录的斜窗宽度只修正新副本；InternScenes k=0 按对象标记前向未知。倾斜固定几何明确拒收，不静默投影。

默认 `strict` yaw 只接纳具有独立语义 front 证据的 MultiScan；其他来源保留位置/尺寸中可信的字段。MansionWorld annotation-footprint 尺寸与 Holodeck padded box 属 proxy，屏蔽 size 监督。`legacy-convention` 是显式的低证据 yaw 消融，不能改称逐资产语义前向真值。完整 size/position/yaw 语料与部分字段监督语料必须分别报告；文本 SFT 的公平对照采用两者共同的完整标签子集。

本地完整标签 optimizer smoke 已验证四项基础 loss、六个模块非零梯度、checkpoint 与推理。结构化 smoke 在 6 个请求中 schema/ID/正尺寸均通过；目标几何通过率 0/6；仅一步 tiny text CE 后的一个生成请求解析失败。这里验证的是软件链路和失败统计，不是 Qwen 训练收益，不能据此声称布局或真实资产闭环有效。

最终全量验收：160,964 个旧 UID = 19,307 个旧默认训练过滤 + 141,341 个保留 + 316 个明确拒收；无未解释 UID。新划分为 124,589 / 8,137 / 8,615，共 1,787,052 个目标对象；position 有效 1,732,321，size 有效 1,224,670，yaw/完整几何有效 581。训练中的 yaw 标签仅 523，跨 58 个 MultiScan 场景。严格完整标签场景为 57 / 2 / 6，已单独导出并逐行 byte-compare。全库均匀 sampler 加默认 1,000 步不能保证 yaw 训练充分，正式训练配置仍需确定。

独立 verifier 检查全体记录的协议、数值/掩码、对象 ID、合法交换组、原 UID/flags/划分、house 与 alias 泄漏和拒收计数，结果 0 错误；26 个冻结输入、2 个外部审核证据、3 个迁移实现与 5 个旧预处理文件哈希吻合。窗口修正作用于 12 个保留房间中的 23 个固定窗口对象；6,706 个保留目标按 k=0 证据标记前向未知；54,731 个倾斜目标使用掩码；7,577 条推断 on 约束被省略。拒收 316 = 无效 polygon 4 + 未证实斜窗 111 + 无可信字段 16 + 倾斜 fixed 185。不得把省略约束或拒收场景误报为全部旧样本无损转入。

最终回归套件 **455 tests + 114 subtests passed**，v2 覆盖率 **86.85%**。外盘新训练包位于 `/Volumes/harddisk/FastFill_v2_20261006/`；对应 [执行手册](fastfill-v2-runbook.md) 和包内独立审核/校验清单。

全量迁移及独立验收的逐来源统计见生成的 manifest 和完整验收 JSON；命令和产物入口见 [v2 README](../fastfill/v2/README.md)。正式 checkpoint、机器和训练预算按用户要求随后确定。

这里必须分开四件事：公式是否正确、代码是否遵守公式、数据标签是否可信、最终场景是否可提交。论文依据和单元测试不能同时证明这四件事。

## 0. 用户纠正后核实的已有筛选语料

此前确实完成了筛选。`fastfill/README.md` §3、`docs/audit-2026-09-28.md`、历史 memory 记录与当前实际 MANIFEST 一致：**Excel 22 项 → 保留 16 个下载数据集 = 11 个产生训练房间 + 2 个仅评测 + 3 个辅助**。11 个训练数据集展开为 **16 个训练 source**（InternScenes 五支、IL3D 两支）；加上 SceneSmith/SpatialGen 两个 held-out source，build 读取 **18 个 IR 文件**。

| 用途 | 已有选择 |
|---|---|
| 11 个训练数据集 | SpatialLM、IL3D、OptiScene(Holodeck 部分)、InternScenes、MansionWorld、SAGE-10k、Structured3D、InteriorGS、HSSD-200、SceneCAD/Scan2CAD、MultiScan |
| 2 个 held-out | SceneSmith、SpatialGen，不能在迁移中移入 train |
| 3 个辅助 | 3D-FRONT 提供 IL3D 源房间/地面；3RScan 提供 reference/rescan aliases；ARKitScenes 提供 video→visit aliases |

本轮重新遍历我们实际 v3.2 的 **160,964 条成品记录**：train=144,150，dev=8,167，test=8,647；旧默认 flag 过滤后 train=124,843，分词/weighting 前。计数证据在 `outputs/fastfill_v2/review-20261005/our-selected-corpus-counts.json`。**这个总量是旧任务 corpus 数量，不能直接声明为新 v2 完整几何有效数量。**

| 训练 source | train | dev | test | 旧默认筛选后 train |
|---|---:|---:|---:|---:|
| SpatialLM | 42,068 | 2,239 | 2,461 | 33,081 |
| MansionWorld | 19,396 | 1,153 | 1,313 | 19,359 |
| IL3D_3dfront | 18,054 | 939 | 872 | 14,520 |
| Structured3D | 12,700 | 793 | 747 | 9,855 |
| InternScenes_gen | 12,552 | 828 | 660 | 12,319 |
| SAGE-10k | 8,977 | 503 | 489 | 8,947 |
| OptiScene_holodeck | 8,946 | 497 | 503 | 8,946 |
| IL3D_synthetic | 6,277 | 349 | 316 | 6,256 |
| InteriorGS | 4,877 | 268 | 311 | 3,315 |
| InternScenes_arkit | 4,528 | 280 | 229 | 3,952 |
| HSSD200 | 1,876 | 86 | 138 | 1,356 |
| InternScenes_scannet | 1,309 | 80 | 104 | 929 |
| InternScenes_mp3d | 1,258 | 89 | 18 | 976 |
| InternScenes_3rscan | 1,200 | 61 | 70 | 963 |
| MultiScan | 114 | 2 | 6 | 58 |
| Scan2CAD | 18 | 0 | 1 | 11 |
| SceneSmith，held-out | 0 | 0 | 364 | 0 |
| SpatialGen，held-out | 0 | 0 | 45 | 0 |
| 合计 | 144,150 | 8,167 | 8,647 | 124,843 |

当前正确的冻结输入是：

```text
.release/v3.2/data/v3.2/MANIFEST.json
.release/v3.2/data/v3.2/QA.json
.release/v3.2/ir/
```

外盘 `/Volumes/harddisk/fastfill_data/` 当前只保存 v3、v3.1，没有 v3.2 子目录。更重要的是，本轮对 HSSD200、InteriorGS、MultiScan、Structured3D 的外盘 `/Volumes/harddisk/fastfill_ir/` 文件重新算 SHA256：四个均与旧 v3 输入一致、均与 v3.2 MANIFEST 不同。**不能把外盘旧 IR 当成最新 v3.2 迁移起点。**

本轮额外重算冻结 v3.2 的全部 **26/26 SHA256**（8 个数据文件 + 18 个 IR），与 MANIFEST 全部一致；8 个数据文件的记录行数也全部匹配。实读 2,251,385,112 bytes，18 个 IR 合计 188,670 行；IR 行数是本轮实测，manifest 本身只规定其哈希。证据在 `outputs/fastfill_v2/review-20261005/our-v3.2-manifest-check.json`。这是实际文件完整性核验，没有重新运行全局 adapter/build/split，也不等于几何标签全正确。

之前的 raw-source 表不能替代我们的筛选与修正管线：已有 HSSD adapter 已测量 opening GLB offset、恢复部分 boxless/URDF 几何、处理 twins；IL3D_3dfront 已从 assets.json/scale 与源 floor 恢复正确尺寸、拆分合并房间；IL3D_synthetic 已用 HSSD catalog 与补充 bounds 恢复 pivot。这些已验证处理应复用。原始导出中的缺口不等于我们现有 IR 仍缺同一字段。

初次复核时新 v2 builder 仅支持 MultiScan，后续已通过独立 bridge 接入冻结语料。下文 208 样本、GT 0/208 等结果只适用于**新单源 smoke 数据**，不适用于我们筛选后的全部 v3.2，也不否定此前 124,843 个参考布局的旧 bbox 验收记录。后者同样不等于 mesh/physics 或新的严格 Validator 已通过。

## 1. 本次证据范围

| 对象 | 实际完成 | 没有完成 |
|---|---|---|
| MultiScan | 全部 CSV 行统计；273 扫描的适配器复现；6,181 个已收录目标框角点对照；全部 208 份 GT 的 Validator 检查；3 个 PLY、85 个对象的 mesh/OBB 数值核验 | 全部扫描视觉叠加、完整房间恢复、物理/任务验收 |
| HSSD | 全部对象 CSV 的有效框统计；资产 catalog、区域及实际资产文件检查 | 所有资产前向/mesh 的逐个复核、articulated 几何恢复、v2 adapter |
| 其他 14 个来源 | 本地实际首记录或有界 JSON 读取、converter/README/source_fields 核对 | 全量 JSON 解析、全量可用比例、全部资产视觉复核 |
| 模型与目标 | 条件编码、slot decoder、assignment/loss 源码核对；反例复现；真实双进程 CPU/gloo 基础训练 | 正式预训练 Qwen 数据训练、CUDA/混合精度大规模训练、模型质量收益 |
| 系统 | Resolver/actual reconciliation/Validator/内存 Host 的本地代码与反例 | 真实 WorldEdge persistent Host、实际资产库检索、mesh/physics/Solver 执行 |

此前离线套件记录为 285 tests + 97 subtests 通过，v2 模块覆盖率 85.57%。这个数字不是整个仓库覆盖率，也没有覆盖本次发现的反例。随机小 Qwen2 和 tiny backbone 的 API/梯度测试不是正式 pretrained Qwen 训练。

## 2. 已复现的实现错误

下表记录初次复核代码的实际反例；当前修正状态见实施更新。复现脚本当时没有修补被测代码。

| 优先级 | 错误与触发条件 | 实际结果与原因 | 修正要求 |
|---|---|---|---|
| P1 | optional box loss + DDP，某 rank 有完整框标签、另一 rank 没有 | gloo `SIGABRT / op.nread == op.preamble.nbytes`。`boxes.py:81` 升为 float64；有 box 的 total 是 float64、空项的 zero 是 float32；`losses.py:32` 用 total dtype 创建 collective count，rank 间 dtype 不同 | 所有 rank 使用固定的 collective count dtype/device，并测试零标签/不均匀标签；检查 collision 分支同类风险 |
| P1 | 在本机 MPS/Metal 上启用 BEV box/collision loss | `Cannot convert a MPS Tensor to float64`。`boxes.py:81–83` 无条件 `.double()` | 明确支持的设备/精度，提供保持梯度的 CPU fallback 或经验证的 MPS 数值路径；默认禁用不等于已支持 |
| P2 | 有效 condition 只在 constraints 中写 `on(lamp, desk)`，不重复写 request.support_parent | `on` 支撑检查 pass，同时 `validation.py:74` 产生 hard `support_unknown`，最终错误拒绝 | 输入边界统一支撑表达、检查冲突与环；reconciliation、matching 与 Validator 共享同一解释 |
| P2 | 训练 target 中用布尔值/数字字符串伪装尺寸，或 yaw=true | `batch.py:116` 调用 float()，`batch.py:152` 允许 bool 作为 int，最终成为可信 size=[1,1,1]、yaw=1 | target 独立 schema 严格检查数值类型，未知与缺失用 null/有效性掩码，不接受类型强制修复 |
| P2 | 只有完整 size 标签，但尺寸全部已由 fixed_size 条件确定，position/yaw 缺失 | `train.py:31` preflight 保留样本；criterion 四项 count 均为 0、loss=0，无可学习监督 | 依据 label validity 与 learn masks 的交集计算可训练性，并记录剔除原因 |

另外三个较小的边界问题也已复现：

1. `CatalogResolver` 只过滤 category/capabilities/fixed/range/front，并按 size 排序。请求 red wooden desk 时，可以选择同尺寸 blue plastic desk；description/attributes 没有参与语义检索或条件满足检查。它是 catalog 参考实现，不能宣称已经完成完整语义 Asset Resolver。若属性是硬要求，需要 typed attribute/capability 契约。
2. 直接调用 `validate_scene(required_levels=("bbox", "mesh_typo"))`，未知 level 被忽略；直接给 validator `capabilities="capability"`，required=["a"] 可因字符串字符集合而通过。当前 CLI 的 level choices 与 Asset constructor 挡住这两类输入，但公共 Validator 边界仍不完整。
3. `_safe_zero` 先求和再乘零；多个有限 float32 size=1e38 会先溢出，造成 NaN loss。默认 size exp clamp 避开常见触发，但“输入逐元素有限”没有保证数值安全，应使用不会溢出的图连接零。

复现证据：

- `outputs/fastfill_v2/review-20261005/model_review_repros.py` 与 `model_review_repros_results.json`。
- `outputs/fastfill_v2/review-20261005/ddp_box_mask_repro.py` 与 `ddp_box_mask_failure.txt`。
- `outputs/fastfill_v2/review-20261005/runtime_repros.py`。

基础四项 loss 的真实 2-process CPU/gloo smoke 已通过：两个 optimizer steps，global position/size/yaw count=2，六个模块梯度范数均非零，保存 checkpoint 与 manifest。该次 `torchrun` 使用另一个本地 Python/torch 环境（torch 2.6、transformers 4.49、peft 0.15）；其他 `python` 复现使用当前 miniforge 环境。不能把它写成 CUDA 或全部版本的分布式验收。

## 3. 已正确实现的基础契约与尚未交付范围

| 契约 | 当前代码证据与判断 |
|---|---|
| 不重新学习 What | 请求对象数/ID/category 从输入继承，无 objectness、background、NMS 分支 |
| 结构化主模型 | condition-only Qwen memory；外部 bidirectional TransformerDecoder；request-bound slots；padding mask；连续 position、正 size、yaw logits/residual heads |
| 尺寸与中心 | local full size、bottom-center；yaw 不改变 local size；几何中心转换分别使用各自高度；固定尺寸维度与可靠 floor z 可屏蔽监督 |
| matching 与梯度 | 固定身份默认；仅合法 exchangeable groups 做 detached cost assignment；loss 使用原始可微预测；关系/支撑置换工具已有 |
| 基础 loss / yaw | SmoothL1(beta=1)，log-size ratio，固定 /3 坐标 reduction，full-valid 向量；GT bin residual；同一等价 yaw 候选的联合分类/回归代价 |

这些判断有代码与已有离线测试支持，不能提升为模型质量结论。

完整交付仍有以下范围没有完成：

1. 数据：尚未建立已有 18 个选中 IR 到 v2 的桥接；应复用现有 adapters、筛选、跨源 alias 去重与 split，而不是重新选 16 个 raw 家族。新增 field validity、asset-disjoint 测试、无目标泄漏的对象顺序/匿名 ID 与条件增强仍需交付。
2. 训练与消融：validation 当前为 mean-of-batches geometry loss；没有按原始约束可用性和真实资产通过率选择 checkpoint。raw-size、sin/cos、yaw/box matching cost 等配置及完整基线未交付。
3. 运行验收：现有 Validator 是保守 upright OBB 路径；openings 非空直接 unknown；wall attachment 没有实际检查；mesh/physics/solver 要求会保守阻止提交。代理框接触与 hollow geometry 还没有解决。
4. 资产与提交：有 metadata catalog 与内存原子 Host 示例，没有真实资产索引/完整语义排序、mesh 规范化链、真实 WorldEdge atomic adapter 和 Solver。`fallback_rate=0` 表示当前没有 fallback 实现，不是经过实验的优秀指标。
5. 实验：尚无正式 Qwen 重训、模型相对 v1/text/baselines 的收益、多随机种子或完整资产闭环结果。

模型的确定性点回归也不自动解决布局多解性或平均化。Hungarian 处理编号歧义，不保证每个位置合理、不重复或网络 permutation-equivariant。冻结设计已承认这个风险，后续应实测。

## 4. 新 MultiScan smoke 数据的明确判定

原始 CSV 共 10,957 对象、273 scans、273 region rows。新 v2 build 收录 208 scans、6,181 targets；split 为 train=159、validation=16、test=33。拒绝 65 scans：up/OBB 轴不一致 11、tilted fixed geometry 30、front/OBB 轴不一致 19、tilted requested object 5。

**6,181 个框的最大角点 round-trip 残差为 2.0947646e-15 米**，支持当前半长→全长、局部轴/front 规范化与 bottom-center 转换正确。MultiScan 官方注明这是 semantic OBB，front/up 与 bbox 轴平行；文档样本与本地 scene_00000_00 obj1 一致。[官方注释协议](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/files/annotation.html)

额外检查三个扫描中 85 个对象的实际 PLY 顶点，最大 mesh 超出 OBB 约 1.45e-7 米；这只覆盖样本，不替代全部视觉/物理验收。

| 对所有 208 份当前 GT 的检查 | 结果 |
|---|---:|
| schema 通过 | 208 |
| boundary_unknown / ceiling_unknown | 各 208 |
| support_unknown | 6,181 对象 |
| target-target 代理框重叠 | 6,597 对；203 场景 |
| target-fixed 代理框重叠 | 4,903 对；195 场景 |
| 当前严格 Validator 全通过 | 0/208 |

这说明参考数据与当前严格 bbox 验收策略不相容，**不说明所有扫描中的真实 mesh 碰撞**。柜体、支撑、安装与嵌入可以有合法的代理框重叠，不能据此删除 203 场景或者缩小家具。

另有 1,101 个 target bottom centers 在输入扫描地板 convex hull 之外。hull 是观测地板，不是完整房间墙界；当前 boundary_known=false/floor_known=false 诚实表达了这个限制。因 required boundary/support 全未知，即使模型逐字输出 GT，目标几何通过率仍是 0%。上一轮 tiny checkpoint 的该指标不能单独用于评价模型。

1,751 个 targets 的源 mobility_type=fixed / is_movable=False；其中存在 wall cabinets、sinks 等，当前被作为匿名新增请求而没有安装/支撑角色。目标总计含 336 ceiling lights、85 wall cabinets、72 sinks。不能直接断言这些对象都必须变成固定条件：若任务是安装它们，仍可以请求生成；但必须表达真实角色和验收条件。当前 corpus 没有完成这个任务定义。

`house_id=source scene_id` 正确隔离同一个扫描空间的 rescans；它不是已证实的整栋住宅 ID。208 样本可保留为**严格框标签的初始监督/过拟合集合**，不能标注成 208 个已验证可提交房间。

## 5. HSSD 与其他本地来源

外盘五个根目录都存在。以下 16 个 raw 家族是源文件库存口径，**不是替代已有 16 个训练 source 的新筛选方案**；大 JSON 结论来自有界样本读取，不代表全量 eligibility。已有 adapter 对 raw 字段的修正应先复用并复核。

HSSD 完整 CSV 有 56,647 对象：53,021 rigid、3,626 articulated；168 场景、2,351 room regions、15,694 assets。实际存在 GLB 与 URDF，并有 asset local_bbox_min/max/extent。53,007 个非空 OBB 中只有 **52,996 个严格有限且正尺寸**，11 个零尺寸；其余 3,640 没 OBB（3,626 articulated + 14 rigid）。objects.csv 的 1860 行 picture x 半长为 0，3875/3876 行 tablecloth y 半长为 0；不能加 epsilon 当有效标签。

`bbox_source=mesh` 53,662 行不等于有实例框：部分 articulated 记录继承资产 mesh 来源但 OBB 空。2,964 metadata boxes 中包含 2,956 doors/windows，raw 早期 base-at-origin 假设不能直接当标签；现有 HSSD adapter 已有测 opening offset 与重建部分实例的处理，应使用修正 IR 并核验新协议。逐资产 semantic front 和 joint-state 能力边界仍需明确。**HSSD 目前缺的是 v2 桥接，不是让旧修正管线全部从头再做。**

| 来源 | 实际字段/几何 | v2 派生数据需要的工作 |
|---|---|---|
| MultiScan | semantic OBB、half extents、center、axes/front/up、scan mesh | 保留已验证框；补角色、支撑、完整条件与适合代理框的验收策略 |
| HSSD | 实例 OBB、pivot/scale、asset local bounds、regions、GLB/URDF | 优先接入；恢复 front/up、开口 bbox offset、articulated geometry、房间关联 |
| SpatialLM | full size、center、yaw、墙/门窗/房高 | center→bottom、front、invalid wall rings、源 house/重复 grouping |
| InternScenes | center/full size、完整 ZXY angles；Gen 有 boundary_points | tilt、center→bottom、front；区分 Gen/Real2Sim floor 质量 |
| SAGE10k | full size、bottom pose、完整 Euler、raw place_id | 真实 pitch/roll；front、asset placement ID；约束厘米单位与 pose 泄漏 |
| IL3D | bbox、完整 USD rotation、floor mesh、flags/asset_record | pivot/scale/轴/front、源异常及跨源 provenance |
| BillLin66 3D-FRONT | floor mesh、room type、asset pivot、size×scale、quaternion | asset bbox offset/front；真实首记录 size/category 为空，按 mask 处理 |
| MansionWorld | floor/height、dimensions/poses、surface parent groups | 各路径 anchor/front、真实支撑 surface；过滤位置丰富的 condition text |
| SceneSmith | mesh extent/offset、DMD origin/full pose、room | centered/bottom origins 分支、front；不能统一减 h/2 |
| Structured3D | half coeffs(mm)、basis、centroid、room/openings | 2*coeffs/1000、raw axes/tilt；front/category；不沿用 heuristic yaw |
| SpatialGen | full extents、center/full transform、floor/ceiling | **官方 test-only 48 场景**；floor/ceil mm、box m，mirror/tilt/front/polygon |
| InteriorGS | 八 bbox corners、room/holes、geometric bottom | long-axis yaw modulo pi 不等于 semantic front；zero extent/归房 |
| 3RScan | centroid/full axesLengths/axes、relations/attributes | room partition、semantic front；可考虑部分几何监督 |
| ARKitScenes | raw obbAligned centroid/axes/full extents、tilt | 从 raw OBB 恢复 bottom，不能沿用 world-AABB bottom；room/front/visit split |
| SceneCAD/Scan2CAD | scan world AABB；CAD local half bbox/TRS | 两套框分开；CAD scale/tilt/front；scan shell 不是完整 room |
| OptiScene/3D-SynthPlace | cm bbox、source pose/asset ID、矩形房间代理 | frame 手性/yaw、pivot、坏/NaN 标签；旧 prompt 的 actual bbox 不进主 v2 condition |

当前 `data.py:49` 明确只支持 multiscan，CLI choices 也只有它。其他 15 个来源“未接入”不能理解成“无可用标签”。`audit.py` 首记录路径未识别 SAGE 顶层 furniture，输出空 key 是采样器漏读；真实首记录有 59 个对象。手工 source_catalog 说明不是动态全量审计证明。

## 6. 文献与 MinkowskiEngine

详细证据、固定 commit、官方链接、PDF 哈希与源码行号见 [文献复核记录](../outputs/fastfill_v2/review-20261005/papers/paper-code-verification.md)。定稿 §1.4/§11 的主要事实正确：

1. OptiScene 先检索资产 bbox，再做位置/方向文本预测；官方训练是 causal LM token SFT + 两轮 DPO。它没有证明隐藏 actual size 下的联合尺寸预测。[OptiScene v2](https://arxiv.org/pdf/2506.07570v2)
2. ATISS/ DiffuScene 都联合对象属性；ATISS 是自回归似然，DiffuScene 是噪声 MSE，不能统称 Hungarian set loss。DiffuScene 的 intersection 正则是对象之间的比较。[ATISS](https://proceedings.neurips.cc/paper/2021/file/64986d86a17424eeac96b08a6d519059-Paper.pdf)、[DiffuScene](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_DiffuScene_Denoising_Diffusion_Models_for_Generative_Indoor_Scene_Synthesis_CVPR_2024_paper.pdf)
3. CasLayout 支持固定家具清单、部分未知尺寸及 rearrangement；不能因为缺人工关系图就排除适配比较。[CasLayout v1](https://arxiv.org/html/2604.27361v1)
4. V-DETR detached matching、负 GIoU cost / 1-GIoU loss、GT-bin residual、proposal-relative L1 regression、ScanNet K=1、3DV-RPE 与 one-to-many 的区分均核对正确。论文回归相对初始 proposal；attention reference 每层更新。固定代码 SHA 为 `9062d75fe2c91e5d4a771b5325483fc330a3e827`。[criterion](https://github.com/V-DETR/V-DETR/blob/9062d75fe2c91e5d4a771b5325483fc330a3e827/criterion.py)、[transformer](https://github.com/V-DETR/V-DETR/blob/9062d75fe2c91e5d4a771b5325483fc330a3e827/models/vdetr_transformer.py)
5. 本地 MinkowskiEngine 是独立稀疏卷积库 checkout `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`，未构建或改 tracked 文件。FastFill v2 没有引入它。V-DETR 的点云依赖不属于 FastFill 训练要求。[V-DETR README](https://github.com/V-DETR/V-DETR/blob/9062d75fe2c91e5d4a771b5325483fc330a3e827/README.md)

文献措辞需纠正一处：§11.6 的 V-DETR v1 PDF 是本次从官方获取；本地用户 PDF 只有 OptiScene。若没有另一个用户文件及哈希，不应写“用户提供的 V-DETR PDF”。K=12、四项权重、Qwen 外部 decoder、请求身份限制和 WorldEdge 闭环仍是本方案设计，不是上述论文已验证的组合。

## 7. 在我们已有语料上重建的范围与执行顺序

**重建的是新的派生数据、规范资产 metadata 与 split manifests；harddisk 原始数据、旧 checkpoint 与已有 v1/v3 数据全部保留。**

1. 先修复明确代码错误，补反例回归。基础默认模型仍可 offline smoke；不能启用尚未修正的 Metal box path 或不均匀标签 DDP box path。
2. 使用 `.release/v3.2/ir` 与匹配 MANIFEST，保留原 16 个训练 source、2 个 held-out、3 个辅助数据的用途；新建 v2 bridge，复用已经完成的正确源适配与筛选。
3. 从可信 IR/source 构造新的 condition/target/validity/provenance。把待摆 size 从 condition 移到 target；已有 fixed geometry 的真实 size/pose 仍是合法条件。不能只删除旧 user.size：`scene.canonical` 的顺序与 ID 用 target footprint/size/position 排序，会继续泄漏目标信息，须生成不依赖 GT 的请求身份/顺序。
4. 继承旧 UID/group/alias 和已分配 split；严格 v2 eligibility 造成剔除时记录影响，不把 test 重分进 train。重新核验全局泄漏，增加 asset-disjoint 评测。旧厘米/整数度量化、untilt envelope、floor/support z-snap、bbox 推断 support 不能一律冒充真实局部尺寸/pose/语义前向；按对象字段追溯与掩码处理。Scan2CAD、斜墙 IL3D 窗与 SpatialGen 的已确认旧缺陷见 `docs/known-issues.md`，需要定向修正。
5. 先做 canonical mesh/OBB 抽检与小集合过拟合，再固定真实 Qwen/hardware/train config。正式训练按用户此前选择仍未启动；没有足够证据声称收益或可提交率改善。

旧 v1 模型的文本任务是 bbox-conditioned pose prediction；v3/v3.1/v3.2 是旧数据版本，不是本次结构化模型版本。旧 IR 中经独立审计的字段可以复用，旧 prompt/answer 不能仅改字段名直接变成隐藏 actual size 的主 v2 训练数据。

在仓库根目录可以重复小反例：

```bash
PYTHONPATH=. python outputs/fastfill_v2/review-20261005/model_review_repros.py
PYTHONPATH=. python outputs/fastfill_v2/review-20261005/runtime_repros.py
```

DDP 失败证据含完整命令，见 `outputs/fastfill_v2/review-20261005/ddp_box_mask_failure.txt`；无需再次触发崩溃才能接受该已捕获结果。初次复核时表内错误尚未修复；当前修正与测试状态见本文顶部“实施更新”。离线实现与数据验收仍不等于正式模型训练或真实资产系统验收。
