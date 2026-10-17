# FastFill v2 review2：修复与重新冻结

**历史快照记录。** 以下“当前／本次”均指 review2 冻结时。后来独立全审的 C1–C4、D1–D2 及当前处置见 [后续处置](fastfill-v2-full-review-followup.md)；当前服务器与数据见 [多源完成记录](fastfill-v2-multisource-20261006.md)。旧测试、hash 和未执行状态不改写为本轮结果。

日期：2026-10-06。对象为当前独立 `fastfill/v2` 实现及新派生数据。旧训练包 SHA256 `4b643eb5f593cafbe6c45b962a91d35630882b868bcca87acd4d9832fa6e929a` 对应历史快照，保持不变；不要将它与本版修复结论混用。

## 1. 用户审核是否正确

用户提供的 1 个 P1 和 3 个 P2 均有有效反例。测试通过并没有覆盖这些边界，因此旧包不能宣称全部实现问题已修复。

| 问题 | 当前修复 | 回归证据 |
|---|---|---|
| DDP 可选框项混合 dtype | `_record` 与 `_validation` collective 统一 float32，backward 保留原 loss 张量 | 真实双进程 Accelerator/Gloo：一个 rank 无框、数量不同、全部无框及 collision 空项；真实日志和 validation 路径数值核验 |
| hard `on` 与 `support_parent` collision 不一致 | regularizer 复用 `effective_support_requests`，soft on 不做支撑豁免 | 同几何的 loss/梯度相同；target-target、target-fixed、冲突和非支撑碰撞 |
| 固定物体能力被忽略 | 固定物体在目标/实际阶段均检查声明的能力；已知缺失 fail，未知 hard unknown | 能力满足/缺失/未知及 Host 不部分提交 |
| 地板下界遗漏 | 新增对象独立检查可信 floor 下界，不由物体支撑声明覆盖 | floor/物体支撑、地下支撑、容差及失败不提交 |

既有墙、门等几何可能合法穿过地板边界，不能将新增对象的下界规则无差别套到全部固定结构。固定物体不能为地下新增物体提供绕过房间边界的通道。

反例与 RED→GREEN 证据位于当前仓库 `outputs/fastfill_v2/review-20261006/`，上传包中在 `audit/review2/`。这是当前修复的审核结果，不改写用户对旧压缩包的报告。

## 2. 额外确认并修复的问题

| 范围 | 修复内容 |
|---|---|
| 训练记录 | validation 改为更新后运行，与 step checkpoint 一致；训练输入读取前/后和结束时校验 fingerprint，记录 validation hash |
| 数值与配置 | 长上下文 slot pooling 至少 float32；collision 体积以 double 计算，Metal 保留 CPU 梯度转换；匹配布尔项/权重严格验证 |
| 文本训练 | 全 cohort 完整 context preflight 在模型加载前，禁止中途随机遇到超长行；输入 fingerprint 绑定实际训练数据 |
| 实际资产验收 | 验证支撑面高度/足迹元数据；显式语义前向硬要求无证据时 unknown；失败请求的生成延迟计入全部请求 |
| 原有测试兼容 | v1 build 测试按目录组件排除审核/测试目录，合法 `v2/review_data.py` 仍进入代码 hash；v1 生产实现未变 |

另外用 Shapely 对照 200 组非退化 BEV IoU/GIoU，最大误差 `3.33e-15`，没有差异。该对照补充框算子证据，不代替梯度和真实 mesh 验收。

## 3. 数据修订及逐行保持范围

新数据写入 `/Volumes/harddisk/FastFill_v2_20261006_review2/data/selected-v3.2/`。来源仍为我们已选的 16 个家族及冻结 v3.2 原划分；不下载新语料，不写原始源数据。

review2 撤销 14,101 场景中的 **14,739 条**主体 GT yaw 无效或来源证据含 `front_unknown_reason` 的旧 source-derived `faces`。原约束和撤销理由写入 provenance。明确用户要求不删除。全部 UID、split、请求对象、target、validity 和几何数值保持，父数据五文件 hash 前后核验；原 316 条拒收记录逐字节保留。

完整标签子集从新父数据重新选取，仍为 **57 train / 2 validation / 6 test** 场景、572 个对象，逐行保持新父数据内容。主集仍为 124,589 / 8,137 / 8,615，共 1,787,052 个对象。

35,095 个可信位置目标、12,209 个场景具有低于所声明可信地板超过 `1e-4 m` 的 bbox 底面诊断。其中训练集为 28,231 个对象、10,469 个场景。仅记录诊断，不修改标签、删除对象或将地板改为 unknown。独立抽查的原始地板 mesh 和 bbox 表明：部分负值已经存在于原来源，SceneSmith 个别毫米级负值来自原 physics 位姿。不能把该诊断直接解释为所有 mesh 穿透，也不能将它忽略后宣称全库几何可提交。

默认 128 对象预算下，206 train / 11 validation / 11 test 场景超预算；训练 preflight 拒绝整场景并记录，评测保留失败分母。context 的实际拒绝率需使用正式 Qwen tokenizer 和选定预算计算。

全量派生核验检查每条数据是否精确等于获准政策的父行转换，同时验证 UID、split、输出 hash 和计数。父数据协议/split 的零错误与新 revision 的一致性，均不代表全部来源的语义前向、支撑、mesh、物理真值得到验证。

## 4. 论文和 RoomGenBench 复核

OptiScene 的具体资产 bbox 是输入，官方训练为 token SFT 与后续 DPO；FastFill 主任务隐藏未来资产尺寸并用连续 heads，不能借它的数字证明本模型有效。ATISS/ DiffuScene 支持联合对象属性表示，但前者是自回归似然，后者是扩散目标，不能统一称 DETR loss。

V-DETR matcher/criterion 的离散对应、尺度回归和 GT-bin residual 可参考；检测 objectness、one-to-many、点云网络和 MinkowskiEngine 均没有移植。另在 V-DETR 可选 DIoU 路径确认了将 BEV `[x,y,w,h,yaw]` 的前三项用于中心距离的问题；本系统没有复制这条路径，默认 BEV GIoU 不受其影响。

RoomGenBench 本地 commit `30f2e059d046ccd4aa928d0109a09a52f11693b5` 是给定布局到 room mesh 的对照。用户截图中的 input/reference 使用已有布局；普通方法再强制 fit mesh 到目标框。它可作为显示效果目标和资产对照参考，不能替代 FastFill 布局训练，也不能用于宣称未经拉伸的资产尺寸适配。

本次核对的固定官方代码、论文主张和本地 RoomGenBench 文件见包内 `audit/review2/papers-reaudit.md` 与 `roomgenbench-comparison.md`。论文访问/源码检查不等于已运行或复现论文训练。

## 5. 训练前仍需固定的内容

训练集可信 yaw 只有 523 个对象、跨 58 个 MultiScan 场景；完整验证只有两场景。现有均匀 sampler 不保证稀有 yaw 充分曝光。正式训练前应固定采样/数据资格、预算和分来源评测，不能以大主集数量代替完整监督覆盖。

用户已选择 GPU 服务器，硬件和 Qwen 路径待提供。先进行真实 Qwen/CUDA 有界 pilot，验证 LoRA、梯度、显存和 checkpoint，然后开展正式对照。当前没有真实 Qwen 质量结果，没有实际资产 mesh/physics/Solver 或真实 WorldEdge Host 验收。

执行命令见 [RUNBOOK](fastfill-v2-runbook.md)；每个 optimizer step、产物和完整房间路径见 [pipeline 说明](fastfill-v2-training-pipeline.md)。当前全套 **640 tests + 114 subtests passed**，v2 覆盖率 **87.55%**（测试包括 v1/v2，覆盖率仅计算 v2）。确切日志与覆盖率以包内 `RUNBOOK.json`、`audit/review2/pytest-frozen.txt`、`coverage-frozen.json` 为准。原包的 455 项通过/86.85% 是原快照记录，不能自动套到本版。
