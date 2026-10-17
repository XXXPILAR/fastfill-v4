# FastFill v2 多源数据、服务器与下游核验记录

日期：2026-10-06。主任务为房间、指定对象与空间条件输入，一次预测每个请求对象的目标局部全尺寸、底面中心和 yaw，再交给资产解析和实际几何验收。以下记录说明已完成的工作及证据边界，不声称正式全量模型已训练或真实 Host 已接通。

## 房间数为什么变化

| 阶段 | Train | Dev / Validation | Test | 合计 |
|---|---:|---:|---:|---:|
| v3 / v3.1 原文件 | 144,140 | 8,167 | 8,647 | 160,954 |
| v3.2 修正后原文件 | 144,150 | 8,167 | 8,647 | 160,964 |
| 继承旧 trainer 默认 flags 过滤后 | 124,843 | 8,167 | 8,647 | 141,657 |
| v2 协议与有效监督资格后 | 124,589 | 8,137 | 8,615 | 141,341 |
| 真实 Qwen tokenizer / 128 对象 / 8192 context 资格后 | 124,375 | 8,125 | 8,602 | 141,102 |

v3.2 恢复 InteriorGS 房屋 `0673_841492` 的 10 个训练房间，没有旧 split 迁移。19,307 个训练场景命中旧默认 `oob_objects`、`fixed_collision`、`overlapping_furniture` 的并集；三个 flag 独立计数为 9,820、10,749、1,799，存在重叠，不能直接相加。它们是历史框代理质量选择，不能逐条解释为 mesh / physics 已证实错误。

旧 v3 / v3.1 文件虽有 144,140 行，默认过滤后、分词前的训练候选是 124,836；当前 v2 的 124,589 相比它少 247。某次历史运行真正用了多少，仍须核对当时参数、source weights 和 tokenizer 日志。此处没有把默认策略认定为每个历史 run 的实际曝光。

v2 另拒绝 316 间：倾斜固定几何 185、斜向窗缺少来源修正 111、零可靠监督 16、非法地板多边形 4；split 为 train 254 / validation 30 / test 32。新多源 builder 保留父包全部 141,341 间；D1 / D2 是字段资格降级，没有删除房间。真实 tokenizer 又明确拒绝 239 间，全部是对象或 context 预算，不静默截断。

完整逐 UID 计数与拒绝台账见 [lineage-counts.json](../outputs/fastfill_v2/multisource-20261006/lineage-counts.json)。旧 19,307 间仍保存在原始版本；`legacy_build --include-flagged` 可以另建候选视图，需重新核验转换失败、掩码与 tokenizer。它们不是连续 heads 必须删除的集合，也不能未经复核整体认定为合格。当前记录没有恢复重建这批候选。

## 已完成的数据

主包 [data](../outputs/fastfill_v2/multisource-20261006/data) 为 `full_condition_multisource_main`，包含 1,787,052 个请求对象、392,099 个固定对象，保留原房间条件、ID / 顺序 / split、支撑与 63,668 条稀疏显式关系。条件不读取未来目标几何或资产 ID。

延续 Excel 中原选定的 16 个数据集家族体系：11 个产生训练房间的家族展开为 16 个 train source tags；SceneSmith / SpatialGen 为 test-only；另三个家族提供房间或分组谱系。SpatialLM 是其中一个训练来源，不是全量主数据的替代物。三字段 [data-minimal-reference](../outputs/fastfill_v2/multisource-20261006/data-minimal-reference) 是独立输入消融。

主数据数值保持父记录原样，变更记录共 261 条：12 个 Scan2CAD 场景地板改为估计来源 / unknown，133 个原全有效 position 向量降级；13 个超出默认 size 输出范围的完整 size 向量降级；19 个受影响交换组的 103 个成员回退固定身份。主数据有效 P / S / semantic yaw 对象分别为 1,732,188 / 1,224,657 / 581。主数据没有提升 SpatialLM 几何轴 yaw。

独立全量校验 [verification-main-final.json](../outputs/fastfill_v2/multisource-20261006/verification-main-final.json) 与 [verification-minimal-final.json](../outputs/fastfill_v2/multisource-20261006/verification-minimal-final.json) 均 `ok=true`，覆盖全部场景和对象。12,509,364 是检查的目标数值坐标槽总量，包含被屏蔽或未知槽，不是全部有效监督坐标数。

主 manifest SHA256：`9eb3a9314f9a8537ff75c1b6d8758c01400a67fddd1c814d7a500a6e19411871`。主 train / validation / test SHA256：

```text
39e9ef5a0a3e79a394b1f9f1bd40416eda9527e191fa350a1759b3e2177f1e55
9cfaa187af7fbcc3ed4d4ad79c4db53ff7dbed9ef188698348ccd953e9dd04f3
59809f7629bb89e54b863b9b90323619a66b82d7bf011e0376e855dc870a2f2f
```

[eligible-local](../outputs/fastfill_v2/multisource-20261006/eligible-local) 是根据服务器全部 239 条排除台账，从本地主包按原始行复制的实际入训资格视图；三个文件 SHA 与服务器 tokenizer 生成的文件完全相同。它没有冒称重新执行 tokenizer。源文件与旧 checkpoint 均保留。

## bbox 输出与 V-DETR 的对应

“OptiScene + V-DETR”准确描述了任务与训练机制的来源：OptiScene 的房间 / 请求条件保留，未来资产尺寸从输入转为输出；V-DETR 提供 detached assignment 后连续框回归、角度分类及 GT-bin residual 的参考。当前模型是 Qwen 条件编码器加外部双向对象 decoder，输出 size / bottom-center / yaw。

固定 V-DETR commit `9062d75fe2c91e5d4a771b5325483fc330a3e827` 的 `criterion.py` 本轮重新获取并核对，SHA256 `dcb7c3a53e9254ad91c6ce4a157d5a77dec672f5ce717d8b116581343a209784`。官方使用 proposal-relative center / log-size L1、yaw CE + GT-bin Huber，以及 1-GIoU；本方案使用 room-normalized bottom-center SmoothL1、预测 / 目标 log-size ratio SmoothL1、yaw CE + GT-bin residual SmoothL1。Hungarian 仅在有完整 cost 标签的合法可交换组内使用，固定角色直接对应。

主配置四个基础 loss 权重均为 1；box / collision / boundary 为 0。已有可选 box 是 BEV oriented convex-hull GIoU，不是完整 V-DETR 3D 检测器或 3D GIoU 复现。JSON 序列化不参与梯度。已知请求数不需要 objectness、检测类别预测或 NMS。

## 服务器实际结果

服务器：`yxd-dev`，包目录 `/home/jovyan/shanliantian/FastFill_v2_multisource_20261006`。模型：固定 Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`。环境：torch 2.13.0+cu126、transformers 5.14.1、peft 0.21.0、accelerate 1.15.0。前次 pilot 快照核验 162 个代码文件及六个主数据文件；本轮权威代码更新为 `project-current-audit-20261006`，与 Git 实现提交 `435d4e6532bdbc24d130df5edc11c5aa002a0460` 绑定；大数据采用同一已钉 hash 的父包在服务器重建，所有输出逐文件 SHA 与本地已追溯 IR 的版本相同，非盲目复用旧数据。

真实 tokenizer 全量 preflight 已完成，16 个 train tags 全部保留，heldout 来源没有进入训练。实际可学 train labels：position 1,495,767、size 1,063,882、semantic yaw 523。所有 58 个 train yaw 场景均保留。预检报告、配置及排除台账见 [server-evidence.json](../outputs/fastfill_v2/multisource-20261006/server-evidence.json)。

真实 Qwen 单卡四更新 pilot 使用 49 场景、16 个训练来源，25 个 microbatch，累积窗口 8 / 8 / 8 / 1；实际曝光 P 895 / S 736 / yaw 3，所有对应模块在有监督窗口获得非零梯度，save / reload 预测差为 0。同环境完整测试为 873 passed、114 subtests passed、6 环境 skips；v2 语句覆盖 87.63%。这些是代码运行证据，不是模型质量百分比。

[固定训练样本前后比较](../outputs/fastfill_v2/multisource-20261006/pilot-fixed-sample-comparison.json) 为只读推理：按原 config、库版本与初始化顺序重建 seed42 baseline，和已有四步 checkpoint 比较同一 MultiScan 三对象样本。总 loss 2.86292 → 2.41201，四项 head loss 全下降；平均位置误差 6.16088 m → 3.74547 m、log-size 0.58069 → 0.51352、yaw 1.72268 → 1.68800 rad，BEV IoU 仍为 0。baseline 不是保存的训练前 checkpoint；单条训练样本、四步改善不能证明泛化或场景可用。

按用户说明检查 GPU 1–7，服务器检查识别为合作式占位进程；测试期间自动释放，本流程未终止占位进程，GPU0 的 VLLM 保留。[七卡 NCCL 收据](../outputs/fastfill_v2/multisource-20261006/gpu7-nccl-readiness.json) 证明七 rank FP32 / BF16 collectives 正确且每卡释放后约 138.1 GiB 空闲。

七卡真实 Qwen / LoRA 的 B2-K8 与 B1-K16 均完成一次 49 场景生产更新，LoRA、decoder、四 heads 梯度非零。B2 曾出现 allocator OOM / retry 警告后成功；B1 日志未出现 OOM 警告。B1 是更保守的候选，短程成功不认证所有 context 的最坏显存。模型加载提示 `lm_head.weight` 为 unexpected 是因为加载 Qwen 条件 `AutoModel`，此路径不使用语言输出 head。

七 rank 实际 prepared loader 已检查 epoch 数、padding、稀有监督、全局空窗及不足额尾窗。B2-K8 三轮为 3,333 次更新，而单卡 B2-K8 的 23,322 次不能直接用于七卡，否则约 21 轮。B1-K16 每 rank 每轮 17,768 microbatches / 1,111 updates，尾窗 8，三轮 3,333 updates；每轮恰好 124,375 唯一场景及一条 padding 重复，yaw 标签每轮 523，有 yaw 的更新窗口分别为 56 / 57 / 57。validation 的 padding 为两条，批均值诊断仍计入 padding。完整窗口同为 112 场景，但每 microbatch 的全局有效对象均值分母不同，稀有 yaw 在窗口中的权重也可能不同，不能称梯度等价。日志普通 loss / counts 取最后一个 microbatch；窗口 active count 和单独曝光台账才覆盖整窗。

候选配置：[training-3epochs-world7-B1K16.json](../outputs/fastfill_v2/multisource-20261006/readiness7-B1K16/training-3epochs-world7-B1K16.json)，SHA256 `834654b3895a703e75213db2a692b3e385a5d79b54767802e1c6387599b7dbcd`。服务器同路径的 `readiness7-B1K16/` 保存实际七 rank 收据。仅在决定启动完整基线时执行：

```bash
cd /home/jovyan/shanliantian/FastFill_v2_multisource_20261006/project-current-audit-20261006
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  /home/jovyan/shanliantian/FastFill_v2_20261006_server/env/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=7 --module fastfill.v2.train \
  --config ../readiness7-B1K16/training-3epochs-world7-B1K16.json \
  --data ../preflight-main/eligible/train.jsonl \
  --validation ../preflight-main/eligible/validation.jsonl \
  --output ../run-main-world7-B1K16-3epochs
```

该命令在本轮没有执行。不要使用 test 选模型，不覆盖既有输出。固定样本比较和有界七卡更新见 [world7-pilots-receipt.json](../outputs/fastfill_v2/multisource-20261006/world7-pilots-receipt.json)。

## 下游与评测边界

实际 RoomGenBench 是 bbox + 每对象文本 → mesh 方法 → 统一装配。轴转换的内存实测通过：局部尺寸轴置换、yaw−90°，bottom-center 保持一致。详见 [实际接口审核](fastfill-v2-roomgenbench-interface-20261006.md)。原共享 bench 只接受预注册场景；本轮 FastFill 动态装配入口已接任意 handoff／独立 output，完整 condition、已声明 support、fixed／constraints 和 unknown flags 均保留，未知高度明确为 display reference。生成方法的实际 runner／权重、mesh 支撑核验仍需接入。其 anisotropic mesh-fit 和 placeholder 展示不是 FastFill 真实资产不任意缩放、失败整场不提交的验收合同。

新增 [constrained-evaluation](../outputs/fastfill_v2/multisource-20261006/constrained-evaluation) 仅为 heldout reference-derived positive NEAR 视图：validation 6,776、test 7,193，共 13,969；没有 train 文件。按完整有效中心标签与固定近邻范围筛选，并加一条硬 near；生产 near 指标是 footprint gap，中心范围仅是充分的参考选择。新关系导致 6,185 场景中的 7,663 交换组 / 29,201 成员回退固定身份。对照有 / 无 near 时须保持同一身份降级，避免额外混淆；该视图不认证 faces / support / 所有约束能力。它尚未执行真实 tokenizer 资格或模型评测，不算额外训练数据。

训练 yaw 仍仅 523 对象 / 58 场景。均匀三轮的计划曝光为 1,569 yaw 标签，不说明朝向已学会；需要单独报告各 head 和有效窗口、必要时另冻有追溯的采样协议。现 CLI 保存 Accelerate state，但尚无自动 optimizer resume。选定 BF16 路径已核验；本轮另修复 FP16 overflow 误计成功更新，并补 CPU GradScaler 入口回归，未冒称 Qwen FP16 GPU pilot。

当前可以开展完整主数据的有记录基线研究实验；不据此宣布布局质量、真实资产 / mesh / physics / Solver / 持久 Host 验收完成。当前实际 checker / WorldEdge Host adapter 等仍有实现缺口，详见接口审核与 [训练流水线](fastfill-v2-training-pipeline.md)。本轮只完成有界 pilot，没有留下正式全量训练进程。

数据路径：本地 `outputs/fastfill_v2/multisource-20261006`；外盘独立新包 `/Volumes/harddisk/FastFill_v2_multisource_20261006`；服务器同名新包。review2 / review3、原 IR、历史版本和旧 checkpoint 均未覆盖。

## 当前发布与清理

代码已发布至私有 GitHub `Slian22/fastfill`，实现提交 `435d4e6532bdbc24d130df5edc11c5aa002a0460`。私有 HF 数据 revision 为 `ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88`；canonical24文件及47个发布文件的远端大小／SHA均一致。下载入口见 [数据发布用法](fastfill-v2-dataset-release.md)。历史 HEAD 文件已移除，旧固定 revision 仍可读取。

服务器清理删除了废弃的 review3／单源 pilot 解包数据、重复传输包、探针与未完成上传，保留 raw sources、环境、冻结档案、checkpoint 和当前主／资格／消融／NEAR 视图。净删除逻辑量为 **1,717,202,511 bytes**，最终缺席目标12个；三份矩形消融曾为清理候选，随后为canonical发布从原JSONL原样筛选恢复，SHA完全一致且现保留。共享JuiceFS的free计数没有同步变化，不能把du估计当成实际磁盘回收量。逐路径收据见 [清理证据](evidence/fastfill-v2-audit-20261006/server-cleanup-receipt.json)。

本轮最终完整本地回归为 **903 passed、114 subtests passed**；v2独立覆盖运行765 passed，statement coverage **88.35%**。服务器当前完整套件897 passed／114 subtests／6 Metal skips；v2服务器statement coverage为87.71%，合并旧v1＋v2为76.71%，范围不同。这不是模型质量通过率，当前同步和论文源码范围以 [本轮交付收据](fastfill-v2-release-20261006.md) 为准。

参考源码及三份论文PDF已发布服务器 `references-20261006`：1772/1772导出文件SHA匹配，1755源码／文本与固定Git blob一致。它是源码／论文参考包，没有编译MinkowskiEngine、运行原detector或安装RoomGenBench生成方法权重。当前FastFill v2不依赖ME。
