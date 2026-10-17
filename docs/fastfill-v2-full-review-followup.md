# FastFill v2：review2 全量审计的当前处置

更新日期：2026-10-06。用户提供的全量审计针对不可变 `FastFill_v2_20261006_review2`，其中“4 项待修”是该快照的有效历史结论。以下对应当前开发代码与新的 `multisource-20261006` 主数据；不改写历史反例、旧包、原始数据或 checkpoint。

当前主任务是完整房间条件、指定实例和空间要求，联合预测局部尺寸、底面中心和 yaw，最终接 RoomGenBench 的 text+bbox 到 mesh 路径。历史 SpatialLM 最小 XY pilot 及多源 `reference_extent` 视图是独立消融。

## C1–C4、D1–D2

| 项目 | 处置证据 | 当前状态 |
|---|---|---|
| C1 尾累积窗口缩小 | `train.py` 在 clip 前按 K / 实际 microbatch 数修正尾窗；真实入口、两 rank Gloo、空监督窗及 CPU GradScaler 回归 | 已修复。累积策略仍是 microbatch 均值的平均，不是整个窗口全对象加权均值 |
| C2 禁用 loss 的空监督更新 | objective-aware preflight；整个累积窗口、所有 rank 的启用目标计数；全空时一致跳过，数值 loss=0 但有标签时仍更新 | 已修复。review3 专项记录保留原反例 |
| C3 文本与 ID 混淆 | 只置换对象 ID、support_parent 和明确的关系引用；普通 category/description/属性、key、surface ID、组名保留 | 已修复；同类不同约束角色继续禁止交换 |
| C4 资产闭环失败 exit 0 | predict 保存失败诊断后 exit 2，入口 `raise SystemExit(main())`；实际 CLI 子进程回归 | 已修复；纯预测成功、离线闭环成功 exit 0 |
| D1 Scan2CAD 估计地板与贴齐 | 新主数据 12 场景显式 `floor_known=false`、estimated-floor/upstream-snap provenance；140 对象中 133 个原全有效 position 向量屏蔽，另 7 个原已无效 | 已完成保守资格处理；未恢复冻结 IR 不存在的逐对象 pre-snap z，也未认证实际地板 |
| D2 13 个不可表达 size | 保留原值，将 10 train / 1 validation / 2 test 的完整 size 向量屏蔽；配置相关 preflight 仍检查有效可学习目标范围 | 已完成默认配置资格处理；不把纳米级尺寸解释为已核 mesh 真值 |

D1/D2 后，19 个原可交换组的 103 个成员回退固定身份，避免使用缺少完整 position/size 标签的 matching cost。全部变更 261 条进入 journal；场景数、ID、split、请求顺序、target 数值、支撑与关系保留。当前有效 P / S / semantic yaw 为 1,732,188 / 1,224,657 / 581。

旧 `legacy_bridge` 仍能重现历史父语料；直接运行旧 `data --source selected-v3.2` 不会自动生成这个资格修订。当前主版本通过 `qualified_data` 派生，训练应绑定它的 manifest 和实际 tokenizer 资格视图，不能混用历史默认构建命令。

## 数据、服务器与实验边界

主集 124,589 / 8,137 / 8,615 场景；真实 Qwen tokenizer 资格 124,375 / 8,125 / 8,602，全部拒绝都有台账且未截断。16 个训练 source tags 保留；SceneSmith/SpatialGen 仍 test-only。523 个训练 semantic yaw 只来自 58 个场景，大样本量不能代替朝向监督覆盖。

服务器环境、固定 Qwen3-8B、完整预检、单卡四更新和七卡一次更新 pilot、保存／重载已完成。完整三轮基线尚未启动。详细证据与配置见 [多源完成记录](fastfill-v2-multisource-20261006.md)，当前代码的新问题和本轮回归见 [设计与实现审核](fastfill-v2-design-audit-20261006.md)。

这些处置建立了可追溯的字段监督数据，不证明全部 GT 经过物理验收。完整 65 场景 cohort 的历史严格框级通过为 0/65，原因含未知边界／支撑与 OBB 相交；它不能改称 65 个可提交房间，也不能据此删除全部参考扫描。

## 历史证据

review2 的 141,341 场景全量原标签诊断、37,990,247 候选对象对和来源追溯结果保持各自范围。C1/C3/C4 反例与修复证据位于 `outputs/fastfill_v2/accumulation-tail-audit-20261006`、`direct-bbox-20261006/test-evidence`；C2 见 [review3 专项](fastfill-v2-review3-objective.md)。D1/D2 和主版本全量逐行证据为 `outputs/fastfill_v2/multisource-20261006/verification-main-final.json`。

历史 640 / 659 / 653 测试数不能自动替换成当前测试数。服务器此前 873 passed + 114 subtests / 6 skips 与 87.63% statement coverage 是对应已钉代码的记录；本轮修改后的结果应使用新的测试和发布收据。
