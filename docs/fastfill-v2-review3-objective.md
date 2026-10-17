# FastFill v2 review3：启用目标与累积窗口

**历史专项记录。** 本页只解释 C2 与 review3 的当时运行；C1/C3/C4、当前多源 D1/D2 资格与最新服务器证据见 [后续处置](fastfill-v2-full-review-followup.md) 和 [多源完成记录](fastfill-v2-multisource-20261006.md)。本文的历史代码／数据数量和测试数保留原范围。

日期：2026-10-06。本次只处理 review2 独立复审确认的非默认配置 P2，完善新服务器执行步骤。review2 的包身份、640 tests 与数据修订记录保留为历史事实；本版不覆盖旧包。

## 复审判断

用户反例成立：review2 的 `_preflight` 不读取 `LossConfig`；yaw-only 配置仍保留只有 position/size 标签的样本。真实入口中 loss 与所有梯度均为零，AdamW 仍执行 weight decay，记录了一个不存在 yaw 学习信号的更新。

这个反例不阻塞 review2 默认四项全开的完整标签 pilot，但会污染 yaw-only 等消融的更新次数和训练曝光。不能以“loss 等于零”判断是否有监督：预测正确时也可能是零。

## 本版修复契约

1. 预检和 criterion 共享实际启用目标的资格规则，考虑字段 validity、固定坐标、BEV 标签资格及 scene regularizer 的真实项；没有任何启用目标的样本明确拒绝。
2. 累积整个窗口的有效目标数；同步边界用 int64 汇总全部 rank。整个窗口全空时不执行 AdamW，不递增更新步数。
3. 前一个 microbatch 有有效目标、最后一个没有时保留已累积梯度；某 rank 为空、其他 rank 有目标时仍参加全局 backward/更新。
4. 有有效目标但数值 loss 正好为零时仍允许正常更新；没有目标的窗口清零，完全无更新的 epoch 明确退出，避免死循环。
5. 日志区分窗口有效目标与最后 microbatch 的几何诊断，记录跳过的空窗口；更新后的 validation 与 checkpoint 约定保留。

默认四项 loss、预测参数化和 dataset 标签不变。无目标标签的 boundary/collision-only 训练仍可有合法正则项，不能因为没有 GT 标签而一律拒绝；支撑豁免的无碰撞项也不能冒充有效监督。

## 数据及旧文件

训练数据完整沿用 review2：141,341 场景、1,787,052 对象；完整 cohort 仍为 57/2/6。主目录五文件和 cohort 四文件与已校验的 review2 副本逐项 hash 比较，不重新划分或更改几何。

本次外盘 `/Volumes/harddisk` 未挂载。使用本地独立审核解包副本，在重新执行其完整性校验后生成新包到仓库 `outputs/fastfill_v2/FastFill_v2_20261006_review3/`，没有在 `/Volumes` 创建伪造盘目录。新包不意味着原外盘数据消失；重新挂载后可复制新包到外盘新目录。

## 服务器与放行范围

用户提供 8 张 H20Z、driver 570.124.06、2.8 TiB 主机 RAM；允许使用 GPU 1–7。冻结 review3 时尚未执行服务器回归；随后在独立 Conda 环境使用物理 GPU 1、PyTorch 2.13.0+cu126，完成 Linux 回归和真实 Qwen3-8B BF16 的 20 步 pilot，保留原任务。具体结果见 [服务器启动手册](fastfill-v2-server-start.md)。

review3 冻结当时，v2 尚未 commit/push 或上传服务器/HF。随后实现已同步至私有 GitHub `Slian22/fastfill`（`071fc3486329ce87fb41d29985b96c9931fbde4c`），数据发布至私有 HF [liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2)，服务器已核验冻结包与 Qwen3-8B 权重。训练实际状态见 [服务器启动手册](fastfill-v2-server-start.md)，不以发布或完整性校验替代模型验证。

新增 **19** 项回归通过；Mac 完整套件 **659 tests + 114 subtests passed**，v2 覆盖率 **87.68%**。随后 Linux 服务器为 **653 tests + 114 subtests passed**、6 个 Metal 专用测试跳过，statement coverage **87.65%**。父数据 hash 和新归档核验的确切结果记录在包内 `RUNBOOK.json` 与 `audit/objective-review/`，服务器测试后 279 个冻结文件仍全匹配。20 步真实 pilot 已验证训练与保存/载入链路，六个测试场景的严格目标几何验收为 0/6。这些结果不证明稀有 yaw 足够、主语料全部物理合法或真实资产/Host/GLB 已验收。正式训练仍需冻结采样与预算。
