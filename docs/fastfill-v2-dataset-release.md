# FastFill v2 数据发布与训练用法

## 当前主数据：多来源完整条件

当前主数据是 `multisource-20261006/data`，124,589 / 8,137 / 8,615 场景，1,787,052 个请求对象。真实 Qwen tokenizer 资格视图为 124,375 / 8,125 / 8,602，位于服务器 `preflight-main/eligible` 和本地 `eligible-local`。新主版本保留完整条件，处理 D1/D2 资格，保留 581 个有效语义 yaw；来源、精确 hash 与完成证据见 [多源记录](fastfill-v2-multisource-20261006.md)。

当前数据固定 HF revision 为 **`ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88`**，配套实现提交为 `435d4e6532bdbc24d130df5edc11c5aa002a0460`。主数据、实际资格视图、简化输入消融和 heldout NEAR 视图同置于 `multisource-20261006`。canonical 数据共 24 文件、6,277,486,224 bytes；连同审计、清单和 dataset card，47 个远端文件的大小／SHA 全部匹配。发布后仓库仍为 private。历史 review3 与 SpatialLM pilot 已从 HEAD 移除，固定旧 revision 保留。

NEAR 和简化视图各自的真实 tokenizer 预检／模型评测尚未完成。主资格 `eligible/` 才是当前已完成真实 tokenizer preflight 的训练视图。SHA256SUMS 列 46 项，包括根 README 和发布 manifest；checksum 自身另以远端验收核对。完整发布证据见 [本轮发布与服务器收据](fastfill-v2-release-20261006.md)。

```bash
export FASTFILL_DATASET_ROOT=/path/to/fastfill-v2-current
hf download liantian/fastfill-v2 --repo-type dataset \
  --revision ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88 \
  --local-dir "$FASTFILL_DATASET_ROOT"
cd "$FASTFILL_DATASET_ROOT"
sha256sum -c multisource-20261006/SHA256SUMS
export FASTFILL_MAIN_DATA="$FASTFILL_DATASET_ROOT/multisource-20261006/eligible"
```

访问方式（2026-10-07 更新）：按用户决定，`liantian/fastfill-v2` 改为 **public + gated（gated=manual，人工审批）**。仓库页面公开可见；下载前需在 HF 页面申请，经人工批准后用已登录的 HF 账号下载，命令不变。本文件其余各处的 private 字样是对应发布当时的状态。本轮（2026-10-07）未访问 HF，以上设置没有在本文件内做远端核验。原数据来源许可和访问限制仍然有效，gated 公开不额外授予再分发或商业使用权。服务器已保存相同主数据与资格文件时直接使用其已验 SHA 路径，无需重复下载。

第二轮（2026-10-07）的数据契约变更（K1 盒对称 `size_axis_swap_allowed`、K2 房高冲突只标记、K3 地面支撑只取源标注）需要本地重建后才进入训练数据；重建数据尚未发布到 HF。旧行仍可读取（`size_axis_swap_allowed` 迁移为全 false）。重建与正式训练命令见 [v2 README 的 2026-10-07 round 2 一节](../fastfill/v2/README.md#2026-10-07-round-2)。

## 历史 SpatialLM pilot：最小 XY 输入 → bbox 布局

历史 **direct-bbox-20261006** 从冻结语料按 SpatialLM 白名单派生，输入只有 room type、XY room size、furniture list。其 train/validation/test 为 **9,601 / 539 / 624** 场景，共 **33,545** 对象。yaw 是 π 周期框轴，不认证语义前向；固定对应，无重建的可交换组。模型条件 H=null，源房高仅用于资格/provenance。来源白名单不等于其他数据集均未通过几何资格。

同一 **private** 仓库的新固定 revision：**35f5272330d37771eea2d11925c42aeec9d917d4**。20 个新上传文件的远端 size/SHA256 全匹配，匿名读取 HTTP 401，16 个历史 review3 快照文件不变。本地最终目录是 outputs/fastfill_v2/direct-bbox-20261006/data-xy-final；服务器在 /home/jovyan/shanliantian/FastFill_v2_direct_bbox_20261006 新建独立目录。规则、监督和下游兼容边界见 [直接 bbox 手册](fastfill-v2-direct-bbox.md)。

```bash
hf auth login
export FASTFILL_DATASET_ROOT=/home/jovyan/shanliantian/data/fastfill-v2
hf download liantian/fastfill-v2 --repo-type dataset \
  --revision 35f5272330d37771eea2d11925c42aeec9d917d4 \
  --include 'direct-bbox-20261006/*' --local-dir "$FASTFILL_DATASET_ROOT"
cd "$FASTFILL_DATASET_ROOT/direct-bbox-20261006"
sha256sum -c SHA256SUMS
export FASTFILL_BBOX_DATA="$FASTFILL_DATASET_ROOT/direct-bbox-20261006/data"
```

从当前代码仓库根目录运行，每次输出选不存在的新目录。服务器已传入同一数据时改 FASTFILL_BBOX_DATA 指向已校验 data，避免重复下载。凭证不要写入脚本或日志。

```bash
# 离线 smoke：新 XY 数据、不下载基础权重。
python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data "$FASTFILL_BBOX_DATA/train.jsonl" \
  --validation "$FASTFILL_BBOX_DATA/validation.jsonl" \
  --output outputs/bbox-tiny-new --max-samples 2 --steps 1 --dry-run

# 真实 20 步 pilot：32 个训练前缀、完整 validation。
# --max-samples 是前缀，不是随机科研子集或正式训练。
CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_bbox_pilot.json \
  --backbone /home/jovyan/shanliantian/models/Qwen3-8B \
  --data "$FASTFILL_BBOX_DATA/train.jsonl" \
  --validation "$FASTFILL_BBOX_DATA/validation.jsonl" \
  --output outputs/bbox-qwen-pilot-new --max-samples 32

CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.evaluate \
  --checkpoint outputs/bbox-qwen-pilot-new/model \
  --data "$FASTFILL_BBOX_DATA/test.jsonl" \
  --output outputs/bbox-qwen-eval-new --device cuda --max-length 4096

CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.predict \
  --checkpoint outputs/bbox-qwen-pilot-new/model \
  --request fastfill/v2/configs/direct_request.json \
  --output outputs/bbox-prediction-new.json \
  --export-dir outputs/bbox-handoff-new --device cuda --max-length 4096
```

正式全数据训练须移除 --max-samples，另存曝光/步数/batch/验证预算明确的配置；20 步只是链路检查。structured/text 对照用同一新数据，不能混成历史 65 场景 cohort。文本 baseline 为 token CE，没有 π 等价文本目标的 marginal loss；文本目标采用来源规范 yaw，bbox 指标按合法对称性评测。当前多为稀疏住宅布局：平均 3.12 对象、最多 26，只有 135 场景达到 10 对象，restaurant 没有合格样本；截图密集布局须补来源审计和评测。OBB 房界资格不认证支撑/collision/mesh/physics。

## 历史 review3：丰富条件和部分字段监督

发布日期：2026-10-06。数据仓库：[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2)，**private**。固定 commit：`96f4946624b46bf8dc99bf94311b5d31290ea09a`，快照子目录：`review3-20261006`。

这里同步的是我们已经生成并审核的 v2 派生数据，不是重下载一套新原始库。review3 的九个数据文件与 review2 字节一致；review3 修复启用目标及梯度累积窗口的训练判定，没有再次修改数据。当前源代码与服务器启动文档可以更新，历史冻结包保留自己的版本说明。

## 1. 我们使用哪些数据

沿用此前筛选的 **16 个数据集家族**：11 个训练生产家族、2 个仅评测家族、3 个提供来源/审计证据的辅助家族。它们展开为 18 个冻结 IR source tags，其中 16 个在训练中出现；不能将重打包或重叠原始视图当作新的独立场景。

| 家族角色 | 内容 |
|---|---|
| 训练生产家族 | IL3D、InteriorGS、InternScenes、MansionWorld、OptiScene、SceneCAD/Scan2CAD、SpatialLM、Structured3D、SAGE10k、HSSD、MultiScan |
| 仅评测家族 | SceneSmith、SpatialGen |
| 辅助原始来源 | 3D-FRONT、3RScan、ARKitScenes；相关几何通过已选家族/适配路径进入，避免重复计数 |

使用冻结 `.release/v3.2/ir`、旧保存消息的 UID/split/flag 和对应审计记录构建，不使用外盘中较旧的 `fastfill_ir` 替代冻结来源。继承 train/dev/test（dev 改名 validation），没有重新随机划分。精确的 18 个 source tag、每个 split 场景数、父数据与输出 hash 见下载数据的 `data/selected-v3.2/manifest.json` 和 `DATASET_MANIFEST.json`。

| 数据 | 场景 train / validation / test | 总对象 | 用法 |
|---|---:|---:|---|
| 主集 `selected-v3.2` | 124,589 / 8,137 / 8,615 | 1,787,052 | 部分可信字段的掩码结构化监督 |
| 完整标签 `complete-label-cohort` | 57 / 2 / 6 | 572 | 四项完整几何 pilot 与同数据 Text/Structured 对照；全部来自 MultiScan |

主集有完整位置标签 1,732,321 个、完整局部尺寸标签 1,224,670 个、可信语义 yaw 标签 **581 个**。训练集 yaw 只有 **523 个对象、58 个场景**；大规模对象数量不代表朝向覆盖。完整标签子集的 57 个训练场景有 514 个对象，另外 9 个可信 yaw 对象位于一个不满足全对象完整标签条件的训练场景。

## 2. 模型如何读取

每行包含 `condition`、`target`、`validity` 和 `provenance`。模型条件只有房间、请求对象和可用约束；源 asset ID、真实目标尺寸/位姿不能从 provenance 混入输入。位置是米制、右手 Z-up 房间坐标中的 bbox bottom-center，尺寸是对象局部全长，yaw 为弧度。未知标签使用显式 validity mask，不补零。

结构化主集按可信字段训练 normalized position、log-size ratio、yaw-bin CE 和 GT-bin residual；固定请求身份，只有真正独立可交换组可使用 detached Hungarian 对应。Text SFT 必须使用每个请求对象都完整标注的共享 cohort，训练 assistant-token CE，不能把部分标签主集直接冒充完整 JSON 答案。pipeline 细节见 [训练说明](fastfill-v2-training-pipeline.md)。

使用选定 **Qwen3-8B** tokenizer/config 做完整对象与上下文资格预检；不能截断对象、约束或支撑引用。review3 预检结合实际启用的 loss，拒绝没有当前 objective 有效目标的样本；训练在所有 rank 的整个梯度累积窗口没有有效目标时跳过 optimizer update。有标签且预测正确导致的零 loss 仍应正常更新。

正式主集采样、朝向标签曝光、训练预算及对象/context 资格仍需 pilot 后冻结。65 个完整场景、2 个完整验证场景只够检查有限范围的训练和受控对照，不能据此宣称跨来源完整几何泛化。

## 3. 下载和完整性校验

服务器工作根目录为 `/home/jovyan/shanliantian`；数据放在其新建的 `data/fastfill-v2` 下，预训练模型另放 `models/Qwen3-8B`。私有下载需要有访问权的 Hugging Face 登录；不要把 token 放入脚本、命令日志或 Git。

```bash
hf auth login
export FASTFILL_DATASET_ROOT=/home/jovyan/shanliantian/data/fastfill-v2
hf download liantian/fastfill-v2 \
  --repo-type dataset \
  --revision 96f4946624b46bf8dc99bf94311b5d31290ea09a \
  --local-dir "$FASTFILL_DATASET_ROOT"
cd "$FASTFILL_DATASET_ROOT/review3-20261006"
sha256sum -c SHA256SUMS
export FASTFILL_MAIN_DATA="$FASTFILL_DATASET_ROOT/review3-20261006/data/selected-v3.2"
export FASTFILL_COMPLETE_DATA="$FASTFILL_DATASET_ROOT/review3-20261006/data/complete-label-cohort"
```

校验清单包含 15 项：9 个数据文件、快照 README、数据发布 manifest 和 4 个来源/父子审计文件。校验和文件不自包含。下载库的根 README 是 dataset card，快照 README 供独立拷贝阅读。总数据文件大小为 **1,471,089,503 bytes**，不含 mesh、贴图或模型权重；上传的是派生 JSONL，不需要在服务器重新跑原始资产构建。

从代码仓库根目录可使用下列已实际跑通的 tiny smoke 检查下载数据与训练接口。输出目录每次必须选一个新路径。这是离线管道检查，不代表 Qwen 模型质量。

```bash
python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data "$FASTFILL_COMPLETE_DATA/train.jsonl" \
  --validation "$FASTFILL_COMPLETE_DATA/validation.jsonl" \
  --output outputs/hf-dataset-tiny-smoke-new \
  --max-samples 2 --steps 1 --dry-run
```

真实 Qwen3-8B 的 BF16 pilot、保存加载、测试集评测和硬件检查见 [服务器手册](fastfill-v2-server-start.md)。其配置中的 `--data`/`--validation` 可换为上述完整 cohort 路径；主集实验再换为 `FASTFILL_MAIN_DATA`，同时记录实际资格/采样规则。真实 Qwen 训练不使用 `--dry-run`。

## 4. 审计和物理边界

review2 撤销了 14,739 条缺少主体可信 yaw/front 证据的旧来源派生 `faces`，涉及 14,101 场景；保留 12 条可信的来源派生 `faces`。原约束与原因写入 provenance，用户明确面向要求不适用这种来源撤销规则。UID、split、对象及顺序、target 和 validity 保留。

地板诊断在 `1e-4 m` 阈值下标记了 35,095 个对象、12,209 个场景。GT 未被静默抬高、缩小、删除或撤销已知地板；既有小偏差，也有明显偏差。此数据是字段监督语料，**不是完整 GT 都能通过严格 Validator 的物理认证语料**。schema/hash/split 零错误不证明 mesh、支撑、能力、物理稳定性、Host 或 Solver 验收通过。

本次上传前后均通过认证 API 确认仓库 `private=true`，匿名读取返回 HTTP 401。九个数据文件的远端大小及 SHA256 全部与冻结本地文件一致：大文件比较远端 LFS SHA256，小文件下载后计算 SHA256。证据保存在本地 `outputs/fastfill_v2/hf-private-release-20261006/publication-verification.json`。与数据配套的历史 review3 训练包 SHA256 为 `973ef1743a62bdce94150643d1df2049934edfe8a1bef5c1947e5066b07fb04c`；包内 pilot 骨干说明是当时快照，当前服务器按已明确的 Qwen3-8B 操作。

旧 `liantian/fastfill-data` 和 `liantian/fastfill-v3` 未修改。保留 private 是当前发布要求，原数据来源许可和访问限制仍有效；本发布没有赋予额外公开再分发或商业使用权。
