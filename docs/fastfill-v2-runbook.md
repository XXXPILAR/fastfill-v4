# FastFill v2 执行手册

**当前主任务是完整房间条件＋指定对象＋空间要求 → 目标尺寸／底面中心／yaw → RoomGenBench 的 text+bbox 到 mesh 路径。** 当前数据、实际入训资格、七卡配置和启动命令见 [多源完成记录](fastfill-v2-multisource-20261006.md)；设计纠正与交付边界见 [本轮设计审核](fastfill-v2-design-audit-20261006.md)。三字段输入是独立消融，不能代替主方案。

下面各节保留 **历史 review3 冻结包** 的命令、数据数量、测试与 pilot 记录。`selected-v3.2` 是父语料，尚未应用新主版本的 D1/D2 资格处理；当前训练使用 `multisource-20261006/data` 的真实 tokenizer 资格视图。旧 catalog／memory Host 命令是离线契约测试，不能当成已经接通 RoomGenBench mesh 生成或持久 Host。

日期：2026-10-06。本文对应冻结设计及本次审计后的可上传训练包。本版为 review3，修复启用目标资格与空累积窗口更新；沿用 review2 数据，旧包与旧 hash 保留为历史快照。真实 Qwen3-8B 已完成单卡 BF16 的 20 更新步 pilot、保存和重新加载评测；Linux 套件为 653 tests + 114 subtests passed、6 个 Metal 跳过，v2 statement coverage 87.65%。全库正式训练、模型质量收益和真实资产闭环仍未验证；六个 pilot 测试场景均未通过严格目标几何验收。实测证据及原因见 [服务器启动手册](fastfill-v2-server-start.md)。冻结压缩包内文档保留冻结时状态，当前手册记录后续实测。

## 1. 上传包和目录

外盘本次未挂载，本地交付目录为仓库 `outputs/fastfill_v2/FastFill_v2_20261006_review3/`。可上传压缩包位于该目录的 `upload/FastFill_v2_training_20261006_review3.tar.gz`，解包后保留顶层目录 `FastFill_v2_20261006_review3/`。

```text
FastFill_v2_20261006_review3/
├── project/                     # fastfill v1/v2 代码、配置、测试和文档
│   ├── fastfill/
│   └── docs/
├── data/
│   ├── selected-v3.2/           # 主训练数据：train/validation/test.jsonl + manifest
│   └── complete-label-cohort/   # 共享完整标签子集及其构造清单
├── audit/                       # 数据审计、来源证据和本次验证报告
├── RUNBOOK.md                   # 本执行手册的包内副本
├── BUNDLE_MANIFEST.json         # 相对文件路径、大小和 SHA256 总清单
├── RUNBOOK.json                 # 交付清单、版本、数据与测试记录
├── SHA256SUMS                   # 包内已交付文件的校验值
├── verify_bundle.py             # 标准库校验器，支持 macOS/Linux
├── outputs/                     # 新训练、评测、配置和日志的输出目录
└── upload/                      # 本地压缩包及压缩包校验记录
```

训练使用包内 JSONL，不需要重新下载原始数据。原始外接盘数据和冻结 `.release/v3.2` 均保持只读；包内源代码、审计证据和数据也应保留为已校验版本。新配置、模型缓存和运行结果分别写入 `outputs/` 或 `models/` 等指定目录。所有训练和评测输出必须使用尚不存在的新路径。

本次 H20Z 服务器的上传、CUDA 安装、Qwen 下载及 pilot 命令见 [服务器启动手册](fastfill-v2-server-start.md)。启用目标修复见 [review3 记录](fastfill-v2-review3-objective.md)。

完整训练步骤、模型产物与 RoomGenBench 的任务区别见 [训练 pipeline 说明](fastfill-v2-training-pipeline.md)。

## 2. 数据范围与监督边界

主数据来自我们已经选定的冻结 v3.2 语料：5 个原始根目录、16 个数据集家族，对应 18 个 IR source 名称。原冻结消息共 160,964 场景；按旧训练 flag 规则保留 124,843 个训练场景。v2 继承旧 split，不重新随机划分；旧 `dev` 只改名为 `validation`，旧 flag 过滤只作用于训练集。

本次 v2 主构建结果如下。精确来源分布、hash 和剔除记录以 `data/selected-v3.2/manifest.json` 及 `rejections.jsonl` 为准。

| 项目 | 已验证数量 |
|---|---:|
| train 场景 | 124,589 |
| validation 场景 | 8,137 |
| test 场景 | 8,615 |
| 合计场景 | 141,341 |
| 目标对象 | 1,787,052 |
| 完整有效 position 向量 | 1,732,321 |
| 完整有效 size 向量 | 1,224,670 |
| 有效 yaw / 完整 size+position+yaw 对象 | 581 / 581 |
| v2 额外剔除场景 | 316 |

316 个额外剔除场景分别为 4 个无效轮廓、111 个缺少修正证据的斜窗、16 个没有可靠几何字段的场景、185 个带不支持倾斜固定物体的场景。未知字段使用 validity mask，不补零，也不将负尺寸或代理框静默修成有效标签。

迁移使用原 IR 的连续几何精度；旧答案中的取整数值不作为新监督。请求 ID 和输入对象顺序不依赖目标位置，原资产 ID 只保留在 provenance。已记录的窗宽修正和 InternScenes `k0` 证据控制转换与标签掩码；MansionWorld 的 footprint 代理尺寸、Holodeck 的 padded proxy 尺寸不提供真实局部 size 监督。旧 bbox 推导的 `on` 关系不转成真实支撑要求。

review2 只撤销旧参考布局派生、但主体 GT yaw 无效或来源证据含 `front_unknown_reason` 的 `faces`；原约束和撤销理由写入 provenance。明确用户要求不按此规则删除。UID、split、对象与几何标签不变；逐行父子核验保存在 `audit/reviewed-data-verification.json`。共撤销 14,739 条 faces，涉及 14,101 场景。主目录 `manifest.json` 记录父数据 hash 和本次修改计数，完整标签子集从新主目录重新派生。

35,095 个可信目标 bbox、12,209 个场景增加地板下界冲突诊断，保留原标签。原始 bbox/physics 位姿可能与严格 bbox 地板验收冲突；不据此把地板改为 unknown 或自动抬高标签。运行时独立地板下界阻止地下新增物体提交。这份主集尚不是全部经过 mesh/物理有效性筛选的数据；如定义严格场景 cohort，需要另报整场景过滤规则与覆盖。

默认 `strict` 策略仅对有规范前向证据的 MultiScan 对象启用语义 yaw。其他来源的可靠位置、尺寸字段可参加掩码监督，但不能据此声称整个主数据具备完整四项监督。`complete-label-cohort/` 必须保留自己的构造清单、来源 split 和排除原因；文本 SFT 与受控结构化对照都使用这一共享子集。完整标签子集为 **57 train / 2 validation / 6 test** 场景，共 65 条；不能代替主数据覆盖率报告。严格 yaw 标签在训练集只有 523 个对象，均来自 58 个 MultiScan 场景。默认全库均匀采样与 1,000 步配置不能保证充分训练 yaw head；正式协议需要先确定标签覆盖与采样方案，再评估是否补充可核验的前向证据。

我们的完整语料审计详见 `project/docs/fastfill-v2-review-20261005.md`；原始盘库存表另见 `project/docs/fastfill-v2-data-audit.md`，模型、loss 与运行契约详见 `project/fastfill/v2/README.md`。这里的框有效性不等同于可提交的真实场景。未知房间边界、地面接触、支撑表面、语义前向或必需能力会阻止相应运行时验收。

## 3. 解包和校验

以下命令中的 `/your/server/workdir` 改为服务器上的实际目录。压缩包顶层目录保持不变；不要将 Mac 的 `/Volumes/...` 训练路径带到服务器。

```bash
tar -xzf FastFill_v2_training_20261006_review3.tar.gz -C /your/server/workdir
export FASTFILL_BUNDLE=/your/server/workdir/FastFill_v2_20261006_review3
export PYTHONDONTWRITEBYTECODE=1
cd "$FASTFILL_BUNDLE"
sha256sum -c SHA256SUMS
python3 verify_bundle.py
```

`verify_bundle.py` 还支持 `python3 verify_bundle.py /path/to/FastFill_v2_20261006_review3`。它流式检查已交付文件的 hash、文件是否齐全及意外增删。新 `outputs/`、`models/`、`upload/` 与 Python/pytest 生成缓存不属于冻结交付文件。校验失败时先定位差异，再使用完整包恢复相应文件；不要通过改写 `SHA256SUMS` 将差异当成通过。

Linux 使用上面的 `sha256sum`；macOS 可用 `shasum -a 256 -c SHA256SUMS`。完整训练数据无需在服务器重新构建。迁移 manifest 中的原始 Mac 路径仅为 provenance；本地全量来源验收已写入 `audit/selected-full-verification.json`。服务器的 `verify_bundle.py` 校验传输后的文件完整性；没有原始冻结 IR/消息与审计输入时，不应直接运行 `legacy_verify` 并声称重新验证了全部来源几何。

## 4. Python 环境和设备检查

在已安装 Conda 的机器上创建独立 Python 3.12 环境。先按服务器的设备/驱动安装对应 PyTorch wheel，再安装项目依赖。PyTorch 官方[安装选择器](https://pytorch.org/get-started/locally/)提供按 OS、包管理器和 compute platform 选择的命令；这里不预设服务器 CUDA 版本。

```bash
conda create -n fastfill-v2 python=3.12 -y
conda activate fastfill-v2
cd "$FASTFILL_BUNDLE/project"
python -m pip install --upgrade pip
# 在此先执行适合该服务器的 PyTorch 安装命令。
python -m pip install -r fastfill/v2/requirements.txt
python -m pip install pytest pytest-cov
mkdir -p ../outputs
python -m pip freeze > ../outputs/environment-new.txt
python -c 'import torch; print({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda, "devices": torch.cuda.device_count()})'
```

`requirements.txt` 提供最低版本范围，并非已固定的服务器 lockfile。正式实验保存安装后的 `environment-new.txt`、GPU 型号、驱动版本、训练配置和数据清单。当前代码不依赖 MinkowskiEngine；V-DETR 只作为 matcher/criterion 与角度监督参考。

## 5. 离线测试和结构化 smoke

下面的测试与 tiny smoke 不下载 Qwen 权重。先完成包校验，再从 `project/` 运行。覆盖率数据、pytest cache 和报告写入新的 `outputs/` 路径。

```bash
cd "$FASTFILL_BUNDLE/project"
COVERAGE_FILE=../outputs/.coverage-new python -m pytest \
  fastfill/tests fastfill/v2/tests \
  -o cache_dir=../outputs/pytest-cache-new \
  --cov=fastfill.v2 --cov-config=fastfill/v2/.coveragerc \
  --cov-report=term-missing \
  --cov-report=json:../outputs/coverage-new.json -q

python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data ../data/complete-label-cohort/train.jsonl \
  --output ../outputs/smoke-new \
  --dry-run --max-samples 2

python -m fastfill.v2.evaluate \
  --checkpoint ../outputs/smoke-new/model \
  --data ../data/complete-label-cohort/test.jsonl \
  --output ../outputs/smoke-eval-new \
  --max-length 32768 --device cpu
```

成功时结构化训练目录包含 `model/`、`tokenizer/`、`run_manifest.json` 和 `training_log.json`；评测目录包含 `report.json` 和逐请求记录。检查各项有效标签计数、loss 和对应 head 梯度，而不仅是进程退出码。`tiny` 是离线管道检查用的因果 byte-token GRU，不代表 Qwen 质量，也不代表模型已学会可用布局。

`--max-samples 2` 只读取并检查训练文件的有界前缀。它不是从全库随机选取的科研子集。没有该限制时，当前结构化训练会将训练和验证 JSONL 载入内存；完整文本 SFT 也采用内存载入。正式运行前必须根据实际机器确认 host RAM、context、对象数与 batch 预算。

## 6. 正式结构化训练

已上传的私有数据及固定 revision 下载命令见 [数据发布与使用](fastfill-v2-dataset-release.md)。主集提供各字段掩码，完整标签 cohort 用于有界联调；两者不能混称为同一监督覆盖率。服务器已传入 review3 冻结包时无需再次下载同一数据。

用户已明确正式骨干为 **Qwen3-8B**，服务器为 `ssh yxd-dev`，8×H20Z（允许使用 GPU 1–7）、2.8 TiB 主机 RAM。在 `/home/jovyan/shanliantian` 下操作；模型固定下载到 `/home/jovyan/shanliantian/models/Qwen3-8B`，先用同一 8B 骨干做 BF16 小样本 pilot。具体命令见 [服务器启动手册](fastfill-v2-server-start.md)。正式采样、context、训练曝光和预算仍需根据预检结果冻结。

review3 冻结包中的 `configs/structured.json` 保留历史开发模板 Qwen2.5-0.5B-Instruct；它不是这次选定的正式模型。运行时另存 Qwen3-8B 配置，不能直接按旧模板启动。本包不含基础权重或已训练正式模型。LoRA、128 维两层对象解码器、12 yaw bins 和四项基础几何 loss 是开发起点；box/collision/boundary 默认关闭，默认 1,000 步与 batch 2 也不是已验证正式预算。

将配置复制到一个新的 `outputs/` 配置路径，并按实际机器修改 dtype、混合精度、训练步数、batch、梯度累积和 context。`FASTFILL_BACKBONE` 可以是实际可用的本地 checkpoint 路径，或已经选择并允许服务器下载的 Hugging Face checkpoint 名称。以下示例在该选择完成后执行；`qwen-server-new.json` 需先写好。

```bash
cd "$FASTFILL_BUNDLE/project"
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('fastfill/v2/configs/structured.json').read_text())
with Path('../outputs/qwen-server-new.json').open('x') as stream:
    stream.write(json.dumps(config, indent=2) + '\n')
PY
# 编辑 ../outputs/qwen-server-new.json，确定实际机器与实验配置后继续。
export FASTFILL_BACKBONE=/path/to/chosen-qwen-checkpoint
cd "$FASTFILL_BUNDLE/project"
python -m fastfill.v2.train \
  --config ../outputs/qwen-server-new.json \
  --backbone "$FASTFILL_BACKBONE" \
  --data ../data/selected-v3.2/train.jsonl \
  --validation ../data/selected-v3.2/validation.jsonl \
  --output ../outputs/selected-qwen-new

python -m fastfill.v2.evaluate \
  --checkpoint ../outputs/selected-qwen-new/model \
  --data ../data/selected-v3.2/test.jsonl \
  --output ../outputs/selected-qwen-eval-new \
  --device cuda --max-length 4096
```

评测的 `--max-length` 应与选定训练/部署配置一致；上面的 4,096 仅展示现有开发起点。context 或对象数超预算时拒绝完整样本，不能截断对象/关系来制造通过率。数据中的部分字段按 validity mask 参加 loss，原始输出与后处理结果分开记录。

多 GPU 可先配置 Accelerate，再以相同参数运行 `accelerate launch -m fastfill.v2.train ...`，使用另一个全新的输出目录。每个 checkpoint 写 `state-step-<n>/`（恢复状态）与 `model-step-<n>/`（可部署）；`--resume <state-step-n>` 在新输出目录继续训练（config/数据须一致）。部署/评测使用 `model/` 或任一 `model-step-<n>/`。`evaluate.py` 的 report.json 另含 `baselines`（房间中心／类别均值位置／类别中位尺寸／均匀 yaw，默认留一法，`--baseline-fit <jsonl>` 改用拟合文件）、`collapse`、`by_source`。

## 7. 文本 SFT 与受控结构化对照

文本模型要求每个请求对象有完整 size/position/yaw 标签；直接将全部部分标签主数据交给文本 SFT 会被拒绝。受控比较的结构化模型和文本模型必须使用同一个 `complete-label-cohort/`，同样的 condition 信息、context/object 资格规则、checkpoint、split、资产库和修复预算，并报告新增对象解码器参数及训练/推理成本。全量掩码结构化训练是另一个数据范围的实验。

先执行文本离线 smoke。只测一个请求的生成检查使用 `--max-samples 1`；质量对照必须移除此限制并覆盖固定测试集全部请求。

```bash
cd "$FASTFILL_BUNDLE/project"
python -m fastfill.v2.text_sft \
  --data ../data/complete-label-cohort/train.jsonl \
  --output ../outputs/text-smoke-new \
  --backbone tiny --dry-run --batch-size 1 --max-length 32768

python -m fastfill.v2.evaluate --baseline text \
  --checkpoint ../outputs/text-smoke-new \
  --data ../data/complete-label-cohort/test.jsonl \
  --output ../outputs/text-smoke-eval-new \
  --device cpu --max-length 32768 --max-new-tokens 4096 --max-samples 1
```

正式受控结构化训练可使用第 6 节命令，将 train/validation 路径替换成 `complete-label-cohort/`，输出改成 `../outputs/controlled-structured-new`。固定对应消融用 `configs/structured_fixed.json` 的新机器配置；可交换组匹配用 `structured.json` 的新机器配置。

以下是正式文本入口的对应示例。步骤、batch、LoRA、dtype 和 context 均需与已选定实验协议一致；示例数值仍为开发起点。文本输入长度要容纳完整 condition、完整 JSON 答案和 EOS；测试生成还要容纳 `max_new_tokens`。

```bash
python -m fastfill.v2.text_sft \
  --backbone "$FASTFILL_BACKBONE" \
  --data ../data/complete-label-cohort/train.jsonl \
  --output ../outputs/controlled-text-new \
  --device cuda --steps 1000 --batch-size 2 --lr 0.0001 \
  --lora-rank 8 --lora-alpha 16 \
  --backbone-dtype float32 --max-length 4096

python -m fastfill.v2.evaluate --baseline text \
  --checkpoint ../outputs/controlled-text-new \
  --data ../data/complete-label-cohort/test.jsonl \
  --output ../outputs/controlled-text-eval-new \
  --device cuda --max-length 8192 --max-new-tokens 4096
```

文本训练 loss 为 assistant token CE。JSON 解析后的几何距离是评测指标，不具有向离散生成 token 自动反传的梯度。单步 tiny smoke 的 JSON 解析失败必须保留为失败，不能用 repair 掩盖原始 schema 指标。

## 8. 单请求推理和资产闭环

`REQUEST.json` 包含 room、指定对象和约束，不含未来选定资产的真实尺寸。`CATALOG.json` 需由实际资产适配器提供资产引用、实际局部尺寸、raw-to-canonical 变换、能力及有来源依据的支撑表面；普通 bbox 顶面不是支撑表面证据。

```bash
cd "$FASTFILL_BUNDLE/project"
python -m fastfill.v2.predict \
  --checkpoint ../outputs/selected-qwen-new/model \
  --condition ../outputs/REQUEST.json \
  --output ../outputs/prediction-new.json \
  --device cuda --max-length 4096

python -m fastfill.v2.serve \
  --condition ../outputs/REQUEST.json \
  --prediction ../outputs/prediction-new.json \
  --catalog ../outputs/CATALOG.json \
  --output ../outputs/runtime-new.json \
  --max-asset-retries 2 --max-repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory

python -m fastfill.v2.evaluate \
  --checkpoint ../outputs/selected-qwen-new/model \
  --data ../data/selected-v3.2/test.jsonl \
  --catalog ../outputs/CATALOG.json \
  --output ../outputs/asset-eval-new \
  --device cuda --max-length 4096 \
  --asset-retries 2 --repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory
```

`predict/serve` 的预算选项前缀是 `--max-asset-retries`、`--max-repair-calls`；`evaluate` 使用 `--asset-retries`、`--repair-calls`。`--commit-in-memory` 只验证本地原子 Host 契约，不写入真实 WorldEdge 持久世界。真实 Host 需要版本检查、幂等键及全场景原子提交适配器。

target、首次解析 actual 和最终修复 actual 分开保存。系统不会通过删除请求对象或静默缩小资产满足验收。未知必需校验会阻止提交；`--required-levels bbox mesh` 在 `serve/evaluate` 要求 mesh 证据，而当前离线几何校验器没有这份证据，会明确失败。没有实际资产库时，只能报告模型输出和目标代理框指标，不能声称资产覆盖率、真实几何通过率或 Solver 成功。

当前 catalog 是类别/能力/尺寸参考实现，没有完整自然语言语义检索。非空 typed `attributes`（如颜色、材质）要求因缺少可信资产属性契约而拒收；实际校验返回硬性 `attributes_unverified`，不能仅换外部 Resolver 或写一份未经校验的 metadata 就放行。支持这些要求需要同时增加资产证据契约与对应 Validator。普通 description 文本本身不构成属性满足证据。

## 9. 结果记录与下一步

每次正式实验保留配置、数据 manifest、环境版本、checkpoint、训练日志、逐请求输出和报告。测试集全部请求作为 schema/ID/validity 和系统成功率的分母；缺标签的参考误差另报 eligible 数量。原始模型、解析后、修复后指标必须分开。生成延迟与运行时延迟合成实际端到端时间，reference scoring/matching 的评测耗时另记；提供外部预测时生成延迟未知。

本次最终套件 **659 tests + 114 subtests passed**，v2 代码覆盖率 **87.68%**；父数据协议逐行审核 **141,341 条、0 错误**（不等于零物理违规）。本包中的本地 smoke、回归测试和审计证明实现能被检查，不证明正式模型收益。包内 `RUNBOOK.json` 与 `audit/` 保存本次确切测试/覆盖率和 smoke 记录；运行本节命令后以新的报告替换自己的实验结论。下一项需要确定的信息是正式训练 checkpoint、服务器 GPU/host RAM、训练预算以及真实资产库/Host 适配器。模型质量结论须等训练、同条件对照及资产闭环评测完成后给出。
