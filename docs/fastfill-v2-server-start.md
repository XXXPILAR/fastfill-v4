# FastFill v2：Qwen3-8B 服务器启动

**当前服务器主入口是 `/home/jovyan/shanliantian/FastFill_v2_multisource_20261006`。** 完整条件主集为 124,589 / 8,137 / 8,615 场景；真实 Qwen tokenizer／对象预算资格为 124,375 / 8,125 / 8,602。环境、Qwen 权重、七卡 NCCL 和有界生产 pilot 已验收，完整三轮训练尚未启动。实际配置与命令见 [多源完成记录](fastfill-v2-multisource-20261006.md)。最终下游是 RoomGenBench；本训练入口不依赖 vLLM 或 MinkowskiEngine。当前实现提交 `435d4e6532bdbc24d130df5edc11c5aa002a0460` 的权威代码目录为该主入口下的 `project-current-audit-20261006`，旧 `project/` 保留为 pilot 快照。

## 2026-10-07 第二轮正式运行：4 卡回归 ＋ 3 卡 grid

第二轮代码（K1–K10，基于 7ea1a0d 的未提交改动）和按其重建的 main 数据都还没有同步到服务器；下面两项是启动前提，本轮均未执行。数据重建命令见 [v2 README 的 round 2 一节](../fastfill/v2/README.md#2026-10-07-round-2)。两组运行同时占用用户允许的 GPU 1–7，GPU0 保留：

```bash
cd <服务器上的第二轮代码目录>
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
PY=/home/jovyan/shanliantian/FastFill_v2_20261006_server/env/bin/python
MAIN=<服务器上重建后的 main 目录>
D=$(sha256sum $MAIN/train.jsonl | cut -d' ' -f1); V=$(sha256sum $MAIN/validation.jsonl | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES=1,2,3,4 $PY -m torch.distributed.run --standalone --nproc_per_node=4 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_4gpu_regression.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-4gpu-regression
CUDA_VISIBLE_DEVICES=5,6,7 $PY -m torch.distributed.run --standalone --nproc_per_node=3 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_3gpu_grid.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-3gpu-grid
```

- 两份配置的全局 batch 都是 96（4×1×24、3×1×32），3,887 次更新 = 124,375 个场景上的 3 轮，warmup 117，每 500 步验证并保存 `state-step-<n>/` 和 `model-step-<n>/`。`--nproc_per_node` 与配置不符会改变全局 batch。
- 启动后立即检查各自输出目录的 `run_manifest_start.json`：`world_size` 为 4／3，`backbone.revision` 为 Qwen3-8B 的 snapshot revision（本地目录则只有 `config_sha256`），`supervised_samples` 是训练器按 8192 token 预检后实际入训的场景数。不是 124,375 时停止，按 `steps=⌈3N/96⌉`、`warmup=round(0.03·steps)` 另写配置再启动。
- `--expect-*-sha256` 不匹配时，在创建输出目录之前报错退出。中断后用 `--resume ../run-…/state-step-<n>` 继续，输出必须是新目录，配置和数据必须一致。
- 选模：结束后读 `run_manifest.json` 的 `selection_metric.best.step`（minimal 投影 collapse score，越低越好；它没有精度项，先对照 `selection_metric.ground_truth` 和该步 `validation.minimal.unweighted.position`，最后一步 3887 也有验证和 `model-step-3887`），评测该 `model-step-<n>`：`$PY -m fastfill.v2.evaluate --checkpoint ../run-…/model-step-<n> --data $MAIN/test.jsonl --projection full minimal --device cuda --output <新目录>`。不传 `--max-length` 时自动使用 checkpoint 记录的 8192。

本页其余内容保留历史 review3 与 SpatialLM XY pilot 的准备和运行记录。它们的旧目录、数量和 checkpoint 均有各自版本，不能作为当前主任务的全量训练结果。服务器整理后的实际路径以最新发布／清理收据为准。

日期：2026-10-06。用户已指定正式骨干为 **Qwen/Qwen3-8B**，服务器通过 SSH alias `yxd-dev` 访问，工作目录为 `/home/jovyan/shanliantian/FastFill_v2_20261006_server`。此前 Qwen2.5-0.5B 只是未执行的可选链路模板，不是本次正式模型。

本次依赖审计要求 `anyio>=4.14.2`、`fsspec>=2026.6.0`；在独立环境安装补丁版本，不修改用户的 base 环境。旧冻结包的 requirements 不包含这两个新增下限，以下安装命令明确补齐。

[OptiScene 论文 §4.1 Training Setup](https://arxiv.org/html/2506.07570v1#S4.SS1) 写明使用 Qwen3-8B 进行 SFT/DPO。[官方 README 训练示例](https://github.com/PolySummit/OptiScene#training-pipeline) 使用 Qwen2.5-7B-Instruct；两处并不完全一致，本次按用户选择固定 Qwen3-8B，不将 README 示例模型当作论文同一配置。

本手册使用冻结 review3 包。包内代码、数据和清单保持不变；独立环境、运行配置与训练输出放在包目录之外，基础模型按用户要求重新下载到 `/home/jovyan/shanliantian/models/Qwen3-8B`。**服务器包的 279 文件校验、模型五个官方 SHA256、真实 Qwen3-8B 的 20 更新步及保存后重新加载评测已通过。** 这只确认运行链路；六个测试场景均未通过严格目标几何验收。v2 实现已同步至私有 GitHub `Slian22/fastfill`，实现提交为 `071fc3486329ce87fb41d29985b96c9931fbde4c`。私有数据下载及固定版本见 [数据使用手册](fastfill-v2-dataset-release.md)。

## 历史 SpatialLM XY pilot：服务器状态和实验入口

新工作根目录 `/home/jovyan/shanliantian/FastFill_v2_direct_bbox_20261006`。新train/validation/test **9601/539/624** 行通过真实Qwen3 tokenizer全量预检，最大 **716/490/526 tokens**；100,635个尺寸轴都在head范围。独立env和固定revision模型已安装，无需vLLM或MinkowskiEngine。

新协议20步pilot实际完成：seed42随机32train/16validation/16test，保存加载及无资产五对象导出通过，全部模块每步有梯度；测试schema/ID/正尺寸16/16，bbox代理无失败仅1/16。全数据正式训练尚未启动，效果不能验收。pilot是修复C1/C3/C4之前的独立快照，K=1/fixed/no-catalog不触发反例；修复及最终hash见 [全量审计处置](fastfill-v2-full-review-followup.md)。

全数据配置 qwen3_8b_bbox_full_fixed.json 是**当前合格XY数据的首个固定对应baseline**：单卡batch4、累积4、BF16、上下文1024、四项loss权重1。恰好9601个入训场景下，ceil(ceil(9601/4)/4)=601次更新/epoch，1803次更新覆盖3完整epoch；每epoch完整validation，只保存最终部署模型，不自动选最佳epoch或恢复optimizer。预算是明确实验起点，不是已验证最优配置；数据资格或卡数变化需重新计算曝光。

```bash
myconda
conda activate /home/jovyan/shanliantian/FastFill_v2_20261006_server/env
export FASTFILL_BBOX_WORK=/home/jovyan/shanliantian/FastFill_v2_direct_bbox_20261006
cd "$FASTFILL_BBOX_WORK/project-final"
CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_bbox_full_fixed.json \
  --backbone /home/jovyan/shanliantian/models/Qwen3-8B \
  --data "$FASTFILL_BBOX_WORK/data/train.jsonl" \
  --validation "$FASTFILL_BBOX_WORK/data/validation.jsonl" \
  --output "$FASTFILL_BBOX_WORK/outputs/qwen3-8b-xy-full-fixed-3epochs-new"
```

输出每次选新路径，结束后对全部624test请求评测。readiness仅针对该资格集的数值监督实验，不扩成历史16源全部合格、密集32–122对象场景有效或mesh/physics通过。3epoch wall time未实测；pilot训练主体6.67秒、含加载/评测/导出46.23秒，不能线性视为全量保证。保留GPU上其他任务。

## 历史 review3 环境准备和 pilot 记录

## 本次服务器实测

| 检查 | 2026-10-06 实际结果 |
|---|---|
| 独立 Conda / CUDA | Python 3.12.14；torch 2.13.0+cu126；Transformers 5.14.1；PEFT 0.21.0；Accelerate 1.15.0；物理 GPU 1 的 H20Z/BF16 检查通过 |
| 完整离线测试 | 653 tests + 114 subtests passed；6 个 Metal 专用测试在 Linux 跳过；v2 statement coverage 87.65%，不含 branch coverage |
| 真实骨干 pilot | 两个 train 场景、两个 validation 场景；20 个 optimizer 更新步；8,741,790 个可训练参数；每步 LoRA、decoder 和四个 head 的梯度范数均大于零；无空目标窗口 |
| 保存后评测 | 全部 6 个 test 场景成功生成；schema、ID 完整性、正尺寸均为 6/6；严格目标几何通过为 0/6；未接入资产、Host 或 Solver |
| 完整性与依赖 | 测试后冻结包仍为 279 文件 hash 全匹配；`pip check` 无冲突；68 个实际依赖的已知漏洞查询为 0，查询跳过为 0 |

训练 loss 首步为 3.68819，末步为 0.99809；两个 step 的 batch 分别包含 7 和 6 个对象，仅说明这次小样本优化运行。两个验证场景的 batch objective 均值在 step 5/10/15/20 为 3.04668/3.09148/3.09480/3.16789，没有显示验证收益。测试的 37 个有效对象平均 bottom-center 误差为 2.46728 米、log-size 误差为 0.59863、yaw 误差为 1.43627 弧度、BEV IoU 为 0.01368。

六个测试场景均缺少可信地板边界、房间边界、房高和支撑证据；未知硬检查阻止通过。此外一个场景出现 collision 失败。因此 `0/6` 不能解释为六个场景均已确认发生几何穿插，也不能把 schema 通过写成可提交场景。没有执行资产解析、持久 Host 提交、mesh/physics/Solver 或 GLB 展示。本次不是全库正式训练。

服务器证据在工作目录 `outputs/`：`pytest-server.log`、`coverage-server.json`、`cuda-preflight.json`、`model-integrity.json`、`pilot-eligibility.json`、`qwen3-8b-pilot-20steps/{run_manifest,training_log}.json`、`qwen3-8b-pilot-eval/{report.json,outcomes.jsonl}`、`bundle-integrity-after-validation.json`。独立解包运行没有 `.git`，训练 manifest 的 `code_commit` 为 null；版本依据冻结包 hash 和逐生产文件 hash，不伪造 commit。

依赖查询依据实际 `pip freeze`；仅将已安装 `torch==2.13.0+cu126` 映射到公告数据库的公开版本 `2.13.0`，将 Conda 的 packaging 本地构建 URL 映射到运行时确认的 `packaging==26.3`。原始环境记录保留，查询结果不等于对 CUDA 二进制进行安全证明。此次查询使用补丁版本 AnyIO 4.15.1 和 fsspec 2026.9.0。

## 为什么无需安装 vLLM

OptiScene 官方 [environment.yml](https://github.com/PolySummit/OptiScene/blob/409f1317e1a8afe852f4cdc99dfefaed30aeb425/environment.yml#L11) 包含 vLLM，但同一 commit 的 [SFT](https://github.com/PolySummit/OptiScene/blob/409f1317e1a8afe852f4cdc99dfefaed30aeb425/scripts/sft_train.py)、[DPO](https://github.com/PolySummit/OptiScene/blob/409f1317e1a8afe852f4cdc99dfefaed30aeb425/scripts/dpo_train.py) 及 [推理](https://github.com/PolySummit/OptiScene/blob/409f1317e1a8afe852f4cdc99dfefaed30aeb425/scripts/inference.py#L34) 代码均没有调用 vLLM；推理使用 `model.generate()`。环境文件列包不能单独证明运行入口依赖该包。

本仓库 v1 的文本服务调用 vLLM，v1 文本评测可选 vLLM。v2 结构化路径是 `AutoModel` hidden states → 对象 decoder → 连续 heads，使用 PyTorch/PEFT/Accelerate 反向传播，没有文本 rollout 或 vLLM 适配。这次环境不安装 vLLM；如后续使用 v1 服务或另行实现文本 rollout，单独准备对应环境与兼容性验证。

## 1. 登录并初始化用户 Conda

从 Mac 登录后，先运行用户提供的 `myconda`：

```bash
ssh yxd-dev
myconda
cd /home/jovyan/shanliantian
```

自动化 SSH 命令使用 `zsh -lic`，在该 shell 内先运行 `myconda`。每个新 SSH 进程都需要重新初始化和激活环境；上一个连接的环境变量不会传到下一个连接。

```bash
ssh yxd-dev 'zsh -lic "
myconda
cd /home/jovyan/shanliantian
python -V
"'
```

随后在服务器 shell 中设置本次工作路径：

```bash
export FASTFILL_WORKDIR=/home/jovyan/shanliantian/FastFill_v2_20261006_server
export FASTFILL_BUNDLE="$FASTFILL_WORKDIR/FastFill_v2_20261006_review3"
export FASTFILL_ENV="$FASTFILL_WORKDIR/env"
export FASTFILL_BACKBONE=/home/jovyan/shanliantian/models/Qwen3-8B
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$FASTFILL_WORKDIR/outputs" /home/jovyan/shanliantian/models
```

用户提供 8×H20Z，每卡 143,771 MiB，driver 570.124.06；主机 RAM 2.8 TiB。用户允许使用物理 GPU 1–7，pilot 先用 GPU 1，保留已有任务。用户报告 `/home/jovyan` 约 1.8 TiB 可用；运行时仍需检查当前空间、显存和占用：

```bash
nvidia-smi
free -h
df -h "$FASTFILL_WORKDIR"
```

## 2. 上传、解包并校验冻结包

Mac 上的新包位于仓库 outputs；原始外盘目录不需要重复上传。以下命令在 Mac 执行：

```bash
scp /Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/FastFill_v2_20261006_review3/upload/FastFill_v2_training_20261006_review3.tar.gz \
    /Users/slian/Desktop/3D/Worldedge/OptiScene/outputs/fastfill_v2/FastFill_v2_20261006_review3/upload/FastFill_v2_training_20261006_review3.tar.gz.sha256 \
    yxd-dev:/home/jovyan/shanliantian/FastFill_v2_20261006_server/
```

服务器执行以下命令。已解包时先运行校验，避免重复解包覆盖已存在内容：

```bash
cd "$FASTFILL_WORKDIR"
sha256sum -c FastFill_v2_training_20261006_review3.tar.gz.sha256
if [ ! -e "$FASTFILL_BUNDLE" ]; then
  tar -xzf FastFill_v2_training_20261006_review3.tar.gz
fi
cd "$FASTFILL_BUNDLE"
sha256sum -c SHA256SUMS
python3 verify_bundle.py
```

review3 压缩包 SHA256 为 `973ef1743a62bdce94150643d1df2049934edfe8a1bef5c1947e5066b07fb04c`。它包含主数据、完整标签子集、代码、测试及审核记录，不含基础 Qwen 权重或正式训练模型。

## 3. 独立环境、CUDA 检查和 Qwen3-8B 权重

环境使用用户 Conda 的独立 prefix，不修改已有训练环境。如果该 prefix 已建立，直接激活并核对依赖；仅在缺失时创建或安装：

```bash
conda create --prefix "$FASTFILL_ENV" python=3.12 -y
conda activate "$FASTFILL_ENV"
cd "$FASTFILL_BUNDLE/project"
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r fastfill/v2/requirements.txt
python -m pip install transformers==5.14.1 peft==0.21.0 accelerate==1.15.0
python -m pip install 'anyio>=4.14.2' 'fsspec>=2026.6.0'
python -m pip install pytest pytest-cov huggingface_hub
python -m pip freeze > "$FASTFILL_WORKDIR/outputs/environment-server.txt"
```

该 CUDA 12.6 PyTorch build 与用户 570.124.06 驱动相容；不需要更换系统驱动或安装系统 CUDA Toolkit。`nvidia-smi` 的 12.8 表示驱动支持信息，`torch.version.cuda` 可以为 12.6。参见 [PyTorch 官方安装表](https://pytorch.org/get-started/previous-versions/) 和 [CUDA 12.6 驱动要求](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)。

先验证选中的单卡及 BF16：

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
import torch
from accelerate import Accelerator
assert torch.cuda.is_available(), 'CUDA PyTorch 不可用'
assert torch.cuda.is_bf16_supported(), '选中设备不支持 BF16'
assert Accelerator(cpu=False, mixed_precision='bf16').device.type == 'cuda'
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert (x @ x)[0, 0].item() == 16
print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
       'visible_gpu_count': torch.cuda.device_count(),
       'physical_gpu_1': torch.cuda.get_device_name(0)})
PY
```

按用户要求重新下载完整基础模型到指定 models 目录，不引用其他项目的缓存作为本次模型路径。使用官方固定 revision：

```bash
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir "$FASTFILL_BACKBONE"
```

参见 [Qwen3-8B 官方模型页](https://huggingface.co/Qwen/Qwen3-8B) 和 [HF CLI 文档](https://huggingface.co/docs/huggingface_hub/en/guides/cli)。这是基础权重下载，不是下载我们已训练的模型。无需为本入口安装 MinkowskiEngine、V-DETR 或 vLLM。

## 4. 离线测试和真实 tokenizer 预检

完整离线套件使用 CPU/Gloo，显式隐藏 GPU；它不证明 CUDA/NCCL/BF16 大模型路径通过。Metal 测试在 Linux 上正常跳过。测试及 tiny smoke 的工件放在本次工作目录 outputs：

```bash
cd "$FASTFILL_BUNDLE/project"
CUDA_VISIBLE_DEVICES='' COVERAGE_FILE="$FASTFILL_WORKDIR/outputs/.coverage-server" \
  python -m pytest fastfill/tests fastfill/v2/tests \
  -o "cache_dir=$FASTFILL_WORKDIR/outputs/pytest-cache-server" \
  --cov=fastfill.v2 --cov-config=fastfill/v2/.coveragerc --cov-report=term-missing -q

CUDA_VISIBLE_DEVICES='' python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/train.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/offline-server-smoke" \
  --dry-run --max-samples 2
```

创建新的 Qwen3-8B pilot 配置；不修改冻结包里的默认配置。每次实验使用新的配置与输出路径：

```bash
export FASTFILL_PILOT_CONFIG="$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot.json"
python - <<'PY'
import json, os
from pathlib import Path
cfg = json.loads(Path('fastfill/v2/configs/structured.json').read_text())
cfg = {**cfg,
       'model': {**cfg['model'], 'backbone': os.environ['FASTFILL_BACKBONE'],
                 'backbone_dtype': 'bfloat16', 'local_files_only': True},
       'training': {**cfg['training'], 'cpu': False, 'steps': 20, 'batch_size': 1,
                    'max_length': 4096, 'mixed_precision': 'bf16',
                    'gradient_accumulation_steps': 1,
                    'checkpoint_every': 10, 'validate_every': 5}}
with Path(os.environ['FASTFILL_PILOT_CONFIG']).open('x') as stream:
    stream.write(json.dumps(cfg, indent=2) + '\n')
PY
```

其余采用可执行的开发起点：decoder 128、4 heads、2 layers，yaw 12 bins，LoRA rank 8/alpha 16，learning rate 1e-4，四项几何 loss 权重均为 1，box/collision/boundary 权重为 0。这些不是已调优的正式超参数。

使用真实 Qwen3 tokenizer 和实际启用 objective 检查两个训练场景、全部两个验证场景：

```bash
python - <<'PY'
import json, os
from pathlib import Path
from fastfill.v2.batch import load_tokenizer, tokenize_condition
from fastfill.v2.io import read_samples
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.train import _preflight, _training_config
cfg = json.loads(Path(os.environ['FASTFILL_PILOT_CONFIG']).read_text())
tok = load_tokenizer(cfg['model']['backbone'], local_files_only=True)
for split, limit in [('train', 2), ('validation', None)]:
    rows = read_samples(Path(os.environ['FASTFILL_BUNDLE']) / 'data/complete-label-cohort' / f'{split}.jsonl',
                        training=split == 'train', max_samples=limit)
    kept, rejected = _preflight(rows, tok, ModelConfig(**cfg['model']),
                                _training_config(cfg['training']), LossConfig(**cfg['loss']))
    print(split, {'kept': len(kept), 'rejected': rejected,
                  'condition_tokens': [len(tokenize_condition(r['condition'], tok)[0]) for r in rows]})
    assert not rejected, 'pilot 集合不完整；显式调整预算后重新预检，不能截断对象'
PY
```

结构化模型读取确定性 condition 文本、提取 hidden states，使用外部连续 heads；它不调用 Qwen 的文本采样或 thinking 生成。文本 SFT 是独立实验路径，不是这次 pilot 的前置。

## 5. 单卡 Qwen3-8B 的 20 更新步 pilot

只有以上检查通过后执行；不加 `--dry-run`。该参数只接受显式 tiny 骨干。

```bash
CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.train \
  --config "$FASTFILL_PILOT_CONFIG" \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/train.jsonl" \
  --validation "$FASTFILL_BUNDLE/data/complete-label-cohort/validation.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-20steps" \
  --max-samples 2

CUDA_VISIBLE_DEVICES=1 python -m fastfill.v2.evaluate \
  --checkpoint "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-20steps/model" \
  --data "$FASTFILL_BUNDLE/data/complete-label-cohort/test.jsonl" \
  --output "$FASTFILL_WORKDIR/outputs/qwen3-8b-pilot-eval" \
  --device cuda --max-length 4096
```

进程内 `cuda:0` 对应物理 GPU 1。训练以 BF16 载入基础骨干、使用 BF16 autocast；连续尺寸指数、pooling 累加等保留代码规定的 float32 数值保护。20 步用于检查真实 8B 权重、CUDA、LoRA、四项梯度、validation、保存和重新载入，不作为布局质量证据。

LoRA 产物不包含完整基础模型；评测和部署仍需同一固定 revision 的 Qwen3-8B，并保留训练配置中的基础路径。2026-10-06 起 `--resume <state-step-n>`（或 `training.resume`）可在**新**输出目录继续训练：恢复模型、optimizer、LR 调度、GradScaler、RNG、累积计数与数据顺序，要求 config（除 `training.resume`）、`--max-samples` 与训练数据 hash 一致；每个 checkpoint 同时导出 `model-step-<n>/`。

全库正式训练还需冻结 batch/context、对象资格、稀有 yaw 采样、地板诊断策略、训练曝光预算和实测显存。训练集可信 yaw 仅 523 个对象，不能把默认均匀 1,000 步配置直接视作充分的正式方案。真实资产、WorldEdge Host、mesh/physics/Solver 和 GLB 展示另行验收。

完整实验与产物说明见 [执行手册](fastfill-v2-runbook.md) 和 [pipeline](fastfill-v2-training-pipeline.md)。
