# 服务器运行说明（FastFill v4：基线 / D / E）

这个仓库就是完整代码，不需要再打补丁。仓库根目录对应原来的 `fastfill-main`，`code_v4/` 是数据构建器。

## Isambard-AI（Slurm）：一键提交

登录节点没有 GPU，所有 GPU 工作都通过 `sbatch` 交给计算节点。在 `$PROJECTDIR` 下 clone（和旧的 `fastfill` 并列），然后：

```bash
HF_HOME=$HOME/.cache/huggingface-v4 ../fastfill/env/bin/hf auth login   # V4 自己的登录（XXXpilar，Write），不碰 V3 的 ~/.cache/huggingface
sbatch fastfill/v2/ops/isambard_v4_setup.sbatch  # 1 卡、≤3 h：环境、Qwen3-8B、数据、测试、E 配置的 GPU 冒烟
# 日志 runs/isambard/ff4-setup-<job>.out 最后一行是 SETUP OK 后：
bash fastfill/v2/ops/isambard_v4_submit.sh       # base / D / E 三组，各占一个节点（4 卡），各带一个续训作业
# 三组都 done 后：
sbatch fastfill/v2/ops/isambard_v4_ablation.sbatch
```

- 和 V3 分开：代码、数据（`data/v4fixed`）、模型权重、runs、日志、上传的模型仓库都在这个 checkout 里，V3 的目录一个文件也不改。
  只共用 conda 环境：软链接到 `FASTFILL_ENV`（默认 `../fastfill/env`），setup 只读它、不往里装包，并检查它跑的是本 checkout 的代码（所有作业都设 `PYTHONPATH=$PWD`）。
  Qwen3-8B 从 `../fastfill/models/Qwen3-8B` 复制一份（`FASTFILL_OLD=...` 改来源），没有就重新下载。
- 进度看 `runs/autorun-v4-{base,D,E}/STATUS.json` 和 `runs/isambard/ff4-*.out`；最好的模型以私有方式上传到 `XXXpilar/fastfill-v4-models`（用 `FASTFILL_MODEL_REPO` 改）。
- 消融结果在 `runs/ablation-v4/`（验证集里按种子抽 1000 个房间，`ROWS=...` 可改），另在 GPT-5.6 回答过的 20 个房间（`fastfill/v2/ops/probe20/`）上对比；
  汇总表 `runs/ablation-v4/REPORT.md`。哪组没跑完就用它最新导出的 model-step，没有的组跳过，不会让整个消融失败。

## 下载代码和数据

```bash
git clone https://github.com/XXXPILAR/fastfill-v4.git      # 私有仓库：需要先 gh auth login 或配置 token
cd fastfill-v4
git submodule update --init RoomGenBench                   # 只拉这一层，不要加 --recursive
python -m pytest -q fastfill/v2/tests/test_masked_layout.py fastfill/v2/tests/test_remask.py fastfill/v2/tests/test_support.py
huggingface-cli download XXXpilar/fastfill-v4-v2-fixed --repo-type dataset --local-dir ../data_v4_v2_fixed   # 私有数据集：需要 huggingface-cli login
```

不要用 `git clone --recursive`：RoomGenBench 里还嵌套着 infinigen、TRELLIS、Hunyuan3D 等大型仓库，会下载几十 GB，FastFill 用不到。
数据集里还有 `smoke/`（3,657 / 400 个房间），可以先拿它做一次快速测试。

RoomGenBench 子模块固定在提交 `30f2e059`。V-DETR 和 MinkowskiEngine 没有登记，训练和评测都用不到。
下面命令里的 `../data_v4_v2_fixed` 指和仓库目录并列的数据目录，按你服务器上的实际路径改。

## 数据

正式训练要用新数据 `data_v4_v2_fixed`：没有任何目标 bbox，保留固定障碍物。旧的 `data_v4_v2` 不要再用。
新数据可以直接上传本地的 `data_v4_v2_fixed/`，也可以在服务器上重建：

```bash
cd code_v4 && python -m fastfill.build_v2 --like <data_v4>/step2/MANIFEST.json --v2-root .. --out ../data_v4_v2_fixed
```

本地重建的结果：train 125,792 / validation 8,225 / test 8,700 个房间，和旧数据一一对应；保留 20,362 个固定障碍物，删除 15,988 个无标注盒子。

## 训练

三组都用 `max_objects = 256`（原来是 128）：家具超过 128 件的房间（训练集 1,343 个，占家具 10.1%，大多来自 SAGE-10k）不再整间被拒。
128 件以内的房间结果不受影响。`max_length` 仍是 8192，文本超长的大房间照旧被拒并记录在 run_manifest 的 rejected 里。
为了公平，基线也要用 256 重训（`main7-cell05-main-o256-e5.json`，和旧基线只差这一项）；不要拿旧的 128 基线和新 D/E 比。
旧 checkpoint（128）不能直接续训成 256。

```bash
python -m fastfill.v2.train --config fastfill/v2/configs/main7-cell05-main-o256-e5.json \
    --data ../data_v4_v2_fixed/train.jsonl --validation ../data_v4_v2_fixed/validation.jsonl --output <runs>/baseline_o256
```

D 组：遮蔽训练 + 位姿噪声（位置 0.15 m、朝向 10°、尺寸 10%，10% 换成随机位置）。
E 组：在 D 的基础上加 `support_head` 和 `loss.support = 0.5`。

```bash
python -m fastfill.v2.train --config fastfill/v2/configs/main7-cell05-D-fixed-noise-e5.json \
    --data ../data_v4_v2_fixed/train.jsonl --validation ../data_v4_v2_fixed/validation.jsonl --output <runs>/D_fixed_noise
python -m fastfill.v2.train --config fastfill/v2/configs/main7-cell05-E-support-e5.json \
    --data ../data_v4_v2_fixed/train.jsonl --validation ../data_v4_v2_fixed/validation.jsonl --output <runs>/E_support
```

用 `autorun.py` 时要加 `--start-config <上面的配置> --name-suffix <区分名>`，因为自动生成的 run 名不区分这些新开关。

## 消融

```bash
python -m fastfill.v2.ops.remask_ablation --checkpoint <runs>/D_fixed_noise/model --data ../data_v4_v2_fixed/validation.jsonl \
    --output <runs>/abl_D --modes argmax spread remask1L remask4 remask4L remask4LX
python -m fastfill.v2.anchor_probe --checkpoint <runs>/D_fixed_noise/model --data ../data_v4_v2_fixed/validation.jsonl \
    --output <runs>/anchor_probe_D.json
# E：再加一次去掉全部支撑声明的版本，看模型自己预测支撑的准确率
python -m fastfill.v2.ops.remask_ablation --checkpoint <runs>/E_support/model --data ../data_v4_v2_fixed/validation.jsonl \
    --output <runs>/abl_E_nosupport --modes argmax spread remask4L remask4LX --strip-support
```

模式名：`remask{轮数}`，`L` 表示合法格子过滤，`X` 表示撤回低置信度家具。
关键判据：`anchor_probe` 里 B − A 的 95% 置信区间要明显落在 0 以下，才说明模型真的在利用已经放好的家具。
