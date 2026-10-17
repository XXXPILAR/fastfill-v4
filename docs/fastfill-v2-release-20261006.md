# FastFill v2 本轮交付、发布与服务器收据

日期：2026-10-06。本页绑定当前可复现代码、数据和服务器状态；设计收益、正式模型质量及真实物理闭环仍由后续实验验证。

## 当前固定版本

| 项目 | 已发布／已验收固定点 |
|---|---|
| 代码实现 | 私有 [Slian22/fastfill](https://github.com/Slian22/fastfill/commit/435d4e6532bdbc24d130df5edc11c5aa002a0460)，实现提交 `435d4e6532bdbc24d130df5edc11c5aa002a0460`；后续文档提交只补发布收据 |
| 当前数据 | 私有 [liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2/tree/ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88/multisource-20261006)，revision `ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88` |
| Qwen | Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218` |
| 参考源码 | V-DETR `9062d75...`；MinkowskiEngine `02fc608...`；RoomGenBench `30f2e059...`；Git submodules 保留完整 commit，原 upstream tracked 源码未改 |
| 服务器主目录 | `/home/jovyan/shanliantian/FastFill_v2_multisource_20261006` |
| 当前实现快照 | 主目录下 `project-current-audit-20261006`；旧 `project` 只作为已测 pilot 快照保留 |
| Python 环境 | `/home/jovyan/shanliantian/FastFill_v2_20261006_server/env/bin/python`；独立环境，Python3.12／torch2.13+cu126／transformers5.14.1 |
| 实际入训数据 | 主目录 `preflight-main/eligible/{train,validation,test}.jsonl`，124,375／8,125／8,602 场景 |

HF canonical24数据文件合计 **6,277,486,224 bytes**。全部47发布文件的远端大小／SHA已逐项核对，且从最终固定 revision 在服务器**实际重新下载**全部47文件、重算SHA；实际下载6,281,154,832 bytes，47/47一致。`sha256sum -c multisource-20261006/SHA256SUMS` 实际46/46 OK、exit0，metadata也在校验中。临时下载已删除。发布前后 `private=true`，旧35项文件只从HEAD移除，旧revision35f527仍列出它们。证据：[远端核验](evidence/fastfill-v2-audit-20261006/publication/hf-after-verification.json)、[实际下载核验](evidence/fastfill-v2-audit-20261006/publication/actual-download-verification.json)。

## 参考源码／论文同步

服务器 `references-20261006` 已原子发布，经核验 **1,772/1,772 导出文件 SHA匹配**、1,755源码／小文本文件与固定Git blob一致；V-DETR 63导出文件（53源码）、MinkowskiEngine232（190源码）、RoomGenBench1,460（65源码）。这些是同步统计，不是将RoomGenBench全部文件标成全文审查。

源码位于 `references-20261006/source/{V-DETR,MinkowskiEngine,RoomGenBench}`，三篇／份PDF位于该目录的papers，全部与本地固定SHA一致：V-DETR用户details-v2、官方arXiv v1、ME官方CVPR2019。原上游预编译.so/.egg、较大图片／站点二进制、RoomGenBench七个nested submodule及方法模型未纳入该源码导出，排除清单与完整tracked库存保留于REFERENCE_MANIFEST。GitHub通过固定gitlinks保留完整upstream版本；服务器源码包不冒称原detector／生成模型已安装运行。

本轮全文阅读范围是V-DETR53源码／10,488行，ME232文本／52,848行，当前FastFill v2 36生产模块；RoomGenBench审查的是输入／装配／评分关键接口。原native与论文实验没有复现。详细 [参考审核](fastfill-v2-reference-code-audit-20261006.md) 和最终同步收据分别记账。

## 本轮改动与验证

review2 C1–C4已在当前代码修复；新主集处理D1估计地板／贴齐资格和D2不可表达size，完整保留数值、身份和split。本轮另修复FP16 overflow误计完成更新，以及不完整交换组关闭其他组Hungarian参考指标。动态RoomGenBench入口保存canonical条件、声明支撑、fixed与constraints，消费GLB/sidecar，区分ok／fallback／missing／fit_mismatch，失败保留所有实例并返回exit2。

本地当前环境完整套件：**903 passed、114 subtests passed，51.74秒**。独立v2覆盖运行：765 passed，statement coverage **88.35%**。服务器同一实现完整套件：**897 passed、114 subtests passed、6个Metal环境skip，83.84秒**。没有把skip记为失败或模型可用率；没有将原CUDA detector的只读审查计作运行测试。服务器合并 v1＋v2 的全源码 statement coverage 为 **76.71%（7,294/9,509）**；它不是 v2 覆盖率。此次新增／修改的 v2 独立服务器覆盖为 **87.71%（4,689/5,346）**，通过80%门槛；该覆盖运行759 passed、6 skipped、24 subtests，73.52秒。optional装配依赖为trimesh5.1.1／Pillow12.3.0，pip check clean；原环境／pilot保留。

服务器代码manifest的209/209 regular files分别与本地固定commit和服务器现场逐项匹配，代码未变；optional下游symlink独立记录。代码／环境／测试和清理收据见 [服务器证据目录](evidence/fastfill-v2-audit-20261006/server-current)。

当前代码独立review未发现确证P1/P2提交阻断；常见provider token/private-key扫描无命中。最终实际服务器55个public依赖的pip-audit无已知漏洞；NVIDIA分发包排除、torch本地build tag仅为advisory查询去除，原安装版本未改；检查范围、版本和排除项随证据保存。这不证明全部native算子或来源资产安全／正确。

## 服务器清理

已删除旧review3和单源pilot的解包数据副本、拼接可核的transfer-parts、逐member核验的传输tar、探针和INCOMPLETE upload。最终12个目标缺席，**净逻辑删除1,717,202,511 bytes**。三份strict_rectangle曾删后从保留JSONL原样筛选恢复，bytes/SHA精确一致，现保留给独立消融与HF发布；恢复已从净删除量扣除。

raw sources、独立环境、冻结档案、模型／checkpoint、当前主／资格／简化／NEAR视图以及GPU进程保留。JuiceFS共享free计数没有同步反映变化；本页只报告逻辑文件量，不将du估计冒称已回收物理磁盘。[逐路径清理／恢复收据](evidence/fastfill-v2-audit-20261006/server-cleanup-receipt.json)包含完整核验理由。

## 当前训练命令与下游

```bash
cd /home/jovyan/shanliantian/FastFill_v2_multisource_20261006/project-current-audit-20261006
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
  /home/jovyan/shanliantian/FastFill_v2_20261006_server/env/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=7 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_world7.json \
  --data ../preflight-main/eligible/train.jsonl \
  --validation ../preflight-main/eligible/validation.jsonl \
  --output ../run-main-world7-B1K16-3epochs
```

该完整三轮命令**没有在本轮启动**；此前只有有界单卡／七卡pilot。GPU0服务和1–7合作式占位保留。3,333更新为已核七rank B1/K16三轮候选，不代表已选最优训练预算。

动态下游命令、坐标与fit政策见 [RoomGenBench接口](fastfill-v2-roomgenbench-interface-20261006.md)。真实learned mesh生成器运行、实际mesh碰撞／支撑、physics、Solver及持久Host均未在本轮验收；已运行的是合成GLB接口回归，失败场景不冒称可提交。

设计逐项纠正见 [设计审核](fastfill-v2-design-audit-20261006.md)，全量参考代码与论文范围见 [参考审核](fastfill-v2-reference-code-audit-20261006.md)，房间数量变化与字段覆盖见 [多源记录](fastfill-v2-multisource-20261006.md)。
