# FastFill 参考代码与论文：全量阅读结论

审核日期：2026-10-06。参考代码以 Git submodule 固定，不把上游 detector 的训练结果当作 FastFill 的布局结果。详细反例、逐文件 hash 与阅读清单随本仓库保存于 [审核证据](evidence/fastfill-v2-audit-20261006/artifact-manifest.json)；原始执行输出另保存在本地和服务器审核目录。

## 阅读范围

| 对象 | 固定版本 | 实际全文范围 |
|---|---|---|
| V-DETR | `9062d75fe2c91e5d4a771b5325483fc330a3e827` | 53/53 源文件，10,488 行；另读 README、依赖、结果、arXiv 2308.04409v1 15 页全文及全页图、仓库 details-v2 一页图 |
| MinkowskiEngine | `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228` | 141 个 Python／文档／构建／示例／测试文本，23,316 行；91 个 C++／CUDA／vendor／native 测试文件，29,532 行；官方 CVPR2019 论文 10 页全文 |
| RoomGenBench | `30f2e059d046ccd4aa928d0109a09a52f11693b5` | 本次核对输入合同、prepare、assemble、render、site、资产状态和评分数据流；不是全仓库模型实现／权重验收 |

MinkowskiEngine 合计 232 个文本文件、52,848 行。244 个 tracked 文件均有完整性库存，其余图片等二进制不能按 hash 算作源码阅读。全文覆盖与执行分支覆盖不同：本轮没有编译这些旧 native 扩展、运行原 detector 训练、复现论文 AP 或完成真实 mesh／physics 验收。

## V-DETR：公式可参考，几何实现不能直接复制

用户设计 §11 的主要公式基本正确：proposal-normalized center L1、proposal-relative log-size L1、yaw 分类及 GT-bin residual Huber、matching 的负 overlap 与训练的 `1-overlap`、ScanNet 单 angle bin、3DV-RPE 与重复 GT 的 one-to-many。六项 loss 是主要 decoder 监督，还有 point classification 和 auxiliary 层等项。`gious` 字段受 IoU／DIoU／GIoU 配置控制；原角度 matching 默认关闭。

明确缺陷和移植限制包括：

- 原 parser 未定义 constructor 使用的 `random_fps`；默认 no-color encoder 又读取未赋值的 `xyz`。README 的新训练入口未自包含。
- 原 GIoU 对 0.1 mm 薄轴同框得到 **18.052633**，正确值应为 1；旋转同框 π/4 的交集又可错误归零。criterion 只判断正 GT yaw，只有负 yaw 时误用非旋转路径。这些均有 CPU 定向反例。
- 可选 shared self-attention 混排 batch；CE background 越过尺寸 anchor 表；superpoint validation 使用未定义 choices；非 tensor gather 错复用上一张量。默认／可选影响范围分别记录，不能称所有普通 ScanNet 路径均触发。
- 固定发布树缺少部分 ScanNet 导出文件、Cython 源码及 SUN RGB-D dataset 实现。旧二进制不是当前 Python／CUDA ABI 兼容证据。

FastFill 使用独立 room-normalized bottom-center 与 log-size ratio SmoothL1、BEV convex-hull GIoU。它们不是原 proposal L1 与 3D AABB-enclosure GIoU 的逐项复现。原 V-DETR camera box 的 yaw 映射还需反号及 center 转換；同为 Z-up 不能省略角点等价核验。

完整报告：[V-DETR 审核](evidence/fastfill-v2-audit-20261006/reference-vdetr/audit.md)。

## MinkowskiEngine：下载源码不等于环境已编译

当前服务器 FastFill 环境没有安装 MinkowskiEngine 或 Backend。FastFill 的 Qwen 条件骨干和连续 heads 不引用它；它是原 V-DETR 点云骨干的依赖。不能把旧 Python3.8／torch1.12／CUDA11.3 requirements 装进已验证的 Qwen 环境。

Python 层确证包括 SyncBatchNorm 转换后被旧 BatchNorm 覆盖、`sparse()` 原地改坐标缓存、TensorField 算术与 `cat_slice` 使用过时 API、collation 只检查总长度而允许每场景特征错配。源码抽取的 CPU 反例实际执行；它们没有运行 native 后端。

原生层另确认 label 冲突写错 voxel、空 kernel-map 悬空引用、GPU sparse-average 用原 row ID 访问 packed counts、stride host over-read、COPY_GEMM shared-map 容量不足、direct max-pool 的负数／空洞索引问题及 handle ownership 缺失。证据是完整 C++／CUDA 控制流和标准库整数机制模拟，**不是 GPU sanitizer 实跑**。报告单列未找到生产 caller 的 vendor/helper 问题，避免混入默认路径缺陷。

`setup.py` 在 metadata 入口也删除 build、卸载已有 ME，因此本轮没有执行它。构建依赖、CUDA stream、ABI 和实际 native 前后向仍须隔离环境验证，不能写成服务器已复现这个 detector。

完整报告：[Python／构建／论文审核](evidence/fastfill-v2-audit-20261006/reference-minkowski-python/REPORT.md)、[native 审核](evidence/fastfill-v2-audit-20261006/reference-minkowski-native/report.md)。

## 对主方案的约束

参考论文支持连续几何监督及检测框机制；Minkowski 卷积论文主要是稀疏语义分割，不提供本任务的 bbox loss。FastFill 仍应由自己的字段资格、实际 tokenizer 预检、合法身份匹配、head 梯度、heldout 布局及 RoomGenBench mesh 结果验收。公开参考代码存在缺陷不推翻这条研究路线，也不授权将主集缩成单一来源。

初始化三个参考源码：

```bash
git submodule update --init V-DETR MinkowskiEngine RoomGenBench
```

该命令只检出固定源码；RoomGenBench 的方法模型、第三方子模块、权重、Blender 与外部服务是另外的运行依赖。论文版本、SHA 与来源在各审核报告的 paper manifest 中，不宣称已核对所有论文的最终出版版本。
