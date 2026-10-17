# MinkowskiEngine Python/API/构建/示例/文档/论文只读审查

审查日期：2026-10-06。源码：本地 `MinkowskiEngine/`，HEAD `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`（包内版本 0.5.5）。Native `src/`、`pybind/`、C++/CUDA 测试另由 `/root/multisource_verify` 审查；本报告不将其静态结论冒称为 Python owner 的 GPU 执行结果。

**这个 checkout 是参考源码，不是当前 FastFill 环境已安装并验证的算子依赖。** 本次在本机及服务器 FastFill interpreter 的中立目录检查：两处均没有 MinkowskiEngine distribution、MinkowskiEngine Python module 或 MinkowskiEngineBackend module；当前服务器也没有 ninja/open3d distribution，`nvcc` 不在 PATH。未进行环境安装或 GPU 编译，不能据此声称完整磁盘没有 CUDA toolkit。

当前 FastFill 的 Qwen + 连续 position/log-size/yaw heads 路线不依赖 MinkowskiEngine。`fastfill/v2/README.md` 明确写出两条比较路线都不需要 ME。ME 可作为旧 V-DETR 点云编码环境的参考，但不能用“下载了仓库”“论文用了稀疏卷积”证明 FastFill 的布局监督、数据标签或真实资产闭环正确。

本报告发现多个公开 Python API 的确定缺陷、构建配置问题和示例评测口径问题。没有修改源码、训练包、模型、数据或环境，没有运行示例下载器、setup.py、ME native 后端或真实训练。

## 覆盖与执行证据

| 范围 | 全文文件 | 行数 |
|---|---:|---:|
| 生产 Python package | 31 | 8,316 |
| docs 文本/config | 35 | 2,662 |
| examples | 21 | 5,043 |
| Python/shell/README 测试文件，含 tests/cpp Python wrappers | 43 | 5,896 |
| root/build/ancillary | 11 | 1,399 |
| 合计 | 141 | 23,316 |

全部 244 个 git-tracked 文件都有 SHA256/字节数库存；本 owner 对 141 个非 native 文本逐文件全文阅读，其余为另一 owner 的 native 文本或图片。并非只搜索 loss 或只读 README。完整逐文件表：[READING_CHECKLIST.md](READING_CHECKLIST.md)，机器库存：[reading_inventory.json](reading_inventory.json)。组合 tool 输出截断的地方另行补读，不把截断输出标成完整阅读。

独立隔离反例从当前文件 AST 提取**未经改写的指定类/方法/函数**，用真实 CPU PyTorch 操作和必要的坐标/后端容器 mock。这样可以执行纯 Python 缺陷而不导入 ME native 扩展。实际运行环境为 macOS arm64、Python 3.13.1、torch 2.6.0、numpy 2.4.6。13 个预期异常检查成立，另有 5 个确定错误行为；额外 Timer 的 NumPy2 异常也记录。它们不是 18 个独立生产缺陷的夸大计数，也不是 native CUDA suite 的通过数。

脚本：[python_api_repros.py](python_api_repros.py)。实际结果、原函数 SHA256、环境：[python_api_repro_results.json](python_api_repro_results.json)。复现命令：

```sh
python3 outputs/fastfill_v2/audit-four-20261006/reference-minkowski-python/python_api_repros.py \
  --repo /Users/slian/Desktop/3D/Worldedge/OptiScene/MinkowskiEngine \
  --output outputs/fastfill_v2/audit-four-20261006/reference-minkowski-python/python_api_repro_results.json
```

## 公开 Python API 的确定问题

以下路径相对 ME 仓库。严重度针对“将本 checkout 直接纳入新训练环境”的影响；它们不是 FastFill v2 自身的新增 C1–C4 缺陷。

| 编号 | 位置与触发 | 结果与含义 | 建议 |
|---|---|---|---|
| MP1 · P1 | `MinkowskiEngine/MinkowskiNormalization.py:148`，`MinkowskiSyncBatchNorm.convert_sync_batchnorm(MinkowskiBatchNorm(3))` | 构造 SyncBatchNorm 后继续遍历原层 child `bn`，把新同步层覆盖回原 BatchNorm1d。实际输出 outer 是 MinkowskiSyncBatchNorm，但 inner 是原 BatchNorm1d，`same_original_inner_object=true`。分布式训练可静默没有跨 rank 同步。 | 递归时不要用原 `bn` 覆盖转换后的 `bn`；检查真实类型并做双 rank 统计同步回归。 |
| MP2 · P2 | `MinkowskiEngine/MinkowskiTensorField.py:458`，field+field 或 field+torch.Tensor | `_binary_functor` 访问不存在的 `coordinate_map_key`，当前对象只有 `coordinate_field_map_key`；两个分支均 AttributeError。field+field 还传错误 constructor keyword。 | 统一正确 field key/keyword，保留 manager 资格检查；覆盖四种二元算术与合法 field 对象。 |
| MP3 · P2 | `MinkowskiEngine/MinkowskiOps.py:398`，dense 输入默认 `remove_zeros=True` / 提供 coordinates | 默认却走 `to_sparse_all` 保留零点；有 coordinates 且默认 remove_zeros 时走 `to_sparse` 丢弃 supplied coordinates。与自身 doc、CHANGELOG 的 override 策略相反。 | 条件应在没有 supplied coordinates 时按 remove_zeros 选路；指定 coordinates 应使用它。反例测的是精确分支，不是模拟 native 零点行为。 |
| MP4 · P1 | `MinkowskiEngine/MinkowskiSparseTensor.py:346`，`SparseTensor.sparse()` 非零最小空间坐标 | `coords=self.C[:,1:]` 为 view，`coords -= min_coords` 改写对象缓存坐标。真实 CPU tensor 示例 `[[0,5],[0,6]]` 调用后变成 `[[0,0],[0,1]]`。native manager 的坐标表不会因这个 Python 原地操作同时重建，后续 Python/native 几何可分歧。 | clone 后计算；验证调用前后原 C 与 manager 坐标均不变。 |
| MP5 · P2 | 同一 `sparse()`，只传文档允许的 max_coords | max 分支校验误写 `min_coords.numel()`，NoneType AttributeError。max 路径另调用 Python manager 没有的 `get_batch_indices()`，未在本隔离反例中执行到该处。 | 分别校验 min/max；按当前 manager API 求 batch 范围；验证 max-only 和显式 min/max。 |
| MP6 · P2 | `MinkowskiEngine/MinkowskiSparseTensor.py:631`，文档示例的 `soutput.cat_slice(sinput)`，X 是 SparseTensor | 分类型之前无条件调用 `X.inverse_mapping(self.coordinate_map_key)`；SparseTensor 的 inverse_mapping 是 Tensor，实际 TypeError `'Tensor' object is not callable`。TensorField 的同名成员才是 method。 | 在分支中取 mapping，验证重复点恢复/拼接长度与顺序，不能只修调用后忽略 feature 长度。 |
| MP7 · P2 | `MinkowskiEngine/MinkowskiKernelGenerator.py` custom offsets / `get_kernel()` cross/transposed | 多元素 IntTensor offsets 在 constructor 被用于 truth test，RuntimeError；cross get 对 list 做 `%2`，TypeError；transposed get 用不存在 `self.stride`，AttributeError。CUSTOM converter 对 dtype 使用 `isinstance(dtype,torch.IntTensor)` 也错误。 | tensor 存在性用 `is not None`；使用 kernel_stride；把形状/奇偶验证统一；确认 native 支持的 RegionType。|
| MP8 · P2 | `MinkowskiEngine/utils/collation.py:96`，coords 样本长度 [1,2]、feats [2,1] | 只检查全局总点数相等，接受每样本不匹配。拼接结果把 batch0 第二个特征 101 配给 batch1 坐标20。错误输入应在边界失败，当前静默跨样本错配。不同列表数还可能被 zip 截断，labels 长度也未系统验证。 | 检查列表数/每样本 N、维数和标签语义；分类 labels 可是每场景1条，不能简单强制 labels=N；按输入协议区分。 |

MP7 的 CUSTOM converter 反例为到达 dtype 检查提供了 mock `RegionType.HYBRID`。当前 native enum 的具体存在性由 native owner 确认：若该 enum 不存在，CUSTOM converter 会先在 `elif RegionType.HYBRID` 上失败。这不影响 constructor 多元素 truth-test、cross list `%`、transposed 缺属性三项确定反例。`get_kernel()` 并非当前 convolution forward 的必经路径；convolution 通常直接把 generator 字段交给 native，因此不能把这些 helper 错误写成所有普通 cube 卷积必然崩溃。

还有公开/辅助 stale API：`MinkowskiStableInstanceNorm.forward` 用 v0.4 的 `x.coords_key/x.coords_man` 和旧 constructor 参数，当前有效 SparseTensor 会 AttributeError；SEBasicBlock/SEBottleneck 给已改为 `dimension` 的父类传 `D`，两个 constructor 都 TypeError。已跑隔离反例。它们不证明 V-DETR 默认主路径调用了这些辅助类。`utils/coords.py` 的旧 helper 未在 utils exports 开启，不能把未导出的旧代码与默认生产调用混为一谈。

其它静态观察保留为次级 API 问题：`MinkowskiNetwork.init` 仍访问旧 coords API；AdaptiveLogSoftmax wrapper 的统一 forward(input) 与底层所需 target/返回 tuple 不匹配；ChannelwiseConvolution 的 `coords` 参数未用于选择输出坐标。未为这些各自执行完整后端集成，不扩张确证计数。

## 构建/install 不能直接迁移

`setup.py:122–123` 在执行 setuptools setup 前，无条件 `rm -rf build` 和 `pip uninstall MinkowskiEngine -y`。所以即使 `python setup.py --name/--help` 也不能作为纯检查命令运行；本次仅读源码。它还调用裸 `pip`，不保证卸载当前 Python interpreter 内的 package。clone 和成功编译 `MinkowskiEngineBackend._C` 是两个不同状态。

构建参数存在确定连接问题：`--cuda_home` 被解析到本脚本变量、只打印，未设置 `torch.utils.cpp_extension.CUDA_HOME` 或环境；扩展工厂实际使用后者。`--blas_library_dirs` 解析成 list 后被直接格式化为 `-Wl,-rpath,['/path', ...]`，没有为 Extension 传 library_dirs 搜索路径。显式 BLAS 路径不能按文档保证生效。缺省 BLAS 检测依赖 `numpy.distutils.system_info`，当前 NumPy2/Python3.12 环境该旧 API 不可依赖；应显式处理新的构建协议。

| 来源 | 宣称/固定环境 | 与当前环境的关系 |
|---|---|---|
| ME `requirements.txt` | torch>=1.4,<1.7；numpy 未固定 | 与同仓 README torch>=1.7 矛盾；不能作为当前 lockfile。 |
| ME README | Python>=3.6、PyTorch>=1.7、CUDA>=10.1.243、CUDA与PyTorch一致、GCC>=7.4、ninja | 只给历史最低要求，没有 torch2.13/CUDA12.6/H20Z 组合的构建回归。 |
| ME quick_start | 更早的 torch>=1.3/GCC>=6 | stale 说明，需以具体编译版本验证。 |
| ME Dockerfile | PyTorch1.12.0/CUDA11.3，ARCH list 止于8.6 | 不是 H20Z 当前编译验证；不能据无9.0条目就断言所有 PTX运行必失败，也不能据此承诺 Hopper 构建。 |
| V-DETR README/requirements | conda Python3.8；torch1.12.1+cu113；numpy1.19.5/scipy1.5.1/mmcv-full1.6.1；另编译 ME/pointnet2/cython | 属于旧点云检测实现环境，应隔离重现，不能直接向 FastFill env pip install 旧 requirements。 |
| 当前服务器 FastFill env | Python3.12.14、torch2.13.0+cu126、numpy2.4.6、transformers5.14.1、accelerate1.15.0 | 已有 CUDA PyTorch 不代表有 nvcc/native 编译链；neutral cwd 没有 ME 或 Backend。 |

本机/服务器此次 metadata/toolchain 只读回执：[environment_presence.json](environment_presence.json)。服务器完整先前环境回执在 `../../../multisource-20261006/live-environment-recheck.json`（注意这是其它任务的更大回执）；本报告只引用 package/CUDA 环境，不将其 pilot 结果等同 ME 运行。

旧 Makefile 仍含历史 compute3.x/caffe2/C++14 选项，CHANGELOG 0.5称不再作为安装方式；不要用它替代 torch2.x 扩展构建验收。`tests/cpp/setup.py` 依赖 CXX env 和 CUDA_HOME，即使 CPU target include_dirs 也 join CUDA_HOME；test_all.sh 指定 `region_cpu` 不在 SOURCE_SETS，应为 kernel_region_cpu。这些是测试构建表面维护问题，不是已经完成的 native 运行回归。

## 示例、测试与评测口径

ME 示例并不实现我们的“给房间/请求对象→每对象 bbox”任务。主要示例包括 ModelNet40 分类、室内语义分割、线点分类、椅子体素重建/补全/VAE：CE、occupancy BCE、VAE KL+BCE 各有对应监督。重建时用 GT target keep masks 保留目标 voxel 是其显式训练机制，不能无分析移植到 FastFill 条件输入或推理约束。

ModelNet40 分类示例有实际评测协议问题：`classification_modelnet40.test(...,phase='val')` 内部无条件调用 make_data_loader('test')，ModelNet40H5 也将 val/test 都映射到 test；训练期间“validation”指标来自最终 test。若复制为正式模型选择协议会使用 test 信息，不能把打印的 validation 当独立 dev。当前源码未运行数据下载/分类训练，本项来自完整数据流静态核查。

examples/classification_modelnet40、reconstruction、completion、vae 调用 DataLoader iterator `.next()`；在本机实际 PyTorch2.6 iterator 该属性不存在，已记录 AttributeError。`examples/common.Timer.reset` 用 NumPy2 删除的 `np.Inf`，已记录 AttributeError。reconstruction.InfSampler 的 `shuffle=False` 会对 int 调用 `.tolist()`；completion visualization/VAE visualization 保留旧 SparseTensor/coords API。它们说明历史 demo 需要适配，不能以代码存在当成新环境可运行证明。

测试全文已读，但**没有执行原 ME suite**：缺 native 扩展，部分模块 import 时下载 `1.ply`/weights；CUDA tests 不都用 skip；convolution 默认有100,000次 leak loop，interpolation 10,000,000次；若盲目 run 全 suite 会触发网络/资源操作并长时间运行。重复 test 方法覆盖先定义的方法、TensorField 若干方法未以 `test_` 命名不会由 unittest发现；广播 CPU/GPU差值检查未取绝对值；部分测试只 print/检查均值方差，无法证明逐点严格对齐。gradcheck wrapper 向 modern torch 传被删除的 check_sparse_nnz keyword，实际精确函数 TypeError，需先适配测试框架后才能引用新通过数。

Native owner 已告知 GPU sparse-average 对 gapped row ID 的索引问题，以及 origin_map/global pooling 边界问题；细节/复现由 [native 报告](../reference-minkowski-native/report.md) 负责。本 Python owner 独立确认可达入口：spmm_average 的公开 wrapper 没有要求 row IDs 稠密；Global Sum/Avg/Max 的 default 是 PYTORCH_INDEX 并原样交给 native；GlobalMax 明确处理 TensorField key。非连续 batch ID 受 CHANGELOG 0.3.2 说明和 `tests/python/sparse_tensor.test_extraction` 的 [0,2] 输入支持，不能事后把所有 gapped batch 认定为非法。此处未运行 native/CUDA反例。`utils.quantize` 的 Tensor路径只调用 `.int()`；真实CPU PyTorch int32 非连续切片 `(stride=(3,2))` 会原对象返回且不转contiguous，已保存 [quantize_stride_receipt.json](quantize_stride_receipt.json)。native owner 确認 CPU quantize helper按dense `row*ncols`走data_ptr，该slice逻辑两条重复`[[0,0],[0,0]]`会被dense指针读成`[[0,1],[0,0]]`；本回执仅证明Python stride保留与静态地址机制，不宣称运行了ME binary。

## 论文来源与科学相关性

下载并全文读取官方 CVF 10页原文：Christopher Choy、JunYoung Gwak、Silvio Savarese，CVPR2019，*4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks*。官方 PDF、逐页提取、SHA256和来源回执在 [paper/receipt.json](paper/receipt.json)；图像没有另行视觉逐图验证，数学/损失结论基于原文完整文字和公式提取。

论文提出 sparse COO 表示、广义卷积 Eq.(3)、任意 input/output coordinate set、quantization/maps、sparse pooling、混合 space/time kernel、7D space–time–color TS-CRF。实验主线是 3D/4D **语义分割**：cross-entropy（p4），mIoU/mAcc，Momentum SGD与Poly schedule（p6）。没有 room-object bbox 的 position/log-size/yaw loss、合法可交换组 Hungarian、资产解析/Host 提交验收协议。README 的 detection 链接属于另篇 GSDN ECCV2020论文。

论文坐标表示把 batch放最后，而此代码 v0.4+ batch放第一，migration docs/code warning明确；照抄论文表示而忽略当前 collate协议会错位。论文可支持“稀疏点云编码器”技术选择，不能替代本项目 bbox 标签有效性、模型可表达范围、优化窗口、稀有 yaw 曝光或物理验收的证据。

## 结论边界

全文读完不等于所有 branch 已执行，隔离 Python 反例不等于 native/GPU 集成通过。要重现旧 V-DETR，应另建固定版本环境，修复/绕过被实际使用的上游缺陷，验证 `_C` 编译、GPU架构、native前后向/梯度/stream/空输入、DDP SyncBN与实际数据路径；不要改变已工作的 FastFill env 来满足旧依赖。

本次不能给 ME 当前 checkout 签“全部正确”，也不能据 ME 与 FastFill不同任务而否定 FastFill研究路线。FastFill的放行应由它自己的多来源 manifest、真实 tokenizer preflight、有效监督/累积窗口回归、Qwen pilot与heldout评测决定。ME 此处是已核源码/论文的参考，尚无安装或 native 训练验收承诺。
