# V-DETR PointNet2／几何子审计，2026-10-06

冻结 HEAD：`9062d75fe2c91e5d4a771b5325483fc330a3e827`。本子任务只读上游，唯一写入是本证据目录。没有安装依赖、编译 C++／CUDA、执行 CUDA kernel、修改 V-DETR／FastFill／训练数据，或运行正式训练。

## 阅读与执行覆盖

`native-coverage.json` 逐文件记录 SHA256、字节数、行数和实际全文阅读范围。PointNet2 全部源码是 **20 文件：6 headers＋5 C++＋4 CUDA＋5 Python**；不是 23 文件。本任务完整读过这 20 文件和 `box_ops3d.py`／`box_util.py`，另外完整读过最初分配的 `pc_util.py`／`nms.py`／`random_cuboid.py`／`cython_compile.py`。合计 **26 源文件、4,323 行**。生成的 Ninja／egg 元数据另外 **6 文本文件、72 行**；3 个生成二进制只检查 hash 和格式，没有声称全文源码阅读。egg 内的 7 个文本成员也全部显示并读过，native binary／pyc 未反编译。

`source-read-copy.txt` 是所读 32 个文本文件的逐行副本，方便总审计做路径集对账；生成副本或重算 hash 本身没有被当作“已经读代码”的证据。`criterion.py` 与 `datasets/scannet.py` 只在本任务读了相关调用片段，完整阅读由父任务负责。

`reproduce.py` 执行的是未修改的生产几何函数和 Python helper。当前本地 `utils/__init__.py` 会因缺少可选 `plyfile` 阻断 import，因此脚本跳过 **package initializer**，保持原 `box_util.py`、`misc.py` 的函数不变。PointNet2 helper 的导入使用空 `_ext` sentinel，**没有调用任何假 kernel 或把它当作 native 测试**。所有结果保存在 `reproduction-results.json` 与 `reproduction-stdout.txt`。这是 CPU 数学／Python 反例，不是整体 V-DETR 环境验收。

## 已证实的几何问题

| ID | 位置，相对 V-DETR | 反例与原因 | 影响范围 |
|---|---|---|---|
| VNG-01 | `utils/box_util.py:449–465,614–618`；`box_ops3d.py:137–147` 同类实现 | `box3d_vol_tensor` 把 squared edge length clamp 到 `1e-6`，实际把每轴最小边长改成 **1 mm**。交集和 enclosure 仍用实际 corner。相同轴对齐 `[2,0.0001,1]` m 框的 GIoU 为 **18.0526332855**，`[2,0.00001,1]` 为 **198.0050201416**，预期都为 1。 | 即使非旋转分支也成立；直接移植到具有薄轴的 FastFill 目标不安全。未测量原 ScanNet 数据的触发数量。 |
| VNG-02 | `utils/box_util.py:562–565,593–594`；Cython caller `669–698`；`box_ops3d.py:383–415` | rotated path 用固定的两个对角 corner 计算“非旋转面积”，零面积就跳过真正 polygon clipping。这些点在旋转后不是 min/max。相同 `[2,1,1]` m 框、yaw=π/4 或 π/2，真实交集=**2 m³**，tensor/JIT `return_inter_vols_only=True` 返回 **0**，float32／float64 都复现。 | 旋转路径确证；当前 ScanNet angle=0 配置通常不触发。缺失的 Cython extension 没有执行，但 caller 确实把这个错误 quick-reject 值传给它。 |
| VNG-03 | `utils/box_util.py:52–61,97–103,393–407` | polygon clipping 用严格 `>` 判断边界，coincident／parallel edge 没有容差和 denominator 保护。相同 `[2,1,1]` 框、yaw=π/4，NumPy IoU 返回 **0**，并实际记录除零／invalid warning；真实 IoU=1。yaw=0.3 时 float32 tensor 交集为 **1.95533657**，真实值=2，float64 对照=2。 | NumPy 评测和旋转 tensor fallback 的共同数值问题；不能靠正常非退化随机例子证明 coincident edge 正确。 |
| VNG-04 | `criterion.py:635` | `rotated_boxes=torch.any(gt_box_angles > 0)` 漏掉只含负 yaw 的 batch，并且只看 GT。按实际 predicate，yaw=-0.3 的同框被送入非旋转公式，交集 **2.4976348877 m³ > 真正 2 m³**，GIoU **1.1028376818**。 | 扩展到 signed yaw 的条件分支问题，父任务全审 criterion；ScanNet K=1 全零 yaw 不触发该特定反例。 |
| VNG-05 | `utils/box_util.py:580,616–623` | 全零 corners 的 GIoU 返回 **NaN**。先除以零 enclosure，再乘 `good_boxes=0`，`NaN×0` 仍是 NaN。 | malformed／零体积输入的边界问题；没有证明正常 ScanNet predictions 产生这种输入。仅将 padded GT 置零也不足以证明一定触发：预测 enclosure 可能仍大于零。 |
| VNG-06 | `utils/box_ops3d.py:475` | 模块 top level 调用 `autojit`，全文件没有定义／import。实际 import 确证 **NameError: name 'autojit' is not defined**。 | legacy helper 模块不可导入；父任务核查实际训练 caller，不能当作 `box_util` import 或全训练入口失败。 |

VNG-01 的精确中间量已保存在 JSON，不是轴顺序误解：camera bounds 是 x=±1、y=±0.5、z=±0.00005；交集与 enclosure 均为 `0.00019999999494757503 m³`，生产单框体积却为 `0.0020000000949949026 m³`。union=`0.00380000019504223 m³` **大于 enclosure**，IoU≈0.0526315749，GIoU 第二项≈**+18.0000014552**。这里不是 intersection 大于 production volume，而是 inconsistent clamp 导致 union 大于 enclosing volume。

对照：另用独立 axis-aligned scalar formula 核验 **200 对**不同中心、不同尺寸的框（每轴 0.1–4 m，固定 seed），GIoU 最大绝对误差 **1.4418837346×10⁻⁷**，反传 corner 梯度全部有限。这支持正常尺度非旋转分支的公式方向，不覆盖薄轴、零体积、rotated edge topology 或所有 dtype／设备。

`nums_k2=None` 的 rotated JIT 调用也实际失败，因为签名是 `Tensor`，未声明 `Optional[Tensor]`，内部虽有 None 分支仍不提供可用 Optional 合同。现有 criterion 传入 count tensor，所以仅作为 API 兼容限制记录，不计为默认调用错误。

## PointNet2 Python helper 的确证问题

| ID | 位置 | CPU 结果和范围 |
|---|---|---|
| VNP-01 | `pointnet2_utils.py:384–388,419` | `GroupAll.__init__` 没保存 `ret_grouped_xyz`，forward 无条件读该属性。传 False／True 均实际得到 `AttributeError`。只有 local grouping 的正常 caller 不会触发这个 global grouping 路径。 |
| VNP-02 | `pointnet2_utils.py:43–45` | `RandomDropout.forward` 调用 `pytorch_utils.feature_dropout_no_scaling`，全文 helper 没这个函数，实际 `AttributeError`。另外传入 `self.train`（bound method），并非 `self.training`；这是修复缺失 helper 后仍要处理的模式语义问题。未证明当前训练实例化这个类。 |
| VNP-03 | `pointnet2_modules.py:117–119,200–202,299–301,444–446` | 构造函数原地修改调用方 MLP spec。复用 `[[0,8]]` 构建两次，外部 spec 变成 `[[3,8]]` 后再变 `[[6,8]]`。第二次网络期望的输入通道会受第一次影响。正常 caller 每次提供新 list 时不触发复用反例。 |

这些 Python 检查没有执行 `_ext`，不能证明 native extension 已安装或 kernel 梯度正确。随包 `pointnet2_test.py` 只做一项 CUDA float32 interpolation gradcheck，使用 `atol=rtol=0.1`；它不覆盖 BallQuery、FPS、gather/group、index/device validation 或 double precision。

## Native API 与 CUDA 静态边界审查

全部 C++ 和 CUDA 文件已经全文阅读。以下是可从源码证明的 boundary gap／可移植性限制，**没有 CUDA 硬件执行、race sanitizer 或 GPU fault 复现**，也没有证明正常内部 caller 传入这些恶意／错误 tensor。

1. **Shape、batch、index bounds 未校验。** `sampling.cpp`、`group_points.cpp`、`interpolate.cpp`、`ball_query.cpp` 只检查 contiguous、Float32／Int32、是否 CUDA。没有核对 dimensions、XYZ last axis=3、batch 相等、idx/weight shape 相等、index 在 `[0,N)`／`[0,M)`。例如 gather `points(1,1,2)` 与 `idx(1,1)=[2]` 会通过这些检查，`sampling_gpu.cu:18–19` 直接读 `points[2]`，越过该 tensor 两个元素；grad kernel `44–46` 会原子写到同样的越界地址。group `27–28,62–64` 与 interpolate `97–102,138–144` 也直接 dereference caller index。安全复现只分析地址公式，未启动错误 kernel。
2. **同设备与 CUDA guard 未校验。** 两个 tensor 同为 CUDA 可以位于不同 device；C++ 不比较 `.device()`，也不设置 `CUDAGuard`。全部 wrappers 用无 device 参数的 `getCurrentCUDAStream()`，依赖外部 current CUDA device。正常 DDP 可以正确设置 current device，但这不是 standalone op 的输入保证。
3. **空／极小集合没有完整合同。** FPS 没检查 N>0 或 N≥nsamples；`cuda_utils.h:17–20` 对 work_size=0 求 log，随后转换／移位没有有效定义。FPS 在 N=0、m>1 会读 dataset[0]；zero grid 还可能是 invalid launch。ThreeNN 的 known M=0 会输出初始 index=0 和巨大距离，随后 interpolate 若直接使用会访问空 features。BallQuery 在没有任何点落入球时保留初始化 index=0，既无 sentinel 也无有效邻域 mask；其返回值不是“该球确有一个合法邻点”的证明。
4. **FPS 不是一般坐标平移不变。** `sampling_gpu.cu:103–104` 排除 `x²+y²+z²≤1e-3` 的点，不止全零 padding；原点附近半径约 3.16 cm 的真实点也被排除。已保存 scalar policy illustration：3 个非零点 0.01／0.015／0.025 m 全被跳过，采两点为 `[0,0]`；整体平移 1 m 后为 `[0,2]`。这是源逻辑说明，不是 GPU执行结果；可能原意是排除 zero padding，但代码没有单独 padding mask。
5. **CPU／MPS 与 mixed precision 不支持。** C++ 的 CPU branches 明确 assert `CPU not supported`；所有浮点输入要求 Float32，reject FP16、BF16、Float64。不能把 FP32-only extension 当作原生 AMP／double backend。所有 backward grad accumulation 使用 atomicAdd，未提供确定性加总实现。
6. **错误处理会终止整个进程。** `CUDA_CHECK_ERRORS` 在 launch error 时 `exit(-1)`，不是抛出可捕获 PyTorch exception；异步内存错误没有在每次调用后同步检测。empty launch 或错误 tensor 的行为不能视为结构化失败协议。
7. **生成 binary 不能代替当前环境构建。** tracked `.so` 和 egg 是 Linux x86_64／CPython 3.8，旧 Ninja 记录 CUDA toolchain、C++14 与 sm70／75／80／86 flags，并包含原作者绝对构建路径。这里只核对 hash／metadata，没有加载该 native binary，也没有证明它匹配当前 Python／PyTorch／CUDA ABI 或服务器 GPU。`setup.py` 需要 CUDAExtension，sources glob 相对工作目录，原安装流程应在正确目录构建；这不代表全 repo root 运行 setup 自动可用。

原 tracked tree 没有 `utils/box_intersection.pyx`，虽然 `cython_compile.py` 引用它。没有读取或编译不存在的文件。CPU 反例明确走真实 `box_util` tensor/JIT fallback，Cython kernel 数值路径未验收。

## 其他完整读过的 geometry utilities

`pc_util.py` 的 canonical point cloud 是 **X-right、Y-forward、Z-up**，camera 转换是 `(X,-Z,Y)`。box helper 使用 **box center** 和局部 full lengths `(l,w,h)`，camera 里绕 Y 旋转；其正 angle 从 depth 的 +X 转向 **−Y**，与常用 Z-up positive CCW yaw 相反。接入 FastFill 的 bottom-center／positive Z yaw 时必须显式升高 `h/2`、转换中心轴并反号 yaw；直接传其底面中心／相同正 yaw 会错。这是不同协议，不是上游自己的坐标 bug。

已实际执行角点等价检查：尺寸 `[4,2,1]`、Z-up bottom-center `[1,2,3]`，几何中心 `[1,2,3.5]` 转 camera `[1,-3.5,2]`。V-DETR `box_angle=+π/2` 的**局部 +X 有向长轴**转回 depth 指向 `[0,-1,0]`，而 canonical `yaw=+π/2` 指向 +Y。由于 cuboid 有 π 对称，不能仅比较 ±π/2 的无序角点集来判正负，所以另外使用非轴对齐 `yaw=0.7`：传 `box_angle=-0.7` 后，八个 world corner 与 independent canonical Rz(0.7) 的最大最近点误差 **0**；传相同符号 +0.7，误差 **1.7709333747 m**。完整结果在 `coordinate_convention`。没有核实 MMCV native diff_iou 的实际 installed library，不能据此假定它与两种 yaw 符号自动一致。

GIoU helper 的 enclosure 是 **axis-aligned 3D bounding prism**，不是 BEV convex hull 或完整 3D convex hull；其 docstring 的“rotate along Z”与实际 camera Y-up 数学不一致。`box_ops3d` 的 legacy convex-hull variant 又含 non-differentiable SciPy hull；不可把名称当作与 FastFill BEV convex-hull GIoU 等价。

`nms.py` 的 faster defaults 是 AABB IoU；old_type 和慢版 2D 是 intersection／other-box-volume。same-class NMS 在 overlap 计算后乘 class equality，没有 OBB／semantic-yaw／请求 ID 保全合同。不应拿 detector NMS 删除 FastFill 请求来提高通过率。

`random_cuboid.py:75–77` 额外确证把 `target_boxes.sum()>0` 当作“存在 bbox”。有效 negative-coordinate center `[-9,-9,-9]`＋positive size `[1,1,1]` 的 sum=-24，center filtering 被跳过。控制原 RNG 选择后，cropped cloud 只含 `[-10,-10,-10]`、`[-9.9,-9.9,-9.9]`，却原样返回上述 **中心在 crop 外** 的 bbox，违反其已声明 center 过滤／至少一个 box 的策略。机器结果在 `ancillary_random_cuboid_negative_coordinate_case`。没有测量真实 ScanNet 的触发 incidence；父任务可合并这项重复阅读证据。

## 对父任务三个 Python 反例的交叉复核

额外读过父任务的 `python_repros.py`／JSON 及相关生产函数片段，三条论断成立，作用范围应保持分开：

- `random_fps`：`ModelVDETR.__init__:87` 无条件读取 args 属性，fresh parser 没有声明；它在 GenericMLP 等构造之前即失败。AST 用 None 代替其他构件不会导致这个缺属性错误，因此该反例有效。是 fresh CLI 的 construction blocker，但本任务没有实际运行完整原训练入口或 native dependencies。
- no-color：`run_encoder:259` 使用本地 `xyz`，它到 `280` 才被赋值；`use_color=False` 是 parser 默认，README 命令也没开启 color。这个路径在 list comprehension 中先失败，ME stub 不会被调用；它是补上／绕过 random_fps 问题后暴露的下一处默认 blocker。
- `ShareSelfAttention`：query/key 从 `(N,B,C)` permute 到 `(B,N,C)`，value 却原样 linear 后 reshape 成 `(B,N,...)`；B>1 会跨 batch 混排。父任务用零 query/key、identity value/projection 和 uniform attention，预期每 batch 独立均值，反例成立。`share_selfattn=False` 为默认，README 每 GPU B=1 也隐藏跨 batch 混排；不能把这项 optional bug 说成默认全部训练污染。

这些交叉复核不新增整文件阅读统计：模型／transformer／main 全文由父任务覆盖。父任务 CPU 结果使用 Torch 2.6.0，本子任务数学结果使用 Torch 2.13.0，分别保存版本，没有合并冒称同一环境执行。

## 复现与使用

在工作区根目录运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python outputs/fastfill_v2/audit-four-20261006/reference-vdetr/native-geometry/reproduce.py
```

需要当前已有的 torch／numpy／scipy／shapely／torchvision。没有自动安装依赖。脚本只覆写本目录的机器结果。

结论：正常尺度、axis-aligned 的 detector 参考公式提供了部分可复核证据；上游的 rotated geometry、薄轴 clamp 与 native boundary 合同不满足“原样用于任意家具 bbox loss”的资格。这里确认的是 pinned **参考 V-DETR** 的问题和移植边界，不是宣称当前 FastFill 已调用这些有缺陷的函数，也没有推翻连续 heads 可反传的主研究路线。
