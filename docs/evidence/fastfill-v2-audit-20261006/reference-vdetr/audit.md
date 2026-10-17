# V-DETR 固定版本全源码只读审查 · 2026-10-06

对象：`V-DETR/`，HEAD `9062d75fe2c91e5d4a771b5325483fc330a3e827`。本报告审查参考实现的实际行为，不将它的检测指标视为 FastFill 布局生成或物理闭环的验证结果。

**结论：用户设计 §11 对核心回归／监督公式的概括基本正确，但不能据此认定整个 upstream 可直接运行或直接移植。** 当前发布源码有新训练的入口缺口、可选模式错误和几何边界缺陷；尤其薄轴与旋转 GIoU 不适合原样移入 FastFill。这里只写审查证据，没有修复 upstream，没有安装旧环境、编译 CUDA、运行原 detector 训练／测试或调用资产／Host。

## 1. 实际覆盖与证据边界

| 维度 | 实际完成 | 没有完成 |
|---|---|---|
| 源码阅读 | 53/53 个 `.py/.cpp/.cu/.h/.cuh` 源码，10,488 行全文阅读；38 Python、15 native | 不等于每条分支执行；不含二进制反编译 |
| Python 分工 | 主审查者读其余 31 Python；子审查者读 20 PointNet 源文件＋两个核心 box 文件，另有四个工具重复阅读 | 文件合计按路径集合去重，不相加代理的重复覆盖 |
| 非代码 | README、requirements、LICENSE、`.gitignore`、公布 ScanNet 结果全文；PointNet setup／生成元数据也阅读 | 没有安装或验证旧依赖组合 |
| 论文 | 本地 arXiv **2308.04409v1** 15 页逐页提取全文阅读，并查看五张三页联系图；仓库 details-v2 PDF 一页是图，已单独渲染查看 | 未将 v1 冒称 ICLR 最终版；未复现实验、核对所有其他引用论文或独立确认图中指标 |
| 执行 | CPU 数学对照、未修改函数／AST 选取定义的定向反例；版本和绕过范围记录在 JSON | 没有 MMCV/Minkowski/PointNet 原生运行、GPU smoke、原模型训练、ScanNet 全量读取或真实物理测试 |
| 完整性 | 固定 HEAD 与 53 文件逐项 SHA256/行数；审查后逐项复核 | 既存未跟踪 `.DS_Store` 不是源码修改；二进制只能证明 hash，不证明行为 |

证据：[源码与阅读覆盖清单](source-read-coverage.json)、[源码 hash 起点](source-inventory-before.json)、[Python 反例输出](python_repros.json)、[Python 反例脚本](python_repros.py)、[native/geometry 覆盖](native-geometry/native-coverage.json)、[native/geometry 反例](native-geometry/reproduction-results.json)、[论文清单](paper-manifest.json)。

读取图工具不可用时按已授权回退直接全文读取；未用 hash 或关键词命中代替全文阅读。`utils/box_intersection.pyx` 与 `scannet/scannet_utils.py` 在固定树中缺失，不能将不存在的源码记为读过。

## 2. 用户 §11 的关键断言核对

| 断言 | 核对 | 代码／论文依据与限定 |
|---|---|---|
| center 采用 proposal-normalized L1 | 正确 | `criterion.py:640`：目标为 `(GT_center-pre_center)/(pre_size+1e-5)`，同预测 `center_reg` 比较；不是对归一化房间坐标直接做 L1 |
| size 采用 log ratio 的 L1 | 正确 | `criterion.py:516,643`：`log((GT_size+1e-5)/(pre_size+1e-5))`；模型 `vdetr_transformer.py:280` 用 `exp(size_reg)*pre_size` 解码 |
| angle residual 除以 `π/K`，GT bin 选择 residual，Huber | 正确 | `criterion.py:370` 起，GT class one-hot 选择对应 residual；Huber delta=1；预测 residual 解码乘回 `π/K` |
| matching 用负 overlap，loss 用 `1-overlap` | 默认 GIoU 正确 | matcher `-outputs['gious']`；`loss_giou` 为 `1-gious`。`iou_type` 可选 IoU/DIoU，字段名 `gious` 不代表它永远是 GIoU |
| 六项 loss 代表全部仓库监督 | 若如此表述则过度 | 六项是 decoder 的 semantic、center、size、GIoU、angle cls/reg；还有 encoder point cls、各 auxiliary 层重复监督、cardinality 诊断 |
| ScanNet `K=1`，angle=0 | 正确 | config 只有一个 bin，原标签全零；model 对 angle logits/residual 乘零仍保留 DDP 图连接。此设置不证明自由 yaw 学习 |
| 3DV-RPE 用点到 bbox 八顶点相对偏置 | 正确 | `GlobalShareCrossAttention` 的八个 CPB MLP、signed log、5D `grid_sample`，八份 bias 相加至 `qkᵀ`；论文 pp.4–5 Eq.1–5 |
| GT repetition 提供 one-to-many | 正确，须解释机制 | 单次 Hungarian 仍一对一；GT 被 repeat 后同一原对象出现多份列，形成对原 GT 的多预测监督。默认 repeat=5，不是“新增五倍真实数据”；首个 binary FFN auxiliary 使用未重复原 GT |
| 默认权重 GIoU2/class3/center1/size0.5、angle matching0、angle cls0.1/reg0.5 | 正确 | `main.py` parser；另外 matcher objectness0、point cls loss0.05、focal alpha0.25 |

角度分类 matching 的实际可选 cost 是**负 raw angle logit**，不是 CE，也不是负 softmax 概率。两个 angle matching 权重默认均为零；报告不把这个自定义 cost 选择一律定性为错误。若启用，应明确其尺度与语义，而不是说 matcher 与训练角度 CE 完全相同。

六项 decoder loss 只对 Hungarian 已匹配 proposal 的几何／角度项计数；分母为跨 worker all-reduce 的 GT 数均值（clamp 最小1），DDP 再平均梯度。semantic focal 同时覆盖未匹配负例。合法的可交换组限制属于 FastFill 协议，原 detector 的全 proposal×GT assignment 没有该条件协议。

## 3. Proposal、reference 与角度约定

实际主链为：RGB/XYZ 点云 → MinkResNet34＋FPN sparse features → FPS 4,096 点 → point semantic head 给初始 class／size anchors → light FFN 给粗 bbox → objectness top-1,024 → 八层 Transformer cross-attention／bbox heads。默认 `q_content=random` 会将所选 FFN 内容改成 learned embedding；这是发布代码的默认，不应仅凭论文描述断定一直使用 sampled content。

初始 center 来自 FPS 点位置。默认尺寸 anchor 来自 18 类平均尺寸；README 训练命令启用 `--hard_anchor`，此时都是 `[1,1,1]` m。第一 FFN center/size 回归相对这些 seed anchors。后续各层 bbox delta 相对**第一 FFN 所选 proposal 的 center/size**；用于 RPE 的 vertices/center/size/angle 每层则来自前一层预测并 detach。不是每层把上次 residual 再乘一次，也不是向 GT 提取 reference。

3DV-RPE 的 `object_coords` 选项对相对 offsets 做相应坐标旋转；ScanNet 无 yaw，该选项与 world coords 的差异不体现。默认 parser 的 angle_type 是空字符串，实质不进入 object_coords 分支。非 ScanNet 应显式配置并验证角点对应关系。

原 point cloud 用 `(X,Y,Z)`、Z-up；box utility 将 center 转到 camera `(X,-Z,Y)`，corner 生成使用 camera y 轴旋转。**同为 Z-up 并不保证与 FastFill 的 yaw 符号、长宽轴、底面中心相同。** 原 bbox center 是体积中心；从该 Z-up 体积 center 转 FastFill 底面 center 用 `center_z-sz/2`，反向需加 `sz/2`，再按需要转换 camera 轴。

CPU 角点等价检查确认：尺寸 `[4,2,1]` m、底面 center `[1,2,3]` 对应体积 center `[1,2,3.5]` 与 camera center `[1,-3.5,2]`。V-DETR `box_angle=+π/2` 的局部 +X 长轴指向 depth −Y；FastFill 正CCW yaw指向+Y。用非轴角 `yaw=0.7` 比较无序角点，`box_angle=-0.7` 误差0，直接传同符号+0.7误差1.770933 m。需反号是该协议映射的结果，不是原协议自身的错误。MMCV overlap 的角度约定未通过实际安装库验收，不能假定它与此两种符号自动等价。

GIoU 的 enclosing volume 是 corner-wise **轴对齐包围体积**，不是 BEV polygon convex hull×height。FastFill 若采用 convex-hull BEV GIoU，数学定义不同，应标为项目选择，不能把两种 score 都称“完全照抄 V-DETR”。IoU/DIoU 可选分支调用 MMCV 的 3D rotated overlap，DIoU 用轴对齐 enclosure 对角线长度；本轮未运行 MMCV。

## 4. 默认入口与发布包缺口

### V1 · 新训练构造读取 parser 未声明的 `random_fps`

`models/model_vdetr.py:87` 直接访问 `args.random_fps`；完整 `main.py:make_args_parser` 不声明或设置这个字段。用未修改 parser 生成 `--dataset_name scannet` 的 Namespace，再调用 AST 选取原 `ModelVDETR` constructor，得到 `AttributeError: 'Namespace' object has no attribute 'random_fps'`，尚未走任何 native 运算。README 新训练命令没有补此字段。

这是固定发布源码的新训练配置缺口。`--auto_test` 重载已有 checkpoint 只对当前 Namespace 已有字段赋值，也不会恢复此缺失字段（`hasattr(args,arg_name)` 条件）。未下载 checkpoint 或运行完整模型，不把此证据称为原 GPU 入口实跑。

### V2 · 默认 no-color branch 引用尚未赋值的 `xyz`

`models/model_vdetr.py:258` 列表推导遍历 `xyz`，而本函数稍后才给 `xyz=out.C[:,1:]*voxel_size`。应遍历入参 `point_clouds`。隔离调用未修改原 method、只提供 ME namespace import sentinel，确认 `UnboundLocalError`。parser `use_color=False`；README train 命令未加 `--use_color`。即使补上 V1，也会遇到此错误。默认不是“彩色配置已经避免该问题”。

### 发布依赖／工具未自包含

README 声明 Python3.8、PyTorch1.12.1+cu113、MMCV-full1.6.1、MinkowskiEngine、PointNet CUDA 与 Cython box intersection。MinkowskiEngine 未固定 commit。requirements 的 NumPy1.19.5/Scipy1.5.1 与旧 CUDA 环境不能直接当作当前 Qwen 服务器环境验收；`pandas` 是 Dataset 的无条件 import，但未列在 requirements。

固定 tree 缺 `scannet_utils.py` 和 `scannet/meta_data/`，ScanNet 导出无法仅靠本包完成；`cython_compile.py` 要求的 `box_intersection.pyx` 也缺失（`.gitignore` 全局忽略 `*.pyx`）。box utility 会回退 tensor 实现，所以缺 Cython 文件不是默认 loss 必然 import 失败；README 编译步骤则无法照做。PointNet 预编译 Linux Python3.8 `.so`/egg 已 hash，仅是旧构建物，未证明兼容当前 ABI/CUDA。

CLI 接受 `sunrgbd`，但 Dataset registry 只实现 `scannet`，SUN RGB-D config/adapter 被注释为未来发布。只能说论文验证了 SUN RGB-D，不能说该固定发布包已提供其完整训练路径。

## 5. 其他确证问题与触发范围

| 问题 | 精确位置／反例 | 影响范围 |
|---|---|---|
| **薄轴 GIoU 越界，非旋转也会** | `box_util.py:box3d_vol_tensor` 将平方边长 clamp到1e-6，等效最小边长1mm。相同 `[2,0.0001,1]` m box 的 enclosure/intersection=0.0002，计算 volume=0.002，union≈0.0038，self GIoU=**18.052633**；10µm轴=198.00502，正确均为1 | 默认 helper 的真实数值缺陷；尚未统计 upstream ScanNet 真实发生数量。FastFill 有薄物体，直接移植尤其危险 |
| **rotated GIoU quick reject 错用两个角点当 AABB** | `box_util.py:562` 起；相同 `[2,1,1]`、yawπ/4或π/2，Shapely self intersection2，原 tensor 返回0；float64仍失败 | 原 ScanNet GT0不触发；自由 yaw 数据不能直接用此旋转路径 |
| **GIoU rotated flag 只看正 GT yaw** | `criterion.py:635`：`any(gt_angle>0)`；仅负yaw−0.3的相同box误走非旋转路径，intersection2.49763、GIoU1.10284，均不合法 | GT yaw只有0/负值的扩展数据；ScanNet默认不触发。还应检查 pred yaw以及GT yaw共同适用的 geometry contract |
| **NumPy rotated self-IoU 的重合边退化** | `box_util.py` polygon clipping/convex hull，相同π/4 box self IoU0，记录除零/invalid warnings | AP helper 在旋转几何的边界反例；ScanNet轴对齐正常对照不支持说全部AP错误 |
| **零体积 helper 仍 NaN** | 原 tensor GIoU 对全零corners返回NaN，good-box mask乘零不能消除NaN | padding/degenerate几何防御；没有据此断言默认训练实际崩溃 |
| **DIoU 中心距离错含 width、漏 z** | `criterion.py:65` 的2D切片前三个字段为x,y,w；z-only距离平方实际0/期望4，width-only实际4/期望0 | `--iou_type diou` 可选。**上轮已确证，本轮重现；不是新发现** |
| **ShareSelfAttention 的 value 缺少 batch/sequence 转置** | `vdetr_transformer.py:639` 对 N×B×C直接reshape B×N×…；CPU B2,N3、uniform attention，期望 batch均值10/110，实际36.667/83.333 | `--share_selfattn` 默认关闭；B1不会暴露跨batch错配。默认普通 MultiheadAttention 不受此项影响 |
| **CE background 被当作 size anchor索引** | `model_vdetr.py:347–353`，CE point head19类包含background18，平均size表18行；指定背景为topclass则原forward IndexError | `cls_loss=celoss` 可选；默认 focal18类避开背景索引问题 |
| **all_gather_dict 非Tensor值复用上一Tensor** | `utils/dist.py:175` 在if外赋值；输入scan_idx=[7], point_clouds=[cloud]返回point_clouds=[7]Tensor，list首项则UnboundLocalError | 当前collator的point_clouds是list。默认AP未使用empty过滤，错误值通常未被消费；打开相应过滤将有错误。不是正确list gather |
| **superpoint validation 的 choices 未赋值** | `datasets/scannet.py:501`，sampling仅在augment=True分支；原getitem+本地tiny fixture、augment=False、use_superpoint=True，UnboundLocalError | 默认superpoint关闭；evaluate也未将superpoint传给model |
| **颜色auto contrast的常数通道产生NaN** | `datasets/scannet.py:213` 的`255/(hi-lo)`；常数RGB、p=1得到非finite数组 | color_contrastp默认0；这是可选augmentation对常数通道未防御 |
| **RandomCuboid用坐标和判断是否有box** | `random_cuboid.py:75` `target_boxes.sum()>0`；有效负坐标box使总和≤0，crop保留cloud但跳过box中心筛选 | 泛化到负坐标的crop资格缺陷；没有测量默认ScanNet发生率 |

所有上表数值／Python错误均有 CPU 定向输出；`negative_only` 与薄轴的几何细节在 child JSON。几何正常控制为 200 对 axis-aligned boxes、每轴0.1–4m，最大GIoU绝对误差1.44e-7、梯度有限。这支持相应普通域的计算一致性，不证明旋转、薄轴、零体积或每个拓扑切换点正确。

## 6. Native API 与遗留代码的证据

PointNet bindings检查部分contiguous/dtype/CUDA，但缺严格shape/batch/index范围/same-device验证，也缺device guard；CUDA gather/group/interpolate kernel直接使用index读写。畸形index可越界是源码上的确证风险，**本轮没有运行CUDA，没有证明正常FPS生成的index实际越界**。详细输入边界、launch/空输入、FPS origin排除规则与不可移植二进制说明见 [native 子报告](native-geometry/audit.md)。

`sampling_gpu.cu:103` 排除平方距离原点≤1e-3的点，使其采样政策不平移不变；CPU模拟展示近原点点集索引`[0,0]`、平移1m后`[0,2]`。这是源码政策的数学说明，不能冒称CUDA实跑。

两个遗留 Python helper 的确证反例：`GroupAll.forward` 读取constructor未保存的 `ret_grouped_xyz`；`RandomDropout` 调用不存在的 `feature_dropout_no_scaling`。`PointnetSAModule` 还会就地修改 caller 的 mlp specification，重复构造可累加输入维度。当前V-DETR主路径只用FPS/gather，不能把这些未用 helper 的崩溃称为默认训练崩溃。

`utils/box_ops3d.py:475` 调用未定义 `autojit`，完整import得到NameError。全树无生产caller引用该模块，默认criterion依赖的是`utils.box_util`。这是发布遗留模块不能独立import，不是默认GIoU的依赖问题。

## 7. 其他协议／评测限制

- `engine.evaluate` 注释说exact，但构造APCalculator没有传`exact_eval=True`，实际remove_empty_box=False；报告按实际默认解释。不能说其默认已经完成点落框过滤。
- 多进程validation用普通DistributedSampler；当样本数不能整除rank数时会padding。APCalculator按累积序号计数，没有按scan_idx去重。默认完整312条ScanNet val与8rank恰好整除，但子集／其他rank数需要检查分母；本轮未运行原分布式evaluation。
- `--rotated_nms` 分支只print，没有设置pred_mask；该选项未形成完成的NMS实现。ScanNet默认false。`angle_conf`在K1时angle_prob0，会将score全乘零；它属于不适用于无角度任务的可选组合。
- 从raw mesh计算的normals没有随axis-alignment、flip、rotation同步变换；no-color配置也没有追加normals特征。默认use_normals=False；若启用，不能把原normals当作正确增强后的坐标特征。
- `wandb_activate` 使用`type=bool`，文本`False`会被Python当True；默认true。这个CLI开关不能按常见`--wandb_activate False`关闭。未联网、未登录或暴露凭据。
- Dataset将每场GT固定padding到64；没有定义超过64时的资格处理。更多对象的FastFill不能照搬该cap与不完整mask策略。

这些静态边界不代表真实ScanNet数据已验证触发。修复或移植时应围绕相应触发加实际回归；不将建议写成已经完成的修复。

## 8. 论文与当前实现必须分开使用

arXiv v1 pp.3–5建立object normalization、3DV-RPE与固定初始proposal residual；pp.6–8/11给出ScanNet/SUN RGB-D实验、one-to-many与point数/encoder的ablation。v1论文表1的77.8/66.0包含TTA，README发布77.85/65.91明确无TTA，并解释cleaning/refinement后变化。它们是不同版本记录。本轮没有训练其模型或复现任何AP值。

v1段落有“RGB前三维、XYZ后三维”的输入文字，发布Dataset/model实际是XYZ前三维、RGB后三维。论文的size ratio示意也不能代替源码log-ratio细节。ScanNet训练mesh数量在v1 pp.6写约12K；本轮不依赖该数字确认真实dataset规模。复现应按实际冻结数据manifest与源码约定，不照抄纸面简写。

## 9. 对 FastFill 的具体含义

可以借鉴且已核实的部分是：连续head接受结构化监督；proposal/reference允许对center、log-size建稳定回归目标；angle residual应按GT bin监督；matching cost与可微loss分开；RPE可以把已知geometry写入attention。以上研究理由不等于FastFill已经实现了点云detector，也不等于新模型已经学会布局。

FastFill全条件主线仍是房间／固定物体／支撑／关系／请求对象→目标size、底面center、yaw→真实资产解析及actual bbox复核→Validator／有界repair→原子Host提交。V-DETR只解决已扫描场景中的点云检测，完全不负责资产、mesh、physics、Solver或Host。`directbbox`三字段是独立输入消融；从检测包默认只支持ScanNet不能推导FastFill应只训练SpatialLM或丢掉16个数据集家族。

MinkowskiEngine与PointNet CUDA是**原点云encoder**的依赖，不是Qwen条件语言模型＋连续head的必需依赖。需要原detector对照时应隔离旧环境；仅借鉴loss公式时应对项目自己的geometry定义、dtype、mask、thin/rotated/zero边界和合法matching组独立验收，尤其不能直接复制这里已失败的GIoU。

本次全源码审查支持上述有限结论，不能证明没有其他缺陷，也不能替代FastFill的全量data资格、真实tokenizer/context preflight、每项head gradient、save/load、小样本改善与实际闭环验收。
