# FastFill v1：单次调用的对象条件布局模型

v3.2 修复与独立数据版本的说明见 [RELEASE_v3.2.md](RELEASE_v3.2.md)。保留 v3/v3.1 作为原有实验数据；运行中的训练不要切换数据目录。版本号描述数据版本，模型文本协议仍为 FastFill v1。

一句话：给定房型、真实房间几何（含已固定在房间里的门、窗、柱子、楼梯等），以及**全部**待摆对象（每件的局部实际尺寸，可选描述与空间约束），模型**一次生成**所有对象的数值位置、yaw 和支撑关系。

$$F_\theta(R, G, O, C_{\text{layout}}) \rightarrow L$$

2026-09-24 晚改版（数据 v2）：不再设物体数上限，落地家具 + 放在家具顶面上的小物件全部保留（含 SAGE），倾斜、悬空的家具修正后保留；门、窗、柱子、楼梯等结构件和没有名字的落地箱子作为 `fixed`（固定件）告诉模型、不摆；挂墙、吊顶、柜子里的东西不摆也不再拒收房间。40960 只是模型的序列长度上限，训练参数由实测数据决定（§4）。

整体框架沿用 OptiScene：对象由上游给定，模型只负责摆放，单轮 Qwen3-8B + LoRA SFT。
在此基础上修正了 OptiScene 已核实的几处缺陷：
- 房间只给面积 → 改为给地面多边形；
- h/w/d 轴序混乱 → 改为局部尺寸，并约定 +X 为正面；
- 同描述物体按列表位置对应 → 每件物体一个 id；
- 训练与推理用的 prompt 不一致 → 改为同一个 prompt。

范围和协议以用户冻结的 FastFill v1 定稿为准，接口见附录 B，实现在 `interface.py`。

## 1. 流水线与文件

```
HF 导出(22 源) ──download──▶ 本地汇总 JSON ──adapters/*──▶ IR(统一约定) ──build──▶ train/dev/test ──train──▶ LoRA
                                                                                       │                        │
                                                                          *_rooms.jsonl(评测用 IR)      merge_lora ─▶ merged
                                                                                       └──────── evaluate ◀───────┘
WorldEdge 请求(附录 B) ──interface.request_to_room──▶ 模型一次生成 ──interface.placements_from_text──▶ placements(yaw_rad)
```

| 文件 | 作用 |
|---|---|
| `download.py` / `download_full.sh` | 下载导出数据。前者默认只下汇总文件（33 GB），后者用 `hf download` 逐个数据集下全量 |
| `adapters/*.py` | 每个数据源一份，把各自的约定转成统一 IR。每份都附有证据：front test、kits 源码、20 间房逐字段核对 |
| `scene.py` | IR 约定、SYSTEM_PROMPT、序列化、canonical 排序和 id、`parse`（永不抛异常） |
| `anchors.py` | 剔除结构件（`fixed_geometry` 挑出要告诉模型的固定件），推断落地、放在他物上、挂墙、吊顶四类锚点，保证支撑关系闭合 |
| `build.py` | 过滤和拒收规则、固定件、取整对齐、跨源去重、防泄漏划分、增强、约束抽取 |
| `split.py` | 用 house/scan、别名、家具指纹、布局指纹做并查集，按连通分量整体划分 |
| `validate.py` | 越界、支撑、分桶碰撞、OOR 检查，以及约束检查 `holds` |
| `train.py` | LoRA SFT：只监督答案，批量分词，按长度分组，支持 `--resume` 和 `--dry_run` |
| `evaluate.py` | 生成并打分，所有请求都计入统计，分布内和 held-out 分开报告，每项都附 GT 对照 |
| `interface.py` | 附录 B 请求与模型文本之间的双向转换（输出 `position_m` / `yaw_rad` / `support_parent`） |
| `serve.py` | OpenAI 兼容服务端点：给 EmbodiedGen 的 `FastFillBackend` 调用，内部走 `interface.py` |
| `tools/qa_report.py`、`tools/ablation_check.py` | 重建后的 QA（泄漏、闭合、约束、参考布局合法率、固定件统计）和两版只差约束的核对 |
| `merge_lora.py` | 合并 LoRA，保留训练保存的 tokenizer、原始及续训 manifest，并写出 transformers 4.x/5.x 兼容字段 |

## 2. 统一标准（所有数据源进训练前都转成这一套）

**坐标与位姿**
- 单位是米，右手系，Z 轴向上。
- 使用房间局部坐标，边界 AABB 的左下角为 (0,0)，地面 z=0。
- 位置取 bbox **底面中心**。
- 尺寸是对象**局部坐标**下的 extent `[sx,sy,sz]`，不是世界坐标下的 AABB。
- 局部 +X 是**语义正面**：训练源须通过正面约定核查；单个对象缺少可靠正面时，按 `no_front` 规则作为固定件或过滤，不要求模型预测它的 yaw。
- yaw 是绕 +Z 从房间 +X 转到对象正面的角度。
  - 接口层和 IR 用弧度（`yaw_rad`）。
  - 模型文本里写整数度，因为 `90` 是精确值且只占 2 个 token；`1.570796` 不精确，还要多 3–6 个 token。
  - 取整误差 ≤0.5°，对应 1 m 处约 0.9 cm。

**房间与对象范围**
- 边界：`polygon` 表示墙体，`hull` 表示扫描地面的凸包。只由物体范围拼出的代理边界不进训练。顶点数不设上限。
- 待摆对象：落地物体，加上放在其他物体顶面上的物体（带 `on`），所有源一视同仁（SAGE 的桌面小物件也算）。**物体数不设上限**，只要求每间房至少 1 件可摆的物体（只有 1 件的保留并标 `single_object`）；唯一的长度限制是模型的 40960 token（超长样本在训练时整条丢弃、不截断，`train.py --dry_run` 会把它们逐条列出来）。
- 不摆、也不拒收房间的物体：挂墙（画、镜子、挂柜……）、吊顶（吊灯类不管源数据标在多高一律按吊顶）、放在柜子里或不在支撑物顶面上的东西、床上的被子毯子。
- **固定件 `fixed`**：已经在房间里、模型不摆但必须避开的箱子，连同位姿一起告诉模型（`fixed` 列表：id、size、pos、yaw）。四类（`stats.json` 里按 `fixed:<类>` 计数）：
  - `structure`（`anchors.fixed_geometry`）：门、窗、柱子、楼梯、电梯井、壁炉、暖气片、隔断、栏杆、低梁等结构件，底部低于 1.5 m 的。贴面和无占地的构件（地面、天花、踢脚线、线板、墙板、插座、地漏……）、窗帘百叶纱帘、占房间 ≥80% 的板块不算；`wall` 盒子只有在墙线房间里伸进房间（离墙 10 cm 以外 ≥0.05 m²，即隔断）才算，扫描凸包房间里的 `wall` 一律不算（墙本来就在凸包里）。
  - `generic`：没有名字的落地箱子（`unknown`、`otherprop`、`object`……）：它们确实占着地方，但没有名字就没法作为待摆对象（id 为 `object_k`）。
  - `wall`：源数据标成挂墙、但箱子底就在地面上的东西（一整排从地面起的壁柜）。
  - `floating`：墙线（设计师）房间里悬空 ≤0.5 m、下面没有东西撑着的可挂墙家具（浴室柜、悬浮电视柜、搁架）：那是挂墙件，不摆，但要避开，保留它真实的高度。扫描源（InternScenes、MultiScan、Scan2CAD）的 z 有噪声，这类家具一律贴地摆放。
  - `no_front`：源数据不给正面朝向的可摆物体（InteriorGS 的定制柜）：答案要每件的 yaw，所以把它当嵌入式固定件告诉模型，房间保留（标 `no_front_fixed`），不再整间拒收。
  - 隔断板（`partition panel`、divider、screen）按结构件处理，不当贴面。
  - 统一过滤（`build._keep_fixed`，服务端 `interface.py` 同一套）：顶面低于 15 cm 的不算（地垫、踢脚、门把手这类）；沉到地面以下的截到地面；整个在房间外的丢；占房间 ≥50% 的丢。**结构件（柱子、楼梯、门、窗……）永远保留**，即使参考布局里有家具压在上面（那会作为参考布局自己的固定件碰撞记入评测的 GT 值）；**非结构件**（无名箱子、落地的挂墙件、悬空挂墙件、无朝向件）被已摆物体盖住 ≥50% 时才丢（那是同一件家具的第二个标注框）。
  - 固定件和待摆对象共用一个 id 命名空间（固定件先编号），答案里只出现待摆对象。
- 倾斜的箱子按"用它的 yaw 的直立包围盒"保留（InternScenes 的 adapter 把倾斜框换成其 8 个角点的直立包围盒；扫描源里主要是书、枕头、杯子、篮子等小件）。落地箱子沉入地面 ≤0.3 m 的贴到地面保留，更深的丢弃；下面没有东西撑着、悬空 ≤0.5 m 的家具：不能挂墙的（床、沙发、桌椅、冰箱……）和扫描房间里的一律贴到地面，设计师房间里的可挂墙家具按 `floating` 固定件处理。放在家具顶面上的画框、镜子按"放在上面"保留，挂在墙上的不摆。
- **硬过滤（只拒收没法构造训练目标的样本，2026-09-25 起）**：源整体不保证正面朝向；有物体缺包围盒（看不见的障碍）；数值不是有限数，或**要摆的**物体尺寸 ≤0（不摆的零厚度贴纸、地垫不再拒收整间房）；同一房间里 id 重复；没有边界多边形/凸包，或多边形无效；一件待摆物体都没有。
- **质量标记（房间保留，写在每条样本的 `flags` 里，`stats.json` 按 `flag:<名>` 计数；`--reject_flagged <名...>` 可以把选中的标记改回拒收）**：`oob_objects`（参考布局有物体出墙 >10 cm 的件数）、`hidden_m2` / `hidden_obstacle`（被丢掉、又不是固定件或贴面的落地箱子露出的地面 m²，≥0.3 时标 hidden_obstacle；主要是楼梯间）、`small_area_m2`（<1 m²）、`room_filling_object`（一件非地毯物体占房间 ≥80%）、`overlapping_furniture`（同类家具大面积重叠或大件硬家具穿插）、`single_object`、`z_snapped`（离地或沉入超过 10 cm 被贴地的件数；每件被贴地的物体都在 IR 里保留原始 z `z_src`）、`tilted`（倾斜换成直立包围盒的件数）。这些是"这条样本的某个字段可不可靠"的标签，不是对数据集的评价；训练时按标签加权或排除是采样层的事。
- 每个 split 里完全相同的房间（边界相同，且每件物体和固定件的类别、尺寸、位姿都一一相同，允许整体转 90°）只保留一份（3D-FRONT/SpatialLM 同一设计重复最多 26 次；`--keep_duplicates` 可关）。
- 不再在构建阶段截断大源（`--cap` 默认为空）。采样层在 `train.py`：`--exclude_flags` **默认去掉 `oob_objects`、`overlapping_furniture`、`fixed_collision` 三类**（参考答案是服务端会拒绝的布局：物体出墙 >10 cm；同一件家具重复穿插；家具穿过门、柱子、楼梯等非窗固定件），数据里仍保留，传 `--exclude_flags` 不带值就全用；`--source_weight 源=w` 让某个源的每条样本平均用 w 次（0.5 减半、2 加倍，按 uid 稳定）。启动时打印每个源的条数和去掉的条数；`QA.json` 的 `train rows used by default` 给出默认子集的参考答案合法率。

**缺失字段与取整**
- 源数据里缺失的字段一律省略，不伪造。房型、层高缺失时直接不写这个键，训练时也会随机丢掉这两个字段。
- 物体高过房间层高时，丢弃层高字段。
- 类别只做可追溯的归一化，原始标签保留在 IR 里；3D-FUTURE 的 'Corner/Side Table'、'Three-Seat / Multi-seat Sofa' 和 Infinigen 的 *factory 名字单独映射成 side table / three seat sofa 等。房型归一化时会去掉实例编号、楼层号和公寓号。
- 所有校验、约束和评测都用**模型实际看到的取整值**：位置和尺寸取到厘米，yaw 取整数度。生成的约束须在这些参考值上成立；参考布局仍可能有已标记的几何问题，合法率由 `QA.json` 实测，不保证全部样本为 100%。

## 3. 数据（v3，2026-09-25 构建：`/Volumes/harddisk/fastfill_data/v3` 和 `v3.1`；HF 数据集 `liantian/fastfill-v3`）

train 144,140 / dev 8,167 / test 8,647（另有 dev_constrained、test_constrained：每间 1–4 条在参考布局上成立的约束）。
- v3 = 无约束基线 (R,G,O)→L；v3.1 = 同一批房间，其中 42,743 条训练样本带 1–4 条约束 (R,G,O,C)→L。两版 dev/test 逐字节相同（True）；`tools/ablation_check.py`：去掉 constraints 后两版训练样本差异 0 条。
- `QA.json`：泄漏（train 与 dev/test 之间 group、别名、家具指纹、布局指纹、held-out 源）全部为 0；答案恰好摆了输入里的全部物体 144,140/144,140；固定件出现在答案里 0 次；约束在参考布局上成立 18662/18662（dev）、19785/19785（test）。
- 参考布局本身的合法率（dev）：93.2%。默认训练子集（去掉 `oob_objects`、`overlapping_furniture`、`fixed_collision`）124,836 条，参考答案通过越界、支撑及层高检查的有 124,834 条；另 2 条的支撑距离因厘米取整落到 5 cm 容差边界，旧版严格 contains 判为失败。该比例不衡量源包围盒正确性、语义质量或全部碰撞。模型指标一律和同一批房间的 GT 值对比。
- token（Qwen3 分词器，`train.py --dry_run`）：训练集共 1.86 亿 token；单条 P50 896、P90 2,226、P95 3,440、P99 6,142、最长 135,965；≤4096 的 96.6%，≤8192 的 99.69%。每间待摆物体 P50 6、P90 28、P95 46、P99 84、最长 2,080；固定件 P50 2、P90 7、P95 8、P99 12、最长 70。
- **超过 40960 的训练样本 9 条**：`interiorgs:0044_839926::room_0`（InteriorGS，2080 件，135,965 token）；`interiorgs:0045_839925::room_0`（InteriorGS，680 件，43,739 token）；`interiorgs:0134_840039::room_0`（InteriorGS，1274 件，80,252 token）；`interiorgs:0141_840177::room_0`（InteriorGS，1575 件，99,434 token）；`interiorgs:0175_840149::room_0`（InteriorGS，1229 件，67,984 token）；`interiorgs:0179_840114::room_0`（InteriorGS，1164 件，75,057 token）；`interiorgs:0394_840186::room_0`（InteriorGS，983 件，57,821 token）；`interiorgs:0408_840112::room_0`（InteriorGS，904 件，57,764 token）；`MansionWorld::mansionworld/public_entertainment_4f_200_fp001#0/F2_event_hall`（MansionWorld，628 件，44,392 token）。`train.py` 加载时整条跳过、不截断。dev 最长 16,584、test 最长 27,156。
- 固定件共 477,666 个，分布在 135,205 间写出的房间里（占 84%）：structure 441,532、generic 22,040、floating 8,800、no_front 5,128、wall 166。
- v3 修复了 v2 中的 MansionWorld 桌面物件、InteriorGS 门窗、部分倾斜框、3D-FRONT 精确边界和补门等问题。后续审核仍发现 Structured3D/HSSD/MultiScan 的倾斜框转换及 InteriorGS 上游归房遗漏，修复见 v3.2。v2 / v2.1 的 test 集不同，数字不可直接比较。

各源房间数（train / dev / test）、单间最多待摆物体、固定件数、带固定件的房间数（都按写出的房间计）：

| 训练源 | train / dev / test | 最多物体 | 固定件 | 带固定件的房间 |
|---|---|---|---|---|
| SpatialLM | 42,068 / 2,239 / 2,461 | 43 | 156,103 | 46,334 |
| MansionWorld | 19,396 / 1,153 / 1,313 | 628 | 86,316 | 21,762 |
| IL3D_3dfront | 18,054 / 939 / 872 | 27 | 39,427 | 13,567 |
| Structured3D | 12,700 / 793 / 747 | 181 | 73,356 | 14,238 |
| InternScenes_gen | 12,552 / 828 / 660 | 140 | 59,378 | 14,040 |
| SAGE-10k | 8,977 / 503 / 489 | 189 | 12,991 | 9,969 |
| OptiScene_holodeck | 8,946 / 497 / 503 | 21 | 5 | 5 |
| IL3D_synthetic | 6,277 / 349 / 316 | 20 | 1,096 | 999 |
| InteriorGS | 4,867 / 268 / 311 | 2080 | 23,508 | 5,347 |
| InternScenes_arkit | 4,528 / 280 / 229 | 63 | 9,486 | 4,012 |
| HSSD200 | 1,876 / 86 / 138 | 623 | 3,529 | 1,345 |
| InternScenes_scannet | 1,309 / 80 / 104 | 90 | 4,194 | 1,257 |
| InternScenes_mp3d | 1,258 / 89 / 18 | 112 | 5,170 | 1,247 |
| InternScenes_3rscan | 1,200 / 61 / 70 | 160 | 2,588 | 915 |
| MultiScan | 114 / 2 / 6 | 80 | 344 | 101 |
| Scan2CAD | 18 / 0 / 1 | 32 | 0 | 0 |
| SceneSmith（只评测） | 0 / 0 / 364 | 152 | 26 | 25 |
| SpatialGen（只评测） | 0 / 0 / 45 | 25 | 149 | 42 |

扫描 188,670 间。硬过滤拒收（间）：object_count 19,606、boundary 722、boundary_shape 142、incomplete_objects 113。
- `object_count`：没有一件可摆的物体（只有墙上、顶上的东西或没有名字的箱子），没有训练答案。
- `boundary` / `boundary_shape`：没有边界或边界多边形无效。`incomplete_objects`：有物体没有包围盒（HSSD 的可动柜门、冰箱等），会成为看不见的障碍。`bad_numbers`：数值非法，或要摆的物体尺寸 ≤0。
- 另外，每个 split 里完全相同的房间只留一份（duplicate_layout 6,137 间），跨源同一场景只留一个源（dedup_dropped 996 间）。

质量标记（写出的房间里，按标记计）：tilted 14,073、single_object 13,344、fixed_collision 11,918、oob_objects 10,959、z_snapped 5,059、no_front_fixed 3,480、hidden_m2 2,443、room_filling_object 2,193、overlapping_furniture 1,989、hidden_obstacle 1,257、small_area_m2 59。

**数据集取舍：Excel 的 22 个 → 下载保留 16 个（11 个产生训练房间、2 个只做评测、3 个是辅助数据）→ 删除 6 个**

每个保留的数据集都有实际用途（下面"用在哪"一栏可在 `adapters/*.py` 里对应核实）。房间数见上面 §3 的表。

*A. 产生训练房间（11 个下载 → 16 个训练源）*

| Excel 数据集 | 训练源 | 为什么用 |
|---|---|---|
| SpatialLM | SpatialLM | 设计师户型，官方墙线多边形，房型 100% 有值 |
| IL3D | IL3D_3dfront、IL3D_synthetic | 3D-FRONT 设计房间的规范版本（类别、尺寸完整）；程序合成房间 |
| OptiScene | OptiScene_holodeck | 参考项目的数据，只用 Holodeck 生成部分（其中 3D-FRONT 部分与 IL3D 重复） |
| InternScenes | InternScenes_gen；arkit / scannet / mp3d / 3rscan | Infinigen 生成房间（每间都有门窗固定件）；四个真实扫描子集，OBB 带语义正面 |
| MansionWorld | MansionWorld | 程序生成楼宇，多边形边界，房型 100% |
| SAGE-10k | SAGE-10k | Agent 生成的单房间，落地家具和桌面小物件都用（v2 起） |
| Structured3D | Structured3D | 官方房间多边形和层高 |
| InteriorGS | InteriorGS | 官方房间 profile 多边形 |
| HSSD-200 | HSSD200 | 设计场景的官方区域多边形；它的 `asset_catalog.csv` 还是 IL3D_synthetic 计算物体网格边界必需的 |
| SceneCAD & Scan2CAD | Scan2CAD | ScanNet 扫描 + CAD 对齐得到的朝向 |
| MultiScan | MultiScan | 真实扫描，标注了正面向量；多数房间没有可用边界（boundary）被拒 |

*B. 只做评测（held-out，从不进训练）*

| Excel 数据集 | test | 为什么用 |
|---|---|---|
| SceneSmith | 364 | 训练里没有的生成器，测泛化 |
| SpatialGen | 45 | 同上 |

*C. 辅助数据：不产生房间，但删掉会让别的数据变错*

| Excel 数据集 | 被谁用 | 删掉的后果 |
|---|---|---|
| 3D-FRONT | IL3D_3dfront：把 IL3D 按房型合并的房间拆回真实源房间，并取真实地面边界和层高 | 约 1,300 个 IL3D 房间会混着 2–10 个房间的物体，边界也不对 |
| 3RScan | InternScenes_3rscan：reference / rescan 对应表 | 同一空间的多次重扫描可能分到 train 和 test 两边（泄漏） |
| ARKitScenes | InternScenes_arkit：video → visit 对应表 | 同一次 visit 的多段扫描可能分到 train 和 test 两边（泄漏） |

这三个本身不直接进训练：3D-FRONT 原始导出类别和尺寸大面积缺失，已由 IL3D 版本覆盖；3RScan、ARKitScenes 原始导出没有房间边界，房间来自 InternScenes 版本。

*D. 删除（6 个）*

| Excel 数据集 | 删除原因 |
|---|---|
| HM3DSem | 没有 yaw，尺寸是世界坐标 AABB；region 不是房间 |
| MetaScenes | 没有 yaw（世界坐标 AABB）；场景是 ScanNet，已由 InternScenes_scannet 覆盖 |
| ScanNet | 没有 yaw；1,513 个扫描已由 InternScenes_scannet（带正面）覆盖 |
| ScanNet++ | 包围盒没有正面信息（2026-09-24 用数据复核，3 种方法均在对照数据集上先验证有效）：X 轴就是较短的水平边（97–100% 的盒子，对照数据集约 50%），即按边长排序而不是按正面；贴墙的一侧在 ±X/±Y 四个方向各约 25%（对照数据集 88–98% 是背面 −X 贴墙）；椅子相对桌子的朝向四个方向均匀（n=2,908）。另外 221 个 apartment 未拆成房间，边界是矩形代理 |
| ASE | 约 60% 的实例没有框 |
| M3DLayout | 3D-FRONT、Matterport 部分是重复；Infinigen 部分的边界就是家具范围（3000/3000），不是真实房间 |

- 删除依据来自各数据集的 HF README 和本地实测（front test、边界检查）。ASE、ScanNet++、M3DLayout_infinigen 转好的 IR 仍保留在 `/Volumes/harddisk/fastfill_ir/`，以后可以直接加进 build；`fastfill.adapters` 遇到已删除的源会跳过并提示。
- 保留的 16 个数据集已逐文件对照 HuggingFace 核对完整（含 MultiScan 的网格和贴图，共 83 GB）。重新下载用 `fastfill/download_full.sh <目录> <名字...>`（需要 zsh；名字写错或 30 次重试都失败会报错退出）。

**许可证（按各数据集本地 README / LICENSE 核对，2026-09-24）**

| 训练源 | 许可 / 条款 |
|---|---|
| SpatialLM | CC BY-NC 4.0 |
| IL3D_3dfront | IL3D 卡片写 Apache-2.0，但内容是 3D-FRONT：受 3D-FRONT 使用协议约束，只能科研、不能商用、不能用结果提供对外服务、不能分发数据（§3.2、§3.3.1–3.3.5） |
| IL3D_synthetic | 用 HSSD 资产布置，按 HSSD：CC BY-NC 4.0 |
| HSSD200 | CC BY-NC 4.0 |
| InternScenes（gen / arkit / scannet / mp3d / 3rscan） | CC BY-NC-SA 4.0，另加各上游扫描数据集的条款 |
| MultiScan | CC BY-NC 4.0（需申请） |
| Structured3D | 仅非商业科研，不得再分发 |
| InteriorGS | 需申请，按 interiorgs-terms-of-use |
| MansionWorld | CC BY 4.0，但 HF 下载时要勾选"仅非商业科研" |
| SAGE-10k、OptiScene_holodeck | Apache-2.0（约占训练集 17%） |
| Scan2CAD | 本地 README 未写许可；ScanNet 条款为非商业 |
| SpatialGen（评测） | CC BY-NC 4.0 |
| SceneSmith（评测） | Apache-2.0 |

结论：所有源都允许非商业科研，混在一起训练做内部研究可以；模型继承全部限制，**只能内部非商业使用**。公开权重有冲突（InternScenes 的 ShareAlike 与 3D-FRONT、Structured3D 的禁止分发条款），真要发布需要去掉 InternScenes 或 IL3D_3dfront 重训，并先做法律核查。

**防泄漏**
- 以下四种关系任意一种相连的房间，整体划入同一个 split：同一 house 或 scan、同一次 visit、相同家具指纹、相同布局指纹（旋转 90° 视为同一布局，换资产也算）。InternScenes_gen 按房子编号分组（同一个编号出现在不同房型文件夹里就是同一栋房子的不同房间）。
- 在写出的文件上复核过：train 与 dev/test 之间 group、布局指纹、家具指纹的重合数都是 0；held-out 源在 train 中为 0 行。

## 4. 在服务器上训练

**环境（CUDA 12.8）**：用 node03 上已验证的组合（torch 2.10 cu128、vLLM 0.19.0、transformers 4.57.6），版本固定在 `requirements-cu128.txt`（在 HF 仓库的 `code/` 下）。训练、合并、推理必须在同一个环境里完成。

**代码和数据**都在HF 数据集 `liantian/fastfill-v3`：`code/`（本目录的快照 + `scripts/merge_lora.py` + `requirements-cu128.txt`）、`v3/`、`v3.1/`。下面所有命令都在下载下来的 `code/` 目录里运行。

```bash
conda create -n fastfill python=3.11 -y && conda activate fastfill       # 已有 node03 的 fastfill 环境就跳过这两行
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -U "huggingface_hub>=0.36" && hf auth login                   # 用一个 read token
hf download liantian/fastfill-v3 --repo-type dataset --local-dir ~/fastfill-v3
cd ~/fastfill-v3/code && pip install -r requirements-cu128.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -m fastfill.scene && python -m fastfill.validate && python -m fastfill.interface   # 自检
hf download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
python -m fastfill.train --model ~/models/Qwen3-8B --data ~/fastfill-v3/v3 --out /tmp/x --dry_run   # 用真实分词器逐条统计（见下）
```

**先测数据，再定参数**（2026-09-24 决定：让数据决定训练参数，不让训练参数决定删哪些家具）
- `--max_len 40960`（默认）只是模型的序列上限，不是显存设置：`train.py` 按每批里最长的那条补齐，显存由数据决定。超过 40960 的样本整条丢弃、不截断。
- `--dry_run`：用 Qwen 分词器对 train/dev/test 逐条分词，打印每条 token 数、答案 token 数、每间房物体数和固定件数的 P50/P90/P95/P99/最大值，各长度桶的条数，以及超过 `--max_len` 的样本清单（uid、源、token 数、物体数）。超长的极少数拿出来单独讨论，不自动删。
- `--mem_test 8192,16384,32768,max`：在 1 张卡上（`CUDA_VISIBLE_DEVICES=0`）按训练的真实配置各跑一步（`--bs` 行、按 collator 的方式右填充：第 0 行是该长度，其余行是它的一半，所以 attention mask 也和训练时一样被实体化；梯度检查点和训练一致）。先用 512 token 做一次预热步，再读取 static（权重 + LoRA 参数/梯度 + AdamW 状态）；表中每行给出 peak 和这一步新增的部分（激活 / logits / attention mask）；某个长度 OOM 时打印后跳到下一个。`max` = train 与 dev 中最长的一行（Trainer 每 `--save_steps` 会在 dev 上评估一次；表前先打印 train / dev 各自最长行的 token 数），数字 = 该长度的合成序列（全部 token 都监督，是上界）。用你打算训练的 `--bs` 跑它，再定 grad_accum。
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m fastfill.train --model ~/models/Qwen3-8B --data ~/fastfill-v3/v3 --out /tmp/x --bs 1 --mem_test 8192,16384,32768,max
  CUDA_VISIBLE_DEVICES=0 python -m fastfill.train --model ~/models/Qwen3-8B --data ~/fastfill-v3/v3 --out /tmp/x --bs 2 --mem_test 8192,16384,max
  ```
- 评估固定 `per_device_eval_batch_size=1`；`--eval_max_len N` 把 dev 里超过 N token 的行排除出评估集（打印排除数量，此时 eval loss 只覆盖剩下的子集），N 按 mem_test 的表来选（最长样本见 §3）。

**训练**：有效 batch = bs × grad_accum × GPU 数，保持在 32 左右。换 GPU 数量时要同步调整 grad_accum；bs 按 `--mem_test` 的结果定（下面的 bs=4 是 v1 数据 ≤2.8k token 时的值）。

```bash
# 8 卡
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node 8 -m fastfill.train \
    --model ~/models/Qwen3-8B --data ~/fastfill-v3/v3 --out outputs/ff-v3 \
    --bs 1 --grad_accum 4 --lr 1e-4 --epochs 2 --save_steps 500          # bs 按 --mem_test 的结果改：bs 2 -> grad_accum 2，bs 4 -> 1
# 只用 1–7 号卡：前面加 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7，并改 --nproc_per_node 7（有效 batch 28 也可以）
# 单卡：CUDA_VISIBLE_DEVICES=0 python -m fastfill.train ... --grad_accum 8 --save_steps 1000
#       （不指定卡时，单进程会把所有卡用成 DataParallel，又慢又占 0 号卡显存）
# 中断后续训：同一条命令末尾加 --resume。在第 2 个 epoch 中途断开时，该 epoch 剩余部分的数据顺序会变
# （HF 的已知行为，最多约 1/4 样本在这个 epoch 多训或少训一次），可以接受
```

**资源估算**（v1 数据、bs=4 时的解析估算，v2 的样本更长，以 `--mem_test` 实测为准）
- 2 个 epoch 的用时：8×H100 约 1.5–2 h；8×A100 约 3–4 h（token 总量比 v1 多 39%）。
- 每个进程都会先在 CPU 上加载一份 16 GB 的权重，8 卡时主机内存峰值约 170 GB。

**合并与评测**

```bash
python -m fastfill.merge_lora --base_model_path ~/models/Qwen3-8B --lora_path outputs/ff-v3/final --output_path outputs/ff-v3-merged
python -m fastfill.evaluate --gt-only --rooms ~/fastfill-v3/v3/test_rooms.jsonl --out eval/gt            # GT 参考线
python -m fastfill.evaluate --model outputs/ff-v3-merged --rooms ~/fastfill-v3/v3/test_rooms.jsonl --out eval/ff-v3
python -m fastfill.evaluate --model outputs/ff-v3-merged --rooms ~/fastfill-v3/v3/test_constrained_rooms.jsonl --out eval/ff-v3-cons
```

**看评测结果时注意**
- 优先装 vLLM；走 HF generate 时设 `CUDA_VISIBLE_DEVICES=0`，否则 8B 模型会被切到多张卡上串行跑；`--batch` 默认 1（40,960 上下文下多于一行的 KV cache 放不进一张卡）。
- prompt 长度 ≥ `--max_len` 的房间不送模型，记为 `prompt_too_long`，按失败请求计（valid False、约束全 False、parsed False），其余房间照常评。
- `metrics.json` 的 `in_dist` 和 `held_out` 分开看，碰撞指标一律和同一批房间的 `*_gt` 对比，不要和 1.0 比。
- 合法（valid）= 不出界、支撑正确、不顶破天花板（物体顶 ≤ 层高 + 5 cm；层高未知时不查）。
- 越界率、碰撞率、OOR 只在完整解析的答案上算；一个完整答案都没有时显示 `null`（未知），不是 0。
- **模型指标只看原始输出**（`valid_rate` 等）。`*_repaired` 是接口把物体推回墙内之后的结果，属于确定性后处理，不算模型能力；`inside_1mm_rate` 是按后端 1 mm 标准的贴墙合格率，原始和修复后分开报。
- **v3（无约束版）训练时没有见过约束输入**，所以 test_constrained 上的约束满足率是"无约束监督基线（零样本）"，不能当作约束能力的结论。
- `rooms_with_fixed_collision`（和 `_gt`）是摆放和固定件重叠的房间比例，对照同一批房间的 GT 值看；序列长度用 `--max_len`（默认 40960，vLLM 的 max_model_len），提示词本身就超过它的房间记为 `prompt_too_long` 失败，不会让整轮评测中断。

**起服务接 WorldEdge（EmbodiedGen `FastFillBackend`）**

```bash
python -m fastfill.serve --model outputs/ff-v3.1-merged --port 8001       # vLLM，贪心解码，一次处理一个房间；--max_len 40960
# EmbodiedGen 那边：WORLDEDGE_FASTFILL_URL=http://<node03>:8001/v1  WORLDEDGE_FASTFILL_MODEL=<服务名，默认是 checkpoint 目录名>
```

- 后端把附录 B 请求的 JSON 原样作为 user 消息发过来；`serve.py` 用 `interface.py` 转成模型文本，模型输出后再转回 `placements`（`position_m` / `yaw_rad` / `support_parent`），放在回复的 content 里。
- 以下情况返回 HTTP 422、不返回布局，由调用方（Harness）决定怎么办：请求不合法（数值非法、坐标系不是米/Z 向上/弧度、重复 id、未知 anchor、约束引用自己或成环、约束引用固定件的 id 等）；prompt 超过 `--max_len`（`prompt_too_long`）；模型输出解析不了（包括漏摆或多摆物体）；修复后仍验收不过（出界、悬空、顶破天花板、有物体超出墙面 1 mm、任何一条 hard 约束不成立）。评测里对应的指标是 `accepted_rate`（同样要求不穿过任何固定件）。FastFill 要么把全部物体合法摆好，要么明确失败，**绝不自己丢物体**。放不下的清单也走这条路。
- 训练数据里物体常常穿墙几厘米（test 参考答案 39.8% 的房间 > 1 mm，OptiScene_holodeck 系统性 4.5 cm），而后端只容忍 1 mm。所以落地物体穿墙或穿进 keepout 不超过 10 cm 时，`interface.py` 会把它推回墙内，放在它上面的东西一起移动。这是**确定性修复，不算模型成功**：回复里同时给 `raw_placements`（模型原始提案）、`placements`（修复后）和 `validation.raw` / `validation.repaired`（各自的验收结果，含每条约束是否满足），日志和实验三者分开记。
- 约束字段（这就是后端文档里说的"FastFill v1 文档"）。**每个谓词只有一份实现：`validate.holds`**。数据里抽约束、模型 prompt 的定义文字、评测、服务端验收都调用它，所以训练、条件、验收三处语义一致；别的系统要判断 FastFill 约束是否满足，读回复里的 `validation.*.constraints`，不要自己重新实现一遍。

| type | 字段 | 定义（`validate.holds`） |
|---|---|---|
| `faces` | `subject`, `target` | subject 的正面方向与"subject 中心 → target 中心"的夹角 ≤ 30° |
| `between` | `subject`, `anchors: [a, b]` | a、b 中心连线与 subject 的旋转 footprint 相交 |
| `supported_by` | `subject`, `parent`, `surface: "top"` | subject 的支撑父物体就是 parent（放在它顶面） |
| `against_wall` | `subject` | subject 旋转 footprint 的某一整条边，每一点离边界都 ≤ 0.1 m |
| `near` | `subject`, `target`, `max_gap_m` | 两个**旋转后 footprint 之间的最短距离** ≤ max_gap_m（间隙，不是中心距离）；训练见过 0.1 / 0.2 / 0.3 / 0.5 |
| `keepout` | `polygon_xy`, `hard` | `hard=true`（默认）：外扩 0.08 m 后从地面多边形里挖掉；被禁放区围出来的 < 0.5 m² 小死角当作不可用地面。`hard=false`：只报告满足情况，不切地面、不新增模型条件 |

- 别的系统里叫 near 但定义不同的关系（例如"中心距 ≤ 1.5 m"），**不能**直接改个字段名当作 FastFill 的 `near` 发过来：两者是不同的谓词，需要调用方自己决定怎么对应。

- 物体上带 `"anchor": "wall"` 或 `"ceiling"` 的（镜子、挂画、吊灯），v1 不摆，放进 `unsupported` 返回；引用它们的约束也进 `unsupported`，其余物体照常摆。
- 回复带 `model`（服务名，默认 checkpoint 目录名）和 `checkpoint`（绝对路径）；EmbodiedGen 按 `WORLDEDGE_FASTFILL_MODEL` 缓存回复，所以两边名字要一致。422 的回复也带模型原文 `raw_text` 和 `unsupported`。没有可摆物体的请求直接回 200、不调用模型。
- 训练里从没出现过的约束形式（扫描凸包房间里的 `against_wall`、主语没有语义正面的 `faces`、涉及放在别的物体上的对象的 `near`/`faces`/`between` 等）照样检查，但在 `validation` 里标 `"trained": false`。
- 禁放区离墙不到 1 cm 时贴到墙上（避免厘米取整后留下一条自相交的细缝）；取整后仍然折叠的切口返回 `keepout_not_representable`。
- 评测的 `metrics.json` 对合法率、验收率、贴墙 1 mm 合格率、出界率都给出同一批房间的参考布局值（`*_gt`）。
- 请求里可选的 `room.fixed_geometry`：已经在房间里、不能动的箱子（门、窗、柱子、楼梯，或者留在原地的家具），每项 `{"id","category"?,"size_xyz_m","position_m"（底面中心）,"yaw_rad"?}`，按 `fixed` 给模型；和它们重叠的摆放记在 `validation.*.fixed_collisions`；其中**穿过门、柱子、楼梯等非窗固定件的**（`fixed_blocking`）判为验收失败（422）。家具挡窗只报告（窗框伸进房间 5 cm，贴墙的柜子、床头常常压到它）；家具之间的包围盒重叠（椅子塞桌下）也只报告。推回墙内的修复会连同放在上面的东西一起检查，不会把杯子推进固定柜体。按训练数据同样归一化：|z| ≤ 0.10 归零；底部 ≥ 1.5 m、占地 ≥ 房间一半、或完全在墙外的固定件不给模型，在 `unsupported` 里以 `fixed_ignored` 报告；推回墙内的修复不会把物体更深地压进固定件（找不到不加深重叠的移动就原地不动，由验收判定）。约束不能引用固定件。EmbodiedGen 目前不发这个字段（门口用 keepout），加不加由后端负责人定。
- 所有约束都按"必须满足"发给模型；验收时 `hard` 默认 true，失败则 422；`"hard": false` 的约束只报告结果，不导致失败。
- `against_wall` 按真实墙线验收，不算禁放区的切口（模型会把切口看成墙，所以请求里同时有禁放区和 against_wall 时，模型偶尔会因此失败）。
- keepout 把房间切成两块或在中间留出孤岛时，请求直接报错（`keepout_not_representable`）。

## 5. 模型输入输出（真实样本，ARKit 扫描卧室，1 扇窗 + 2 扇门是固定件，6 件待摆，其中 2 盏灯放在柜子上）

```json
user:      {"boundary_type":"hull","boundary":[[0,0.04],[0.04,0],[5.66,0],[5.7,0.04],[5.7,2.93],[5.66,2.97],[0.04,2.97],[0,2.93]],
            "fixed":[{"id":"window_1","size":[0.11,1.16,1.23],"pos":[0.2,1.5,0.89],"yaw":356},{"id":"door_1","size":[0.05,0.86,2.1],"pos":[5.33,2.74,0],"yaw":274},
                     {"id":"door_2","size":[0.03,0.79,2.03],"pos":[5.52,1.39,0],"yaw":179}],
            "objects":[{"id":"bed_1","size":[2.07,1.63,0.61]},{"id":"cabinet_1","size":[0.5,0.84,0.77]},{"id":"cabinet_2","size":[0.52,0.42,0.54]},
                       {"id":"cabinet_3","size":[0.53,0.41,0.52]},{"id":"lamp_1","size":[0.29,0.2,0.33]},{"id":"lamp_2","size":[0.21,0.18,0.34]}]}
assistant: {"placements":[{"id":"bed_1","pos":[2.31,1.13,0],"yaw":91},{"id":"cabinet_1","pos":[4.2,2.6,0],"yaw":269},{"id":"cabinet_2","pos":[3.38,0.36,0],"yaw":91},
            {"id":"cabinet_3","pos":[1.26,0.4,0],"yaw":91},{"id":"lamp_1","on":"cabinet_2","pos":[3.38,0.25,0.54],"yaw":90},{"id":"lamp_2","on":"cabinet_3","pos":[1.23,0.24,0.52],"yaw":89}]}
```

- `fixed`（可选）：已经在房间里、不能动的箱子，带 size / pos / yaw，模型不摆它们、只避开；答案里不出现。
- 对象按占地面积从大到小排列，id 是 `类别_序号`（固定件先编号，和待摆对象共用命名空间），答案顺序与输入一致。
- 可选的 `desc` 取资产描述的第一句，已去掉摆放从句，训练时整间房以 50% 的概率丢弃。
- 可选的 `constraints` 有五种：`["on",a,b]`、`["faces",a,b]`、`["near",a,b,d]`、`["against_wall",a]`、`["between",a,b,c]`。

## 6. 重建数据 / 下一步

```bash
python -m fastfill.download --out $DATA                     # 汇总文件 33 GB（全量：fastfill/download_full.sh $DATA）
python -m fastfill.adapters --root $DATA --out $IR          # 21 个 adapter → IR（约 20 万个房间）
python -m fastfill.build --ir $IR --out data/fastfill_v3 \
  --sources SpatialLM IL3D_3dfront InteriorGS HSSD200 Structured3D InternScenes_arkit InternScenes_mp3d InternScenes_scannet \
            InternScenes_3rscan Scan2CAD MultiScan OptiScene_holodeck IL3D_synthetic SAGE-10k MansionWorld InternScenes_gen SpatialGen SceneSmith
# v3.1（约束监督）：同一条命令加 --constraint_frac 0.3，输出到另一个目录；test 文件不变，可以与 v3 配对比较
python -m fastfill.tools.qa_report $IR data/fastfill_v3           # -> QA.json（泄漏、闭合、约束、参考合法率、固定件）
python -m fastfill.tools.ablation_check data/fastfill_v3 data/fastfill_v3.1
```

**已知限制**
- **序列长度**：唯一的硬上限是模型的 40960 token；超长样本训练时整条丢弃（`--dry_run` 列出清单）。物体最多的房间来自 InteriorGS（649 件）和 HSSD（623 件）。
- **门窗来源**：SpatialLM、3D-FRONT、MansionWorld、InteriorGS 来自导出里每个房间的 `doors` / `windows`（位置、宽高、窗台高；门厚取自门框轮廓，其余按 0.10 m；被拆成几块的门和飘窗合并成一个，低于 15 cm 的窗台板丢掉；3D-FRONT 只挂在隔壁房间名下的门按几何补给它通向的每个房间，约 62% 的房间有门）；Structured3D 来自 `room_context.doors` / `windows`；SAGE-10k 的门来自每个场景的 `layout_*.json`；InternScenes、HSSD、MultiScan、扫描源用自带的门窗标注框（HSSD 元数据生成的门窗框按网格包围盒重新定中心）。Holodeck、IL3D_synthetic 的导出里没有门窗。门窗框放在墙线上，厚墙里（离地面多边形 0.3 m 以内）的也保留；飘窗凸出墙外、不碰墙线的不给模型。
- **服务端边界**：`clean_boundary` 不是幂等的（厘米取整后再按 1 cm 简化，第二次可能再删掉顶点），所以把已经规范化过的边界再发给服务，得到的多边形可能和训练时略有不同；原始边界和 EmbodiedGen 发的矩形不受影响。
- **MansionWorld 桌面物件**（v3 起读入）：导出里有、并和原始 floor JSON 对上的都进来了；AI2-THOR 自带、导出时就没有尺寸的约 2.5 万件（笔记本电脑、盆栽、书……）没有来源尺寸，不编造，不摆。
- **不在支撑物顶面上的东西**（SAGE 带书架的书桌桌面、柜子里、搁板层上：底部离父物体包围盒顶面超过 5 cm）不摆也不作为固定件：答案格式规定 `on` 的 z 就是父物体的顶面，表示不了它们。SAGE 里约 37% 的带 place_id 的小物件属于这种情况。
- **支撑关系**：绝大多数是从 bbox 顶面接触推断出来的，证据偏弱。
- **边界**：IL3D_3dfront 在 v3 起用 3D-FRONT 的精确地面多边形（无效时退回地板凸包，`meta` 里记录用了哪种）。
- **房型标签**：3D-FRONT 自身有房型标错的噪声。
- **朝向**：扫描源约 15% 的床正面转了 90°（适配器层面的问题，尚未修）；MansionWorld 和 Holodeck 的马桶宽大于深。
- **碰撞检查**：只在 bbox 级别，结果须对照 GT 参考线。
- **尚未支持**：对称物体 yaw 等价类的训练；把 keepout 也表示成固定件（现在仍是从地面多边形里挖掉）。
- **许可**：见第 3 节许可证表。训练出的模型只能用于内部非商业研究；不能公开权重，也不能对外提供服务。
