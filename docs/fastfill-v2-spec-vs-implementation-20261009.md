# FastFill v2 设计规范（2026-10-05 定稿）与当前实现对照

- 日期：2026-10-09；仓库：`main` @ `fb9f475`
- 规范来源：2026-10-05 定稿的设计规范（要点转录稿，保留了每一条可核对的断言）
- 对照基准：以当前状态为准，包括 `fastfill/v2/*.py`、`fastfill/v2/README.md`、`docs/`、数据 `outputs/fastfill_v2/rebuild-main-20261008a/main/`（train/validation/test.jsonl、manifest.json），以及 `V-DETR/` 子模块（用于核对 §11）
- 运行配置：`fastfill/v2/configs/main7-cell05-main-20261007b-e5.json`，运行时再加 `--set model.max_objects=256`，并设 `loss.yaw_cls` 为 0.5 或 0.08（`fastfill/v2/ops/isambard_autorun.sbatch:17`；`fastfill/v2/ops/isambard_submit.sh:3-8`）

引用约定：不带目录的 `xxx.py:N` 指 `fastfill/v2/xxx.py`；`ops/…`、`tests/…`、`README.md` 都在 `fastfill/v2/` 下；`config:N` 指上面的运行配置文件；`docs/…` 指仓库根目录下的 `docs/`；train、validation、test 指上述数据目录里的 jsonl。

判定含义：**成立**（CORRECT，与代码一致）；**已过时**（OUTDATED，曾经成立，代码已改）；**错误**（WRONG，从未成立或本身不对）；**未实现**（NOT-IMPLEMENTED，规范说已交付或必需，代码没有）；**无法核实**（UNVERIFIED，本地无法核对）。

## 结论

规范的骨架仍然成立：输出接口 `{id, target_size_local_m, bottom_center_m, yaw_rad}`、底面中心与右手 Z-up 约定、只依赖输入的房间归一化、Qwen + LoRA 条件编码加双向 slot 解码器、每个请求一个 slot 且没有 objectness、`exp` 形式的 log-size 头、K=12 的 yaw bin + residual 公式、只在已认证可交换组内做 detached Hungarian（α_p=α_s=1）、权重为 0 的 `1−GIoU` box loss、按 `Σ|log(actual/target)|` 排序的 Resolver，以及 §11.4 中 V-DETR 的默认配置，这些都与代码一致。

有两处判定为错误。第一，评测脚本的平凡基线默认在评测集上逐行留一拟合，而 autorun 的选模分数以这些基线为分母，违反“统计量只由训练数据确定”。第二，选模依据不含资产解析率和约束通过率，只有单参考误差和几项分布偏差。

已过时的部分集中在模型和数据契约：
- 位置头是 16×16 网格分类 + 格内残差 + z 回归，不是连续回归；
- 位置和尺寸用 L1，不用 SmoothL1；
- K1 契约允许 6.3% 的对象交换 w、d，与 §6.4 的加粗句相反；
- 93.7% 的对象只学 yaw mod π，最终朝向靠 spread 的“背向墙”启发式；
- 门窗以 `room.fixed_objects` 盒子出现，`room.openings` 从未使用；
- 交付输出默认先经过贪心的 spread 后处理。

影响最大的缺口有六个：
1. 选模、主表和 LLM 对照都用三字段投影。这个投影去掉门窗、固定物体、约束和支撑声明，所以选出的模型和报告的数字都不反映对门窗的处理，而训练目标仍按被去掉的障碍摆放。
2. 场景协调性没有任何量化指标。FastFill 比 LLM harness 看起来更不协调，只是目视观察；VLM 评审工具已经写好，但还没运行。
3. §8 的资产闭环、Harness、持久 Host、Solver 都没有实现，与规范“作为主系统整体交付”的说法不符。
4. 碰撞只靠推理端的 spread 缓解，所有场景正则的权重都是 0。
5. 文本 SFT 对照、大部分核心消融、资产不重叠子集和真实尺寸子集都没有做。
6. 只有单种子，置信区间按房间而不是按房屋聚类。

## 判定统计

| 判定 | 条数 | 说明 |
|---|---:|---|
| 成立 CORRECT | 15 | 对核心接口、公式和配置抽查确认，不是穷举 |
| 已过时 OUTDATED | 16 | |
| 错误 WRONG | 2 | §3.4 统计量来源、§7.3 选模依据 |
| 未实现 NOT-IMPLEMENTED | 27 | |
| 无法核实 UNVERIFIED | 3 | §4.3 顺序/ID 稳定性、§7.3 过拟合诊断、§11.6 方法定位段 |
| 规范条目合计 | 63 | |
| 规范未提及、当前已有 | 4 条单列 | 三字段投影、RoomGenBench 交接、VLM 评审渲染、Isambard/autorun 运维；spread 解码、K1 交换规则和 LLM 智能体对照分别并入 §4.1、§6.4 和 §9 |

## 1. 任务与系统职责

### 1.1-a 输出接口 — 成立
- 规范：「F_θ(R, O_req, C) → L̂ = {(id_i, ŝ_i, p̂_i, θ̂_i)}。id 与请求一一对应」
- 现状：`schema.py:293-311` 的 `validate_layout` 只接受 `id/target_size_local_m/bottom_center_m/yaw_rad`，要求尺寸为正、yaw 在 [-π,π)、ID 不重复，并且输出 ID 集合必须等于请求 ID 集合。

### 1.1-b R 的内容（门窗开口、已有物体） — 已过时
- 规范：「R 包含可获得的房型、地板轮廓、房高、门窗开口，以及必须保留的已有物体…已有物体属于条件，不占新增请求的输出 slot；生成与验证同时考虑它们」
- 现状：
  - `schema.py:53-54` 接受 `room.openings`，但这个键在 train 124,584 行、validation 8,137 行和 test 8,620 行中都没有出现。
  - 门窗只以 `room.fixed_objects` 盒子存在：train 中有 door 盒 178,022 个、window 盒 133,767 个。含门或窗盒的行在 train 为 101,486/124,584（81.5%；把 doorframe 等也算上为 101,659），在 validation 为 6,794–6,804/8,137，在 test 为 6,913–6,926/8,620。
  - 固定物体的 ID 为 `fixed_%04d`，不在 `condition.objects` 中，不占 slot（`legacy_bridge.py:200,217`）。它们在验证时作为 `fixed_collision` 检查（`validation.py:325-332`），在 spread 解码时作为障碍（`evaluate.py:204-211`）。
  - `render_minimal_condition`（`batch.py:137-156`）会去掉 fixed_objects、多边形细节、约束和 support_parent。训练以 `minimal_form_p 0.5`（config:81）对 65,987 个合格房间使用这个投影，其中约 4.78 万个房间含门窗，而这些房间的目标布局仍按被去掉的障碍摆放。
  - 选模也用同一投影（`train.py:92`；`autorun.py:291`）。只有 autorun 的测试报告另外给出完整条件投影（`autorun.py:559`，`projections.full`）。
- 改为：完整条件下，R 包含房型、地板多边形、房高和 room.fixed_objects。门窗以 category 为 door/window 的固定盒子写在 fixed_objects 里，约 81% 的训练行有；room.openings 字段保留，但当前数据没有使用。已有物体（包括门窗）只作条件，不占输出 slot，验证时检查碰撞，spread 解码时作为障碍。三字段投影会去掉全部固定物体、约束和支撑声明，但目标布局仍按原有障碍摆放。训练中 minimal_form_p=0.5 使用这种投影，选模也用它，因此选模指标不反映模型对已有物体和门窗的处理；完整条件的结果只出现在测试报告的 projections.full 中。

### 1.3 适用范围（倾斜对象分流） — 未实现
- 规范：「主协议覆盖直立、仅绕竖直轴旋转的对象；显著 pitch/roll、非刚性等应显式分流」
- 现状：
  - `legacy_bridge.py:97-110` 的 `_validity` 在 `upright=False` 时把位置、尺寸、yaw 掩码都设为 False，但对象仍作为请求 slot 保留（`legacy_bridge.py:220-245`）。
  - train 的 field_evidence 中有 84,377 个 `tilted:true` 对象，占 1,998,842 个请求对象的 4.2%；validation 有 5,105 个，test 有 6,154 个。
  - 只有倾斜的固定几何会让整行失败（`legacy_bridge.py:205-206`）。没有任何字段标记非刚性对象。
- 改为：主协议只监督直立、仅绕竖直轴旋转的对象。当前实现没有显式分流：倾斜的源对象仍作为请求 slot 保留，只是位置、尺寸、朝向监督全部屏蔽（约占训练请求对象的 4.2%），推理时照常输出直立框。倾斜的固定几何会让整行被拒收；非刚性对象没有标记。

## 2. 坐标、尺寸与数据协议

### 2.1 局部 +X 规范前向（与 §6.4 对称性、语义朝向合并） — 已过时
- 规范：「对象局部 +X 规范前向…」；§6.4「真正旋转对称的对象定义等价 yaw 标签集合……有前向要求的对象保持语义角度」
- 现状：
  - 前向策略为 `axis`（manifest 的 `parent_front_policy`；`legacy_bridge.py:14-21, 107-110`；`multisource_data.py:288-289`）。
  - train 的 `yaw_symmetry_order` 分布：order 2 为 1,872,154 个（93.7%），order 4 为 126,178 个（6.3%），order 1 为 510 个（只来自 MultiScan）。order 4 几乎全部来自交换来源策略，只有 Scan2CAD 的少量 `__SYM_ROTATE_UP_4` 对象是真实的对称标注。
  - loss 在 n 个对称候选上取联合最小（`losses.py:188-205`），这一机制与 §6.4 一致；但它意味着 order ≥ 2 的对象只学 yaw mod π。
  - spread 解码会把离墙 0.3 m 以内的地面对象翻转 π，让它背向墙（`evaluate.py:283-287`；`README.md:253-266`）。
  - 数据中没有 `faces_direction` 约束；train 只有 12 条 `faces` 约束，validation 和 test 都是 0。
- 改为：约定上对象局部 +X 为规范前向，但当前主数据的 yaw 周期主要由来源策略决定，不由真实旋转对称性决定。训练集 93.7% 的对象为 order 2，只学 yaw mod π；6.3% 为 order 4（盒对称档）；只有 MultiScan 的 510 个对象（0.03%）保留语义前向。因此 +X 的正负方向不是语义朝向的真值。最终朝向由 spread 解码按“背向最近墙”的启发式翻转 π 得到；数据中没有 faces_direction 约束。

（§2.1 中“局部尺寸不随 yaw 变化”与 K1 交换规则的冲突见 §6.4-b。）

### 2.2 bottom-center 与几何中心 — 成立
- 规范：「position head、matching cost、position loss 均使用底面中心 p。框算子要求几何中心时 ĉ=p̂+(0,0,ĥ/2)」
- 现状：`geometry.py:61-62` 的 `bottom_to_center` 加 h/2；`geometry.py:74-78` 的直立框从底面 z 向上延伸 h。

### 2.4-a 归一化只依赖输入 — 成立
- 规范：「o_R、d_R 仅由输入确定…缺失房高不能用监督对象高度补齐」
- 现状：`schema.py:316-324` 的 `normalize_room` 由地板多边形外包框给出原点和水平尺度，缺少房高时用固定的 3 m；`batch.py:176-178` 明确写着 “never infer scales from targets”。

### 2.4-b 未知标记 — 未实现
- 规范：「未知边界、门窗、支撑或角度携带有效性标记；未知 ≠ 0」
- 现状：
  - 边界、地面、房高和目标字段都有未知标记：`boundary_known`（train 中 6,751 行为 false）、`floor_known`、`height_m` 为 null，目标字段用 NaN 加 validity 掩码（`batch.py:193-200`）。
  - 缺少 support_parent 时一律按 unknown 处理，不当作地面（`validation.py:216-217` 的 `support_unknown`）。
  - 门窗没有这样的标记：train 中 23,098/124,584 行（18.5%）没有任何门窗盒，无法区分“确实没有门窗”和“门窗未标注”。`render_minimal_condition` 写入 `boundary_known` 为 true，去掉门窗时也不留开口未知标记（`batch.py:137-156`）。
- 改为：边界（boundary_known）、地面（floor_known）、房高（null）和目标字段（validity 掩码，用 NaN 而不是 0）都有未知标记。缺少 support_parent 时一律按“未知”处理，而不是当成地面。门窗没有已知/未知标记：没有门窗和门窗未标注无法区分；三字段投影去掉门窗时也不留未知标记。

## 3. 数据构造与审计

### 3.2 几何审计（视觉复核） — 未实现
- 规范：「几何审计：…场景关系（房间轮廓、门窗、楼层、支撑、悬挂）、视觉复核」
- 现状：
  - 自动审计已有：`legacy_verify.py`、`multisource_verify.py`、`review_data.py` 和 `qualified_data.py`。
  - `docs/fastfill-v2-data-audit.md:175` 写明 “No full source mesh overlay/render audit ... has been performed”；`docs/fastfill-v2-review-20261005.md:87` 把“全部资产视觉复核”列为未完成。
  - `bbox_visualization.py` 和 `ops/vlm_judge.py` 能渲染盒子，但没有任何数据视觉复核记录。
- 改为：已实现的自动审计覆盖尺寸与变换、标签有效性、地面、高度冲突、支撑选择和 split 泄漏；数据层面的视觉复核或网格叠加审计尚未执行（现有渲染工具只用于预测评审）。

### 3.3-a 条件泄漏控制（描述中的尺寸） — 未实现
- 规范：「条件只保留测试时可获得的信息…不得把资产 ID、精确尺寸摘要、目标位置描述…放入条件」
- 现状：
  - 资产 ID 和支撑表面不进入条件：没有固定物体带 `support_surfaces`，请求对象的键只有 `id/category/description/support_parent`。
  - 对象描述直接用源文本，没有任何尺寸过滤（`legacy_bridge.py:163-166`；`provenance.descriptions='source_desc_or_category'`）。
  - 用米制尺寸正则匹配，约 1.5k 个训练对象、104 个验证对象、91 个测试对象的描述含尺寸，例如 “modern 90cm wide range cooker…” 和 “rectangular white bathtub suitable for a 1.7m length”。
  - 训练中的约束是稀疏关系（max_k=4，`build.py:347`），属于规范的 C 输入，§3.1 允许这样做，所以不算泄漏。
- 改为：条件中不放资产 ID 和支撑表面；但对象描述直接沿用源文本，没有过滤尺寸（约 1.5k 个训练对象、91 个测试对象的描述含米制尺寸）。这些尺寸应当过滤，或者作为“真实尺寸要求”单独标注和统计。

### 3.3-b 真实尺寸要求子集 — 未实现
- 规范：「真实固定尺寸/范围保留并标注为子集，单独统计」
- 现状：train、validation、test 中没有任何请求对象带 `fixed_size_local_m` 或 `size_bounds_local_m`，manifest 里也没有这样的子集。schema 和模型已支持这两个字段（`schema.py:158-173`、`batch.py:242-251`、`model.py:215-220`）。
- 改为：schema 与模型已支持固定尺寸和尺寸范围，但当前数据中没有任何带真实尺寸要求的对象，也没有对应的子集统计。

### 3.4-a 资产不重叠评测子集 — 未实现
- 规范：「另设资产不重叠评测子集」
- 现状：在 `fastfill/v2` 中 grep 不到任何与资产不重叠相关的代码或数据。`docs/fastfill-v2-review-20261005.md:133` 和 `:214` 仍把它列为待交付。现在只有来源级留出：manifest 的 `evaluation_only_sources` 为 SceneSmith 和 SpatialGen。
- 改为：资产不重叠评测子集尚未建立；目前只有来源级留出（SceneSmith、SpatialGen 仅用于测试）。

### 3.4-b 增强：整体平移/旋转、语义描述改写 — 未实现
- 规范：「增强：整个房间与对象联合平移/旋转、语义描述改写」
- 现状：
  - `_rigid_xy`（`batch.py:334-370`）对房间、固定物体、约束和目标统一做绕 +Z 的 k·90° 旋转，然后重新锚定到原来的最小角。没有平移，也没有任意角度旋转；mirror 为 false（config:76）。
  - 每个房间都预先锚定为多边形最小角在 (0,0)（test 8,620/8,620 行），所以平移增强会让输入离开规范坐标系。
  - 没有语义改写：`category_only_description_p 0.5` 只是把整条描述替换成类别名（`batch.py:408-415`）。
- 改为：当前只做房间与对象联合的 90° 整数倍旋转，旋转后重新锚定到外包最小角（rotate90=true，mirror=false），门窗、方向约束和目标同步变换。没有平移增强，因为预处理把房间外包最小角固定在原点。没有语义改写，只有以 0.5 概率把描述整体替换为类别名的降级增强。

### 3.4-c 实际增强列表 — 已过时
- 规范：「增强：…对象输入顺序变化、匿名 ID 重命名」
- 现状：
  - 实际增强由 `AUGMENT_DEFAULTS` 和运行配置给出（`batch.py:283-285`；config:74-82）：`shuffle_objects` 打乱顺序并重编号为 `obj_%04d`（`batch.py:301-321`），`drop_constraints_p 0.3`、`drop_support_p 0.2`、`category_only_description_p 0.5`、`minimal_form_p 0.5`（`batch.py:408-421`）。
  - minimal 只作用于边界已知的轴对齐矩形房间（`batch.py:127-134`），在 train 中有 65,987/124,584 行（53.0%）。它去掉固定物体（含门窗）、约束和 support_parent，因此地面 z 需要模型自己学，并且会重新计算分组（`_regroup`）。
- 改为：实际训练增强为：rotate90；对象顺序打乱并重编号为 obj_%04d；以 0.3 概率丢弃约束；以 0.2 概率丢弃支撑声明；以 0.5 概率把描述降为类别名；以 0.5 概率对边界已知的轴对齐矩形房间改用三字段投影（只保留外包矩形、房型、房高和物品清单）。不做镜像、平移和尺度增强。

### 3.4-d 统计量来源 — 错误
- 规范：「统计量只由训练数据确定」
- 现状：
  - 模型输入符合规范：归一化按各自房间的输入计算（`batch.py:178-180`），`size_reference` 是常数 [1,1,1]；训练器的基线用训练行（`train.py:649`，`samples[:20000]`）。
  - 但 `evaluate.py:929-935` 在没有传入 `--baseline-fit` 时，直接在评测集上拟合纯标签基线（`fit_source` 为 `evaluation_set_leave_one_out`），而 autorun 从不传这个参数。
  - `autorun.score` 以这些基线为分母（`autorun.py:44-69`），SUMMARY 也把它们标为 “(eval-set LOO)”。
- 改为：模型输入的归一化只依赖各自房间，训练日志里的基线统计来自训练集。但评测脚本的纯标签基线默认在评测集上逐行留一拟合，autorun 的选模分数以这些基线为分母。要符合本条，应显式用 --baseline-fit 指向训练集。

## 4. 模型：条件编码与结构化对象解码

### 4.1-a 编码器与解码器结构（含 §4.2 解码器部分） — 成立
- 规范：「Qwen 条件编码器（预训练 + LoRA/可训练层）→ 完整条件 hidden states → N 个绑定请求对象的 slots → 对象解码器：双向 self-attention + 条件 cross-attention」
- 现状：
  - `model.py:125-140` 要求骨干为 Qwen 系列，并用 PEFT LoRA 包装；主配置为 Qwen3-8B、LoRA rank 8，作用于 q/k/v/o（config:3, 20-29）。
  - 解码器是 `nn.TransformerDecoder`（`model.py:150-153`），self-attention 不加因果掩码，cross-attention 作用于全部条件 token 的 memory（`model.py:191-199`）。
  - 模型输入只有条件 token、对象区间和条件给定的固定值（`model.py:30-33`）。

### 4.1-b 主架构输出端 — 已过时
- 规范：「→ 连续几何张量 → 确定性校验与序列化」
- 现状：
  - `--grid-decode` 默认为 spread（`predict.py:39-40`；`evaluate.py:1067`）。`serialize_predictions`（`evaluate.py:340-381`）先调用 `spread_grid_xy`（`evaluate.py:121-328`），再调用 `validate_layout`（`schema.py:293`）。
  - spread 是贪心放置：先放地面物体，再放高处物体，同类中按占地面积从大到小；每个物体在 top-64 格子中选位置，允许最多 15% 重叠。它还会把未声明支撑的低处物体贴地、把墙面物体投影到墙上、把声明了支撑的子物体放到父物体上，并在靠墙时把 yaw 翻转 π。
  - `README.md:613-617`：spread 是默认交付方式，autopilot 的选模和测试都用它；`--grid-decode argmax` 保留原始头输出。
- 改为：→ 头输出（grid 位置 logits/residual、尺寸、yaw）→ 默认用确定性的贪心 spread 解码（evaluate.spread_grid_xy：按 top-k 网格逐个放置以避开碰撞和出界；z 按支撑声明确定，未声明的低处物体贴地；墙面物体贴墙；靠墙物体的 yaw 可翻转 π）→ validate_layout 做 schema 校验 → JSON 序列化。--grid-decode argmax 保留原始头输出，两种结果需分开报告。

### 4.2 对象特征 — 未实现
- 规范：「对象特征从其条件 token 区间、类别特征、约束角色及 slot seed 构造」
- 现状：
  - `model.py:163-181` 的 `_slots`：每个 slot 是该对象 token 区间（经 `memory_projection` 后）的均值池化，加上 `slot_seed[slot 序号]`。没有类别嵌入，也没有约束角色特征。
  - 约束在条件中是更靠前的独立片段（`batch.py:67`，`CONDITION_FIELD_ORDER` 把 constraints 放在 objects 之前）。
  - `schema.py:56-59` 允许 `constraint_role` 字段，但 train 的 1,998,842 个对象中没有一个带它，对象键只有 id/category/description/support_parent。
- 改为：slot 的初始特征 = 该对象 JSON token 区间 hidden states 的均值（投影后）+ 按请求序号学习的 slot seed。类别和 support_parent 只以文本形式出现在该区间内；约束只能通过编码器的因果注意力和解码器的 cross-attention 间接获得。没有单独的类别嵌入或约束角色特征。

### 4.3-a 每请求一个 slot、padding 屏蔽、无 objectness — 成立
- 规范：「每个请求一个 slot；padding 在注意力、匹配、loss 一致屏蔽；无 background/objectness」
- 现状：`slot_mask` 用于解码器的 `tgt_key_padding_mask` 和所有输出（`model.py:196-199, 224-227`）；匹配要求 slot 连续且与请求一一绑定（`matching.py:119-120`）；loss 只在有效 slot 上计算（`losses.py:127-133`）。模型只有位置、尺寸、yaw 三个头，没有 objectness 头（`model.py:156-160`）。

### 4.3-b 输入顺序与 ID 重命名稳定性（与 §9 消融合并） — 无法核实
- 规范：「需实测输入顺序打乱和 ID 重命名的稳定性」；§9「消融：输入顺序扰动、ID 重命名」
- 现状：只有 `ops/diag_ablation.py` 的变体 e（`:12`）。它用 `batch._shuffle_objects` 一步完成顺序打乱和 `obj_%04d` 重命名，两种效应没有分开。这个任务已排进 `ops/run_plan.sh:124-147`（Isambard 上的 abl-* 任务），但本地没有结果。`evaluate.py` 没有顺序或 ID 扰动指标；训练时 `shuffle_objects=true`。
- 改为：输入顺序扰动只有诊断脚本 ops/diag_ablation.py 的变体 e（打乱顺序并同时重命名为 obj_%04d），没有单独的 ID 重命名测试，评测里也没有对应指标。结果需等 Isambard 分析作业产出。

### 4.3-c support parent 输入 — 已过时
- 规范：「上游已知 support parent 直接作为输入并在运行时核验」
- 现状：
  - 有 support_parent 时，它渲染在对象区间里，地面声明还会固定 z（`batch.py:258-270`）。运行时 Validator 会核验它（`validation.py:213-244`），spread 也按它放置（`evaluate.py:365`）。
  - `render_minimal_condition` 只保留 id/category/description（`batch.py:155`）。这个投影用于合格房间约 50% 的训练样本（`minimal_form_p 0.5`，另有 `drop_support_p 0.2`），也用于选模（`train.py:576`；`autorun.py:291`）和主测试表。
  - train 中 1,369,298/1,998,842 个对象（68.5%）带 support_parent。
- 改为：上游声明的 support_parent 在完整条件或直接请求中作为对象文本输入（地面声明还会固定 z），由运行时 Validator 核验，并指导 spread 放置。但三字段投影（训练中合格房间约一半的样本，以及选模和主测试表）会去掉 support_parent，这条路径上支撑检查全部为 unknown。

### 4.4-a 尺寸头、固定尺寸、角度头 — 成立
- 规范：「尺寸 ŝ=s_ref⊙exp(u)。固定尺寸维度直接采用给定值并屏蔽监督…角度头 K 个 logits + K 个归一化 residual」
- 现状：尺寸为 `size_reference * exp(clamp(u))`（`model.py:212-213`）；固定尺寸用 `torch.where` 直接替换（`model.py:217-220`）；yaw 头是 K 个 logits 加 K 个 residual，residual 经 tanh（`model.py:159-160, 221-223`）。

### 4.4-b 位置头（与 §7.2 数值头合并） — 已过时
- 规范：「位置头输出归一化底面中心 p̂̃（连续回归）」；§7.2「数值头 normalized bottom-center」
- 现状：
  - 运行配置设 `position_head "grid_residual"`、`position_grid 16`（config:18-19）。
  - `model.py:156-157` 是 `Linear(128, 3*256+1)`，`model.py:203-209` 把输出拆成 256 个格子 logits、256 对以半格为单位的 tanh XY residual，以及一个回归的 z。
  - 解码取 argmax 格子中心加该格 residual（`geometry.py:50-58`）。超出 [0,1) 的位置被夹到边界格，residual 为 ±1（`geometry.py:37-47`）。
  - `ModelConfig` 的默认值仍是 `'regression'`（`model.py:53`）。
- 改为：主配置的位置头为 grid_residual：在房间外包矩形归一化的 XY 上做 16×16 网格分类（256 个 logits），每个格子再预测一对 tanh XY residual（以半格为单位），z 单独连续回归。原始解码 = argmax 格子中心 + 该格 residual，再按 2.4 反归一化；交付默认还经过 spread 解码。纯连续回归的 regression 头保留为可选项。

### 4.5 梯度边界 — 已过时
- 规范：「连续 head 输出直接进入几何 loss」
- 现状：
  - grid 头下，XY 由格子 CE 加 GT 格 residual 监督（`losses.py:101-114, 164-170`）。
  - argmax 解码出的 `position_normalized` 只进入 detached 的 `term_sums['position']`（`losses.py:160-161`）和 box、碰撞、边界三项；这三项在运行配置中权重都是 0（config:40,45-46；`losses.py:218, 226-235`）。
  - 尺寸、z 和 yaw 的输出直接进入各自的 loss；Hungarian 在 no_grad 下运行（`matching.py:89`）。
- 改为：尺寸、z 和 yaw 头的输出直接进入 loss；XY 通过网格 CE 加 GT 格 residual 监督，argmax 解码出的位置不进入任何 loss（box、碰撞、边界权重均为 0），只用于记录。Hungarian 在 no-grad 的代价上计算。

## 5. 实例对应与受约束 Hungarian matching

### 5.1 / 5.4 分组规则与关系置换 — 成立
- 规范：「完全相同且关系允许交换的匿名实例 → 组内 Hungarian；交换依赖其他组同步置换 → 保留固定身份」
- 现状：`certify_group` 要求组内成员除 id 外的字段完全相同，并验证交换后约束图和支撑引用不变，否则报错并退回固定身份（`matching.py:30-45`）；`group_labels` 只把位置标签完整的成员放进组（`matching.py:48-77`）。§5.4 的关系置换已实现为 `permute_relations`（`matching.py:175-193`），但只在测试中调用（`tests/test_supervision.py:54`），因为训练中没有关系或支撑监督目标。

### 5.2-a 匹配成本（与 §7.2 assignment 合并） — 已过时
- 规范：「C_ij = α_p‖p̂̃_i − p̃*_j‖₁ + α_s‖log(ŝ_i⊘s*_j)‖₁」；§7.2「assignment 固定身份 + 组内 position/size cost，α_p=α_s=1」
- 现状：
  - 解码后位置的 L1 成本只在 regression 头下使用（`matching.py:154-155`）。
  - grid 头下，每对的成本为 α_p·[position_cell·CE + position_residual·residual + z/3]，只在可学习的坐标上计算（`matching.py:139-153`；`README.md:192`）；运行配置中两个系数为 0.5 和 0.4。
  - 允许轴交换的目标，尺寸成本取两种轴序中较小的一个（`matching.py:161-163`）。与 loss 不同，这里不考虑 sx/sy 被固定的情况。
  - 组内只要有一个成员尺寸标签不完整，就去掉尺寸项（`matching.py:156`）；train 中 69,404/324,456 个组（21.4%）属于这种情况。
  - 评测时的匹配用解码后位置的 L1。α 为 1/1、hungarian 为 true 与规范一致（config:41-44）。
- 改为：assignment 采用固定身份，加上已认证可交换组内的 Hungarian，α_p=α_s=1。grid 头的位置成本就是该配对的 grid 位置 loss：α_p[0.5·CE(目标 j 所在格) + 0.4·该格 XY residual 误差均值 + |Δz|/3]。允许轴交换的目标，尺寸成本取 (sx,sy) 与 (sy,sx) 中较小者。组内有成员尺寸标签不完整时只用位置成本。regression 头和评测匹配使用解码后位置的 L1。

### 5.2-b 匹配消融（与 §9 合并） — 未实现
- 规范：「主配置 position+size cost；box 或 yaw cost 为消融」；§9「固定对应 vs 组内 Hungarian、cost 加 yaw/box」
- 现状：`match_batch` 只接受 `alpha_position` 和 `alpha_size`，没有 yaw 或 box 选项（`matching.py:80-90`）。`README.md:212`（“yaw is not in the matching cost”）和 `:748-750`（“additional matching cost ... future ablations”）确认了这一点。固定对应与 Hungarian 的对比有配置（`configs/structured_fixed.json`、`configs/structured.json`）和 `evaluate --fixed-correspondence`，但只跑过 2026-10-05 的冒烟测试（`outputs/fastfill_v2/hungarian-smoke-*`）。
- 改为：主配置只用 position+size 成本。匹配代价没有 yaw 或 box 选项，相关消融未实现。固定对应与组内 Hungarian 的对比只有配置和冒烟测试，没有正式对比。

### 5.3-a detached 匹配与可微 loss — 成立
- 规范：「no-grad 构造 cost，组内 one-to-one assignment；用索引从原始预测计算可微 loss」
- 现状：`match_batch` 带 `@torch.no_grad()` 装饰（`matching.py:89`），用 `linear_sum_assignment` 做一对一分配（`matching.py:168`）；loss 先按分配索引 gather 目标，再与原始预测计算（`losses.py:135-137`）。

### 5.3-b 记录 tie-breaking 与无效样本 — 未实现
- 规范：「保存 tie-breaking、无效样本计数」
- 现状：平局由 SciPy 的行/列顺序决定（`matching.py:167-168`），没有记录。`README.md:212` 记有一例：位置成本相同时，yaw_cls 会随 slot 顺序取 0 或 20。训练中位置标签不完整的组被静默跳过（`matching.py:134`），没有计数。分组计数只在评测中有（`evaluate.py:551-557`：`incomplete_groups`、`position_only_groups`、`matching_scope`），构建 manifest 中有 `qualification_change_counts.exchangeable_group=98`。
- 改为：平局按 SciPy 的行/列顺序隐式决定，不做记录（已知 yaw 不同但位置成本相同时，平局结果会影响 yaw loss）。训练时不统计被跳过的无效组；只有评测报告 incomplete_groups/position_only_groups，数据构建 manifest 记录被降级的组。

## 6. 训练目标

### 6.1 损失函数 ρ — 已过时
- 规范：「ρ = beta 为 1 的 SmoothL1（Huber）」
- 现状：`LossConfig` 默认 `position_type=size_type='l1'`（`losses.py:53-54`），运行配置也是如此（config:47-48）；`_elementwise` 在 l1 下返回 `abs()`（`losses.py:73-75`）。grid 的 XY residual 和 z 也用 `position_type`（`losses.py:111,113`）。`docs/fastfill-v2-loss-calibration-20261006.md:44` 记录了从 SmoothL1(beta=1) 改为 L1 的过程。只有 yaw residual 仍硬编码为 beta=1 的 SmoothL1（`losses.py:197`）。
- 改为：位置项（含 grid residual 和 z）与 log 尺寸项默认用 L1，主配置也是 L1（loss.position_type = size_type = "l1"）；smooth_l1 只是可选项，β 由 loss.smooth_l1_beta 给定。只有 yaw residual 固定使用 β=1 的 SmoothL1。

### 6.2-a L_p 形式 — 已过时
- 规范：「L_p = 1/max(1,|M_p|) Σ_i 1/3 Σ_q m^p_iq ρ(p̂̃_iq − p̃*_π(i)q)…按固定三坐标平均」
- 现状：grid_residual 下，position = position_cell·CE(GT 格) + position_residual·mean_xy L1(GT 格 residual) + L1(z)/3（`losses.py:101-114, 164-170`），两个系数为 0.5 和 0.4（config:50-51）。z 被条件固定时贡献 0，但该 slot 仍留在分母中（`losses.py:112-113, 170`）。`term_sums['position']` 只是没有梯度的解码诊断量。按三坐标除以 3 的平均对 regression 头和尺寸项仍然成立（`losses.py:154-155`）。
- 改为：主配置使用 grid_residual 位置头：L_p = 0.5·CE(GT 格) + 0.4·mean_{x,y} L1(GT 格内 XY residual，以半格为单位) + L1(z)/3。z 被条件固定时 z 项为 0，但不缩小分母。按三坐标除以 3 的平均只适用于 regression 头和尺寸项。

### 6.2-b 掩码、分母与全局计数 — 成立
- 规范：「条件固定坐标屏蔽但不缩小分母…分布式用全局有效计数」
- 现状：固定坐标在求和前置 0，然后除以固定的 3（`losses.py:151-155`）；`_mean` 在分布式下对有效计数做 `all_reduce`，并对 DDP 的梯度平均做补偿（`losses.py:79-88`）。

### 6.3 yaw bins 与 residual — 成立
- 规范：「k*=⌊wrap_[0,2π)(θ*+Δ/2)/Δ⌋ mod K；r*=wrap(θ*−φ_k*)/(Δ/2)…只取 GT bin 的 residual…K=12」
- 现状：`encode_yaw` 和 `decode_yaw` 与公式一致（`geometry.py:13-34`）；CE 加上只取 GT bin 的 residual loss（`losses.py:194-197`）；K=12（config:9）。

### 6.4-a 对称候选的联合最小 — 成立（但标签来源已过时，见 §2.1）
- 现状：每个候选计算整套“分类 + residual”代价，再选同一候选的联合最小（`losses.py:194-205`）。问题在于 order 主要由来源策略决定，见 §2.1。

### 6.4-b 尺寸轴固定（与 §2.1 局部尺寸轴合并） — 已过时
- 规范：「s=(w,d,h) 为局部 X、Y、Z 全长度…局部尺寸不随 yaw 变化」；§6.4 原文加粗句「**尺寸轴也必须固定，不能通过随意交换 w,d 隐式制造等价标签。**」
- 现状：
  - 契约 K1（`README.md:87`；`legacy_bridge.py:22-35, 108-110`）按来源/类别规则设置 `validity.size_axis_swap_allowed`。全部 split 中有 147,673 个这样的对象（manifest 的 `source_size_axis_swap_allowed_objects`），train 中有 126,173 个（6.3%）。
  - 对这些对象，loss 在 k=0..3 上对 (sx,sy,yaw+k·π/2) 取 detached 联合最小，奇数 k 时交换 sx 和 sy（`losses.py:12-19, 157-158, 171-206`）。匹配取两种轴序中较小者（`matching.py:161-163`）。
  - 只有 sx 或 sy 被固定时才锁定轴序（`losses.py:174, 193`），而数据中没有任何固定尺寸。
- 改为：s=(w,d,h) 原则上是局部 X、Y、Z 方向的全长度，且不随 yaw 变化。但从第二轮契约 K1 起，size_axis_swap_allowed 为真的对象（按来源/类别规则设定，训练集 12.6 万个，占 6.3%）可以写成 (sy, sx, yaw + kπ/2)。loss 在 k∈{0,1,2,3} 四个候选中取 detached 联合最小值，匹配取两种轴序中较小的代价，评测报告盒等价误差和 *_plain_convention。只有请求固定了 sx 或 sy 时才锁定轴序，而当前数据没有固定尺寸。

### 6.5 框重叠监督 — 成立
- 规范：「L_box = 1−GIoU；基础主配置 λ_b=0；BEV 与 3D oriented box 不同」
- 现状：只实现了显式命名的 `bev_oriented_giou_convex_hull` 算子（`losses.py:120-121`），loss 为 `1 - bev_giou(...)`（`losses.py:218-222`）；运行配置 `box: 0.0`（config:40-41）。

### 6.6-a L_total — 未实现
- 规范：「L_total = L_sup + λ_col L_collision + λ_bd L_boundary + λ_req L_requirements」
- 现状：`LossConfig` 有 collision 和 boundary 两项，没有 requirements 项（`losses.py:41-58`）。collision 是两个直立 OBB 的交体积除以两盒平均体积（`regularizers.py:26-50`）；配对包括全部固定物体（门窗也在内），只排除声明的支撑父物体（`regularizers.py:71-78`）。boundary 用半平面计算，只支持已知的凸多边形，否则报错（`regularizers.py:17-23, 53-61`）。两项在运行配置中权重都是 0（config:45-46）。
- 改为：L_total = L_sup + λ_col L_collision + λ_bd L_boundary，L_requirements 尚未实现。collision 是成对直立 OBB 的交体积除以两盒平均体积，声明的支撑对除外，并包括与门窗等全部固定物体的碰撞；boundary 只支持已知凸多边形。主配置中两者权重均为 0。

### 6.6-b 场景正则消融与防缩小（与 §9 合并） — 未实现
- 规范：「需消融并防止"缩小家具换取低碰撞"」；§9「消融：box loss/场景正则」
- 现状：`fastfill/v2/configs` 中没有任何配置把 collision 或 boundary 设为大于 0。只有 `configs/structured_bev_box.json` 设 box 为 0.1，但没有跑过对比（`README.md:746-749`）。没有防缩小项；尺寸只靠完整尺寸标签对象上的 log-size 监督约束（train 中 1,351,470/1,998,842 个对象，67.6%）。碰撞只在推理时由 spread 处理。
- 改为：box、碰撞和边界 loss 可以配置，但正式训练里权重全为 0，没有做消融，也没有防止“缩小家具换低碰撞”的专门机制。尺寸只靠完整尺寸标签对象（约 67.6%）上的 log-size 监督约束；碰撞目前只靠推理端的 spread 后处理缓解。
- 补充：§7.3「box loss 后期开启要显式记录」不单列为缺口。这条要求本身是有条件的；而且 resume 拒绝任何配置变更（`train.py:598-599`），box 权重只能出现在新运行的配置记录里。

## 7. 训练配置与执行

### 7.1 流程 — 未实现
- 规范：「审计样本 → 联合训练结构化模型（四项 L_geom）→ 验证集选模与增强配置 → 真实资产闭环评估 → 冻结部署包」
- 现状：
  - autorun 的阶段划分见 `autorun.py:1-19`。增强参数在配置中写死（config:74-82），从不参与选择。
  - 选模用 `autorun.score`，在最新 3 个（B 阶段）或 4 个（F 阶段）checkpoint 中选择（`autorun.py:249-258, 529`），条件为三字段投影，解码为 spread（`autorun.py:290-292`）。
  - 最佳模型在验收检查之前就上传到私有 HF 仓库（`autorun.py:532`，标为 “UNACCEPTED backup”）。
  - RoomGenBench 导出为 `layout_boxes` 的 bbox 场景（`ops/run_checkpoint.sh:22`）；按 `autorun.py:450-451`，“their validator/physics are not run”。流程中没有资产库或资产解析步骤。
- 改为：当前流程为：preflight 审计 → 单配置联合训练（增强参数固定，不在验证集上选择）→ autorun 在最新 3 或 4 个 checkpoint 中按 score 选模（三字段投影 + spread 解码）→ 把最佳模型上传到私有 HF 仓库，作为验收前的备份 → 测试评测（三字段与完整条件）→ 把 5 个 RoomGenBench 房间导出为 layout_boxes bbox 场景。真实资产闭环评估尚未接入。

### 7.2 起始配置与 λ 标定 — 成立（数值头与 assignment 见 §4.4-b、§5.2-a）
- 规范：「固定 Qwen checkpoint…K=12 yaw bins + residual…四项 loss 各 λ 从 1 起步再标定」
- 现状：`LossConfig` 中四项默认值都是 1（`losses.py:42-45`），运行配置为标定后的 position 1.0、size 0.6、yaw_cls 0.08、yaw_reg 2.0（config:36-39；Isambard 的另一组把 yaw_cls 设为 0.5）。骨干固定为 Qwen3-8B 的本地 checkpoint（config:3, 31）。

### 7.3-a 验证集选模依据 — 错误
- 规范：「验证集选择依据同时包含对象完整性、原始几何可用性、真实资产解析率和约束通过率；单参考几何误差只是补充」
- 现状：
  - `train.py:92-96` 和 `710-712` 记录的 `selection_metric` 是 `validation.minimal.geometry_objective`：在三字段投影的验证行上算加权单参考 loss，验证行先按 `exclude_flags` 过滤（config:84-88）。
  - `autorun.score`（`autorun.py:44-69`）的组成：
    - log-size、对称 yaw、底面中心三项误差各自除以平凡基线的比值；
    - 分布偏差惩罚：|中央 1/4 区域占比 − GT|、|离墙距离 − GT|/GT、5×超出 GT 的重叠、5×超出 GT 的出界；
    - 10×无布局请求的占比。
  - score 不含资产解析率，也不含 Validator 或约束通过率；验证集本身也没有任何约束。autorun 只按这个 score 选模（`README.md:209`）。
- 改为：当前选模有两层。train.py 记录的 selection_metric 是三字段投影验证集上的加权几何目标，属于单参考 loss。autorun 实际在最新 3 或 4 个 checkpoint 中按 score 选择：三项单参考误差（log-size、对称 yaw、底面中心）相对平凡基线的比值，加上中心区比例、平均离墙距离、BEV 重叠率和出界比例相对 GT 的偏差惩罚，再加无布局请求占比×10。score 不含真实资产解析率和约束/校验通过率；验证集本身也没有约束。

### 7.3-b 过拟合、梯度可达、round-trip — 无法核实
- 规范：「少量可信样本可过拟合、梯度可达、round-trip 一致」
- 现状：
  - 梯度可达：`tests/test_model.py:43`、`tests/test_model_loss_r2.py:143`。
  - round-trip：`tests/test_model.py:92`、`tests/test_model_loss_r2.py:101,125`、`tests/test_geometry_core.py`。
  - 单个 fixture 上的 loss 下降：`tests/test_execution.py:64`。玩具双峰样本拟合：`tests/test_model_loss_r2.py:193`。
  - 没有找到在真实可信小样本集上过拟合的运行记录。
- 改为：单元测试已覆盖各 head（含 grid 头）的梯度可达性，yaw、grid 编码与 checkpoint 的 round-trip，单样本 loss 下降，以及一个玩具双峰样本的拟合；在真实可信小样本集上过拟合的运行记录尚未找到。

### 7.4-a 文本 SFT 对照的条件与数据 — 已过时
- 规范：「文本 SFT 对照：相同 condition 与目标字段，Qwen/LoRA + token CE 输出 JSON」
- 现状：
  - `text_sft.py` 使用相同的 `tokenize_condition` 和目标字段，Qwen/LoRA（CAUSAL_LM，q/k/v/o），只在 assistant token 上算 CE（`text_sft.py:25-36, 83-98`），这些与规范一致。
  - `_read_samples` 对每一行调用 `_answer`，遇到不完整标签直接报错（`text_sft.py:30-31, 108-112`）。所以除非先过滤数据，它在当前 train.jsonl 上会失败；过滤后只剩 82,991/124,584 行。
  - 它没有增强，也没有三字段投影；采样是有放回的（`text_sft.py:208`）；没有验证集选模；默认骨干为 `Qwen/Qwen2.5-0.5B-Instruct`。
- 改为：text_sft.py 使用与主模型相同的条件渲染和目标字段，以 Qwen/LoRA + assistant-token CE 输出 JSON。但它要求每一行的标签都完整，遇到不完整行直接报错，不会跳过，所以必须先把数据过滤成完整行子集（当前训练集 82,991/124,584 行）。它也没有增强、没有三字段投影、没有验证集选模，因此与当前主模型的训练设置已不对等。

### 7.4-b 文本 SFT 对照与 structured vs 文本 SFT 消融（与 §9 合并） — 未实现
- 规范：「可与主模型并行开发」；§9「对照：文本 SFT」「核心消融：structured vs 文本 SFT」
- 现状：`README.md:769-773` 记录只跑过一次 2026-10-05 的冒烟，其中一次单请求生成的 JSON/schema 解析失败；`README.md:748` 写明 “No such quality comparison has been run.” autorun 和 `ops/*` 都不调用 text_sft。实际做的对比是未微调的 API LLM 智能体（`llm_baseline.py`、`llm_structured.py`）。
- 改为：文本 SFT 对照尚未正式训练，只跑过一次冒烟，且生成的 JSON 解析失败；structured 与文本 SFT 的对比消融没有做。目前与主模型并列比较的是未微调 API LLM 的 prompt/harness/structured 基线，输入为相同的三字段字段。

## 8. WorldEdge 运行流程与资产闭环

### 8-a 运行流程与交付定位 — 未实现
- 规范：「Harness → FastFill v2 → Asset Resolver → … → Host atomic Commit → Persistent World → Solver…」「该流程作为主系统整体交付。」
- 现状：`runtime.py:338-448` 的 `run_pipeline` 包含 schema、`CatalogResolver`、reconcile、bbox 级 `validate_scene`、资产重选、可选的位姿修复和内存中的 `AtomicMemoryHost`。没有 Harness、没有持久世界、没有 Solver；`solver_success` 硬编码为 None（`runtime.py:333`；`evaluate.py:892`）。`README.md:8-10, 20-21, 715-716` 写明真实 Host 适配器和 mesh/physics/Solver 仍属待集成工作。仓库中没有资产库文件，autorun 和 ops 从不传 `--catalog`。实际交付的下游是 RoomGenBench 交接（`README.md:25-32`）。
- 改为：当前只实现离线参考链：FastFill 预测 → CatalogResolver → reconcile → bbox 级 Validator →（有限次资产重选、可选平移修复）→ 内存 AtomicMemoryHost。没有接入 Harness、持久世界、Solver 和真实 WorldEdge Host 适配器，也没有在真实资产库上运行过。实际交付的下游是 RoomGenBench 交接（predict --export-dir + fastfill.v2.roomgenbench）。

### 8-b Resolver 契约 — 成立
- 规范：「硬过滤 + d_size=‖log(s_actual⊘ŝ)‖₁ 排序；不默认非均匀缩放」
- 现状：`CatalogResolver.resolve` 先按资格过滤，再按 `Σ|log(actual/target)|` 加 ref 排序（`runtime.py:141-144`）；`_transform` 只做 yaw 旋转和平移，再乘资产的规范变换，不做缩放（`runtime.py:147-150`）。

### 8-c 失败处理 — 未实现
- 规范：「未通过 → 有限次重选/修复/重规划」
- 现状：重选见 `runtime.py:384-409`，默认 `max_asset_retries=2`（`runtime.py:191`）。修复只有 `BoundedTranslationRepair`（`runtime.py:204-249`），即单步最多 0.25 m 的 XY 平移，默认关闭（`max_repair_calls=0`，见 `runtime.py:192`、`serve.py:51`、`evaluate.py:1060`）。重规划（再次调用 FastFill）不存在。
- 改为：未通过时依次做有限次资产重选（默认 2 次）和可选的有界平移修复（默认 0 次，单步 ≤0.25 m，只改位姿）。重规划（重新调用 FastFill）没有实现。

### 8-d Validator 层级：开口与通行 — 未实现
- 规范：「目标几何（房间内外、开口与通行、约束、代理碰撞与支撑）」
- 现状：`validation.py:393-394` 只在 `room.openings` 非空时输出 `openings_unchecked`（unknown），而当前没有任何行有 openings。门窗是 fixed_objects，只作为盒子障碍检查碰撞（`fixed_collision`，`validation.py:325-332`），没有自动的门前净空或通行检查。用户显式声明的 `clearance`（最小距离）和 `keepout`（多边形）约束是支持的（`validation.py:287-297`），但数据中没有一行带这类约束。三字段投影会去掉门窗，RoomGenBench 的 `benchmark_request` 也不含门窗（`roomgenbench.py:253-265`）。
- 改为：开口净空与通行检查没有实现。门窗只是 room.fixed_objects 里的盒子障碍，只做碰撞检查；room.openings 字段从未使用。只有用户显式给出的 clearance、keepout 约束可以表达净空要求，但数据中没有。三字段投影和 RoomGenBench 请求都不含门窗。

### 8-e Validator 层级：实际资产与仿真 — 未实现
- 规范：「实际资产几何 / 仿真任务」
- 现状：在 actual 阶段，几何仍是由 `actual_size_local_m` 构造的直立 OBB（`validation.py:100-110`）；模块 docstring 写明只验证盒子（`validation.py:1-7`）。mesh、physics、solver 三层始终为 unknown（`validation.py:398-399`），把它们设为必需会阻止提交（`README.md:709-712`）。
- 改为：实际资产层只是把目标框换成资产实际尺寸的直立 OBB，没有 mesh 检查。mesh、physics、solver 三层始终报 unknown，没有接入任何检查器；把它们设为必需会一律阻止提交。

## 9. 评测与消融

### 9-a 分母 — 已过时
- 规范：「全部请求为分母」
- 现状：在被评测的投影内部，失败请求留在分母中（`evaluate.py:846-866`：`stage_denominator`、推理失败、超容量）。三字段投影会跳过非矩形或边界未知的房间：测试报告 `outputs/fastfill_v2/comparison-20261007b/test-main7-cell05-main-20261007b-e5/report.json` 的顶层为 `requests=4455`、`skipped_non_rectangular_rooms=4165`；它的 `projections.full` 为 `requests=8620`，跳过 0 间。选模和 SUMMARY 主表都用三字段的数字。
- 改为：在被评测的投影内，所有请求（含推理失败和超容量请求）都计入分母。但三字段投影会跳过非矩形或边界未知的房间（测试集 8620 间中跳过 4165 间），选模和主表都用这一投影。完整条件的分母（8620 间）只出现在测试报告的 projections.full 中，报告时必须同时写出被跳过的房间数。

### 9-b 指标：输出与需求 — 未实现
- 规范：「schema 有效率、完整率、尺寸与能力满足率」
- 现状：`requested_ids_exactly_once` 和 `positive_valid_size` 与 `schema_success` 同时赋值（`evaluate.py:700, 995`），所以完整率从未单独测量。尺寸和能力检查有代码（`validation.py:171-197`），但 test 中没有一个对象带 `fixed_size_local_m`、`size_bounds_local_m`、`required_capabilities` 或 `attributes`，actual 阶段也从未运行，这两个比率都是空的。
- 改为：目前只有 schema 有效率是独立指标；“完整率”和“正尺寸”与它是同一个布尔值。尺寸/能力满足率在代码里有检查，但现有数据没有任何尺寸范围或能力要求，资产阶段也从未运行，所以这两项指标为空。

### 9-c 指标：几何可用性 — 未实现
- 规范：「几何可用性（边界、通行、碰撞、支撑、面向，注明 bbox/mesh/physics 层级）」
- 现状：边界、天花板、碰撞、`fixed_collision` 和支撑按检查码计数（`evaluate.py:800-812`），层级也有记录（`geometry_level`、`required_levels`、mesh/physics/solver_unchecked）。没有通行指标。面向只能通过 faces/faces_direction 约束检查，而 validation 和 test 中没有这类约束。三字段视图下所有对象都是 `support_unknown`（`validation.py:216-217`）；完整视图下，放在物体上的支撑总是 `support_surface_unknown`，因为没有对象带 support_surfaces（`validation.py:232-235`）。
- 改为：已报告边界、天花板、碰撞（含与固定物）和支撑的 pass/violation/unknown 计数，并标注 bbox 层级。通行没有指标；面向只能通过 yaw 参考误差间接体现。三字段视图下支撑全部为 unknown；完整视图下放在物体上的支撑也因为没有 support_surfaces 而为 unknown，只有地面/墙面支撑能判定通过。

### 9-d 指标与消融：资产与任务闭环、修复与验收层级 — 未实现
- 规范：「资产与任务闭环」「无修复 vs 固定预算修复、目标框 vs 实际资产验收」
- 现状：资产指标（`retrieval_coverage`、`capability_satisfaction`、`first_pass_actual_geometry_validation`）只在传入 `--catalog` 时出现（`evaluate.py:864-892`）。仓库中没有资产库文件，autorun 和 ops 从不传 `--catalog` 或 `--repair-calls`，`solver_success` 恒为 None。替代品是 RoomGenBench 的 `--require-placement` 检查（`ops/run_checkpoint.sh:22`；`ops/llm_compare.py:40` 的 `interface_pass_rate`）。
- 改为：资产闭环和任务指标从未运行：没有资产库，Solver 恒为 None。修复预算对比和目标框/实际资产验收对比也都没有运行。目前替代它们的是 RoomGenBench 交接的 require-placement 通过率，这只是接口层面的检查，不代表物理或任务成功。

### 9-e 对照：未微调语言模型（含规范未提及的 LLM 智能体对照） — 已过时
- 规范：「相同条件下未微调语言模型」
- 现状：
  - 没有未微调 Qwen 的运行。取而代之的是外部 LLM 智能体（gpt-6.1-sol）：`llm_baseline` 的 prompt/harness 模式和 `llm_structured` 的 structured/structured-harness 模式。
  - 它们回答前 300 条三字段验证行（`llm_baseline.py:189`；`llm_structured.py:242`），使用相同字段和同一个 evaluate 评分器（`README.md:786-849`）。
  - 结果：FastFill 尺寸误差约 0.24，LLM 为 0.39–0.41；LLM 的 yaw 接近均匀随机；位置差异不显著。
  - 这 300 行属于 FastFill 的选模队列（`README.md:849`）。本地 `outputs/fastfill_v2/comparison-20261007b` 中只有 `llm-prompt-300` 和 `llm-harness-300`。
- 改为：“未微调模型”对照改为外部 LLM 智能体（gpt-6.1-sol，prompt、harness、OptiScene 式 structured、structured+harness 四种模式），在 300 条三字段验证行上与 FastFill 使用同一组字段和同一评分器。FastFill 的尺寸误差明显更低；LLM 的 yaw 接近随机；位置差异不显著。这 300 行属于 FastFill 的选模队列，不是留出集；也没有做含门窗的全条件对照。整体场景协调性目前没有量化指标，目视渲染中 FastFill 不如 LLM harness，有待 VLM 评审确认。

### 9-f 对照：类别尺寸先验 + 布局模型 — 未实现
- 规范：「类别尺寸先验 + 同条件布局模型」
- 现状：只有纯标签基线：房间中心、类别均值位置、类别中位尺寸、均匀 yaw（`evaluate.py:50-51, 619-690`），以及 diag_ablation 中不看输入、用 TRAIN 统计的类别格先验（`ops/diag_ablation.py:1-2`）。没有把先验尺寸与布局模型组合起来的流程。
- 改为：只有基于标签统计的平凡基线（房间中心、类别均值位置、类别中位尺寸、均匀 yaw，以及 diag_ablation 的类别格先验），没有“类别尺寸先验 + 布局模型”对照。

### 9-g 对照：ATISS/DiffuScene — 未实现
- 规范：「ATISS/DiffuScene 等」
- 现状：在 `fastfill/v2` 中 `grep -i 'atiss|diffuscene'` 没有结果；只存了论文（`outputs/fastfill_v2/review-20261005/papers/atiss-neurips2021.pdf`、`diffuscene-cvpr2024.pdf`）。
- 改为：没有实现或运行 ATISS、DiffuScene 等已发表方法的对照。

### 9-h 消融：参数化 — 未实现
- 规范：「log-size vs raw-size、yaw bins vs sin/cos」
- 现状：模型只有指数形式的尺寸和 bin + residual 形式的 yaw（`model.py:158-160, 212-225`），没有 raw-size 或 sin/cos 选项；`README.md:748-749` 把它们列为未来的消融。实际做过的头部对比是 regression 与 grid_residual 位置头，以及 position_cell 取 0.04/0.2/0.5/1.0（`README.md:610-612`；`configs/qwen3_8b_main_4gpu_*`）。
- 改为：代码里没有 raw-size 或 sin/cos yaw 选项，这两项消融没有实现。实际做过的头部对比是 regression 与 grid_residual 位置头，以及 position_cell 权重（0.04/0.2/0.5/1.0）配置。

### 9-i 统计 — 未实现
- 规范：「多种子或置信区间，按房屋聚类」
- 现状：运行配置用 seed 42（config:59），Isambard 的两组只在 yaw_cls 上不同（`ops/isambard_submit.sh:3-8`）。置信区间是房间级配对 bootstrap（`ops/compare.py:14-16`；`ops/stratify.py:324-335`），重采样的是房间而不是房屋。test 集的 8,620 个房间来自 3,319 个房屋，其中 1,300 个房屋有不止一个房间，最多的有 252 个。
- 改为：只有单种子（42）。置信区间是以房间为单位的配对 bootstrap，没有按房屋聚类；而测试集里有 1300 个多房间房屋（最多 252 间），区间会偏窄。

（§9 中的顺序/ID 消融见 §4.3-b，匹配消融见 §5.2-b，box 与场景正则消融见 §6.6-b，文本 SFT 对照见 §7.4-b。）

## 10. 实现交付与验收

### 10-a 交付模块 4：实验与评测包 — 未实现
- 规范：「实验与评测包（文本 SFT 对照、核心消融、分阶段指标）」
- 现状：文本 SFT 没有正式运行（`README.md:748, 769-773`）。核心消融中只排了顺序/ID（diag e）和输入消融 b–d（`ops/run_plan.sh:124-147`）。分阶段指标有代码，但没有资产库，资产阶段从未产出数字。
- 改为：评测包已实现全请求分母、参考指标、平凡基线、坍缩诊断和房间级 bootstrap 比较；文本 SFT 对照和大部分核心消融没有做，资产阶段的分阶段指标从未产出。

### 10-b 交付模块 5：部署与复现包 — 未实现
- 规范：「部署与复现包（Host 原子提交协议等）」
- 现状：Host 协议和 `AtomicMemoryHost` 已有（`runtime.py:252-297`），但没有真实的 Host 适配器（`README.md:715-716`）。复现方面有 `verify_bundle.py`、私有 HF 模型（`README.md:841`）和 `ops/isambard_*` 作业。
- 改为：复现部分（HF 私有模型、校验脚本、Isambard 作业）已有；Host 只有协议和内存参考实现，没有可部署的 WorldEdge 持久 Host 适配器。

### 10-c 交付范围 — 已过时
- 规范：「这些模块共同构成 FastFill v2 的完整交付范围。」
- 现状：实际交付物包括 RoomGenBench 交接（`README.md:25-32`）、三字段投影选模（`train.py:92`；`autorun.py:291`）、LLM 智能体对照（`README.md:786-849`）、spread 解码（`README.md:613-617`）和 VLM 评审渲染器（`ops/vlm_judge.py`），这些都不在规范里。
- 改为：实际交付范围已调整为：全条件训练的结构化模型 + 三字段投影选模 + RoomGenBench 交接 + LLM 智能体对照 + spread 解码 + 渲染评审工具；WorldEdge Host 闭环降为离线参考协议。

## 11. 文献与 V-DETR 实现依据

### 11.3 FastFill 回归 — 已过时
- 规范：「FastFill 的 room-normalized bottom-center 与 SmoothL1 log-size 借鉴相对尺度但不是复现」
- 现状：运行配置中 `loss.position_type` 和 `size_type` 都是 `'l1'`（config:47-48），模型用 `position_head 'grid_residual'`、grid 16（config:18-19；`losses.py:101-114`）。`docs/fastfill-v2-reference-code-audit-20261006.md:26` 仍写 SmoothL1，也已过时。
- 改为：FastFill 使用房间归一化坐标：XY 是 16×16 格分类加格内残差，z 单独回归；尺寸是 log 比值的 L1（不是 SmoothL1）。这只借鉴了 V-DETR 的相对尺度思想，不是其复现。

### 11.4 V-DETR 角度与 matcher 默认值 — 成立
- 规范：「GT residual 除以 π/K…官方 ScanNet 配置 angle bin 数为 1…parser matching 默认 GIoU 2、class 3、center 1、size 0.5、angle 0；训练 angle class/residual 0.1、0.5」
- 现状：`V-DETR/datasets/scannet.py:41` 为 `num_angle_bin = 1`；`V-DETR/criterion.py:387` 用 `np.pi / num_angle_bin` 归一化 residual；`V-DETR/main.py:118-124` 中 matcher 的 giou/cls/center/size/anglecls/anglereg 默认值为 2/3/1/0.5/0/0；`V-DETR/main.py:133-134` 中 angle cls/reg 的 loss 权重为 0.1/0.5；size 回归用 L1（`V-DETR/criterion.py:518`）。

### 11.6 方法定位 — 无法核实
- 规范：「方法定位表述（英文一段）」
- 现状：转录稿没有保留这段原文。`docs/fastfill-v2-review-20261005.md:205` 记录：V-DETR v1 的 PDF 是从官方来源获取的，不是用户提供的；K=12、四项权重、Qwen + 外部解码器的组合以及 WorldEdge 闭环都是本设计自己的选择，没有被所引论文验证。
- 改为：方法定位段应写明：V-DETR PDF 是本项目从官方获取的 v1 版本；Qwen 条件编码 + 外部对象解码器、K=12、各项权重和 WorldEdge 闭环是本方案的设计，没有被所引论文验证。

## 规范里没有、但当前已有的部分

1. **三字段投影（K5 契约）**：由 `batch.render_minimal_condition`（`batch.py:137-156`）和 `evaluate.project_minimal`（`evaluate.py:68-74`）实现，训练中 `minimal_form_p 0.5`。选模（`autorun.py:291`）、LLM 对照（`llm_baseline.py:189`）和渲染都用它。它去掉固定物体（含门窗）、约束和支撑声明，并跳过非矩形房间（test 中 4,165/8,620 间）。完整条件的测试数字只在 `projections.full` 中。
   - 规范应补充：约一半训练样本，以及全部选模和 LLM 对照，都只用房间类型、外包矩形尺寸和家具清单，丢掉门窗、固定物、约束和支撑。完整条件（含门窗）的效果只在测试报告的 projections.full 中，门窗与通行需要单独的全条件评测。
2. **spread 解码**：是默认的交付后处理（`evaluate.py:121-328`；`README.md:613-617`），详见 §4.1-b。
3. **K1 轴交换规则与按来源设定的 yaw 对称阶**：见 §6.4-b 和 §2.1。
4. **RoomGenBench 交接**：`predict --export-dir` 加 `fastfill/v2/roomgenbench.py`；`benchmark_request` 见 `roomgenbench.py:253-265`，support_parent 取自 place_id，不含门窗。`ops/run_checkpoint.sh:22` 用 `--method layout_boxes --require-placement` 运行。五个 SAGE 基准房间被留出到测试集（manifest 的 `roomgenbench_holdout_groups`）。
   - 规范应补充：导出 condition/layout/SceneSpec/GLB，五个 SAGE 基准房间留出到测试集，并做 require-placement 检查；请求带 place_id 支撑声明，但不含门窗。
5. **LLM 智能体对照**：外部 gpt-6.1-sol 的四种模式，在 300 条三字段验证行上比较（`README.md:786-849`），详见 §9-e。
6. **VLM 评审渲染**：`ops/vlm_judge.py:1-21` 提供 render/anchors/handoff/contact。它用 numpy z-buffer 画俯视平面图和两个角视图，锚点包括 room-centre、random、shuffled、yaw-flipped、mirrored、identical（`ops/vlm_judge.py:51`），本身不调用任何模型。评审本身还没运行。
   - 规范应补充：用匿名盒子渲染图和锚点做成对盲评，弥补单参考指标无法衡量场景合理性的问题。
7. **Isambard/autorun 运维**：`autorun.py`（A–F 阶段、score 选模、HF 上传、spread 与 argmax 两种测试）；`ops/isambard_setup|autorun|analysis.sbatch`；`ops/isambard_submit.sh`（两组：yaw_cls 0.5 与对照组 0.08）；`ops/run_plan.sh`（运行 stratify、compare、llm_compare 和 report）。
   - 规范应补充：服务器 autopilot 负责选模与发布；Isambard 上跑 yaw_cls 0.5 与 0.08 两组对照训练和分析作业（分层统计、房间 bootstrap 比较、HTML 报告）。
8. **验证集过滤**：选模用的验证行先剔除带 `oob_objects`、`fixed_collision`、`overlapping_furniture` 标记的行（config:84-88）。规范没有提到这一步。

## 建议下一版规范怎么改（按影响排序）

1. **把门窗放进被评测的条件。** 规定门窗的唯一表示：要么用 `room.openings`，要么写明 `fixed_objects` 中 category 为 door/window 的盒子，并加上已知/未知标记。选模、主表和 LLM 对照中至少有一条使用含固定物和门窗的完整条件投影；三字段投影降为附加结果，并且必须写出跳过的房间数。增加门前净空和通行指标（§1.1、§2.4、§8、§9）。
2. **加入布局质量指标和 VLM 评审。** 把 `autorun.score` 的公式写进规范，并承认它是单参考误差加分布偏差的组合。§9 增加“场景合理性”一层：用 VLM 成对盲评加锚点。平凡基线必须用训练集拟合（`--baseline-fit`）。“FastFill 不如 LLM harness 协调”在量化之前只能写作目视观察（§3.4、§7.3、§9）。
3. **改写模型与损失章节。** 位置头写成 grid_residual（16×16 CE + 格内 residual + z 回归），regression 改为消融。匹配成本写成 grid 位置 loss 加 swap-min 尺寸成本。ρ 改为 L1。spread 解码写进交付链，并要求 raw（argmax）与 spread 结果分开报告（§4.1、§4.4、§5.2、§6.1、§6.2、§11.3）。
4. **改写对称与朝向规则。** 删除或改写 §6.4 的加粗句：承认 K1 盒等价交换，限定适用的来源和类别，并要求评测同时报告盒等价误差和 `*_plain_convention` 误差。写明当前 93.7% 的对象只学 yaw mod π，语义朝向由 spread 的背向墙启发式给出；如果需要语义前向，必须单列数据来源（§2.1、§6.4）。
5. **收窄资产闭环的范围。** 把 §8 和 §10 的“主系统整体交付”改为“离线参考协议”，把 RoomGenBench 交接写为当前实际下游。Harness、持久 Host、Solver、mesh/physics 检查和重规划列为后续集成。资产不重叠子集和真实尺寸子集要么建立，要么从交付中删除（§3.3、§3.4、§8、§10）。
6. **说清碰撞的处理方式。** 写明场景正则权重为 0，碰撞只靠 spread 解决；如果保留“防缩小”的要求，就要给出具体机制和消融计划（§6.6）。
7. **统计与对照。** 至少两个种子，置信区间按房屋聚类做 bootstrap。文本 SFT 要么正式做（先修复 `text_sft.py` 遇到不完整行就报错的问题，并对齐增强和投影），要么从对照列表中删除。ATISS/DiffuScene 和“类别先验 + 布局模型”也一样，要么排期，要么删除（§7.4、§9）。
8. **更新数据章节的细节。** 增强列表改为实际的 6 项（见 §3.4-c）；描述中的米制尺寸要过滤或单列；倾斜对象要显式标记，而不只是屏蔽监督；记录训练中被跳过的匹配组数和平局（§1.3、§3.3、§3.4、§5.3）。
