# FastFill v2：简化输入消融与 RoomGenBench bbox 交付

2026-10-06。本文记录 **简化输入消融及历史 SpatialLM pilot**：room type + room size + furniture list → 局部尺寸、底面中心和 yaw → bbox 下游输入。当前主任务保留房间、固定对象、支撑与空间约束，见 [主流水线](fastfill-v2-training-pipeline.md)。最终下游 RoomGenBench 按 text+bbox 生成／装配 mesh；仅导出 bbox 不能宣称实际 mesh 场景已验收。

当前多源简化视图为 `multisource-20261006/data-minimal-reference`，它使用显式 `reference_extent`，保留未知物理边界／地板及部分标签。以下 9,601 / 539 / 624 数量属于历史 `direct-bbox-20261006` 严格矩形 pilot，不是当前多源训练集。

## 1. 上游只提供三个字段

```json
{
  "room_type": "living room",
  "room_size_m": [5.0, 4.0],
  "furniture_list": ["sofa", "coffee table", {"category": "chair", "count": 2}]
}
```

`room_size_m` 是矩形房间 X/Y 全长，单位米；可显式传第三维真实房高，默认 XY 任务不要求房高。原点为房间左下 XY 角，名义地板为 z=0。缺房高保持 unknown，归一化竖直尺度使用协议固定 3 米，不读取目标高度。列表支持字符串或带 category/description/count 的对象，展开后每个实例恰好一个稳定 ID；显式 ID 只能用于 count=1。超过对象预算时拒绝整个请求，不截断。

不从家具类别自动生成支撑、门窗、墙体、面向或交互能力证据。若上游只有这些字段，就不能声称模型条件包含真实开口或支撑约束。

## 2. RoomGenBench 实际需要什么

本地只读核对了 `RoomGenBench/bench/CONTRACT.md`、`prepare_inputs.py`、`assemble.py`、`render.py` 和 `build_site.py`。它的任务是 **text + 已给定 bbox → 对象 mesh → 房间 mesh**，不是学习 bbox 布局。截图的 `REFERENCE / Input layout` 来自 `layout_boxes`：直接画给定框，不检索资产、不读取 SAGE mesh。`sage_gt` 是另一个使用数据集 mesh 的参考。

| 输入 | 下游用途 | 本实现输出 |
|---|---|---|
| 房型、房间尺寸/位置 | 场景范围、地板与相机 | room_type、room.dimensions/position；未知 H 保持 null |
| 每个对象 ID、type、description | 实例对应及 mesh 生成提示 | 继承请求 ID/类别/描述 |
| 局部全尺寸 | 给生成 mesh 的目标 bbox | dimensions.width/length/height，单位米 |
| 底面中心与旋转 | 放置生成 mesh | position XYZ，rotation XYZ 角度制 |
| asset_key 与注册表 | 生成结果 GLB 的文件名及复用索引 | assets.jsonl；key = `slug(type)[:24]_sha1(description)[:8]`（benchmark 约定），同 type＋description 共用，取首实例尺寸；不是已检索资产 |

FastFill 是右手 Z-up、局部 X/Y/Z 全尺寸、底面中心、弧度 yaw。SAGE/RoomGenBench 的局部约定为 width=X、length=Y、规范轴向 +Y；导出采用 **width=d、length=w、yaw_deg=degrees(wrap(yaw−π/2))**。这同时保留世界 OBB 角点和规范轴映射。当前几何 yaw 标签不认证椅背/屏幕的语义前向，不能把这一轴转换解释为语义朝向已经解决。GLB 世界坐标转为标准 Y-up：`(x,y,z) → (x,z,-y)`。

**兼容范围有边界。** 导出的 SceneSpec 已实际通过原 `layout_boxes` 装配函数：XY 请求、H=null，三个各向异性旋转框的 GLB 世界角点误差最大 2.74e-8 米，未调用任何资产加载，源文件 hash 未变。原 benchmark CLI 和 site 的场景列表固定为五个房间，需要另行注册新 scene_key。原 site 还读取 building_style，原 renderer 对 null 房高没有处理；XY 请求可直接使用本实现的 GLB/SVG，不能声称原 site/render 已原样兼容。缺声明支撑时，底面距已知地面 ≤ 0.02 m 的物体记 `place_id="floor"`（`support_status=inferred_floor_contact`），否则 place_id=null，原 assembler 会将其着色/标记为 on_object；这不是支撑证据。本实现预览按类别着色并明确 unknown，不伪造 wall 支撑以获得同色效果。RoomGenBench checkout 保持不变。

## 3. 一次推理与交付

从代码仓库根目录运行，MODEL 必须是针对当前最小条件训练的模型；历史 rich-condition pilot 不是这个任务的质量证据。每次输出选不存在的新路径。

```bash
python -m fastfill.v2.predict \
  --checkpoint "$FASTFILL_MODEL" \
  --request fastfill/v2/configs/direct_request.json \
  --output outputs/bbox-prediction-new.json \
  --export-dir outputs/bbox-handoff-new \
  --device cuda --max-length 4096
```

生成原始预测 JSON，以及下游目录中的六个文件：

| 文件 | 含义 |
|---|---|
| scene.json | FastFill bbox 场景；保留局部 size、bottom-center、yaw，补 geometric center 和八角点 |
| roomgenbench_scene.json + assets.jsonl | 下游 SceneSpec 与生成索引；没有资产检索结果 |
| layout.glb | 带每个对象 ID 的彩色盒子＋房间范围；无家具 mesh/贴图 |
| preview.svg | 可直接打开的等距 bbox 预览 |
| diagnostics.json | 目标 OBB 边界/地板/已知天花板/重叠诊断；不修正原预测 |

没有 `--catalog`，不调用 Asset Resolver 或 Host。proxy 重叠不能自动判定支撑接触是否合法；此报告不认证 mesh、支撑、物理或任务成功。无效布局仍保留原始框，不缩小、不删除、不移高来美化输出。

## 4. 训练数据如何变化

原来筛选的 16 个家族和 review3 谱系保留。历史 `direct-bbox-20261006` builder **先限定 SpatialLM 来源**，再筛选房型／地板／矩形边界、完整局部框和 upright 几何 yaw，将 condition 投影为公开 request adapter。因此它不能证明其他来源不合格。当前多源主集不提升缺少证据的语义 yaw。

| split | 场景 | 对象 |
|---|---:|---:|
| train | 9,601 | 30,049 |
| validation | 539 | 1,642 |
| test | 624 | 1,854 |
| 合计 | 10,764 | 33,545 |

源 IR hash 固定；每个源 ID、local size、bottom-center、wrapped yaw 与父数据逐项一致。继承原 UID/房屋组/split，不重新随机划分；actual 派生行的原点平移全为零，target 数值未改变。XY 模式移除输入房高，真实源房高仅在 provenance/资格检查中保存，模型不读取它。所有 OBB 角点须在 XY 房界内，底面不低于地板、顶部不超已知源房高，容差 1 mm；没有认证碰撞/支撑/mesh/physics。

yaw 表示矩形局部轴，按 **π 周期（yaw_symmetry_order=2）**监督/评测；不认证语义前向。当前派生版使用固定请求对应，未重建 exchangeable groups；不能宣称已启用组内 Hungarian。匿名重复对象的固定顺序可能限制集合生成，应作为后续合法交换组消融。

最大对象数为 **26**，平均 3.12 个；只有 135 个场景达到 10 个对象。截图 32/51/53/55/122 对象的密集房间未被这批合格训练数据覆盖；128 slots 的架构预算不能作为密集布局质量证据。房型主要是住宅卧室/卫生间等，restaurant 没有当前合格样本。来源中的 dining table combination 等组合类别作为一个请求实例保留，不擅自拆成独立桌椅。当前描述只保留 category，丰富描述泛化也需独立评测。

原始 harddisk 数据未改。外盘未挂载时，本地最终生成在 `outputs/fastfill_v2/direct-bbox-20261006/data-xy-final`，没有在 `/Volumes/harddisk` 创建假挂载。服务器/私有 HF 版本、下载与 hash 检查见 [数据发布手册](fastfill-v2-dataset-release.md)。

## 5. 训练流程和产物

Qwen3-8B 只编码最小 condition → request-bound slots → 双向对象 decoder + cross-attention → positive local size、normalized XYZ bottom-center、12 yaw-bin logits/residual。直接用四项可微几何监督训练；文本 SFT 是独立 baseline，无需先训练文本模型，无默认 DPO，不需要 vLLM 或 MinkowskiEngine。

每步 loss 使用有效原张量，监督尺寸是 log-ratio SmoothL1；对两个等价 yaw 候选选择同一候选的分类＋residual 联合代价。基础配置 box/collision/boundary 权重为零；边界资格不保证模型预测也在界内。部署产物为 LoRA/骨干配置、decoder/geometry heads、tokenizer、日志和数据 hash。它们组成 bbox 布局预测器，不会自动产生家具 mesh 或贴图。

详细优化步骤与历史 pilot 边界见 [pipeline](fastfill-v2-training-pipeline.md)。先对当前 XY 数据做真实 Qwen tokenizer/object 预算预检和小样本 pilot，再冻结正式训练曝光、对照和密集场景测试；历史两个 MultiScan 场景的 20 步结果不得当作当前新输入模型已经训练完成。
