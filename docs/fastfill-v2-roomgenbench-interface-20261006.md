# FastFill → RoomGenBench 接口核对

审核日期：2026-10-06。本次检查本地 `RoomGenBench/` 的输入、方法调用、装配和评分源码，并在 FastFill 中新增独立动态接入口及回归；未修改下游仓库。已用合成 GLB 跑真实 trimesh 装配；未运行真实资产生成模型、Blender、physics／Solver 或 Host，也未修改训练数据或 checkpoint。RoomGenBench 独立 Git HEAD 为 `30f2e059d046ccd4aa928d0109a09a52f11693b5`，审核时 tracked working tree 无修改。图服务以 `persistence=False` 建索引后，用图搜索和源码片段检查；没有写入仓库图文件。

**任务衔接成立：FastFill 的预测 target bbox＋文本可以成为 RoomGenBench 的输入。** RoomGenBench 消费已有布局并生成／检索 mesh，不决定家具类别、数量或摆法。[prepare_inputs.py](../RoomGenBench/bench/prepare_inputs.py) 的 `main`（32 行）复制父场景的对象清单，[assemble.py](../RoomGenBench/bench/assemble.py) 的 `assemble`（257 行）按这些对象逐项装配。重复资产可以共用 mesh，实例数量仍由输入布局确定。

用户所说“把 OptiScene 输入 bbox 移到输出，借鉴 V-DETR”描述了任务与结构监督方向：未来局部尺寸成为 target，连续 heads 学习 size／bottom-center／yaw，匹配仅在合法组内进行。当前 [model.py](../fastfill/v2/model.py) 是 condition LM＋请求 slots＋双向 object decoder，不是直接复用点云检测器；[text_sft.py](../fastfill/v2/text_sft.py) 是独立 token CE 对照。这里核对接口，不以这个类比证明方法质量。

主训练继续使用原选定 16 个家族体系：11 个训练房间家族展开为 16 个训练 source tags，两个家族仅评测，三个辅助家族。RoomGenBench 的五个固定示例是展示／mesh 比较集合，**不能据此把训练语料缩成五个房间或单来源**。主数据保留完整条件和 partial masks，资格处理见[训练流程](fastfill-v2-training-pipeline.md)。

## 下游实际合同

| 阶段 | 必需信息／约定 | 源码位置 |
|---|---|---|
| 每唯一资产输入 `assets.jsonl` | `asset_key`、`type`、`description`、米制完整局部 `dimensions={width,length,height}`、`place=floor/wall/on_object`、`scenes`、`n_instances`。key 是生成文件 stem，不要求已有真实 mesh ID。原准备脚本按 type＋description 去重，以首条实例的尺寸作为生成输入；2026-10-06 起 `direct_layout.asset_key` 采用同一约定 `slug(type)[:24]_sha1(description)[:8]`，同 type＋description 实例共用一个 key。 | [CONTRACT.md](../RoomGenBench/bench/CONTRACT.md) 20 行；[prepare_inputs.py](../RoomGenBench/bench/prepare_inputs.py) 23、51 行 |
| 每场景输入 | `scene_key`、`room_type`、`room`、`objects`；每对象含稳定 `id`、`type`、`description`、`asset_key`、`position={x,y,z}`、`rotation={x,y,z}`、`dimensions`、`place_id`（FastFill 导出另带 `place` 与 `support_status`，见下文第 2 点）。位置是 Z-up 底面中心；旋转是 degree Euler XYZ，矩阵为 Rz·Ry·Rx；SAGE 局部 front 为 +Y。 | [prepare_inputs.py](../RoomGenBench/bench/prepare_inputs.py) 39 行；[assemble.py](../RoomGenBench/bench/assemble.py) 50、58 行 |
| 房间 shell | dimensions、position、walls、doors；门引用 wall ID、沿墙位置、宽高。实际 assembler 按 width／length 生成 Z=0 的矩形 floor；它不读取 FastFill 的 floor／boundary known flags。 | [assemble.py](../RoomGenBench/bench/assemble.py) 83 行 |
| 方法资产输出 | `results/<method>/assets/<asset_key>.glb` 与同 stem JSON；GLB 为 Y-up、front +Z，scale／offset 可任意；sidecar 含 `asset_key`、`method`、`status=ok/fallback/failed`、`prompt`、`seconds` 等。 | [CONTRACT.md](../RoomGenBench/bench/CONTRACT.md) 39 行 |
| 统一装配 | 展开 GLB node transforms，转到 SAGE frame，yaw snap／必要时 tip，非等比缩放至 target box，再底面中心归零、按布局旋转和位移。输出 GLB 世界坐标为 `(x,z,-y)`。失败资产保留实例并显示 placeholder。 | [assemble.py](../RoomGenBench/bench/assemble.py) 133、177、257 行 |

## FastFill 已有转换与实测

[direct_layout.py](../fastfill/v2/direct_layout.py) 的 `layout_to_roomgenbench`（107 行）保留请求 ID、category／description、数量和底面中心，并作如下转换：

| FastFill 预测 | RoomGenBench 输入 |
|---|---|
| canonical 局部 size `[sx,sy,sz]`，局部 +X 朝向 | `width=sy, length=sx, height=sz`，局部 +Y 朝向 |
| `yaw_rad` | `rotation={x:0,y:0,z:degrees(wrap(yaw−π/2))}` |
| Z-up `bottom_center_m` | `position={x,y,z}`，米制底面中心不变 |
| requested ID | 同一 `id`；生成 registry key 独立标为 `downstream_generation_key_only`，不是实际 catalog ref |

`predict --condition ... --export-dir ...` 可以导出完整主条件的预测框视图；简化输入消融使用 `--request ... --room-size-semantics reference_extent`。第二轮起，用 `minimal_form_p` 训练的主模型也可直接用默认 `--request`（rectangular，只给房型、房间尺寸和物品清单）推理：`request_to_condition` 调用 `batch.render_minimal_condition`，渲染出的条件文本与训练时同房型、同尺寸、同清单的三字段投影逐字节相同；此时没有地面支撑声明，`place` 全部来自几何推断或为 unknown。[export_handoff](../fastfill/v2/direct_layout.py)（194 行）写出 `condition.json`、`layout.json`、`scene.json`、`roomgenbench_scene.json`、`assets.jsonl`、`diagnostics.json`、`preview.svg`、`layout.glb`。它保存完整 canonical 条件；`roomgenbench` 接入口使用独立路径装配任意 handoff，不注册或修改共享 bench 输入，也不运行生成模型。

以下是在仓库根目录可重复执行的只读内存检查；不创建输出文件，不加载训练权重：

```bash
python -B - <<'PY'
from pathlib import Path
import importlib.util
import json
import numpy as np
from fastfill.v2.direct_layout import request_to_condition, layout_to_roomgenbench

spec = importlib.util.spec_from_file_location("rgb_readonly", Path("RoomGenBench/bench/assemble.py"))
rgb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rgb)
condition = request_to_condition({"room_type": "unknown", "room_size_m": [5, 4],
    "furniture_list": ["chair"]}, room_size_semantics="reference_extent")
layout = {"schema_version": "fastfill.v2", "objects": [{"id": "obj_0000",
    "target_size_local_m": [1.4, .7, 1.1], "bottom_center_m": [2, 1.5, .2], "yaw_rad": .7}]}
scene = layout_to_roomgenbench(condition, layout)
obj = scene["objects"][0]
w, d, h = (obj["dimensions"][k] for k in ("width", "length", "height"))
points = np.array([[x, y, z] for x in (-w/2, w/2) for y in (-d/2, d/2) for z in (0, h)])
actual = (rgb.place_matrix(obj) @ np.column_stack((points, np.ones(8))).T).T[:, :3]
sx, sy, sz = layout["objects"][0]["target_size_local_m"]
raw = np.array([[x, y, z] for x in (-sx/2, sx/2) for y in (-sy/2, sy/2) for z in (0, sz)])
a = .7
rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
expected = raw @ rotation.T + [2, 1.5, .2]
corners = lambda values: sorted(tuple(np.round(v, 9)) for v in values)
shell, _ = rgb.build_shell(scene["room"])
print(json.dumps({"corner_sets_match": corners(actual) == corners(expected),
    "rotation_z_deg": obj["rotation"]["z"], "floor_known": scene["room"]["floor_known"],
    "shell_names": [name for name, _ in shell], "height": scene["room"]["dimensions"]["height"],
    "missing_site_gt_fields": [k for k in ("building_style", "sage_scene") if k not in scene]}))
PY
```

本轮实际结果：按 10⁻⁹ m 取整的世界 corner 集合相同；下游 z rotation 为 `−49.892954340842365°`。同时 `floor_known=false` 仍生成 `shell_floor`；height 保持 null；缺少 building_style／sage_scene。这个检查证明该 fixture 的坐标转换和消费行为，不证明模型质量或所有 mesh 的语义正面。

## 原下游边界与本轮适配

1. **动态输入与执行入口**：assembler 固定五个 SCENES，从 `bench/inputs/scenes/<key>.json` 读取，CLI 的 `--scenes` 接受 key 而非任意 JSON path；各方法读取固定 `bench/inputs/assets.jsonl`。[assemble.py](../RoomGenBench/bench/assemble.py) 36、334 行。本轮 `fastfill.v2.roomgenbench` 已提供任意 handoff、独立 output root、GLB／sidecar 消费入口；生成方法的权重、外部服务及 runner 仍需另配。
2. **可靠 placement／support**：旧 bridge 所有 `place_id=None`、registry `place=unknown`；本轮已转换明确 `support_parent` 与 hard-on 条件，保留 surface ID 和支撑状态。Holodeck 的 `THRESHOLD[place]` 只接受 floor／wall／on_object，[retrieve.py](../RoomGenBench/methods/holodeck_retrieval/retrieve.py) 49、99 行。2026-10-07 第二轮起（`direct_layout.infer_support`），每个场景对象都带 `place`（floor／wall／on_object／unknown）和 `support_status`（declared／inferred／unknown）：请求声明了 `support_parent` 或 hard on 时按声明给出，状态 `declared`；否则底面距已知地面 ≤ 0.02 m 时为 `floor`（`floor_known=false` 的房间，例如 `reference_extent` 请求，不做地面推断），否则在底部严格更低的其它预测框中，取顶面距本物体底面 ≤ 0.03 m 且旋转后的投影包含本物体中心的最高一个，`place="on_object"`、`place_id` 为该框 ID，这两种状态都是 `inferred`；都不满足时 `place="unknown"`、`support_status="unknown"`。请求没有声明就绝不输出 wall，也不按类别杜撰。`assets.jsonl` 的 `place` 与之相同，装配 receipt 带 `place`、`support_parent`、`support_status`。第二轮之前导出的 handoff 不再通过 `roomgenbench._handoff` 的场景一致性检查（新增 `place`，旧状态名 `inferred_floor_contact` 改为 `inferred`），需用 `predict --export-dir` 重新导出。`room.walls` 由地面多边形各边生成（厚 0.1 m，高 = 房高或默认 2.7），`room.doors/windows` 来自类别含 door/window 的 fixed_objects（最近墙、`position_on_wall∈[0,1]`、沿墙投影宽度、高度；窗另有 `sill_height`）。动态入口的 `--require-placement` 明确拒绝未知 placement；完整 condition、fixed 与 relations 保留于 sidecar；`fastfill.v2.roomgenbench` 调用参考 `build_shell`（门属于 shell，其余 fixed 仍为 bbox proxy），receipt `walls` 为参考 walls_meta，`room.shell_geometry_kind="reference_build_shell_polygon_walls"`。参考 `build_shell` 以包围盒中心判定墙外法向，非凸房间会把部分墙挤进室内；`fastfill.v2.roomgenbench._shell_parts` 逐墙调用参考几何，但外法向改由地面多边形有向面积决定（receipt `room.wall_normal_source="floor_polygon_winding"`）。
3. **未知房间信息**：`height=None` 在 [render.py](../RoomGenBench/bench/render.py) 70 行被读作 H，随后 `H+1.5` 无法执行。本轮动态入口将未知高度显式记为 display_reference，原 height 仍为 null，避免该旧 renderer 分支；原 Blender renderer 未改动。assembler 不执行 known flags，生成的矩形 shell 仅是 proxy，不是原房间资格认证。
4. **默认 site／GT 分支**：[build_site.py](../RoomGenBench/bench/build_site.py) 132 行要求 building_style；`sage_gt` 另需 sage_scene 与 source mesh。新请求没有 SAGE GT 时不能编造对应字段或继续报告对 SAGE 的 shape 指标。通用 generated-method／layout_boxes 路径不要求先有 GT asset ID。

RoomGenBench 的目标是统一 **mesh-fit benchmark**：[fit_asset](../RoomGenBench/bench/assemble.py) 对 mesh 非等比缩放、yaw snap 和 tip；FastFill 主 actual-asset runtime 则保留 target／actual 区分、不得缩小或删物体来解决约束。这两种结果必须单列。RoomGenBench 可成为 mesh 生成与展示端，不能把 fit、placeholder 或渲染成功当成真实资产、物理稳定与原子 Host 通过。当前 FastFill 的 mesh／physics／Solver 检查始终 unknown，具体 checkers 和持久化 WorldEdge Host 尚未接入。

## 当前动态命令

先用 `predict --condition ... --export-dir ...` 生成 canonical handoff。以下入口直接接受该目录，无需占用原 benchmark 的五个固定场景：

```bash
pip install -r fastfill/v2/requirements-roomgenbench.txt

python -m fastfill.v2.roomgenbench \
  --handoff outputs/my-handoff \
  --output-dir outputs/my-roomgen-boxes \
  --roomgenbench-root RoomGenBench

# 生成器另行读取 assets.jsonl，输出每 asset_key 的 GLB 与 JSON sidecar。
python -m fastfill.v2.roomgenbench \
  --handoff outputs/my-handoff \
  --output-dir outputs/my-roomgen-meshes \
  --method my_generator --assets-dir /path/to/generated/assets \
  --roomgenbench-root RoomGenBench --require-placement
```

输出 GLB、同名场景 JSON 和 receipt.json；目录必须不存在，不能与输入重叠。`layout_boxes` 成功只表示代理装配完成。生成模式全部实例为 `ok` 且 GLB 成功读取／拟合时 exit 0；missing／failed／fallback 仍保留实例并记录状态，exit 2。`generated_mesh_success` 不表示实际几何／功能／物理／Host 通过。receipt 保存 GLB／sidecar hash、原生 glTF／SAGE 尺寸、拟合后尺寸、拟合日志及明确的 anisotropic fit policy。上游薄轴 clamp 导致未拟合到目标范围时记为 fit_mismatch，不计入 generated success。

该入口复用固定下游的 load_glb_parts、fit_asset、place_matrix，通过隔离 import 读取 helpers；不改共享全局输入路径或源码。测试覆盖任意场景、+X/+Y 角点等价、非零 floor、未知 height、声明支撑、fixed/constraints、缺失或失败 sidecar、placeholder、路径重叠和全实例分母。合成 box GLB 仅用于接口回归，不充当真实生成资产成果。

## 评分分母

[render_eval.py](../RoomGenBench/bench/render_eval.py) 103 行为失败资产渲染 placeholder；[score_aggregate.py](../RoomGenBench/bench/score_aggregate.py) 50 行的全资产描述／类别等均值包含它们，coverage 与 generated-only 诊断另报。稳定 requested IDs／实例不能因生成失败消失。

[score_geom.py](../RoomGenBench/bench/score_geom.py) 110 行在 child 或 supporter mesh 缺失时跳过该 support pair；aggregate 的 support `n` 是实际可评 pairs，**不是全部请求支撑对分母**。接入 FastFill 时应另报告所有请求／支撑对总数、缺失资产数、可评数与成功率，保留失败。shared bench 没有 scene collision／用户关系 constraints／atomic Host 验收；其 VQA、shape-to-SAGE 和 support rays 不能替代主目标框指标与实际资产验收。

## 审核源文件 SHA256

以下下游文件是审核固定点；FastFill 两个 adapter hash 是本轮实现前的读取点，当前发布 hash 以新证据 manifest 为准。不是宣称整个下游仓库或真实模型／资产缓存均已验收。

| 文件 | SHA256 |
|---|---|
| `RoomGenBench/README.md` | `4cfc4695484c075a1188cd3a33100db1c5e486bb734cd34f2cdd520f5df18660` |
| `RoomGenBench/bench/CONTRACT.md` | `9ec2351cb8538e138b977f8f8e8f30a0492e60473318c0362e79d4f2c0577a28` |
| `RoomGenBench/bench/prepare_inputs.py` | `fa4b55c877e6d17ca3e07fcf298e03b969ded9249c70d632192e41fabc071380` |
| `RoomGenBench/bench/assemble.py` | `3bb3b1a280c7074d0902a7ca0210ed433f5fc386da1bac735dc37f076763035f` |
| `RoomGenBench/bench/render.py` | `70d526b7d3a04d90510f6dcf259d6976c5b87a015a7e30a1bcedb6bd76fc26f5` |
| `RoomGenBench/bench/build_site.py` | `fe465d6391d717ebfab8b1e2a763166c7289ad760e629b0c603605529375acc4` |
| `RoomGenBench/bench/render_eval.py` | `d141fc78d8235011e68cd363b036d587f1f1bf460b03f28ca6ae016504fc5b02` |
| `RoomGenBench/bench/score_geom.py` | `ddef22d516019caa877efff77b4516906483122f20268f955a2912acc7dfdd16` |
| `RoomGenBench/bench/score_aggregate.py` | `afac424e2e12eea64f59d4a89476b03ec9a1b07e59ef8528ed31eb8a554d256d` |
| `RoomGenBench/methods/holodeck_retrieval/retrieve.py` | `afda408738449a2f9f5e9240f99b6eaf7d2d2456c8d52f49fcf960dee1f062ad` |
| `fastfill/v2/direct_layout.py` | `05217abfbc0627827cfac0f77ad1f40886eac51241af809537b7cfe651ab3808` |
| `fastfill/v2/predict.py` | `36aa199e649f37badccb55d1ffe91cc967fe5b98b59c145338d372c47d2a9e16` |
