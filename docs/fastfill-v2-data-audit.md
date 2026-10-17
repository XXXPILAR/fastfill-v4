# FastFill v2 raw-source inventory audit — 2026-10-05

Historical raw-source inventory. Current full-condition multi-source derivatives,
qualification changes and server evidence are documented in
[the completed multi-source record](fastfill-v2-multisource-20261006.md).
The audit-only decisions below belong to the initial raw adapter, not a claim
that the current main uses only MultiScan.

This document records the initial **raw-source inventory**, not a replacement
for OUR retained dataset selection. The primary implementation now migrates the
frozen `.release/v3.2` corpus through `fastfill.v2.data --source selected-v3.2`.
The retained 16 dataset families expand to 18 IR sources (16 train source tags
and SceneSmith/SpatialGen held out). See
[the selected-corpus review](fastfill-v2-review-20261005.md) and
[the implementation guide](../fastfill/v2/README.md) for migration and results.
The 208 scans below describe a separate initial raw MultiScan smoke corpus; they
are **not** the population selected by the existing v3.2 pipeline.

In that initial implementation, the first full geometry source was **MultiScan**, with
strict upright/frame checks. Its source room footprint and floor contact are
incomplete; neither becomes a verified room-boundary or physical-support label.
The remaining sources are inventoried and remain **audit-only** in the initial
v2 builder. That means their adapters are not yet eligible; it does not mean
their geometry could never be usable.

All reads used `/Volumes/harddisk/3D_Room_Collections`. No source directories,
files, assets, names, or metadata were changed. Existing v1/v3/v3.1/v3.2 outputs
are not inputs to the v2 builder and remain separate.

## Evidence and scope

`python -m fastfill.v2.audit` records the local README/converter hashes, a real
record's field keys, and the presence of all sources. It bounds large JSON
sampling to the first record and 4 MiB, and counts CSV rows directly. The
catalog's `declared_counts` are source-documentation totals, not a new numerical
recount of every JSON corpus. Inspecting a README or converter does not establish
every asset's semantic front or prove mesh correctness.

The current actual inventory contains all five requested top-level directories:

| Directory | Actual nested datasets |
|---|---|
| `BillLin66__3D_Room_Collections` | 3D-FRONT, SpatialGen |
| `imChuling__3D_Room_Collections` | 3RScan, ARKitScenes, IL3D, InteriorGS, InternScenes, MansionWorld, OptiScene, SceneCAD/Scan2CAD, SceneSmith, SpatialLM, Structured3D |
| `liantian__3D_Room_Scene_Collections` | SAGE10k |
| `XXXpilar__hssd_clean` | HSSD rigid objects, articulated metadata, regions, asset catalog and GLB/URDF geometry |
| `XXXpilar__multiscan-clean` | MultiScan objects, regions, articulation, relations, PLY/OBJ meshes |

The machine report stores separate fields for room geometry/type, category,
object/asset identity, bbox semantics, full/half extent, units, position reference,
rotation, handedness/up-axis, scale, support, relations, meshes, eligibility, and
blockers for all **16** nested dataset families. IL3D and InternScenes also record
their source subsets. Repackaged single-room/multi-room views are not additional
independent data.

## Source geometry decisions

| Dataset | Size and position evidence | Rotation/frame and scale | Current decision |
|---|---|---|---|
| BillLin66 3D-FRONT | Asset size/bbox × placement scale; position is asset origin, not proven bottom center. First real row has missing size/category. | Source Y-up converts `(x,-z,y)` to right-handed Z-up; raw house quaternion available; scale already applied. | Audit-only: recover mesh-local bbox center/minimum and semantic front before full supervision. |
| SpatialGen | Scaled local full XYZ extents; transform translation is bbox center; floor/ceiling are millimeters, boxes meters. | Full transform retains tilt and mirrors; extent extraction applies column norms once. | Official **test-only**; no train split. Reject significant tilt/mirror and invalid floor polygon; canonical semantic front still needs evidence. |
| 3RScan | `obb.axesLengths` full lengths along raw `normalizedAxes`; position is `centroid`. | Export yaw null; preserve raw box axes, not world AABB. | Audit-only: scan is not a true room; no independent room boundary or semantic front. Attributes/affordances and 3DSSG relations exist, mesh absent. |
| ARKitScenes | `obbAligned.axesLengths` are full local lengths. Export position is world AABB bottom; use raw OBB `centroid` to recover actual OBB geometry. | Raw `normalizedAxes.T`, metric Z-up; significant tilt exists. Raw unaligned `obb` is a different field/frame. | Audit-only: room/front unavailable; `custom.abandon` is a placeholder asset ID. Group by `visit_id`, accounting for the documented cross-split visit. |
| IL3D 3D-FRONT/HSSD/synthetic | Layout bbox maps X/Z/Y; declared bottom origin and retained asset metadata must be verified against asset bounds. | Source USD `rotateXYZ`, full Z-up rotation matrix retained; do not guess bbox/scale interaction. | Audit-only: pivot/frame/front and source anomalies unresolved. Do not inherit v1's oblique-window world-span bug. |
| InteriorGS | Eight source corners; exported long/short horizontal sizes and bbox bottom. | Export yaw is geometric long-axis orientation modulo π, with no semantic front. Source meter/right-handed Z-up. | Audit-only: geometry-only orientation must not become furniture semantic front; preserve zero extents/ambiguous room assignment as invalid. |
| InternScenes Real2Sim/Gen | Source `bbox[3:6]` full local lengths; `bbox[0:3]` is center. | Angles radians in raw ZXY order; source right-handed Z-up. | Audit-only: semantic front remains unverified. v1's upright geometric fallback is not semantic-front evidence. Gen/Real2Sim containers and floor hull quality differ. |
| MansionWorld | Floor/wall annotation bounding boxes, bottom heights; surface groups reference parents. | AI2-THOR Y-up maps `(x,-z,y)`; metadata geometry and per-asset pivot/front require source audit. | Audit-only: metadata-only assets; actual support surfaces cannot be obtained from bbox tops. Exclude position-rich room/asset text from condition. |
| OptiScene / 3D-SynthPlace | Claimed `[height,width,depth]` full bbox in centimeters; position is source anchor with unresolved pivot. | Export `(x,z,y)` changes handedness; source horizontal rotation and full Euler convention need reconciliation. | Audit-only: known invalid/NaN geometry and mesh checks required. Do not use existing prompts' actual bbox as main-v2 condition. 17,528 source scenes ≠ 17,480 text rows. |
| SceneCAD / Scan2CAD | ScanNet furniture is **world AABB**; separate CAD `bbox` is local **half** extent × TRS scale; CAD position is geometric center. | Full raw CAD/scan/axisAlignment transforms retained, ShapeNet Y-up. | Audit-only: original tilt cannot silently become yaw; do not inherit v1 tilt bug. Scan shell is not a partitioned room or room-height label. |
| SceneSmith | Visual glTF/SDF full mesh extent; DMD translations are mixed asset origins. | AngleAxis and full raw pose available; visual scales/node transforms already applied. | Audit-only: keep asset bbox offset/minimum; documented 46 centered-vs-bottom wall-asset errors forbid blanket center subtraction. Semantic front needs source evidence. |
| SpatialLM | Source Bbox local full scale XYZ; geometric center; yaw-only format. | Angle radians in raw text; already metric final extents. | Audit-only: front currently empirically calibrated; invalid wall rings and cross-export duplicate houses require common source split. |
| Structured3D | Raw `coeffs` are **half extents in mm** along raw basis rows; centroid and room origin kept. | Do not reuse export's heuristic size-axis reorder/yaw as model pose. | Audit-only: semantic front, tilt, unknown categories and errata need explicit masks. Floor polygon/height/opening evidence exists, furniture mesh absent. |
| SAGE10k | Source local full dimensions; asset bottom origin; `place_id` retained in raw per-scene layout. | Real `Rz Ry Rx` physics pitch/roll, degrees. Placement-constraint XY is **cm**, unlike pose meters. | Audit-only: source README's “nonsemantic x/y” statement is insufficient. v1 +Y front is empirical; no silent standing-up, clipping or height lifting. Exact pose in placement constraints is label leakage. |
| HSSD | `obb_half_extents` already instance-scaled; `obb_center` vs asset-local translation clearly separate. Articulated rows lack source box. | Meter/right-handed Y-up; quaternion WXYZ; **never reapply `non_uniform_scale`** to OBB extents. | Audit-only: per-asset up/front not retained in asset catalog; opening metadata center offset, articulated twin identity and frame evidence require reliable source-level handling. GLB/URDF assets exist. |
| MultiScan | OBB center, local **half** extents, three raw box axes plus explicit `front`/`up`. All 10,957 rows have strictly positive extents. | Meter/right-handed Z-up; no extra scale. Reject non-orthogonal/tilted frames and non-horizontal/misaligned fronts. | **Admitted after strict filter.** Source floor hull and slab top stay uncertain; mesh assets are whole scans, not a ready Resolver asset catalog. |

Source evidence lives beside each export: `README.md`, its extraction/aggregation
scripts, retained raw `source_fields`, manifests and source records. The machine
audit includes their relative paths and hashes. `docs/known-issues.md` supplies
the previously confirmed v1 errors; those adapters are not silently reused.

## Initial actual MultiScan build

The initial raw-source smoke output is the immutable directory
`data/build/fastfill_v2/multiscan-v2-20261005/`; the complete inventory is
`data/build/fastfill_v2/source-audit-v2-final-20261005.json`. The inventory records
its auditor hash and local source evidence hashes. Both builder and audit
protect the canonical external source root even when `--source-root` points to
a different dataset location.

The earlier `multiscan-audited-20261005/` build predates exchangeable request
groups, and `multiscan-exchangeable-20261005/` predates the extra canonical-root
output guard. Both remain retained. The final split JSONL files are identical
to the exchangeable build; the final manifest records the updated builder hash.
No build overwrites any existing dataset or source file.

| Split | Samples | Supervised requested objects | Underlying source scenes |
|---|---:|---:|---:|
| Train | 159 | 4,699 | 87 |
| Validation | 16 | 252 | 9 |
| Test | 33 | 1,230 | 11 |
| Total | **208** | **6,181** | **107** |

All 273 original scans were inspected. Strict rejection removed 65 scans:
30 with unrepresentable tilted fixed geometry, 19 with a front not aligned to a
box axis under the protocol tolerance, 11 with an inconsistent up axis, and
5 with significant tilt on requested objects. These are **eligibility rejection
counts**, not proof that every source annotation is wrong.

The geometry tolerance is `1e-5`, configured by the versioned adapter constant.
Of all 10,957 raw objects, 10,729 passed the initial orthonormal/+Z upright
check; the final scene-level check additionally verifies front/up box-axis
alignment and all retained obstacles. No epsilon/absolute-value size repair is
applied. Unsupported geometry rejects the whole scene rather than removing a
potential obstacle. Floor/ceiling slabs remain room geometry; other architectural
and upright vertical-front objects remain explicit existing fixed boxes.

Requested IDs are anonymous sample-local IDs. Conditions contain category and
ordinary templated description; they contain **no source asset identity,
target size, or target pose**. Repeated category/description requests are grouped
only because no input relation, role, support or capability distinguishes them.
The matcher rechecks exchangeability; adding such roles in a future adapter
requires revisiting these group assignments.

`house_id` is MultiScan's underlying `scene_id`; rescans share one split.
SHA256 of `(seed, source, house_id)` assigns 80/10/10 before any derivation.
With seed 42, the realized small-source split is the table above. Categories,
source geometry, sample ordering and target permutations do not affect split.
Per-source CSV hashes, adapter/builder hashes, house assignments and every
rejection are saved in `manifest.json`.

Room footprint/floor are copied from **independent structural** source fields.
No furniture-derived expansion, median furniture-bottom floor estimate, GT
ceiling fill, or reference-layout relation text enters condition. The source
convex hull covers only scanned floor slabs and may omit occupied regions;
`boundary_known=false`, `floor_known=false`, and
`boundary_quality=partial_scanned_floor_convex_hull` make this visible to the
model and Validator. Unknown physical support is omitted rather than labeled
as floor contact. Unreliable ceiling/height is `null`.

The 208 admitted conditions and targets pass the v2 schema validator. Request
counts range up to 101, with median 27.5; fixed-object count is at most 19.
An independent all-target source-corner comparison checked **6,181** OBBs against
their original CSV center/axes/half-extents, with maximum nearest-corner residual
**2.095e-15 m**. All 208 samples also pass geometry collation with a fitting
128-slot/12,795-byte-token budget. This establishes box conversion, not original
mesh/semantic correctness or source-floor validity.
Training must reject whole samples exceeding object/context budgets, or create
a separately verified complete subscene. It must not truncate object/support
references to meet a budget. The byte tokenizer's context consumption differs
from Qwen's tokenizer; use a fitting context limit for offline smoke tests.

## Rebuild and checks

Use a **new** output directory each time:

```bash
python -m fastfill.v2.audit \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/source-audit-new.json

python -m fastfill.v2.data \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/multiscan-new

python -m pytest fastfill/v2/tests/test_data.py -q
```

The source adapter tests cover source-axis/extent conversion, metric units,
source-corner round-trip, local-size invariance to yaw, strict invalid-size and
tilt rejection, source-only floor/boundary, existing obstacles, anonymous
exchangeability, split stability, and source/symlink/output-write protection.
The tests use source semantics rather than inherited v1 labels.

No full source mesh overlay/render audit, reusable asset extraction, fresh GPU
training, model-quality comparison, or actual-asset success measurement has been
performed by this data build. MultiScan's 159 train scenes are an initial
correctness/overfit source, not a demonstrated sufficient corpus for production
joint-size/pose generalization. Admitting other sources needs source-specific
canonical front/pivot/scale evidence and regression fixtures first.
