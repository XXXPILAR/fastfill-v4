# FastFill v2 — experimental joint geometry generator

Main task: **room geometry + object requests + fixed objects + available support
and relation constraints** predicts one target local full size, bottom-center and
yaw per stable requested ID. Continuous heads provide structural supervision;
Hungarian matching is restricted to legal exchangeable groups. Deployment design
then resolves actual assets, reconciles actual geometry and validates the whole
scene before atomic Host commit. Current code implements the offline reference
contracts; real mesh/physics/Solver checkers and a persistent WorldEdge Host
adapter remain integration work.
The separate three-field `reference_extent` view is a simplified-input ablation
that exports bbox JSON and GLB/SVG without an asset library. It does not replace
the main full-condition task or its deployment acceptance requirements.
Qwen encodes condition tokens only. An external bidirectional object decoder
cross-attends to all condition hidden states and produces continuous geometry.
Text SFT is an independent token-CE comparison. Neither requires MinkowskiEngine.

The architecture follows the 2026-10-05 design. The direct minimal-input boundary
is an independent ablation. It is **not a validated method**.
No production Qwen training, quality improvement, real Host deployment, mesh or
physical simulation success is claimed. Existing `fastfill/train.py`,
`evaluate.py`, `serve.py`, and v3/v3.1/v3.2 dataset/experiment lineage are retained.
Model version v2 is separate from the older dataset's v3-series numbering.

The final mesh consumer is **RoomGenBench**. `predict --export-dir` now saves
canonical `condition.json` / `layout.json` alongside SceneSpec, asset registry,
GLB/SVG and diagnostics. `python -m fastfill.v2.roomgenbench` accepts any such
handoff and an independent output directory, optionally consuming generated
GLB/JSON sidecars. It preserves declared support, constraints and fixed bbox
proxies, counts failed/fallback/missing assets, and records native/fitted sizes.
See the [downstream contract and commands](../../docs/fastfill-v2-roomgenbench-interface-20261006.md).
This mesh-fit adapter does not run learned generation or certify physics/Host.

The formal runs (round 2) are `configs/qwen3_8b_main_4gpu_regression.json`
(4 ranks x B1 x K24, regression position head) and
`configs/qwen3_8b_main_3gpu_grid.json` (3 ranks x B1 x K32, grid_residual head):
both global batch 96, 3,887 updates (3 epochs of 124,375 scenes), warmup 117,
validation and checkpoint every 500. They inherit everything else from
`configs/qwen3_8b_main_world7.json` (the earlier seven-rank candidate). Pass
them explicitly with the qualified dataset; generic defaults remain
development configurations. Launch commands are in the round-2 section below. Latest audit, source pins and publication state are
in the [design audit](../../docs/fastfill-v2-design-audit-20261006.md).

## What is implemented

| Module | Behavior |
|---|---|
| `schema.py`, `geometry.py`, `batch.py` | Strict finite-number/ID protocol, condition-only normalization/token spans, masks, fixed geometry, complete-sample budget rejection |
| `model.py` | Qwen-family condition backbone (LoRA/full/frozen), request-bound slots, bidirectional self-attention + cross-attention, positive exponential size, position (`model.position_head` regression or grid_residual), yaw logits/residuals, checkpoints |
| `matching.py`, `losses.py` | Fixed identity or explicitly certified within-group Hungarian (groups from `validity.exchangeable_group`), detached assignment, differentiable original tensors, complete-field masks, L1 / smooth-L1 selectable position and size terms, symmetry-aware paired yaw loss, box-symmetric size/yaw candidates for `size_axis_swap_allowed`, per-term sums and counts for window logging |
| `boxes.py`, `regularizers.py` | Optional BEV oriented convex-hull GIoU, normalized collision-volume and convex-room boundary losses; no implicit 3D GIoU |
| `audit.py`, `adapters.py`, `data.py`, `legacy_build.py`, `legacy_bridge.py`, `legacy_evidence.py` | Source inventory, immutable migration of the selected v3.2 corpus with inherited splits and evidence-based field masks; explicit raw MultiScan audit/smoke adapter |
| `qualified_data.py` | New full-condition main derivative with pinned parent/source hashes, D1/D2 qualification changes, fixed-identity fallback for groups losing complete geometry, per-member change records and unchanged target numbers |
| `train.py`, `text_sft.py`, `evaluate.py` | Joint optimizer with warmup/cosine schedule, decoder LR, seeded augmentation, resume and per-checkpoint export; assistant-only text CE; checkpoint inference, all-request failures, legal matching reference metrics, mean-predictor baselines, collapse metrics and separated asset/system metrics |
| `direct_layout.py`, `bbox_visualization.py`, `minimal_data.py`, `multisource_data.py` | Independent three-field ablation, explicit reference-extent semantics, source-audited partial-mask projection, canonical/RoomGenBench JSON with polygon-wall shell, door/window openings, declared/inferred/unknown `place` and benchmark-convention shared `asset_key`, no-asset bbox GLB/SVG and proxy diagnostics |
| `predict.py` | Main `--condition` checkpoint inference; separate `--request` ablation and optional offline catalog execution; `--max-length` defaults to the checkpoint's training value |
| `runtime.py`, `validation.py`, `serve.py` | Catalog resolver contract, target/actual separation, pivot-to-canonical transforms, verified support propagation, bounded reselect/translation repair, atomic in-memory Host reference |

The latest [objective update](../../docs/fastfill-v2-review3-objective.md) adds enabled-loss preflight and a globally counted accumulation-window optimizer guard.

## 2026-10-06 audit fixes

Contract changes from the three-group audit (schema, serialization, yaw policy, matching, losses, trainer, evaluation, RoomGenBench handoff):

- **Row schema.** `condition.objects[i]` carries only `id`, `category`, `description` and optional `support_parent`. Exchangeable groups moved to `validity.exchangeable_group` (`list[str|null]`, target order) and `validity.yaw_symmetry_order` (`list[int]`, 1 = semantic front, 2 = axis mod pi) is always present. `io.read_samples` migrates old rows (`schema.migrate_legacy_row`) once; `validate_condition` rejects `exchangeable_group` in objects.
- **Condition text.** `batch.condition_segments` renders `schema_version, room, constraints, objects` in that fixed order (objects last, so causal object tokens see the room), drops `room.boundary_quality` and every `_`-prefixed key; object span tagging and the prompt header are unchanged. The direct request path therefore renders the same room fields as a training rectangle.
- **Yaw policy at build.** Default `--front-policy axis`: yaw valid = upright, finite, `front_known`; `yaw_symmetry_order` 1 for MultiScan, 2 elsewhere; round 2: `validity.size_axis_swap_allowed` (box may equally be (sx, sy, yaw) or (sy, sx, yaw + pi/2)) is true for InternScenes arkit/3rscan/mp3d/scannet, InteriorGS, HSSD200 categories containing `chair` and MultiScan categories containing `bed`, and those objects carry order 4 (old rows migrate to all false). Floor support comes from the source annotation only: an `anchor = floor` (not inferred) object on a known floor is declared `support_parent: floor` and its target z snapped when within 2 cm (`field_evidence.legacy_z_snap_applied_to_target`), otherwise left undeclared with free z; the manifest counts `floor_declarations_written` / `floor_declaration_z_snapped` / `floor_declaration_skipped_floating` and `source_size_axis_swap_allowed_objects`. `strict` and `legacy-convention` remain selectable (`legacy_bridge.FRONT_POLICIES`). Descriptions are the source `desc` or the category (`provenance.descriptions = source_desc_or_category`). Groups need complete **position** only; signature = category/description/support_parent.
- **Qualification.** Any size axis < 3 mm masks the whole size vector (`degenerate_axis_lt_3mm`); a target above `room.height_m + 0.05` keeps labels and the declared height and is flagged in `provenance.height_conflict = {objects, max_excess_m}` (journal field `provenance.height_conflict`, reason `target_exceeds_declared_height_flag_only`; round 2 reverts the round-1 height drop); the five RoomGenBench SAGE layouts move to the test split with `provenance.holdout_reason = roomgenbench_benchmark_room`. `legacy_verify` / `multisource_verify` recompute `source_yaw_valid_objects`, `source_yaw_symmetry_order_counts`, `exchangeable_group_counts`, `yaw_policy`, `descriptions` and accept the holdout.
- **Matching and losses.** `match_batch` reads `batch["exchangeable_group"]` (per request slot). Group cost = `alpha_position * L1(position)` always, plus `alpha_size * L1(log size)` only when every member has a complete size label; position-incomplete groups keep fixed identity. `loss.position_type` / `loss.size_type` in {`l1`, `smooth_l1`} (default `l1`), `loss.smooth_l1_beta` (default 1). Main-config weights 1.0 / 0.6 / 0.08 / 7.0 (round 2 caps yaw_reg at 2.0) come from the [calibration note](../../docs/fastfill-v2-loss-calibration-20261006.md). The criterion exposes `term_sums` / `term_counts` so window logs are count-weighted means over every microbatch.
- **Trainer.** New sections `optimizer` (`warmup_steps` default 3 % of steps, `schedule` cosine to 10 % LR, `decoder_lr`), `augmentation` (`rotate90`, `mirror`, `shuffle_objects`, `drop_constraints_p`, `drop_support_p`, `category_only_description_p`, round 2 `minimal_form_p`; training rows only, seeded per (seed, epoch, row)), `validation.exclude_flags` (default the three training flags), `training.resume`, `training.export_model_every_checkpoint`. Unknown keys raise. `--resume <state-step-n>` continues into a new output directory with identical config/data; each checkpoint writes `state-step-<n>/` and `model-step-<n>/`; `train.log` tees stdout; bf16 windows with nonfinite gradients are skipped and counted; a room-centre / category-median-size / uniform-yaw baseline is logged before training.
- **Evaluation.** `report.json` adds `baselines` (room centre, category mean position, category median size, uniform yaw; leave-one-out or `--baseline-fit <jsonl>`), `collapse` (predicted vs ground-truth overlap, stacking, wall distance, central fraction), `by_source` and `yaw_error_rad_by_symmetry_order`.
- **RoomGenBench export.** `room.walls` from the floor polygon (0.1 m thick, height = room height or 2.7), `doors` / `windows` from fixed objects attached to the nearest wall, `place` / `support_status` (round 2, `direct_layout.infer_support`): a declared support parent or hard `on` gives `floor` / `on_object` / `wall` with `declared`; otherwise `floor` when the bottom is within 2 cm of a known floor height (none when `floor_known` is false, e.g. `reference_extent`), else `on_object` for the highest strictly-lower predicted box whose top is within 3 cm and whose footprint contains the object's centre and whose declared or earlier inferred support chain does not lead back to the object (no `A -> B -> A` from a declaration), both `inferred`; else (round 10) `wall` when the bottom is over 0.15 m above a known floor, a footprint side lies within `validation.WALL_GAP_M` (0.1 m) of a known boundary and no box lies under the object's centre, `inferred`; anything else `unknown`. `assets.jsonl` and assembly receipts carry the same `place` and `support_status`; `asset_key = slug(type)[:24] + "_" + sha1(description)[:8]` shared by identical type+description (benchmark convention), per-instance dimensions in the scene. Tests need the reference assembler at `RoomGenBench/` or `FASTFILL_ROOMGENBENCH_ROOT`. See the [server setup guide](../../docs/fastfill-v2-server-start.md) for the supplied H20Z host. The historical [review2 audit](../../docs/fastfill-v2-review-20261006.md) separates repaired defects, data provenance and remaining training limits.

Review repairs on top of the above (same day):

- **Augmentation.** `rotate90` / `mirror` snap rigid-motion float noise (`8.8 - 1.1` renders as `7.7`, full-precision source values are kept), and `train._Rows` falls back to the preflighted row when an augmented condition exceeds `max_length` (`run_manifest.augmentation_fallbacks`). After `drop_constraints_p` / `drop_support_p` / `category_only_description_p` fire, exchangeable groups are recomputed with the build rule (`matching.group_labels`): requests made textually identical by a drop are matched by geometry instead of fixed identity; build-time groups whose wider candidate fails certification keep their members under a `kept_<label>` label.
- **Benchmark rooms.** `io.ROOMGENBENCH_HOLDOUT_GROUPS` is the single producer copy (`multisource_verify` mirrors it independently); `io.read_samples(training=True)` (train.py, text_sft.py) and `cohort` refuse rows whose `provenance.group` is a benchmark room or that carry `holdout_reason`.
- **RoomGenBench assembly.** `roomgenbench._shell_parts` builds each wall with the reference `build_shell` but takes the outward normal from the floor-polygon winding (the reference uses the bbox centre, which extrudes some walls of non-convex rooms into the room); receipt `room.wall_normal_source = floor_polygon_winding`.
- **Smaller.** `decode_yaw` computes bin centres in at least float32 under bf16 residuals; evaluation baselines are row-level leave-one-out (every label of the evaluated row is excluded, not only the object's own); the declared-height conflict adds a 1e-6 m epsilon so float32-rounded sizes do not flag a real height; `legacy_bridge` reads Scan2CAD `sym` (`__SYM_ROTATE_UP_4` -> `yaw_symmetry_order` 4, `__SYM_ROTATE_UP_INF` -> yaw unsupervised; `legacy_verify` accepts orders 1/2/4).

## 2026-10-07 round 2

Contract K1–K10 on top of commit 7ea1a0d. Rows built before round 2 still load
(`schema.migrate_legacy_row` adds an all-false `size_axis_swap_allowed`), but
K1–K3 only reach the training data through the rebuild below.

- **K1 box symmetry.** `validity.size_axis_swap_allowed` (bool, target order, always written): the labelled box may equally be (sx, sy, yaw) or (sy, sx, yaw + pi/2). The axis policy sets it for InternScenes arkit/3rscan/mp3d/scannet and InteriorGS (all objects), HSSD200 categories containing `chair` and MultiScan categories containing `bed`; those objects carry `yaw_symmetry_order` 4. Collate exposes `batch["size_axis_swap_allowed"]` (b x n bool). Loss and matching: see "Geometry and exact losses". Evaluation scores these objects' `log_size_error` / `yaw_error_rad` as the box-equivalent minimum over the same four candidates, keeps `*_plain_convention` (written axis order, yaw modulo pi for swap objects and modulo the recorded order otherwise) and counts `box_equivalent_objects`; the category-median-size baseline is box-equivalent too.
- **K2 height conflict.** Flag only; the round-1 height drop is reverted (see "Qualification" above).
- **K3 floor support.** Source `anchor = floor` only, 2 cm snap, farther anchors stay undeclared with a free z (see "Yaw policy at build" above); `legacy_verify` recomputes the manifest counts.
- **K4 MansionWorld sizes.** IR objects carry only `id, category, size, pos, yaw, anchor, parent, tilted, desc, structure`; no field separates annotated bbox sizes from footprint proxies, so the source-level mask stays (every MansionWorld and OptiScene_holodeck size invalid). Using `anchor` as an indirect proxy is an open decision.
- **K5 three-field contract.** `batch.render_minimal_condition` keeps `schema_version`, room {frame, axis-aligned bounding rectangle, floor_z_m, floor_known, boundary_known = true, height_m, room_type}, `constraints: []` and objects {id, category, description}. `direct_layout.request_to_condition` calls it, so a default (`rectangular`) request renders byte-identically to the projection of a training row with the same room type, size and inventory (`boundary_quality` is never rendered). The separate `reference_extent` ablation profile still sets `boundary_known` false after the projection, as its own frozen data does. Augmentation `minimal_form_p` (default 0.5) fires only when `batch.minimal_form_eligible`: a 4-point axis-aligned rectangle (1 cm tolerance, +1e-9 m float margin so rotations agree) whose `boundary_known` is not false, so hull and `reference_extent` rectangles are never rewritten as known; it regroups exchangeable groups and drops floor-fixed z for that sample. `evaluate.project_minimal` is the shared projection (same predicate; other rows count as `skipped_non_rectangular_rooms`) for `evaluate --projection full minimal` and the trainer's periodic validation, so a `reference_extent` ablation run has no minimal projection and no `selection_metric.best`.
- **K6 hand-off place.** `place` / `support_status` in {declared, inferred, unknown}; see "RoomGenBench export" above. Hand-offs exported before round 2 fail the scene check of `roomgenbench._handoff`; re-export them with `predict --export-dir`.
- **K7 grid head.** See "Geometry and exact losses".
- **K8 binding.** `run_manifest_start.json` before the first update (config, data / validation / implementation sha256, resolved backbone path + HF snapshot revision + config.json sha256 (a plain local directory is identified by config.json only, which does not pin the weights), tokenizer sha256, augmentation, world size, admitted `supervised_samples` / `rejected_samples` / `validation_samples` / `validation_minimal_samples`, selection metric); `run_manifest.json` at the end (adds `selection_metric.best`); `checkpoint_manifest.json` in `model/` and every `model-step-<n>/`. `--expect-data-sha256` / `--expect-validation-sha256` abort before any output. `evaluate` / `predict` take `max_length` from the checkpoint (`io.load_checkpoint_config`) unless `--max-length` is explicit, and model settings from its `model_config.json`; evaluation reports record `checkpoint`, `checkpoint_binding`, `max_length_source`, `data_sha256`, `implementation_sha256`, `code_commit` / `code_dirty`.
- **Selection metric.** `validation.minimal.geometry_objective` (`train.SELECTION_METRIC`) = the weighted geometry objective on the minimal projection of the validation rooms, lower is better. Each validation runs the rectangular subset a second time. Collapse (predicted BEV overlap rate at IoU > 0.3 + central-quarter fraction) is logged beside it, never minimised on its own, as `collapse` (every request) and `collapse_matched` (the predictions the criterion's matching assigns to label-complete labels); the labels' own collapse score on the same projection is recorded once as `selection_metric.ground_truth` in `run_manifest_start.json` and `run_manifest.json`. Only `collapse_matched` covers the same objects as that anchor (size-masked sources such as MansionWorld are only in `collapse`): a `collapse_matched` score below it means a more spread-out layout than the data, not a more accurate one. The final update is always validated and checkpointed as well (unless the interval is 0), so `steps` need not be a multiple of the interval. Keep `validate_every == checkpoint_every` so the best step has a `model-step-<n>/`.
- **K9 configs.** Every Qwen config uses `max_length` 8192 (including the Qwen2.5-0.5B `structured*` templates); main and `pilot_1gpu` use `yaw_reg` 2.0, `position_cell` 0.04, `position_residual` 0.4, `minimal_form_p` 0.5; formal configs as at the top of this README.
- **K10 validator.** `validate_scene` check status is `pass` / `violation` / `unknown` (was `fail`) with `counts`; `ok` is unchanged (every hard check must pass, unknown is not pass). This changes the JSON of serve.py / runtime reports. `evaluate` reports `target_validation_checks` per check code and `target_validation_not_run`; ceiling checks on `provenance.height_conflict` rows count as `ceiling_on_height_conflict_rows`. `direct_layout.bbox_diagnostics` (hand-off proxy diagnostics) keeps its own pass/fail words.

Rebuild (CPU, about 1–1.5 h; `OUT` must be new and outside `.release` and the
external disk). The parent hash pins still name review3, so the qualified and
verify steps pin the fresh parent explicitly:

```bash
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
R=/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/v3.2
E=/Users/slian/Desktop/3D/Worldedge/OptiScene/.release/audits/2026-09-28/source-check
OUT=outputs/fastfill_v2/rebuild-20261007
python -m fastfill.v2.legacy_build --release-root $R --evidence-root $E --front-policy axis --workers 4 --output $OUT/bridge
python -m fastfill.v2.legacy_verify --data-root $OUT/bridge --output $OUT/reports/legacy-verify.json
python -m fastfill.v2.review_data build --parent-root $OUT/bridge --output $OUT/reviewed
python -m fastfill.v2.review_data verify --parent-root $OUT/bridge --data-root $OUT/reviewed
python -m fastfill.v2.qualified_data --parent-root $OUT/reviewed --output $OUT/main --ir-root $R/ir \
  --frozen-manifest $R/data/v3.2/MANIFEST.json --pin-current-parent
python -m fastfill.v2.multisource_verify --full-condition --parent-root $OUT/reviewed --data-root $OUT/main \
  --ir-root $R/ir --parent-manifest $OUT/reviewed/manifest.json --output $OUT/reports/main-verify.json
# Optional three-field ablation view and its independent verification:
python -m fastfill.v2.multisource_data --parent-root $OUT/reviewed --ir-root $R/ir \
  --frozen-manifest $R/data/v3.2/MANIFEST.json --output $OUT/data-minimal-reference
python -m fastfill.v2.multisource_verify --parent-root $OUT/reviewed --data-root $OUT/data-minimal-reference \
  --ir-root $R/ir --parent-manifest $OUT/reviewed/manifest.json --output $OUT/reports/minimal-verify.json
```

A bounded smoke build adds `--source NAME` (repeatable),
`--max-scenes-per-source N` and `--include-uid UID` (repeatable) to `legacy_build` (not forwarded by `fastfill.v2.data`).

Formal runs, side by side on GPUs 1–7 (GPU0 stays free). `--nproc_per_node`
must match the config (any other world size changes the global batch of 96):

```bash
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
MAIN=/path/to/rebuild-20261007/main
D=$(sha256sum $MAIN/train.jsonl | cut -d' ' -f1); V=$(sha256sum $MAIN/validation.jsonl | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES=1,2,3,4 python -m torch.distributed.run --standalone --nproc_per_node=4 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_4gpu_regression.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-4gpu-regression
CUDA_VISIBLE_DEVICES=5,6,7 python -m torch.distributed.run --standalone --nproc_per_node=3 --module fastfill.v2.train \
  --config fastfill/v2/configs/qwen3_8b_main_3gpu_grid.json \
  --data $MAIN/train.jsonl --validation $MAIN/validation.jsonl \
  --expect-data-sha256 $D --expect-validation-sha256 $V --output ../run-main-3gpu-grid
```

Check `run_manifest_start.json` right after launch: `world_size` 4 / 3 and
`supervised_samples`. 3,887 updates are 3 epochs only for 124,375 admitted
scenes; the trainer's own preflight now uses 8192 tokens, so for another count N
write a new config with `steps = ceil(3 N / 96)` and `warmup_steps = round(0.03 steps)`.
Pick the checkpoint by `run_manifest.json` `selection_metric.best.step`, never by
test; then `python -m fastfill.v2.evaluate --checkpoint <run>/model-step-<best>
--data $MAIN/test.jsonl --projection full minimal --device cuda --output <new dir>`.

Recorded local validation (2026-10-07, CPU, tiny backbone, no Qwen weights):
`pytest fastfill/tests fastfill/v2/tests` (with `FASTFILL_ROOMGENBENCH_ROOT` set): 1119 passed, 114 subtests passed, no skips (after the H2 review repairs, `tests/test_review_repairs_r2.py`). End-to-end smoke in the session scratchpad: a 200-scene bounded
build (HSSD200, MultiScan, InternScenes_arkit, MansionWorld, IL3D_synthetic x 40)
passed `legacy_verify`, review build/verify, qualified build and
`multisource_verify --full-condition` (ok, no errors) plus the minimal view and
its verifier (478 swap objects, 1,134 floor declarations of which 91 snapped,
4 floating anchors skipped, one height-conflict flag); a tiny grid_residual run
with `minimal_form_p` 0.5 trained 6 steps, and `--resume state-step-4` reproduced
the step-6 weights exactly (max |diff| 0); `evaluate --projection full minimal`
on 50 cross-source rows bound `max_length` to `checkpoint_manifest.json`
(50 full / 17 rectangular minimal requests, 135 box-equivalent objects); one
room was exported by `predict --condition` and `--request` and assembled by the
real RoomGenBench assembler. These are code-path checks, not model quality.

Open decisions: K4 above; the literal HSSD200 `chair` rule matches 27 objects
while 2,982 HSSD200 `seat` objects are not marked (and `bed net` / `desk and chairs`
are).

## 2026-10-07 round 3

On top of 816f0c2. Row and checkpoint formats are unchanged: the first 50
validation rows of the 20261006 package and of `rebuild-20261007/main` load
through `io.read_samples`, `collate_samples` (with and without augmentation) and
`match_batch`, and existing `model-step-*` / `state-step-*` directories load.
The two data items reach training only through the round-2 rebuild commands
above (into a new `OUT`).

- **Subset groups.** `matching.group_labels`: when the full swap of a signature set fails certification, its members that no constraint reference field and no `support_parent` names still form the set's `anonymous_<n>` group if at least two remain (three identical chairs, one named by a `faces` constraint: the other two are now matched by geometry). `legacy_verify` checks both halves of the rule on swap certification (a whole set that certifies shares one label, else its unreferenced members do); augmentation regroups with it. Known residual: a referenced but mutually symmetric subgroup inside a failed set (two display cabinets each `against_wall` beside a third, free one) stays ungrouped (7 legal pairs in the first 30,000 `rebuild-20261007/main` train rows, none grouped illegally). Rows stored before this rule keep their groups unless augmentation regroups them (any drop or the minimal form); on those 30,000 rows the new rule has 8.1 % more group members (82,193 -> 88,817), so resuming a pre-round-3 run on this commit trains augmented and unaugmented samples of the same rows under the two rules. `qualified_data` still drops the whole group when a member loses position (only Scan2CAD demotes positions, whole rows at a time). 300-scene smoke: 28 rows gained members (1,980 -> 2,087).
- **Room height independent of targets.** `legacy_bridge` keeps the source height (MultiScan only when `height_reliable`) even where the frozen prep dropped it because of a target; the drop stays as `provenance.legacy_height_dropped` and the bridge writes the K2 flag `provenance.height_conflict` itself (`legacy_bridge.height_conflict`, shared). `qualified_data` recomputes the flag on its qualified masks and journals only a difference (`before` = parent flag, reason `height_conflict_follows_qualified_masks`); `legacy_verify` and `multisource_verify --full-condition` recompute it independently, and `legacy_verify` rejects a null `room.height_m` on a `legacy_height_dropped` row (MultiScan: when `source_meta.height_reliable`). `qualified_data` records `legacy_bridge.py` in `implementation_sha256` beside its own files.
- **Per-target size swap in matching.** A pair takes the (sx, sy) / (sy, sx) minimum only when its target has `size_axis_swap_allowed`; one swap member no longer makes the whole group swap-tolerant.
- **Collapse metrics.** The predicted column counts every requested object (predicted boxes), so a legal ID swap cannot change it; the ground-truth column still needs complete position and size labels and finite yaw, so size-masked sources (MansionWorld, OptiScene_holodeck) appear only in the predicted column. `predicted_matched` holds the predicted objects the reference matching assigns to the ground-truth column's slots (the same objects; a legal swap only permutes them), and `autorun.score` / SUMMARY.md compare it, not `predicted`, with the ground truth: the labels themselves now score a zero distribution penalty (before: 0.038 on 1,500 `rebuild-20261007` validation rows). New `out_of_room_fraction` (bottom centre more than `evaluate.OUT_OF_ROOM_M` = 0.05 m outside the floor polygon); `collapse_score` is unchanged and scores 0 for a wall-hugging layout, so read it with `out_of_room_fraction` and the reference errors.
- **RoomGenBench doors.** A second door on one wall is `shell_wall_<i>_door_1`, ... (the first keeps `shell_wall_<i>_door`) instead of overwriting the first in the GLB.
- **Resume binding.** States also save `world_size` and `validation_data_sha256`. Resuming with either changed raises unless `--allow-resume-change`, which records `resumed_with_changes = {field: {saved, current}}` in both manifests; a validation change restarts `selection_metric.best` at the resumed step, and the states carry that step (`selection_after_step`) so later plain resumes keep it (`run_manifest.json` `selection_metric.best_after_step`). A world-size change keeps each rank's old `batches_done`, so the data order is not exact after it. States saved before round 3 (the running formal runs) resume with a stderr warning and `resume_unverified = ["world_size", "validation_data_sha256"]` in both manifests; the flag is not part of the config, so `config_sha256` is unchanged.
- **Kept by design: the 2 cm floor declaration.** Review F04 (the full-condition `support_parent: floor` hint depends on whether the source anchor is within 2 cm of the floor) is left as is. It withholds the hint only from the 894 floating source floor anchors of `rebuild-20261007` (`floor_declaration_skipped_floating`, against 682,758 declarations), and the minimal (three-field) form, which `minimal_form_p` 0.5 trains on and model selection scores, drops `support_parent` anyway.

Recorded local validation (2026-10-07, CPU, no Qwen weights): `pytest fastfill/tests fastfill/v2/tests` (with `FASTFILL_ROOMGENBENCH_ROOT` set): 1166 passed, 114 subtests passed. A 300-scene bounded build (HSSD200, IL3D_3dfront, SAGE-10k, MultiScan, InternScenes_arkit x 60) passed `legacy_verify`, review build/verify, qualified build, `multisource_verify --full-condition` and the minimal view with its verifier. The full rebuild was not run.

## 2026-10-07 round 4

On top of 2ce6b58. Row, checkpoint, CLI and config formats are unchanged; `match_batch` only gains the keyword `loss_config` (default `LossConfig()`, the criterion passes its own) and `spread_grid_xy` the keywords `requests` / `fixed_z` (default: every object undeclared).

- **L1 grid matching cost = the loss's position terms.** For grid_residual predictions (`position_cell_logits` present) `match_batch` charges each (prediction slot, target) pair `losses._grid_position` under the criterion's config: `position_cell * CE(target cell) + position_residual * GT-cell residual + z/3` on the prediction slot's learned (not `fixed_position_mask`) coordinates, times `alpha_position`; the size term is unchanged. Before, it compared only decoded positions, so two slots with the same argmax cell tied and SciPy's row order picked the assignment while the full cell CE was charged: the rebuilt review counterexample (2 x 2 grid) cost 3.2598 or 5.5098 by slot order with assignment [0, 1] both times; now 3.2598 both ways (`tests/test_round4_matching.py`). The regression head, evaluation matching (decoded layouts carry no grid keys) and the training preflight keep the decoded L1.
- **L3 spread decoding honours declarations.** `serialize_predictions` passes `validation.effective_support_requests` (hard `on` constraints merged into `support_parent`) and `fixed_position_mask[..., 2]` to `spread_grid_xy`, and the room frame carries `room.fixed_objects`. Fixed objects block cells like placed ones and can carry raised objects; an object with a fixed z stands at that z, one declared on `floor` on the floor, one declared on `wall` keeps its predicted z (it never drops onto furniture or the floor; round 6 hangs it on a wall by rule, see below); one declared on a request or fixed object is placed after its parent (support-chain depth) on the parent's top: at its most probable top_k cell whose centre lies in the parent's (spread) footprint, else at the free point of a 5 x 5 lattice over that footprint (inset by the child's half extents) nearest its argmax decode, else at the least overlapping of these (support beats collision). Undeclared raised objects rest, as before, on the highest placed or fixed floor-standing object (fixed bottom within 0.15 m of the floor, so a wall shelf does not catch a cup). Any other object no candidate can hold takes, as before, the least overlapping cell at its standing height; without declarations, fixed objects or fixed z the decode is bit-identical to 2ce6b58. Unknown parents and support cycles raise ValueError. Review counterexample (cup declared on a fixed table with a 1 m top): spread now writes [1, 1, 1], which validates (support pass, no `fixed_collision`); before [1, 1, 0] (`tests/test_round4_spread.py`). `--grid-decode argmax` is unchanged.
- **D4 kept: Scan2CAD rotational symmetry as exported.** `legacy_bridge` keeps `sym` `__SYM_ROTATE_UP_4` -> `yaw_symmetry_order` 4, including the five objects the review found with sx != sy (for them a 90-degree yaw error costs no yaw loss). The annotation is the source's (collaborators' export), and the dataset catalog (`excel/3D_Room_Scene_Collection.xlsx`, SceneCAD&Scan2CAD row: no collision repair, category rewriting or outlier deletion; degenerate source annotations kept as is) rules out silently repairing or rewriting source annotations. Their positions are already masked (`multisource_data`: every Scan2CAD position, `Scan2CAD_estimated_floor_and_upstream_snap_uncertainty`), so they enter neither matching nor the position loss.

Recorded local validation (2026-10-07, CPU, no Qwen weights): `pytest fastfill/tests fastfill/v2/tests` (with `FASTFILL_ROOMGENBENCH_ROOT` set): 1208 passed, 114 subtests passed. A 4-step tiny-backbone grid_residual training (`position_grid` 8, Hungarian on, the 3 `tests/fixtures` validation rows relabelled `train` in a scratch copy) ran finite losses with 5 grid-cost assignments; `evaluate --grid-decode spread` and `argmax` on those 3 rows with its `model-step-4` both completed (0 schema failures; the untrained model fails target geometry on all 3 either way, spread with 1 instead of 5 collision and 1 instead of 4 boundary violations). The fixture rows declare no supports or fixed objects; L3 is covered by the unit tests.

Round-4 review repair (same day): the first L3 version kept the raw argmax decode for *every* object no candidate could hold, which changed undeclared decoding (200 `rebuild-main-20261007b` validation rows, minimal three-field view, synthetic oracle / noisy / random grid predictions: 61 / 68 / 39 rows changed, boundary violations 93 -> 124 / 83 -> 127 / 91 -> 101, 63 new `floor_lower_bound` violations under noisy). Now only declarations change the decode: the same 600 runs are bit-identical to 2ce6b58 (0 changed rows). On 100 MansionWorld object-support rows (872 declared children; HEAD / first L3 / now) children off their parent's top: noisy 412 / 122 / 0, oracle 93 / 28 / 0; collision violations noisy 474 / 536 / 518, oracle 309 / 336 / 331 (forcing children onto the parent without the lattice cost 803 / 385); `fixed_collision` noisy 51 / 24 / 25. 100 IL3D_3dfront floor-support rows: boundary 108 -> 107 (oracle) and 111 -> 111 (noisy), `fixed_collision` 13 -> 7 and 16 -> 10 against 2ce6b58. `pytest fastfill/tests fastfill/v2/tests`: 1210 passed, 114 subtests passed.

## 2026-10-07 round 5 (audit c750c31)

On top of 077c158; the audit (REPORT.md sections 2-4) read c750c31. Row, checkpoint, CLI and config formats are unchanged; `spread_grid_xy` only gains the keyword `keep_yaw` (default: no slot kept). Each item's audit reproduction, rerun on this tree, is quoted before -> now.

- **E1 candidate cache bound to the command and the code.** `autorun.evaluate_candidates` reuses a report only when it is scorable, of the same data and checkpoint, `projection` minimal, `grid_decode` spread and carries `implementation_sha256`. One selection uses one implementation: the cached reports' when every candidate is cached under the same one (a restarted autopilot beside its training keeps its six select1 reports, no subprocess), else the code on disk (`autorun.implementation_sha256`, equal to `io.run_metadata`'s); other reports are set aside (`.stale-<ts>`) and evaluated again, which is refused while a `fastfill.v2.train` process is alive, and a fresh report of other code (disk changed meanwhile) raises. `<tag>-selection.json` records each entry's hash. Audit case (an argmax / full report of `WRONG_OLD_CODE`): selected after 0 evaluations -> set aside, 1 evaluation with `--projection minimal --grid-decode spread`.
- **E2 LLM caches bound to their inputs.** `llm_baseline` writes `data_sha256` (rows file) and `implementation_sha256` (llm_baseline.py) into summary.json; autorun reuses a baseline only when these and `request_parameters` match, and its scored report only when that report's `data_sha256` / `predictions_sha256` match the baseline's rows / predictions; anything else is removed and run again. Audit case: both stale modes registered after 0 calls -> 2 runs, none registered. Summaries written before round 5 lack the fields, so a restarted autopilot reruns both modes once (paid API calls, CPU only).
- **E3 F-phase LLM-rows race:** closed in 077c158 (`rows_job` / `raw_job` start as None and are waited only when launched). Audit race on the current F block: `UnboundLocalError` -> no error.
- **E4 raw id compliance.** Since 077c158 duplicated, unknown or missing ids make an answer invalid instead of being cleaned (audit ids `[obj_0000, obj_0000, obj_0001, unrequested]`: accepted as two clean objects -> ValueError). Round 5 reports it apart from the retry: per row `attempts` and `first_answer_valid` (null: no request answered), in summary.json `first_answer_invalid` (rows whose first answer received was no valid layout; a failed request, e.g. URLError or 5xx after `chat`'s retries, is retried but never counts there) beside `unanswered` (rows no request answered) and `failed` (rows still without a layout). Review case (row 0: URLError, then a valid answer; row 1 valid): `first_answer_invalid` 1 -> 0.
- **E5 spread keeps a declared facing.** `serialize_predictions` passes `keep_yaw` for every object named by a `faces_direction` or `faces` constraint (hard or soft): spread never turns it back to the wall and checks its footprint at the decoded yaw. Audit case (hard `faces_direction` [-1, 0], decoded yaw pi): spread yaw 0, violation -> pi, pass. Three-field rows carry no constraints: on 200 minimal and 200 full `rebuild-main-20261007b` validation rows (synthetic grid outputs, no faces constraints) spread is byte-identical to 077c158.
- **T1 DDP diagnostics count each row once.** Validation, its minimal projection, the baseline and the label collapse are no longer prepared by Accelerate (whose even batches repeat leading rows): each rank reads its unpadded stride shard (`train._diagnostic_loader`), runs the unwrapped model and a rank-local criterion, and only the reductions after each loop are collective. `batches` is the global batch count and `geometry_objective_mean_of_batches` the mean of every batch's own objective (unchanged in one process; multi-GPU values are not comparable with runs before round 5, e.g. the running c750c31 training's). `test_training_record_review`'s two-process validation case now expects the mean of the ranks' rank-local objectives. Audit case (3 rows, 2 ranks): counts 4, position 0.2 -> 3, 0.216667 (exact 0.216667). This does reach the running c750c31 training if it crashes: autorun resumes it with a plain `--resume` into this train.py, so one log would hold both conventions. States therefore record `diagnostics: "rank-shards"`; resuming a multi-rank state without it, with validation, records `resumed_with_changes.diagnostics = {saved: null, current: "rank-shards"}` in both manifests and restarts `selection_metric.best` at the resumed step (carried on as `selection_after_step`), with no `--allow-resume-change` needed since training is unchanged. Review case (2 ranks, bf16, accumulation 2, grid + Hungarian, augmentation, dropout 0; c750c31 killed at state-step-2, resumed here): losses and weights identical to the uninterrupted run both before and after; best step 2 (old convention, 3.13699), `best_after_step` 0, `resumed_with_changes` {} -> best step 4 (3.14978), `best_after_step` 2, the `diagnostics` entry. Autorun selects checkpoints by evaluate reports, not by this manifest field.
- **T2 resume with dropout.** Diagnostic loaders have their own generator, so diagnostics never draw from the global RNG. Audit case dropout 0.2 with validation: 36 tensors differ after resume (max 0.0107) -> 0, in all four configurations. The RNG stream differs from older runs; resuming an older state stays exact at dropout 0 (the formal runs' setting).
- **T3 Metal.** `match_batch` builds the grid pair cost, and `_Window.summary` / `_reduce_collapse` reduce, in float64 on the CPU for an mps device (CPU / CUDA unchanged). Audit case: TypeError -> runs, loss 3.2598 as on the CPU.
- **Kept: yaw is not in the matching cost.** Matching charges position (+ size) only. When two slots tie exactly on those and differ in yaw, SciPy's row order breaks the tie, so a permutation of the same prediction set can change the yaw loss (audit case, unchanged: position 2.7200 both ways, yaw_cls 0 vs 20). A permutation-invariant full objective needs a symmetry-aware yaw term in the pair cost (or the full pair cost); not changed while the formal training runs.

Recorded local validation (2026-10-07, CPU + Metal, no Qwen weights): `pytest fastfill/tests fastfill/v2/tests` (with `FASTFILL_ROOMGENBENCH_ROOT` set): 1236 passed, 114 subtests passed.

## 2026-10-07 round 6 (RoomGenBench contract)

On top of f73ced4. Row, checkpoint and config formats are unchanged; `request_to_condition` and `fastfill.v2.roomgenbench` only gain optional fields / options.

- **Requests may declare support.** A `furniture_list` entry may carry `support_parent`: `"floor"`, `"wall"` or another entry's id (task analysis supplies it). `request_to_condition` copies it onto the condition object; an unknown id or self-reference raises naming the entry, a cycle raises in `validate_condition`. Entries without it render byte for byte as before. `layout_to_roomgenbench` reports a declared parent as `place_id` with `support_status` `declared`.
- **Versioned RoomGenBench request builder.** `python -m fastfill.v2.roomgenbench --requests-from RoomGenBench/bench/inputs/scenes --requests-out <new dir>` writes one `<scene_key>.json` per scene with every object, wall ones included (313 objects; the server's unversioned `runs/roomgenbench/make_requests.py` sent 255, without `place_id`), each `place_id` as `support_parent`, checked with `predict --request`'s default `max_objects` (128). Ids are `obj_%04d` in scene order (`obj_0007` is `scene["objects"][7]`), the training id style: RoomGenBench's own ids put the restaurant at 9,362 Qwen3 condition tokens, over the 8,192 training `max_length` (`collate_samples` refuses, it never truncates); now 6,727 (bathroom 2,976, bedroom 2,511, gym 1,680, living room 2,986). The `--handoff` usage is unchanged.
- **Spread hangs declared `wall` objects.** No training row declares `wall`, so `spread_grid_xy` places them by rule: each top_k cell is projected onto its nearest side of the room's bounding rectangle, back flush and facing in (`keep_yaw` keeps the yaw), slid inside along the wall; the first projection without overlap (placed and fixed objects sharing its height interval) wins, else the least overlapping one. It keeps its predicted z when that is over 0.15 m above the floor; a z near the floor is no evidence here (the untrained five-room check put 33 of 58 there), so it then hangs centred `hang_centre` = 1.5 m above the floor (an eye-level convention, not fitted to RoomGenBench; the knob is a `spread_grid_xy` keyword) and is placed after the floor furniture. Either z is clamped to [floor, ceiling - height] (`_room_frame` now carries `height_m`; unknown height: only >= floor). One wider or deeper than the room is centred along that axis. Without a `wall` declaration spread is byte-identical to HEAD (200 full + 200 minimal `rebuild-main-20261007b` validation rows, synthetic grid heads). Known ceiling: bounding rectangle only, axis-aligned footprints for turned objects.
- **`validate_scene` checks `wall` supports.** A declared `wall` parent (or a hard `on` to `wall`) reports `wall_support`: pass when a footprint side lies within the tolerance of the room boundary (the `against_wall` geometry), else violation; `wall_unknown` without a known boundary. Before, every wall object was a `support_parent_missing` violation and a hard `on` to `wall` also a `constraint_reference` violation.
- **LLM report reuse bound to the evaluator.** `autorun.llm_baselines` reuses a scored `report.json` only when its `implementation_sha256` equals the fastfill/v2 code on disk, else evaluates again (CPU `--predictions`, no API calls).
- **`global_counts` reaches the scene regularizers.** `GeometryCriterion.forward(..., global_counts=False)` (rank-local validation) no longer all-reduces the collision/boundary counts; the training default is unchanged (loss, collision and boundary equal to HEAD to 1e-6 on the fixed two-rank batch).

Five-room check (real CLIs as in `run_checkpoint.sh`, a tiny untrained grid_residual checkpoint): `roomgenbench --requests-from` -> `predict --request --export-dir --device cpu` -> `roomgenbench --handoff --method layout_boxes --require-placement`. All five rooms exit 0; every object is requested and declared (floor / wall / on_object): bathroom 55 (8 / 32 / 15), bedroom 51 (16 / 13 / 22), gym 32 (21 / 5 / 6), living room 53 (14 / 2 / 37), restaurant 122 (50 / 6 / 66); every `place_id` maps back to RoomGenBench's. The old server requests on the same checkpoint: 255 objects, 43 `unknown`, all five rooms rejected by `--require-placement`. With the near-floor fallback the same run hangs 0 of 58 wall objects at z <= 0.15 m (before 33; z 0.15-2.06 m), and `validate_scene` reports `wall_support` pass for all 58. Requests carry no doors (the three-field contract), so wall objects can still cover a doorway.

## 2026-10-08 round 10 (wall and supported-item targets)

The frozen v3.2 prep (`anchors=[floor, object]`, top-surface-only support) dropped every wall object and every item on an inner shelf; the five RoomGenBench rooms kept 228 of their 313 objects (58 wall, 27 on-object missing). The bridge now adds, on top of the unchanged frozen selection (`legacy_bridge.selection_extensions`; `field_evidence.selection_rule`):

- `wall_anchor`: a source wall anchor of `WALL_ANCHOR_SOURCES` (SAGE-10k `place_id` wall, MansionWorld `wall_objects`, SceneSmith `wall_mounted`), declared `support_parent: "wall"` when the boundary is known. IL3D's per-asset `on_wall` flags (most of those boxes stand off the wall) stay excluded; HSSD's `objects.csv` support column never reached the IR. Labels follow the source rules (MansionWorld size stays masked; yaw keeps the axis policy's order 2).
- `support_inside_parent` / `support_on_added_parent`: a source child (not bbox-inferred) whose centre lies on its frozen-prep or added parent's footprint (+5 cm) and whose bottom lies within the parent box (+-5 cm), e.g. books on a bookcase's inner shelves, items on a wall shelf; it declares its source parent. Floating, off-footprint and below-parent children stay dropped.

Frozen-prep targets, rooms, splits and constraints are unchanged; only the `obj_NNNN` numbering of rooms that gain objects moves. `legacy_verify` rechecks the rule from each row (wall declared exactly on `wall_anchor` targets of those sources in a known boundary; an added child declares its source parent, matches its parent's rule and lies on its footprint within its box) and the new manifest counts `wall_declarations_written` / `selection_rule_counts`; `multisource_verify --full-condition` checks that every `wall` declaration carries its `wall_anchor` evidence. The five benchmark rooms now carry all 313 objects with RoomGenBench's `place_id`, still held out in test. Smoke: `legacy_build --include-uid UID` (repeatable) takes a frozen UID past `--max-scenes-per-source`. Known limit: rooms over 128 objects grow (the training preflight drops them whole; raising `max_objects` or splitting rooms is an open decision).

Review repairs (same round):

- **Verifiers see self-consistent bad builds.** `legacy_verify` restates the rule's constants (`WALL_ANCHOR_SOURCES`, `SUPPORT_TOL_M`, the rule names) instead of importing them from `legacy_bridge`, and per row requires exactly as many `frozen_prep` targets as the frozen v3.2 row has placements, no `frozen_prep` target with a source `wall`/`ceiling` anchor, and every `wall`-declared target within `validation.WALL_GAP_M` of the boundary; `multisource_verify --full-condition` checks the last two (its own footprint-to-outline distance). On the 325-room smoke bridge the reviewer's mutations "wall anchors relabelled frozen_prep with their children", "children relabelled frozen_prep" and "childless wall objects moved to the room centre" now fail (before: passed). A builder that silently drops eligible added targets still passes: that needs the source IR and the frozen fixed set.
- **`validate_scene` accepts the source wall gap.** `wall_support` passes when a footprint side lies within `WALL_GAP_M` = 0.1 m (the `against_wall` default; was the 1e-4 m tolerance) of the boundary, for FastFill, the LLM baselines and the RoomGenBench reference check alike; the check records `tolerance_m`. Ground-truth wall objects stand 2-8 cm off the wall.
- **Spread keeps wall and inner-shelf heights.** A declared child whose predicted bottom is more than `inner` = 0.05 m below its parent's top (no frozen-prep child is; every `support_inside_parent` one is) keeps that height inside the parent, which no longer blocks it. An undeclared raised object over no floor-standing object that holds its predicted bottom and within `near_wall` of a wall keeps its predicted height at its free cell instead of dropping to the floor or onto the sofa under it (it is not projected onto the wall: no declaration certifies it). A declared wall object predicted in the swapped axis form (thinner along local Y, local +X along its argmax wall) turns a quarter so the thin side stays on the wall. Oracle decode of the smoke rows (one-hot GT cells, GT size / yaw / z): minimal-projection wall objects keeping z within 2 cm, validation 0 -> 111 of 111, test 2 -> 746 of 765; full-condition added children, validation 11 -> 92 of 92.
- **Hand-off infers `wall`.** `infer_support` proposes `wall` for an undeclared box over 0.15 m above a known floor with a footprint side within `WALL_GAP_M` of a known boundary and no box under its centre; a painting above a plant stays `unknown`.
- **Structured LLM harness lets wall objects hang.** Rule (3) of the instruction describes wall objects and `support_parent: wall`; `structured_problems` drops llm_baseline's "floats ... put it on a surface or the floor" for an object declared on or hung on a wall and reports a declared wall object off the wall. llm_baseline.py (`prompt` / `harness`) stays byte-identical to 510f1e0, so its harness still sends raised wall objects to the floor; both modes still see the three-field projection, which has no support declarations.
- **rotate90 / mirror skip rooms with `openings`** (free-form metadata `_rigid_xy` cannot turn) instead of raising in the DataLoader; no current row has openings.

## 2026-10-08 round 14 (evaluation math audit)

Evaluation and reporting only: training, data build and `llm_baseline.py` are unchanged. Report fields are only
added (existing ones keep their values), except the RoomGenBench reference check, whose schema moves to v2 (below).
Regression tests: `tests/test_round14_eval_math.py`, `tests/test_autorun.py`.

- **Box-preserving wall facing (spread).** The model learns yaw only modulo pi (the yaw loss takes the min over
  {y, y + pi}), but spread turned a floor-standing object whose footprint ended within `near_wall` of a wall to its most
  probable yaw bin within 45 degrees of facing away from that wall: on test about 24% of objects turned, mostly by about
  90 degrees, adding 0.033 rad yaw error. Now the decoded yaw only flips by pi, when the flipped yaw faces away from the
  nearest wall within 45 degrees, and is kept otherwise; the box and its footprint never change (the yaw logits only switch
  the rule on). Declared `wall` objects (round 10) and `keep_yaw` slots are unchanged. Oracle check on 2,000 seeded
  validation rows (validation split only; 16 x 16 one-hot GT cells with exact residuals, GT size and z; the "mod pi"
  yaw head ties the GT bin and its opposite, as a model that learns yaw modulo pi): three-field projection (1,001 rooms,
  12,840 objects), old rule -> new rule, objects whose yaw spread changed 3,186 (24.8%) -> 2,296, off their predicted
  box 875 (579 by about 90 degrees) -> 0, mean yaw error modulo pi 0.098 rad -> 0 (the oracle's own); full condition
  (2,000 rooms, 25,893 objects) 6,250 -> 4,104 changed, 2,112 -> 0 off-box, 0.110 -> 0 rad. With the GT bin alone
  (exact head) the old rule moved 992 / 2,510 boxes and the new one none. 98,748 of the 98,769 yaw-valid validation
  objects have symmetry order 2 or 4, so the half turn costs no yaw error; positions also differ for 1,657 / 3,729
  objects, since the old turns changed footprints. By design the half turn also turns objects that faced the wall on
  purpose (a chair facing a desk against it): in that oracle 347 / 12,840 (three-field) and 820 / 25,893 (full) objects
  left their exact GT facing, invisible to the order >= 2 yaw metrics. Only the nearest wall counts: in a corner, an
  object facing the second-nearest wall is 90 degrees from the flip window and keeps facing it.
- **Rotated supports (spread).** A declared child's "centre on the parent" test and its 5 x 5 parent lattice, and an
  undeclared raised object's "over a floor-standing object" tests, used the parent's axis-aligned bounds; they now work
  in the parent's own frame (unchanged for yaws that are multiples of 90 degrees). Case: a cup declared on a fixed table
  turned 45 degrees, argmax cell inside the table's bounds but off its top: support violation -> pass.
- **Paired baselines.** `baselines_paired` (also per source) is scored only on the requests the model reference scores
  (those with a layout), so on the model's own objects; over-capacity and failed rows used to count only in the
  baselines. `baselines` (every row) stays. `autorun.score` and SUMMARY.md use the paired one; a cached candidate
  report without it is evaluated again.
- **Validator collisions and clean rooms.** `validation.model` / `validation.ground_truth`: rooms, `rooms_with_collision`,
  `collision_pairs` (requested pairs), `fixed_collision_pairs`, `clean_rooms` / `clean_room_rate` (no hard violation;
  unknown checks are not violations) and `clean_rooms_with_hard_unknown` (counted separately), and
  `clean_room_rate_known_only` over the `rooms_without_hard_unknown` (null when every room has one, as in the
  three-field projection, whose undeclared supports are all unknown). All rates are over the rooms with a layout, so
  read them with the no-layout count when two methods differ in it; the ground-truth column is paired. The ground truth is the
  same rooms' labels as written under the same `validate_scene` (`outcomes[*].ground_truth_validation`).
  `collapse.<column>.bev_overlap_rate_iou_gt_0.3_room_mean` weighs every room with a pair equally (the pooled rate is
  dominated by the largest rooms). `failed_requests` keeps its value and is now named for what it is,
  `failed_requests_strict` (no layout, or a hard check not passed, unknown included); `hard_violation_requests` counts
  no layout or a violated hard check only. SUMMARY.md shows all of these beside the IoU > 0.3 overlap, which hides
  collisions of smaller overlap (its table delimiter row has one cell per header cell, as GFM requires to render it).
- **Reference check v2.** `log_size_error` / `yaw_error_rad` are evaluate's joint box-equivalent errors (the same box
  written a quarter turned with sx / sy swapped scores 0; `box_equivalent_*` keep their values), here for every object
  (evaluate uses them only for `size_axis_swap_allowed` objects); the separately minimised ones are
  `log_size_error_marginal_min` / `yaw_error_rad_marginal_min`. Exchangeable requests, evaluate's groups
  (`matching.group_labels`: identical request fields but the id, so the same support parent, and a certified swap), are
  matched to the truth by a Hungarian assignment on bottom-centre distance (`predicted_id`), so a vase is never paired
  with an identical one on another table; the per-id numbers stay as `*_by_id`. The uniform-yaw baseline follows the
  primary yaw error: `uniform_yaw_baseline_error_rad` (per object and pooled) is the exact expected box-equivalent yaw
  error of a uniform yaw on the truth's own box, (m^2 + (pi/2 - m)^2) / pi with m = min(pi/2, pi/4 + c/2) and c the
  sx/sy swap's log-size cost (pi / 8 for a square footprint up to pi / 4); RoomGenBench rooms 0.416-0.515 rad.
  `uniform_yaw_baseline_error_rad_marginal_min` is the former pi / 4, the baseline of `yaw_error_rad_marginal_min`.
- **Square grid.** `serialize_predictions` takes the grid from `math.isqrt` and raises on a non-square cell count.
- **Saved head outputs.** `evaluate --checkpoint ... --save-head-outputs DIR` writes `DIR/<projection>/row-<row>.npz`
  (compressed) per request: `position_cell_logits`, `position_cell_residuals`, `position_normalized` (its z is the
  regressed height), `size`, `yaw_logits`, `yaw_residuals`, `slot_mask`, `ids` and the `condition` JSON; a failure
  before the forward pass (e.g. over capacity) stores `error_type` / `error_message`. Then
  `python -m fastfill.v2.evaluate --from-head-outputs DIR --data ROWS --grid-decode {spread,argmax} [--projection ...] --output OUT`
  re-decodes and scores offline on the CPU with the outcomes of the online path (tested equal for both decodes and
  both projections, an over-capacity row included). Head outputs of another condition or other ids abort the run.
  The saving run also writes `DIR/manifest.json` (its checkpoint, binding, max_length, grid_decode, data and code
  sha256), which the offline report carries as `head_outputs_manifest`. `DIR` and `--output` must be separate
  directories, neither inside the other (checked before evaluating, so no run is lost to the clash).
- **Docs.** "Geometry and exact losses" below now states the training normalisation, the final `position_cell` weight
  (0.5) and the spread default; `autorun.score`'s docstring names its yaw baseline, the uniform-random expectation
  pi / (2 * order), not a constant yaw.

## Main full-condition data and preserved release history

The new main derivative is `outputs/fastfill_v2/multisource-20261006/data`, with
portable package `/Volumes/harddisk/FastFill_v2_multisource_20261006` and server
copy `/home/jovyan/shanliantian/FastFill_v2_multisource_20261006`. Final build
counts, hashes and actual tokenizer eligibility belong to its new manifests and
preflight, not to a historical smoke or single-source pilot.

The original selected 16-family system remains: 11 training-room families expand
to 16 training source tags, SceneSmith and SpatialGen remain evaluation-only,
and three families supply auxiliary inputs. The new main preserves full parent
room/fixed/support/relation conditions and all target numbers. Scan2CAD's
estimated floor is marked unknown and its whole position vectors are masked;
13 targets outside the default size head range receive whole-size masks.
If any member loses complete position labels, all members of that
exchangeable group lose its tag, with logged fixed-identity fallback. IDs,
order, support, relations and targets remain unchanged. Main yaw keeps the 581
inherited valid semantic labels, with no SpatialLM geometric-yaw promotion.
The independent simplified-input view is `data-minimal-reference`; its XY
translation and geometric-axis yaw policy are separate experimental protocols.

The local source audit is in [docs/fastfill-v2-data-audit.md](../../docs/fastfill-v2-data-audit.md).
The historical broad training source is **our selected, frozen `.release/v3.2` corpus**,
derived from the 5 local roots / 16 dataset families and represented by 18 source
names in the frozen IR. Its saved splits contain 160,964 scenes (144,150 train,
8,167 dev, 8,647 test); the existing training flag policy retains 124,843 of the
old training scenes. These are historical selection counts, not new v2 geometry
eligibility counts. The new bridge writes
`data/build/fastfill_v2/selected-v3.2-20261005/`; its `manifest.json` records the
actual migrated splits, field-valid counts and explicit rejection reasons.

Migration reads the frozen release and audit evidence, without redownloading or
rewriting source data. It verifies source/legacy-code/evidence hashes, inherits
each saved scene's train/dev/test assignment (dev becomes validation), and applies
the old flag filter only to training. The converted dataset is staged and exposed
atomically at a new output path. Request IDs and input order are independent of
target geometry; asset/source identifiers remain provenance. Continuous targets
come from the original IR precision, rather than rounded legacy answer text.

Field eligibility is explicit. Recorded window-width evidence corrects new
condition geometry; unresolved oblique-window geometry is rejected. InternScenes
`k0` vertical-axis/front evidence controls position, size and yaw masks. MansionWorld
footprint proxies and padded Holodeck boxes do not provide true local-size labels.
The default axis front policy admits every upright finite source yaw with front
evidence and records its symmetry order (1 = MultiScan semantic front, 2 = axis
mod pi); the strict policy keeps semantic yaw for MultiScan only, masking the rest.
Legacy convention yaw is an explicit alternate policy, not verified semantic front. Old inferred `on` relations are omitted, rather than
turning bbox contact into support evidence. Tilted fixed geometry is rejected;
unsupported requested-object geometry is masked. A scene with no reliable geometry
field is rejected. Thus the historical broad structured corpus contains partial labels;
it must not be described as fully supervised size/position/yaw for every object.

Known room/support evidence is preserved, and missing evidence remains unknown.
For example, selected MultiScan conditions retain `boundary_known=false`,
`floor_known=false` and missing height/contact labels. The runtime refuses commit
when a required check is unknown. Dataset migration does not establish asset
coverage, mesh validity, physical stability or successful room/asset validation.

The final selected build passed an independent streaming audit of **141,341
scenes / 1,787,052 targets**, with no protocol/split/accounting errors:
**124,589 train / 8,137 validation / 8,615 test**. The 316 extra v2 rejections
are recorded individually. Valid complete fields are position 1,732,321;
size 1,224,670; yaw/full geometry 581. Only 57/2/6 scenes have every requested
object fully labeled under the strict policy. The training set has only 523 yaw
labels across 58 MultiScan scenes: choose a coverage/sampling protocol before
production training; uniform 1,000-step development training is insufficient
as an evidence claim. Source/field integrity does not certify all meshes/fronts.

The portable upload bundle is `outputs/fastfill_v2/FastFill_v2_20261006_review3/`.
Its [execution manual](../../docs/fastfill-v2-runbook.md) gives Linux/macOS
checksum, environment, smoke, training, evaluation and runtime commands.
The repository rebuild commands below require the original frozen `.release`
inputs; the portable bundle uses the supplied data and the execution manual.
Historical review3 local tests: **659 passed + 114 subtests**, v2 coverage **87.68%**.

## Simplified-input bbox ablations and historical SpatialLM pilot

Read the [direct bbox guide](../../docs/fastfill-v2-direct-bbox.md) for the exact
request and downstream boundary. The historical XY derivative has 9,601/539/624 scenes,
33,545 objects, all complete geometry labels. It is qualified from SpatialLM,
inherits parent splits, omits source height from condition, and supervises
pi-periodic **bbox axes** (not semantic front). Correspondence is fixed; no
exchangeable groups were reconstructed. Maximum scene size is 26 objects.
The source whitelist was applied before other geometry checks; these counts do
not prove other sources ineligible. This historical pilot is separate from the
new full-condition multi-source main. The new `data-minimal-reference` ablation
retains partial masks from multiple sources rather than demanding complete yaw
from every source, and explicitly keeps physical boundary/floor unknown.

```bash
python -m fastfill.v2.predict \
  --checkpoint "$FASTFILL_MODEL" \
  --request fastfill/v2/configs/direct_request.json \
  --room-size-semantics reference_extent \
  --output outputs/bbox-prediction-new.json \
  --export-dir outputs/bbox-handoff-new --device cuda
```

This writes geometry JSON, RoomGenBench SceneSpec/registry, GLB, SVG and proxy
diagnostics without a catalog or Host. Unknown height/support stays unknown;
the original RoomGenBench fixed scene/site/render harness needs separate scene
registration and height/metadata handling. Export success is not model quality.
Main inference uses `--condition` and its original qualification fields. Commands
below exercise the full-condition trainer and the offline asset-runtime contract.

The [training pipeline walkthrough](../../docs/fastfill-v2-training-pipeline.md) explains data, each optimizer step, deployment artifacts, and the RoomGenBench task boundary. review2 withdraws unsupported source-derived `faces` while retaining UID/splits/targets/masks; it adds below-floor diagnostics without rewriting labels. The older bundle remains a historical snapshot.

## Run locally

Run commands from the OptiScene repository root. Choose a **new output path**
for each build/run. No CLI is allowed to write below
`/Volumes/harddisk/3D_Room_Collections`. Runtime CLI produces a new report file;
training/evaluation produce new directories.

```bash
python -m pip install -r fastfill/v2/requirements.txt
python -m pip install pytest pytest-cov

python -m fastfill.v2.audit \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/audit-new.json

python -m fastfill.v2.data \
  --source selected-v3.2 \
  --release-root .release/v3.2 \
  --evidence-root .release/audits/2026-09-28/source-check \
  --front-policy axis --workers 4 \
  --output data/build/fastfill_v2/selected-v3.2-new
```

`selected-v3.2` is the default `data` CLI source. The dated primary build path is
`data/build/fastfill_v2/selected-v3.2-20261005/`; use a fresh suffix for reproduction.
Four workers preserve the same deterministic output order as one worker, with
bounded conversion submissions. No existing output is overwritten.

A reproducible complete-label comparison cohort has also been built at
`data/build/fastfill_v2/selected-complete-v3.2-20261006/` (57/2/6 scenes).
Each retained JSONL row is byte-identical to its parent row and retains its split.
To reproduce into a fresh directory:

```bash
python -m fastfill.v2.cohort \
  --data-root data/build/fastfill_v2/selected-v3.2-20261005 \
  --output data/build/fastfill_v2/selected-complete-new
```

Use this same cohort for both structured and text controlled comparisons.
Text training checks complete labels before loading the backbone.

The local complete-label smoke subset at
`outputs/fastfill_v2/selected-full-geometry-smoke-20261005/` contains **57 train /
2 validation / 6 test** scenes from default-kept selected MultiScan, with one
incomplete-label scene excluded. It exercises every geometry head and allows a
text/structured plumbing comparison on the same labels. It is a smoke subset,
not the primary corpus or a completed model-quality benchmark.

For an offline optimizer/checkpoint smoke:

```bash
python -m fastfill.v2.train \
  --config fastfill/v2/configs/smoke.json \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/train.jsonl \
  --output outputs/fastfill_v2/selected-all-heads-smoke-new \
  --dry-run --max-samples 2

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-all-heads-smoke-new/model \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-all-heads-eval-new \
  --max-length 32768 --device cpu

python -m fastfill.v2.text_sft \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/train.jsonl \
  --output outputs/fastfill_v2/selected-text-smoke-new \
  --backbone tiny --dry-run --batch-size 1 --max-length 32768

python -m fastfill.v2.evaluate --baseline text \
  --checkpoint outputs/fastfill_v2/selected-text-smoke-new \
  --data outputs/fastfill_v2/selected-full-geometry-smoke-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-text-eval-new \
  --device cpu --max-length 32768 --max-new-tokens 4096 --max-samples 1

python -m pytest fastfill/tests fastfill/v2/tests \
  --cov=fastfill.v2 --cov-config=fastfill/v2/.coveragerc --cov-report=term-missing -q
```

The explicit `tiny` backbone is a causal byte-token GRU for offline plumbing,
never a Qwen replacement in experimental conclusions. Tests also instantiate
local randomly initialized Qwen2 + PEFT models without downloading weights, to
verify LoRA/full/frozen gradient paths and checkpoint round-trips.

The raw-drive adapter remains available explicitly for an optional audit/smoke
build. It creates separate house-grouped splits and **does not replace or inherit
the selected v3.2 corpus**:

```bash
python -m fastfill.v2.data --source multiscan \
  --source-root /Volumes/harddisk/3D_Room_Collections \
  --output data/build/fastfill_v2/multiscan-audit-new
```

## Train Qwen and evaluate

The reviewed data is published at
[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2)
(public + gated with manual approval since 2026-10-07 by the user's decision; private at each earlier release), fixed
commit `96f4946624b46bf8dc99bf94311b5d31290ea09a`, snapshot `review3-20261006`.
Use the [dataset guide](../../docs/fastfill-v2-dataset-release.md) to download and
verify the main masked corpus and the complete-label pilot cohort. Raw v1 SFT
messages are not structured v2 samples.

`configs/structured.json` chooses `Qwen/Qwen2.5-0.5B-Instruct` as a development
default, with LoRA, 128-dimensional two-layer decoder, 12 yaw bins, all four base
weights 1, and no scene/box loss. The selected production backbone is now
**Qwen3-8B** on the user's H20Z server (`ssh yxd-dev`). Override the frozen
development template with a separate runtime configuration; the server model
path is `/home/jovyan/shanliantian/models/Qwen3-8B`. See the
[server instructions](../../docs/fastfill-v2-server-start.md) for BF16 preflight
and the bounded 20-step pilot; `configs/qwen3_8b_pilot.json` provides the selected
8B pilot settings. Freeze sampling, context and the full training
budget after pilot measurements; the template's default budget is not validated.

```bash
python -m fastfill.v2.train \
  --config fastfill/v2/configs/structured.json \
  --data data/build/fastfill_v2/selected-v3.2-20261005/train.jsonl \
  --validation data/build/fastfill_v2/selected-v3.2-20261005/validation.jsonl \
  --output outputs/fastfill_v2/selected-qwen-structured-new

# Optional multi-GPU launch using the same configuration and fresh run path:
accelerate launch -m fastfill.v2.train \
  --config fastfill/v2/configs/structured.json \
  --data data/build/fastfill_v2/selected-v3.2-20261005/train.jsonl \
  --validation data/build/fastfill_v2/selected-v3.2-20261005/validation.jsonl \
  --output outputs/fastfill_v2/selected-qwen-distributed-new

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --data data/build/fastfill_v2/selected-v3.2-20261005/test.jsonl \
  --output outputs/fastfill_v2/selected-qwen-eval-new --device cuda
```

Tune dtype/batch/context/budget for the selected machine in a new config. The
defaults are development starting points, not measured optimal settings.
An unrestricted structured run currently loads the train and validation JSONL
samples into memory; plan host RAM against the migrated corpus size. Structured
`--max-samples` bounds a streamed training prefix for smoke runs, not a complete
randomized training subset. Text SFT also loads its input JSONL into memory.

A full text SFT comparison uses the materialized 57/2/6 complete-label cohort
with complete size/position/yaw labels for every requested object and inherited
splits. Its builder and artifact are provided above. Select shared context/object
eligibility and training budgets for the formal research protocol, then use that
**same cohort for the controlled structured comparison**. Comparing text trained
on complete labels against structured training on all partial-label scenes would mix the
training objective with data coverage. The full masked structured run is a
separate experiment whose scope and eligible-label counts must be reported.

Structured training records complete-sample context rejection, source hashes,
implementation hashes, package versions, unweighted losses, effective global
counts, head/backbone gradient norms and trainable parameter count. Validation
loss is a mean-of-batches diagnostic; final all-request metrics come from
`evaluate.py`. Every checkpoint writes an Accelerate `state-step-<n>/` and a
deployable `model-step-<n>/`; `--resume <state-step-n>` continues into a new
output directory (see the audit-fixes section). Deployment checkpoints contain
backbone adapter/full weights where needed and decoder/heads.

## Geometry and exact losses

- Right-handed Z-up meters. Local +X is the canonical semantic front for the main
  protocol; sources without front evidence have yaw masked. The separate
  geometric-yaw ablation does not certify semantic front. `(w,d,h)` are full
  X/Y/Z lengths, invariant under yaw. Positions are bbox bottom-center.
- Position normalization uses input room origin and XY extent; missing height
  uses fixed 3m. Trusted floor-support z and requested fixed size dimensions are
  used directly. Unconstrained head tensors are retained separately for diagnosis.
- `size = positive_reference * exp(clamp(u,-10,10))`; reference is fixed `(1,1,1)`
  meters, with no validation/test statistics. Head residuals use `tanh` by default.
- L1 by default (`loss.position_type` / `loss.size_type`; `smooth_l1` with
  `loss.smooth_l1_beta` selectable) for normalized bottom-center and log size ratio. Each term uses
  only complete valid field vectors and at least one learned coordinate, then
  sum learned coordinate losses /3 / valid instance count. Missing labels are
  filtered before arithmetic. Fixed dimensions do not reduce the `/3` denominator.
- Yaw CE + SmoothL1 on **GT-bin** residual in training; argmax predicted bin in
  inference. Explicit justified symmetry orders choose the same equivalent angle
  by the joint weighted CE/residual minimum. Box symmetry never implies semantic
  front symmetry. Detached group cost uses position L1 (grid_residual head: the
  loss's own per-pair position terms, round 4), plus log-size L1 only when
  every group member has a complete size label; no global
  matching, detection classification, objectness, NMS or repeated GT.
- `model.position_head = grid_residual` (round 2): logits over
  `position_grid` x `position_grid` cells (default 16) of normalized XY plus a
  tanh XY residual per cell in half-cell units; training uses CE on the GT cell +
  L1 on the GT cell's residual (`loss.position_cell` / `loss.position_residual`
  relative weights; the final configuration `configs/qwen3_8b_main_3gpu_grid_cell05.json`,
  which the autopilot retrains as `main7-cell05-*`, uses **0.5** / 0.4; the first formal
  grid configs used 0.04 / 0.4, and `position_cell` 0.2 and 1.0 were also compared) + the regression z share |dz|/3.
  The raw head decodes the argmax cell + its residual (`predictions["position_normalized"]`,
  `--grid-decode argmax`, "the model alone"); the **default delivery is spread**:
  `predict`, `evaluate` and the autopilot's selection and test reports use
  `--grid-decode spread`, the collision-aware re-decode `evaluate.spread_grid_xy`
  (post-processing of the same head outputs; see round 14 below for its wall-facing rule).
  `predictions["position_normalized"]` is always decoded, so matching, regularizers, evaluation and hand-off are
  head-agnostic. `term_sums["position"]` is the decoded-position L1/3 under both
  heads; `position_cell` / `position_residual` / `position_z` are logged as well.
- Box symmetry (round 2): for `validity.size_axis_swap_allowed` objects the loss
  picks the detached joint minimum of size + yaw CE + yaw residual over
  k in {0..3} (yaw + k pi/2, size xy swapped for odd k); a fixed sx or sy
  (a fixed sz alone pins nothing) pins the written order and leaves yaw only the box's pi symmetry (order 2); matching uses min(cost(sx,sy), cost(sy,sx)) per pair.
  `loss.yaw_reg` is capped at 2.0 in the main configs
  ([calibration, round-2 section](../../docs/fastfill-v2-loss-calibration-20261006.md)).

**Training normalisation.** Each loss term of one optimizer update is a **global
per-microbatch mean averaged over the accumulation window**, not a per-object mean
over the window. Per microbatch, `losses._mean` divides the term's sum by its valid
count all-reduced over the ranks (times the world size, which DDP's gradient
averaging divides out again), so the microbatch term is the mean over the valid
objects of that global microbatch. Accelerate then divides every microbatch loss by
`gradient_accumulation_steps` K (an epoch-tail window of m < K microbatches is
rescaled by K / m in `train._rescale_flushed_window_gradients`), so the update follows
the plain mean of the K microbatch means: an object in a microbatch with few valid
labels weighs more than one in a dense microbatch, and a microbatch with no valid
label of a term contributes 0 to that term's average. The update is skipped only
when the whole window holds no enabled objective on any rank. Window logs
(`term_sums` / `term_counts`) are count-weighted per-object means, a different
reduction from the one optimised.

Optional `box_operator=bev_oriented_giou_convex_hull` is piecewise differentiable
oriented rectangle intersection with convex-hull enclosure of both boxes.
Vertices are calculated in a common local float64 frame to protect small valid
boxes from global-coordinate cancellation. Intersection/hull topology decisions
are detached; position/size/yaw tensors retain gradients within topology regions.
Contacts/coincident edges are nonsmooth; numerical-gradient tests use
nondegenerate boxes. BEV does not supervise height/z. GIoU and true BEV IoU have
separate functions/metrics. This implementation is correctness-oriented and
Python-loop based; benchmark before large-batch GPU use. No 3D GIoU is claimed.

Collision/convex-room boundary regularizers are separate flags, default zero.
Unknown/concave boundary input is refused by the boundary training operator.
Declared support pairs are exempt from volume regularization; runtime contact
still needs actual evidence. These losses do not replace runtime validation.

## WorldEdge contract and offline asset loop

```bash
python -m fastfill.v2.predict \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --condition REQUEST.json --output outputs/fastfill_v2/prediction-new.json \
  --device cuda

python -m fastfill.v2.serve \
  --condition REQUEST.json --prediction PREDICTION.json --catalog CATALOG.json \
  --output outputs/fastfill_v2/runtime-new.json \
  --max-asset-retries 2 --max-repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory

python -m fastfill.v2.evaluate \
  --checkpoint outputs/fastfill_v2/selected-qwen-structured-new/model \
  --data EVALUATION.jsonl --catalog CATALOG.json \
  --output outputs/fastfill_v2/asset-eval-new --device cuda \
  --asset-retries 2 --repair-calls 2 --repair-step-m 0.25 \
  --commit-in-memory
```

Catalog records carry `ref`, category, actual local size, a raw-to-canonical
4x4 transform, optional semantic front, capability names and verified horizontal
support polygons with local heights. `retrieval_tolerance_log` (or legacy alias
`retrieval_tolerance`) is scalar/three-vector of maximum absolute log-size ratio.
Category/capability/fixed/range checks precede size ranking. This deterministic
catalog example does not implement nuanced natural-language or learned asset
retrieval. Nonempty typed `attributes` requests are rejected because the current
Asset contract has no verified attribute evidence; actual validation reports
hard `attributes_unverified`. An external resolver alone cannot bypass this:
support requires an explicit asset evidence contract and a matching validator.
Description text is not verified proof of color/material or other attributes.
Known semantic-front metadata must already point along canonical +X. A +Y-front
asset must first normalize its transform, local size axes and support polygons;
the runtime rejects that unnormalized metadata. Missing front remains unknown.
Catalog JSON accepts an array or `{"assets": [...]}`. Asset provenance is a
list of source references.

Target size/pose, first-pass actual objects and final repaired actual objects are
retained separately. No asset shrink/delete occurs. Reconciliation applies
`T(bottom_center) @ Rz(yaw) @ canonical_transform`; actual support surfaces, not
bbox tops, update dependent child heights. Translation repair moves support
descendants consistently, is bounded, and revalidates actual geometry.

Runtime geometry uses upright OBBs and polygon checks. Unknown required support,
boundary or semantic front blocks commit. Nonempty openings without a verified
clearance converter produce hard `openings_unchecked` diagnostics. Actual
capabilities are rechecked after asset filtering. Conservative OBBs can reject
valid contents of hollow furniture or shelves; this initial validator does not
infer cavity geometry or bypass penetration checks by furniture category.
Mesh/physics/Solver are explicitly
unavailable: the current Validator unconditionally emits unknown for each, with
no connected checker or external evidence-ingestion interface. Requiring these
validation levels therefore blocks commit. Time budgets
are checked between callbacks; external callbacks must enforce their own I/O
timeouts. `AtomicMemoryHost` demonstrates all-or-fail versioned/idempotent commit.
**Real WorldEdge persistent Host integration requires a concrete atomic adapter**;
this repo exposes the protocol and does not mutate a remote or existing world.
The existing sibling WorldEdge resolver's `resolve(FloorObjectSpec)` contract is
for v1 and cannot silently substitute for v2 target-before-asset resolution.

Evaluation always retains failed requests in schema/ID/validity and system-rate
denominators. Valid-label reference means disclose their eligible-object counts.
Malformed text JSON, missing assets and incomplete validation have distinct
diagnostics. Latency is p50/p95; raw model, asset-first-pass and final stages are
logged separately. Bbox acceptance does not establish task/physics success.
Operational end-to-end latency adds model generation and asset/runtime time,
excluding reference scoring and matching. Supplied-prediction evaluation reports
runtime latency and leaves end-to-end latency unknown; evaluation wall time is
recorded separately. Incomplete labels in anonymous groups use diagnosed fixed
reference correspondence; ineligible reference scores never change model-schema
success or skip the asset loop. `--required-levels bbox mesh` in either CLI
requests mesh evidence and blocks offline commit while that evidence is unknown.

## Required comparisons

1. Preserve existing v1 as an explicitly **bbox-conditioned oracle-size** reference
   using its own existing entry point. It has richer size information; label that
   difference rather than reporting it as the same hidden-size task.
2. Train v2 text SFT and its controlled structured counterpart on the same
   full-condition complete-label view, condition information and documented
   sample exposure. Current text SFT explicitly rejects incomplete labels;
   do not silently erase the main corpus's partial masks. Simplified-input
   comparisons use separate manifests. Freeze asset/repair budgets for the
   main downstream runtime evaluation.
3. Train structured fixed correspondence with `configs/structured_fixed.json`.
4. Train structured exchangeable-group Hungarian with `configs/structured.json`.
5. If useful, compare `configs/structured_bev_box.json` at equal training budget.

No such quality comparison has been run. Raw/log size, sin/cos yaw, additional
matching cost and scene regularizers remain separately selected future ablations,
not claimed improvements. No V-DETR default weight is claimed optimal.

Official [V-DETR criterion](https://github.com/V-DETR/V-DETR/blob/main/criterion.py)
and [main](https://github.com/V-DETR/V-DETR/blob/main/main.py) were consulted only
for matcher/criterion separation and angle supervision. The implementation here
is specific to requested layout generation. Point-cloud encoders, 3DV-RPE,
one-to-many, large detection query banks and MinkowskiEngine are not dependencies.

## Recorded local validation — 2026-10-05

The selected-corpus structured smoke used two complete-label training scenes and
one optimizer step with the explicit tiny backend. Its checkpoint evaluation
completed the **6 selected test scenes / 37 requested objects**, with schema,
ID and positive-size success for each. Target-geometry validation was **0%**;
unknown boundary/support evidence remains a blocker and usable layouts have not
been demonstrated. Reports are at
`outputs/fastfill_v2/selected-all-heads-smoke-20261005/run_manifest.json` and
`outputs/fastfill_v2/selected-all-heads-eval-20261005/report.json`.

The selected text smoke performed a token-CE optimizer step on the same
complete-label training cohort. Its bounded one-request generation check failed
JSON/schema parsing. Report:
`outputs/fastfill_v2/selected-text-eval-20261005/report.json`. These plumbing runs
are not a controlled quality comparison. A separate single-fixture overfit test
checks decreasing geometry loss and gradients to all heads.

Offline regression tests include immutable release/hash checks, inherited splits,
train-only flag filtering, atomic migration failure and deterministic one/two-worker
output equivalence, together with geometry/model/runtime checks. Final suite and
coverage totals are recorded by the final verification run; historical coverage
artifacts do not describe the entire repository.

The server and Qwen3-8B checkpoint have been chosen and the historical pilot ran.
Full training exposure/budget and quality comparisons remain unvalidated.
No quality gain or actual persistent-world success has been measured.

## LLM comparison protocol

FastFill and every LLM mode answer the same requests from the same fields and are scored by one scorer.

- **Same rooms.** The first 300 three-field rows of the fixed validation sample: autorun's `validation-<sha12>-3000.jsonl`
  (validation.jsonl shuffled with `random.Random(0)`, first 3000 lines), of which `llm_baseline --max-samples 300` keeps the
  first 300 that `evaluate.project_minimal` accepts. Today's run wrote them to `<runs>/llm-prompt-300/rows.jsonl`; every
  further method reads that file as `--data` (the projection is idempotent). Check: the new run's `rows.jsonl` is byte-identical
  to `llm-prompt-300/rows.jsonl` and its summary.json `data_sha256` (of `--data`) equals `sha256(llm-prompt-300/rows.jsonl)`. (Today's
  summaries record the hash of the 3000-line file they read, so the two `data_sha256` values differ by construction.)
- **Same input fields, nothing else.** Room type, room size (width x length x height, floor polygon, floor z) and the
  furniture list (id, category, description; `support_parent` only when the request declares one, which three-field rows
  never do). No sizes, priors, positions or statistics from data. FastFill gets exactly this projection
  (`batch.render_minimal_condition`). The structured modes' template (`llm_structured.STRUCTURED`: instruction + one fixed
  train demonstration, recorded as `prompt_sha256`) is identical for every request; only the `[Task ...]` sections change.
- **Modes.** `llm_baseline` `prompt` and `harness` (today's runs) and the OptiScene-style `llm_structured` `structured` and
  `structured-harness` (its docstring). The structured-harness checks use only the request and the answer: parse errors,
  ids not exactly once, non-finite or non-positive numbers, below the floor, above the ceiling, a declared support not met,
  beyond a wall (5 cm margin), overlapping footprints of objects sharing a height interval (over 15% of the smaller one,
  as in `harness` and spread decoding) and a raised object with nothing under it. No category size priors.
- **The checks are not a quality score.** The ground truth itself fails them: on the 300 rows,
  `llm_structured.structured_problems(row["target"], row["condition"])` finds 482 problems in 100 rows (1.61 per row):
  448 overlaps in 93 rows (chairs tucked under tables), 32 objects beyond a wall in 20 rows, 1 above the ceiling, 1 below
  the floor, no floating object. The harness pushes answers towards zero of them, the ground truth does not; quote this
  reference line next to any `checks_final`.
- **Repairs <= 2** (`--repairs 2`, recorded as `repairs` in the structured summaries). In every mode an answer that is
  still no valid layout is asked again from scratch once.
- **One scorer, failures counted.** `python -m fastfill.v2.evaluate --data <out>/rows.jsonl --predictions
  <out>/predictions.jsonl --output <dir>`. A row without a layout stays in predictions.jsonl with `layout: null` and counts
  as no layout; summary.json reports `failed`, `unanswered` and `first_answer_invalid`. The harness diagnostics stay out of
  the comparison table: `llm_baseline`'s `problems_*` count `problems()` (capped at 40) on valid layouts, `llm_structured`'s
  `checks_*` count the checks above, uncapped, and for an unreadable first answer its parse or number errors.
- **Cost.** summary.json reports `api_calls` (chat calls, each with its own HTTP retries; a call that raised counts too)
  and `mean_latency_s`; the structured summaries add `usage` (summed tokens of the calls whose response reported them, their
  number in `calls_reported`: a call that raised or a response without usage is not among them).
- **FastFill raw and spread apart.** The best checkpoint on the same rows.jsonl with `--projection full --grid-decode argmax`
  (the model alone) and with `--grid-decode spread` (collision-aware post-processing) are two separate rows of the table.

```bash
python -m fastfill.v2.llm_structured --data <runs>/llm-prompt-300/rows.jsonl --env <api env> --mode structured-harness \
  --repairs 2 --max-samples 300 --output <runs>/llm-structured-harness-300   # likewise --mode structured
cmp <runs>/llm-structured-harness-300/rows.jsonl <runs>/llm-prompt-300/rows.jsonl
python -m fastfill.v2.evaluate --data <runs>/llm-structured-harness-300/rows.jsonl \
  --predictions <runs>/llm-structured-harness-300/predictions.jsonl --output <runs>/llm-structured-harness-300-eval
```

The structured modes live in `llm_structured.py` so that `llm_baseline.py` stays byte-identical to 510f1e0
(`test_llm_structured`): autorun reuses today's `llm-prompt-300` / `llm-harness-300` only while `sha256(llm_baseline.py)`
matches their summary.json, and otherwise deletes them and pays for new answers, which differ (the reasoning model runs
without a seed or temperature). Any new `fastfill/v2/*.py`, this module included, changes evaluate's `implementation_sha256`,
so a restarted autopilot re-scores the `-eval` reports (evaluate only, no API calls). Back up the `llm-*-300` directories
before any edit of `llm_baseline.py`.

## 2026-10-08 status of the final version

- Baseline model: `main7-cell05-main-20261007b-e5/model-step-6357` (HF private `liantian/fastfill-v2-models`), data
  `main-20261007b`, 7 GPUs × 5 epochs. Test, 4455 three-field rooms: log-size error 0.296 (category median 0.394),
  yaw error 0.673 raw / 0.700 spread (uniform random 0.783), bottom-centre error 2.643 raw / 2.663 spread m (room
  centre 2.264 m; single-reference position error is minimised by central placements and cannot rank layouts alone).
- LLM agents (gpt-6.1-sol, reasoning effort medium; prompt, harness, OptiScene-style structured, structured+harness)
  were each run once on the same frozen 300 validation rows with the same fields as FastFill and re-scored with the
  current evaluator (`runs/llm-*-300-eval-4125b82`). On those rows FastFill's size error is about 0.24 vs 0.39–0.41
  for every LLM mode (≈ category median); LLM yaw is at uniform-random level; position differences between methods
  were not significant. These 300 rows were part of FastFill's checkpoint-selection cohort (not held out for FastFill).
- Data `main-20261008a` (round 10, commit 155a123): adds wall-mounted targets with explicit `wall` support where the
  footprint touches the boundary, and source-declared supported items; training targets 1.58M → 2.00M; both verifiers
  pass. Final candidate `main7-cell05-main-20261008a-e5-yawcls05` (max_objects 256, yaw_cls 0.5) started 03:49 UTC;
  a `yaw_cls 0.08` control on the same data follows, so the yaw-weight effect is attributable.
- Final analysis tooling: `fastfill/v2/ops/` (decisions on validation only; test report-only).
