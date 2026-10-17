# FastFill v2 training-path audit — 2026-10-06

This is the current-tree audit for the assigned training/model/loss/matching/batch/geometry/evaluation modules. Every production file listed in `scope-manifest.json` was read in full. The audit did not run formal Qwen training, server GPUs, assets, physics, Solver, or a real Host. Two newly confirmed defects were fixed under explicit ownership; no dataset or frozen package was changed.

## Concrete findings

### Fixed — P2 — FP16 overflow was counted as a completed optimizer step

Before the fix, `train.py` incremented `step`, constructed a successful training record and later allowed validation/checkpointing before `optimizer.step()`. Accelerate's wrapped optimizer can deliberately skip `optimizer.step()` after GradScaler detects nonfinite gradients. A finite scalar loss with coefficient `1e38` and CPU GradScaler produced an overflowing scaled derivative: the actual optimizer recorded zero calls, but the old loop logged `step=1` with a NaN gradient norm and stopped.

The new regression first failed with `observed=[]` instead of the expected finite second-window optimizer call. The loop now calls the wrapped optimizer first, checks `accelerator.optimizer_step_was_skipped`, and advances/logs/validates/checkpoints only after a real update. The manifest separately records `skipped_gradient_overflow_windows`. An epoch in which every eligible attempted update overflowed fails without publishing a completed-step claim. BF16/no-scaler behavior and the declared accumulation mean are unchanged.

Regression: `test_amp_overflow_does_not_count_as_completed_optimizer_step`.

### Fixed — P2 — one incomplete exchange group disabled legal matching for every group in evaluation

`evaluate.reference_metrics` previously set `use_hungarian=False` globally when any exchangeable group lacked complete position/size cost labels. A four-object counterexample contained a complete group whose predictions were exactly swapped and an unrelated incomplete group. The old metric reported mean bottom-center error `1.0 m`; group-local policy requires the complete group to match and the incomplete group alone to use fixed identity, producing `0.0 m`.

Evaluation now removes only incomplete groups from the candidate graph. Remaining groups still pass the production `match_batch` constraint/support-role certification before detached Hungarian assignment. Schema success, request denominators, runtime execution and the main qualified dataset are unchanged.

Regression: `test_incomplete_group_does_not_disable_matching_for_other_complete_group`.

## Review2 C1–C4 status

- C1 accumulation tail: fixed. A flushed `m<K` tail multiplies accumulated gradients by `K/m` before Accelerate unscale/clip. Empty-label microbatches remain in the explicitly declared microbatch-mean denominator. Actual one-rank and two-rank complete/tail regressions pass.
- C2 active supervision: fixed. Preflight uses enabled loss weights, complete-field validity and fixed-coordinate masks. Runtime aggregates eligible objectives across the entire accumulation window and all ranks; a globally empty window skips weight decay and does not increment the update count.
- C3 exchange certification: fixed. Only explicit schema reference fields are renamed. Category, description, attributes, dictionary keys, group names and surface-local IDs stay in their own namespaces. Constraint/support invariance is re-certified for every matched group.
- C4 predict exit code: fixed. Runtime closure failure is saved and returns status 2 through `raise SystemExit(main())`; raw bbox success and successful asset closure return 0.

## Loss and distributed math

For a term with local differentiable sum `T_r`, world size `W` and all-reduced valid-object count `C`, `_mean` returns `W*T_r/C`. DDP averages parameter gradients across ranks, yielding `sum_r grad(T_r)/C`. Ranks with zero terms participate in the same count reductions and graph-connected zero. This is correct for the documented global valid-instance mean per microbatch.

Accelerate divides every backward loss by configured accumulation `K`. A complete window therefore averages K microbatch objectives. For a loader tail with m microbatches, the explicit `K/m` gradient rescale restores the mean over the actual tail. The rescale occurs while gradients may still be AMP-scaled and before unscale/clip. Window eligibility is reduced across all ranks. After the overflow fix, only a successful wrapped optimizer call advances the completed-update schedule.

The default four structured terms implement:

- room-normalized bottom-center SmoothL1;
- log predicted/target size SmoothL1;
- yaw-bin cross entropy;
- GT-bin normalized yaw-residual SmoothL1.

Position and size require a complete label vector, select non-fixed learned coordinates before arithmetic, divide the coordinate sum by three, and divide globally by valid objects. This deliberately gives fixed coordinates zero contribution while preserving the documented fixed three-coordinate denominator.

For yaw symmetry order `n`, the criterion constructs `yaw + k*2pi/n`, computes CE and residual loss for each candidate, and takes a detached argmin of the weighted joint CE+residual cost. Gradients then reach the selected class logits and selected GT-bin residual. This is mathematically consistent for the enabled default `yaw_cls=1, yaw_reg=1`. The optional box/collision/boundary terms decode through a discrete argmax and therefore do not ordinarily train yaw logits; configurations that disable yaw CE while expecting those terms to learn the discrete yaw bin are not equivalent to the default protocol.

Hungarian cost is detached normalized-position L1 plus log-size L1 and runs only inside groups whose conditioning graph is invariant and whose position/size cost labels are complete. Fixed identities are unchanged.

## Model/batch/geometry conclusions

- The model reads condition tokens only. Request slots are bound to exact object token spans, then interact bidirectionally and cross-attend the full condition memory.
- Slot prefix pooling accumulates in at least float32, avoiding the reviewed BF16 cancellation and FP16 prefix overflow problem.
- The size exponent executes in float32 with configuration-validated finite positive bounds. Training preflight rejects learned supervised axes outside the configured output range instead of clamping labels.
- Floor-supported known-Z and exact fixed-size request coordinates are enforced as fixed output coordinates; contradictory demonstrated geometry fails batching.
- The BEV polygon operator and scene regularizers are piecewise differentiable proxies, not mesh/physics certification. Optional box loss is BEV convex-hull GIoU, not full 3D GIoU.
- `qualified_data.py` preserves target numbers, masks D1/D2 evidence, marks Scan2CAD floor as estimated/unknown, and removes exchangeability from groups losing complete matching cost labels. This audit did not repeat the separate full-data provenance scan.

## Text SFT comparison status

The code provides a valid assistant-token CE baseline with the exact same serialized condition prefix as the structured model. It intentionally rejects any sample without complete position, size and yaw for every request. Therefore the full partial-label main train file cannot be passed directly to text SFT.

A fair quality comparison has not been run. It must freeze the complete-label cohort and use that same cohort for a controlled structured run, the same Qwen3 base/revision and condition profile, shared tokenizer/context/object eligibility, comparable LoRA/optimizer budget, and explicitly equalized sample exposure. The current structured loader uses shuffled epochs while text SFT samples with replacement, so equal nominal optimizer steps do not by themselves give equal examples. Text SFT also lacks the structured CLI's DDP accumulation and validation schedule; wall-clock or step-only comparisons would mix training harness differences with model form. The all-partial-label structured run remains a separate experiment.

The generic `structured.json` and text CLI defaults are development Qwen2.5 settings. The frozen seven-GPU main candidate explicitly supplies the Qwen3-8B B1/K16 configuration; omitting `--config` would not run that formal experiment.

## Test and coverage evidence

- RED overflow regression: failed because no optimizer call occurred while old code logged completed step 1.
- GREEN overflow regression: 1 passed.
- RED mixed-group metric regression: failed with `1.0 != 0.0`.
- GREEN mixed-group regression plus existing incomplete-group runtime regression: 2 passed.
- Related training/model/loss/matching/evaluation suite: 277 passed in 34.06 seconds.
- Direct bbox/handoff/visualization suite: 60 passed in 2.74 seconds.
- Combined targeted statement coverage for the 15 assigned production files: 1,493/1,699 = 87.88%, rounded to 88%. Per-file results and exact current SHA256/line counts are in `scope-manifest.json`; machine coverage is in `targeted-coverage.json`.

These targeted results supplement, rather than rename, the earlier 873-passed server suite. A repository-wide post-fix suite and server sync are owned by the release coordinator.
