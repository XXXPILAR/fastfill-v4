# FastFill

FastFill v2 predicts one local bbox size, bottom-center position and yaw for each
requested object from **room geometry + object requests + fixed objects +
available support and relation constraints**. The main data protocol keeps the
original selected multi-source collection, full conditions and partial
field-validity masks. Continuous heads supply structural geometry supervision;
Hungarian matching is restricted to legal exchangeable groups. Runtime then
reconciles predictions with real assets and validates the whole scene before an
atomic Host commit. The implemented memory Host and bbox checks do not certify
real mesh, physics, Solver or WorldEdge Host operation. Concrete checkers and a
persistent WorldEdge Host adapter remain integration work; the current Validator
reports those geometry levels unknown and blocks when they are required.
See the [training pipeline](docs/fastfill-v2-training-pipeline.md). The separate
[direct bbox guide](docs/fastfill-v2-direct-bbox.md) describes a simplified-input
ablation that exports geometry without assets.

The downstream mesh interface is **RoomGenBench**. Reference repositories are
pinned Git submodules; initialize their source checkouts with
`git submodule update --init V-DETR MinkowskiEngine RoomGenBench`.
Their cloned source and papers are reference material, not proof that the old
point-cloud detector has been built in the FastFill CUDA environment. Read the
[design corrections](docs/fastfill-v2-design-audit-20261006.md) and
[reference source audit](docs/fastfill-v2-reference-code-audit-20261006.md).

The preserved FastFill v1 path takes a floor polygon, optional room type/height,
fixed obstacles, objects with known geometry and optional relations, and predicts
position, yaw and support parent. The v1 commands below retain that protocol.

This is the current implementation for [Slian22/fastfill](https://github.com/Slian22/fastfill). The repository uses `main` as its development branch. Its Python package is `fastfill`; commands run from the repository root. The earlier `src/fastfill_train` Floor/Surface implementation remains in Git history and is a different protocol.

## Code and experiment versions

| Item | Version and purpose |
|---|---|
| Current v2 main experiment | Qwen3-8B + bidirectional object decoder; full-condition multi-source joint geometry prediction and asset-runtime validation |
| v2 simplified-input ablation | Three request fields with explicit reference extents; independent bbox export and downstream comparison |
| Preserved v1 implementation | v3.2 data-pipeline, text SFT, evaluation and service |
| Preserved v1 constraint-supervision experiment | Paired **v3 / v3.1** datasets; keep the same evaluation inputs and training configuration |
| v3.2 dataset | Separately rebuilt constrained dataset with source geometry corrections and recovered rooms |
| Model text protocol | FastFill v1: `placements`, `pos`, integer-degree training targets for `yaw`, optional `on` |

Dataset version numbers are not model architecture or training-stage numbers. v3.2 changes geometry and rows, so comparing v3 with v3.2 is not a constraint-only ablation. Existing training runs should retain their original data and code snapshots. See [version and repository map](docs/versions.md) and [known limitations](docs/known-issues.md).

## Data

Data is maintained separately at [liantian/fastfill-v3](https://huggingface.co/datasets/liantian/fastfill-v3). Repository access and upstream dataset terms continue to apply.

FastFill v2 uses the separate **private** dataset
[liantian/fastfill-v2](https://huggingface.co/datasets/liantian/fastfill-v2), historical snapshot
`review3-20261006`, fixed commit `96f4946624b46bf8dc99bf94311b5d31290ea09a`.
The historical review3 splits contain 124,589/8,137/8,615 scenes with
field-validity masks; the historical complete-geometry cohort contains 57/2/6 scenes.
The historical single-source `direct-bbox-20261006` pilot contains
9,601/539/624 scenes and 33,545 objects from SpatialLM. Its builder applied a
source whitelist; these counts do not establish that other sources are
ineligible. Its yaw is pi-periodic bbox-axis orientation, not certified semantic
front. That private pilot snapshot is fixed at
`35f5272330d37771eea2d11925c42aeec9d917d4`.

The main `multisource-20261006` derivative retains the original **16-family
selection**: 11 families expand to 16 training source tags, two families remain
evaluation-only (SceneSmith and SpatialGen), and three provide auxiliary inputs.
It preserves the full parent condition (room geometry, fixed objects, object
requests, support metadata and constraints) and partial supervision instead of
requiring every source to have complete position, size and yaw labels. Target
numbers stay unchanged; conservative Scan2CAD qualification and 13 whole-size
masks address the audited D1/D2 issues. Main yaw remains the 581 inherited valid
semantic-yaw targets, with no SpatialLM geometric-yaw promotion.
If a D1/D2 mask change leaves any exchangeable-group member without complete
position/size supervision, all members lose that group's exchangeable tag and
fall back to fixed identity. Each changed member is logged; IDs, request order,
support, relations and target numbers remain intact.
The protocol and conservative Scan2CAD/size-range policies are documented in
the [training pipeline](docs/fastfill-v2-training-pipeline.md). The completed
build locations are `outputs/fastfill_v2/multisource-20261006/data`,
`/Volumes/harddisk/FastFill_v2_multisource_20261006`, and the server copy
`/home/jovyan/shanliantian/FastFill_v2_multisource_20261006`. Final build hashes,
split counts and actual Qwen tokenizer eligibility are frozen in the new
manifests and preflight. The private multi-source release is fixed at
`ba1c3bf018c49bc841b696c25f8c2e1d1ff61a88`, with main data, actual eligible
view, independent minimal/rectangle ablations and held-out NEAR data. The
24 canonical data files total 6,277,486,224 bytes; all 47 release files have
verified remote sizes and SHA256. Historical directories were removed from
current HF HEAD while their pinned revisions remain available.
The independent three-field `reference_extent` ablation is built under
`outputs/fastfill_v2/multisource-20261006/data-minimal-reference`; its XY coordinate
translation, unknown physical boundary/floor flags and geometric-axis yaw policy
must not be substituted for the main task. The inherited main validation/test
lack explicit relations. A separate held-out reference-derived NEAR view is now
built and independently checked; its own tokenizer/model evaluation remains
pending. Rare semantic-yaw exposure has been counted for the candidate three-epoch
run, which has not been launched. See the [completed data/server record](docs/fastfill-v2-multisource-20261006.md)
and [design audit](docs/fastfill-v2-design-audit-20261006.md).
See the [dataset download and usage guide](docs/fastfill-v2-dataset-release.md).
The v1 dataset table below remains historical and is not the v2 training input.

| Directory | Train | Dev | Test | Purpose |
|---|---:|---:|---:|---|
| `v3/` | 144,140 | 8,167 | 8,647 | No training constraints |
| `v3.1/` | 144,140 | 8,167 | 8,647 | Same rooms/targets; 42,743 training rows carry constraints |
| `v3.2/` | 144,150 | 8,167 | 8,647 | Corrected constrained dataset; 10 added train rows, 25 changed train rows and 1 changed test row |

The dataset repository also preserves `code/` for v3/v3.1 and `code_v3.2/` for v3.2. This Git repository contains implementation and tests; raw scenes, meshes, weights and generated training JSONLs are not checked in.

## Install and verify

The commands below are for FastFill v1. Install a PyTorch build suitable for your CPU/CUDA environment first. FastFill v2 has its own [server environment and training guide](docs/fastfill-v2-server-start.md).

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -B -m pytest -q -p no:cacheprovider fastfill/tests
```

The offline regression suite covers geometry conversion, build transactions, QA, runtime validation, and a tiny real CPU LoRA merge. It does not replace GPU training, model-quality evaluation, deployment testing or mesh-level validation. vLLM is optional for **v1 text evaluation** and required by the **v1 text service**. The v2 structured training and evaluation entry points do not use vLLM.

The optional `fastfill.tools.visualize` utility also requires `pip install matplotlib`.

## Train and evaluate

```bash
# Choose v3 OR v3.1 for the paired experiment; use a separate run for v3.2.
hf download liantian/fastfill-v3 --repo-type dataset \
  --include 'v3/*' --include 'v3.1/*' --include 'v3.2/*' --local-dir data

python -m fastfill.train --model /path/to/Qwen3-8B \
  --data data/v3.1 --out outputs/ff-v3.1 --dry_run

python -m fastfill.train --model /path/to/Qwen3-8B \
  --data data/v3.1 --out outputs/ff-v3.1

python -m fastfill.merge_lora --base_model_path /path/to/Qwen3-8B \
  --lora_path outputs/ff-v3.1/final --output_path outputs/ff-v3.1-merged

python -m fastfill.evaluate --model outputs/ff-v3.1-merged \
  --rooms data/v3.1/test_constrained_rooms.jsonl --out eval/ff-v3.1-constraints
```

Select memory, batch size and training budget using the actual tokenizer profile and target hardware. Report both original and deterministically repaired outputs, completeness, constraint satisfaction and collision metrics. `valid` and an accepted HTTP response do not guarantee a collision-free or physically stable scene.

## Implementation

| Path | Responsibility |
|---|---|
| `fastfill/adapters/` | Source-to-canonical-IR conversions |
| `fastfill/build.py`, `split.py`, `anchors.py` | Dataset preparation, leakage groups and support handling |
| `fastfill/scene.py` | Shared text protocol, coordinate conventions and parsing |
| `fastfill/train.py`, `merge_lora.py` | LoRA SFT and adapter merging |
| `fastfill/evaluate.py`, `validate.py` | Model and reference scoring |
| `fastfill/interface.py`, `serve.py` | WorldEdge request conversion and HTTP service |
| `fastfill/tools/`, `fastfill/tests/` | Data QA, ablation checks and regressions |

See the [technical documentation](fastfill/README.md) and [v3.2 release notes](fastfill/RELEASE_v3.2.md). The implementation follows the object-conditioned layout-learning approach of [OptiScene](https://github.com/PolySummit/OptiScene). Its original `main.py` / DPO scripts and schema are not entry points for this package; preference training requires data in a consistent FastFill protocol.

## FastFill v2 implementation

The separate experimental [FastFill v2 package](fastfill/v2/README.md) jointly
predicts target local size, bottom-center position and yaw for a specified object
inventory. See the [selected-corpus review](docs/fastfill-v2-review-20261005.md) before
retraining. The [training pipeline walkthrough](docs/fastfill-v2-training-pipeline.md)
and [execution manual](docs/fastfill-v2-runbook.md) describe the reviewed upload bundle,
optimizer steps and deployment artifacts. The [server setup guide](docs/fastfill-v2-server-start.md)
gives the upload, CUDA environment and **Qwen3-8B** BF16 pilot commands. The
selected model is separate from the historical 0.5B development template;
`fastfill/v2/configs/qwen3_8b_pilot.json` records the historical pilot settings;
`qwen3_8b_bbox_pilot.json` is the separate 20-step historical minimal-input template.
`qwen3_8b_main_world7.json` freezes the reviewed Qwen3-8B seven-rank B1/K16
three-epoch candidate (3,333 updates). It contains the server model path and
must be passed explicitly; the generic development default is not this run.
The SpatialLM pilot reaches at most 26 objects per scene. That limit does not
describe the multi-source main corpus; its actual object/context eligibility,
rare-yaw exposure and dense-layout evaluation must be frozen with the new run.
Existing FastFill v1 entry points and v3-series datasets are preserved.
