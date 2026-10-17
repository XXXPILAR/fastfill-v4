# Versions and repository ownership

The maintained source repository is https://github.com/Slian22/fastfill, with one development branch, `main`.

## Implementation history

- The earlier independent repository used `src/fastfill_train`, a Floor/Surface codec and vendored WorldEdge contracts. It is retained in Git history; its inputs and model protocol differ from the current implementation.
- Current development originated in the `OptiScene/fastfill` directory. The package performs one-call object-conditioned layout generation and provides the v3-series data pipeline.
- The v3.2 release receipt records GitHub commit `5ab744d3de42f0ffc7eadd692cac90dd18d8e392`, originally on `fastfill-v3.2`. That immutable commit remains the release identifier even after branches are consolidated.
- The corresponding Hugging Face release receipt records dataset commit `d264159e7095e4ce817d46c2fa0bd0cf5dc3d28d` in `liantian/fastfill-v3`, adding `v3.2/` and `code_v3.2/`.

## Reproducing experiments

v3 and v3.1 are the original paired experiment: identical rooms, reference answers, splits and evaluation files, with optional training constraints added to v3.1. Use their original run manifests and `code/` snapshot when reproducing existing runs. Current code has subsequent corrections and should not be silently substituted into a historical result.

v3.2 was rebuilt from 18 selected source IR files. HSSD200, InteriorGS, MultiScan and Structured3D IR changed; the other 14 input hashes stayed the same. The output added 10 train rows and modified 25 train rows and one test row. No original split memberships moved. It is a separate corrected constrained dataset, not the constrained arm of a newly rebuilt paired experiment.

The repository's source consolidation does not rebuild data, retrain models, alter Hugging Face versions or change running jobs. The dataset version, implementation commit, checkpoint and evaluation inputs must each be recorded.

## Local working-copy names

Historical local layouts can contain both `Worldedge/fastfill` (old independent checkout) and `Worldedge/OptiScene/fastfill` (new development package). Directory names alone do not identify the checked-out version. Inspect the Git commit and use the root README's `python -m fastfill...` commands for this implementation.

Old data copies and frozen code snapshots should be retained until their experiment references are recorded. Removing Git branch names does not remove backed-up history or imply that those data versions should be deleted.
