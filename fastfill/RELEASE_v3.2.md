# FastFill v3.2 release

Verification: code repairs, full dataset build, independent scan and exact project QA replay passed. The local publication receipt records GitHub commit `5ab744d3de42f0ffc7eadd692cac90dd18d8e392` and Hugging Face commit `d264159e7095e4ce817d46c2fa0bd0cf5dc3d28d`. Repository consolidation preserves these release identifiers; see [versions](../docs/versions.md).

v3.2 is a separately rebuilt, constrained dataset derived from v3.1 (constraint probability 0.3, the same 18 selected sources). v3 and v3.1 remain immutable baselines for existing training jobs. Do not change a dataset in the middle of a run.

## Changes

- Re-express tilted Structured3D, HSSD200 and MultiScan boxes as enclosing upright boxes using their rotated geometry.
- Propagate unknown horizontal fronts for 245 upright MultiScan objects with vertically annotated fronts; their pose and size are unchanged. Separately, 120 tilted MultiScan objects receive corrected envelopes.
- Recover InteriorGS objects that the upstream export left unassigned only when their source box center belongs to exactly one valid repaired room polygon; record recovery decisions and unresolved cases.
- Protect builds against empty/missing inputs, malformed room geometry and partial writes. The output directory must be new or empty; completed output is published by an atomic directory rename.
- Align support tolerance boundaries and singular/plural window handling; enforce API anchor requirements, keep soft keepouts soft, and bound HTTP request reads.
- Validate written-file hashes, training constraints and actual IR-to-message reproduction in QA; compare ablation metadata too.
- Preserve per-prompt generation limits during HF evaluation and adapter tokenizer/run provenance during merging. Download failures return a failing exit code.

The API now accepts only positive Content-Length values up to 1 MiB; invalid lengths return 400 and oversized requests return 413. Anchor requirements are checked on returned placements; violations return 422. Soft keepouts report satisfaction without altering the floor or adding unseen conditioning fields to the model prompt.

The text protocol and system prompt remain unchanged. Runtime validation changes can be used with existing weights. New geometry and recovered rooms mean v3.2 is not a constraint-only ablation of v3; compare models on the same versioned evaluation inputs.

## Existing-run interpretation

The September 27 evidence identifies 21 retained erroneous boxes across the three adapters. Joining that complete list to the saved v3/v3.1 row index gives 15 boxes (including fixed context boxes) in 10 default-selected training rooms, plus one affected Structured3D test room. This contradicts the claim that only one training object is affected or all affected HSSD rows are excluded.

The old 124,834 / 124,836 figure measures the implemented out-of-bounds, support and ceiling checks. It is not a general data-purity rate and does not test raw box correctness or all collisions. The known issues do not by themselves justify discarding an ongoing run; the actual server configuration was not inspected here.

## Reproduction

Use new writable directories for IR and data; preserve the original versions. The release records source IR hashes, code hashes, build arguments, and every output file hash in MANIFEST.json. Source geometry changes can change deduplication and connected split components; a release comparison must enumerate those changes before cross-version benchmark claims.

```bash
python -m pytest -q -p no:cacheprovider fastfill/tests
python -m fastfill.merge_lora --base_model_path BASE --lora_path RUN/final --output_path MERGED
```

Regression suite: 138 tests and 90 subtests passed, including the tiny real CPU merge, mocked HTTP handlers, source-envelope fixtures and existing synthetic self-checks. Combined line coverage of the 13 changed/new production modules: 90.8% (2,075 / 2,285 executable lines). This is not whole-repository coverage.

## Dataset verification

The rebuild scanned 188,670 rooms across 18 sources. Four adapters were rerun for 32,125 IR rooms; the other 14 selected IR files match the v3.1 input hashes. Output: 144,150 train / 8,167 dev / 8,647 test. The default truthy-flag filter selects 124,843 training rows before token-length filtering or source weighting.

Compared with v3.1, 25 existing train rows and one test row changed; 10 train rows were added and none removed. Split membership is unchanged. Eight recovered InteriorGS rooms pass the default flag filter, while one existing HSSD room now fails it; the net default-selected increase is seven. No new dev/test source group or alias overlaps the old training split. Dev inputs are unchanged; compare old and new models on a common version of the corrected test input.

The independent scanner uses no project imports. It checked all 25 IR/processed JSONLs, 383,016 rows and 2,251,366,019 bytes. All 15 leakage intersections (five keys across three split pairs) are zero. All 98,808 train, 18,662 dev and 19,785 test constraints hold. Written schemas, reference closure, support cycles, finite numbers and manifest hashes passed. All 124,843 default-selected references pass the scanner's out-of-bounds/support/ceiling checks; this does not assert mesh-level correctness or absence of all collisions. Raw IR retains 90 invalid boundaries, 99 nonpositive sizes and 19 clockwise boundaries; the written schemas pass after preparation.

All 20 original v3/v3.1 files match their pre-build hashes. The release code snapshot matches all 41 implementation hashes in MANIFEST.json. The separate publisher's seven offline tests passed. Its publication receipt records status `published`; the later consolidation audit could not independently access the remote services.

The cached Qwen3-8B tokenizer (revision b968826d9c46dd6066d109eabc6255188de91218, offline only) checked all 36 added/modified SFT rows: 575–3,470 tokens including the answer end token, with none above the default 40,960-token limit. This is a targeted length check, not a full-corpus token census; no model weights were loaded.

Project QA reproduced all 160,964 saved SFT rows exactly from IR (system/user/assistant messages and metadata), with zero mandatory failures. This replay verifies persisted rows; it does not independently rerun the global deduplication and split-selection algorithm. The separate scanner and version comparison check the resulting splits and identities.

## Subsequent audit

Request-read timeouts and InternScenes near-vertical semantic-front flags remain open issues; see [known issues](../docs/known-issues.md). The recorded geometry checks and release hashes do not establish that all defects are resolved.
