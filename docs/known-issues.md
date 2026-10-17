# Verified limitations and remaining issues

These notes describe the audited v3.2 implementation. A frozen and hash-verified release is reproducible; it does not imply that all defects are resolved.

The [2026-09-28 source-by-source recheck](audit-2026-09-28.md) records the download inventory, raw-source checks, frozen-output comparisons, and the limits of each conclusion. No implementation or frozen dataset was changed during that audit.

## Open implementation issues

- `fastfill/serve.py` bounds Content-Length but does not set a request-read timeout. The synchronous HTTP server can wait indefinitely for an incomplete request body. Add a request deadline before relying on this server for clients that may stall; concurrent model execution is a separate design decision.
- `fastfill/adapters/scan2cad.py` does not recompute upright extents for objects tilted by more than its explicit 10-degree noise tolerance. Independent source-corner checks confirm wrong dimensions in eight targets across six saved v3.2 train rows; seven targets in five rows pass the default training flag filter, before token-length filtering and weighting.
- `fastfill/adapters/il3d_3dfront.py` treats an exported opening's world-axis span as its length along an oblique wall. Confirmed ordinary rectangular windows are too short: 39 fixed-input boxes across 24 saved rooms, including 25 boxes in 15 train rows that pass the default flag filter. This is a fixed-input geometry issue, not a window placement-target issue.
- `fastfill/adapters/spatialgen.py` retains local extents while discarding full tilt. Two affected paintings remain as targets in v3.2 test rows 5419 and 5423. Six other affected source objects are filtered before final messages; this source has no train rows.
- `fastfill/tools/qa_report.py` repairs training reference containment without passing saved prompt constraints. Evaluation and the API preserve hard constraints during repair, so the QA repaired-containment statistic can disagree with them. A synthetic case is reproduced; the incidence in the saved dataset has not been measured.

## Semantic-front evidence and intermediate IR

- InternScenes deliberately substitutes a horizontal geometric axis when source local X is the closest-to-vertical axis. Its geometry helper passes independent corner checks. The retained source documentation does not establish a semantic front for every asset, so the earlier unconditional description of this as a verified near-vertical **semantic** front was too strong. The issue is unverified semantic interpretation of a known geometric fallback, not a demonstrated box-conversion error in the TV/monitor examples. In v3.2, 5,874 such target objects occur in 2,927 train rows after default flag filtering; these are exposure counts, **not error counts**. Setting `front_known=False` under the existing build policy can remove objects and alter room selection; no blanket relabeling was performed.
- SceneSmith has 46 wall-mounted source objects whose asset origin is at the bottom, although the adapter assumes vertical center. Their frozen IR is shifted downward. None is directly retained as a target or fixed object in the current saved test messages; possible indirect effects through anchoring, filtering, or flags have not been excluded.
- InteriorGS has a reproduced thin-object edge case in its 2 cm shortcut. The three inspected lamps are absent from final targets and fixed inputs. This is not evidence that all final room content is unaffected indirectly.
- Several adapters use empirically calibrated front conventions. Wall-contact statistics and a geometric box axis do not establish per-asset semantic truth. Preserve this distinction when reporting facing accuracy or proposing data migrations.

## Metric and capability boundaries

- `valid` checks room containment with tolerance, support and ceiling limits. It excludes neither every object-object collision nor every physical failure.
- Runtime acceptance also checks the 1 mm floor-containment rule, non-window fixed obstacles and hard relations. Collisions between objects to place are reported but do not reject the layout. Report severe collisions separately.
- Support means the child's bottom is near the parent's box top and its center is near the parent's footprint. This is not mesh contact, center-of-mass or rigid-body stability verification.
- Fixed geometry may represent existing furniture as obstacles, but relation constraints cannot refer to fixed IDs. Adding a cup onto an existing fixed table is not supported by this protocol. `on` relates objects generated in the same request.
- Integer-degree training targets introduce up to half a degree of angular quantization; geometry still uses floating-point calculations.
- Constraints in v3.1/v3.2 are extracted from reference layouts where they already hold. Counterfactual tests with the same room/objects and different feasible constraints are needed to demonstrate condition-responsive control.
- The earlier standalone, single-room side-by-side plotting scripts are not this package's benchmark. They use a different output schema and should not be used as evidence of FastFill v3-series performance.

## Verification scope

The current offline suite passed 138 tests and 90 subtests in the recorded audit. It includes a tiny real CPU adapter merge; generation and HTTP tests also use mocks. Production GPU training, vLLM deployment, arbitrary mesh correctness and full physical simulation were not validated by that suite. Dependency vulnerability auditing requires a reachable vulnerability database and was unavailable in the consolidation environment.
