"""New full-condition research revision: mask uncertain labels, preserve geometry.

The three-field reference-extent dataset is a separate ablation. This main view
retains room polygons, existing objects, support, identity and constraints.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile

from .io import fingerprint, safe_output
from .legacy_bridge import HEIGHT_TOLERANCE_M, height_conflict
from .multisource_data import (DEGENERATE_AXIS_M, HOLDOUT_REASON, ROOMGENBENCH_HOLDOUT_GROUPS, SPLITS, SOURCES,
                              _frozen_ir_guard, _holdout, _input_hashes, _loads, _parent_rows, _size_policy,
                              _split_integrity, _write, qualify_parent_masks)
from .schema import migrate_legacy_row, validate_condition

POLICY = "full-condition-mask-review-v1"
YAW_POLICY = "inherit parent yaw validity and yaw_symmetry_order; no geometric promotion"
PARENT_HASHES = {
    "manifest.json": "7135a59f097e665ec6bb0973aae0233f323be447556e4ef903b868afbe779652",
    "train.jsonl": "ad885654907242e9cb7b222653c9da5f9a5237c5586215e7ced94ed74c830c92",
    "validation.jsonl": "5a0a45d6b2d82f8fcbf3a1f2d7f908ce9531e484efdac870b869dd9164a5b773",
    "test.jsonl": "efc393bd36f1110ca3e037d89ddc70c60e7b6cd676bdc1c47b2f7f81e9648ea9",
    "rejections.jsonl": "02925d15b20a613807ada2f8741c851c24c1b036720feaa8b1c0fadc8317f95a",
}


def qualify_sample(parent, *, model_config=None, holdout_groups=()):
    """Pure full-task derivation; never promote yaw or rewrite target numbers."""
    parent = migrate_legacy_row(parent)
    source = parent["provenance"]["source"]
    split = parent["provenance"]["split"]
    if source not in SOURCES or split not in SPLITS or source in {"SceneSmith", "SpatialGen"} and split != "test":
        raise ValueError("unknown source or evaluation-only split violation")
    if "dataset_qualification_policy" in parent["provenance"]:
        raise ValueError("parent is already qualification-reviewed")
    validate_condition(parent["condition"])
    requests = [obj["id"] for obj in parent["condition"]["objects"]]
    targets = [obj["id"] for obj in parent["target"]["objects"]]
    source_ids = parent["provenance"]["target_source_ids"]
    if (requests != targets or len(set(targets)) != len(targets)
            or len(source_ids) != len(targets) or len(set(source_ids)) != len(source_ids)):
        raise ValueError("target identity/order must equal stable parent request order")
    masks, changes = qualify_parent_masks(parent, model_config=model_config)
    row = deepcopy(parent)
    row["validity"] = {**row["validity"], **masks}
    added = {"dataset_qualification_policy": POLICY, "qualification_changes": changes}
    if source == "Scan2CAD":
        room, meta = parent["condition"]["room"], parent["provenance"].get("source_meta", {})
        row["condition"]["room"]["floor_known"] = False
        added["qualification_changes"] = [{"field": "room.floor_known", "before": room.get("floor_known"),
            "after": False, "reason": "Scan2CAD_floor_is_estimated_not_independent_physical_measurement"}, *changes]
        added["estimated_floor_provenance"] = {"floor_source": "estimated",
            "parent_floor_known": room.get("floor_known"), "parent_floor_z_m": room.get("floor_z_m"),
            "source_meta_floor_z": meta.get("floor_z"), "source_meta_n_floor_snapped": meta.get("n_floor_snapped"),
            "per_object_pre_snap_z": "unavailable_in_frozen_IR",
            "per_object_snap_membership": "unknown_do_not_infer_from_zero_z"}
    # K2 flag only: labels and the declared height stay. The flag follows the qualified masks; a
    # differing parent (bridge) flag is replaced or removed and journaled with its value as ``before``.
    before = parent["provenance"].get("height_conflict")
    conflict = height_conflict(row["condition"]["room"].get("height_m"), parent["target"]["objects"],
                               masks["position"], masks["size"])
    if conflict:
        added["height_conflict"] = conflict
    if conflict != before:
        added["qualification_changes"] = [*added["qualification_changes"], {
            "field": "provenance.height_conflict", "before": deepcopy(before), "after": deepcopy(conflict),
            "reason": "target_exceeds_declared_height_flag_only" if before is None else "height_conflict_follows_qualified_masks"}]
    # Position demotion can invalidate a previously legal exchangeable group.
    # The protocol falls back to fixed identities; object count/roles stay intact.
    groups = row["validity"].get("exchangeable_group", [])
    invalid_groups = {g for i, g in enumerate(groups) if g is not None and not all(masks["position"][i])}
    for i, group in enumerate(groups):
        if group in invalid_groups:
            row["validity"]["exchangeable_group"][i] = None
            added["qualification_changes"] = [*added["qualification_changes"], {
                "object_id": targets[i], "target_source_id": source_ids[i], "field": "exchangeable_group",
                "before": group, "after": None, "reason": "mask_review_removed_complete_position_exchangeability"}]
    provenance, moved = _holdout(row["provenance"], split, holdout_groups)
    if moved:
        added["qualification_changes"] = [*added["qualification_changes"], moved]
    row["provenance"] = {**provenance, **added}
    if not conflict:
        row["provenance"].pop("height_conflict", None)
    validate_condition(row["condition"])
    return row


def _protected(parent, target, ir):
    roots = [parent, Path("/Volumes/harddisk/3D_Room_Collections").resolve()]
    if ir is not None:
        roots.append(Path(ir).resolve())
    if any(target == root or root in target.parents or target in root.parents for root in roots):
        raise ValueError("new output must be outside protected inputs and ancestors")


def build_dataset(parent_root, output, *, expected_hashes=None, ir_root=None, frozen_manifest=None, model_config=None,
                  holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS):
    """Publish only after parent pins, split integrity and before/after hashes pass."""
    parent, target = Path(parent_root).resolve(), safe_output(output)
    holdout_groups = tuple(holdout_groups)
    _protected(parent, target, ir_root)
    implementation_before = {str(path): fingerprint(path) for path in  # legacy_bridge: the shared K2 height_conflict
                             (Path(__file__).resolve(), *(Path(__file__).with_name(name).resolve()
                                                          for name in ("multisource_data.py", "legacy_bridge.py")))}
    pins = PARENT_HASHES if expected_hashes is None else expected_hashes
    parent_inputs = {str(path): fingerprint(path) for path in [parent / (s + ".jsonl") for s in SPLITS]
                     + [parent / "manifest.json", parent / "rejections.jsonl"]}
    if any(parent_inputs.get(str(parent / name)) != digest for name, digest in pins.items()):
        raise ValueError("parent bytes differ from pinned reviewed inputs")
    if expected_hashes is None and (ir_root is None or frozen_manifest is None):
        raise ValueError("release build requires canonical frozen IR manifest and IR root")
    evidence = {}
    if ir_root is not None:
        ir = Path(ir_root).resolve()
        frozen = Path(frozen_manifest).resolve()
        evidence = _input_hashes(parent, ir, frozen)
        _frozen_ir_guard(ir, frozen, evidence)
    before = {**parent_inputs, **evidence}
    integrity = _split_integrity(parent)
    parent_manifest = _loads((parent / "manifest.json").read_text())
    counters = {key: Counter() for key in ("split_samples", "split_objects", "source_split_samples",
                "source_split_objects", "validity_counts", "constraint_counts", "qualification_change_counts",
                "fixed_objects_by_split", "source_split_validity", "supervised_yaw_scenes_by_split",
                "source_yaw_symmetry_order_counts", "source_size_axis_swap_allowed_objects", "holdout_samples_by_parent_split")}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qualified-build-", dir=target.parent) as temp:
        stage = Path(temp)
        with ExitStack() as stack:
            changelog = stack.enter_context((stage / "changes.jsonl").open("x"))
            destinations = {split: stack.enter_context((stage / (split + ".jsonl")).open("x")) for split in SPLITS}
            for parent_split, line_number, parent_row in _parent_rows(parent, holdout_groups):
                row = qualify_sample(parent_row, model_config=model_config, holdout_groups=holdout_groups)
                split, source = row["provenance"]["split"], row["provenance"]["source"]
                _write(destinations[split], row)
                count = len(row["target"]["objects"])
                for key, ident, n in (("split_samples", split, 1), ("split_objects", split, count),
                        ("source_split_samples", source + ":" + split, 1), ("source_split_objects", source + ":" + split, count),
                        ("fixed_objects_by_split", split, len(row["condition"]["room"].get("fixed_objects", []))),
                        ("holdout_samples_by_parent_split", parent_split, split != parent_split)):
                    counters[key][ident] += n
                for field in ("position", "size", "yaw"):
                    n = sum(all(mask) if isinstance(mask, list) else mask for mask in row["validity"][field])
                    counters["validity_counts"][field] += n
                    counters["source_split_validity"][source + ":" + split + ":" + field] += n
                counters["supervised_yaw_scenes_by_split"][split] += any(row["validity"]["yaw"])
                for order in row["validity"].get("yaw_symmetry_order", []):
                    counters["source_yaw_symmetry_order_counts"][f"{source}:{order}"] += 1
                counters["source_size_axis_swap_allowed_objects"][source] += sum(row["validity"]["size_axis_swap_allowed"])
                for constraint in row["condition"].get("constraints", []):
                    counters["constraint_counts"][constraint["type"] + ":" + split] += 1
                for change in row["provenance"]["qualification_changes"]:
                    counters["qualification_change_counts"][change["field"]] += 1
                    _write(changelog, {"uid": row["provenance"]["scene_id"], "source": source,
                                      "split": parent_split, "parent_line": line_number, **change})
                if counters["split_samples"][split] % 20000 == 0:
                    print(json.dumps({"split": split, "scenes": counters["split_samples"][split]}), flush=True)
        print(json.dumps({"split_samples": dict(counters["split_samples"])}), flush=True)
        shutil.copyfile(parent / "rejections.jsonl", stage / "rejections.jsonl")
        if any(fingerprint(path) != digest for path, digest in before.items()):
            raise ValueError("input bytes changed during immutable main build")
        if any(fingerprint(path) != digest for path, digest in implementation_before.items()):
            raise ValueError("producer implementation changed during immutable main build")
        manifest = {"schema_version": "fastfill.v2", "builder": POLICY, "dataset_qualification_policy": POLICY,
            "task_role": "full_condition_multisource_main", "parent_root": str(parent),
            "input_sha256": before, "inputs_unchanged": True, "split_integrity": integrity,
            "source_families": SOURCES, "evaluation_only_sources": ["SceneSmith", "SpatialGen"],
            "auxiliary_families": ["3D-FRONT", "3RScan", "ARKitScenes"],
            "size_output_policy": _size_policy(model_config), "yaw_policy": YAW_POLICY,
            "parent_front_policy": parent_manifest.get("front_policy"), "parent_yaw_policy": parent_manifest.get("yaw_policy"),
            "condition_policy": "preserve full parent conditions; qualify Scan2CAD estimated floor, mask degenerate (<3mm axis) sizes, "
                                "flag (never drop) a declared room height exceeded by a valid target in provenance.height_conflict, "
                                "revert groups losing complete position to fixed identity",
            "degenerate_axis_m": DEGENERATE_AXIS_M, "height_tolerance_m": HEIGHT_TOLERANCE_M,
            "roomgenbench_holdout_groups": sorted(holdout_groups), "holdout_reason": HOLDOUT_REASON,
            "target_geometry_modified": False, "source_data_modified": False, "parent_data_modified": False,
            "admission_policy": "retain every reviewed parent scene; separate actual-tokenizer active-objective/object/context preflight",
            "implementation_sha256": implementation_before,
            **{key: dict(value) for key, value in counters.items()}}
        manifest["output_sha256"] = {path.name: fingerprint(path) for path in sorted(stage.iterdir())}
        with (stage / "manifest.json").open("x") as stream:
            _write(stream, manifest)
        target.mkdir(exist_ok=False)
        for path in stage.iterdir():
            path.rename(target / path.name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ir-root", required=True)
    parser.add_argument("--frozen-manifest", required=True)
    parser.add_argument("--holdout-group", action="append", dest="holdout_groups",
                        help="provenance.group written to test (repeatable); default: RoomGenBench benchmark rooms")
    parser.add_argument("--pin-current-parent", action="store_true",
                        help="pin the parent bytes as found now (recorded in input_sha256) instead of PARENT_HASHES")
    args = parser.parse_args(argv)
    holdout = ROOMGENBENCH_HOLDOUT_GROUPS if args.holdout_groups is None else tuple(args.holdout_groups)
    pins = {name: fingerprint(Path(args.parent_root) / name) for name in PARENT_HASHES} if args.pin_current_parent else None
    result = build_dataset(args.parent_root, args.output, ir_root=args.ir_root, frozen_manifest=args.frozen_manifest,
                           holdout_groups=holdout, expected_hashes=pins)
    print(json.dumps({"output": args.output, "split_samples": result["split_samples"],
                      "qualification_changes": result["qualification_change_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
