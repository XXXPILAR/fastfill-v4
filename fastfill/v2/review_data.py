"""Publish a new evidence-qualified revision; never repair parent geometry.

Only frozen source-derived ``faces`` constraints lacking semantic-yaw evidence
are omitted. Explicit user requirements survive. Below-floor target positions
are diagnostics, not clamped labels. This module intentionally uses stdlib only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

SPLITS = ("train", "validation", "test")
INPUT_FILES = tuple(f"{split}.jsonl" for split in SPLITS) + ("manifest.json", "rejections.jsonl")
OUTPUT_FILES = tuple(name for name in INPUT_FILES if name != "manifest.json")
POLICY = "source-derived-facing-evidence-v1"
BUILDER = "immutable-parent-review-v1"
FLOOR_TOLERANCE_M = 1e-4
LEGACY_CONSTRAINT_SOURCE = "frozen_sparse_legacy_request"
RAW_SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")
REVIEW_FIELDS = ("review_parent_scene_id", "review_policy", "review_source_qualification",
                "review_omitted_constraints", "review_geometry_diagnostics")


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def _read_json(value):
    return json.loads(value, parse_constant=_reject_constant)


def _parent_inputs(root):
    for name in INPUT_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f"required parent input missing: {root / name}")
    raw = (root / "manifest.json").read_bytes()
    manifest = _read_json(raw)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "fastfill.v2":
        raise ValueError("parent manifest must identify fastfill.v2 data")
    hashes = {name: _digest(root / name) for name in INPUT_FILES if name != "manifest.json"}
    return manifest, {**hashes, "manifest.json": hashlib.sha256(raw).hexdigest()}


def _recheck(root, hashes):
    if any(_digest(root / name) != expected for name, expected in hashes.items()):
        raise ValueError("parent input or manifest changed during review")


def _new_output(parent_root, output):
    if Path(output).is_symlink():
        raise FileExistsError(f"immutable review output already exists as a symlink: {output}")
    parent, target = Path(parent_root).resolve(), Path(output).resolve()
    for protected in (parent, RAW_SOURCE_ROOT.resolve()):
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError("output must be outside source data and its ancestors")
    if target.exists():
        raise FileExistsError(f"immutable review output already exists: {target}")
    return parent, target


def _complete(mask, dimensions):
    return mask is True or (isinstance(mask, list) and len(mask) == dimensions
                            and all(value is True for value in mask))


def _mask(sample, field, index, dimensions):
    values = sample["validity"].get(field, [])
    return index < len(values) and _complete(values[index], dimensions)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _indices(sample):
    if not isinstance(sample, dict) or sample.get("schema_version") != "fastfill.v2":
        raise ValueError("parent rows must use fastfill.v2 schema")
    condition, target = sample.get("condition", {}), sample.get("target", {})
    if not isinstance(condition, dict) or not isinstance(target, dict):
        raise ValueError("condition and target must be objects")
    collections = (condition.get("objects"), target.get("objects"))
    ids = []
    for objects in collections:
        if not isinstance(objects, list) or any(not isinstance(obj, dict)
                or not isinstance(obj.get("id"), str) or not obj["id"] for obj in objects):
            raise ValueError("request and target objects require nonempty string IDs")
        names = [obj["id"] for obj in objects]
        if len(names) != len(set(names)):
            raise ValueError("duplicate request or target ID")
        ids.append(names)
    if set(ids[0]) != set(ids[1]):
        raise ValueError("target IDs must equal requested IDs exactly once")
    validity = sample.get("validity")
    if not isinstance(validity, dict) or any(not isinstance(validity.get(field, []), list)
            or len(validity.get(field, [])) > len(ids[1]) for field in ("position", "size", "yaw")):
        raise ValueError("invalid geometry validity rows")
    if not isinstance(condition.get("room"), dict) or not isinstance(condition.get("constraints", []), list):
        raise ValueError("room and constraints must be structured fields")
    provenance = sample.get("provenance")
    if not isinstance(provenance, dict) or any(not isinstance(provenance.get(key), str)
            or not provenance[key] for key in ("scene_id", "legacy_uid")):
        raise ValueError("parent provenance requires stable scene_id and legacy_uid")
    evidence = provenance.get("field_evidence", [])
    if (not isinstance(evidence, list) or len(evidence) > len(ids[1])
            or any(not isinstance(row, dict) for row in evidence)):
        raise ValueError("field evidence must follow target object indices")
    if any(key in provenance for key in REVIEW_FIELDS):
        raise ValueError("parent already contains review provenance")
    return {name: index for index, name in enumerate(ids[1])}


def _trusted_yaw(sample, index):
    """Facing needs a semantic front: a valid yaw whose symmetry order is 1 (axis-only mod-pi yaw is not evidence)."""
    orders = sample["validity"].get("yaw_symmetry_order", [])
    if not _mask(sample, "yaw", index, 1) or index < len(orders) and orders[index] != 1:
        return False
    angle = sample["target"]["objects"][index].get("yaw_rad")
    if not _finite(angle) or not -math.pi <= angle < math.pi:
        raise ValueError("valid yaw label must be finite and wrapped to [-pi, pi)")
    evidence = sample["provenance"].get("field_evidence", [])
    field = evidence[index] if index < len(evidence) else {}
    recorded = field.get("recorded_source_evidence", {})
    if not isinstance(recorded, dict):
        raise ValueError("recorded source evidence must be an object")
    return "front_unknown_reason" not in recorded


def _floor_diagnostics(sample):
    room = sample["condition"]["room"]
    known = room.get("floor_known") is True
    floor = room.get("floor_z_m") if known else None
    if known and not _finite(floor):
        raise ValueError("known floor requires a finite floor_z_m")
    conflicts = []
    for index, target in enumerate(sample["target"]["objects"]):
        if not known or not _mask(sample, "position", index, 3):
            continue
        position = target.get("bottom_center_m")
        if not isinstance(position, list) or len(position) != 3 or not all(map(_finite, position)):
            raise ValueError("valid position label must be finite XYZ")
        if position[2] < floor - FLOOR_TOLERANCE_M:
            conflicts.append({"object_id": target["id"], "bottom_minus_floor_m": position[2] - floor})
    return {"known_floor": known, "floor_z_m": floor, "tolerance_m": FLOOR_TOLERANCE_M,
            "below_known_floor": conflicts,
            "minimum_bottom_minus_floor_m": min((c["bottom_minus_floor_m"] for c in conflicts), default=None),
            "labels_modified": False}


def review_sample(parent):
    """Pure derivation: only omit unverified frozen legacy facing constraints."""
    indices = _indices(parent)
    legacy = parent["provenance"].get("constraint_source") == LEGACY_CONSTRAINT_SOURCE
    kept, omitted = [], []
    for constraint in parent["condition"].get("constraints", []):
        if not isinstance(constraint, dict):
            raise ValueError("constraint must be an object")
        if legacy and constraint.get("type") == "faces":
            object_id = constraint.get("object_id")
            if object_id not in indices:
                raise ValueError("source-derived facing refers to an unknown request ID")
            if not _trusted_yaw(parent, indices[object_id]):
                omitted.append({"constraint": deepcopy(constraint),
                                "reason": "unverified_source_derived_facing"})
                continue
        kept.append(deepcopy(constraint))
    result = deepcopy(parent)
    if "constraints" in parent["condition"]:
        result["condition"] = {**result["condition"], "constraints": kept}
    result["provenance"] = {**result["provenance"],
        "review_parent_scene_id": parent["provenance"]["scene_id"], "review_policy": POLICY,
        "review_source_qualification": "Frozen legacy constraints are source-derived annotations, not newly "
            "synthesized user requirements; unverified source-derived facing is omitted. "
            "Explicit/nonlegacy requirements and existing exchangeability groups are preserved.",
        "review_omitted_constraints": omitted, "review_geometry_diagnostics": _floor_diagnostics(parent)}
    return result


def _rows(root, split, seen):
    with (root / f"{split}.jsonl").open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                sample = _read_json(raw)
                provenance = sample["provenance"]
                uid = provenance["legacy_uid"]
                if not isinstance(uid, str) or not uid or uid in seen or provenance.get("split") != split:
                    raise ValueError("duplicate/missing UID or provenance split disagrees with its file")
                seen.add(uid)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid row {split}.jsonl:{number}: {exc}") from exc
            yield sample


def _empty_counts():
    return {key: Counter() for key in ("split_samples", "split_objects", "source_split_samples",
        "source_omitted_faces", "source_floor_conflict_objects", "totals")}


def _count(counts, sample, split):
    source = sample["provenance"].get("source", "unknown")
    omitted = len(sample["provenance"]["review_omitted_constraints"])
    conflicts = len(sample["provenance"]["review_geometry_diagnostics"]["below_known_floor"])
    counts["split_samples"][split] += 1
    counts["split_objects"][split] += len(sample["target"]["objects"])
    counts["source_split_samples"][f"{source}:{split}"] += 1
    for key, value in (("omitted_faces", omitted), ("rows_with_omitted_faces", bool(omitted)),
                       ("floor_conflict_objects", conflicts), ("rows_with_floor_conflicts", bool(conflicts))):
        counts["totals"][key] += value
    if omitted:
        counts["source_omitted_faces"][source] += omitted
    if conflicts:
        counts["source_floor_conflict_objects"][source] += conflicts


def _summary(counts):
    result = {key: dict(value) for key, value in counts.items() if key != "totals"}
    for key in ("split_samples", "split_objects"):
        result[key] = {split: counts[key][split] for split in SPLITS}
    return {**result, **counts["totals"], "samples_written": sum(counts["split_samples"].values()),
            "objects_written": sum(counts["split_objects"].values())}


def _write_records(parent, staging):
    seen, counts = set(), _empty_counts()
    for split in SPLITS:
        with (staging / f"{split}.jsonl").open("xb") as output:
            for record in _rows(parent, split, seen):
                child = review_sample(record)
                output.write((json.dumps(child, ensure_ascii=False, allow_nan=False,
                                         separators=(",", ":")) + "\n").encode())
                _count(counts, child, split)
    shutil.copyfile(parent / "rejections.jsonl", staging / "rejections.jsonl")
    return _summary(counts)


def build_reviewed(parent_root, output):
    """Stream all rows and atomically publish to a new immutable directory."""
    parent, target = _new_output(parent_root, output)
    manifest, hashes = _parent_inputs(parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary)
        result = {"schema_version": "fastfill.v2", "builder": BUILDER, "review_policy": POLICY,
            "parent_root": str(parent), "parent_builder": manifest.get("builder"),
            "parent_front_policy": manifest.get("front_policy"),
            "front_policy": manifest.get("front_policy"), "yaw_policy": manifest.get("yaw_policy"), "parent_sha256": hashes,
            "split_rule": "inherit every parent UID and split without re-splitting or filtering",
            "labels_modified": False, "source_data_modified": False,
            "implementation_sha256": _digest(__file__), **_write_records(parent, staging),
            "output_sha256": {name: _digest(staging / name) for name in OUTPUT_FILES}}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        _recheck(parent, hashes)
        if target.exists() or target.is_symlink():
            raise FileExistsError("immutable review output appeared during derivation")
        os.rename(staging, target)
    return result


def verify_reviewed(data_root, parent_root):
    """Check fingerprints, every deterministic row, preserved identities and totals."""
    child, parent = Path(data_root).resolve(), Path(parent_root).resolve()
    _, parent_hashes = _parent_inputs(parent)
    manifest, child_hashes = _parent_inputs(child)
    if (manifest.get("builder") != BUILDER or manifest.get("review_policy") != POLICY
            or manifest.get("parent_sha256") != parent_hashes):
        raise ValueError("review manifest policy or parent fingerprints do not match")
    expected_hashes = {name: child_hashes[name] for name in OUTPUT_FILES}
    if manifest.get("output_sha256") != expected_hashes:
        raise ValueError("review output hashes do not match manifest")
    if child_hashes["rejections.jsonl"] != parent_hashes["rejections.jsonl"]:
        raise ValueError("parent rejections were changed")
    parent_seen, child_seen, counts = set(), set(), _empty_counts()
    for split in SPLITS:
        pairs = zip_longest(_rows(parent, split, parent_seen), _rows(child, split, child_seen))
        for number, (original, observed) in enumerate(pairs, 1):
            if original is None or observed != review_sample(original):
                raise ValueError(f"reviewed row differs from deterministic parent revision: {split}:{number}")
            _count(counts, observed, split)
    summary = _summary(counts)
    if any(manifest.get(key) != value for key, value in summary.items()):
        raise ValueError("review manifest counts do not match verified rows")
    if manifest.get("labels_modified") is not False or manifest.get("source_data_modified") is not False:
        raise ValueError("review manifest must declare unchanged labels and parent data")
    _recheck(parent, parent_hashes)
    _recheck(child, child_hashes)
    return {"verified": True, **summary, "manifest_sha256": child_hashes["manifest.json"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--parent-root", type=Path, required=True)
        sub.add_argument("--output" if command == "build" else "--data-root", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = build_reviewed(**args) if command == "build" else verify_reviewed(**args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
