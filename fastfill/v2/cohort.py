"""Create one immutable complete-label cohort for structured/text comparisons.

Eligible JSONL rows are copied byte-for-byte. Original splits and conditions
are preserved; missing or proxy labels are never filled from conditions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .batch import _geometry_rows, _mask_row, room_normalization
from .data import _digest, _output_path
from .io import ROOMGENBENCH_HOLDOUT_GROUPS
from .schema import migrate_legacy_row, validate_condition, validate_layout


SPLITS = ("train", "validation", "test")
INPUT_FILES = tuple(f"{split}.jsonl" for split in SPLITS) + ("manifest.json",)


def _parent_inputs(root):
    paths = [root / name for name in INPUT_FILES]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required parent input missing: {path}")
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "fastfill.v2":
        raise ValueError("parent manifest must identify fastfill.v2 data")
    hashes = {str(path): _digest(path) for path in paths if path != manifest_path}
    return manifest, {**hashes, str(manifest_path): hashlib.sha256(manifest_bytes).hexdigest()}


def _exact_ids(sample):
    condition = sample.get("condition", {})
    targets = sample.get("target", {}).get("objects")
    objects = condition.get("objects")
    if not isinstance(objects, list) or not isinstance(targets, list):
        raise ValueError("request and target objects must be lists")
    ids = []
    for collection in (objects, targets):
        if any(not isinstance(obj, dict) or not isinstance(obj.get("id"), str)
               or not obj["id"] for obj in collection):
            raise ValueError("request/target IDs must be nonempty strings")
        values = [obj["id"] for obj in collection]
        if len(values) != len(set(values)):
            raise ValueError("duplicate request/target IDs")
        ids.append(set(values))
    if ids[0] != ids[1]:
        raise ValueError("target IDs must equal all requested IDs exactly once")
    return len(objects)


def _complete(sample):
    """Only complete rows need expensive geometry validation; none are repaired."""
    if not isinstance(sample, dict) or sample.get("schema_version") != "fastfill.v2":
        raise ValueError("parent rows must use fastfill.v2 schema")
    count = _exact_ids(sample)
    if not count:
        return False
    validity = sample.get("validity", {})
    if not isinstance(validity, dict):
        raise ValueError("validity must be an object")
    masks = []
    for field, dimensions in (("position", 3), ("size", 3), ("yaw", 1)):
        rows = validity.get(field, [])
        if not isinstance(rows, list) or len(rows) > count:
            raise ValueError(f"invalid {field} validity row count")
        masks.extend(_mask_row(validity, field, index, dimensions) for index in range(count))
    if not all(all(row) for row in masks):
        return False
    sample = migrate_legacy_row(sample)  # pre-C1 parents keep groups in objects; output bytes stay identical
    condition = sample["condition"]
    validate_condition(condition)
    validate_layout(sample["target"], condition)
    origin, scale = room_normalization(condition["room"])
    geometry = _geometry_rows(sample, origin, scale)
    return (all(all(row) for row in geometry["position_valid"] + geometry["size_valid"])
            and all(geometry["yaw_valid"]))


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number is not a valid geometry label: {value}")


def _write_cohort(source, staging):
    counts, skipped, inputs, sources, objects = Counter(), Counter(), Counter(), Counter(), Counter()
    with ExitStack() as stack:
        outputs = {split: stack.enter_context((staging / f"{split}.jsonl").open("xb")) for split in SPLITS}
        for split in SPLITS:
            with (source / f"{split}.jsonl").open("rb") as stream:
                for number, raw in enumerate(stream, 1):
                    if not raw.strip():
                        continue
                    inputs[split] += 1
                    try:
                        sample = json.loads(raw, parse_constant=_reject_constant)
                        eligible = _complete(sample)
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f"Invalid parent row {split}.jsonl:{number}: {exc}") from exc
                    if split != "test" and sample.get("provenance", {}).get("group") in ROOMGENBENCH_HOLDOUT_GROUPS:
                        raise ValueError(f"RoomGenBench benchmark room in {split}.jsonl:{number}; build the cohort from qualified data")
                    if not eligible:
                        skipped[split] += 1
                        continue
                    outputs[split].write(raw)
                    counts[split] += 1
                    objects[split] += len(sample["target"]["objects"])
                    sources[f"{sample.get('provenance', {}).get('source', 'unknown')}:{split}"] += 1
    return counts, skipped, inputs, sources, objects


def build_complete_cohort(data_root, output):
    """Publish a new same-row cohort only after parent fingerprints recheck."""
    if Path(output).is_symlink():
        raise FileExistsError(f"immutable cohort output already exists as a symlink: {output}")
    source, target = _output_path(data_root, output)
    parent, hashes = _parent_inputs(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary)
        counts, skipped, inputs, sources, objects = _write_cohort(source, staging)
        if not counts["train"]:
            raise ValueError("no eligible training rows with complete geometry labels")
        result = {"schema_version": "fastfill.v2", "builder": "complete-supervision-cohort-v1",
                  "data_root": str(source), "parent_builder": parent.get("builder"),
                  "parent_front_policy": parent.get("front_policy"), "parent_hashes": hashes,
                  "samples_written": sum(counts.values()), "split_samples": {s: counts[s] for s in SPLITS},
                  "split_skipped": {s: skipped[s] for s in SPLITS},
                  "split_input_samples": {s: inputs[s] for s in SPLITS},
                  "split_objects": {s: objects[s] for s in SPLITS}, "source_split_samples": dict(sources),
                  "eligibility": "Every requested object has valid full XYZ position, local size and yaw labels",
                  "comparison_scope": "Same complete-label rows for structured and text models; conditions and labels unchanged",
                  "split_rule": "Inherit parent splits without re-splitting",
                  "rows_byte_identical": True, "source_data_modified": False,
                  "output_sha256": {f"{s}.jsonl": _digest(staging / f"{s}.jsonl") for s in SPLITS},
                  "implementation_sha256": _digest(__file__)}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if any(_digest(path) != expected for path, expected in hashes.items()):
            raise ValueError("parent input or manifest changed during complete-cohort selection")
        if target.exists() or target.is_symlink():
            raise FileExistsError("immutable cohort output appeared during selection")
        os.rename(staging, target)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = build_complete_cohort(**vars(parser.parse_args(argv)))
    print(json.dumps({key: result[key] for key in ("samples_written", "split_samples", "split_skipped")}, indent=2))


if __name__ == "__main__":
    main()
