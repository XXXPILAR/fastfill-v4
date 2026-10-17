"""Build immutable FastFill-v2 splits from read-only audited annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from fastfill.v2.adapters import iter_multiscan

DEFAULT_SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")


def grouped_split(source: str, house_id: str, seed: int = 42) -> str:
    """Stable 80/10/10 partition of source/house identity before derivation."""
    key = json.dumps([seed, source, str(house_id)], separators=(",", ":")).encode()
    fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "train" if fraction < .8 else "validation" if fraction < .9 else "test"


def filter_rows_by_flags(rows, flags):
    """Drop rows whose provenance.legacy_flags report any listed flag (same truthiness rule as the train build)."""
    if not isinstance(flags, (list, tuple)) or any(not isinstance(flag, str) or not flag for flag in flags):
        raise ValueError("exclude_flags must be a list of nonempty flag names")
    return [row for row in rows
            if not any(row.get("provenance", {}).get("legacy_flags", {}).get(flag) for flag in flags)]


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_path(source_root, output):
    source, target = Path(source_root).resolve(), Path(output).resolve()
    protected = (source, DEFAULT_SOURCE_ROOT.resolve())
    if any(target == root or root in target.parents or target in root.parents for root in protected):
        raise ValueError("output must be an explicitly new path outside source data and its ancestors")
    if target.exists():
        raise FileExistsError(f"immutable build output already exists: {target}")
    return source, target


def build_dataset(source_root, output, *, sources=("multiscan",), seed=42, max_scenes=None):
    """Build admitted sources without overwrite or writes beneath source_root.

    Only raw MultiScan geometry is admitted initially. All other local sources
    remain audit-only until their canonical front/pivot/scale blockers are
    resolved. `max_scenes` bounds admitted output, not changes the split rule.
    """
    source, target = _output_path(source_root, output)
    if tuple(sources) != ("multiscan",):
        raise ValueError("currently audited builder source is multiscan; uncertain sources are not admitted")
    if max_scenes is not None and max_scenes < 1:
        raise ValueError("max_scenes must be positive")
    # Check readable source files before creating output; sources stay read-only.
    for name in ("objects.csv", "regions.csv", "scans.csv"):
        if not (source / "XXXpilar__multiscan-clean" / name).is_file():
            raise FileNotFoundError(f"required raw source missing: {name}")
    target.mkdir(parents=True, exist_ok=False)
    counts, objects, houses = Counter(), Counter(), {}
    diagnostics = {"scenes_inspected": 0, "rejections": []}
    with ExitStack() as stack:
        handles = {split: stack.enter_context((target / f"{split}.jsonl").open("x"))
                   for split in ("train", "validation", "test")}
        for sample in iter_multiscan(source, diagnostics):
            provenance = sample["provenance"]
            split = grouped_split(provenance["source"], provenance["house_id"], seed)
            houses[f"{provenance['source']}:{provenance['house_id']}"] = split
            record = {**sample, "provenance": {**provenance, "split": split}}
            handles[split].write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            counts[split] += 1
            objects[split] += len(record["target"]["objects"])
            if max_scenes is not None and sum(counts.values()) >= max_scenes:
                break
    manifest = {"schema_version": "fastfill.v2", "source_root": str(source), "builder_version": 1,
                "sources": list(sources), "samples_written": sum(counts.values()),
                "objects_written": sum(objects.values()), "split_samples": dict(counts),
                "split_objects": dict(objects), "split_seed": seed, "split_rule":
                "SHA256(seed, source, underlying scene/house) -> 80/10/10 before derivation",
                "house_split": houses, "bounded_build": max_scenes is not None,
                "source_data_modified": False, "diagnostics": diagnostics,
                "source_sha256": {name: _digest(source / "XXXpilar__multiscan-clean" / name)
                                  for name in ("objects.csv", "regions.csv", "scans.csv")},
                "adapter_sha256": _digest(Path(__file__).with_name("adapters.py")),
                "builder_sha256": _digest(__file__),
                "boundary_validation_eligible": False, "eligibility":
                "upright annotated-front local OBB supervision; scan floor hull/contact remain incomplete"}
    with (target / "manifest.json").open("x") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)
        fh.write("\n")
    return manifest


def main(argv=None):
    from fastfill.v2.legacy_build import DEFAULT_RELEASE, DEFAULT_EVIDENCE
    from fastfill.v2.legacy_bridge import FRONT_POLICIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, required=True, help="New immutable output directory")
    parser.add_argument("--source", choices=["selected-v3.2", "multiscan"], default="selected-v3.2")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--front-policy", choices=FRONT_POLICIES, default="axis")
    parser.add_argument("--include-flagged", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int)
    args = parser.parse_args(argv)
    if args.source == "selected-v3.2":
        from fastfill.v2.legacy_build import build_selected_dataset
        manifest = build_selected_dataset(args.release_root, args.output, evidence_root=args.evidence_root,
            seed=args.seed, max_scenes=args.max_scenes, front_policy=args.front_policy,
            include_flagged=args.include_flagged, workers=args.workers)
    else:
        manifest = build_dataset(args.source_root, args.output, sources=(args.source,),
                                 seed=args.seed, max_scenes=args.max_scenes)
    print(json.dumps({key: manifest[key] for key in ("samples_written", "split_samples")}, indent=2))


if __name__ == "__main__":
    main()
