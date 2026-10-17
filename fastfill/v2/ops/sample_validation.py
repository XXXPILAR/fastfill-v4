"""Source-stratified random validation sample for diag_ablation.py, the same scenes in two data versions.

    python sample_validation.py --new data/main-20261008a/validation.jsonl --old data/main-20261007b/validation.jsonl \
        --checkpoint RUN_A/model-step-N --checkpoint RUN_B/model-step-M --rows 500 --seed 0 --out-dir DIR

diag_ablation.py takes the first --rows eligible rows in file order (the file is grouped by source). This writes
DIR/new.jsonl and DIR/old.jsonl holding the same sampled scenes in the same order, each row eligible exactly as
diag_ablation checks it for EVERY given checkpoint: none of the runs' validation exclude_flags, a three-field projection
exists (evaluate.project_minimal), at most max_objects objects, and the projection collates with the run's tokenizer
and max_length; so ``diag_ablation.py --validation DIR/new.jsonl --rows 500`` scores exactly these rows in this order.
Allocation: proportional to each source's eligible-scene count (largest remainder, at least one per source); within a
source, a seeded random order. DIR/sample.json records the allocation, scene ids and file hashes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from fastfill.v2.batch import collate_samples, load_tokenizer
from fastfill.v2.data import filter_rows_by_flags
from fastfill.v2.evaluate import project_minimal
from fastfill.v2.io import fingerprint, load_checkpoint_config


def allocate(sizes, total):
    """Largest-remainder proportional allocation with a floor of one per non-empty stratum."""
    keys = sorted(k for k in sizes if sizes[k])
    if total < len(keys):
        raise ValueError("fewer rows than sources")
    base = {k: 1 for k in keys}
    rest, pool = total - len(keys), sum(sizes[k] - 1 for k in keys)
    quota = {k: rest * (sizes[k] - 1) / pool if pool else 0 for k in keys}
    for k in keys:
        base[k] += int(quota[k])
    for k in sorted(keys, key=lambda k: (int(quota[k]) - quota[k], k))[:total - sum(base.values())]:
        base[k] += 1
    return {k: min(base[k], sizes[k]) for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", required=True, type=Path)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--checkpoint", action="append", required=True, type=Path)
    p.add_argument("--rows", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(argv)
    load = lambda path: [json.loads(line) for line in open(path) if line.strip()]
    new, old = load(a.new), load(a.old)
    old_by_scene = {r["provenance"]["scene_id"]: r for r in old}
    if len(old_by_scene) != len(old) or {r["provenance"]["scene_id"] for r in new} != set(old_by_scene):
        raise SystemExit("refused: the two files do not hold the same scenes")
    runs = []
    for ckpt in a.checkpoint:
        start = json.loads((ckpt.parent / "run_manifest_start.json").read_text())["resolved_config"]
        model = json.loads((ckpt / "model_config.json").read_text())
        runs.append({"checkpoint": str(ckpt), "exclude_flags": start.get("validation", {}).get("exclude_flags", []),
                     "max_objects": model["max_objects"], "max_length": load_checkpoint_config(ckpt)["max_length"] or 4096,
                     "tokenizer": load_tokenizer("tiny" if model["backbone"] == "tiny" else str(ckpt.parent / "tokenizer"),
                                                 local_files_only=True)})
    flags = sorted({f for run in runs for f in run["exclude_flags"]})
    pool = {}
    for row in filter_rows_by_flags(new, flags):
        if row["provenance"].get("split") != "validation":
            raise SystemExit("refused: a non-validation row")
        twin = old_by_scene[row["provenance"]["scene_id"]]
        if filter_rows_by_flags([twin], flags) and project_minimal(row) is not None and project_minimal(twin) is not None \
                and len(row["condition"]["objects"]) <= min(run["max_objects"] for run in runs):
            pool.setdefault(str(row["provenance"]["source"]), []).append((row, twin))
    quota = allocate({k: len(v) for k, v in pool.items()}, a.rows)
    rng, chosen, skipped = random.Random(a.seed), {}, {}

    def collates(row):
        m = project_minimal(row)
        try:
            for run in runs:
                collate_samples([m], run["tokenizer"], max_length=run["max_length"], max_objects=run["max_objects"])
        except ValueError:
            return False
        return True

    for source in sorted(quota):
        candidates = pool[source][:]
        rng.shuffle(candidates)
        chosen[source] = []
        for row, twin in candidates:
            if len(chosen[source]) == quota[source]:
                break
            if collates(row) and collates(twin):
                chosen[source].append((row, twin))
            else:
                skipped[source] = skipped.get(source, 0) + 1
    picked = [pair for source in sorted(chosen) for pair in chosen[source]]
    random.Random(a.seed + 1).shuffle(picked)  # file order mixes sources
    a.out_dir.mkdir(parents=True, exist_ok=False)
    for name, index in (("new.jsonl", 0), ("old.jsonl", 1)):
        with open(a.out_dir / name, "w") as stream:
            stream.writelines(json.dumps(pair[index]) + "\n" for pair in picked)
    record = {"rows": len(picked), "seed": a.seed, "new": str(a.new.resolve()), "new_sha256": fingerprint(a.new),
              "old": str(a.old.resolve()), "old_sha256": fingerprint(a.old), "exclude_flags": flags,
              "checkpoints": [{k: v for k, v in run.items() if k != "tokenizer"} for run in runs],
              "eligible_scenes_by_source": {k: len(v) for k, v in sorted(pool.items())}, "allocation": quota,
              "collate_skipped_by_source": skipped, "chosen_by_source": {k: len(v) for k, v in chosen.items()},
              "scene_ids": [pair[0]["provenance"]["scene_id"] for pair in picked],
              "new_sample_sha256": fingerprint(a.out_dir / "new.jsonl"), "old_sample_sha256": fingerprint(a.out_dir / "old.jsonl")}
    (a.out_dir / "sample.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("rows", "allocation", "collate_skipped_by_source", "new_sample_sha256")}, indent=1))
    assert len(picked) == a.rows, "a source ran out of collatable rows"


if __name__ == "__main__":
    assert allocate({"a": 10, "b": 1, "c": 89}, 10) == {"a": 2, "b": 1, "c": 7}
    main()
