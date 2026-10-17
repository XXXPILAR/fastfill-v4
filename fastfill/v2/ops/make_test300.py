"""A fixed test subset for the LLM comparisons: N rooms of test.jsonl, stratified by source in proportion to the test
set (largest remainder; a source too small for one room gets none), each source's rooms drawn with one seeded RNG.
Rooms over the model capacity (more than --max-objects objects) are left out before drawing, so every method can
answer every room. Writes rows.jsonl (test order) and MANIFEST.json (seed, counts, sha256, scene ids).

    python fastfill/v2/ops/make_test300.py --test data/v4fixed/test.jsonl --output data/test300 [--rows 300 --seed 0]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


def allocate(sizes: dict, total: int) -> dict:
    """Largest-remainder apportionment of ``total`` over ``sizes`` (ties broken by name)."""
    whole = sum(sizes.values())
    quota = {k: total * v / whole for k, v in sizes.items()}
    share = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (-(quota[k] - share[k]), k))[:total - sum(share.values())]:
        share[k] += 1
    return share


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-objects", type=int, default=256)
    a = p.parse_args(argv)
    lines = [l for l in a.test.read_text().splitlines() if l.strip()]
    by_source, over = defaultdict(list), 0
    for index, line in enumerate(lines):
        row = json.loads(line)
        if len(row["condition"]["objects"]) > a.max_objects:
            over += 1
            continue
        by_source[row["provenance"]["source"]].append(index)
    share, rng, chosen = allocate({k: len(v) for k, v in by_source.items()}, a.rows), random.Random(a.seed), []
    for source in sorted(by_source):
        chosen += rng.sample(by_source[source], share[source])
    chosen.sort()
    a.output.mkdir(parents=True, exist_ok=True)
    rows = "".join(lines[i] + "\n" for i in chosen)
    (a.output / "rows.jsonl").write_text(rows)
    manifest = {"rows": len(chosen), "seed": a.seed, "max_objects": a.max_objects, "over_capacity_excluded": over,
                "test_file": a.test.name, "test_rows": len(lines), "test_sha256": hashlib.sha256(a.test.read_bytes()).hexdigest(),
                "rows_sha256": hashlib.sha256(rows.encode()).hexdigest(),
                "per_source": {k: {"test": len(by_source[k]), "chosen": share[k]} for k in sorted(by_source)},
                "test_line_indices": chosen,
                "scene_ids": [json.loads(lines[i])["provenance"].get("scene_id") for i in chosen]}
    (a.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "over_capacity_excluded", "rows_sha256")}),
          {k: v["chosen"] for k, v in manifest["per_source"].items()})


if __name__ == "__main__":
    main()
