"""Relation templates for the critic and for evaluation, mined from two disjoint halves of the training rows.

``fastfill.v2.ops.layout_quality mine`` reads a seeded 15 % sample of the whole training file, so the templates a critic
optimises and the templates REL is scored with would come from overlapping rooms. Here each training row goes to one
half by the CRC32 of its line (even: critic, odd: evaluation), and each half is mined with layout_quality's own
statistics (``boxes`` / ``pair_stats``, the same 80 % gap quantile and 70 % facing share, min rooms scaled to the half).
This keeps the rooms apart; the two halves still come from one distribution, so the templates are close to each
other, and a critic that optimises them can still inflate REL: REL is reported beside metrics the critic does not use.

    python experiments/mine_split_templates.py --train ../data_v4_v2_fixed/train.jsonl \\
        --critic critic_templates.json --eval experiments/quality50/eval_templates_odd.json
"""
import argparse, json, random, sys, zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow.contract import _main  # noqa: E402,F401  (puts the fastfill package on sys.path)
import numpy as np  # noqa: E402
from fastfill.v2.ops.layout_quality import (FACES_SHARE, NEAR_MAX_M, NEAR_QUANTILE, boxes,  # noqa: E402
                                            pair_stats)
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402


def templates(stats, min_rooms):
    gaps, faces, rooms = stats
    out = []
    for pair, n in rooms.items():
        if n < min_rooms:
            continue
        q = float(np.quantile(gaps[pair], NEAR_QUANTILE))
        if q <= NEAR_MAX_M:
            out.append({"a": pair[0], "b": pair[1], "kind": "near", "max_gap_m": round(q, 3), "rooms": n})
        share = float(np.mean(faces[pair]))
        if share >= FACES_SHARE:
            out.append({"a": pair[0], "b": pair[1], "kind": "faces", "share": round(share, 3), "rooms": n})
    return sorted(out, key=lambda t: -t["rooms"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--critic", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--fraction", type=float, default=.3, help="share of each half sampled (seeded)")
    ap.add_argument("--min-rooms", type=int, default=150, help="as layout_quality's default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    halves = {k: (defaultdict(list), defaultdict(list), defaultdict(int)) for k in (0, 1)}
    seen = {0: 0, 1: 0}
    with open(a.train, "rb") as stream:
        for line in stream:
            if rng.random() > a.fraction:
                continue
            half = zlib.crc32(line) & 1
            row = migrate_legacy_row(json.loads(line))
            seen[half] += 1
            gaps, faces, rooms = halves[half]
            for pair, values in pair_stats(boxes(row, row["target"])).items():
                rooms[pair] += 1
                gaps[pair] += [g for g, _ in values]
                faces[pair] += [f for _, f in values]
    for half, path in ((0, a.critic), (1, a.eval)):
        t = templates(halves[half], a.min_rooms)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"train": str(a.train), "half": "even" if half == 0 else "odd",
                                          "split": "zlib.crc32(line) & 1", "rooms_sampled": seen[half],
                                          "fraction": a.fraction, "seed": a.seed, "min_rooms": a.min_rooms,
                                          "templates": t}, indent=1) + "\n")
        print(f"{'critic' if half == 0 else 'eval'} ({'even' if half == 0 else 'odd'} half): {seen[half]} rooms, "
              f"{len(t)} templates -> {path}")


if __name__ == "__main__":
    main()
