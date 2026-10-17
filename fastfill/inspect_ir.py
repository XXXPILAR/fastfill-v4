"""Diagnostics for IR files: counts, GT geometry checks, and the front-direction test.

    python -m fastfill.inspect_ir /Volumes/harddisk/fastfill_ir/IL3D.jsonl [--limit 5000]

Front test: for wall-hugging categories (bed, sofa, wardrobe, toilet, ...) the side of the footprint that
touches the boundary should be the back, i.e. local -X under the IR convention (front = local +X).
A different dominant side means the adapter's front_offset_deg is wrong.
"""
import argparse
import collections
import json
import math
import re

from shapely.geometry import Point, Polygon

from fastfill.validate import check

WALL_HUGGERS = re.compile(r"\b(bed|sofa|couch|wardrobe|closet|toilet|dresser|bookcase|bookshelf|tv[ _]?stand|"
                          r"refrigerator|fridge|sideboard|nightstand|armoire)\b", re.I)


def wall_side(o, ring, tol=0.15):
    """Which local axis side (+X, -X, +Y, -Y) of the footprint lies on the boundary, or None."""
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    x, y = o["pos"][0], o["pos"][1]
    best, name = tol, None
    for lab, (dx, dy) in {"+X": (hx, 0), "-X": (-hx, 0), "+Y": (0, hy), "-Y": (0, -hy)}.items():
        d = ring.distance(Point(x + c * dx - s * dy, y + s * dx + c * dy))
        if d < best:
            best, name = d, lab
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for path in a.files:
        n, nobj, btypes, rtypes, cats = 0, [], collections.Counter(), collections.Counter(), collections.Counter()
        meta = collections.Counter()
        geo = collections.Counter()
        sides = collections.defaultdict(collections.Counter)
        for line in open(path):
            r = json.loads(line)
            n += 1
            nobj.append(len(r["objects"]))
            btypes[r.get("boundary_type")] += 1
            rtypes[r.get("room_type")] += 1
            meta["rooms_with_tilted"] += bool(r["meta"].get("n_tilted"))
            meta["rooms_with_incomplete"] += bool(r["meta"].get("n_incomplete"))
            for o in r["objects"]:
                cats[o["category"]] += 1
            if r.get("boundary") and r["objects"] and (a.limit is None or geo["rooms"] < a.limit):
                c = check(r)
                geo["rooms"] += 1
                geo["objects"] += c["n"]
                geo["valid"] += c["valid"]
                geo["oob_objects"] += len(c["oob"])
                geo["rooms_with_collision"] += bool(c["collisions"]["main"])
                geo["oor_sum"] += c["oor"]
                ring = Polygon(r["boundary"]).exterior
                for o in r["objects"]:
                    m = WALL_HUGGERS.search(o["category"] or "")
                    if m:
                        sides[m.group(1).lower()][wall_side(o, ring)] += 1
        nobj.sort()
        q = lambda f: nobj[min(int(f * len(nobj)), len(nobj) - 1)] if nobj else None
        print(f"== {path}\nrooms {n}  objects/room p10 {q(.1)} p50 {q(.5)} p90 {q(.9)} max {q(1)}")
        print("boundary_type", dict(btypes), "\nmeta", dict(meta))
        print("room_type top", rtypes.most_common(12))
        print("category top", cats.most_common(25), f"({len(cats)} distinct)")
        if geo["rooms"]:
            g = geo
            print(f"GT geometry on {g['rooms']} rooms: valid {g['valid'] / g['rooms']:.3f}  "
                  f"oob_object_rate {g['oob_objects'] / max(g['objects'], 1):.3f}  "
                  f"rooms_with_main_collision {g['rooms_with_collision'] / g['rooms']:.3f}  mean_oor {g['oor_sum'] / g['rooms']:.3f}")
        print("front test (side touching the wall; expect -X):")
        for k, v in sorted(sides.items(), key=lambda kv: -sum(kv[1].values()))[:10]:
            tot = sum(v.values())
            print(f"  {k:12s} n={tot:6d} " + " ".join(f"{s}:{v[s] / tot:.2f}" for s in ("-X", "+X", "-Y", "+Y", None)))


if __name__ == "__main__":
    main()
