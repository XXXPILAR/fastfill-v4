"""Contrastive constraint test set (FastFill v1 spec section 10.2): the same room and objects asked under two
different constraints that call for different layouts; a model that follows constraints satisfies both, one that
reproduces a memorised layout satisfies at most one.

    python -m fastfill.tools.contrast_set --rooms data/v4/test_clean_rooms.jsonl --out data/v4/test_contrast_rooms.jsonl
    python -m fastfill.evaluate --model M --rooms data/v4/test_contrast_rooms.jsonl --out eval/contrast
    python -m fastfill.tools.contrast_set --report eval/contrast/predictions.jsonl

Each pair is two rooms with the same boundary, fixed boxes and object list:
  A  one constraint that holds on the reference layout (the room's own objects)
  B  one constraint on the same subject that does NOT hold on the reference, with a WITNESS layout (the B room's
     objects) proving B feasible: only the subject is turned or moved, and the witness must pass the validator
     (valid, the subject clear of fixed boxes other than windows, no new main collision); A must fail on it, so
     the two answers have to differ.
Types: faces (a seat turned towards another piece) and near (a piece moved next to another one); subjects and
targets are furniture-sized (>= 0.1 m2 footprint, >= 0.4 m tall). Pairs whose witness search finds nothing are
skipped, never weakened.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import random

from shapely.geometry import Polygon

from fastfill.anchors import FLAT
from fastfill.scene import footprint, head, num
from fastfill.validate import check, holds

NEAR_D = 0.3             # the "near" gap used for both variants (m)
# turning a seat towards something else is a request people make; turning a toilet, bed or desk in place is not
SEATS = {"chair", "armchair", "seat", "sofa", "couch", "loveseat"}
MIN_FOOTPRINT_M2 = 0.1   # subjects and targets are furniture, not bins and toilet brushes
MIN_HEIGHT_M = 0.4
STEP_M = 0.1             # slide step of the near-witness search along each side of the anchor object


def _rng(uid):
    return random.Random(int(hashlib.sha1(f"contrast:{uid}".encode()).hexdigest(), 16))


def _furniture(o):
    """A floor piece of furniture size (not flat, not a small item)."""
    return not o.get("parent") and head(o) not in FLAT and o["size"][2] >= MIN_HEIGHT_M and \
        o["size"][0] * o["size"][1] >= MIN_FOOTPRINT_M2


def _movable(o, objs):
    """Furniture nothing stands on (moving it moves no child)."""
    return _furniture(o) and not any(c.get("parent") == o["id"] for c in objs)


def _witness_ok(room, ref_check, wit, subject):
    """The witness layout is a legal answer: valid, the subject clear of non-window fixed boxes, no new main
    collision involving the subject."""
    c = check(wit)
    if not c["valid"] or any(p[0] == subject for p in c["fixed_blocking"]):
        return False
    old = {tuple(sorted(p)) for p in ref_check["collisions"]["main"]}
    return all(tuple(sorted(p)) in old for p in c["collisions"]["main"] if subject in p)


def faces_pair(room, rng):
    objs, b = room["objects"], room["boundary"]
    ref = check(room)
    idx = {o["id"]: o for o in objs}
    subjects = [o for o in objs if head(o) in SEATS and _movable(o, objs)]
    rng.shuffle(subjects)
    for x in subjects:
        targets = [o for o in objs if o is not x and _furniture(o)
                   and not Polygon(footprint(o)).intersects(Polygon(footprint(x)))]
        yes = [t for t in targets if holds(["faces", x["id"], t["id"]], objs, b, idx)]
        no = [t for t in targets if not holds(["faces", x["id"], t["id"]], objs, b, idx)]
        rng.shuffle(yes)
        rng.shuffle(no)
        for y in yes:
            for z in no:
                yaw = math.radians(round(math.degrees(math.atan2(z["pos"][1] - x["pos"][1],
                                                                 z["pos"][0] - x["pos"][0]))) % 360)
                wit = copy.deepcopy(room)
                w = {o["id"]: o for o in wit["objects"]}
                w[x["id"]]["yaw"] = yaw
                if holds(["faces", x["id"], z["id"]], wit["objects"], b, w) and \
                        not holds(["faces", x["id"], y["id"]], wit["objects"], b, w) and _witness_ok(room, ref, wit, x["id"]):
                    return ["faces", x["id"], y["id"]], ["faces", x["id"], z["id"]], wit
    return None


def _slides(a, c):
    """Candidate centres for `a` (yaw kept) beside each side of c's footprint, 5 cm gap, sliding along the side."""
    ca, sa = math.cos(c["yaw"]), math.sin(c["yaw"])
    hx, hy = c["size"][0] / 2, c["size"][1] / 2
    ext = max(a["size"][0], a["size"][1]) / 2 + 0.05           # a's farthest extent + gap: never overlaps c
    for nx, ny, half in ((1, 0, hy), (-1, 0, hy), (0, 1, hx), (0, -1, hx)):
        off = (hx if nx else hy) + ext
        tx, ty = -ny, nx                                         # along the side
        k = 0
        while k * STEP_M <= half:
            for t in ((0.0,) if k == 0 else (k * STEP_M, -k * STEP_M)):
                lx, ly = nx * off + tx * t, ny * off + ty * t
                yield c["pos"][0] + ca * lx - sa * ly, c["pos"][1] + sa * lx + ca * ly
            k += 1


def near_pair(room, rng):
    objs, b = room["objects"], room["boundary"]
    ref = check(room)
    idx = {o["id"]: o for o in objs}
    subjects = [o for o in objs if _movable(o, objs)]
    rng.shuffle(subjects)
    for a in subjects:
        others = [o for o in objs if o is not a and _furniture(o)]
        yes = [o for o in others if holds(["near", a["id"], o["id"], NEAR_D], objs, b, idx)]
        no = [o for o in others if not holds(["near", a["id"], o["id"], NEAR_D], objs, b, idx)]
        rng.shuffle(yes)
        rng.shuffle(no)
        for bb in yes[:2]:
            for c in no[:4]:
                for x, y in _slides(a, c):
                    wit = copy.deepcopy(room)
                    w = {o["id"]: o for o in wit["objects"]}
                    w[a["id"]]["pos"] = [num(x), num(y), a["pos"][2]]
                    if holds(["near", a["id"], c["id"], NEAR_D], wit["objects"], b, w) and \
                            not holds(["near", a["id"], bb["id"], NEAR_D], wit["objects"], b, w) and \
                            _witness_ok(room, ref, wit, a["id"]):
                        return ["near", a["id"], bb["id"], NEAR_D], ["near", a["id"], c["id"], NEAR_D], wit
    return None


def build_pairs(rooms):
    out, stats = [], collections.Counter()
    for room in rooms:
        rng = _rng(room["uid"])
        for kind, fn in (("faces", faces_pair), ("near", near_pair)):
            got = fn(room, rng)
            stats[f"{kind}:{'pair' if got else 'none'}"] += 1
            if not got:
                continue
            ca, cb, wit = got
            pid = f"{room['uid']}#{kind}"
            meta = lambda r, v: {**r.get("meta", {}), "pair_id": pid, "variant": v, "contrast": kind}
            out.append({**room, "uid": pid + ":A", "constraints": [ca], "meta": meta(room, "A")})
            out.append({**wit, "uid": pid + ":B", "constraints": [cb], "meta": meta(wit, "B")})
    return out, stats


def report(path):
    """Pair-level scores from evaluate's predictions.jsonl: a pair counts when both answers are valid and satisfy
    their constraint; `same_answer` counts pairs answered with the same layout text (constraints ignored)."""
    rows = [json.loads(l) for l in open(path)]
    pairs = collections.defaultdict(dict)
    for r in rows:
        pid, v = r["uid"].rsplit(":", 1)
        pairs[pid][v] = r
    by = collections.defaultdict(collections.Counter)
    for pid, p in pairs.items():
        if set(p) != {"A", "B"}:
            continue
        kind = pid.rsplit("#", 1)[1]
        ok = {v: bool(p[v].get("valid")) and all(h for _, h in p[v].get("constraints", [])) for v in "AB"}
        c = by[kind]
        c["pairs"] += 1
        c["A_ok"] += ok["A"]
        c["B_ok"] += ok["B"]
        c["both_ok"] += ok["A"] and ok["B"]
        c["same_answer"] += p["A"].get("output") is not None and p["A"].get("output") == p["B"].get("output")
    out = {k: {"pairs": c["pairs"], **{m: round(c[m] / c["pairs"], 4) for m in ("A_ok", "B_ok", "both_ok", "same_answer")}}
           for k, c in by.items()}
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", help="test_clean_rooms.jsonl (canonical IR from fastfill.build)")
    ap.add_argument("--out", help="output rooms jsonl for fastfill.evaluate")
    ap.add_argument("--report", help="predictions.jsonl of fastfill.evaluate on the contrast rooms")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return
    rooms = [json.loads(l) for l in open(a.rooms)]
    out, stats = build_pairs(rooms)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"rooms": len(rooms), "pairs": len(out) // 2, **dict(sorted(stats.items()))}, indent=1))


if __name__ == "__main__":
    main()
