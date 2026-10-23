"""Calibrate the hard checks on ground-truth layouts of the validation split (never the test split).

For each candidate threshold: the share of validation ground-truth rooms that pass the check (a real layout should
pass), and the share of corrupted copies it catches (2 objects moved 0.5-1.5 m or turned, as in OptiScene's
injected violations; a broken layout should fail). Ground-truth supports are inferred from geometry: the floor, a
wall, or the box under the object whose footprint holds its centre and whose height span holds its bottom.

    python experiments/calibrate_checker.py --rooms 1000
"""
import argparse, json, math, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shapely.geometry import Point  # noqa: E402
from workflow.contract import fill_room_defaults  # noqa: E402
from workflow.geometry import _opening_strip, footprint  # noqa: E402
from fastfill.v2.validation import on_wall  # noqa: E402

COLLISION_RATIOS = (0.0, 0.05, 0.10, 0.20, 0.30)
DOOR_DEPTHS = (0.8, 0.6, 0.4, 0.3, 0.2)
BOUNDARY_TOLS = (1e-5, 0.02, 0.05, 0.10)
CEILING_TOLS = (1e-5, 0.05, 0.10, 0.20)
SUPPORT_TOP_TOL = 0.03


def boxes(row):
    room = row["condition"]["room"]
    gt = [{"id": o["id"], "target_size_local_m": o["target_size_local_m"], "bottom_center_m": o["bottom_center_m"],
           "yaw_rad": o["yaw_rad"]} for o in row["target"]["objects"]]
    fixed = [{"id": o["id"], "target_size_local_m": o["size_local_m"], "bottom_center_m": o["bottom_center_m"],
              "yaw_rad": o["yaw_rad"]} for o in room.get("fixed_objects", [])]
    return gt, fixed


def infer_parents(room, gt, fixed):
    floor = room["floor_z_m"]
    everything = gt + fixed
    parents = {}
    for b in gt:
        z = b["bottom_center_m"][2]
        if abs(z - floor) <= 0.03:
            parents[b["id"]] = "floor"; continue
        centre = Point(b["bottom_center_m"][:2])
        under = [o for o in everything if o["id"] != b["id"] and footprint(o).buffer(0.02).contains(centre)
                 and o["bottom_center_m"][2] - 0.01 <= z <= o["bottom_center_m"][2] + o["target_size_local_m"][2]
                 + SUPPORT_TOP_TOL and o["bottom_center_m"][2] < z]
        if under:
            parents[b["id"]] = max(under, key=lambda o: o["bottom_center_m"][2] + o["target_size_local_m"][2])["id"]
        elif on_wall(b, room, tolerance=0.1):
            parents[b["id"]] = "wall"
        else:
            parents[b["id"]] = None  # floating: no support found
    return parents


def ancestors(parents, ident):
    seen = set()
    while ident in parents and parents[ident] not in (None, "floor", "wall") and ident not in seen:
        seen.add(ident)
        ident = parents[ident]
        yield ident


def room_checks(row, gt, parents):
    """Per check and threshold: does the room pass?"""
    room = row["condition"]["room"]
    _, fixed = boxes(row)
    poly = __import__("shapely.geometry", fromlist=["Polygon"]).Polygon(room["floor_polygon_xy_m"])
    ceiling = room["floor_z_m"] + room["height_m"]
    shapes = {b["id"]: footprint(b) for b in gt + fixed}
    out = {}
    # collision: worst overlap share (intersection / smaller footprint) among non-exempt pairs
    worst = 0.0
    items = gt + fixed
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if i >= len(gt):
                continue  # two fixed objects: not ours to judge
            if min(a["target_size_local_m"][2], b["target_size_local_m"][2]) <= 0.05:
                continue  # rugs, mats: thin objects may lie under others
            if b["id"] in ancestors(parents, a["id"]) or a["id"] in ancestors(parents, b["id"]):
                continue  # an object and what it stands on (incl. shelf / cabinet interiors)
            az, bz = a["bottom_center_m"][2], b["bottom_center_m"][2]
            if min(az + a["target_size_local_m"][2], bz + b["target_size_local_m"][2]) - max(az, bz) <= 1e-5:
                continue
            inter = shapes[a["id"]].intersection(shapes[b["id"]]).area
            if inter > 1e-5:
                worst = max(worst, inter / max(1e-9, min(shapes[a["id"]].area, shapes[b["id"]].area)))
    out["collision"] = {r: worst <= r + 1e-9 for r in COLLISION_RATIOS}
    # boundary
    out["boundary"] = {t: all(poly.buffer(t).covers(shapes[b["id"]]) for b in gt) for t in BOUNDARY_TOLS}
    out["ceiling"] = {t: all(b["bottom_center_m"][2] + b["target_size_local_m"][2] <= ceiling + t for b in gt)
                      for t in CEILING_TOLS}
    # doors: a floor-standing, non-thin object in the strip in front of a door
    door = {}
    for depth in DOOR_DEPTHS:
        ok = True
        for opening in room.get("openings", []):
            if opening.get("type") != "door":
                continue
            strip = _opening_strip(room["floor_polygon_xy_m"], opening, depth)
            if strip is None:
                continue
            strip = strip.intersection(poly)
            for b in gt:
                if b["target_size_local_m"][2] > 0.05 and b["bottom_center_m"][2] < room["floor_z_m"] + 2.0 and \
                        shapes[b["id"]].intersection(strip).area > 1e-4:
                    ok = False
        door[depth] = ok
    out["door"] = door
    out["support"] = {"found": all(p is not None for p in parents.values())}
    return out


def corrupt(gt, room, rng):
    gt = json.loads(json.dumps(gt))
    floor_objects = [b for b in gt if abs(b["bottom_center_m"][2] - room["floor_z_m"]) <= 0.03
                     and b["target_size_local_m"][2] > 0.05]
    for b in rng.sample(floor_objects, min(2, len(floor_objects))):
        angle, dist = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.5)
        b["bottom_center_m"][0] += dist * math.cos(angle)
        b["bottom_center_m"][1] += dist * math.sin(angle)
        b["yaw_rad"] += rng.choice([0, math.pi / 2, math.pi / 4])
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data_v4_v2_fixed" / "validation.jsonl"))
    ap.add_argument("--rooms", type=int, default=1000)
    a = ap.parse_args()
    lines = [l for l in open(a.data) if l.strip()]
    random.Random(0).shuffle(lines)
    rng = random.Random(1)
    real, broken, n = Counter(), Counter(), 0
    for line in lines:
        if n >= a.rooms:
            break
        row = json.loads(line)
        v = row.get("validity", {})
        if not all(all(x) for key in ("position", "size") for x in v.get(key, [])) or not all(
                (x[0] if isinstance(x, list) else x) for x in v.get("yaw", [])):
            continue
        if not row["target"]["objects"] or len(row["target"]["objects"]) > 120:
            continue
        row["condition"], _ = fill_room_defaults(row["condition"])
        room = row["condition"]["room"]
        gt, fixed = boxes(row)
        parents = infer_parents(room, gt, fixed)
        n += 1
        for check, values in room_checks(row, gt, parents).items():
            for key, ok in values.items():
                real[(check, key)] += ok
        bad = corrupt(gt, room, rng)
        for check, values in room_checks(row, bad, parents).items():
            for key, ok in values.items():
                broken[(check, key)] += ok
    print(f"{n} validation rooms (complete labels, <=120 objects)")
    print(f"{'check':10s} {'threshold':>10s} {'real pass':>10s} {'corrupted pass':>15s}")
    for (check, key) in sorted(real, key=lambda k: (k[0], str(k[1]))):
        print(f"{check:10s} {str(key):>10s} {real[(check, key)] / n:10.1%} {broken[(check, key)] / n:15.1%}")


if __name__ == "__main__":
    main()
