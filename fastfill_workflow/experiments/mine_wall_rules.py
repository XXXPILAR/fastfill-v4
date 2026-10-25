"""Measure the training rooms for the critic's orientation and reachability terms (the even half, as the critic's
relation templates: ``zlib.crc32(line) & 1 == 0``).

  against the wall   per category, the share of floor obstacles standing with the back face's midpoint within
                     ``--gap`` of a wall and the front within ``--angle`` degrees of the wall's inward normal
                     (``critic.wall_back_error``); categories with a share of at least ``--share`` in at least
                     ``--min-rooms`` rooms become rules. Also the share facing a wall (front within 30 degrees of the
                     outward normal, back face away): a high share there means the front convention is flipped.
  facing             for every "faces" template of ``critic_templates.json``, the share of ground-truth pairs that pass
                     layout_quality's 45 degree test and the ray test (``critic.facing_error`` == 0).
  reachability       per person width, the share of objects (and of rooms with every object) reached from a door
                     (``critic.unreachable``).

    python experiments/mine_wall_rules.py --train ../data_v4_v2_fixed/train.jsonl --output critic_wall_rules.json
"""
import argparse, json, math, random, sys, zlib
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow.contract import _main  # noqa: E402,F401  (puts the fastfill package on sys.path)
import numpy as np  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from fastfill.v2.cell_costs import door_zones  # noqa: E402
from fastfill.v2.ops import layout_quality as lq  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402
from workflow import critic  # noqa: E402
from workflow.geometry import DOOR_DEPTH_M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--templates", default=str(HERE / "critic_templates.json"))
    ap.add_argument("--fraction", type=float, default=.1, help="share of the even half sampled (seeded)")
    ap.add_argument("--reach-fraction", type=float, default=.2, help="share of the sampled rooms also measured for reach")
    ap.add_argument("--min-rooms", type=int, default=150)
    ap.add_argument("--share", type=float, default=.8,
                    help="above layout_quality's 0.7 FACES_SHARE: a wall rule pushes harder than a template, and sofas "
                         "(0.70) and desks (0.80) often stand free")
    ap.add_argument("--gap", type=float, default=critic.WALL_GAP_M)
    ap.add_argument("--angle", type=float, default=math.degrees(critic.WALL_ANGLE_RAD))
    ap.add_argument("--widths", default="0.4,0.5,0.6")
    ap.add_argument("--reach-width", default=None, help="person width the critic uses (default: the first width)")
    ap.add_argument("--reach-share", type=float, default=.9,
                    help="categories reached in at least this share of training rooms are checked")
    ap.add_argument("--min-reach-rooms", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    widths = [float(w) for w in a.widths.split(",")]
    faces = [t for t in critic.load_templates(a.templates) if t["kind"] == "faces"]
    wall = defaultdict(lambda: {"n": 0, "pass": 0, "faces_wall": 0, "rooms": set(), "gaps": [], "angles": []})
    face = defaultdict(lambda: [0, 0, 0, 0])  # pairs, 45-degree pass, ray pass, either
    reach = {w: {"objects": 0, "reached": 0, "rooms": 0, "rooms_all": 0, "doorless": 0} for w in widths}
    reach_cat = {w: defaultdict(lambda: [0, 0, set()]) for w in widths}  # objects, reached, rooms
    rooms = 0
    with open(a.train, "rb") as stream:
        for line in stream:
            if rng.random() > a.fraction or zlib.crc32(line) & 1:
                continue
            row = migrate_legacy_row(json.loads(line))
            room = row["condition"]["room"]
            floor = Polygon(room["floor_polygon_xy_m"]).buffer(0)
            if floor.geom_type != "Polygon":
                continue
            rooms += 1
            bs = lq.boxes(row, row["target"])
            obstacles = [b for b in bs if lq.obstacle(b)]
            wl = critic.walls(floor)
            for b in obstacles:
                error, gap, angle = critic.wall_back_error(b, wl, a.gap, math.radians(a.angle))
                s = wall[b["cat"]]
                s["n"] += 1
                s["pass"] += error == 0.
                s["rooms"].add(rooms)
                s["gaps"].append(gap)
                s["angles"].append(math.degrees(angle))
                front = Polygon([tuple(b["centre"] + b["front"] * b["size"][0] / 2 + b["side"] * b["size"][1] / 2 * k)
                                 for k in (1, -1)] + [tuple(b["centre"] + b["front"] * (b["size"][0] / 2 + .3)
                                                            + b["side"] * b["size"][1] / 2 * k) for k in (-1, 1)])
                s["faces_wall"] += any(front.distance(line_) <= 1e-6 and float(np.dot(b["front"], -normal)) >= math.cos(
                    math.radians(30)) for line_, normal in wl)
            by_cat = defaultdict(list)
            for b in obstacles:
                by_cat[b["cat"]].append(b)
            for t in faces:
                for x in by_cat.get(t["a"], ()):
                    targets = by_cat.get(t["b"])
                    if not targets:
                        continue
                    y = min(targets, key=lambda o: x["poly"].distance(o["poly"]))
                    to = y["centre"] - x["centre"]
                    if np.linalg.norm(to) <= 1e-6:
                        continue
                    c = face[(t["a"], t["b"])]
                    c[0] += 1
                    c[1] += math.degrees(math.acos(max(-1., min(1., float(np.dot(x["front"], to / np.linalg.norm(to))))))) \
                        <= lq.FACING_DEG
                    c[2] += critic.facing_error(x, y["poly"], centre_rad=None) == 0.
                    c[3] += critic.facing_error(x, y["poly"]) == 0.
            if rng.random() <= a.reach_fraction:
                doors = [Polygon(z).intersection(floor) for z in door_zones(room, DOOR_DEPTH_M)]
                fixed = lq.fixed_obstacles(row)
                targets = [b for b in obstacles if b["size"][2] >= lq.ACC_MIN_HEIGHT_M]
                for w in widths:
                    missed = critic.unreachable(floor, bs, fixed, doors, w / 2, blockers=False)
                    ids = {i for i, _, _ in missed}
                    for b in targets:
                        c = reach_cat[w][b["cat"]]
                        c[0] += 1
                        c[1] += b["id"] not in ids
                        c[2].add(rooms)
                    r = reach[w]
                    r["objects"] += len(targets)
                    r["reached"] += len(targets) - len(missed)
                    r["rooms"] += 1
                    r["rooms_all"] += not missed
                    r["doorless"] += not doors
    rules = []
    table = []
    for cat, s in sorted(wall.items(), key=lambda kv: -len(kv[1]["rooms"])):
        if len(s["rooms"]) < a.min_rooms:
            continue
        share = s["pass"] / s["n"]
        row = {"category": cat, "objects": s["n"], "rooms": len(s["rooms"]), "share": round(share, 3),
               "faces_wall_share": round(s["faces_wall"] / s["n"], 3),
               "gap_m_p50_p90": [round(float(np.quantile(s["gaps"], q)), 3) for q in (.5, .9)],
               "angle_deg_p50_p90": [round(float(np.quantile(s["angles"], q)), 1) for q in (.5, .9)]}
        table.append(row)
        if share >= a.share:
            rules.append({"category": cat, "share": row["share"], "rooms": row["rooms"]})
    facing = [{"a": k[0], "b": k[1], "pairs": c[0], "pass_45deg": round(c[1] / c[0], 3), "pass_ray": round(c[2] / c[0], 3),
               "pass_either": round(c[3] / c[0], 3)}
              for k, c in sorted(face.items(), key=lambda kv: -kv[1][0]) if c[0]]
    reach_out = {str(w): {"object_share": round(r["reached"] / max(r["objects"], 1), 4),
                          "room_share": round(r["rooms_all"] / max(r["rooms"], 1), 4), "rooms": r["rooms"],
                          "rooms_without_door": r["doorless"]} for w, r in reach.items()}
    width = float(a.reach_width) if a.reach_width else widths[0]
    reach_rules = sorted(cat for cat, (n, hit, rs) in reach_cat[width].items()
                         if len(rs) >= a.min_reach_rooms and hit / n >= a.reach_share)
    reach_table = {str(w): {cat: {"objects": n, "share": round(hit / n, 3), "rooms": len(rs)}
                            for cat, (n, hit, rs) in sorted(reach_cat[w].items(), key=lambda kv: -kv[1][0])
                            if len(rs) >= a.min_reach_rooms} for w in widths}
    out = {"train": str(a.train), "half": "even", "split": "zlib.crc32(line) & 1", "rooms_sampled": rooms,
           "fraction": a.fraction, "seed": a.seed, "min_rooms": a.min_rooms, "share": a.share,
           "gap_m": a.gap, "angle_deg": a.angle, "against_wall": rules, "categories": table, "facing": facing,
           "reach": reach_out,
           "reach_width_m": width, "reach_share": a.reach_share, "reach_categories": reach_rules,
           "reach_by_category": reach_table}
    Path(a.output).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{rooms} rooms; {len(rules)} against-wall rules -> {a.output}")
    for r in table:
        print(f"  {r['category']:<22} share {r['share']:.2f}  faces wall {r['faces_wall_share']:.2f}  rooms {r['rooms']:>5}"
              f"  gap p50/p90 {r['gap_m_p50_p90']}  angle p50/p90 {r['angle_deg_p50_p90']}")
    for f in facing:
        print(f"  faces {f['a']} -> {f['b']}: {f['pairs']} pairs, 45deg {f['pass_45deg']:.2f}, ray {f['pass_ray']:.2f}, "
              f"either {f['pass_either']:.2f}")
    for w, r in reach_out.items():
        print(f"  reach width {w} m: objects {r['object_share']:.3f}, rooms {r['room_share']:.3f} ({r['rooms']} rooms, "
              f"{r['rooms_without_door']} without a door)")
    print(f"  reach categories at {width} m (share >= {a.reach_share}): {reach_rules}")
    for cat, r in reach_table[str(width)].items():
        print(f"    {cat:<22} {r['share']:.2f} ({r['objects']})")


if __name__ == "__main__":
    main()
