"""v4 rooms -> FastFill v2 training rows (bbox = OUTPUT), in the real v2 protocol of fastfill-main.

    python -m fastfill.build_v2 --like ../data_v4/step2/MANIFEST.json --v2-root ../fastfill-main --out ../data_v4_v2

Rows are made by fastfill-main's own ``fastfill.v2.legacy_bridge.convert_selected_room`` (schema-validated
condition / target / validity / provenance, v2 yaw policy, K1, exchangeable groups, round-10 wall anchors and
in-shelf children), fed with this package's v4 preparation instead of the frozen v3.2 one:

  rooms / split   build.prep + dedup + split + exact duplicates of the --like v4 build, same arguments: a room lands
                  in the same split as there (dev -> validation). Train rows flagged DEFAULT_REJECT_FLAGS are left out
                  like fastfill.v2.legacy_build does (--include-flagged keeps them); eval rows keep every flag.
  v4 fronts       the bridge labels from raw IR; v4's bed re-fronting (+-90 deg, horizontal sizes swapped) and
                  back-to-wall flips (+180 deg) are applied to the raw box first and recorded in field_evidence.
  constraints     every room gets extract_constraints (train: salt s2, eval: salt eval = the v4 *_constrained rooms);
                  the v2 trainer drops them with drop_constraints_p.
  no input bbox   no target (placed) object's box enters the condition. Fixed doors / windows (opening_type: a
                  word taxonomy, not a word-boundary regex) become wall-relative room.openings (RoomGenBench shell
                  form: wall index = floor-polygon edge i -> i+1, position_on_wall = center fraction along it, width /
                  height / sill); a door/window farther than OPENING_MAX_WALL_DIST_M from every wall is removed
                  (legacy_flags.removed_detached_openings). Every other fixed box is room context, never placed:
                  the prep kinds in --fixed-kinds (default KEEP_FIXED_KINDS: structure = columns, radiators,
                  partitions, stairs...; floating / wall = built-in wall-hung units; no_front = built-in cabinetry)
                  stay as room.fixed_objects; the others (default: generic = unlabelled 'unknown' / 'otherprop'
                  boxes) are removed, counted in legacy_flags.removed_fixed_obstacles(_by_kind).
  descriptions    metric size phrases ("10 cm tall", "120cm x 60cm", "with an 8 cm diameter") are cut out of the
                  description and the rest kept (field_evidence.description_metric = "spans_removed"); a
                  description with nothing left but the size becomes the category ("category").
  evidence        the v3.2 audit evidence index (.release/audits) is not on this machine and is not applied; rooms the
                  bridge refuses without it (oblique IL3D windows) are listed in rejections.jsonl.
"""
import argparse
import collections
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import fastfill
from fastfill.build import (_input_files, _json_lines, _sha256, _validate_destination, dedup, exact_key,
                            extract_constraints, prep, rng_for)
from fastfill.scene import canonical, messages
from fastfill.split import assign_splits

SPLIT_NAMES = {"train": "train", "dev": "validation", "test": "test"}
SHARED = ("scene.py", "anchors.py", "validate.py", "split.py")   # must equal fastfill-main's: the bridge imports them
REJECT_FLAGS = ("oob_objects", "fixed_collision", "overlapping_furniture")   # = fastfill.v2.legacy_build
OPENING_MAX_WALL_DIST_M = 0.5
FIXED_KINDS = ("structure", "floating", "wall", "no_front", "generic")   # build.prep's fixed_kind
KEEP_FIXED_KINDS = ("structure", "floating", "wall", "no_front")
# opening_type: words of the category (letters only: "window1" -> window, "shower_door" -> shower, door)
DOOR_WORDS = {"door", "doors", "doorway", "doorways", "doorframe", "doorframes"}
NOT_ROOM_DOOR = {"shower", "cabinet", "cupboard", "wardrobe", "oven", "fridge", "refrigerator",
                 "microwave", "dishwasher", "car", "stove", "safe"}    # a door of an appliance or enclosure
WINDOW_WORDS = {"window", "windows"}
NOT_WINDOW = {"pad", "seat", "bench", "sill", "cushion", "curtain", "curtains", "blind", "blinds", "shade", "shades",
              "planter", "box"}                                       # bay window pad, window seat: furniture
METRIC = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|cm|m|meters?|metres?|inch(?:es)?|in|ft|feet|\"|”|米|厘米|毫米)(?![a-z])"
                    r"|\d+\s*[x×*]\s*\d+", re.I)
_NUM = r"\d+(?:[.,]\d+)?"
_UNIT = r"(?:mm|cm|m|meters?|metres?|inch(?:es)?|in|ft|feet|\"|”|米|厘米|毫米)"
# one size phrase: hedges / article, one measure or a dims product (8.5x11 inch, 2cm by 4cm), its dimension word
SIZE_PHRASE = re.compile(
    rf"(?:\b(?:an?|about|approx(?:imately|\.)?|around|roughly|nearly|almost|some|over|under)\s+|~\s*)*"
    rf"(?:(?:{_NUM}\s*{_UNIT}?\s*(?:[x×*]|\bby\b)\s*)+{_NUM}\s*{_UNIT}?|{_NUM}\s*{_UNIT})(?![a-z])"
    rf"(?:\s+(?:in\s+)?(?:tall|high|long|wide|deep|thick|length|height|width|depth|diameter)\b)?", re.I)
# words a clause may keep around a removed size and still say nothing else ("standing 20 cm tall", "height about")
SIZE_ONLY = {"with", "of", "at", "to", "a", "an", "the", "and", "in", "is", "its", "each", "standing", "stands",
             "reaching", "reaches", "extending", "extends", "measuring", "measures", "sized", "size", "dimensions",
             "dimension", "height", "length", "width", "depth", "diameter", "tall", "high", "long", "wide", "deep",
             "thick", "about", "approximately", "around", "roughly", "overall", "total", "format", "standard"}


def attach_v2(v2_root):
    """Make fastfill.v2 importable from fastfill-main while fastfill.build stays this (v4) package."""
    pkg = Path(v2_root).resolve() / "fastfill"
    if not (pkg / "v2" / "legacy_bridge.py").is_file():
        raise FileNotFoundError(f"no fastfill/v2 under {v2_root}")
    here = Path(__file__).resolve().parent
    differ = [n for n in SHARED if _sha256(here / n) != _sha256(pkg / n)]
    if differ:
        raise ValueError(f"fastfill-main and this package differ in {differ}; the bridge would mix two protocols")
    if str(pkg) not in fastfill.__path__:
        fastfill.__path__.append(str(pkg))
    return pkg


def with_v4_fronts(raw, prepared):
    """Raw IR copy whose boxes carry the v4 front fixes made in prep (labels stay unrounded)."""
    fixed = {o["id"]: o for o in prepared["objects"] if o.get("front_fix")}
    if not fixed:
        return raw
    objs = []
    for o in raw["objects"]:
        p = fixed.get(o["id"])
        if p is None:
            objs.append(o)
            continue
        if p["front_fix"] == "rot180":
            o = {**o, "yaw": (o["yaw"] + math.pi) % (2 * math.pi)}
        else:                                       # bed_rot90: the exact quarter turn prep made, sizes swapped
            turn = math.copysign(math.pi / 2, math.remainder(p["yaw"] - p["yaw_src"], 2 * math.pi))
            o = {**o, "yaw": (o["yaw"] + turn) % (2 * math.pi), "size": [o["size"][1], o["size"][0], o["size"][2]]}
        objs.append({**o, "v2_evidence": {**o.get("v2_evidence", {}), "v4_front_fix": p["front_fix"]}})
    return {**raw, "objects": objs}


def legacy_row(prepared, split):
    """The v1 message row the bridge reads constraints from: canonical prepared room + its extracted constraints."""
    c = canonical(prepared)
    cons = extract_constraints(c, rng_for(prepared["uid"], "s2" if split == "train" else "eval"))
    return {"messages": messages(c, cons), "flags": prepared["meta"]["flags"]}


def opening_type(category):
    """"door" / "window" for a room opening, else None (a shower door, a window seat or a radiator is not one)."""
    words = set(re.findall(r"[a-z]+", (category or "").lower()))
    if words & DOOR_WORDS and not words & NOT_ROOM_DOOR:
        return "door"
    if words & WINDOW_WORDS and not words & NOT_WINDOW:
        return "window"
    return None


def keep_fixed(prepared, kinds):
    """prepared with only the fixed boxes the condition shows: openings and the prep kinds in ``kinds``.
    -> (prepared copy, Counter of removed fixed_kind)."""
    kept, removed = [], collections.Counter()
    for f in prepared.get("fixed", []):
        if opening_type(f["category"]) or f.get("fixed_kind") in kinds:
            kept.append(f)
        else:
            removed[f.get("fixed_kind") or "unknown"] += 1
    return {**prepared, "fixed": kept}, removed


def strip_sizes(description):
    """Description without its metric size phrases, or None when nothing but the size (and filler) is left."""
    clauses = []
    for clause in re.split(r"[,;]", SIZE_PHRASE.sub(" ", description)):
        words = re.findall(r"[^\W\d_]+", clause.lower())
        if words and not set(words) <= SIZE_ONLY:
            clause = " ".join(clause.split())
            # a filler word left dangling at the clause end: "A soy candle with" (an 8 cm diameter)
            while (tail := re.search(r"\s+(\w+)[.!]?$", clause)) and tail.group(1).lower() in SIZE_ONLY:
                clause = clause[:tail.start()]
            clauses.append(clause.strip(" .:-"))
    text = ", ".join(c for c in clauses if c)
    return text if text and not METRIC.search(text) and re.search(r"[^\W\d_]", text) else None


def to_openings(room):
    """fixed_objects -> wall-relative room.openings for doors/windows; every other fixed box stays in
    room.fixed_objects. Returns (n fixed boxes kept, n detached openings removed)."""
    points = room["floor_polygon_xy_m"]
    edges = list(zip(points, points[1:] + points[:1]))
    floor = room.get("floor_z_m") or 0.
    openings, other, detached = [], [], 0
    for f in room.pop("fixed_objects", []):
        kind = opening_type(f.get("category", ""))
        if not kind:
            other.append(f)
            continue
        p = f["bottom_center_m"]
        best = None
        for i, (a, b) in enumerate(edges):
            length = math.dist(a, b)
            if length <= 1e-9:                      # repeated vertex: no wall here (direct_layout._shell skips it too)
                continue
            u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            t = min(max(((p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]) / length, 0.), 1.)
            d = math.dist(p[:2], (a[0] + u[0] * t * length, a[1] + u[1] * t * length))
            if best is None or d < best[0]:
                best = (d, i, t, u)
        d, i, t, u = best
        if d > OPENING_MAX_WALL_DIST_M:
            detached += 1
            continue
        c, s = math.cos(f["yaw_rad"]), math.sin(f["yaw_rad"])
        sx, sy, sz = f["size_local_m"]
        width = abs(sx * (u[0] * c + u[1] * s)) + abs(sy * (u[1] * c - u[0] * s))
        o = {"type": kind, "wall_index": i,
             "position_on_wall": round(t, 6), "width_m": round(width, 4), "height_m": round(sz, 4)}
        if o["type"] == "window":
            o["sill_height_m"] = round(p[2] - floor, 4)
        openings.append(o)
    openings.sort(key=lambda o: (o["wall_index"], o["position_on_wall"], o["type"]))
    if openings:
        room["openings"] = openings
    if other:
        room["fixed_objects"] = other
    return len(other), detached


def finish(sample, removed=None):
    """No-bbox condition + metric-description fix on a bridge row; revalidated by the v2 protocol.
    ``removed``: fixed_kind counts of the fixed boxes keep_fixed left out before the bridge."""
    from fastfill.v2.batch import _geometry_rows, room_normalization
    from fastfill.v2.schema import validate_condition
    cond = sample["condition"]
    kept, detached = to_openings(cond["room"])
    flags = sample["provenance"]["legacy_flags"]
    if kept:
        flags["fixed_obstacles_kept"] = kept
    if removed:
        flags["removed_fixed_obstacles"] = sum(removed.values())
        flags["removed_fixed_obstacles_by_kind"] = dict(sorted(removed.items()))
    if detached:
        flags["removed_detached_openings"] = detached
    for obj, ev in zip(cond["objects"], sample["provenance"]["field_evidence"]):
        if obj["description"] != obj["category"] and METRIC.search(obj["description"]):
            stripped = strip_sizes(obj["description"])
            obj["description"] = stripped or obj["category"]
            ev["description_metric"] = "spans_removed" if stripped else "category"
    validate_condition(cond)
    _geometry_rows(sample, *room_normalization(cond["room"]))
    return sample


def select(files, a, stats):
    """build.main's room selection: prep -> dedup -> split -> exact duplicates. -> {uid: (prepared room, split)}"""
    rooms = []
    for f in files:
        for r0 in _json_lines(f):
            st = stats[r0["source"]]
            r, why = prep(r0, a)
            st["scanned"] += 1
            if why:
                st["rejected:" + why] += 1
                continue
            rooms.append({**r, "meta": {**r["meta"], "n_src_objects": len(r0["objects"])}})
        print(f"prepared {f.name}: {len(rooms)} retained rooms so far", flush=True)
    rooms, dropped = dedup(rooms)
    for s, n in dropped.items():
        stats[s]["dedup_dropped"] += n
    splits = assign_splits(rooms, a.dev, a.test)
    first = {}
    for i in sorted(range(len(rooms)), key=lambda i: hashlib.sha1(rooms[i]["uid"].encode()).hexdigest()):
        key = (splits[i], exact_key(rooms[i]))
        if key in first and not a.keep_duplicates:
            stats[rooms[i]["source"]]["duplicate_layout"] += 1
            splits[i] = "duplicate"
        else:
            first[key] = i
    if a.cap:
        raise ValueError("--like build used --cap; not supported here")
    return {r["uid"]: (r, s) for r, s in zip(rooms, splits) if s in SPLIT_NAMES}


def untilt_fixed(raw, prepared):
    """Fixed boxes only become openings or are removed, so their tilt must not refuse the room (the bridge does)."""
    ids = {f["id"] for f in prepared.get("fixed", [])}
    n = sum(bool(o.get("tilted")) for o in raw["objects"] if o["id"] in ids)
    if not n:
        return raw, 0
    return {**raw, "objects": [{**o, "tilted": False} if o["id"] in ids else o for o in raw["objects"]]}, n


def convert(raw, prepared, split, front_policy, fixed_kinds=KEEP_FIXED_KINDS):
    from fastfill.v2.legacy_bridge import convert_selected_room
    row = legacy_row(prepared, split)                # constraints as the v4 build drew them
    prepared, removed = keep_fixed(prepared, fixed_kinds)
    raw, n_tilted_fixed = untilt_fixed(with_v4_fronts(raw, prepared), prepared)
    sample = convert_selected_room(raw, prepared, row, SPLIT_NAMES[split], front_policy=front_policy)
    if n_tilted_fixed:
        sample["provenance"]["tilted_fixed_converted"] = n_tilted_fixed
    sample["provenance"]["constraint_source"] = "v4_extract_constraints_every_room"
    sample["provenance"]["v4_flags"] = copy.deepcopy(prepared["meta"]["flags"])
    return finish(sample, removed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="MANIFEST.json of the v4 build whose rooms and splits to reuse")
    ap.add_argument("--v2-root", required=True, help="fastfill-main checkout (provides fastfill.v2)")
    ap.add_argument("--out", required=True, help="new dataset directory (must be absent or empty)")
    ap.add_argument("--ir", default=None, help="IR dir (default: the --like build's, relative to code_v4)")
    ap.add_argument("--sources", nargs="*", default=None, help="subset of sources (smoke runs; splits then differ)")
    ap.add_argument("--front-policy", default="axis", help="fastfill.v2.legacy_bridge.FRONT_POLICIES")
    ap.add_argument("--include-flagged", action="store_true", help="keep train rows with REJECT_FLAGS")
    ap.add_argument("--fixed-kinds", default=",".join(KEEP_FIXED_KINDS),
                    help=f"prep fixed_kind values kept as room.fixed_objects (of {','.join(FIXED_KINDS)}; '' = none)")
    cli = ap.parse_args()
    fixed_kinds = tuple(k for k in cli.fixed_kinds.split(",") if k)
    if set(fixed_kinds) - set(FIXED_KINDS):
        ap.error(f"--fixed-kinds must be among {FIXED_KINDS}")
    v2pkg = attach_v2(cli.v2_root)
    from fastfill.v2.legacy_bridge import FRONT_POLICIES, YAW_POLICY, SELECTION_POLICY
    if cli.front_policy not in FRONT_POLICIES:
        ap.error(f"--front-policy must be one of {FRONT_POLICIES}")
    like = json.load(open(cli.like, encoding="utf-8"))
    args = like["args"]
    if cli.ir is None:
        cli.ir = str((Path(cli.like).resolve().parent.parent.parent / "code_v4" / args["ir"]).resolve())
    a = SimpleNamespace(**{**args, "ir": cli.ir, "sources": cli.sources or args["sources"]})
    a.source_anchors = {k: (v.split(",") if isinstance(v, str) else v) for k, v in (a.source_anchors or {}).items()}
    files = _input_files(a, ap)
    if cli.sources is None and {p.name: _sha256(p) for p in files} != like["ir_sha256"]:
        ap.error("IR files differ from the --like build: rooms/splits would not match")
    _validate_destination(cli.out)
    stats = collections.defaultdict(collections.Counter)
    kept = select(files, a, stats)
    dest = Path(cli.out).absolute()
    dest.parent.mkdir(parents=True, exist_ok=True)
    totals, rejected = collections.Counter(), collections.Counter()
    with tempfile.TemporaryDirectory(prefix=f".{dest.name}-", dir=dest.parent) as staging:
        outs = {n: open(Path(staging) / f"{n}.jsonl", "w", encoding="utf-8") for n in SPLIT_NAMES.values()}
        rej = open(Path(staging) / "rejections.jsonl", "w", encoding="utf-8")
        for f in files:                              # second pass: the bridge needs the raw IR room
            for raw in _json_lines(f):
                hit = kept.pop(raw["uid"], None)
                if hit is None:
                    continue
                prepared, split = hit
                st = stats[raw["source"]]
                if split == "train" and not cli.include_flagged and any(prepared["meta"]["flags"].get(k)
                                                                        for k in REJECT_FLAGS):
                    st["train_flag_filtered"] += 1
                    continue
                try:
                    row = convert(raw, prepared, split, cli.front_policy, fixed_kinds)
                except ValueError as e:
                    reason = str(e).split(":")[0][:120]
                    st["v2_rejected:" + reason] += 1
                    rejected[reason] += 1
                    rej.write(json.dumps({"uid": raw["uid"], "source": raw["source"], "split": SPLIT_NAMES[split],
                                          "reason": str(e)}, ensure_ascii=False) + "\n")
                    continue
                sp, v, cond = row["provenance"]["split"], row["validity"], row["condition"]
                st[sp] += 1
                st[sp + "_objects"] += len(row["target"]["objects"])
                st["constraints"] += len(cond["constraints"])
                st["openings"] += len(cond["room"].get("openings", []))
                flags = row["provenance"]["legacy_flags"]
                st["fixed_obstacles_kept"] += flags.get("fixed_obstacles_kept", 0)
                st["removed_fixed_obstacles"] += flags.get("removed_fixed_obstacles", 0)
                for kind, n in flags.get("removed_fixed_obstacles_by_kind", {}).items():
                    st["removed_fixed:" + kind] += n
                st["rooms_with_constraints"] += bool(cond["constraints"])
                st["removed_detached_openings"] += row["provenance"]["legacy_flags"].get("removed_detached_openings", 0)
                for p, s, y, order, swap, req, ev in zip(v["position"], v["size"], v["yaw"], v["yaw_symmetry_order"],
                                                         v["size_axis_swap_allowed"], cond["objects"],
                                                         row["provenance"]["field_evidence"]):
                    st["full_geometry"] += all(p) and all(s) and bool(y)
                    st["size_masked"] += not all(s)
                    st["yaw_masked"] += not y
                    st[f"yaw_symmetry_order_{order}"] += 1
                    st["k1_objects"] += swap
                    st["support:" + str(req.get("support_parent") if req.get("support_parent") in
                                       (None, "floor", "wall") else "object")] += 1
                    st["selection:" + ev["selection_rule"]] += 1
                    st["v4_front_fix"] += "v4_front_fix" in ev["recorded_source_evidence"]
                    if ev.get("description_metric"):
                        st["description_metric:" + str(ev["description_metric"])] += 1
                st["exchangeable_members"] += sum(g is not None for g in v["exchangeable_group"])
                st["objects_max"] = max(st["objects_max"], len(row["target"]["objects"]))
                totals[sp] += 1
                totals[sp + "_objects"] += len(row["target"]["objects"])
                outs[sp].write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            print(f"converted {f.name}: {dict(totals)}", flush=True)
        if kept:
            raise RuntimeError(f"{len(kept)} selected rooms not found again in the IR")
        for fh in (*outs.values(), rej):
            fh.close()
        with open(Path(staging) / "stats.json", "w", encoding="utf-8") as fh:
            json.dump({k: dict(v) for k, v in sorted(stats.items())}, fh, indent=1)
        here = Path(__file__).resolve().parent
        manifest = {"schema_version": "fastfill.v2", "builder": "v4-prep-bridge (code_v4/fastfill/build_v2.py)",
                    "like": str(Path(cli.like).resolve()), "like_args": args, "ir": cli.ir, "sources": a.sources,
                    "bounded_build": cli.sources is not None,
                    "split_rule": "v4 build split (dedup + union-find components); dev renamed validation",
                    "front_policy": cli.front_policy, "yaw_policy": YAW_POLICY[cli.front_policy],
                    "selection_extension_policy": SELECTION_POLICY,
                    "train_flag_filter": [] if cli.include_flagged else list(REJECT_FLAGS),
                    "condition_geometry": "no target bbox: doors/windows as wall-relative room.openings, fixed "
                                          f"boxes of kinds {list(fixed_kinds)} as room.fixed_objects, other fixed "
                                          "boxes removed (legacy_flags.removed_fixed_obstacles)",
                    "fixed_kinds_kept": list(fixed_kinds),
                    "evidence_index": "not applied (v3.2 audit evidence not available)",
                    "split_samples": {k: totals[k] for k in SPLIT_NAMES.values()},
                    "split_objects": {k: totals[k + "_objects"] for k in SPLIT_NAMES.values()},
                    "v2_rejections": dict(rejected),
                    "ir_sha256": {p.name: _sha256(p) for p in files},
                    "code_sha256": {**{n: _sha256(here / n) for n in ("build_v2.py", "build.py", "v4fixes.py", *SHARED)},
                                    **{"v2/" + n: _sha256(v2pkg / "v2" / n) for n in
                                       ("legacy_bridge.py", "schema.py", "batch.py", "matching.py", "validation.py")}},
                    "files": {p.name: {"sha256": _sha256(p), "lines": sum(1 for _ in open(p, "rb"))}
                              for p in sorted(Path(staging).glob("*.jsonl"))}}
        with open(Path(staging) / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1)
        _validate_destination(dest)
        os.rename(staging, dest)
    print(json.dumps({k: manifest[k] for k in ("split_samples", "split_objects", "v2_rejections")}, indent=1))


if __name__ == "__main__":
    main()
