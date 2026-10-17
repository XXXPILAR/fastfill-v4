"""Streaming QA: python -m fastfill.tools.qa_report <IR dir> <data dir> -> <data dir>/QA.json.

--ir/--data are equivalent named arguments. No dataset version is assumed.
Checks written outputs against MANIFEST, constraint schema/references/truth on the saved
answers, closure, geometry, selection and leakage. Replays actual SFT text using the
manifest's augmentation, constraint and dropout settings. Only compact UID/digest/key
indexes and counters are retained, never whole datasets or whole files in memory.
Replay checks persisted rows, not the build's global split/dedup/cap selection decisions.
Writes the report before exiting 1 on mandatory failures; trainer drift and geometry
rates remain informational rather than requiring every reference to be pristine.
"""
import argparse
import collections
import copy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

from shapely.geometry import Polygon

from fastfill.build import _code_files, extract_constraints, prep, rng_for
from fastfill.evaluate import inside, score
from fastfill.interface import snap_inside
from fastfill.scene import canonical, messages, parse, rot90
from fastfill.split import content_key, layout_key
from fastfill.validate import check, holds


SPLITS = ("train", "dev", "test")
OUTPUTS = ("train.jsonl", "dev.jsonl", "test.jsonl", "dev_rooms.jsonl", "test_rooms.jsonl",
           "dev_constrained_rooms.jsonl", "test_constrained_rooms.jsonl", "stats.json")
BUILD_CODE = {"__init__.py", "build.py", "anchors.py", "scene.py", "split.py", "validate.py"}
DEFAULT_FLAGS = {"oob_objects", "overlapping_furniture", "fixed_collision"}
DEFAULTS = dict(reject_flagged=[], keep_duplicates=False, min_objects=1, max_vertices=0,
                sources=None, source_anchors={}, boundary_types=["polygon", "hull"],
                anchors=["floor", "object"], oob_tol=.1, hidden_max=.3,
                no_rot90=False, constraint_frac=0., field_dropout=.2, desc_dropout=.5)


def rows(path):
    """Missing outputs are disclosed by provenance; never materialize JSONL files."""
    if Path(path).is_file():
        with open(path, encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc


def _file_info(path):
    digest, lines, last = hashlib.sha256(), 0, b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    return {"sha256": digest.hexdigest(), "lines": lines + int(bool(last) and last != b"\n")}


def _scoped_code(name):
    return not any(p.startswith(("review", ".", "__pycache__")) or p == "tests"
                   for p in Path(name).parts[:-1])


def _provenance(man, ir, data):
    here = Path(__file__).resolve().parents[1]
    current = {str(p.relative_to(here)): _file_info(p)["sha256"] for p in _code_files(here)}
    recorded = {k: v for k, v in man.get("code_sha256", {}).items() if _scoped_code(k)}
    diff = sorted(k for k in current.keys() | recorded.keys() if current.get(k) != recorded.get(k))
    qa = {"build code differing from MANIFEST (must be empty)": [k for k in diff if k in BUILD_CODE],
          "other code changed since the build (does not affect the data)": [k for k in diff if k not in BUILD_CODE],
          "legacy MANIFEST code entries outside build provenance scope":
              sorted(k for k in man.get("code_sha256", {}) if not _scoped_code(k))}
    ir_hashes = man.get("ir_sha256")
    qa["IR files differing from MANIFEST (must be empty)"] = (
        sorted(k for k, v in ir_hashes.items()
               if not (ir / k).is_file() or _file_info(ir / k)["sha256"] != v)
        if ir_hashes else "no ir_sha256 in MANIFEST")
    declared = man.get("files") or {}
    differences = {}
    for name in sorted(set(declared) | set(OUTPUTS)):
        path = data / name
        if not path.is_file():
            differences[name] = {"missing": True}
            continue
        if name not in declared:
            continue  # Disclose missing declarations separately, not as verified hashes.
        actual = _file_info(path)
        mismatch = {k: {"expected": declared[name].get(k), "actual": v}
                    for k, v in actual.items() if declared[name].get(k) != v}
        if mismatch:
            differences[name] = mismatch
    qa["output files differing from MANIFEST (must be empty)"] = differences
    qa["output files missing from MANIFEST (must be empty)"] = sorted(set(OUTPUTS) - set(declared))
    return qa


def _layout(row):
    user = json.loads(row["messages"][1]["content"])
    objs = [{"id": o["id"], "category": o["id"].rsplit("_", 1)[0], "size": o["size"]} for o in user["objects"]]
    pl, err = parse(row["messages"][2]["content"], [o["id"] for o in objs])
    # SFT fixed boxes use integer degrees; every IR geometry consumer uses radians.
    fixed = [{**f, "yaw": math.radians(f["yaw"])} for f in user.get("fixed", [])]
    room = {"boundary": user["boundary"], "boundary_type": user["boundary_type"],
            "height": user.get("height"), "fixed": fixed, "objects": objs}
    lay = {**room, "objects": [{**o, "pos": pl[o["id"]]["pos"], "yaw": pl[o["id"]]["yaw"],
                               "parent": pl[o["id"]]["on"]} for o in objs if o["id"] in pl]}
    return user, room, pl, lay, err


def _valid(row):
    """Bounds/support/ceiling validity of the saved answer, using displayed inputs."""
    _, _, _, lay, err = _layout(row)
    return err is None and check(lay)["valid"]


def _constraint_error(c, ids):
    arity = {"on": 3, "faces": 3, "near": 4, "against_wall": 2, "between": 4}
    if not isinstance(c, list) or not c or not isinstance(c[0], str) or len(c) != arity.get(c[0]):
        return "schema"
    refs = c[1:3] if c[0] == "near" else c[1:]
    if not all(isinstance(v, str) and v for v in refs):
        return "schema"
    if c[0] == "near":
        d = c[3]
        try:
            good = type(d) in (int, float) and math.isfinite(d) and d >= 0
        except OverflowError:
            good = False
        if not good:
            return "schema"
    return None if all(v in ids for v in refs) else "reference"


def _constraints(constraints, objects, boundary, ids=None, answer_error=None):
    counts, failures = collections.Counter(), collections.Counter()
    if not isinstance(constraints, list):
        return counts, collections.Counter(schema=1), 0, 1
    index = {o["id"]: o for o in objects}
    ids = set(index) if ids is None else ids
    ok = 0
    for c in constraints:
        kind = c[0] if isinstance(c, list) and c and isinstance(c[0], str) else "malformed"
        counts[kind] += 1
        error = _constraint_error(c, ids)
        if not error and answer_error:
            error = "answer"
        if not error and not holds(c, objects, boundary, index):
            error = "false"
        if error:
            failures[error] += 1
        else:
            ok += 1
    return counts, failures, ok, len(constraints)


def _percentiles(hist):
    total = sum(hist.values())
    out = dict.fromkeys(("p50", "p90", "p99", "max"))
    for key, fraction in zip(out, (.5, .9, .99, 1.)):
        rank, n = min(int(fraction * total), total - 1), 0
        for value, count in sorted(hist.items()):
            n += count
            if n > rank:
                out[key] = value
                break
    return out


def _train_metrics(records):
    closure, fixed, objects, gone = (collections.Counter() for _ in range(4))
    types, failures = collections.Counter(), collections.Counter()
    n = constrained = fixed_answered = raw = repaired = used = valid = cons_ok = cons_total = 0
    for row in records:
        n += 1
        user, room, pl, lay, err = _layout(row)
        closure["ok" if err is None else err] += 1
        fixed_answered += any(f["id"] in pl for f in user.get("fixed", []))
        fixed[len(user.get("fixed", []))] += 1
        objects[len(user["objects"])] += 1
        constrained += "constraints" in user
        c, f, ok, total = _constraints(user.get("constraints", []), lay["objects"], lay["boundary"],
                                       {o["id"] for o in room["objects"]}, err)
        types.update(c); failures.update(f)
        cons_ok += ok; cons_total += total
        # Exactly train.select's default truthy-flag filter and alphabetical reason precedence.
        hit = sorted(k for k, v in (row.get("flags") or {}).items() if v and k in DEFAULT_FLAGS)
        if hit:
            gone["flag:" + hit[0]] += 1
        else:
            used += 1
            gone["weight"] += 0
            valid += err is None and check(lay)["valid"]
        if err is None:
            floor = Polygon(user["boundary"])
            raw += inside(lay, floor)
            adjusted = copy.deepcopy(pl)
            snap_inside(adjusted, room, floor)
            repaired += inside({**lay, "objects": [{**o, "pos": adjusted[o["id"]]["pos"]}
                                                    for o in lay["objects"]]}, floor)
    return {"train_rows_with_constraints": constrained, "object_set_closure": dict(closure),
            "fixed_boxes_answered (must be 0)": fixed_answered,
            "train reference inside walls within 1 mm": {"raw": f"{raw}/{n}", "repaired": f"{repaired}/{n}"},
            "train rows used by default": {"rows": used, "left_out": dict(gone), "reference_valid": f"{valid}/{used}"},
            "train fixed boxes": {"rows_with_fixed": n - fixed[0], "per_row": _percentiles(fixed)},
            "train objects per row": _percentiles(objects), "constraints_by_type": {"train": dict(sorted(types.items()))},
            "constraint_reference_validation": {"train": f"{cons_ok}/{cons_total}"},
            "constraint_failures": {"train": dict(failures)}}


def _eval_metrics(data, qa):
    qa["reference legality raw vs repaired"] = {}
    metrics = {"complete_rate": None, "valid_rate": "valid", "valid_rate_repaired": "valid_repaired",
               "inside_1mm_rate": "inside_1mm", "inside_1mm_rate_repaired": "inside_1mm_repaired",
               "rooms_with_fixed_collision": "coll_fixed", "rooms_with_severe": "coll_severe"}
    for split in ("dev", "test"):
        types, failures = collections.Counter(), collections.Counter()
        ok = total = 0
        for room in rows(data / f"{split}_constrained_rooms.jsonl"):
            c, f, k, n = _constraints(room.get("constraints", []), room["objects"], room["boundary"])
            types.update(c); failures.update(f); ok += k; total += n
        qa["constraints_by_type"][split] = dict(sorted(types.items()))
        qa["constraint_reference_validation"][split] = f"{ok}/{total}"
        qa["constraint_failures"][split] = dict(failures)
        groups = collections.defaultdict(collections.Counter)
        for room in rows(data / f"{split}_rooms.jsonl"):
            r = score(room, None)
            counts = groups["held_out" if r["held_out"] else "in_dist"]
            counts["rooms"] += 1
            for name, key in metrics.items():
                counts[name] += r["error"] is None if key is None else bool(r.get(key, False))
        for group, counts in groups.items():
            qa["reference legality raw vs repaired"][f"{split}/{group}"] = {
                "rooms": counts["rooms"], **{k: counts[k] / counts["rooms"] for k in metrics}}


def _digest(value):
    # Outer JSON formatting is irrelevant; message content strings are compared exactly.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).digest()


def _fingerprints(row):
    msgs = row["messages"]
    return tuple(_digest(v) for v in (
        *[msgs[i] if i < len(msgs) else None for i in range(3)], msgs[3:],
        {k: v for k, v in row.items() if k != "messages"}))


def _replay(room, split, args):
    cons, with_desc = None, True
    if split == "train":
        aug, s2, drop = (rng_for(room["uid"], k) for k in ("aug", "s2", "dropout"))
        c = canonical(room if args.no_rot90 else rot90(room, aug.randrange(4)))
        cons = extract_constraints(c, s2) if s2.random() < args.constraint_frac else None
        if drop.random() < args.field_dropout:
            c = {**c, "room_type": None}
        if drop.random() < args.field_dropout:
            c = {**c, "height": None}
        with_desc = drop.random() >= args.desc_dropout
    else:
        c = canonical(room)
    return {"uid": room["uid"], "source": room["source"], "flags": room["meta"]["flags"],
            "messages": messages(c, cons, with_desc=with_desc)}


def _record_mismatch(report, split, uid, reasons):
    report["failed"] += 1
    report["mismatches"].update(reasons)
    if len(report["examples"]) < 15:
        report["examples"].append({"split": split, "uid": uid, "reasons": reasons})


def _ir_checks(ir, args, saved):
    keys = {s: collections.defaultdict(set) for s in SPLITS}
    held, found = set(), set()
    src_rows, ir_seen = collections.Counter(), collections.Counter()
    report = {"matched": 0, "failed": 0, "mismatches": collections.Counter(), "examples": [],
              "scope": "Exact messages and row metadata, using current code and MANIFEST settings; "
                       "code/IR/output provenance discrepancies are reported separately."}
    for path in sorted(ir.glob("*.jsonl")):
        if args.sources is not None and path.stem not in args.sources:
            continue
        for raw in rows(path):
            uid = raw["uid"]
            if uid not in saved:
                continue
            ir_seen[uid] += 1
            if uid in found:
                continue  # First match is checked; ambiguous source UIDs are reported separately.
            found.add(uid)
            room, why = prep(raw, args)
            expected = {}
            for split, digests in saved[uid]:
                if room is None:
                    _record_mismatch(report, split, uid, ["prep"])
                    continue
                if split not in expected:
                    expected[split] = _fingerprints(_replay(room, split, args))
                names = ("system", "user", "assistant", "extra_messages", "metadata")
                reasons = [k for k, actual, target in zip(names, digests, expected[split]) if actual != target]
                if reasons:
                    _record_mismatch(report, split, uid, reasons)
                else:
                    report["matched"] += 1
                src_rows[(split, path.stem)] += 1
            if room is None:
                continue
            if room.get("meta", {}).get("eval_only"):
                held.add(path.stem)
            fingerprints = (("furniture keys", content_key(room)), ("layout keys", layout_key(room)))
            for split in {s for s, _ in saved[uid]}:
                keys[split]["worlds"].add(room["group"])
                keys[split]["aliases"].update(room.get("meta", {}).get("group_aliases") or [])
                for name, key in fingerprints:
                    if key:
                        keys[split][name].add(key)
    for uid in saved.keys() - found:
        for split, _ in saved[uid]:
            _record_mismatch(report, split, uid, ["missing_ir"])
    leak = {f"{a}&{b} {name}": len(keys[a][name] & keys[b][name])
            for a, b in combinations(SPLITS, 2) for name in ("worlds", "aliases", "furniture keys", "layout keys")}
    leak.update({"held-out sources in train (rows)": sum(n for (s, src), n in src_rows.items() if s == "train" and src in held),
                 "rows not reproducible from IR": report["failed"],
                 "rows whose uid is missing from the IR": sum(len(saved[u]) for u in saved.keys() - found),
                 "uids written more than once (within or across splits)": sum(len(v) > 1 for v in saved.values()),
                 "persisted uids found more than once in IR": sum(n > 1 for n in ir_seen.values())})
    return {"worlds": {s: len(keys[s]["worlds"]) for s in SPLITS},
            "leakage (all must be 0)": leak, "SFT reproducibility": report}


def _arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ir", nargs="?")
    ap.add_argument("data", nargs="?")
    ap.add_argument("--ir", dest="ir_option")
    ap.add_argument("--data", dest="data_option")
    a = ap.parse_args()
    for key in ("ir", "data"):
        positional, option = getattr(a, key), getattr(a, key + "_option")
        if positional and option and positional != option:
            ap.error(f"conflicting {key} paths")
        value = option or positional
        if not value:
            ap.error(f"{key} path is required")
        setattr(a, key, value)
    return a


def _rejections(data):
    if not (data / "stats.json").is_file():
        return None
    with open(data / "stats.json", encoding="utf-8") as f:
        stats = json.load(f)
    return {
        src: {k.replace("rejected:", ""): v for k, v in d.items()
              if k.startswith(("rejected:", "flag:", "fixed:")) or k in
              ("scanned", "dev", "test", "fixed_items", "rooms_with_fixed", "objects_max")}
        | {"kept_train": d.get("train_S1", 0) + d.get("train_S2", 0)} for src, d in stats.items()}


def _mandatory_failures(qa):
    """Gate explicit invariants, not descriptive geometry rates or non-build code drift."""
    failures = [key for key in (
        "build code differing from MANIFEST (must be empty)",
        "IR files differing from MANIFEST (must be empty)",
        "output files differing from MANIFEST (must be empty)",
        "output files missing from MANIFEST (must be empty)",
        "fixed_boxes_answered (must be 0)",
    ) if qa[key]]
    if any(n for error, n in qa["object_set_closure"].items() if error != "ok"):
        failures.append("object_set_closure")
    if any(any(counts.values()) for counts in qa["constraint_failures"].values()):
        failures.append("constraint_failures")
    if any(qa["leakage (all must be 0)"].values()):
        failures.append("leakage (all must be 0)")
    if qa["SFT reproducibility"]["failed"]:
        failures.append("SFT reproducibility")
    return failures


def main():
    a = _arguments()
    ir, data = Path(a.ir), Path(a.data)
    with open(data / "MANIFEST.json", encoding="utf-8") as f:
        man = json.load(f)
    settings = {**DEFAULTS, **man["args"]}
    anchors = settings["source_anchors"]
    settings["source_anchors"] = dict(anchors) if isinstance(anchors, dict) else {
        k: v.split(",") for k, v in (x.split("=", 1) for x in anchors)}
    args = argparse.Namespace(**settings)
    qa = {"build": a.data, **_provenance(man, ir, data),
          "MANIFEST defaults used": {k: v for k, v in DEFAULTS.items() if k not in man["args"]}}
    saved, counts = collections.defaultdict(list), collections.Counter()

    def indexed(split):
        for row in rows(data / f"{split}.jsonl"):
            counts[split] += 1
            saved[row["uid"]].append((split, _fingerprints(row)))
            yield row

    qa.update(_train_metrics(indexed("train")))
    for split in ("dev", "test"):
        for _ in indexed(split):
            pass
    qa["rows"] = {s: counts[s] for s in SPLITS}
    _eval_metrics(data, qa)
    qa.update(_ir_checks(ir, args, saved))
    qa["rejections_by_source"] = _rejections(data)
    qa["mandatory_failures"] = _mandatory_failures(qa)
    with open(data / "QA.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, indent=1)
    print(json.dumps({k: v for k, v in qa.items() if k != "rejections_by_source"}, indent=1))
    return int(bool(qa["mandatory_failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
