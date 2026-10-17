"""Immutable v2 migration of OUR selected v3.2 corpus, preserving saved splits."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import json
from itertools import islice
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

from fastfill.build import prep
from fastfill.v2.data import _digest, _output_path
from fastfill.v2.legacy_bridge import FLOOR_SNAP_M, FRONT_POLICIES, SELECTION_POLICY, YAW_POLICY, convert_selected_room

DEFAULT_RELEASE = Path(__file__).resolve().parents[2] / ".release/v3.2"
DEFAULT_EVIDENCE = Path(__file__).resolve().parents[2] / ".release/audits/2026-09-28/source-check"
DEFAULT_REJECT_FLAGS = ("oob_objects", "fixed_collision", "overlapping_furniture")
_WORKER_CONTEXT = None


def _convert_job(job, args, evidence, seed, front_policy):
    raw, split, row = job
    prepared, reason = prep(deepcopy(raw), args)
    if reason:
        raise ValueError(f"selected frozen UID rejected by legacy preparation: {raw['uid']}: {reason}")
    try:
        sample = convert_selected_room(evidence.apply_evidence(raw), prepared, row, split,
                                       seed=seed, front_policy=front_policy)
        if not any(all(p) or all(s) or y for p, s, y in zip(
                sample["validity"]["position"], sample["validity"]["size"], sample["validity"]["yaw"])):
            raise ValueError("no reliable geometry fields")
        return sample, None
    except ValueError as exc:
        return None, {"uid": raw["uid"], "source": raw["source"], "split": split, "reason": str(exc)}


def _initialize_worker(args, evidence_root, seed, front_policy):
    from fastfill.v2.legacy_evidence import EvidenceIndex
    global _WORKER_CONTEXT
    # Each subprocess owns immutable settings/evidence; source files are read-only.
    _WORKER_CONTEXT = (SimpleNamespace(**args), EvidenceIndex.from_root(evidence_root, require=True), seed, front_policy)


def _worker_job(job):
    return _convert_job(job, *_WORKER_CONTEXT)


def _selected_jobs(connection, path):
    with path.open() as stream:
        for line in stream:
            raw = json.loads(line)
            selected = connection.execute("SELECT split,row FROM selected WHERE uid=?", (raw["uid"],)).fetchone()
            if selected is not None:
                connection.execute("UPDATE selected SET seen=seen+1 WHERE uid=?", (raw["uid"],))
                yield raw, selected[0], json.loads(selected[1])


def _record_result(result, outs, diagnostics, counts, source_counts, validity_counts, rejected, detail_counts):
    sample, failure = result
    if failure:
        rejected[failure["reason"]] += 1
        diagnostics.write(json.dumps(failure, allow_nan=False) + "\n")
        return
    split, source = sample["provenance"]["split"], sample["provenance"]["source"]
    outs[split].write(json.dumps(sample, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    counts[split] += 1
    source_counts[f"{source}:{split}"] += 1
    validity, floor_known = sample["validity"], sample["condition"]["room"].get("floor_known") is True
    validity_counts["targets"] += len(sample["target"]["objects"])
    for p, s, y, order, group, swap, request, evidence in zip(
            validity["position"], validity["size"], validity["yaw"], validity["yaw_symmetry_order"],
            validity["exchangeable_group"], validity["size_axis_swap_allowed"], sample["condition"]["objects"],
            sample["provenance"]["field_evidence"]):
        validity_counts["position"] += all(p)
        validity_counts["size"] += all(s)
        validity_counts["yaw"] += bool(y)
        validity_counts["full_geometry"] += all(p) and all(s) and bool(y)
        detail_counts[f"{source}:yaw_valid"] += bool(y)
        detail_counts[f"{source}:symmetry_order_{order}"] += 1
        detail_counts[f"{source}:swap_allowed"] += bool(swap)
        detail_counts["exchangeable_members"] += group is not None
        declared = request.get("support_parent") == "floor"
        detail_counts["floor_declarations_written"] += declared
        detail_counts["floor_declaration_z_snapped"] += bool(evidence["legacy_z_snap_applied_to_target"])
        detail_counts["floor_declaration_skipped_floating"] += (evidence["raw_anchor"] == "floor" and floor_known
            and not evidence["source_support_inferred"] and request.get("support_parent") is None)
        detail_counts["wall_declarations_written"] += request.get("support_parent") == "wall"
        detail_counts[f"selection_rule:{evidence['selection_rule']}"] += 1
    detail_counts["exchangeable_groups"] += len({g for g in validity["exchangeable_group"] if g is not None})


def verify_release(release):
    manifest_path = release / "data/v3.2/MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {**{str(release / "data/v3.2" / n): v["sha256"] for n, v in manifest["files"].items()},
              **{str(release / "ir" / n): h for n, h in manifest["ir_sha256"].items()}}
    for path, expected in inputs.items():
        if _digest(path) != expected:
            raise ValueError(f"frozen input hash mismatch: {path}")
    package = Path(__file__).resolve().parents[1]
    for name in ("build.py", "scene.py", "anchors.py", "validate.py", "split.py"):
        if _digest(package / name) != manifest["code_sha256"][name]:
            raise ValueError(f"legacy preparation code differs from frozen manifest: {name}")
    return manifest, inputs


def _index_saved(connection, release, include_flagged):
    connection.execute("CREATE TABLE selected(uid TEXT PRIMARY KEY, split TEXT, row TEXT, seen INTEGER DEFAULT 0)")
    filtered = Counter()
    for old, new in (("train", "train"), ("dev", "validation"), ("test", "test")):
        with (release / "data/v3.2" / f"{old}.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if new == "train" and not include_flagged and any(row.get("flags", {}).get(k) for k in DEFAULT_REJECT_FLAGS):
                    filtered[row["source"]] += 1
                    continue
                connection.execute("INSERT INTO selected(uid,split,row) VALUES(?,?,?)", (row["uid"], new, line))
    connection.commit()
    return filtered


def _included(connection, uids, sources):
    """Smoke-only extra UIDs per frozen source; each must be an unfiltered frozen selected UID of a migrated source."""
    by_source = {}
    for uid in uids:
        hit = connection.execute("SELECT row FROM selected WHERE uid=?", (uid,)).fetchone()
        source = None if hit is None else json.loads(hit[0])["source"]
        if source is None or sources is not None and source not in sources:
            raise ValueError(f"included UID is not an unfiltered frozen selected UID of a migrated source: {uid}")
        by_source.setdefault(source, set()).add(uid)
    return by_source


def _write_migration(connection, release, staging, manifest, evidence, *, seed, max_scenes, front_policy, workers=1,
                     sources=None, max_scenes_per_source=None, include_uids=()):
    args = SimpleNamespace(**manifest["args"])
    counts, source_counts, validity_counts, rejected, detail_counts = (Counter() for _ in range(5))
    included = _included(connection, include_uids, sources)
    with ExitStack() as stack:
        outs = {s: stack.enter_context((staging / f"{s}.jsonl").open("x")) for s in ("train", "validation", "test")}
        diagnostics = stack.enter_context((staging / "rejections.jsonl").open("x"))
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers, initializer=_initialize_worker,
            initargs=(manifest["args"], evidence.evidence_root, seed, front_policy))) if workers > 1 else None
        for name in manifest["ir_sha256"]:
            if sources is not None and name[:-len(".jsonl")] not in sources:
                continue
            jobs, start, full = _selected_jobs(connection, release / "ir" / name), sum(counts.values()), False
            wanted = set(included.get(name[:-len(".jsonl")], ()))  # smoke UIDs taken past the per-source cap
            # Bound submissions; Python 3.12 Executor.map otherwise consumes the entire corpus eagerly.
            while (not full or wanted) and (batch := tuple(islice(jobs, 128))):
                if full:
                    batch = tuple(job for job in batch if job[0]["uid"] in wanted)
                results = pool.map(_worker_job, batch) if pool else (
                    _convert_job(job, args, evidence, seed, front_policy) for job in batch)
                for job, result in zip(batch, results):
                    if full and job[0]["uid"] not in wanted:
                        continue
                    wanted.discard(job[0]["uid"])
                    _record_result(result, outs, diagnostics, counts, source_counts, validity_counts, rejected, detail_counts)
                    if max_scenes is not None and sum(counts.values()) >= max_scenes:
                        return counts, source_counts, validity_counts, rejected, detail_counts
                    full = max_scenes_per_source is not None and sum(counts.values()) - start >= max_scenes_per_source
                    if full and not wanted:
                        break
            connection.commit()
            print(json.dumps({"source_finished": name, "split_samples": dict(counts)}, ensure_ascii=False), flush=True)
    return counts, source_counts, validity_counts, rejected, detail_counts


def build_selected_dataset(release_root, output, *, evidence_root=DEFAULT_EVIDENCE, seed=42, max_scenes=None,
                           front_policy="axis", include_flagged=False, workers=1, sources=None, max_scenes_per_source=None,
                           include_uids=()):
    """``sources`` / ``max_scenes_per_source`` / ``include_uids`` (taken past the per-source cap) bound a smoke build
    (``bounded_build``); a release build passes none of them."""
    release, target = _output_path(release_root, output)
    if front_policy not in FRONT_POLICIES:
        raise ValueError("unknown front policy")
    for label, value in (("max_scenes", max_scenes), ("max_scenes_per_source", max_scenes_per_source)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{label} must be a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    manifest, inputs = verify_release(release)
    if sources is not None:
        sources = tuple(sources)
        if unknown := set(sources) - {name[:-len(".jsonl")] for name in manifest["ir_sha256"]}:
            raise ValueError(f"unknown source filter: {sorted(unknown)}")
    include_uids = tuple(include_uids or ())
    bounded = max_scenes is not None or max_scenes_per_source is not None or sources is not None or bool(include_uids)
    from fastfill.v2.legacy_evidence import EvidenceIndex
    evidence = EvidenceIndex.from_root(evidence_root, require=True)
    evidence_hashes = {str(Path(evidence_root) / name): _digest(Path(evidence_root) / name)
                       for name in evidence.loaded_files}
    code_hashes = {str(Path(__file__).with_name(name)): _digest(Path(__file__).with_name(name))
                   for name in ("legacy_build.py", "legacy_bridge.py", "legacy_evidence.py")}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary)
        with sqlite3.connect(staging / "selection.sqlite") as connection:
            filtered = _index_saved(connection, release, include_flagged)
            counts, source_split, valid, rejected, detail_counts = _write_migration(connection, release, staging, manifest, evidence,
                seed=seed, max_scenes=max_scenes, front_policy=front_policy, workers=workers,
                sources=sources, max_scenes_per_source=max_scenes_per_source, include_uids=include_uids)
            if not bounded and connection.execute("SELECT COUNT(*) FROM selected WHERE seen!=1").fetchone()[0]:
                raise ValueError("frozen selected UID missing or repeated in source IR")
        (staging / "selection.sqlite").unlink()
        if not sum(counts.values()):
            raise ValueError("no selected samples eligible for migration")
        result = {"schema_version": "fastfill.v2", "builder": "selected-v3.2-bridge", "release_root": str(release),
                  "split_rule": "inherit frozen selected UID assignment; dev renamed validation; no re-split", "seed": seed,
                  "front_policy": front_policy, "yaw_policy": YAW_POLICY[front_policy],
                  "workers": workers, "bounded_submission_jobs": 128, "bounded_build": bounded,
                  "source_filter": None if sources is None else list(sources), "max_scenes_per_source": max_scenes_per_source,
                  "included_uids": list(include_uids),
                  "samples_written": sum(counts.values()),
                  "split_samples": dict(counts), "source_split_samples": dict(source_split), "valid_label_counts": dict(valid),
                  "source_yaw_valid_objects": {k.split(":")[0]: v for k, v in detail_counts.items() if k.endswith(":yaw_valid")},
                  "source_yaw_symmetry_order_counts": {k.replace(":symmetry_order_", ":"): v for k, v in detail_counts.items() if ":symmetry_order_" in k},
                  "source_size_axis_swap_allowed_objects": {k.split(":")[0]: v for k, v in detail_counts.items() if k.endswith(":swap_allowed") and v},
                  "exchangeable_group_counts": {k: detail_counts[k] for k in ("exchangeable_groups", "exchangeable_members")},
                  **{k: detail_counts[k] for k in ("floor_declarations_written", "floor_declaration_z_snapped", "floor_declaration_skipped_floating")},
                  "floor_declaration_policy": f"source anchor floor only; target z within {FLOOR_SNAP_M} m of the known floor is snapped, farther keeps a free z without declaration",
                  "wall_declarations_written": detail_counts["wall_declarations_written"],
                  "selection_rule_counts": {k.split(":", 1)[1]: v for k, v in detail_counts.items() if k.startswith("selection_rule:")},
                  "selection_extension_policy": SELECTION_POLICY,
                  "descriptions": "source_desc_or_category",
                  "legacy_train_flag_filter": [] if include_flagged else list(DEFAULT_REJECT_FLAGS),
                  "legacy_train_filtered_by_source": dict(filtered), "v2_rejections": dict(rejected),
                  "selected_sources": manifest["args"]["sources"], "source_hashes": inputs,
                  "implementation_sha256": code_hashes, "evidence_sha256": evidence_hashes,
                  "source_data_modified": False, "geometry_scope": "field-masked canonical IR, no mesh/physics acceptance claim"}
        (staging / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if any(_digest(path) != expected for path, expected in {**inputs, **evidence_hashes, **code_hashes}.items()):
            raise ValueError("frozen input, evidence or migration implementation changed during migration")
        if target.exists():
            raise FileExistsError("immutable output appeared during migration")
        os.rename(staging, target)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--source", action="append", dest="sources", metavar="NAME",
                        help="only migrate this IR source (repeatable); marks the build bounded")
    parser.add_argument("--max-scenes-per-source", type=int, help="stop each source after this many written scenes (bounded smoke)")
    parser.add_argument("--include-uid", action="append", dest="include_uids", metavar="UID",
                        help="also migrate this frozen UID past the per-source cap (repeatable); marks the build bounded")
    parser.add_argument("--front-policy", choices=FRONT_POLICIES, default="axis")
    parser.add_argument("--include-flagged", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    result = build_selected_dataset(**vars(parser.parse_args(argv)))
    print(json.dumps({k: result[k] for k in ("samples_written", "split_samples", "valid_label_counts", "v2_rejections")}, indent=2))


if __name__ == "__main__":
    main()
