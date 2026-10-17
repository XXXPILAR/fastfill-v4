"""Independent streaming verifier for OUR selected-v3.2 FastFill-v2 bridge.

Does not rerun preparation, adapters, splitting, tokenization or tensor batching.
Only UID/underlying-house indices grow with corpus size; samples are streamed.
Geometry eligibility is label eligibility, not mesh, physics or context-budget
acceptance. A new immutable report can be written with ``--output``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .batch import _geometry_rows, room_normalization
from .legacy_bridge import FLOOR_SNAP_M, HEIGHT_TOLERANCE_M, SELECTION_POLICY, YAW_POLICY, size_axis_swap_allowed
from .schema import validate_condition
from .validation import on_wall

# Round 10 selection rule, restated here rather than imported from legacy_bridge: a builder that widens it fails rows.
SELECTION_RULES = ("frozen_prep", "wall_anchor", "support_inside_parent", "support_on_added_parent")
WALL_ANCHOR_SOURCES = frozenset({"SAGE-10k", "MansionWorld", "SceneSmith"})
SUPPORT_TOL_M = .05
SPLITS = ("train", "validation", "test")
DEFAULT_REJECT_FLAGS = ("oob_objects", "fixed_collision", "overlapping_furniture")
TEST_ONLY_SOURCES = frozenset({"SceneSmith", "SpatialGen"})
SOURCE_ROOT = Path("/Volumes/harddisk/3D_Room_Collections")
COUNT_FIELDS = ("scenes", "targets", "position", "size", "yaw", "full_geometry",
                "full_geometry_scenes", "text_eligible_scenes", "learnable_scenes", "zero_supervision_scenes")


class _Audit:
    def __init__(self, max_errors: int = 100):
        self.errors: list[dict[str, str]] = []
        self.error_count = 0
        self.max_errors = max_errors

    def fail(self, context: str, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < self.max_errors:
            self.errors.append({"context": context, "message": message})


def _hash_file(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest, length, lines, last = hashlib.sha256(), 0, 0, b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            length += len(block)
            lines += block.count(b"\n")
            last = block[-1:]
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("file changed during hash read")
    return {"sha256": digest.hexdigest(), "bytes": length, "lines": lines + int(bool(last) and last != b"\n")}


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _json_file(path: Path) -> dict:
    value = json.loads(path.read_bytes(), parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _json_lines(path: Path, audit: _Audit, hashes: dict[str, str]):
    digest = hashlib.sha256()
    try:
        before = path.stat()
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                context = f"{path}:{number}"
                try:
                    row = json.loads(line, parse_constant=_invalid_constant)
                    if not isinstance(row, dict):
                        raise ValueError("expected JSON object")
                except (ValueError, UnicodeDecodeError) as exc:
                    audit.fail(context, f"invalid JSONL record: {exc}")
                    continue
                yield context, row
        after = path.stat()
        hashes[path.name] = digest.hexdigest()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            audit.fail(str(path), "dataset changed during verification")
    except OSError as exc:
        audit.fail(str(path), f"cannot read required JSONL: {exc}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing nonempty {label}")
    return value


def _flag_digest(flags: Any) -> str:
    if not isinstance(flags, dict):
        raise ValueError("legacy flags must be an object")
    return hashlib.sha256(json.dumps(flags, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _check_hashes(label: str, expected: dict, audit: _Audit, *, line_counts=None) -> dict:
    result = {"checked": 0, "matched": 0, "bytes": 0, "files": {}}
    for name, checksum in expected.items():
        path = Path(name)
        try:
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError("invalid expected SHA256")
            actual = _hash_file(path)
            result["checked"] += 1
            result["bytes"] += actual["bytes"]
            matched = actual["sha256"] == checksum
            result["matched"] += matched
            result["files"][name] = {**actual, "expected_sha256": checksum, "matched": matched}
            if not matched:
                audit.fail(name, f"{label} hash mismatch")
            if line_counts and name in line_counts and actual["lines"] != line_counts[name]:
                audit.fail(name, "frozen manifest physical line count mismatch")
        except (OSError, ValueError) as exc:
            audit.fail(name, f"{label} hash check failed: {exc}")
    return result


def _saved_selection(release: Path, filter_flags: tuple, audit: _Audit):
    expected, counts, filtered = {}, Counter(), Counter()
    for old, split in (("train", "train"), ("dev", "validation"), ("test", "test")):
        for context, row in _json_lines(release / "data/v3.2" / f"{old}.jsonl", audit, {}):
            try:
                uid, source = _text(row.get("uid"), "selected UID"), _text(row.get("source"), "selected source")
                flags = row.get("flags", {})
                digest = _flag_digest(flags)
                excluded = split == "train" and any(flags.get(key) for key in filter_flags)
                if uid in expected:
                    raise ValueError("duplicate UID in frozen selected splits")
                placements = len(json.loads(row["messages"][-1]["content"])["placements"])  # the frozen target count
                expected[uid] = (split, source, excluded, digest, placements)
                counts[(source, split)] += 1
                if excluded:
                    filtered[source] += 1
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                audit.fail(context, str(exc))
    return expected, counts, filtered


def _lineage(row: dict, split: str, expected: dict, seen: set, audit: _Audit, context: str):
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("sample provenance must be an object")
    uid = _text(provenance.get("legacy_uid"), "legacy_uid")
    source = _text(provenance.get("source"), "provenance source")
    if uid in seen:
        audit.fail(context, f"duplicate UID in kept/rejected outputs: {uid}")
    seen.add(uid)
    if provenance.get("split") != split:
        audit.fail(context, "provenance split differs from output split")
    if provenance.get("scene_id") != uid:
        audit.fail(context, "scene_id must retain original selected UID")
    if source in TEST_ONLY_SOURCES and split != "test":
        audit.fail(context, f"{source} is test-only")
    assignment = expected.get(uid)
    if assignment is None:
        audit.fail(context, f"UID was not selected in frozen release: {uid}")
    elif assignment[:2] != (split, source):
        audit.fail(context, "UID source/split differs from frozen selected assignment")
    elif assignment[2]:
        audit.fail(context, "default-flag-filtered training UID was retained")
    if assignment and _flag_digest(provenance.get("legacy_flags", {})) != assignment[3]:
        audit.fail(context, "legacy flags differ from frozen selected record")
    return uid, source, provenance


def _check_house(provenance: dict, source: str, split: str, houses: dict, audit: _Audit, context: str):
    house = _text(provenance.get("house_id"), "underlying house_id")
    meta = provenance.get("source_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("source_meta must be an object")
    aliases = meta.get("group_aliases") or []
    if not isinstance(aliases, list) or any(not isinstance(v, str) or not v for v in aliases):
        raise ValueError("group_aliases must be a list of nonempty strings")
    keys = [("house", source, house)] + [("global_alias", alias) for alias in aliases]
    for key in keys:
        previous = houses.get(key)
        if previous is not None and previous != split:
            audit.fail(context, f"split leakage across underlying identity: {key} ({previous}, {split})")
        else:
            houses[key] = split
    if meta.get("eval_only") and split != "test":
        audit.fail(context, "source eval_only scene is test-only")


def _geometry(row: dict) -> dict:
    if row.get("schema_version") != "fastfill.v2":
        raise ValueError("sample schema_version must be fastfill.v2")
    condition = row.get("condition")
    validate_condition(condition)
    requests = condition["objects"]
    if not requests:
        raise ValueError("selected scene must contain requested objects")
    target = row.get("target")
    if not isinstance(target, dict) or set(target) != {"schema_version", "objects"} or target["schema_version"] != "fastfill.v2":
        raise ValueError("target schema_version/fields must follow fastfill.v2")
    targets = target["objects"]
    fields = {"id", "target_size_local_m", "bottom_center_m", "yaw_rad"}
    if not isinstance(targets, list) or any(not isinstance(o, dict) or set(o) != fields for o in targets):
        raise ValueError("invalid target object fields")
    ids = [o["id"] for o in targets]
    if (any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids)
            or set(ids) != {o["id"] for o in requests}):
        raise ValueError("target IDs must equal all requested IDs exactly once")
    validity = row.get("validity")
    if not isinstance(validity, dict):
        raise ValueError("validity must be an object")
    for field in ("position", "size", "yaw", "yaw_symmetry_order", "exchangeable_group", "size_axis_swap_allowed"):
        if not isinstance(validity.get(field), list) or len(validity[field]) != len(targets):
            raise ValueError(f"validity {field} must have exactly one entry per target")
    if any(isinstance(v, bool) or v not in (1, 2, 4) for v in validity["yaw_symmetry_order"]):
        raise ValueError("yaw_symmetry_order entries must be 1, 2 or 4")
    if any(not isinstance(v, bool) for v in validity["size_axis_swap_allowed"]):
        raise ValueError("size_axis_swap_allowed entries must be booleans")
    if any(swap and order != 4 for swap, order in zip(validity["size_axis_swap_allowed"], validity["yaw_symmetry_order"])):
        raise ValueError("size_axis_swap_allowed objects must carry yaw_symmetry_order 4")
    geometry = _geometry_rows(row, *room_normalization(condition["room"]))
    for value, valid in zip(geometry["yaw"], geometry["yaw_valid"]):
        if valid and not -math.pi <= value < math.pi:
            raise ValueError("valid yaw target must be wrapped to [-pi, pi)")
    from .matching import certify_group
    slot = {o["id"]: i for i, o in enumerate(requests)}
    groups = {}
    for target_obj, label in zip(targets, validity["exchangeable_group"]):
        if label is not None:
            groups.setdefault(label, []).append(slot[target_obj["id"]])
    for indices in groups.values():
        certify_group(requests, condition["constraints"], indices)
        if any(not all(geometry["position_valid"][i]) for i in indices):
            raise ValueError("exchangeable matching requires complete position labels")
    geometry["exchangeable_groups"], geometry["exchangeable_members"] = len(groups), sum(map(len, groups.values()))
    return geometry


def _check_unreferenced_groups(row: dict) -> None:
    """Both halves of the ``matching.group_labels`` rule, on swap certification: a set of position-complete
    requests with identical non-id fields shares one group label when its whole exchange certifies; otherwise
    its members that no constraint reference or support_parent names (always exchangeable) do."""
    from .matching import certify_group
    requests, validity, targets = row["condition"]["objects"], row["validity"], row["target"]["objects"]
    constraints = row["condition"]["constraints"]
    referenced = {o.get("support_parent") for o in requests}
    for c in constraints:
        referenced.update(c.get(k) for k in ("object_id", "target_id", "parent_id"))
        referenced.update(ref for k in ("target_ids", "object_ids") for ref in c.get(k) or ())
    label = {t["id"]: g for t, g in zip(targets, validity["exchangeable_group"])}
    position = {t["id"]: all(m) for t, m in zip(targets, validity["position"])}
    sets = {}
    for i, o in enumerate(requests):
        if position[o["id"]]:
            sets.setdefault(json.dumps({k: v for k, v in o.items() if k != "id"}, sort_keys=True), []).append(i)
    for indices in sets.values():
        try:
            certify_group(requests, constraints, indices)
        except ValueError:
            indices = [i for i in indices if requests[i]["id"] not in referenced]
        labels = [label[requests[i]["id"]] for i in indices]
        if len(labels) > 1 and (None in labels or len(set(labels)) > 1):
            raise ValueError("identical position-complete requests (the whole set when its exchange certifies, "
                             "else its unreferenced members) must share one exchangeable group")


def _height_conflict(row: dict) -> dict | None:
    """K2 recomputed: complete position+size targets whose top exceeds the kept source room height."""
    height, validity = row["condition"]["room"].get("height_m"), row["validity"]
    tops = {t["id"]: t["bottom_center_m"][2] + t["target_size_local_m"][2]
            for t, p, s in zip(row["target"]["objects"], validity["position"], validity["size"]) if all(p) and all(s)}
    over = [] if height is None else [i for i, top in tops.items() if top > height + HEIGHT_TOLERANCE_M + 1e-6]
    return {"objects": over, "max_excess_m": max(tops[i] for i in over) - height} if over else None


def _row_counts(geometry: dict) -> dict[str, int]:
    position, size, yaw = [all(mask) for mask in geometry["position_valid"]], [all(mask) for mask in geometry["size_valid"]], geometry["yaw_valid"]
    complete = [p and s and y for p, s, y in zip(position, size, yaw)]
    learnable = any(yaw) or any(valid and not all(fixed) for valid, fixed in zip(position, geometry["fixed_position_mask"])) or any(valid and not all(fixed) for valid, fixed in zip(size, geometry["fixed_size_mask"]))
    return {"scenes": 1, "targets": len(position), "position": sum(position), "size": sum(size), "yaw": sum(yaw),
            "full_geometry": sum(complete), "full_geometry_scenes": int(all(complete)),
            "text_eligible_scenes": int(all(complete)), "learnable_scenes": int(learnable),
            "zero_supervision_scenes": int(not learnable)}


def _floor_declarations(row: dict, floors: Counter):
    """Floor support comes only from a source floor anchor on a known floor: declared targets sit exactly on
    the floor (snapped when within FLOOR_SNAP_M), anchored targets farther away stay undeclared with free z."""
    room = row["condition"]["room"]
    floor_known, floor = room.get("floor_known") is True, room.get("floor_z_m")
    requests = {o["id"]: o for o in row["condition"]["objects"]}
    for target, evidence in zip(row["target"]["objects"], row["provenance"]["field_evidence"]):
        request, z = requests[target["id"]], target["bottom_center_m"][2]
        declared, snapped = request.get("support_parent") == "floor", evidence.get("legacy_z_snap_applied_to_target")
        if not isinstance(snapped, bool):
            raise ValueError("legacy_z_snap_applied_to_target must be a boolean")
        anchored = evidence.get("raw_anchor") == "floor" and floor_known and not evidence.get("source_support_inferred")
        if declared and (not anchored or z != floor):
            raise ValueError("floor support declared without a source floor anchor and an on-floor target")
        if snapped and not declared:
            raise ValueError("target z snapped to the floor without a floor declaration")
        if anchored and request.get("support_parent") is None and abs(z - floor) <= FLOOR_SNAP_M:
            raise ValueError("source floor anchor within the snap tolerance was not declared")
        floors["floor_declarations_written"] += declared
        floors["floor_declaration_z_snapped"] += snapped
        floors["floor_declaration_skipped_floating"] += anchored and request.get("support_parent") is None


def _footprint(box: dict):
    from shapely.geometry import Polygon
    (x, y, _), (w, d, _), yaw = box["bottom_center_m"], box["target_size_local_m"], box["yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    return Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in ((w/2, d/2), (-w/2, d/2), (-w/2, -d/2), (w/2, -d/2))])


def _selection_rules(row: dict, source: str, walls: Counter, frozen_targets: int | None):
    """Round 10, recomputed from the row: exactly ``frozen_targets`` (the frozen v3.2 row's placements) frozen_prep
    targets, none a source wall or ceiling anchor (the frozen anchors=[floor, object] dropped those); ``support_parent``
    "wall" exactly on ``wall_anchor`` targets (a source wall anchor of a WALL_ANCHOR_SOURCES source) in a known
    boundary, each with a footprint side within ``validation.WALL_GAP_M`` of it; an added supported item declares its
    source (not inferred) parent, whose rule matches (frozen_prep: inside, else on_added), and its target centre lies on
    that parent's target footprint (+SUPPORT_TOL_M) with its bottom within the parent box (+-SUPPORT_TOL_M, plus
    FLOOR_SNAP_M for a floor-snapped parent). It cannot see an eligible source object the builder left out."""
    from shapely.geometry import Point
    room = row["condition"]["room"]
    known = room.get("boundary_known") is True
    requests = {o["id"]: o for o in row["condition"]["objects"]}
    boxes = {t["id"]: t for t in row["target"]["objects"]}
    rules = {t["id"]: e.get("selection_rule") for t, e in zip(row["target"]["objects"], row["provenance"]["field_evidence"])}
    for target, evidence in zip(row["target"]["objects"], row["provenance"]["field_evidence"]):
        rule, parent = rules[target["id"]], requests[target["id"]].get("support_parent")
        if rule not in SELECTION_RULES:
            raise ValueError("unknown field_evidence.selection_rule")
        if rule == "frozen_prep" and evidence.get("raw_anchor") in ("wall", "ceiling"):
            raise ValueError("frozen_prep target with a source wall/ceiling anchor (the frozen prep keeps floor/object)")
        wall = rule == "wall_anchor"
        if wall and (evidence.get("raw_anchor") != "wall" or source not in WALL_ANCHOR_SOURCES):
            raise ValueError("wall_anchor target without a source wall anchor of a wall-anchor source")
        if parent == "wall" and not (wall and known):
            raise ValueError("support_parent wall must be declared exactly on wall_anchor targets in a known boundary")
        if parent == "wall" and not on_wall(target, room):
            raise ValueError("wall-declared target has no footprint side within WALL_GAP_M of the room boundary")
        if parent != "wall" and wall and known and on_wall(target, room):  # off-wall anchors stay undeclared
            raise ValueError("support_parent wall must be declared exactly on wall_anchor targets touching a known boundary")
        if wall and evidence.get("wall_contact") != (on_wall(target, room) if known else None):
            raise ValueError("field_evidence.wall_contact differs from the target's geometric wall contact")
        if rule.startswith("support_"):
            box = boxes.get(parent)
            if box is None or evidence.get("raw_anchor") != "object" or evidence.get("source_support_inferred") is not False:
                raise ValueError("added supported item must declare its source (not inferred) parent")
            if (rules[parent] == "frozen_prep") != (rule == "support_inside_parent"):
                raise ValueError("support selection rule disagrees with its parent's selection rule")
            x, y, z = target["bottom_center_m"]
            bottom, slack = box["bottom_center_m"][2], SUPPORT_TOL_M + FLOOR_SNAP_M + 1e-6
            if (_footprint(box).distance(Point(x, y)) > SUPPORT_TOL_M + 1e-6
                    or not bottom - slack <= z <= bottom + box["target_size_local_m"][2] + slack):
                raise ValueError("added supported item is not on its parent's footprint within its box")
        walls["wall_declarations_written"] += parent == "wall"
        walls[rule] += 1
    if frozen_targets is not None and sum(rule == "frozen_prep" for rule in rules.values()) != frozen_targets:
        raise ValueError("frozen_prep target count differs from the frozen v3.2 row's placements")


def _evidence_counts(row: dict, kept: Counter, omitted: Counter, reasons: Counter, corrections: Counter):
    for constraint in row["condition"]["constraints"]:
        kept[constraint["type"]] += 1
    provenance = row["provenance"]
    for item in provenance.get("omitted_legacy_constraints", []):
        if not isinstance(item, dict) or not isinstance(item.get("legacy_constraint"), list) or not item["legacy_constraint"]:
            raise ValueError("invalid omitted legacy constraint record")
        omitted[_text(item["legacy_constraint"][0], "omitted constraint type")] += 1
        reasons[_text(item.get("reason"), "omitted constraint reason")] += 1
    source_ids, evidence = provenance.get("target_source_ids"), provenance.get("field_evidence")
    n = len(row["target"]["objects"])
    if (not isinstance(source_ids, list) or len(source_ids) != n
            or any(not isinstance(v, str) or not v for v in source_ids) or len(set(source_ids)) != n):
        raise ValueError("target_source_ids must preserve one distinct source instance per target")
    if not isinstance(evidence, list) or len(evidence) != n or any(not isinstance(v, dict) for v in evidence):
        raise ValueError("field_evidence must preserve one record per target")
    summary = provenance.get("source_meta", {}).get("v2_evidence", {})
    for field in ("opening_width_corrections", "front_unknown_objects"):
        value = summary.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("source correction counts must be nonnegative integers")
        corrections[field] += value
    for item in evidence:
        recorded = item.get("recorded_source_evidence", {})
        if not isinstance(recorded, dict):
            raise ValueError("recorded_source_evidence must be an object")
        corrections["target_opening_width_corrections"] += "corrected_width_m" in recorded
        corrections["target_front_unknown_objects"] += "front_unknown_reason" in recorded
        corrections["tilted_masked_targets"] += bool(item.get("tilted"))


def _same_counts(observed: dict, declared: Any, label: str, audit: _Audit):
    if (not isinstance(declared, dict)
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in declared.values())
            or {k: v for k, v in observed.items() if v} != {k: v for k, v in declared.items() if v}):
        audit.fail("manifest.json", f"{label} differs from independent streaming counts")


def _report_path(output: Any, data: Path, release: Path, manifest: dict) -> Path | None:
    if output is None:
        return None
    target = Path(output).resolve()
    protected = [data, release, SOURCE_ROOT.resolve()]
    evidence = [Path(p).resolve() for p in manifest.get("evidence_sha256", {})]
    if evidence:
        protected.append(Path(os.path.commonpath([str(p.parent) for p in evidence])))
    if any(target == root or root in target.parents or target in root.parents for root in protected):
        raise ValueError("report output must stay outside read-only data, release, evidence and raw sources")
    if target.exists():
        raise FileExistsError(f"report already exists: {target}")
    return target


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}-", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
        os.link(temporary, path)  # Atomic publication, never replaces another report.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_selected_dataset(data_root: Any, output: Any = None) -> dict:
    data = Path(data_root).resolve()
    manifest = _json_file(data / "manifest.json")
    if manifest.get("builder") != "selected-v3.2-bridge" or manifest.get("schema_version") != "fastfill.v2":
        raise ValueError("selected-v3.2 FastFill-v2 migration manifest required")
    release = Path(_text(manifest.get("release_root"), "release_root")).resolve()
    report_path = _report_path(output, data, release, manifest)
    audit = _Audit()
    frozen_path = release / "data/v3.2/MANIFEST.json"
    frozen_manifest_hash = _hash_file(frozen_path)["sha256"]
    frozen = _json_file(frozen_path)
    source_hashes = {str(release / "data/v3.2" / name): record["sha256"] for name, record in frozen["files"].items()}
    source_hashes = {**source_hashes, **{str(release / "ir" / name): value for name, value in frozen["ir_sha256"].items()}}
    line_counts = {str(release / "data/v3.2" / name): record["lines"] for name, record in frozen["files"].items() if "lines" in record}
    if manifest.get("source_hashes") != source_hashes:
        audit.fail("manifest.json", "source hash inventory differs from frozen selected release")
    if manifest.get("selected_sources") != frozen["args"]["sources"]:
        audit.fail("manifest.json", "selected_sources differs from frozen selection")
    if manifest.get("source_data_modified") is not False:
        audit.fail("manifest.json", "source_data_modified must be false")
    filter_value = manifest.get("legacy_train_flag_filter")
    if filter_value not in ([], list(DEFAULT_REJECT_FLAGS)):
        audit.fail("manifest.json", "unknown legacy_train_flag_filter")
        filter_value = list(DEFAULT_REJECT_FLAGS)
    bounded = manifest.get("bounded_build")
    if not isinstance(bounded, bool):
        audit.fail("manifest.json", "bounded_build must be a boolean")
    expected, selected_counts, filtered = _saved_selection(release, tuple(filter_value), audit)
    hashes = {"manifest.json": _hash_file(data / "manifest.json")["sha256"]}
    counts = {split: Counter({key: 0 for key in COUNT_FIELDS}) for split in SPLITS}
    sources, seen, houses, kept_rows, rejected_rows = {}, set(), {}, Counter(), Counter()
    kept_types, omitted_types, omitted_reasons, corrections, rejection_reasons, details, swaps, floors, walls = (Counter() for _ in range(9))
    for split in SPLITS:
        for context, row in _json_lines(data / f"{split}.jsonl", audit, hashes):
            try:
                uid, source, provenance = _lineage(row, split, expected, seen, audit, context)
                kept_rows[(source, split)] += 1
                _check_house(provenance, source, split, houses, audit, context)
                geometry = _geometry(row)
                if provenance.get("descriptions") != "source_desc_or_category":
                    audit.fail(context, "provenance.descriptions must be source_desc_or_category")
                if provenance.get("height_conflict") != _height_conflict(row):
                    audit.fail(context, "provenance.height_conflict differs from the K2 recomputation on the kept room height")
                # The frozen prep dropped a source height because of a target; the bridge keeps it (MultiScan: if reliable).
                if (provenance.get("legacy_height_dropped") and row["condition"]["room"].get("height_m") is None
                        and (source != "MultiScan" or (provenance.get("source_meta") or {}).get("height_reliable"))):
                    audit.fail(context, "room.height_m is null where the frozen prep dropped it for a target; the bridge keeps the source height")
                details[f"{source}:yaw_valid"] += sum(geometry["yaw_valid"])
                for order in row["validity"]["yaw_symmetry_order"]:
                    details[f"{source}:{order}"] += 1
                swaps[source] += sum(row["validity"]["size_axis_swap_allowed"])
                categories = {o["id"]: o["category"] for o in row["condition"]["objects"]}
                if row["validity"]["size_axis_swap_allowed"] != [manifest.get("front_policy") == "axis" and size_axis_swap_allowed(
                        source, categories[o["id"]]) for o in row["target"]["objects"]]:
                    audit.fail(context, "size_axis_swap_allowed differs from the source/category axis policy")
                details["exchangeable_groups"] += geometry["exchangeable_groups"]
                details["exchangeable_members"] += geometry["exchangeable_members"]
                measured = _row_counts(geometry)
                counts[split].update(measured)
                sources.setdefault(source, {}).setdefault(split, Counter({key: 0 for key in COUNT_FIELDS})).update(measured)
                if measured["zero_supervision_scenes"]:
                    audit.fail(context, "no learnable geometry supervision")
                _evidence_counts(row, kept_types, omitted_types, omitted_reasons, corrections)
                _floor_declarations(row, floors)
                _selection_rules(row, source, walls, (expected.get(uid) or (None,) * 5)[4])
                _check_unreferenced_groups(row)
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                audit.fail(context, str(exc))
    for context, row in _json_lines(data / "rejections.jsonl", audit, hashes):
        try:
            uid, source, split = (_text(row.get(key), key) for key in ("uid", "source", "split"))
            if uid in seen:
                audit.fail(context, f"duplicate UID in kept/rejected outputs: {uid}")
            seen.add(uid)
            assignment = expected.get(uid)
            if assignment is None or assignment[:3] != (split, source, False):
                audit.fail(context, "rejected UID source/split differs from eligible frozen selection")
            rejected_rows[(source, split)] += 1
            rejection_reasons[_text(row.get("reason"), "rejection reason")] += 1
        except (ValueError, TypeError, KeyError) as exc:
            audit.fail(context, str(exc))
    missing = sum(not excluded and uid not in seen for uid, (_, _, excluded, *_) in expected.items())
    if missing and not bounded:
        audit.fail("selection", f"unaccounted selected UID: {missing} eligible saved scenes missing from kept/rejected output")
    split_counts = {split: dict(values) for split, values in counts.items()}
    _same_counts({s: c["scenes"] for s, c in counts.items()}, manifest.get("split_samples"), "split_samples", audit)
    _same_counts({f"{source}:{split}": n for (source, split), n in kept_rows.items()}, manifest.get("source_split_samples"), "source_split_samples", audit)
    _same_counts({key: sum(c[key] for c in counts.values()) for key in ("targets", "position", "size", "yaw", "full_geometry")}, manifest.get("valid_label_counts"), "valid_label_counts", audit)
    _same_counts(dict(filtered), manifest.get("legacy_train_filtered_by_source"), "legacy_train_filtered_by_source", audit)
    _same_counts({k.split(":")[0]: v for k, v in details.items() if k.endswith(":yaw_valid")}, manifest.get("source_yaw_valid_objects"), "source_yaw_valid_objects", audit)
    _same_counts({k: v for k, v in details.items() if ":" in k and not k.endswith(":yaw_valid")}, manifest.get("source_yaw_symmetry_order_counts"), "source_yaw_symmetry_order_counts", audit)
    _same_counts({k: details[k] for k in ("exchangeable_groups", "exchangeable_members")}, manifest.get("exchangeable_group_counts"), "exchangeable_group_counts", audit)
    _same_counts(dict(swaps), manifest.get("source_size_axis_swap_allowed_objects"), "source_size_axis_swap_allowed_objects", audit)
    for key in ("floor_declarations_written", "floor_declaration_z_snapped", "floor_declaration_skipped_floating"):
        if isinstance(manifest.get(key), bool) or manifest.get(key) != floors[key]:
            audit.fail("manifest.json", f"{key} differs from independent streaming counts")
    if isinstance(manifest.get("wall_declarations_written"), bool) or manifest.get("wall_declarations_written") != walls["wall_declarations_written"]:
        audit.fail("manifest.json", "wall_declarations_written differs from independent streaming counts")
    _same_counts({k: walls[k] for k in SELECTION_RULES}, manifest.get("selection_rule_counts"), "selection_rule_counts", audit)
    if manifest.get("selection_extension_policy") != SELECTION_POLICY:
        audit.fail("manifest.json", "selection_extension_policy differs from legacy_bridge.SELECTION_POLICY")
    if manifest.get("yaw_policy") != YAW_POLICY.get(manifest.get("front_policy")):
        audit.fail("manifest.json", "yaw_policy differs from legacy_bridge.YAW_POLICY[front_policy]")
    if manifest.get("descriptions") != "source_desc_or_category":
        audit.fail("manifest.json", "descriptions must be source_desc_or_category")
    _same_counts(dict(rejection_reasons), manifest.get("v2_rejections"), "v2_rejections", audit)
    if (isinstance(manifest.get("samples_written"), bool)
            or manifest.get("samples_written") != sum(c["scenes"] for c in counts.values())):
        audit.fail("manifest.json", "samples_written differs from independent streaming counts")
    evidence_hashes = manifest.get("evidence_sha256", {})
    if not isinstance(evidence_hashes, dict) or len(evidence_hashes) != 2 or not all(any(Path(p).as_posix().endswith(suffix) for p in evidence_hashes) for suffix in ("designer/opening-target-join.json", "internscenes/k0_saved_object_exposure.jsonl")):
        audit.fail("manifest.json", "required two recorded evidence hashes are missing")
        evidence_hashes = evidence_hashes if isinstance(evidence_hashes, dict) else {}
    implementation = manifest.get("implementation_sha256", {})
    expected_code = {str(Path(__file__).with_name(name)): name for name in ("legacy_build.py", "legacy_bridge.py", "legacy_evidence.py")}
    if not isinstance(implementation, dict) or set(implementation) != set(expected_code):
        audit.fail("manifest.json", "migration implementation hash inventory differs from expected modules")
        implementation = implementation if isinstance(implementation, dict) else {}
    package = Path(__file__).resolve().parents[1]
    preparation = {str(package / name): frozen["code_sha256"][name] for name in ("build.py", "scene.py", "anchors.py", "validate.py", "split.py")}
    hash_checks = {"source": _check_hashes("source", source_hashes, audit, line_counts=line_counts),
                   "evidence": _check_hashes("evidence", evidence_hashes, audit),
                   "implementation": _check_hashes("implementation", implementation, audit),
                   "frozen_preparation": _check_hashes("frozen preparation", preparation, audit)}
    if _hash_file(data / "manifest.json")["sha256"] != hashes["manifest.json"]:
        audit.fail("manifest.json", "dataset manifest changed during verification")
    if _hash_file(frozen_path)["sha256"] != frozen_manifest_hash:
        audit.fail(str(frozen_path), "frozen manifest changed during verification")
    accounting = {"selected": len(expected), "filtered_train": sum(filtered.values()),
                  "candidates": len(expected) - sum(filtered.values()), "kept": sum(kept_rows.values()),
                  "rejected": sum(rejected_rows.values()), "unaccounted": missing}
    source_accounting = {}
    for (source, split), selected in sorted(selected_counts.items()):
        n_filtered = filtered[source] if split == "train" else 0
        kept, rejected = kept_rows[(source, split)], rejected_rows[(source, split)]
        source_accounting.setdefault(source, {})[split] = {"selected": selected, "filtered_train": n_filtered,
            "candidates": selected - n_filtered, "kept": kept, "rejected": rejected,
            "unaccounted": selected - n_filtered - kept - rejected}
    report = {"verifier_version": 1, "schema_version": "fastfill.v2", "passed": audit.error_count == 0,
              "data_root": str(data), "release_root": str(release), "bounded_build": bounded,
              "complete_selection_coverage": not bounded and missing == 0,
              "legacy_train_flag_filter": filter_value, "selection_accounting": accounting,
              "source_selection_accounting": source_accounting, "split_counts": split_counts,
              "source_split_counts": {s: {k: dict(v) for k, v in splits.items()} for s, splits in sources.items()},
              "kept_legacy_constraint_types": dict(kept_types), "omitted_legacy_constraint_types": dict(omitted_types),
              "omitted_legacy_constraint_reasons": dict(omitted_reasons), "source_corrections": dict(corrections),
              "selection_rule_counts": {k: walls[k] for k in SELECTION_RULES},
              "wall_declarations_written": walls["wall_declarations_written"],
              "hash_checks": hash_checks, "dataset_sha256": hashes, "frozen_manifest_sha256": frozen_manifest_hash,
              "underlying_house_and_alias_index_entries": len(houses), "errors_total": audit.error_count,
              "errors": audit.errors, "errors_truncated": audit.error_count > len(audit.errors),
              "eligibility_scope": "complete field-valid geometry; text subset excludes context/object budgets; no asset, mesh or physics acceptance claim"}
    if report_path is not None:
        _write_report(report_path, report)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify_selected_dataset(args.data_root, args.output)
    print(json.dumps({"passed": report["passed"], "errors_total": report["errors_total"],
                      "selection_accounting": report["selection_accounting"], "report": str(args.output)}, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
