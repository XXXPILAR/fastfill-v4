"""Immutable multi-source partial supervision for the three-field bbox task.

This module produces a comparison view; the full-condition production task
retains its original room/fixed/support/constraint inputs.

Reference extents describe source structural input, not certified rectangular
physical rooms. Every well-formed parent scene remains in the primary view;
training object/context/active-objective admission is a separate preflight.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import sqlite3
import tempfile

import torch

from .batch import TinyTokenizer, _geometry_rows, room_normalization, tokenize_condition
from .direct_layout import request_to_condition
from .io import HOLDOUT_REASON, ROOMGENBENCH_HOLDOUT_GROUPS, fingerprint, safe_output  # noqa: F401 (re-exported)
from .legacy_bridge import FLOOR_SNAP_M
from .minimal_data import SPATIALLM_IR_SHA256
from .schema import migrate_legacy_row, validate_condition

SPLITS = ("train", "validation", "test")
DEGENERATE_AXIS_M = .003
SOURCES = {
    "HSSD200": "HSSD-200", "IL3D_3dfront": "IL3D", "IL3D_synthetic": "IL3D",
    "InteriorGS": "InteriorGS", "InternScenes_3rscan": "InternScenes",
    "InternScenes_arkit": "InternScenes", "InternScenes_gen": "InternScenes",
    "InternScenes_mp3d": "InternScenes", "InternScenes_scannet": "InternScenes",
    "MansionWorld": "MansionWorld", "MultiScan": "MultiScan",
    "OptiScene_holodeck": "OptiScene", "SAGE-10k": "SAGE-10k",
    "Scan2CAD": "SceneCAD & Scan2CAD", "SpatialLM": "SpatialLM",
    "Structured3D": "Structured3D", "SceneSmith": "SceneSmith", "SpatialGen": "SpatialGen",
}
EVALUATION_SOURCES = {"SceneSmith", "SpatialGen"}
AUXILIARY_FAMILIES = ("3D-FRONT", "3RScan", "ARKitScenes")
UNKNOWN_TYPES = {"", "misc", "other", "other room", "unknown", "undefined", "none"}
PROJECTION = "room_type_reference_extent_furniture_only-v1"
YAW_SEMANTICS = "local_bbox_axes_mod_pi_not_certified_semantic_front"


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON literal: {value}")


def _loads(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_nonfinite)


def _write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n")


def _parent_rows(parent, holdout_groups=()):
    """Yield (parent split, line, row) in file order; holdout rows leaving train/validation come last."""
    deferred = []
    for split in SPLITS:
        with (parent / (split + ".jsonl")).open() as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    row = _loads(line)
                    if split != "test" and row["provenance"].get("group") in holdout_groups:
                        deferred.append((split, number, row))
                    else:
                        yield split, number, row
    yield from deferred


def _holdout(provenance, split, holdout_groups):
    """Benchmark rooms are written to test; returns (provenance, change-or-None)."""
    if split == "test" or provenance.get("group") not in holdout_groups:
        return provenance, None
    return ({**provenance, "split": "test", "holdout_reason": HOLDOUT_REASON},
            {"field": "provenance.split", "before": split, "after": "test", "reason": HOLDOUT_REASON})


def _source_role(source, split):
    if source not in SOURCES:
        raise ValueError("source_taxonomy_unknown")
    if split not in SPLITS or source in EVALUATION_SOURCES and split != "test":
        raise ValueError("source_role_split_conflict")
    return "evaluation_only" if source in EVALUATION_SOURCES else "train_family"


def _split_integrity(parent):
    uids, groups, counts = set(), {}, Counter()
    for split, number, row in _parent_rows(parent):
        p = row["provenance"]
        _source_role(p.get("source"), split)
        uid = p.get("scene_id")
        if not isinstance(uid, str) or not uid or uid in uids or p.get("split") != split:
            raise ValueError(f"parent UID/split integrity failure at {split}:{number}")
        uids.add(uid)
        counts[split] += 1
        for alias in (p.get("group"), p.get("house_id")):
            if not isinstance(alias, str) or not alias:
                raise ValueError("parent underlying group identity missing")
            if alias in groups and groups[alias] != split:
                raise ValueError("parent group alias has cross-split conflict")
            groups[alias] = split
    return {"parent_uids_checked": len(uids), "parent_group_aliases_checked": len(groups),
            "duplicate_uids": 0, "group_cross_split_conflicts": 0, "parent_split_samples": dict(counts),
            "split_rule": "inherit parent without re-splitting"}


def _size_policy(model_config):
    values = asdict(model_config) if is_dataclass(model_config) else model_config or {}
    reference = values.get("size_reference", (1., 1., 1.))
    limit = values.get("size_log_limit", 10.)
    if (not isinstance(reference, (list, tuple)) or len(reference) != 3
            or not all(_finite(v) and v > 0 for v in reference)
            or not _finite(limit) or not 0 < limit <= 30):
        raise ValueError("invalid configured size output range")
    low = [v * math.exp(-limit) for v in reference]
    high = [v * math.exp(limit) for v in reference]
    limits = torch.finfo(torch.float32)
    if any(lo < limits.tiny or hi > limits.max for lo, hi in zip(low, high)):
        raise ValueError("configured size output range must preserve positive finite float32 sizes")
    return {"size_reference": list(reference), "size_log_limit": limit,
            "minimum_size_m": low, "maximum_size_m": high,
            "outside_range_policy": "mask whole size vector; preserve original numeric labels"}


def _vertical_reframe(source_room):
    if source_room["source"] != "MultiScan":
        return 0.
    meta = source_room.get("meta", {})
    if meta.get("n_floor_objects_outside_structure", 0):
        raise ValueError("source_boundary_target_expanded")
    if not all(_finite(meta.get(key)) for key in ("floor_z", "floor_height")):
        raise ValueError("source_vertical_reference_missing")
    return float(meta["floor_z"]) - float(meta["floor_height"])


def _normalized_vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("source/parent geometry must contain three coordinates")
    if any(v is not None and not _finite(v) for v in value):
        raise ValueError("source/parent geometry must contain finite numbers or null")
    return list(value)


def _source_geometry(sample, raw):
    p, room = sample["provenance"], sample["condition"]["room"]
    if (raw.get("source") != p.get("source") or raw.get("uid") != p.get("scene_id")
            or raw.get("group") != p.get("group") or raw.get("group") != p.get("house_id")):
        raise ValueError("source_identity_mismatch")
    if raw.get("boundary") != room["floor_polygon_xy_m"]:
        raise ValueError("source_boundary_mismatch")
    if raw.get("room_type") != room.get("room_type"):
        raise ValueError("source_room_type_mismatch")
    dz = _vertical_reframe(raw)
    if p.get("vertical_reframe_m", 0.) != dz:
        raise ValueError("source_vertical_reference_mismatch")
    raw_objects = {obj["id"]: obj for obj in raw.get("objects", [])}
    if len(raw_objects) != len(raw.get("objects", [])):
        raise ValueError("duplicate source object identity")
    targets, ids, fields = sample["target"]["objects"], p.get("target_source_ids", []), p.get("field_evidence", [])
    if len(ids) != len(targets) or len(set(ids)) != len(ids) or len(fields) != len(targets):
        raise ValueError("source_identity_missing_or_duplicate")
    joined, requests = [], {obj["id"]: obj for obj in sample["condition"]["objects"]}
    for target, ident, evidence in zip(targets, ids, fields):
        obj = raw_objects.get(ident)
        if obj is None:
            raise ValueError("source_identity_missing_object")
        pos = _normalized_vector(obj.get("pos"))
        expected_pos = [*pos[:2], pos[2] + dz if pos[2] is not None else None]
        if evidence.get("legacy_z_snap_applied_to_target") is True:
            # legacy_bridge K3: a declared source floor anchor within FLOOR_SNAP_M of the known floor is snapped.
            floor = room.get("floor_z_m")
            if (requests[target["id"]].get("support_parent") != "floor" or not _finite(floor)
                    or not _finite(expected_pos[2]) or abs(expected_pos[2] - floor) > FLOOR_SNAP_M):
                raise ValueError("source_floor_snap_mismatch")
            expected_pos[2] = floor
        yaw = obj.get("yaw")
        expected_yaw = (yaw + math.pi) % (2 * math.pi) - math.pi if _finite(yaw) else None
        if (expected_pos != target.get("bottom_center_m") or obj.get("size") != target.get("target_size_local_m")
                or expected_yaw != target.get("yaw_rad") or bool(obj.get("tilted")) != evidence.get("tilted")):
            raise ValueError("source_geometry_mismatch")
        joined.append(obj)
    return joined


def _frame(sample):
    room = sample["condition"]["room"]
    validate_condition(sample["condition"])
    points = room["floor_polygon_xy_m"]
    low = [min(p[q] for p in points) for q in (0, 1)]
    dimensions = [max(p[q] for p in points) - low[q] for q in (0, 1)]
    if not all(_finite(v) and v > 0 for v in dimensions):
        raise ValueError("source structural reference extent must be positive")
    floor = room.get("floor_z_m")
    if floor is not None and not _finite(floor):
        raise ValueError("input floor reference must be finite or null")
    room_type = room.get("room_type")
    room_type = room_type if isinstance(room_type, str) and room_type.strip().lower() not in UNKNOWN_TYPES else "unknown"
    return [*low, floor if floor is not None else 0.], dimensions, room_type


def _parent_validity(sample):
    targets, validity = sample["target"]["objects"], sample["validity"]
    result = deepcopy(validity)
    if any(key not in result for key in ("position", "size", "yaw")):
        raise ValueError("parent geometry validity fields missing")
    for key, target_field in (("position", "bottom_center_m"), ("size", "target_size_local_m")):
        if len(result[key]) != len(targets):
            raise ValueError(f"invalid parent {key} mask length")
        for mask, target in zip(result[key], targets):
            if not isinstance(mask, list) or len(mask) != 3 or not all(isinstance(v, bool) for v in mask):
                raise ValueError(f"invalid parent {key} mask")
            values = _normalized_vector(target.get(target_field))
            if any(flag and (not _finite(v) or key == "size" and v <= 0) for flag, v in zip(mask, values)):
                raise ValueError(f"valid {key} target must be finite" + (" and positive" if key == "size" else ""))
    if len(result["yaw"]) != len(targets) or not all(isinstance(v, bool) for v in result["yaw"]):
        raise ValueError("invalid parent yaw mask")
    for mask, target in zip(result["yaw"], targets):
        yaw = target.get("yaw_rad")
        if yaw is not None and not _finite(yaw) or mask and not _finite(yaw):
            raise ValueError("valid yaw target must be finite")
    for key, ok in (("yaw_symmetry_order", lambda v: type(v) is int and v >= 1),
                    ("exchangeable_group", lambda v: v is None or isinstance(v, str) and v),
                    ("size_axis_swap_allowed", lambda v: type(v) is bool)):
        if key in result and (len(result[key]) != len(targets) or not all(ok(v) for v in result[key])):
            raise ValueError(f"invalid parent {key} row")
    return result


def qualify_parent_masks(sample, *, model_config=None):
    """Apply only D1/D2 numeric demotions; preserve full-task yaw semantics."""
    policy = _size_policy(model_config)
    masks, changes = _parent_validity(sample), []
    p = sample["provenance"]
    for i, target in enumerate(sample["target"]["objects"]):
        base = {"object_id": target["id"], "target_source_id": p["target_source_ids"][i]}
        if p["source"] == "Scan2CAD" and any(masks["position"][i]):
            changes.append({**base, "field": "position", "before": masks["position"][i], "after": [False] * 3,
                            "reason": "Scan2CAD_estimated_floor_and_upstream_snap_uncertainty"})
            masks["position"][i] = [False] * 3
        size = target["target_size_local_m"]
        if any(masks["size"][i]) and any(flag and not lo <= value <= hi for flag, value, lo, hi in zip(
                masks["size"][i], size, policy["minimum_size_m"], policy["maximum_size_m"])):
            changes.append({**base, "field": "size", "before": masks["size"][i], "after": [False] * 3,
                            "value_m": deepcopy(size), "reason": "outside_configured_size_head_range"})
            masks["size"][i] = [False] * 3
        if any(masks["size"][i]) and any(_finite(value) and value < DEGENERATE_AXIS_M for value in size):
            changes.append({**base, "field": "size", "before": masks["size"][i], "after": [False] * 3,
                            "value_m": deepcopy(size), "reason": "degenerate_axis_lt_3mm"})
            masks["size"][i] = [False] * 3
    return masks, changes


def _masks(sample, joined, policy):
    masks, changes = qualify_parent_masks(sample, model_config=policy)
    p, geometric = sample["provenance"], []
    for i, (target, raw, evidence) in enumerate(zip(sample["target"]["objects"], joined, p["field_evidence"])):
        base = {"object_id": target["id"], "target_source_id": p["target_source_ids"][i]}
        qualified = (p["source"] == "SpatialLM" and raw.get("tilted") is False and evidence.get("tilted") is False
                     and evidence.get("size_semantics") == "canonical_source_IR" and _finite(raw.get("yaw"))
                     and all(_finite(v) and v > 0 for v in raw["size"]) and all(_finite(v) for v in raw["pos"]))
        geometric.append(qualified)
        if qualified and not masks["yaw"][i]:
            changes.append({**base, "field": "yaw", "before": False, "after": True,
                            "reason": "pinned_SpatialLM_full_local_extents_and_geometric_yaw"})
            masks["yaw"][i] = True
    # Bbox-axis yaw is mod pi; the box-symmetry tier (size_axis_swap_allowed) stays order 4.
    return {**masks, "yaw_symmetry_order": [4 if swap else 2 for swap in masks["size_axis_swap_allowed"]]}, changes, geometric


def _floor_evidence(sample):
    room, source = sample["condition"]["room"], sample["provenance"]["source"]
    return {"floor_source": "estimated" if source == "Scan2CAD" else (
                "source_canonical_reference" if room.get("floor_z_m") is not None else "unknown"),
            "parent_floor_known": room.get("floor_known"), "parent_floor_z_m": room.get("floor_z_m"),
            "upstream_z_snap_possible": source == "Scan2CAD",
            "per_object_pre_snap_z": "unavailable_in_frozen_IR" if source == "Scan2CAD" else "not_inferred"}


def project_sample(sample, source_room, *, split, model_config=None, holdout_groups=()):
    """Project input only, preserve numeric labels and explicitly qualify masks."""
    sample = migrate_legacy_row(sample)
    p = sample["provenance"]
    _source_role(p.get("source"), split)
    if p.get("split") != split:
        raise ValueError("source_split_mismatch")
    joined = _source_geometry(sample, source_room)
    origin, dimensions, room_type = _frame(sample)
    requests, targets = sample["condition"]["objects"], sample["target"]["objects"]
    if [o["id"] for o in requests] != [o["id"] for o in targets]:
        raise ValueError("parent request/target correspondence or order mismatch")
    request = {"room_type": room_type, "room_size_m": dimensions, "furniture_list": [
        {"id": obj["id"], "category": obj["category"], "description": obj["description"], "count": 1} for obj in requests]}
    condition = request_to_condition(request, max_objects=max(128, len(requests)), room_size_semantics="reference_extent")
    masks, changes, geometric = _masks(sample, joined, _size_policy(model_config))
    translated = [{**deepcopy(obj), "bottom_center_m": [value - offset if value is not None else None
                   for value, offset in zip(obj["bottom_center_m"], origin)]} for obj in targets]
    if any(v != 0 for v in origin):
        changes.append({"field": "bottom_center_m", "operation": "common_input_reference_translation",
                        "offset_m": [-v for v in origin], "reason": "structural_input_extent_and_explicit_floor_reference"})
    p, moved = _holdout(p, split, holdout_groups)
    changes += [moved] if moved else []
    provenance = {**deepcopy(p), "minimal_request": request, "condition_projection": PROJECTION,
                  "room_size_semantics": "reference_extent", "room_dimension_mode": "xy",
                  "frame_translation_m": [-v for v in origin], "parent_condition": deepcopy(sample["condition"]),
                  "parent_validity": deepcopy(sample["validity"]), "source_floor_evidence": _floor_evidence(sample),
                  "yaw_label_semantics": YAW_SEMANTICS, "geometric_yaw_qualified": geometric,
                  "qualification_changes": changes, "correspondence": "fixed_request_identity",
                  "source_role": _source_role(p["source"], split), "removed_condition_dependencies": _removed_dependencies(sample),
                  "geometry_checks_scope": "source identity/numerics/masks only; no physical legality certification"}
    result = {"schema_version": "fastfill.v2", "condition": condition,
              "target": {"schema_version": "fastfill.v2", "objects": translated}, "validity": masks, "provenance": provenance}
    _geometry_rows(result, *room_normalization(condition["room"]))
    return result


def _strict_rectangle(sample):
    from shapely.geometry import Polygon
    room = sample["provenance"]["parent_condition"]["room"]
    points = room["floor_polygon_xy_m"]
    dimensions = sample["provenance"]["minimal_request"]["room_size_m"]
    area = dimensions[0] * dimensions[1]
    return (sample["provenance"]["minimal_request"]["room_type"] != "unknown"
            and sample["provenance"]["source"] != "Scan2CAD"
            and room.get("boundary_known") is True and room.get("floor_known") is True
            and _finite(room.get("floor_z_m")) and Polygon(points).is_valid
            and abs(Polygon(points).area - area) <= max(1e-8, area * 1e-6))


def _index_ir(connection, ir):
    connection.execute("CREATE TABLE raw (source TEXT, uid TEXT PRIMARY KEY, row TEXT)")
    counts = Counter()
    for source in SOURCES:
        with (ir / (source + ".jsonl")).open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = _loads(line)
                if raw.get("source") != source or not isinstance(raw.get("uid"), str) or not raw["uid"]:
                    raise ValueError(f"source IR identity invalid: {source}:{number}")
                try:
                    connection.execute("INSERT INTO raw VALUES (?,?,?)", (source, raw["uid"], line))
                except sqlite3.IntegrityError as error:
                    raise ValueError("duplicate source IR UID") from error
                counts[source] += 1
        connection.commit()
    return dict(counts)


def _input_hashes(parent, ir, frozen_manifest):
    paths = [parent / (split + ".jsonl") for split in SPLITS] + [parent / "manifest.json"]
    paths += [ir / (source + ".jsonl") for source in SOURCES] + [frozen_manifest]
    return {str(path): fingerprint(path) for path in paths}



def _frozen_ir_guard(ir, frozen_manifest, before):
    manifest = _loads(frozen_manifest.read_text())
    expected = manifest.get("ir_sha256", {})
    names = {source + ".jsonl" for source in SOURCES}
    if set(expected) != names:
        raise ValueError("frozen manifest must explicitly bind all 18 IR sources")
    for name, digest in expected.items():
        if before[str(ir / name)] != digest:
            raise ValueError(f"frozen IR SHA256 mismatch: {name}")
    return expected


def _removed_dependencies(sample):
    condition = sample["condition"]
    fixed = condition["room"].get("fixed_objects", [])
    fixed_ids = {obj["id"] for obj in fixed}
    objects = condition["objects"]
    return {"fixed_objects": len(fixed), "constraints": len(condition["constraints"]),
            "support_requests": sum(obj.get("support_parent") is not None for obj in objects),
            "fixed_support_requests": sum(obj.get("support_parent") in fixed_ids for obj in objects),
            "exchangeable_requests": sum(g is not None for g in sample["validity"].get("exchangeable_group", [])),
            "interpretation": "omitted comparison-condition dependencies retained verbatim in parent_condition; not a complete-room physical task"}


def _record_stats(stats, sample, byte_length):
    p, masks = sample["provenance"], sample["validity"]
    split, source, count = p["split"], p["source"], len(sample["target"]["objects"])
    for key, ident, value in (("split_samples", split, 1), ("split_objects", split, count),
                              ("source_split_samples", f"{source}:{split}", 1),
                              ("source_split_objects", f"{source}:{split}", count),
                              ("object_count_histogram", str(count), 1), ("utf8_context_histogram", str(byte_length), 1)):
        stats[key][ident] += value
    for field in ("position", "size", "yaw"):
        n = sum(all(v) if isinstance(v, list) else v for v in masks[field])
        stats["source_split_validity"][f"{source}:{split}:{field}"] += n
        stats["validity_counts"][field] += n
    stats["validity_counts"]["geometric_yaw"] += sum(p["geometric_yaw_qualified"])
    stats["diagnostics"]["unknown_room_type_scenes"] += p["minimal_request"]["room_type"] == "unknown"
    stats["diagnostics"]["estimated_floor_scenes"] += source == "Scan2CAD"
    stats["diagnostics"]["above_default_128_object_scenes"] += count > 128
    for field in ("fixed_objects", "constraints", "support_requests", "fixed_support_requests", "exchangeable_requests"):
        n = p["removed_condition_dependencies"][field]
        stats["removed_dependency_counts"][field] += n
        stats["removed_dependency_counts"][field + "_scenes"] += n > 0
    stats["diagnostics"]["no_active_geometry_scenes"] += not any(
        all(pos) or all(size) or yaw for pos, size, yaw in zip(masks["position"], masks["size"], masks["yaw"]))


def _build_rows(parent, stage, connection, model_config, strict_rectangle, holdout_groups):
    names = ("split_samples", "split_objects", "source_split_samples", "source_split_objects", "source_split_validity",
             "object_count_histogram", "utf8_context_histogram", "validity_counts", "diagnostics", "strict_rectangle_samples",
             "removed_dependency_counts")
    stats = {name: Counter() for name in names}
    with ExitStack() as stack:
        streams = {split: stack.enter_context((stage / (split + ".jsonl")).open("x")) for split in SPLITS}
        views = {split: stack.enter_context((stage / ("strict_rectangle_" + split + ".jsonl")).open("x"))
                 for split in SPLITS} if strict_rectangle else {}
        changes = stack.enter_context((stage / "changes.jsonl").open("x"))
        stack.enter_context((stage / "exclusions.jsonl").open("x"))
        for split, number, parent_row in _parent_rows(parent, holdout_groups):
            p = parent_row["provenance"]
            source = connection.execute("SELECT row FROM raw WHERE uid=? AND source=?", (p["scene_id"], p["source"])).fetchone()
            if source is None:
                raise ValueError("parent source UID missing from frozen IR")
            row = project_sample(parent_row, _loads(source[0]), split=split, model_config=model_config, holdout_groups=holdout_groups)
            written = row["provenance"]["split"]
            _write(streams[written], row)
            for change in row["provenance"]["qualification_changes"]:
                _write(changes, {"uid": p["scene_id"], "source": p["source"], "split": split, "parent_line": number, **change})
            byte_length = len(tokenize_condition(row["condition"], TinyTokenizer())[0])
            _record_stats(stats, row, byte_length)
            if strict_rectangle and _strict_rectangle(row):
                _write(views[written], row)
                stats["strict_rectangle_samples"][written] += 1
    return {key: dict(value) for key, value in stats.items()}


def build_dataset(parent_root, ir_root, output, *, model_config=None, strict_rectangle=False,
                  expected_spatiallm_sha256=SPATIALLM_IR_SHA256, frozen_manifest=None,
                  holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS):
    """Stream all 18 frozen IR sources through disk index; publish a new view."""
    parent, ir, target = Path(parent_root).resolve(), Path(ir_root).resolve(), safe_output(output)
    if any(target == root or root in target.parents or target in root.parents for root in (parent, ir)):
        raise ValueError("output must be new and outside parent/source roots and ancestors")
    frozen = Path(frozen_manifest).resolve() if frozen_manifest else ir.parent / "data/v3.2/MANIFEST.json"
    if not frozen.is_file():
        raise ValueError("explicit frozen manifest is required to bind every source IR")
    if target == frozen or target in frozen.parents:
        raise ValueError("output must stay outside frozen manifest and its ancestors")
    policy = _size_policy(model_config)
    before = _input_hashes(parent, ir, frozen)
    frozen_hashes = _frozen_ir_guard(ir, frozen, before)
    if before[str(ir / "SpatialLM.jsonl")] != expected_spatiallm_sha256:
        raise ValueError("SpatialLM IR SHA256 differs from pinned geometric-yaw evidence")
    parent_manifest = _loads((parent / "manifest.json").read_text())
    integrity = _split_integrity(parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".multisource-build-", dir=target.parent) as work:
        stage = Path(work) / "dataset"
        stage.mkdir()
        with sqlite3.connect(Path(work) / "raw.sqlite3") as connection:
            ir_counts = _index_ir(connection, ir)
            stats = _build_rows(parent, stage, connection, model_config, strict_rectangle, tuple(holdout_groups))
        after = _input_hashes(parent, ir, frozen)
        if before != after:
            raise ValueError("parent/source inputs changed during immutable build")
        manifest = {"schema_version": "fastfill.v2", "builder": "multi-source-minimal-partial-v1",
                    "parent_root": str(parent), "ir_root": str(ir), "parent_manifest": parent_manifest,
                    "input_sha256": before, "inputs_unchanged": True, "split_integrity": integrity,
                    "source_ir_sha256": frozen_hashes, "frozen_manifest": str(frozen),
                    "parent_sha256": {name: before[str(parent / name)] for name in
                                      [split + ".jsonl" for split in SPLITS] + ["manifest.json"]},
                    "source_IR_rows": ir_counts, "source_families": SOURCES,
                    "auxiliary_families": list(AUXILIARY_FAMILIES), "evaluation_only_sources": sorted(EVALUATION_SOURCES),
                    "condition_projection": PROJECTION, "room_size_semantics": "reference_extent",
                    "room_dimension_mode": "xy", "yaw_label_semantics": YAW_SEMANTICS,
                    "spatiallm_geometric_yaw_IR_sha256": expected_spatiallm_sha256, "size_output_policy": policy,
                    "size_reference": policy["size_reference"], "size_log_limit": policy["size_log_limit"],
                    "task_role": "minimal_reference_extent_comparison_not_full_condition_main",
                    "condition_boundary_quality": "source_reference_extent",
                    "exclusions": {}, "admission_policy": "preserve every parent scene; separate whole-scene tokenizer/object/active-loss preflight",
                    "roomgenbench_holdout_groups": sorted(holdout_groups), "holdout_reason": HOLDOUT_REASON,
                    "degenerate_axis_m": DEGENERATE_AXIS_M,
                    "strict_rectangle_view": "metadata-only subset; not physical legality certification" if strict_rectangle else None,
                    "geometry_modified": "common structural-input XY and explicit input-floor reference translation only",
                    "source_data_modified": False, "parent_data_modified": False, **stats}
        manifest["output_sha256"] = {path.name: fingerprint(path) for path in sorted(stage.iterdir())}
        manifest["implementation_sha256"] = fingerprint(Path(__file__))
        with (stage / "manifest.json").open("x") as stream:
            _write(stream, manifest)
        target.mkdir(exist_ok=False)
        for path in stage.iterdir():
            path.rename(target / path.name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--ir-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-manifest", required=True, help="canonical sealed v3.2 manifest binding all 18 IR files")
    parser.add_argument("--model-config", help="JSON model config or train config containing model")
    parser.add_argument("--strict-rectangle", action="store_true", help="also emit metadata-qualified rectangle subsets")
    parser.add_argument("--holdout-group", action="append", dest="holdout_groups",
                        help="provenance.group written to test (repeatable); default: RoomGenBench benchmark rooms")
    args = parser.parse_args(argv)
    config = _loads(Path(args.model_config).read_text()) if args.model_config else None
    if config and "model" in config:
        config = config["model"]
    holdout = ROOMGENBENCH_HOLDOUT_GROUPS if args.holdout_groups is None else tuple(args.holdout_groups)
    manifest = build_dataset(args.parent_root, args.ir_root, args.output, model_config=config,
                             strict_rectangle=args.strict_rectangle, frozen_manifest=args.frozen_manifest, holdout_groups=holdout)
    print(json.dumps({"output": str(Path(args.output).resolve()), "split_samples": manifest["split_samples"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
