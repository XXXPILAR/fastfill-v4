"""Derive a minimal rectangular-room bbox task from sealed SpatialLM geometry.

This builder qualifies geometric box axes, not asset semantic fronts. Parent
splits and labels are preserved; a recorded room-frame translation is the only
allowed numeric target transformation. Old datasets remain immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path

from shapely.geometry import Polygon

from .direct_layout import request_to_condition
from .io import fingerprint, safe_output
from .schema import validate_layout

SPATIALLM_IR_SHA256 = "f582b46d45b38e59b7b593d045f946f889d08942c10147bf4f077b7f33ef4410"
SPLITS = ("train", "validation", "test")
UNKNOWN_ROOM_TYPES = {"misc", "other", "other room", "unknown", "undefined", "none", ""}
YAW_SEMANTICS = "local_bbox_axes_not_certified_semantic_front"


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _room_frame(room):
    if not isinstance(room.get("room_type"), str) or room["room_type"].strip().lower() in UNKNOWN_ROOM_TYPES:
        raise ValueError("room_type_unknown")
    if room.get("boundary_known") is not True:
        raise ValueError("boundary_unknown")
    if room.get("floor_known") is not True or not _finite(room.get("floor_z_m")):
        raise ValueError("floor_unknown")
    points = room.get("floor_polygon_xy_m", [])
    if len(points) < 3 or not all(len(p) == 2 and all(_finite(v) for v in p) for p in points):
        raise ValueError("rectangle_invalid_polygon")
    polygon = Polygon(points)
    lo = [min(p[q] for p in points) for q in (0, 1)]
    hi = [max(p[q] for p in points) for q in (0, 1)]
    area = (hi[0] - lo[0]) * (hi[1] - lo[1])
    if not polygon.is_valid or area <= 0 or abs(polygon.area - area) > max(1e-8, area * 1e-6):
        raise ValueError("rectangle_not_axis_aligned")
    height = room.get("height_m")
    if height is not None and (not _finite(height) or height <= 0):
        raise ValueError("height_invalid")
    return [*lo, float(room["floor_z_m"])], [hi[q] - lo[q] for q in (0, 1)], height


def _source_room_metadata(sample, source_room):
    room, provenance = sample["condition"]["room"], sample["provenance"]
    if source_room.get("room_type") != room.get("room_type"):
        raise ValueError("source_room_type_mismatch")
    parent_height, source_height = room.get("height_m"), source_room.get("height")
    recorded_drop = parent_height is None and provenance.get("legacy_height_dropped") is True
    if parent_height != source_height and not recorded_drop:
        raise ValueError("source_room_height_mismatch")
    if source_height is not None and (not _finite(source_height) or source_height <= 0):
        raise ValueError("source_room_height_invalid")
    # Sealed SpatialLM IR uses source meters/Z-up and a canonical zero floor.
    # An explicit source floor supports separately audited test/version frames.
    source_floor = source_room.get("floor_z_m", 0.)
    if not _finite(source_floor) or source_floor != room.get("floor_z_m"):
        raise ValueError("source_floor_frame_mismatch")
    if source_room.get("boundary_type") != "polygon":
        raise ValueError("source_boundary_semantics_mismatch")
    if room.get("frame") != "right_handed_z_up":
        raise ValueError("source_floor_frame_axis_mismatch")
    return source_height


def _source_geometry(sample, source_room):
    provenance = sample["provenance"]
    if (source_room.get("source") != "SpatialLM" or source_room.get("uid") != provenance.get("scene_id")
            or source_room.get("group") != provenance.get("group")
            or source_room.get("group") != provenance.get("house_id")):
        raise ValueError("source_identity_mismatch")
    source_height = _source_room_metadata(sample, source_room)
    if source_room.get("boundary") != sample["condition"]["room"]["floor_polygon_xy_m"]:
        raise ValueError("source_boundary_mismatch")
    source_ids = provenance.get("target_source_ids", [])
    targets = sample["target"]["objects"]
    evidence = provenance.get("field_evidence", [])
    if len(source_ids) != len(targets) or len(set(source_ids)) != len(source_ids) or len(evidence) != len(targets):
        raise ValueError("source_identity_missing_or_duplicate")
    source_objects = {obj["id"]: obj for obj in source_room.get("objects", [])}
    for target, source_id, fields in zip(targets, source_ids, evidence):
        obj = source_objects.get(source_id)
        if obj is None:
            raise ValueError("source_identity_missing_object")
        if obj.get("tilted") is not False or fields.get("tilted") is not False:
            raise ValueError("tilted_object")
        if fields.get("size_semantics") != "canonical_source_IR":
            raise ValueError("source_geometry_proxy_size")
        if not _finite(obj.get("yaw")):
            raise ValueError("source_geometry_invalid_yaw")
        yaw = (obj["yaw"] + math.pi) % (2 * math.pi) - math.pi
        if (obj.get("pos") != target.get("bottom_center_m") or obj.get("size") != target.get("target_size_local_m")
                or yaw != target.get("yaw_rad")):
            raise ValueError("source_geometry_mismatch")
    return source_height


def _complete_geometry(sample):
    targets = sample["target"]["objects"]
    masks = sample.get("validity", {})
    for key in ("position", "size"):
        rows = masks.get(key, [])
        if len(rows) != len(targets) or not all(len(row) == 3 and all(flag is True for flag in row) for row in rows):
            raise ValueError("incomplete_geometry_labels")
    for target in targets:
        p, s, yaw = target.get("bottom_center_m", []), target.get("target_size_local_m", []), target.get("yaw_rad")
        if len(p) != 3 or len(s) != 3 or not all(_finite(v) for v in p + s) or min(s) <= 0 or not _finite(yaw):
            raise ValueError("invalid_geometry_labels")


def _inside_room(targets, origin, dimensions, height, tolerance):
    xmin, ymin, floor = origin
    xmax, ymax = xmin + dimensions[0], ymin + dimensions[1]
    for obj in targets:
        p, s, yaw = obj["bottom_center_m"], obj["target_size_local_m"], obj["yaw_rad"]
        for sx, sy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            x = p[0] + (sx * s[0] * math.cos(yaw) - sy * s[1] * math.sin(yaw)) / 2
            y = p[1] + (sx * s[0] * math.sin(yaw) + sy * s[1] * math.cos(yaw)) / 2
            if not xmin - tolerance <= x <= xmax + tolerance or not ymin - tolerance <= y <= ymax + tolerance:
                raise ValueError("xy_boundary_conflict")
        if p[2] < floor - tolerance:
            raise ValueError("floor_boundary_conflict")
        if height is not None and p[2] + s[2] > floor + height + tolerance:
            raise ValueError("ceiling_boundary_conflict")


def _validate_options(max_objects, tolerance_m, room_dimension_mode):
    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    if not _finite(tolerance_m) or tolerance_m < 0:
        raise ValueError("tolerance_m must be finite and nonnegative")
    if room_dimension_mode not in {"xy", "xyz"}:
        raise ValueError("room_dimension_mode must be xy or xyz")


def project_sample(sample, source_room, *, split, max_objects=128, tolerance_m=.001,
                   room_dimension_mode="xy"):
    """Reject uncertain source fields; use the exact public minimal input adapter."""
    _validate_options(max_objects, tolerance_m, room_dimension_mode)
    provenance = sample["provenance"]
    if provenance.get("source") != "SpatialLM":
        raise ValueError("source_not_qualified")
    if provenance.get("split") != split or split not in SPLITS:
        raise ValueError("source_split_mismatch")
    room = sample["condition"]["room"]
    origin, dimensions, height = _room_frame(room)
    requests, targets = sample["condition"]["objects"], sample["target"]["objects"]
    if not requests or len(requests) > max_objects:
        raise ValueError("max_objects_exceeded_or_empty")
    _complete_geometry(sample)
    qualification_height = _source_geometry(sample, source_room)
    _inside_room(targets, origin, dimensions, qualification_height, tolerance_m)
    inventory = [{"id": obj["id"], "category": obj["category"], "description": obj["category"], "count": 1}
                 for obj in requests]
    request_dimensions = dimensions + ([height] if room_dimension_mode == "xyz" and height is not None else [])
    request = {"room_type": room["room_type"], "room_size_m": request_dimensions,
               "furniture_list": inventory}
    condition = request_to_condition(request, max_objects=max_objects)
    translated = [{**deepcopy(obj), "bottom_center_m": [v - offset for v, offset in zip(obj["bottom_center_m"], origin)]}
                  for obj in targets]
    target = {"schema_version": "fastfill.v2", "objects": translated}
    validate_layout(target, condition)
    n = len(translated)
    return {"schema_version": "fastfill.v2", "condition": condition, "target": target,
            "validity": {"position": [[True] * 3 for _ in range(n)], "size": [[True] * 3 for _ in range(n)],
                         "yaw": [True] * n, "yaw_symmetry_order": [2] * n},
            "provenance": {**deepcopy(provenance), "minimal_request": request,
                           "condition_projection": "room_type_size_furniture_only-v1",
                           "room_dimension_mode": room_dimension_mode,
                           "qualification_source_height_m": qualification_height,
                           "frame_translation_m": [-value for value in origin],
                           "parent_yaw_validity": deepcopy(sample["validity"].get("yaw", [])),
                           "yaw_label_semantics": YAW_SEMANTICS,
                           "geometry_yaw_evidence": "sealed SpatialLM source IR: full local extents + yaw-only Bbox",
                           "correspondence": "fixed_request_identity",
                           "qualification_tolerance_m": tolerance_m,
                           "geometry_checks_scope": "OBB XY/floor/known ceiling only; collision/support/mesh/physics not certified"}}


def _parent_rows(parent):
    for split in SPLITS:
        with (parent / (split + ".jsonl")).open() as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    yield split, number, json.loads(line)


def _split_integrity(parent):
    uids, groups = set(), {}
    for split, number, row in _parent_rows(parent):
        provenance = row["provenance"]
        uid = provenance.get("scene_id")
        if not uid or uid in uids or provenance.get("split") != split:
            raise ValueError(f"parent UID/split integrity failure at {split}:{number}")
        uids.add(uid)
        for alias in (provenance.get("group"), provenance.get("house_id")):
            if not isinstance(alias, str) or not alias:
                raise ValueError("parent underlying group identity missing")
            if alias in groups and groups[alias] != split:
                raise ValueError("parent group alias has cross-split conflict")
            groups[alias] = split
    return {"parent_uids_checked": len(uids), "parent_group_aliases_checked": len(groups),
            "duplicate_uids": 0, "group_cross_split_conflicts": 0, "split_rule": "inherit parent without re-splitting"}


def _source_index(source):
    result = {}
    with source.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "SpatialLM" or row.get("uid") in result:
                raise ValueError("sealed SpatialLM source IR has invalid source or duplicate UID")
            result[row["uid"]] = row
    return result


def build_minimal_dataset(parent_root, source_ir, output, *, max_objects=128, tolerance_m=.001,
                          expected_source_sha256=SPATIALLM_IR_SHA256, room_dimension_mode="xy"):
    """Build a new, hash-pinned derivative. No source files are normalized in place."""
    _validate_options(max_objects, tolerance_m, room_dimension_mode)
    parent, source = Path(parent_root).resolve(), Path(source_ir).resolve()
    target = safe_output(output)
    if parent == target or parent in target.parents or target in parent.parents or source == target or target in source.parents:
        raise ValueError("output must be new and outside parent/source data and ancestors")
    inputs = [parent / (split + ".jsonl") for split in SPLITS] + [parent / "manifest.json", source]
    before = {str(path): fingerprint(path) for path in inputs}
    if before[str(source)] != expected_source_sha256:
        raise ValueError("source IR SHA256 does not match the explicitly audited version")
    integrity = _split_integrity(parent)
    source_rows = _source_index(source)
    counts, object_counts, rejected, inspected = Counter(), Counter(), Counter(), Counter()
    target.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        streams = {split: stack.enter_context((target / (split + ".jsonl")).open("x")) for split in SPLITS}
        exclusions = stack.enter_context((target / "exclusions.jsonl").open("x"))
        for split, number, row in _parent_rows(parent):
            inspected[split] += 1
            uid, source_name = row["provenance"]["scene_id"], row["provenance"]["source"]
            try:
                derived = project_sample(row, source_rows.get(uid, {}), split=split,
                                         max_objects=max_objects, tolerance_m=tolerance_m,
                                         room_dimension_mode=room_dimension_mode)
            except ValueError as error:
                reason = str(error)
                rejected[reason] += 1
                exclusions.write(json.dumps({"uid": uid, "split": split, "source": source_name,
                                             "parent_line": number, "reason": reason}, allow_nan=False) + "\n")
                continue
            streams[split].write(json.dumps(derived, separators=(",", ":"), allow_nan=False) + "\n")
            counts[split] += 1
            object_counts[split] += len(derived["target"]["objects"])
    if any(fingerprint(path) != before[str(path)] for path in inputs):
        raise RuntimeError("input changed during immutable dataset build; output is not a valid release")
    if not counts:
        raise ValueError("no qualified minimal-condition examples; output is not a valid release")
    manifest = {"schema_version": "fastfill.v2", "builder": "spatiallm-direct-bbox-v1",
                "parent_root": str(parent), "source_ir": str(source), "source_ir_sha256": before[str(source)],
                "parent_sha256": {path.name: before[str(path)] for path in inputs[:-1]},
                "samples_written": sum(counts.values()), "objects_written": sum(object_counts.values()),
                "split_samples": {s: counts[s] for s in SPLITS}, "split_objects": {s: object_counts[s] for s in SPLITS},
                "split_input_samples": dict(inspected), "exclusion_counts": dict(rejected), "split_integrity": integrity,
                "input_contract": "room_type + room_size_m" + ("[W,D]" if room_dimension_mode == "xy" else "[W,D,(H)]")
                + " + furniture_list; no assets/fixed/constraints/support",
                "room_dimension_mode": room_dimension_mode,
                "height_policy": "xy excludes source height from model input; any source height remains qualification evidence only; "
                "xyz explicitly opts into known exported room height; missing height stays unknown",
                "source_metadata_agreement": "room type, source/published height, canonical floor frame, polygon boundary "
                "semantics and house/group must agree with pinned IR; recorded height drops never bypass source ceiling checks",
                "yaw_label_semantics": YAW_SEMANTICS, "yaw_symmetry_order": 2,
                "correspondence": "fixed_request_identity", "max_objects": max_objects, "tolerance_m": tolerance_m,
                "qualification": "SpatialLM; known type/floor/boundary; valid polygon area=AABB within 1e-6; "
                "complete position/size; upright; finite source-matched geometric yaw; all OBB corners in XY, "
                "bottom above floor and top below any known ceiling within tolerance",
                "source_data_modified": False, "parent_data_modified": False,
                "target_changes": "recorded common translation only; local sizes and yaw unchanged; no snapping/scaling",
                "geometry_scope": "Boundary-qualified OBBs; collision/support completeness and mesh/physics not certified",
                "old_pilot_scope": "Earlier two-scene MultiScan pilot used richer conditions and did not train this contract",
                "implementation_sha256": fingerprint(Path(__file__)),
                "output_sha256": {path.name: fingerprint(path) for path in sorted(target.glob("*.jsonl"))}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--source-ir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--tolerance-m", type=float, default=.001)
    parser.add_argument("--room-dimensions", choices=("xy", "xyz"), default="xy")
    args = parser.parse_args()
    result = build_minimal_dataset(args.parent_root, args.source_ir, args.output,
                                   max_objects=args.max_objects, tolerance_m=args.tolerance_m,
                                   room_dimension_mode=args.room_dimensions)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
