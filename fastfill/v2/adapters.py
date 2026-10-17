"""Fail-closed source adapters for the v2 geometry contract.

These read original annotations; they never call v1's geometric fallbacks.
MultiScan is the initially admitted source because it supplies annotated front
and up vectors, local OBB half extents, and independent structural geometry.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

SOURCE_DIRECTORY = "XXXpilar__multiscan-clean"
AXIS_TOLERANCE = 1e-5


def _array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def canonical_obb(center, extents, axes, front, up, *, half_extents=True, unit_scale=1.0):
    """Source OBB -> local full size, bottom center, semantic yaw.

    `axes` are rows containing world directions. Full extents are already
    instance-scaled; no asset scale is applied again. Significant pitch/roll,
    non-orthogonal frames, and unknown/vertical fronts are refused.
    """
    c = _array(center, (3,), "center")
    e = _array(extents, (3,), "positive extents")
    a = _array(axes, (3, 3), "axes")
    f = _array(front, (3,), "front")
    u = _array(up, (3,), "up")
    if not np.isfinite(unit_scale) or unit_scale <= 0 or not (e > 0).all():
        raise ValueError("extents and unit_scale must be strictly positive")
    if not np.allclose(a @ a.T, np.eye(3), atol=AXIS_TOLERANCE, rtol=0):
        raise ValueError("OBB axes must be orthonormal")
    if not np.allclose(u, [0, 0, 1], atol=AXIS_TOLERANCE, rtol=0):
        raise ValueError("only upright +Z-up objects are admitted; tilt needs another protocol")
    if abs(f[2]) > AXIS_TOLERANCE or not np.isclose(np.linalg.norm(f), 1, atol=AXIS_TOLERANCE):
        raise ValueError("annotated front must be a unit horizontal box axis")
    if not np.isclose(np.max(np.abs(a @ f)), 1, atol=AXIS_TOLERANCE):
        raise ValueError("annotated front must be aligned with an OBB axis")
    if not np.isclose(np.max(np.abs(a @ u)), 1, atol=AXIS_TOLERANCE):
        raise ValueError("upright up must be aligned with an OBB axis")
    basis = np.stack([f, np.cross(u, f), u], axis=1)
    full = e * (2 if half_extents else 1) * unit_scale
    size = full @ np.abs(a @ basis)
    bottom = c * unit_scale - np.array([0, 0, size[2] / 2])
    yaw = (math.atan2(f[1], f[0]) + math.pi) % (2 * math.pi) - math.pi
    return {"target_size_local_m": size.tolist(), "bottom_center_m": bottom.tolist(), "yaw_rad": yaw}


def _box(row, geometric_front=False):
    axes = np.asarray(json.loads(row["obb_axes"]), dtype=float).reshape(3, 3)
    front = json.loads(row["front"])
    up = json.loads(row["up"])
    if geometric_front:
        # Fixed obstacles need geometric coverage, not a semantic facing label.
        horizontal = [axis for axis in axes if abs(axis[2]) < AXIS_TOLERANCE]
        if not horizontal:
            raise ValueError("fixed geometry is tilted and cannot be silently projected")
        front, up = horizontal[0], [0, 0, 1]
    return canonical_obb(json.loads(row["obb_center"]), json.loads(row["obb_half_extents"]),
                         axes, front, up)


def _room(region):
    polygon = _array(json.loads(region["poly_loop"]), (len(json.loads(region["poly_loop"])), 2), "floor polygon")
    if len(polygon) < 3:
        raise ValueError("independent source floor footprint needs at least three vertices")
    floor = float(region["floor_height"])
    if not math.isfinite(floor):
        raise ValueError("source floor height must be finite")
    height = None
    if region.get("height_reliable") == "True":
        height = float(region["ceiling_height"]) - floor
        if not math.isfinite(height) or height <= 0:
            raise ValueError("reliable room height must be positive")
    return {"frame": "right_handed_z_up", "floor_polygon_xy_m": polygon.tolist(),
            "floor_z_m": floor, "height_m": height, "room_type": region.get("room_type") or None,
            "boundary_quality": "partial_scanned_floor_convex_hull", "boundary_known": False,
            "floor_known": False, "fixed_objects": []}


def multiscan_sample(region, rows, scan):
    """One scan, full requested inventory, with unsupported upright objects fixed.

    The footprint and floor slab top are structural inputs. The floor slab top
    is a coordinate reference, not verified physical contact. Target furniture
    cannot expand the room or estimate its floor. Tilted fixed or requested
    objects reject the scene rather than disappear or become yaw-only labels.
    """
    room = _room(region)
    requests, targets, fixed, source_ids = [], [], [], []
    for row in rows:
        category = row["category"].replace("_", " ").strip()
        structural = row.get("is_architectural") == "True" or row.get("is_opening") == "True"
        if structural and category in {"floor", "ceiling"}:
            continue
        front = np.asarray(json.loads(row["front"]), dtype=float)
        use_fixed = structural or abs(front[2]) > AXIS_TOLERANCE
        box = _box(row, geometric_front=use_fixed)
        if use_fixed:
            fixed.append({"id": f"fixed_{len(fixed):04d}", "category": category,
                          "size_local_m": box["target_size_local_m"],
                          "bottom_center_m": box["bottom_center_m"], "yaw_rad": box["yaw_rad"]})
            continue
        if not category:
            raise ValueError("requested category missing")
        object_id = f"object_{len(requests):04d}"
        requests.append({"id": object_id, "category": category, "description": f"a {category}"})
        targets.append({"id": object_id, **box})
        source_ids.append(row["object_id"])
    if not requests:
        raise ValueError("scene has no eligible requested objects")
    # Requests contain anonymous categories/descriptions only: no relation,
    # support, capability, or size role distinguishes repeated instances.
    # Groups are supervision bookkeeping (validity, target order), never condition text.
    category_count = Counter(request["category"] for request in requests)
    groups = [f"anonymous_{request['category']}" if category_count[request["category"]] > 1 else None
              for request in requests]
    n = len(requests)
    return {"schema_version": "fastfill.v2",
            "condition": {"schema_version": "fastfill.v2", "room": {**room, "fixed_objects": fixed},
                          "objects": requests, "constraints": []},
            "target": {"schema_version": "fastfill.v2", "objects": targets},
            "validity": {"position": [[True] * 3 for _ in range(n)], "size": [[True] * 3 for _ in range(n)],
                         "yaw": [True] * n, "yaw_symmetry_order": [1] * n, "exchangeable_group": groups},
            "provenance": {"source": "MultiScan", "scene_id": region["scan_id"],
                           "house_id": scan["scene_id"], "source_object_ids": source_ids,
                           "source_frame": "scan_local_Z_up_meters", "geometry_evidence":
                           "objects.csv: annotation OBB center/half-extents/axes/front/up; no additional scale",
                           "room_geometry_evidence": "regions.csv poly_loop and floor slab AABB top",
                           "semantic_front_evidence": "source annotated front vector",
                           "upright_axis_tolerance": AXIS_TOLERANCE,
                           "split": None, "relations_used_as_condition": False,
                           "mesh_ply": scan.get("mesh_ply"), "mesh_obj": scan.get("mesh_obj")}}


def iter_multiscan(root: Path, diagnostics: dict) -> Iterator[dict]:
    """Yield verified samples; report every rejected source scene."""
    base = Path(root) / SOURCE_DIRECTORY
    scans, grouped = {}, defaultdict(list)
    with (base / "scans.csv").open(newline="") as fh:
        scans = {row["scan_id"]: row for row in csv.DictReader(fh)}
    with (base / "objects.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            grouped[row["scan_id"]].append(row)
    with (base / "regions.csv").open(newline="") as fh:
        for region in csv.DictReader(fh):
            diagnostics["scenes_inspected"] += 1
            try:
                sample = multiscan_sample(region, grouped[region["scan_id"]], scans[region["scan_id"]])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                diagnostics["rejections"].append({"scene_id": region["scan_id"], "reason": str(exc)})
                continue
            yield sample
