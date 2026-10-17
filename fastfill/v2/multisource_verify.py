"""Independent, streaming verification of the multi-source minimal derivative.

No producer or adapter functions are imported. Expected conditions, numeric
translations and mask changes are recomputed from the immutable parent rows;
SpatialLM yaw qualification additionally uses the hash-pinned source IR.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path

SPLITS = ("train", "validation", "test")
UNKNOWN = {"", "misc", "other", "other room", "unknown", "undefined", "none"}
FULL_CONDITION_POLICY = "full-condition-mask-review-v1"
FULL_CONDITION_YAW_POLICY = "inherit parent yaw validity and yaw_symmetry_order; no geometric promotion"
ROOMGENBENCH_HOLDOUT_GROUPS = ("sage:layout_61ebde9f", "sage:layout_6b049b06", "sage:layout_fef15043",
                               "sage:layout_60ee2ae3", "sage:layout_46cdcce3")
HOLDOUT_REASON = "roomgenbench_benchmark_room"
DEGENERATE_AXIS_M = .003
WALL_GAP_M = .1  # a wall-declared target's footprint lies within this of the boundary (source gaps: <= 8 cm)
HEIGHT_TOLERANCE_M = .05
FLOOR_SNAP_M = .02
SEALED_IR_SHA256 = {
    "HSSD200.jsonl": "6af6bb3e622d5bdad1381e3f9229261b3ea86b27df7c7636b05fcf2c7b8fe74a",
    "IL3D_3dfront.jsonl": "7aa9309d776ecfea09b8e872dd8c21526b668f07dc8553d781ce0add840a852f",
    "IL3D_synthetic.jsonl": "2f021ea40220b96ea69a29b9bba42151866938832ff1bd807a6f660a29e70df2",
    "InteriorGS.jsonl": "612c3afb3b80ae2fbee01b8930d05e7cec89831538d88d0308d0dbafda5c7b73",
    "InternScenes_3rscan.jsonl": "bae023a964ec08795feb25b8a36ea6da0fc826ab44ccaa508da369d57a67a992",
    "InternScenes_arkit.jsonl": "f90f2ccc4870fdf8f5c11df34d21f78a63b9917885371c0bcf3b17391b563c29",
    "InternScenes_gen.jsonl": "75387e2527997ed051c71fad75c3e472a5efe7a128bae18dd3efe5d7ba4dd772",
    "InternScenes_mp3d.jsonl": "3685a34f5406a7dd3c93de3c4d2ff8f5d051a5ef5a4183038ece4d54e7b6a84b",
    "InternScenes_scannet.jsonl": "994b9edbba489599940323947dd42e5fba8da0157346e48940e45c5ab095aa3a",
    "MansionWorld.jsonl": "3b91f38162f1b2f19e6580d3437dc853b590802291bbd008727319f7274bd40c",
    "MultiScan.jsonl": "3fcd69d2a9363453cbeeb8b9aa6f681b1e7c3b622774ccad8d7d791c370124ca",
    "OptiScene_holodeck.jsonl": "6b50155b6c71838f8f8267c20e17cc67f49f37746ed6b92d950602d84fb31b2a",
    "SAGE-10k.jsonl": "8a1e03dddc5baa770752ae14e77298b0d3b42aab77113c119008d6ee978d06b6",
    "Scan2CAD.jsonl": "8dd7b1b766ee884b7ee7d3f8657c42a18fa7af4bd3671d87ab0380c6a18eea36",
    "SceneSmith.jsonl": "0d58aae78ff6061bf41f7d0abc2f508028940a087e10b0c04a779f5e8d92d432",
    "SpatialGen.jsonl": "5390fd84350a3d6070842bc3c49b75fa88031f645e4c06312c7c1fe57e4096d8",
    "SpatialLM.jsonl": "f582b46d45b38e59b7b593d045f946f889d08942c10147bf4f077b7f33ef4410",
    "Structured3D.jsonl": "c4b66c25ea6385de7a738311f8ee92866c12e23706e439389248bdfbaef386a4",
}
PARENT_SHA256 = {
    "train.jsonl": "ad885654907242e9cb7b222653c9da5f9a5237c5586215e7ced94ed74c830c92",
    "validation.jsonl": "5a0a45d6b2d82f8fcbf3a1f2d7f908ce9531e484efdac870b869dd9164a5b773",
    "test.jsonl": "efc393bd36f1110ca3e037d89ddc70c60e7b6cd676bdc1c47b2f7f81e9648ea9",
    "manifest.json": "7135a59f097e665ec6bb0973aae0233f323be447556e4ef903b868afbe779652",
    "rejections.jsonl": "02925d15b20a613807ada2f8741c851c24c1b036720feaa8b1c0fadc8317f95a",
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _equal(a, b):
    if type(a) is bool or type(b) is bool or a is None or b is None:
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _require(value, expected, label):
    if not _equal(value, expected):
        raise ValueError(label + " mismatch")


def _validate_parent(parent):
    targets, masks = parent["target"]["objects"], parent["validity"]
    n = len(targets)
    for field, label in (("position", "bottom_center_m"), ("size", "target_size_local_m")):
        if len(masks.get(field, [])) != n:
            raise ValueError("parent validity " + field + " length")
        for mask, obj in zip(masks[field], targets):
            values = obj.get(label, [])
            if len(mask) != 3 or len(values) != 3 or any(type(flag) is not bool for flag in mask):
                raise ValueError("parent validity " + field + " shape/type")
            if any(value is not None and not _finite(value) for value in values):
                raise ValueError("parent target " + field + " nonfinite/type")
            if any(flag and (not _finite(value) or field == "size" and value <= 0) for flag, value in zip(mask, values)):
                raise ValueError("parent validity " + field + " invalid active target")
    if len(masks.get("yaw", [])) != n or any(type(flag) is not bool for flag in masks["yaw"]):
        raise ValueError("parent validity yaw length/type")
    for flag, obj in zip(masks["yaw"], targets):
        value = obj.get("yaw_rad")
        if value is not None and not _finite(value) or flag and not _finite(value):
            raise ValueError("parent validity yaw invalid target")
    if "yaw_symmetry_order" in masks and (len(masks["yaw_symmetry_order"]) != n
            or any(type(v) is not int or v < 1 for v in masks["yaw_symmetry_order"])):
        raise ValueError("parent validity yaw_symmetry_order length/type")
    if "exchangeable_group" in masks and (len(masks["exchangeable_group"]) != n
            or any(v is not None and (type(v) is not str or not v) for v in masks["exchangeable_group"])):
        raise ValueError("parent validity exchangeable_group length/type")
    if "size_axis_swap_allowed" in masks and (len(masks["size_axis_swap_allowed"]) != n
            or any(type(v) is not bool for v in masks["size_axis_swap_allowed"])):
        raise ValueError("parent validity size_axis_swap_allowed length/type")
    _require([o["id"] for o in parent["condition"]["objects"]], [o["id"] for o in targets], "parent request ID order")


def _migrated(parent):
    """Pre-C1 parents carried exchangeable_group inside condition objects; read it as validity.
    Pre-K1 parents lack size_axis_swap_allowed; read it as all false."""
    objects = parent["condition"]["objects"]
    if "size_axis_swap_allowed" not in parent["validity"]:
        parent = {**parent, "validity": {**parent["validity"], "size_axis_swap_allowed": [False] * len(parent["target"]["objects"])}}
    if not any("exchangeable_group" in o for o in objects):
        return parent
    if "exchangeable_group" in parent["validity"]:
        raise ValueError("parent declares exchangeable_group in both condition objects and validity")
    groups = {o["id"]: o.get("exchangeable_group") for o in objects}
    return {**parent, "condition": {**parent["condition"], "objects": [
                {k: v for k, v in o.items() if k != "exchangeable_group"} for o in objects]},
            "validity": {**parent["validity"], "exchangeable_group": [groups.get(t["id"]) for t in parent["target"]["objects"]]}}


def _holdout(provenance, split, holdout_groups):
    if split == "test" or provenance.get("group") not in holdout_groups:
        return provenance, None
    return ({**provenance, "split": "test", "holdout_reason": HOLDOUT_REASON},
            {"field": "provenance.split", "before": split, "after": "test", "reason": HOLDOUT_REASON})


def _qualified_masks(parent, size_reference, size_log_limit):
    """Recompute D1/D2 and degenerate-axis mask demotions with their journal entries, in protocol order."""
    p, masks, changes = parent["provenance"], deepcopy(parent["validity"]), []
    for i, obj in enumerate(parent["target"]["objects"]):
        base = {"object_id": obj["id"], "target_source_id": p["target_source_ids"][i]}
        if p["source"] == "Scan2CAD" and any(masks["position"][i]):
            changes.append({**base, "field": "position", "before": masks["position"][i], "after": [False] * 3,
                            "reason": "Scan2CAD_estimated_floor_and_upstream_snap_uncertainty"})
            masks["position"][i] = [False] * 3
        size = obj["target_size_local_m"]
        for reason, degenerate in (("outside_configured_size_head_range", any(
                    flag and not ref * math.exp(-size_log_limit) <= v <= ref * math.exp(size_log_limit)
                    for flag, v, ref in zip(masks["size"][i], size, size_reference))),
                ("degenerate_axis_lt_3mm", any(_finite(v) and v < DEGENERATE_AXIS_M for v in size))):
            if any(masks["size"][i]) and degenerate:
                changes.append({**base, "field": "size", "before": masks["size"][i], "after": [False] * 3,
                                "value_m": size, "reason": reason})
                masks["size"][i] = [False] * 3
    return masks, changes


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairs(items):
    result = dict(items)
    if len(result) != len(items):
        raise ValueError("duplicate JSON key")
    return result


def _loads(text):
    def invalid(value):
        raise ValueError("nonfinite JSON literal: " + value)
    def number(value):
        result = float(value)
        return result if math.isfinite(result) else invalid(value)
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid, parse_float=number)


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield _loads(line)


def _expected_condition(parent):
    room = parent["condition"]["room"]
    points = room["floor_polygon_xy_m"]
    if not points or any(len(p) != 2 or not all(_finite(v) for v in p) for p in points):
        raise ValueError("parent reference extent invalid")
    low = [min(p[q] for p in points) for q in (0, 1)]
    dimensions = [max(p[q] for p in points) - low[q] for q in (0, 1)]
    if min(dimensions) <= 0:
        raise ValueError("parent reference extent degenerate")
    room_type = room.get("room_type")
    room_type = "unknown" if not isinstance(room_type, str) or room_type.strip().lower() in UNKNOWN else room_type
    objects = [{"id": obj["id"], "category": obj["category"], "description": obj["description"]}
               for obj in parent["condition"]["objects"]]
    w, d = dimensions
    condition = {"schema_version": "fastfill.v2", "room": {
        "frame": "right_handed_z_up", "room_type": room_type, "floor_z_m": 0.,
        "floor_known": False, "boundary_known": False, "boundary_quality": "source_reference_extent",
        "height_m": None, "floor_polygon_xy_m": [[0., 0.], [w, 0.], [w, d], [0., d]]},
        "objects": objects, "constraints": []}
    request = {"room_type": room_type, "room_size_m": dimensions,
               "furniture_list": [{**obj, "count": 1} for obj in objects]}
    floor = room.get("floor_z_m")
    return condition, request, [*low, floor if _finite(floor) else 0.]


def _check_source(parent, source_room):
    p, room = parent["provenance"], parent["condition"]["room"]
    if source_room is None:
        raise ValueError("source_geometry missing source room")
    for key, value in (("uid", p["scene_id"]), ("source", p["source"]), ("group", p["group"]),
                       ("boundary", room["floor_polygon_xy_m"]), ("room_type", room.get("room_type"))):
        _require(source_room.get(key), value, "source_geometry " + key)
    _require(source_room["group"], p["house_id"], "source_geometry house_id")
    dz = 0.
    if p["source"] == "MultiScan":
        meta = source_room["meta"]
        if meta.get("n_floor_objects_outside_structure", 0) or not all(_finite(meta.get(k)) for k in ("floor_z", "floor_height")):
            raise ValueError("source_geometry MultiScan floor reference missing/target expanded")
        dz = float(meta["floor_z"]) - float(meta["floor_height"])
    _require(p.get("vertical_reframe_m", 0.), dz, "source_geometry vertical_reframe_m")
    objects = source_room["objects"]
    index = {o["id"]: o for o in objects}
    if len(index) != len(objects):
        raise ValueError("source_geometry duplicate source ID")
    requests = {o["id"]: o for o in parent["condition"]["objects"]}
    for obj, source_id, fields in zip(parent["target"]["objects"], p["target_source_ids"], p["field_evidence"]):
        raw = index.get(source_id)
        if raw is None:
            raise ValueError("source_geometry source ID absent")
        pos, yaw = raw["pos"], raw.get("yaw")
        expected = [*pos[:2], pos[2] + dz if pos[2] is not None else None]
        if fields.get("legacy_z_snap_applied_to_target") is True:
            # Bridge K3: a declared floor anchor within FLOOR_SNAP_M of the known floor has its target z snapped.
            floor = room.get("floor_z_m")
            if (requests[obj["id"]].get("support_parent") != "floor" or not _finite(floor) or not _finite(expected[2])
                    or abs(expected[2] - floor) > FLOOR_SNAP_M):
                raise ValueError("source_geometry floor snap without declaration or beyond tolerance")
            expected[2] = floor
        _require(expected, obj["bottom_center_m"], "source_geometry position")
        _require(raw["size"], obj["target_size_local_m"], "source_geometry size")
        _require((yaw + math.pi) % (2 * math.pi) - math.pi if _finite(yaw) else None, obj["yaw_rad"], "source_geometry yaw")
        _require(bool(raw.get("tilted")), fields["tilted"], "source_geometry tilted")


def _spatiallm_qualification(parent, source_room):
    p = parent["provenance"]
    if source_room is None:
        raise ValueError("source_geometry missing SpatialLM room")
    for key, expected in (("uid", p["scene_id"]), ("source", "SpatialLM"), ("group", p["group"])):
        _require(source_room.get(key), expected, "source_geometry " + key)
    source_objects = source_room["objects"]
    index = {o["id"]: o for o in source_objects}
    if len(index) != len(source_objects):
        raise ValueError("source_geometry duplicate source ID")
    qualified = []
    for i, obj in enumerate(parent["target"]["objects"]):
        raw = index.get(p["target_source_ids"][i])
        if raw is None:
            raise ValueError("source_geometry source ID absent")
        fields = p["field_evidence"][i]
        eligible = (raw.get("tilted") is False and fields.get("tilted") is False
                    and fields.get("size_semantics") == "canonical_source_IR" and _finite(raw.get("yaw"))
                    and all(_finite(v) and v > 0 for v in raw["size"]) and all(_finite(v) for v in raw["pos"]))
        if eligible:
            yaw = (raw["yaw"] + math.pi) % (2 * math.pi) - math.pi
            _require(raw.get("pos"), obj.get("bottom_center_m"), "source_geometry position")
            _require(raw.get("size"), obj.get("target_size_local_m"), "source_geometry size")
            _require(yaw, obj.get("yaw_rad"), "source_geometry yaw")
        qualified.append(eligible)
    return qualified


def verify_pair(parent, derived, split, source_room=None, *, size_reference=(1., 1., 1.), size_log_limit=10.,
                holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS):
    """Fail loudly for one row; preserve invalid/null numbers and all identities."""
    _validate_parent(parent)
    parent = _migrated(parent)
    provenance, actual = parent["provenance"], derived["provenance"]
    _require(parent["schema_version"], "fastfill.v2", "parent schema")
    _require(derived["schema_version"], "fastfill.v2", "derived schema")
    _require(provenance["split"], split, "parent split")
    expected_provenance, moved = _holdout(provenance, split, holdout_groups)
    for key, value in expected_provenance.items():
        _require(actual.get(key), value, "provenance " + key)
    if provenance["source"] in {"SceneSmith", "SpatialGen"} and split != "test":
        raise ValueError("evaluation_only source in training")
    condition, request, origin = _expected_condition(parent)
    _require(derived["condition"], condition, "condition")
    _require(actual.get("minimal_request"), request, "minimal_request")
    _require(actual.get("parent_condition"), parent["condition"], "parent_condition")
    _require(actual.get("parent_validity"), parent["validity"], "parent_validity")
    _require(actual.get("frame_translation_m"), [-v for v in origin], "frame_translation_m")
    expected_target = deepcopy(parent["target"])
    for obj in expected_target["objects"]:
        obj["bottom_center_m"] = [value - offset if _finite(value) else value
                                  for value, offset in zip(obj["bottom_center_m"], origin)]
    _require(derived["target"], expected_target, "target")
    n = len(expected_target["objects"])
    if len(provenance["target_source_ids"]) != n or len(set(provenance["target_source_ids"])) != n:
        raise ValueError("target_source_ids missing or duplicate")
    _require([o["id"] for o in condition["objects"]], [o["id"] for o in expected_target["objects"]], "request ID order")
    if len(provenance.get("field_evidence", [])) != n:
        raise ValueError("source_geometry evidence length mismatch")
    if source_room is not None:
        _check_source(parent, source_room)
    masks, changes = _qualified_masks(parent, size_reference, size_log_limit)
    qualifications = _spatiallm_qualification(parent, source_room) if provenance["source"] == "SpatialLM" else [False] * n
    # The recorded protocol qualifies all D1/D2 masks before geometric yaw; both passes keep their exact order.
    for i, obj in enumerate(parent["target"]["objects"]):
        if qualifications[i] and not masks["yaw"][i]:
            masks["yaw"][i] = True
            changes.append({"object_id": obj["id"], "target_source_id": provenance["target_source_ids"][i],
                "field": "yaw", "before": False, "after": True,
                "reason": "pinned_SpatialLM_full_local_extents_and_geometric_yaw"})
    masks["yaw_symmetry_order"] = [4 if swap else 2 for swap in masks["size_axis_swap_allowed"]]
    _require(derived["validity"], masks, "validity")
    _require(actual.get("geometric_yaw_qualified"), qualifications, "geometric_yaw_qualified")
    floor = parent["condition"]["room"].get("floor_z_m")
    estimated = provenance["source"] == "Scan2CAD"
    _require(actual.get("source_floor_evidence"), {
        "floor_source": "estimated" if estimated else "unknown" if floor is None else "source_canonical_reference",
        "parent_floor_known": parent["condition"]["room"].get("floor_known"), "parent_floor_z_m": floor,
        "upstream_z_snap_possible": estimated,
        "per_object_pre_snap_z": "unavailable_in_frozen_IR" if estimated else "not_inferred"}, "source_floor_evidence")
    if any(origin):
        changes.append({"field": "bottom_center_m", "operation": "common_input_reference_translation",
            "offset_m": [-v for v in origin], "reason": "structural_input_extent_and_explicit_floor_reference"})
    changes += [moved] if moved else []
    _require(actual.get("qualification_changes"), changes, "qualification_changes")
    for key, expected in (("condition_projection", "room_type_reference_extent_furniture_only-v1"),
            ("room_dimension_mode", "xy"), ("room_size_semantics", "reference_extent"),
            ("yaw_label_semantics", "local_bbox_axes_mod_pi_not_certified_semantic_front"),
            ("correspondence", "fixed_request_identity")):
        _require(actual.get(key), expected, key)
    return {"objects": n, "target_object_fields_checked": sum(len(o) for o in expected_target["objects"]),
            "target_numeric_coordinates_checked": 7 * n,
            "position_masks_demoted": sum(all(a) and not all(b) for a, b in zip(parent["validity"]["position"], masks["position"])),
            "position_mask_rows_changed": sum(a != b for a, b in zip(parent["validity"]["position"], masks["position"])),
            "size_masks_demoted": sum(all(a) and not all(b) for a, b in zip(parent["validity"]["size"], masks["size"])),
            "size_mask_rows_changed": sum(a != b for a, b in zip(parent["validity"]["size"], masks["size"])),
            "yaw_masks_promoted": sum(not a and b for a, b in zip(parent["validity"]["yaw"], masks["yaw"])),
            "valid_position_objects": sum(all(x) for x in masks["position"]),
            "valid_size_objects": sum(all(x) for x in masks["size"]), "valid_yaw_objects": sum(masks["yaw"]),
            "removed_fixed_objects": len(parent["condition"]["room"].get("fixed_objects", [])),
            "removed_constraints": len(parent["condition"].get("constraints", []))}


def _source_offsets(ir, names):
    index, counts = {}, Counter()
    for name in names:
        with (ir / name).open("rb") as stream:
            while True:
                offset, line = stream.tell(), stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = _loads(line)
                if row.get("source") + ".jsonl" != name or row["uid"] in index:
                    raise ValueError("source IR identity/duplicate UID")
                index[row["uid"]] = (name, offset)
                counts[name[:-6]] += 1
    return index, dict(counts)


def _source_row(ir, index, uid):
    if uid not in index:
        raise ValueError("source_geometry parent UID absent from pinned IR")
    name, offset = index[uid]
    with (ir / name).open("rb") as stream:
        stream.seek(offset)
        return _loads(stream.readline())


def _boundary_gap(box, polygon):
    """Distance between a target box's footprint outline and the floor polygon's outline (vertex-to-edge both ways)."""
    (x, y, _), (w, d, _), yaw = box["bottom_center_m"], box["target_size_local_m"], box["yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((w / 2, d / 2), (-w / 2, d / 2), (-w / 2, -d / 2), (w / 2, -d / 2))]

    def to_edges(point, ring):
        best = math.inf
        for a, b in zip(ring, ring[1:] + ring[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            t = min(max(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy or 1.), 0.), 1.)
            best = min(best, math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy))
        return best
    ring = [tuple(p) for p in polygon]
    return min(min(to_edges(p, ring) for p in corners), min(to_edges(p, corners) for p in ring))


def verify_full_pair(parent, derived, split, *, size_reference=(1., 1., 1.), size_log_limit=10.,
                     holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS):
    """Verify the primary full-condition view; only mask/room qualification and the benchmark holdout may change."""
    _validate_parent(parent)
    parent = _migrated(parent)
    p, actual = parent["provenance"], derived["provenance"]
    _require(parent["schema_version"], "fastfill.v2", "parent schema")
    _require(derived["schema_version"], "fastfill.v2", "derived schema")
    _require(p["split"], split, "parent split")
    expected_provenance, moved = _holdout(p, split, holdout_groups)
    for key, value in expected_provenance.items():
        if key != "height_conflict":  # recomputed on the qualified masks below
            _require(actual.get(key), value, "provenance " + key)
    if p["source"] in {"SceneSmith", "SpatialGen"} and split != "test":
        raise ValueError("evaluation_only source in training")
    condition = deepcopy(parent["condition"])
    targets = parent["target"]["objects"]
    n = len(targets)
    if len(p["target_source_ids"]) != n or len(set(p["target_source_ids"])) != n:
        raise ValueError("target_source_ids missing or duplicate")
    known = condition["room"].get("boundary_known") is True
    walls = [obj.get("support_parent") == "wall" for obj in condition["objects"]]
    for wall, fields, target in zip(walls, p.get("field_evidence", [{}] * n), targets):  # round 10 bridge wall rule
        # The bridge declares wall only where a footprint side touches the boundary (fields["wall_contact"], checked
        # exactly by legacy_verify); here: declared <=> touching wall anchor, and a declared one is near the boundary.
        if wall != (known and fields.get("selection_rule") == "wall_anchor" and fields.get("raw_anchor") == "wall"
                    and fields.get("wall_contact") is True):
            raise ValueError("wall support declared without a selected source wall anchor in a known boundary")
        if fields.get("selection_rule", "frozen_prep") == "frozen_prep" and fields.get("raw_anchor") in ("wall", "ceiling"):
            raise ValueError("frozen_prep target with a source wall/ceiling anchor (the frozen prep keeps floor/object)")
        if wall and _boundary_gap(target, condition["room"]["floor_polygon_xy_m"]) > WALL_GAP_M:
            raise ValueError("wall-declared target footprint lies over WALL_GAP_M from the room boundary")
    masks, changes = _qualified_masks(parent, size_reference, size_log_limit)
    if p["source"] == "Scan2CAD":
        condition["room"]["floor_known"] = False
        changes.insert(0, {"field": "room.floor_known", "before": parent["condition"]["room"].get("floor_known"),
            "after": False, "reason": "Scan2CAD_floor_is_estimated_not_independent_physical_measurement"})
    height, conflict, before = condition["room"].get("height_m"), None, p.get("height_conflict")
    tops = {obj["id"]: obj["bottom_center_m"][2] + obj["target_size_local_m"][2]
            for i, obj in enumerate(targets) if all(masks["size"][i]) and all(masks["position"][i])}
    over = [] if height is None else [ident for ident, top in tops.items() if top > height + HEIGHT_TOLERANCE_M + 1e-6]
    if over:
        conflict = {"objects": over, "max_excess_m": max(tops[ident] for ident in over) - height}
    if not _equal(conflict, before):  # a parent (bridge) flag is replaced/removed when the qualified masks differ
        changes.append({"field": "provenance.height_conflict", "before": before, "after": conflict,
                        "reason": "target_exceeds_declared_height_flag_only" if before is None else "height_conflict_follows_qualified_masks"})
    groups = masks.get("exchangeable_group", [])
    bad_groups = {g for i, g in enumerate(groups) if g is not None and not all(masks["position"][i])}
    members_removed = 0
    for i, group in enumerate(groups):
        if group in bad_groups:
            masks["exchangeable_group"][i] = None
            changes.append({"object_id": targets[i]["id"], "target_source_id": p["target_source_ids"][i],
                "field": "exchangeable_group", "before": group, "after": None,
                "reason": "mask_review_removed_complete_position_exchangeability"})
            members_removed += 1
    changes += [moved] if moved else []
    _require(derived["condition"], condition, "condition")
    _require(derived["target"], parent["target"], "target")
    _require(derived["validity"], masks, "validity")
    _require(actual.get("dataset_qualification_policy"), FULL_CONDITION_POLICY, "dataset_qualification_policy")
    _require(actual.get("qualification_changes"), changes, "qualification_changes")
    added = {"dataset_qualification_policy", "qualification_changes"} | ({"holdout_reason"} if moved else set())
    _require(actual.get("height_conflict"), conflict, "height_conflict")
    if conflict is not None:
        added.add("height_conflict")
    if p["source"] == "Scan2CAD":
        added.add("estimated_floor_provenance")
        meta = p.get("source_meta", {})
        _require(actual.get("estimated_floor_provenance"), {
            "floor_source": "estimated", "parent_floor_known": parent["condition"]["room"].get("floor_known"),
            "parent_floor_z_m": parent["condition"]["room"].get("floor_z_m"),
            "source_meta_floor_z": meta.get("floor_z"), "source_meta_n_floor_snapped": meta.get("n_floor_snapped"),
            "per_object_pre_snap_z": "unavailable_in_frozen_IR",
            "per_object_snap_membership": "unknown_do_not_infer_from_zero_z"}, "estimated_floor_provenance")
    _require(sorted(actual), sorted(set(p) - {"height_conflict"} | added), "provenance keys")
    return {"objects": n, "target_object_fields_checked": sum(len(o) for o in targets),
        "target_numeric_coordinates_checked": 7 * n,
        "position_masks_demoted": sum(all(a) and not all(b) for a, b in zip(parent["validity"]["position"], masks["position"])),
        "position_mask_rows_changed": sum(a != b for a, b in zip(parent["validity"]["position"], masks["position"])),
        "size_masks_demoted": sum(all(a) and not all(b) for a, b in zip(parent["validity"]["size"], masks["size"])),
        "size_mask_rows_changed": sum(a != b for a, b in zip(parent["validity"]["size"], masks["size"])),
        "exchangeable_groups_demoted": len(bad_groups), "exchangeable_members_demoted": members_removed,
        "yaw_masks_promoted": 0, "valid_position_objects": sum(all(x) for x in masks["position"]),
        "valid_size_objects": sum(all(x) for x in masks["size"]), "valid_yaw_objects": sum(masks["yaw"]),
        "wall_declared_objects": sum(walls), "removed_fixed_objects": 0, "removed_constraints": 0}


def _output_path(output, roots):
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError("verification output already exists: " + str(target))
    for root in (*roots, Path("/Volumes/harddisk/3D_Room_Collections")):
        root = root.resolve()
        if target == root or root in target.parents or target in root.parents:
            raise ValueError("verification output must stay outside source/data roots and ancestors")
    return target


def _row_pairs(parent, data, holdout_groups):
    """Pair parent rows with derived rows per split; holdout rows from train/validation pair with the tail of derived test."""
    derived = {split: _rows(data / (split + ".jsonl")) for split in SPLITS}
    deferred, line = [], 0
    for split in SPLITS:
        line = 0
        for line, old in enumerate(_rows(parent / (split + ".jsonl")), 1):
            if split != "test" and old["provenance"].get("group") in holdout_groups:
                deferred.append((split, line, old))
            else:
                yield split, line, split, old, next(derived[split], None)
        if split != "test":
            for new in derived[split]:
                yield split, line + 1, split, None, new
    for split, number, old in deferred:
        yield split, number, "test", old, next(derived["test"], None)
    for new in derived["test"]:
        yield "test", line + 1, "test", None, new


def parent_pins(parent_manifest):
    """Parent pins in PARENT_SHA256 shape, from a reviewed parent's manifest.json (its output hashes plus its own hash)."""
    path = Path(parent_manifest)
    return {**_loads(path.read_text())["output_sha256"], "manifest.json": _sha256(path)}


def verify_dataset(parent_root, data_root, ir_root, *, output=None, expected_ir_sha256=None, full_condition=False,
                   holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS, expected_parent_sha256=None):
    """Verify every inherited row and file hash, returning a machine-readable report."""
    parent, data, ir = [Path(p).resolve() for p in (parent_root, data_root, ir_root)]
    destination = _output_path(output, (parent, data, ir)) if output is not None else None
    seals = SEALED_IR_SHA256 if expected_ir_sha256 is None else expected_ir_sha256
    manifest = _loads((data / "manifest.json").read_text())
    if full_condition:
        _require(manifest.get("dataset_qualification_policy"), FULL_CONDITION_POLICY, "manifest full-condition policy")
        if expected_ir_sha256 is None:
            _require(manifest.get("task_role"), "full_condition_multisource_main", "manifest task_role")
            _require(manifest.get("yaw_policy"), FULL_CONDITION_YAW_POLICY, "manifest yaw_policy")
    if expected_ir_sha256 is None or "roomgenbench_holdout_groups" in manifest:
        _require(manifest.get("roomgenbench_holdout_groups"), sorted(holdout_groups), "manifest roomgenbench_holdout_groups")
    paths = [parent / (s + ".jsonl") for s in SPLITS] + [parent / "manifest.json", data / "manifest.json"]
    if expected_ir_sha256 is None or (parent / "rejections.jsonl").exists():
        paths.append(parent / "rejections.jsonl")
    paths += [data / name for name in manifest["output_sha256"]] + [ir / name for name in seals]
    paths += [Path(name) for name in manifest.get("input_sha256", {}) if Path(name) not in paths]
    implementation = manifest.get("implementation_sha256", {})
    implementation = implementation if isinstance(implementation, dict) else {str(Path(__file__).with_name("multisource_data.py")): implementation}
    paths += [Path(name) for name in implementation if Path(name) not in paths]
    paths.append(Path(__file__).resolve())
    before = {str(path): _sha256(path) for path in paths}
    errors, examples, counts, source_counts = Counter(), [], Counter(), {}
    split_counts = {split: Counter() for split in SPLITS}

    def fail(kind, where, detail):
        errors[kind] += 1
        if len(examples) < 100:
            examples.append({"kind": kind, "where": where, "detail": str(detail)})

    for kind, root, hashes in (("ir_hash", ir, seals), ("output_hash", data, manifest["output_sha256"])):
        for name, expected in hashes.items():
            if before[str(root / name)] != expected:
                fail(kind, name, "SHA256 differs")
    if expected_ir_sha256 is None:
        pins = PARENT_SHA256 if expected_parent_sha256 is None else expected_parent_sha256
        for name, expected in pins.items():
            if before[str(parent / name)] != expected:
                fail("parent_frozen_hash", name, "parent differs from the pinned reviewed parent")
    for name, expected in manifest.get("input_sha256", {}).items():
        if before[name] != expected:
            fail("input_hash", name, "declared input hash differs")
    for name, expected in implementation.items():
        if before[name] != expected:
            fail("implementation_hash", name, "producer code differs from recorded release")
    if full_condition and (data / "rejections.jsonl").exists() and before[str(data / "rejections.jsonl")] != before[str(parent / "rejections.jsonl")]:
        fail("parent_rejections", "rejections.jsonl", "historical exclusions were rewritten")
    recorded_paths = paths[:4] if full_condition else [*paths[:4], *[ir / name for name in seals]]
    for path in recorded_paths:
        recorded = manifest.get("input_sha256", {}).get(str(path))
        if recorded != before[str(path)]:
            fail("input_hash", str(path), "manifest input hash differs/missing")
    source_index, source_rows = ({}, {}) if full_condition else _source_offsets(ir, seals)
    if not full_condition and manifest.get("source_IR_rows") != source_rows:
        fail("manifest_ir_rows", "manifest", "IR row counts differ")
    policy = manifest.get("size_output_policy", {})
    reference, limit = policy.get("size_reference", (1., 1., 1.)), policy.get("size_log_limit", 10.)
    identities = {"parent": set(), "derived": set()}
    groups = {"parent": {}, "derived": {}}
    summaries = {name: Counter() for name in ("qualification_change_counts", "constraint_counts",
                 "fixed_objects_by_split", "supervised_yaw_scenes_by_split", "source_yaw_symmetry_order_counts",
                 "source_size_axis_swap_allowed_objects", "holdout_samples_by_parent_split")}
    journal = _rows(data / "changes.jsonl") if "changes.jsonl" in manifest["output_sha256"] else None
    journal_checked = 0
    if journal is None and expected_ir_sha256 is None:
        fail("missing_journal", "changes.jsonl", "required per-record audit trail missing")
    for parent_split, number, split, old, new in _row_pairs(parent, data, holdout_groups):
        where = f"{parent_split}:{number}"
        if old is None or new is None:
            fail("missing_row", where, "parent/derived row counts differ")
            continue
        for label, row, row_split in (("parent", old, parent_split), ("derived", new, split)):
            p = row["provenance"]
            uid = p.get("scene_id")
            if not uid or uid in identities[label]:
                fail("duplicate_uid", where, label + ": " + str(uid))
            identities[label].add(uid)
            for group in {p.get("house_id"), p.get("group")}:
                if not group or group in groups[label] and groups[label][group] != row_split:
                    fail("group_cross_split", where, label + ": " + str(group))
                groups[label][group] = row_split
        try:
            checked = verify_full_pair(old, new, parent_split, size_reference=reference, size_log_limit=limit, holdout_groups=holdout_groups) if full_condition else (
                verify_pair(old, new, parent_split, _source_row(ir, source_index, old["provenance"]["scene_id"]),
                            size_reference=reference, size_log_limit=limit, holdout_groups=holdout_groups))
            split_counts[split].update(checked)
            split_counts[split]["samples"] += 1
            key = old["provenance"]["source"] + ":" + split
            source_counts[key] = source_counts.get(key, Counter()) + Counter({**checked, "samples": 1})
            if full_condition:
                summaries["fixed_objects_by_split"][split] += len(new["condition"]["room"].get("fixed_objects", []))
                summaries["supervised_yaw_scenes_by_split"][split] += any(new["validity"]["yaw"])
                summaries["holdout_samples_by_parent_split"][parent_split] += split != parent_split
                for order in new["validity"].get("yaw_symmetry_order", []):
                    summaries["source_yaw_symmetry_order_counts"][old["provenance"]["source"] + ":" + str(order)] += 1
                summaries["source_size_axis_swap_allowed_objects"][old["provenance"]["source"]] += sum(
                    new["validity"].get("size_axis_swap_allowed", []))
                for constraint in new["condition"].get("constraints", []):
                    summaries["constraint_counts"][constraint["type"] + ":" + split] += 1
                for change in new["provenance"]["qualification_changes"]:
                    summaries["qualification_change_counts"][change["field"]] += 1
            if journal is not None:
                for change in new["provenance"]["qualification_changes"]:
                    _require(next(journal, None), {"uid": old["provenance"]["scene_id"],
                        "source": old["provenance"]["source"], "split": parent_split, "parent_line": number, **change}, "changes journal")
                    journal_checked += 1
        except (ValueError, KeyError, TypeError, IndexError) as error:
            fail("row_mismatch", where, error)
    split_counts = {split: dict(stats) for split, stats in split_counts.items()}
    for stats in split_counts.values():
        counts.update(stats)
    if journal is not None and next(journal, None) is not None:
        fail("changes_journal", "changes.jsonl", "unexpected additional entries")
    unchanged = all(_sha256(path) == before[str(path)] for path in paths)
    if not unchanged:
        fail("input_changed", "files", "inputs changed during verification")
    for key, field in (("split_samples", "samples"), ("split_objects", "objects")):
        expected = {s: v[field] for s, v in split_counts.items() if v.get(field)}
        if manifest.get(key) != expected:
            fail("manifest_counts", key, "recomputed counts differ")
    if full_condition:
        expected_stats = {key: dict(value) for key, value in summaries.items()}
        expected_stats.update({"source_split_samples": {s: v.get("samples", 0) for s, v in source_counts.items()},
            "source_split_objects": {s: v.get("objects", 0) for s, v in source_counts.items()},
            "validity_counts": {f: counts.get("valid_" + f + "_objects", 0) for f in ("position", "size", "yaw")},
            "source_split_validity": {s + ":" + f: v.get("valid_" + f + "_objects", 0)
                for s, v in source_counts.items() for f in ("position", "size", "yaw")}})
        for key, expected in expected_stats.items():
            if (expected_ir_sha256 is None or key in manifest) and manifest.get(key) != expected:
                fail("manifest_counts", key, "independently recomputed total differs")
    report = {"ok": not errors, "verifier": "independent-multisource-full-condition-v1" if full_condition else "independent-multisource-reference-extent-v1",
        "counts": dict(counts), "split_counts": split_counts,
        "source_split_counts": {k: dict(v) for k, v in sorted(source_counts.items())},
        "error_counts": dict(errors), "errors_first_100": examples, "files_sha256": before,
        "input_hashes_unchanged": unchanged, "sealed_ir_files_checked": len(seals), "journal_entries_checked": journal_checked,
        "roomgenbench_holdout_groups": sorted(holdout_groups),
        "scope": "all parent-derived identities, conditions, target numbers, mask/change policy and file/code hashes; no mesh/physics certification"}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-root", "data-root", "ir-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--full-condition", action="store_true", help="verify primary full-condition D1/D2 qualification only")
    parser.add_argument("--holdout-group", action="append", dest="holdout_groups",
                        help="provenance.group expected in test (repeatable); default: RoomGenBench benchmark rooms")
    parser.add_argument("--parent-manifest", help="reviewed parent manifest.json whose output hashes replace PARENT_SHA256 pins")
    args = parser.parse_args(argv)
    holdout = ROOMGENBENCH_HOLDOUT_GROUPS if args.holdout_groups is None else tuple(args.holdout_groups)
    pins = parent_pins(args.parent_manifest) if args.parent_manifest else None
    report = verify_dataset(args.parent_root, args.data_root, args.ir_root, output=args.output,
                            full_condition=args.full_condition, holdout_groups=holdout, expected_parent_sha256=pins)
    print(json.dumps({"ok": report["ok"], "counts": report["counts"], "error_counts": report["error_counts"], "output": args.output}))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
