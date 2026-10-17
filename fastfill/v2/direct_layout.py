"""Room type + room dimensions + furniture inventory -> a bbox handoff.

This boundary never resolves assets, certifies support, or commits a world. The
geometric outputs remain predictions even when their proxy diagnostics fail;
hand-off placement is either the request's declaration or an inferred geometric
candidate (``infer_support``), never a verified contact.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .batch import render_minimal_condition, wall_relative_opening
from .geometry import wrap_yaw
from .io import safe_output
from .schema import _id, _keys, validate_condition, validate_layout, vector


WALL_THICKNESS_M = .1
DEFAULT_WALL_HEIGHT_M = 2.7
FLOOR_CONTACT_M = .02
ON_OBJECT_CONTACT_M = .03
RAISED_M = .15  # a wall candidate is raised, as spread_grid_xy's raised objects


def asset_key(category, description):
    """RoomGenBench bench/prepare_inputs.py convention: slug(type)[:24] + sha1(description)[:8]."""
    slug = "".join(c if c.isalnum() else "_" for c in category.lower()).strip("_")[:24]
    return f"{slug}_{hashlib.sha1(description.encode()).hexdigest()[:8]}"


def _request_objects(entries, max_objects):
    """Expanded request objects; an optional ``support_parent`` ("floor", "wall" or another entry's id) is kept."""
    if not isinstance(entries, list) or not entries:
        raise ValueError("furniture_list must be a nonempty list")
    objects = []
    for entry in entries:
        item = {"category": entry} if isinstance(entry, str) else entry
        _keys(item, {"id", "category", "description", "count", "support_parent"}, {"category"}, "furniture")
        category = _id(item["category"], "furniture category")
        description = item.get("description", category)
        if not isinstance(description, str):
            raise ValueError("furniture description must be a string")
        count = item.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("furniture count must be a positive integer")
        if "id" in item and count != 1:
            raise ValueError("an explicit furniture ID requires count=1")
        if len(objects) + count > max_objects:
            raise ValueError("furniture inventory exceeds max_objects; do not truncate the request")
        support = {"support_parent": _id(item["support_parent"], "furniture support_parent")} if "support_parent" in item else {}
        expanded = [{"id": _id(item.get("id", f"obj_{len(objects) + index:04d}")),
                     "category": category, "description": description, **support} for index in range(count)]
        objects = objects + expanded
    ids = {obj["id"] for obj in objects}
    for obj in objects:  # cycles: validate_condition ("support graph contains a cycle")
        parent = obj.get("support_parent")
        if parent is not None and (parent == obj["id"] or parent not in ids | {"floor", "wall"}):
            raise ValueError(f"furniture {obj['id']!r}: support_parent {parent!r} must be "
                             "\"floor\", \"wall\" or the id of another furniture entry")
    return objects


def request_to_condition(request, *, max_objects=128, room_size_semantics="rectangular"):
    """Meters-only room adapter with an explicit interpretation of room size.

    room_size_m is [X length, Y length] or [X length, Y length, height].
    The room origin is its lower XY corner and its nominal floor is Z=0.
    Missing height stays unknown; no mesh, assets, doors or windows are assumed.
    rectangular declares a known rectangle and floor. reference_extent uses
    the same coordinates for normalization without declaring a physical room
    boundary or a measured floor. The profile is configuration, not a fourth
    request field. A furniture entry may declare ``support_parent`` ("floor", "wall"
    or another entry's id); it is copied to the condition object (the full-condition
    field the shared three-field projection drops), so entries without it are unchanged.
    """
    _keys(request, {"room_type", "room_size_m", "furniture_list"},
          {"room_type", "room_size_m", "furniture_list"}, "direct request")
    room_type = _id(request["room_type"], "room type")
    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    if room_size_semantics not in ("rectangular", "reference_extent"):
        raise ValueError("room_size_semantics must be rectangular or reference_extent")
    dimensions = request["room_size_m"]
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) not in (2, 3):
        raise ValueError("room_size_m must contain two or three full lengths in meters")
    dimensions = vector(dimensions, len(dimensions), "room size", positive=True)
    w, d = dimensions[:2]
    known_rectangle = room_size_semantics == "rectangular"
    room = {"frame": "right_handed_z_up", "room_type": room_type,
            "floor_polygon_xy_m": [[0., 0.], [w, 0.], [w, d], [0., d]], "floor_z_m": 0., "floor_known": known_rectangle,
            "height_m": dimensions[2] if len(dimensions) == 3 else None}
    # The rendered text is the shared three-field projection (batch.render_minimal_condition), byte for
    # byte what a rectangular training row becomes under minimal_form_p; on top come only the unrendered
    # quality tag and the reference_extent profile's unknown boundary.
    objects = _request_objects(request["furniture_list"], max_objects)
    condition = render_minimal_condition({"schema_version": "fastfill.v2", "room": room,
                 "objects": objects, "constraints": []})
    condition["room"].update(boundary_known=known_rectangle,
            boundary_quality="explicit_rectangular_request" if known_rectangle else "source_reference_extent")
    for rendered, obj in zip(condition["objects"], objects):
        if "support_parent" in obj:
            rendered["support_parent"] = obj["support_parent"]
    validate_condition(condition)
    return condition


def _corners(position, size, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    w, d, h = size
    xy = [(x * c - y * s + position[0], x * s + y * c + position[1])
          for x, y in [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]]
    return [[x, y, position[2] + dz] for dz in [0., h] for x, y in xy]


def layout_to_scene(condition, layout):
    """Attach request semantics and explicit OBB geometry without changing sizes."""
    validate_condition(condition)
    validate_layout(layout, condition)
    predictions = {obj["id"]: obj for obj in layout["objects"]}
    objects = []
    for request in condition["objects"]:
        prediction = deepcopy(predictions[request["id"]])
        size, position = prediction["target_size_local_m"], prediction["bottom_center_m"]
        yaw = prediction["yaw_rad"]
        bbox = {"center_m": [position[0], position[1], position[2] + size[2] / 2],
                "size_local_m": list(size), "yaw_rad": yaw,
                "corners_m": _corners(position, size, yaw)}
        objects = objects + [{**prediction, "yaw_rad": yaw, "category": request["category"],
                              "description": request["description"], "bbox": bbox}]
    return {"schema_version": "fastfill.bbox-scene.v1", "frame": "right_handed_z_up",
            "units": "meters", "geometry_kind": "predicted_local_obb_envelope",
            "room": deepcopy(condition["room"]), "objects": objects}


def _shell(room):
    """Walls along the floor polygon edges; doors/windows from fixed objects on the nearest wall.

    The world frame is shared with SAGE (Z-up, same XY), so polygon vertices are
    wall endpoints directly; only object-local axes differ. Opening width is the
    footprint's extent projected on the wall direction; windows add sill_height.
    """
    points, floor = room["floor_polygon_xy_m"], room.get("floor_z_m") or 0.
    height = room.get("height_m") or DEFAULT_WALL_HEIGHT_M
    indexed = [(i, a, b) for i, (a, b) in enumerate(zip(points, points[1:] + points[:1])) if math.dist(a, b) > 1e-9]
    edges = [(a, b) for _, a, b in indexed]
    walls = [{"id": f"wall_{index:02d}", "start_point": {"x": a[0], "y": a[1], "z": floor},
              "end_point": {"x": b[0], "y": b[1], "z": floor}, "height": height, "thickness": WALL_THICKNESS_M}
             for index, (a, b) in enumerate(edges)]
    openings = {"doors": [], "windows": []}
    # Wall-relative room.openings (wall_index = polygon edge i -> i+1) are already in the shell's own form.
    wall_of = {i: wall["id"] for (i, _, _), wall in zip(indexed, walls)}
    for k, o in enumerate(room.get("openings") or []):
        if not wall_relative_opening(o) or o["wall_index"] not in wall_of:
            raise ValueError("room openings must be wall-relative on a nonzero polygon edge")
        kind = "doors" if o["type"] == "door" else "windows"
        opening = {"id": o.get("id", f"opening_{k:02d}"), "wall_id": wall_of[o["wall_index"]],
                   "position_on_wall": o["position_on_wall"], "width": o["width_m"], "height": o["height_m"]}
        openings[kind].append(opening if kind == "doors" else {**opening, "sill_height": o.get("sill_height_m", 0.)})
    for fixed in room.get("fixed_objects", []):
        category = str(fixed.get("category", "")).lower()
        kind = "doors" if "door" in category else "windows" if "window" in category else None
        if kind is None:
            continue
        p = fixed["bottom_center_m"]
        best = None  # ponytail: nearest wall by point-to-segment distance, no cutoff for detached openings
        for wall, (a, b) in zip(walls, edges):
            length = math.dist(a, b)
            u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
            t = min(max(((p[0] - a[0]) * u[0] + (p[1] - a[1]) * u[1]) / length, 0.), 1.)
            distance = math.dist(p[:2], (a[0] + u[0] * t * length, a[1] + u[1] * t * length))
            if best is None or distance < best[0]:
                best = (distance, wall["id"], t, u)
        _, wall_id, t, u = best
        c, s = math.cos(fixed["yaw_rad"]), math.sin(fixed["yaw_rad"])
        sx, sy, sz = fixed["size_local_m"]
        width = abs(sx * (u[0] * c + u[1] * s)) + abs(sy * (u[1] * c - u[0] * s))
        opening = {"id": fixed["id"], "wall_id": wall_id, "position_on_wall": t, "width": width, "height": sz}
        openings[kind].append(opening if kind == "doors" else {**opening, "sill_height": p[2] - floor})
    return {"walls": walls, **openings}


def _footprint_contains(box, xy):
    """Is the XY point inside the box's rotated footprint (closed test)?"""
    c, s = math.cos(box["yaw_rad"]), math.sin(box["yaw_rad"])
    dx, dy = xy[0] - box["bottom_center_m"][0], xy[1] - box["bottom_center_m"][1]
    w, d, _ = box["target_size_local_m"]
    return abs(c * dx + s * dy) <= w / 2 and abs(-s * dx + c * dy) <= d / 2


def _reaches(parents, ident, target):
    """Does ident's support chain in ``parents`` (id -> parent, acyclic) lead to target?"""
    while ident in parents and ident != target:
        ident = parents[ident]
    return ident == target


def infer_support(obj, declared, boxes, floor, room=None, parents=None):
    """K6 hand-off placement: (support_parent, status), status in {declared, inferred, unknown}.

    A declared parent (request support_parent or hard on) is reported as is.
    Otherwise a bottom within FLOOR_CONTACT_M of the floor height is a floor
    candidate; else the highest other predicted box that starts strictly below
    this object, whose top is within ON_OBJECT_CONTACT_M of this bottom and
    whose footprint contains this footprint centre, is an on_object candidate
    (strictly lower parents keep inferred chains acyclic; a declared parent need
    not be lower, so a box whose chain in ``parents``, id -> declared or already
    inferred parent, leads back to this object is skipped: declared A on B with B
    predicted on A's top would otherwise give A -> B -> A); else, given a ``room``
    with a known boundary and a known floor, a box whose bottom is over RAISED_M above
    the floor, with a footprint side within ``validation.WALL_GAP_M`` of the boundary and
    no other box under its centre (footprint containing it, starting below this bottom:
    not a shelf's content, nothing it could stand on) is a wall candidate; anything else
    is unknown.
    """
    if declared is not None:
        return declared, "declared"
    z = obj["bottom_center_m"][2]
    if floor is not None and abs(z - floor) <= FLOOR_CONTACT_M:
        return "floor", "inferred"
    top = lambda box: box["bottom_center_m"][2] + box["target_size_local_m"][2]
    below = [box for box in boxes if box["bottom_center_m"][2] < z and abs(z - top(box)) <= ON_OBJECT_CONTACT_M
             and _footprint_contains(box, obj["bottom_center_m"][:2])
             and not _reaches(parents or {}, box["id"], obj["id"])]
    if below:
        return max(below, key=top)["id"], "inferred"
    from .validation import on_wall
    if (room is not None and floor is not None and z - floor > RAISED_M and on_wall(obj, room) and not any(
            box["bottom_center_m"][2] < z and _footprint_contains(box, obj["bottom_center_m"][:2]) for box in boxes)):
        return "wall", "inferred"
    return None, "unknown"


def place_of(parent):
    """RoomGenBench `place` vocabulary for a support parent (declared or inferred)."""
    if parent is None:
        return "unknown"
    return parent if parent in {"floor", "wall"} else "on_object"


def layout_to_roomgenbench(condition, layout):
    """SceneSpec adapter: SAGE local +Y axis, full sizes, degree yaw.

    Swapping the two local horizontal lengths and subtracting pi/2 preserves
    world corners and maps FastFill's canonical +X axis to SAGE's +Y axis.
    Geometric bbox-axis yaw does not certify an asset's semantic front. No
    asset IDs or verified support surfaces are fabricated. Explicit request
    support and hard on constraints are preserved (`support_status` declared);
    otherwise a layout object's predicted `support_parent` (support head) is used (`predicted`); otherwise
    `infer_support` proposes floor / on_object / wall candidates from the
    predicted boxes (inferred, never closing a support cycle with the declared or
    earlier inferred parents; floor only when `floor_known` is not false, wall
    only when `boundary_known` is not false) or
    leaves the object unknown. `place` is the
    RoomGenBench vocabulary of `place_id`. asset_key follows the benchmark
    convention (shared by identical type+description; per-instance dimensions
    stay on the scene object). The room shell is built by _shell.
    """
    scene = layout_to_scene(condition, layout)
    from .validation import effective_support_requests
    requests = {obj["id"]: obj for obj in effective_support_requests(condition)}
    room = scene["room"]
    points = room["floor_polygon_xy_m"]
    low = [min(p[q] for p in points) for q in (0, 1)]
    high = [max(p[q] for p in points) for q in (0, 1)]
    dimensions = {"width": high[0] - low[0], "length": high[1] - low[1], "height": room.get("height_m")}
    floor = room.get("floor_z_m")
    # A declared-unknown floor (reference_extent: floor_z_m 0, floor_known false) is no floor-contact evidence.
    contact_floor = None if room.get("floor_known") is False else floor
    # Declared support (a fixed object's hard on included), then each inference as made: none closes a cycle.
    parents = {c["object_id"]: c.get("parent_id", c.get("target_id")) for c in condition["constraints"]
               if c.get("type") == "on" and c.get("hard", True)}
    parents = {**parents, **{ident: row.get("support_parent") for ident, row in requests.items()}}
    predicted = {obj["id"]: obj["support_parent"] for obj in layout["objects"] if obj.get("support_parent") is not None}
    objects = []
    for obj in scene["objects"]:
        if requests[obj["id"]].get("support_parent") is None and obj["id"] in predicted:
            parent, status = predicted[obj["id"]], "predicted"
        else:
            parent, status = infer_support(obj, requests[obj["id"]].get("support_parent"), scene["objects"], contact_floor,
                                           room if room.get("boundary_known") is not False else None, parents)
        parents = {**parents, obj["id"]: parent}
        objects = objects + [{"id": obj["id"], "type": obj["category"], "description": obj["description"],
                "asset_key": asset_key(obj["category"], obj["description"]),
                "asset_key_kind": "downstream_generation_key_only",
                "position": dict(zip(("x", "y", "z"), obj["bottom_center_m"])),
                "rotation": {"x": 0., "y": 0., "z": math.degrees(wrap_yaw(obj["yaw_rad"] - math.pi / 2))},
                "dimensions": {"width": obj["target_size_local_m"][1],
                               "length": obj["target_size_local_m"][0],
                               "height": obj["target_size_local_m"][2]},
                "place_id": parent, "place": place_of(parent),
                "support_surface_id": requests[obj["id"]].get("support_surface_id"),
                "support_status": status}]
    scene_key = "fastfill_" + hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(",", ":"),
                                                      ensure_ascii=False).encode()).hexdigest()[:16]
    downstream = {"scene_key": scene_key, "room_type": room.get("room_type"),
            "geometry_only": True, "objects": objects,
            "fixed_objects": deepcopy(room.get("fixed_objects", [])),
            "constraints": deepcopy(condition["constraints"]),
            "room": {"dimensions": dimensions, "position": {"x": low[0], "y": low[1], "z": floor},
                     "ceiling_height": room.get("height_m"), **_shell(room)}}
    if room.get("boundary_quality") == "source_reference_extent":
        return {**downstream, "room_size_semantics": "reference_extent",
                "room_interpretation": "XY reference range and nominal Z origin; physical boundary and floor unknown",
                "room": {**downstream["room"], "boundary_known": room["boundary_known"],
                         "floor_known": room["floor_known"], "boundary_quality": room["boundary_quality"]}}
    return downstream


def bbox_diagnostics(scene, *, tolerance_m=1e-3):
    """Report proxy boundary/collision observations without requiring asset evidence."""
    from shapely.geometry import Polygon
    room, objects = scene["room"], scene["objects"]
    boundary = Polygon(room["floor_polygon_xy_m"])
    footprints = {obj["id"]: Polygon([p[:2] for p in obj["bbox"]["corners_m"][:4]]) for obj in objects}
    checks = []
    for obj in objects:
        ident, p, size = obj["id"], obj["bottom_center_m"], obj["target_size_local_m"]
        boundary_status = "pass" if boundary.buffer(tolerance_m).covers(footprints[ident]) else "fail"
        checks = checks + [{"code": "boundary", "object_ids": [ident],
                            "status": boundary_status if room.get("boundary_known", True) else "unknown"}]
        floor = room.get("floor_z_m")
        floor_status = "unknown" if floor is None or not room.get("floor_known", True) else (
            "pass" if p[2] >= floor - tolerance_m else "fail")
        checks = checks + [{"code": "floor_lower_bound", "object_ids": [ident], "status": floor_status}]
        height = room.get("height_m")
        ceiling_status = "unknown" if height is None or floor is None or not room.get("floor_known", True) else (
            "pass" if p[2] + size[2] <= floor + height + tolerance_m else "fail")
        checks = checks + [{"code": "ceiling", "object_ids": [ident], "status": ceiling_status}]
    for i, first in enumerate(objects):
        for second in objects[i + 1:]:
            low = max(first["bottom_center_m"][2], second["bottom_center_m"][2])
            high = min(first["bottom_center_m"][2] + first["target_size_local_m"][2],
                       second["bottom_center_m"][2] + second["target_size_local_m"][2])
            if high - low > tolerance_m and footprints[first["id"]].intersection(footprints[second["id"]]).area > 1e-8:
                checks = checks + [{"code": "obb_overlap", "object_ids": [first["id"], second["id"]],
                                    "status": "fail", "interpretation": "proxy overlap; support/contact semantics unknown"}]
    report = {"scope": "target_bbox_proxy", "tolerance_m": tolerance_m, "checks": checks,
            "counts": dict(Counter(c["status"] for c in checks)),
            "asset_retrieval": "not_attempted", "mesh": "not_checked", "support": "unknown",
            "physics": "not_checked", "commit": "not_attempted"}
    if room.get("boundary_quality") == "source_reference_extent":
        return {**report, "room_size_semantics": "reference_extent",
                "room_interpretation": "XY reference range and nominal Z origin; physical boundary and floor unknown"}
    return report


def write_bbox_glb(path, scene):
    from .bbox_visualization import glb_bytes
    target = safe_output(path)
    payload = glb_bytes(scene)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)


def export_handoff(directory, condition, layout):
    """Write a new handoff directory. Validation failures never repair predictions."""
    from .bbox_visualization import glb_bytes, preview_svg
    output = safe_output(directory)
    scene = layout_to_scene(condition, layout)
    downstream = layout_to_roomgenbench(condition, layout)
    registry = {}
    for obj in downstream["objects"]:
        # Benchmark convention: the first instance's dimensions and place are the generation input.
        entry = registry.get(obj["asset_key"]) or {"asset_key": obj["asset_key"], "type": obj["type"],
                    "description": obj["description"], "dimensions": deepcopy(obj["dimensions"]),
                    "place": obj["place"], "support_status": obj["support_status"],
                    "placement_eligible": obj["place_id"] is not None, "scenes": [downstream["scene_key"]],
                    "n_instances": 0, "asset_key_kind": "downstream_generation_key_only"}
        registry = {**registry, obj["asset_key"]: {**entry, "n_instances": entry["n_instances"] + 1}}
    payloads = {"condition.json": json.dumps(condition, indent=2, allow_nan=False) + "\n",
                "layout.json": json.dumps(layout, indent=2, allow_nan=False) + "\n",
                "scene.json": json.dumps(scene, indent=2, allow_nan=False) + "\n",
                "roomgenbench_scene.json": json.dumps(downstream, indent=2, allow_nan=False) + "\n",
                "assets.jsonl": "".join(json.dumps(registry[key], allow_nan=False) + "\n" for key in sorted(registry)),
                "diagnostics.json": json.dumps(bbox_diagnostics(scene), indent=2, allow_nan=False) + "\n",
                "preview.svg": preview_svg(scene), "layout.glb": glb_bytes(scene)}
    # Compute and validate everything before creating the output directory.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        with (output / name).open("xb" if isinstance(payload, bytes) else "x") as stream:
            stream.write(payload)
    return output
