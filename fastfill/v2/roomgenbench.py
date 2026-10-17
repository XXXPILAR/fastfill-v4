"""Dynamic RoomGenBench assembly with explicit paths and truthful mesh/proxy receipts.

Only the upstream geometry helpers are imported. Its fixed five-scene CLI,
global paths, reference GT branch and source files are never changed. Asset
generation is a separate step; this adapter consumes generated GLB/sidecars.
``--requests-from`` instead writes the benchmark scenes as FastFill requests (``benchmark_request``).
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import sys
from pathlib import Path
import re

from .direct_layout import layout_to_roomgenbench
from .io import fingerprint, safe_output
from .schema import finite


FIT_POLICY = "roomgenbench_anisotropic_yaw_snap_tip"


def _read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _reference(root):
    source = Path(root).resolve() / "bench" / "assemble.py"
    if not source.is_file():
        raise FileNotFoundError(f"RoomGenBench assembler not found: {source}")
    spec = importlib.util.spec_from_file_location("_fastfill_roomgenbench_geometry", source)
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # never write into the downstream checkout
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module, {"source": str(source), "sha256": fingerprint(source)}


def _handoff(path, require_placement):
    directory = Path(path).resolve()
    condition, layout = (_read_json(directory / name) for name in ("condition.json", "layout.json"))
    scene = layout_to_roomgenbench(condition, layout)
    if _read_json(directory / "roomgenbench_scene.json") != scene:
        raise ValueError("RoomGenBench handoff must match the canonical condition and layout")
    if require_placement and any(o["place_id"] is None for o in scene["objects"]):
        raise ValueError("this downstream policy requires explicit placement for every requested object")
    return condition, scene


def _extents(parts):
    import numpy as np
    vertices = np.vstack([part.vertices for part in parts])
    if not np.isfinite(vertices).all():
        raise ValueError("asset vertices must be finite")
    lengths = vertices.max(0) - vertices.min(0)
    if (lengths <= 0).any():
        raise ValueError("asset must have positive three-dimensional extents")
    return lengths.tolist()


def _asset_parts(module, obj, method, assets_dir):
    """Missing/failed/invalid assets retain a requested instance as a placeholder."""
    key = obj["asset_key"]
    sidecar, glb = assets_dir / f"{key}.json", assets_dir / f"{key}.glb"
    if not sidecar.is_file():
        return [], {"status": "missing", "reason": "sidecar_missing"}
    try:
        record = _read_json(sidecar)
        if (not isinstance(record, dict) or record.get("asset_key") != key
                or record.get("method") != method or not isinstance(record.get("status"), str)
                or record["status"] not in {"ok", "fallback", "failed"}):
            raise ValueError("sidecar requires matching asset_key/method and explicit ok/fallback/failed status")
    except (OSError, ValueError) as error:
        return [], {"status": "invalid_sidecar", "reason": str(error)}
    status = module.effective_status(method, sidecar)
    if status == "failed":
        return [], {"status": "failed", "reason": "method_reported_failure"}
    if not glb.is_file():
        return [], {"status": "missing", "reason": "glb_missing"}
    try:
        parts = module.load_glb_parts(glb)
        native = _extents(parts)
        sage_parts = [part.copy() for part in parts]
        for part in sage_parts:
            part.apply_transform(module.GLTF_TO_SAGE)
        fit_log = {"status": status, "native_size_gltf_m": native,
                   "native_size_sage_local_m": _extents(sage_parts)}
        fitted = module.fit_asset(parts, obj["dimensions"], fit_log)
        extents = _extents(fitted)
        target = [obj["dimensions"][k] for k in ("width", "length", "height")]
        # Upstream clamps thin native axes before scaling. Report a resulting
        # target mismatch rather than accepting the method's original ok status.
        if any(abs(a - b) > max(1e-9, 1e-5 * b) for a, b in zip(extents, target)):
            fit_log = {**fit_log, "status": "fit_mismatch", "source_status": status}
        return fitted, {**fit_log, "fitted_size_sage_local_m": extents,
                        "source_glb_sha256": fingerprint(glb), "source_sidecar_sha256": fingerprint(sidecar)}
    except Exception as error:
        return [], {"status": "failed", "reason": f"asset_load_or_fit_failed: {type(error).__name__}: {error}"}


def _box_parts(module, dimensions):
    import trimesh
    lengths = [dimensions[k] for k in ("width", "length", "height")]
    transform = trimesh.transformations.translation_matrix([0, 0, lengths[2] / 2])
    return [module.colored_box(lengths, transform, module.PLACEHOLDER_RGBA)]


def _objects(module, scene, method, assets_dir, output):
    logs = []
    for index, obj in enumerate(scene["objects"]):
        parts, fit_log = ([], {"status": "box"}) if method == "layout_boxes" else _asset_parts(module, obj, method, assets_dir)
        kind = "generated_fitted_mesh" if fit_log["status"] == "ok" else (
            "fitted_mesh_with_size_mismatch" if fit_log["status"] == "fit_mismatch" else
            "fallback_mesh" if parts else "target_bbox_proxy" if method == "layout_boxes" else "placeholder_bbox")
        parts = parts or _box_parts(module, obj["dimensions"])
        node = f"obj_{index:03d}"
        for part_index, part in enumerate(parts):
            name = node if len(parts) == 1 else f"{node}_p{part_index}"
            output.add_geometry(part, node_name=name, geom_name=name,
                                transform=module.SAGE_WORLD_TO_GLTF @ module.place_matrix(obj))
        logs = logs + [{"id": obj["id"], "asset_key": obj["asset_key"], "type": obj["type"],
                       "place": obj["place"], "support_parent": obj["place_id"],
                       "support_status": obj["support_status"], "geometry_kind": kind,
                       "node": node, **fit_log}]
    return logs


def _shell_parts(module, room):
    """Reference build_shell geometry with wall normals from the polygon winding, not the bbox centre.

    build_shell extrudes each wall away from the room's bounding-box centre,
    which pushes some walls of a non-convex polygon into the room. Walls trace
    the floor polygon in order, so its signed area fixes the interior side;
    each wall is built alone with a stand-in centre placed on that side.
    build_shell names every door panel of a wall ``<wall>_door``; the second and later
    panels of one wall get a running suffix (``_1``, ``_2``, ...) so no scene node is overwritten.
    """
    parts, walls = module.build_shell({**room, "walls": [], "doors": []})  # floor slab only
    points = [(w["start_point"]["x"], w["start_point"]["y"]) for w in room["walls"]]
    ccw = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])) > 0
    for index, wall in enumerate(room["walls"]):
        a, b = wall["start_point"], wall["end_point"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        inward = (-dy, dx) if ccw else (dy, -dx)
        cx, cy = (a["x"] + b["x"]) / 2 + inward[0], (a["y"] + b["y"]) / 2 + inward[1]
        single = {"dimensions": {"width": 2., "length": 2.}, "position": {"x": cx - 1., "y": cy - 1.},
                  "walls": [wall], "doors": [d for d in room["doors"] if d["wall_id"] == wall["id"]]}
        wall_parts, meta = module.build_shell(single)
        seen = Counter()
        for name, part in wall_parts:
            if name != "shell_floor":
                name = name.replace("shell_wall_0", f"shell_wall_{index}", 1)
                seen[name] += 1
                parts.append((name if seen[name] == 1 else f"{name}_{seen[name] - 1}", part))
        walls += [{**meta[0], "name": f"shell_wall_{index}"}]
    return parts, walls


def _display_geometry(module, condition, scene, output, display_height):
    """Reference shell (polygon walls, door cutouts) plus fixed bbox proxies; the condition is retained.

    Unknown room height renders walls at the display reference height. The
    reference shell puts its floor at Z=0, so it is translated to the floor.
    Doors are part of the shell; other fixed objects become bbox proxies.
    """
    import trimesh
    room, floor = condition["room"], condition["room"].get("floor_z_m")
    height = room.get("height_m")
    display_floor = 0. if floor is None else floor
    downstream = scene["room"] if height is not None else {**scene["room"], "walls": [
        {**wall, "height": display_height} for wall in scene["room"]["walls"]]}
    shell, walls = _shell_parts(module, downstream)
    lift = module.SAGE_WORLD_TO_GLTF @ trimesh.transformations.translation_matrix([0., 0., display_floor])
    for name, part in shell:
        output.add_geometry(part, node_name=name, geom_name=name, transform=lift)
    doors = {door["id"] for door in downstream["doors"]}
    proxies = 0
    for index, fixed in enumerate(room.get("fixed_objects", [])):
        if fixed["id"] in doors:
            continue
        w, d, h = fixed["size_local_m"]
        # Same +X -> +Y conversion as the requested-object adapter.
        obj = {"position": dict(zip("xyz", fixed["bottom_center_m"])),
               "rotation": {"x": 0., "y": 0., "z": 180 * fixed["yaw_rad"] / 3.141592653589793 - 90},
               "dimensions": {"width": d, "length": w, "height": h}}
        node = f"fixed_bbox_{index:03d}"
        output.add_geometry(_box_parts(module, obj["dimensions"])[0], node_name=node, geom_name=node,
                            transform=module.SAGE_WORLD_TO_GLTF @ module.place_matrix(obj))
        proxies += 1
    return walls, proxies, {**downstream["dimensions"],
            "height": display_height if height is None else height,
            "height_source": "display_reference" if height is None else "condition",
            "source_height_m": height, "floor_known": room.get("floor_known", floor is not None),
            "boundary_known": room.get("boundary_known", True),
            "shell_geometry_kind": "reference_build_shell_polygon_walls", "wall_normal_source": "floor_polygon_winding",
            "display_floor_z_m": display_floor, "source_floor_z_m": floor}


def assemble_handoff(handoff, output_dir, *, method="layout_boxes", assets_dir=None,
                     roomgenbench_root=None, require_placement=False, display_height_m=3.):
    """Consume an arbitrary FastFill handoff and optionally generated mesh assets.

    The returned receipt always uses all requested objects as its denominator.
    `generated_mesh_success` means every requested GLB loaded/fitted with status
    ok; it does not assert native dimensions, room feasibility or deployment.
    """
    if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", method) or method == "sage_gt":
        raise ValueError("method must be a short snake_case generated method or layout_boxes; sage_gt is unavailable")
    if method != "layout_boxes" and assets_dir is None:
        raise ValueError("generated mesh assembly requires --assets-dir")
    height = finite(display_height_m, "display reference height", positive=True)
    target = safe_output(output_dir)
    inputs = [Path(handoff).resolve()] + ([Path(assets_dir).resolve()] if assets_dir is not None else [])
    if any(target == source or target in source.parents or source in target.parents for source in inputs):
        raise ValueError("output directory must not overlap handoff or asset inputs")
    condition, scene = _handoff(handoff, require_placement)
    reference = Path(roomgenbench_root or Path(__file__).resolve().parents[2] / "RoomGenBench").resolve()
    if target == reference or target in reference.parents or reference in target.parents:
        raise ValueError("output directory must remain independent of the reference repository")
    module, source_info = _reference(reference)
    import trimesh
    output = trimesh.Scene()
    walls, proxies, room = _display_geometry(module, condition, scene, output, height)
    objects = _objects(module, scene, method, Path(assets_dir).resolve() if assets_dir is not None else None, output)
    counts = dict(Counter(obj["status"] for obj in objects))
    generated_success = method != "layout_boxes" and bool(objects) and counts.get("ok", 0) == len(objects)
    receipt = {"schema_version": "fastfill.roomgenbench-assembly.v1", "scene_key": scene["scene_key"],
        "scene": scene["scene_key"], "room_type": scene["room_type"], "walls": walls,
        "method": method, "assembly_complete": True, "generated_mesh_success": generated_success,
        "requested_objects": len(objects), "counts": counts, "objects": objects, "room": room,
        "fixed_bbox_proxies": proxies,
        "condition": deepcopy(condition), "fit_policy": FIT_POLICY if method != "layout_boxes" else "target_bbox_proxy",
        "original_asset_geometry_acceptance": "not_checked", "support_verification": "not_checked",
        "validator": "not_attempted", "physics": "not_attempted", "host_commit": "not_attempted",
        "reference_assembler": source_info}
    # Build/serialize everything before creating the immutable output directory.
    payload = output.export(file_type="glb")
    metadata = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    target.mkdir(parents=True, exist_ok=False)
    (target / f'{scene["scene_key"]}.glb').write_bytes(payload)
    (target / f'{scene["scene_key"]}.json').write_text(metadata)
    (target / "receipt.json").write_text(metadata)
    return receipt


def benchmark_request(scene):
    """FastFill request for a RoomGenBench scene (bench/inputs/scenes/*.json): every object, wall ones included.

    Entries are named ``obj_%04d`` in scene order, the training id style (``obj_0007`` is
    ``scene["objects"][7]``): the scene's long ids put the restaurant at 9,362 Qwen3 condition tokens,
    over the 8,192 training ``max_length`` (6,727 with these). Each declares its ``place_id`` ("floor",
    "wall" or the supporter, renamed alike) as ``support_parent``; room_size_m is [width, length, height].
    """
    dims = scene["room"]["dimensions"]
    ids = {o["id"]: f"obj_{index:04d}" for index, o in enumerate(scene["objects"])}
    return {"room_type": scene["room_type"], "room_size_m": [dims["width"], dims["length"], dims["height"]],
            "furniture_list": [{"id": ids[o["id"]], "category": o["type"], "description": o["description"],
                                "support_parent": ids.get(o["place_id"], o["place_id"])} for o in scene["objects"]]}


def write_benchmark_requests(scenes_dir, output_dir, *, roomgenbench_root=None):
    """Write ``<scene_key>.json`` per scene into a new directory outside the reference checkout;
    returns per-scene counts (requested, floor, wall, on_object)."""
    from .direct_layout import request_to_condition
    target = safe_output(output_dir)
    reference = Path(roomgenbench_root or Path(__file__).resolve().parents[2] / "RoomGenBench").resolve()
    if target == reference or reference in target.parents:
        raise ValueError("request directory must remain outside the reference repository")
    scenes = sorted(Path(scenes_dir).glob("*.json"))
    if not scenes:
        raise ValueError(f"no RoomGenBench scene JSON in {scenes_dir}")
    payloads, summary = {}, []
    for path in scenes:
        scene = _read_json(path)
        request = benchmark_request(scene)
        request_to_condition(request)  # schema check only, with predict --request's default max_objects
        places = Counter(o["support_parent"] if o["support_parent"] in ("floor", "wall") else "on_object"
                         for o in request["furniture_list"])
        payloads[f'{scene["scene_key"]}.json'] = json.dumps(request, indent=2) + "\n"
        summary.append({"scene_key": scene["scene_key"], "requested": len(request["furniture_list"]),
                        **{place: places[place] for place in ("floor", "wall", "on_object")}})
    if len(payloads) != len(scenes):
        raise ValueError("RoomGenBench scene keys must be unique")
    target.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (target / name).write_text(payload)
    return summary


def ground_truth_layout(condition, scene):
    """FastFill layout of a benchmark scene: the exact inverse of ``layout_to_roomgenbench``.

    Request ``obj_%04d`` is ``scene["objects"][i]`` (``benchmark_request``); size = [length, width,
    height], bottom centre = position, yaw = radians(rotation.z) + pi/2 wrapped; rotation.x / .y
    (benchmark tilts) have no FastFill counterpart and are dropped. Misaligned ids, categories or
    counts raise ValueError.
    """
    import math
    from .geometry import wrap_yaw
    from .schema import SCHEMA_VERSION, validate_layout
    requested, truth = condition["objects"], scene["objects"]
    if len(requested) != len(truth):
        raise ValueError(f"request has {len(requested)} objects, benchmark scene {len(truth)}")
    objects = []
    for index, (request, obj) in enumerate(zip(requested, truth)):
        if request["id"] != f"obj_{index:04d}" or request["category"] != obj["type"]:
            raise ValueError(f"request object {index} ({request['id']}, {request['category']}) is not "
                             f"benchmark object {index} (obj_{index:04d}, {obj['type']})")
        d = obj["dimensions"]
        objects.append({"id": request["id"], "target_size_local_m": [d["length"], d["width"], d["height"]],
                        "bottom_center_m": [obj["position"][k] for k in "xyz"],
                        "yaw_rad": wrap_yaw(math.radians(obj["rotation"]["z"]) + math.pi / 2)})
    return validate_layout({"schema_version": SCHEMA_VERSION, "objects": objects}, condition)


def reference_check(handoff_dir, scene_path):
    """Same-rule comparison of a hand-off with its benchmark scene's ground truth.

    validate_scene(hand-off condition) runs on both layouts (per check code: pass / violation /
    unknown). Exchangeable requests (evaluate's groups, ``matching.group_labels``: identical request
    fields but the id, support parent included, and a certified swap) are matched: within each group
    predictions are assigned to truth objects by a Hungarian assignment on bottom-centre distance
    (``predicted_id`` names the one matched to each truth row). Per-object FastFill-vs-truth errors
    are pooled over all objects and per ground-truth place: bottom-centre distance and, primary, the
    box-equivalent (size, yaw) of evaluate's ``_box_equivalent_errors`` (one joint minimum over yaw +
    k pi/2 with sx/sy swapped for odd k: the same box scores 0), here for every object (evaluate uses it
    only for ``size_axis_swap_allowed`` ones); ``*_marginal_min`` minimise separately (mean |log size
    ratio| over the (sx, sy) swap, yaw modulo pi), and ``*_by_id`` score the prediction of the same id,
    without matching. ``box_equivalent_*`` repeat the primary values under their schema-v1 names.
    Baselines: room centre (``room_center_position_error_m``, floor-level centre of the room's XY bounds)
    and uniform yaw on the same box-equivalent footing: ``uniform_yaw_baseline_error_rad`` is the exact
    expected box-equivalent yaw error of a uniformly random yaw with the truth's own size (per object and
    pooled), ``uniform_yaw_baseline_error_rad_marginal_min`` the pi / 4 of ``yaw_error_rad_marginal_min``.
    """
    import math
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from .evaluate import _box_equivalent_errors, _log_size_error, _uniform_yaw_error, _yaw_error
    from .direct_layout import place_of
    from .matching import group_labels
    from .schema import normalize_room
    from .validation import CHECK_STATUSES, validate_scene
    condition, _ = _handoff(handoff_dir, False)
    layout, scene = _read_json(Path(handoff_dir) / "layout.json"), _read_json(scene_path)
    truth = ground_truth_layout(condition, scene)
    checks = {}
    for name, candidate in (("fastfill", layout), ("ground_truth", truth)):
        report = validate_scene(condition, candidate["objects"])
        by_code = {}
        for check in report["checks"]:
            by_code.setdefault(check["code"], dict.fromkeys(CHECK_STATUSES, 0))[check["status"]] += 1
        checks[name] = {"ok": report["ok"], "counts": report["counts"], "by_code": by_code}
    origin, scale = normalize_room(condition["room"])
    center = [origin[0] + scale[0] / 2, origin[1] + scale[1] / 2, origin[2]]
    predicted, truth_by_id = {o["id"]: o for o in layout["objects"]}, {o["id"]: o for o in truth["objects"]}
    groups = {}  # evaluate's exchangeable groups (every truth position is known); ungrouped requests match themselves
    labels = group_labels(condition["objects"], condition.get("constraints", []), [[True] * 3] * len(condition["objects"]))
    for request, label in zip(condition["objects"], labels):
        groups.setdefault(label or request["id"], []).append(request["id"])
    matched = {}  # truth id -> the predicted id assigned to it
    for ids in groups.values():
        cost = [[math.dist(predicted[a]["bottom_center_m"], truth_by_id[b]["bottom_center_m"]) for b in ids] for a in ids]
        for a, b in zip(*linear_sum_assignment(np.asarray(cost))):
            matched[ids[b]] = ids[a]

    def errors(p, t):
        (sx, sy, sz), (box_size, box_yaw) = t["target_size_local_m"], _box_equivalent_errors(p, t, True, True)
        return {"position_error_m": math.dist(p["bottom_center_m"], t["bottom_center_m"]),
                "log_size_error": box_size, "yaw_error_rad": box_yaw,
                "log_size_error_marginal_min": min(_log_size_error(p["target_size_local_m"], s) for s in ((sx, sy, sz), (sy, sx, sz))),
                "yaw_error_rad_marginal_min": _yaw_error(p["yaw_rad"], t["yaw_rad"], 2)}

    rows = []
    for t, obj in zip(truth["objects"], scene["objects"]):
        row = errors(predicted[matched[t["id"]]], t)
        rows.append({"id": t["id"], "predicted_id": matched[t["id"]], "category": obj["type"], "place": place_of(obj["place_id"]),
                     **row, "room_center_position_error_m": math.dist(center, t["bottom_center_m"]),
                     "uniform_yaw_baseline_error_rad": _uniform_yaw_error(4, True, t["target_size_local_m"]),
                     "box_equivalent_log_size_error": row["log_size_error"], "box_equivalent_yaw_error_rad": row["yaw_error_rad"],
                     **{f"{key}_by_id": value for key, value in errors(predicted[t["id"]], t).items()}})
    metrics = [key for key in rows[0] if key not in ("id", "predicted_id", "category", "place")] if rows else []
    pool = lambda group: {"objects": len(group), **{key: float(np.mean([r[key] for r in group])) if group else None
                                                    for key in metrics}}
    places = sorted({r["place"] for r in rows})
    return {"schema_version": "fastfill.roomgenbench-reference-check.v2", "handoff": str(Path(handoff_dir).resolve()),
            "scene": str(Path(scene_path).resolve()), "scene_key": scene.get("scene_key"), "objects": len(rows),
            "ground_truth_tilted_objects": sum(max(abs(o["rotation"].get(k, 0)) for k in "xy") > 1.
                                               for o in scene["objects"]),  # > 1 degree about X or Y; dropped upright
            "checks": checks, "errors": {"all": pool(rows), **{place: pool([r for r in rows if r["place"] == place])
                                                               for place in places}},
            "uniform_yaw_baseline_error_rad": pool(rows)["uniform_yaw_baseline_error_rad"] if rows else None,
            "uniform_yaw_baseline_error_rad_marginal_min": math.pi / 4, "per_object": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", help="directory from predict --export-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--method", default="layout_boxes")
    parser.add_argument("--assets-dir", help="method GLB/JSON sidecar directory")
    parser.add_argument("--roomgenbench-root")
    parser.add_argument("--require-placement", action="store_true")
    parser.add_argument("--display-height-m", type=float, default=3.)
    parser.add_argument("--requests-from", help="instead of assembling: RoomGenBench scenes directory to turn into requests")
    parser.add_argument("--requests-out", help="new directory for the <scene_key>.json requests of --requests-from")
    parser.add_argument("--reference-check", metavar="HANDOFF", help="instead of assembling: compare HANDOFF with --scene")
    parser.add_argument("--scene", help="benchmark scene JSON (bench/inputs/scenes/<room>.json) for --reference-check")
    parser.add_argument("--output", help="new JSON report path for --reference-check")
    args = parser.parse_args(argv)
    if args.reference_check is not None or args.scene is not None or args.output is not None:
        if None in (args.reference_check, args.scene, args.output) or args.handoff or args.output_dir or args.requests_from or args.requests_out:
            parser.error("--reference-check, --scene and --output go together, without other modes")
        target = safe_output(args.output)
        report = reference_check(args.reference_check, args.scene)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"output": str(target), "objects": report["objects"],
                          **{name: {"ok": c["ok"], **c["counts"]} for name, c in report["checks"].items()}}))
        return 0
    if args.requests_from is not None or args.requests_out is not None:
        if args.requests_from is None or args.requests_out is None or args.handoff or args.output_dir:
            parser.error("--requests-from and --requests-out go together, without --handoff/--output-dir")
        for row in write_benchmark_requests(args.requests_from, args.requests_out, roomgenbench_root=args.roomgenbench_root):
            print(json.dumps(row))
        return 0
    if args.handoff is None or args.output_dir is None:
        parser.error("the following arguments are required: --handoff, --output-dir")
    receipt = assemble_handoff(args.handoff, args.output_dir, method=args.method, assets_dir=args.assets_dir,
        roomgenbench_root=args.roomgenbench_root, require_placement=args.require_placement, display_height_m=args.display_height_m)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "counts": receipt["counts"],
                      "generated_mesh_success": receipt["generated_mesh_success"]}))
    return 0 if args.method == "layout_boxes" or receipt["generated_mesh_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
