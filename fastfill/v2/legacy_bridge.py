"""Migrate selected v3.2 geometry, retaining lineage and target-independent IDs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from fastfill.scene import canonical, norm_cat, rot90, target_json
from fastfill.v2.matching import group_labels
from fastfill.v2.schema import validate_condition
from fastfill.v2.validation import on_wall

FRONT_POLICIES = ("axis", "strict", "legacy-convention")
YAW_POLICY = {
    "axis": "upright finite yaw with source front_known; yaw_symmetry_order 1 for MultiScan semantic fronts, 2 (axis mod pi) elsewhere; "
            "Scan2CAD sym __SYM_ROTATE_UP_4 -> 4, __SYM_ROTATE_UP_INF -> yaw unsupervised; "
            "box-symmetry tier size_axis_swap_allowed (uncertain local axis pairing) -> 4",
    "strict": "only MultiScan documented semantic fronts; yaw_symmetry_order 1",
    "legacy-convention": "legacy source front convention treated as semantic front; yaw_symmetry_order 1 (lower-evidence ablation)",
}
# K1 box-symmetry tier: the labelled box may equally be (sx, sy, yaw) or (sy, sx, yaw + pi/2).
SWAP_SOURCES = frozenset({"InternScenes_arkit", "InternScenes_3rscan", "InternScenes_mp3d", "InternScenes_scannet", "InteriorGS"})
# HSSD's generic chair template is categorised "seat"; MultiScan beds face a long side (and "bed net" is not a bed).
SWAP_CATEGORIES = {"HSSD200": lambda c: "chair" in c or c == "seat", "MultiScan": lambda c: c == "bed"}
# K3: a source floor anchor within this distance of the known floor is declared and its target z snapped; farther is floating.
FLOOR_SNAP_M = .02
# K2: a target top above room height + this (+1e-6 float slack) is flagged; labels and the height stay.
HEIGHT_TOLERANCE_M = .05


def size_axis_swap_allowed(source, category):
    """Data policy for ``validity.size_axis_swap_allowed`` from the source and normalized category."""
    rule = SWAP_CATEGORIES.get(source)
    return source in SWAP_SOURCES or (rule is not None and rule(category))


def height_conflict(height, targets, position_valid, size_valid):
    """K2 ``provenance.height_conflict`` value ``{objects, max_excess_m}``, or None without a conflict.

    Only targets with complete position and size masks count. 1e-6: float32-rounded source sizes
    (2.6500000953674316 in a 2.6 m room) are not a conflict.
    """
    tops = {t["id"]: t["bottom_center_m"][2] + t["target_size_local_m"][2]
            for t, position, size in zip(targets, position_valid, size_valid) if all(position) and all(size)}
    over = [] if height is None else [ident for ident, top in tops.items() if top > height + HEIGHT_TOLERANCE_M + 1e-6]
    return {"objects": over, "max_excess_m": max(tops[ident] for ident in over) - height} if over else None


def _key(seed, uid, ident):
    return hashlib.sha256(json.dumps([seed, uid, ident], separators=(",", ":")).encode()).hexdigest()


def _legacy_map(prepared, row):
    tagged = {**prepared, "objects": [{**o, "_source_id": o["id"]} for o in prepared["objects"]],
              "fixed": [{**o, "_source_id": o["id"]} for o in prepared.get("fixed", [])]}
    expected = json.loads(row["messages"][-1]["content"])
    for k in range(4):
        candidate = canonical(rot90(deepcopy(tagged), k))
        if json.loads(target_json(candidate)) == expected:
            return {o["id"]: o["_source_id"] for o in candidate["objects"] + candidate.get("fixed", [])}
    raise ValueError("frozen target cannot be reproduced from selected IR preparation")


def _constraint(c, mapping):
    kind = c[0]
    result = {"type": kind, "object_id": mapping[c[1]]}
    if kind in {"near", "faces", "on"}:
        result["parent_id" if kind == "on" else "target_id"] = mapping[c[2]]
    if kind == "near":
        result["max_distance_m"] = c[3]
    if kind == "between":
        result["target_ids"] = [mapping[c[2]], mapping[c[3]]]
    if kind not in {"near", "faces", "on", "between", "against_wall"}:
        raise ValueError(f"unsupported frozen constraint {kind}")
    return result


def _frame(raw):
    meta = raw.get("meta", {})
    dz = 0.
    if raw["source"] == "MultiScan":
        if meta.get("n_floor_objects_outside_structure", 0):
            raise ValueError("target-expanded MultiScan boundary cannot enter condition")
        dz = float(meta["floor_z"]) - float(meta["floor_height"])
    room = {"frame": "right_handed_z_up", "floor_polygon_xy_m": deepcopy(raw["boundary"]),
            "boundary_known": raw.get("boundary_type") == "polygon", "boundary_quality": raw.get("boundary_type", "unknown"),
            "floor_known": raw["source"] != "MultiScan", "floor_z_m": 0. if raw["source"] != "MultiScan" else None,
            "height_m": raw.get("height")}
    if raw["source"] == "MultiScan":
        room["height_m"] = raw["height"] + dz if meta.get("height_reliable") and raw.get("height") else None
    if raw.get("room_type"):
        room["room_type"] = raw["room_type"]
    return room, dz


def _validity(obj, raw, front_policy, category):
    numeric = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    upright = not obj.get("tilted", False)
    size = upright and all(numeric(v) and v > 0 for v in obj["size"])
    position = upright and all(numeric(v) for v in obj["pos"])
    front = obj.get("front_known", raw.get("meta", {}).get("front_known")) is True
    if front_policy == "strict":
        front = front and raw["source"] == "MultiScan"
    sym = obj.get("sym")  # Scan2CAD rotational symmetry: __SYM_NONE / __SYM_ROTATE_UP_2 / _4 / _INF
    yaw = upright and front and numeric(obj.get("yaw")) and sym != "__SYM_ROTATE_UP_INF"
    symmetry = (4 if sym == "__SYM_ROTATE_UP_4" else 2) if front_policy == "axis" and raw["source"] != "MultiScan" else 1
    swap = front_policy == "axis" and size_axis_swap_allowed(raw["source"], category)
    if swap:
        symmetry = 4
    if raw["source"] in {"OptiScene_holodeck", "MansionWorld"}:
        size = False  # Padded boxes and annotation footprints cannot certify local asset extents.
    return [position] * 3, [size] * 3, yaw, symmetry, swap


SELECTION_RULES = ("frozen_prep", "wall_anchor", "support_inside_parent", "support_on_added_parent")
# Round 10: sources whose adapter wall anchor is an explicit per-object source field (SAGE place_id "wall",
# MansionWorld wall_objects / mount_type, SceneSmith asset_group wall_mounted). IL3D's per-asset placement flags plus a
# z gate are not (most of its "wall" boxes stand off the wall); HSSD's objects.csv support column never reached the IR.
WALL_ANCHOR_SOURCES = frozenset({"SAGE-10k", "MansionWorld", "SceneSmith"})
SUPPORT_TOL_M = .05
SELECTION_POLICY = ("frozen prep selection plus round 10: source wall anchors of " + ", ".join(sorted(WALL_ANCHOR_SOURCES))
                    + " (support_parent wall in a known boundary) and source-declared children whose centre lies on their "
                    f"selected or added parent's footprint and whose bottom lies within its box (+-{SUPPORT_TOL_M} m)")


def selection_extensions(raw, prepared, tol=SUPPORT_TOL_M):
    """Round 10: raw IDs the frozen prep dropped only by a selection rule, as {id: rule}.

    wall_anchor: a source wall anchor (``WALL_ANCHOR_SOURCES``), which the frozen anchors=[floor, object] exclude.
    support_inside_parent / support_on_added_parent: a source child (not bbox-inferred) of a frozen-prep or added
    parent, dropped by the top-surface-only test or with its excluded parent: its centre lies on the parent footprint
    (+tol) and its bottom within the parent's box (+-tol), e.g. books on the inner shelves of a bookcase. Floating,
    off-footprint and below-parent children stay dropped (geometry). Structure, soft covers, hanging (ceiling) items,
    generic labels, front_known false, negative sizes and boxes the frozen prep shows as fixed keep their legacy treatment.
    """
    from fastfill.anchors import annotate
    from fastfill.scene import GENERIC, footprint
    from shapely.geometry import Point, Polygon
    source_anchor = {o["id"]: o.get("anchor") for o in raw["objects"]}
    kept = {o["id"] for o in prepared["objects"]}
    taken = kept | {o["id"] for o in prepared.get("fixed", [])}
    eligible = lambda o: (o["id"] not in taken and o.get("front_known") is not False and min(o["size"]) >= 0
                          and norm_cat(o["category"]) and not GENERIC.match(norm_cat(o["category"])))
    annotated = annotate(deepcopy(raw))
    added = {o["id"]: "wall_anchor" for o in annotated if raw["source"] in WALL_ANCHOR_SOURCES
             and o["anchor"] == "wall" and source_anchor[o["id"]] == "wall" and eligible(o)}
    present = {o["id"]: o for o in annotated if o["id"] in kept or o["id"] in added}  # unrounded source boxes
    children = [o for o in annotated if o["anchor"] == "object" and not o.get("anchor_inferred")
                and source_anchor[o["id"]] == "object" and eligible(o)]
    grew = True
    while grew:  # fixed point: a child may sit (<= tol) below its parent's bottom
        grew = False
        for o in children:
            p = present.get(o.get("parent"))
            if (o["id"] not in present and p is not None and Polygon(footprint(p)).distance(Point(o["pos"][:2])) <= tol
                    and p["pos"][2] - tol <= o["pos"][2] <= p["pos"][2] + p["size"][2] + tol):
                added[o["id"]] = "support_on_added_parent" if p["id"] in added else "support_inside_parent"
                present[o["id"]], grew = o, True
    return added


def _description(obj, category):
    text = obj.get("desc", obj.get("description"))
    return text.strip() if isinstance(text, str) and text.strip() else category


def convert_selected_room(raw, prepared, row, split, *, seed=42, front_policy="axis"):
    """Selected objects come from legacy prep plus ``selection_extensions``; labels come from unrounded IR.
    An added wall anchor declares ``support_parent: wall`` (known boundary only), an added child its source parent;
    ``field_evidence.selection_rule`` names the rule (``frozen_prep`` for the legacy selection).

    axis (default) admits every upright finite source yaw with front_known and
    records yaw_symmetry_order 2 (axis mod pi) except MultiScan semantic fronts (1);
    Scan2CAD rotational symmetry raises the order to 4 or leaves yaw unsupervised.
    ``validity.size_axis_swap_allowed`` marks the box-symmetry tier (``SWAP_SOURCES``,
    HSSD200 chairs, MultiScan beds): the labelled box may equally be (sx, sy, yaw)
    or (sy, sx, yaw + pi/2), so those objects carry order 4 under the axis policy.
    strict yaw only admits independently documented MultiScan semantic fronts.
    legacy-convention is an explicit lower-evidence ablation, never a claim of
    per-asset semantic-front verification. Descriptions are the source desc or
    the category; exchangeable groups live in validity. No targets enter the condition.
    Floor support comes from the source annotation only: anchor floor (not inferred)
    with a known floor declares ``support_parent: floor`` and snaps a target z within
    ``FLOOR_SNAP_M`` of the floor (``field_evidence.legacy_z_snap_applied_to_target``);
    a farther target keeps a free z and no declaration.
    Room height is the source height (MultiScan: only when ``height_reliable``), never
    dropped because a target exceeds it: the frozen prep's target-dependent drop is kept
    only as ``provenance.legacy_height_dropped`` and a K2 exceedance as ``provenance.height_conflict``.
    """
    if front_policy not in FRONT_POLICIES:
        raise ValueError("unknown front policy")
    raw_objects = {o["id"]: o for o in raw["objects"]}
    prepared_objects = {o["id"]: o for o in prepared["objects"]}
    added = selection_extensions(raw, prepared)
    selected = [raw_objects[o["id"]] for o in prepared["objects"]] + [raw_objects[i] for i in added]
    selected = sorted(selected, key=lambda o: _key(seed, raw["uid"], o["id"]))
    fixed_selected = sorted(prepared.get("fixed", []), key=lambda o: _key(seed, raw["uid"], o["id"]))
    ids = {o["id"]: f"obj_{i:04d}" for i, o in enumerate(selected)}
    ids = {**ids, **{o["id"]: f"fixed_{i:04d}" for i, o in enumerate(fixed_selected)}}
    room, dz = _frame(raw)
    fixed = []
    for item in fixed_selected:
        o = raw_objects.get(item["id"], item)
        if o.get("tilted"):
            raise ValueError("tilted fixed geometry cannot enter the yaw-only condition protocol")
        if min(o["size"]) <= 0:
            raise ValueError("selected fixed geometry has nonpositive size")
        if (raw["source"] == "IL3D_3dfront" and "window" in norm_cat(o["category"])
                and abs(math.sin(2 * o["yaw"])) > 1e-5
                and not o.get("v2_evidence", {}).get("corrected_width_m")):
            raise ValueError("oblique window requires source-mesh correction before conditioning")
        fixed.append({"id": ids[o["id"]], "category": norm_cat(o["category"]) or "object",
                      "size_local_m": list(o["size"]), "bottom_center_m": [*o["pos"][:2], o["pos"][2] + dz],
                      "yaw_rad": (o["yaw"] + math.pi) % (2 * math.pi) - math.pi})
    if fixed:
        room["fixed_objects"] = fixed
    validity = {"position": [], "size": [], "yaw": [], "yaw_symmetry_order": [], "size_axis_swap_allowed": []}
    requests, targets, source_ids, evidence = [], [], [], []
    for obj in selected:
        category = norm_cat(obj["category"]) or "object"
        request = {"id": ids[obj["id"]], "category": category, "description": _description(obj, category)}
        z, snapped, contact = obj["pos"][2] + dz, False, None
        # Preserve explicit source support, never promote bbox-inferred support.
        if added.get(obj["id"]) == "wall_anchor":
            box = {"bottom_center_m": [*obj["pos"][:2], z], "target_size_local_m": list(obj["size"]),
                   "yaw_rad": (obj["yaw"] + math.pi) % (2 * math.pi) - math.pi}
            contact = on_wall(box, room) if room["boundary_known"] else None  # None: no boundary to touch
            if contact:  # declare only a contact the geometry shows
                request = {**request, "support_parent": "wall"}
        elif obj.get("parent") in ids and not obj.get("anchor_inferred"):
            request = {**request, "support_parent": ids[obj["parent"]]}
        elif (obj.get("anchor") == "floor" and not obj.get("anchor_inferred") and room["floor_known"]
                and abs(z - room["floor_z_m"]) <= FLOOR_SNAP_M):
            request = {**request, "support_parent": "floor"}
            snapped, z = z != room["floor_z_m"], room["floor_z_m"]
        requests.append(request)
        pv, sv, yv, order, swap = _validity(obj, raw, front_policy, category)
        for key, value in (("position", pv), ("size", sv), ("yaw", yv), ("yaw_symmetry_order", order),
                           ("size_axis_swap_allowed", swap)):
            validity[key].append(value)
        targets.append({"id": ids[obj["id"]], "target_size_local_m": list(obj["size"]),
                        "bottom_center_m": [*obj["pos"][:2], z],
                        "yaw_rad": (obj["yaw"] + math.pi) % (2 * math.pi) - math.pi})
        source_ids.append(obj["id"])
        evidence.append({"tilted": bool(obj.get("tilted")), "front_policy": front_policy,
                         "size_semantics": {"OptiScene_holodeck": "padded_proxy",
                                            "MansionWorld": "annotation_footprint_proxy"}.get(
                                                raw["source"], "canonical_source_IR"),
                         "source_support_inferred": bool(obj.get("anchor_inferred")),
                         "legacy_support_inferred": bool(prepared_objects.get(obj["id"], {}).get("anchor_inferred")),
                         "selection_rule": added.get(obj["id"], "frozen_prep"),
                         **({"wall_contact": contact} if added.get(obj["id"]) == "wall_anchor" else {}),
                         "raw_anchor": obj.get("anchor"), "legacy_z_snap_applied_to_target": snapped,
                         "recorded_source_evidence": deepcopy(obj.get("v2_evidence", {}))})
    old_user = json.loads(row["messages"][1]["content"])
    constraints, omitted_constraints = [], []
    if old_user.get("constraints"):
        old_sources = _legacy_map(prepared, row)
        mapping = {old: ids[src] for old, src in old_sources.items()}
        for c in old_user["constraints"]:
            child = prepared_objects.get(old_sources[c[1]], {})
            if c[0] == "on" and child.get("anchor_inferred"):
                omitted_constraints.append({"legacy_constraint": c, "reason": "bbox_inferred_support_not_verified_source_label"})
            else:
                constraints.append(_constraint(c, mapping))
    validity["exchangeable_group"] = group_labels(requests, constraints, validity["position"])  # request order == target order
    condition = {"schema_version": "fastfill.v2", "room": room, "objects": requests, "constraints": constraints}
    validate_condition(condition)
    conflict = height_conflict(room["height_m"], targets, validity["position"], validity["size"])
    result = {"schema_version": "fastfill.v2", "condition": condition,
            "target": {"schema_version": "fastfill.v2", "objects": targets}, "validity": validity,
            "provenance": {"source": raw["source"], "scene_id": raw["uid"], "legacy_uid": raw["uid"],
                           "house_id": raw.get("group", raw["uid"]), "group": raw.get("group"),
                           "split": split, "legacy_flags": deepcopy(row.get("flags", {})),
                           "target_source_ids": source_ids, "field_evidence": evidence,
                           "source_meta": deepcopy(raw.get("meta", {})), "vertical_reframe_m": dz,
                           "legacy_height_dropped": bool(prepared.get("meta", {}).get("height_dropped")),
                           **({"height_conflict": conflict} if conflict else {}),
                           "omitted_legacy_constraints": omitted_constraints,
                           "constraint_source": "frozen_sparse_legacy_request", "request_order": "SHA256(seed,uid,source_id)",
                           "descriptions": "source_desc_or_category"}}
    from fastfill.v2.batch import _geometry_rows, room_normalization
    _geometry_rows(result, *room_normalization(room))
    return result
