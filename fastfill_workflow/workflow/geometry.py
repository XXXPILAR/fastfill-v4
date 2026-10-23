"""Exact 2D footprints plus height intervals for bbox-level hard checks."""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import LineString, Polygon
from fastfill.v2.validation import on_wall

EPS = 1e-5
# the evaluator's tolerance (fastfill.v2.validation.validate_scene tolerance_m): boundary and vertical bounds
BOUNDARY_TOL = 1e-4
# an object stands on another when its footprint centre lies on the parent's surface (within this margin) and its
# bottom is at the surface height; its footprint may overhang (ground-truth shelves and tables show it)
SUPPORT_CENTRE_MARGIN = .02


def footprint(box: dict) -> Polygon:
    sx, sy = box["target_size_local_m"][:2]
    x, y = box["bottom_center_m"][:2]
    yaw = box["yaw_rad"]
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2),
               (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    return Polygon([(x + c * a - s * b, y + s * a + c * b) for a, b in corners])


def as_box(fixed: dict) -> dict:
    return {"id": fixed["id"], "target_size_local_m": fixed["size_local_m"],
            "bottom_center_m": fixed["bottom_center_m"], "yaw_rad": fixed["yaw_rad"]}


def _vertical_overlap(a: dict, b: dict) -> bool:
    az, bz = a["bottom_center_m"][2], b["bottom_center_m"][2]
    return min(az + a["target_size_local_m"][2], bz + b["target_size_local_m"][2]) - max(az, bz) > EPS


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _signed_area(points: list) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])) / 2


def _opening_strip(points: list, opening: dict, depth: float) -> Polygon | None:
    index = opening.get("wall_index")
    if not isinstance(index, int) or not 0 <= index < len(points):
        return None
    a, b = points[index], points[(index + 1) % len(points)]
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= EPS:
        return None
    ux, uy = dx / length, dy / length
    sign = 1 if _signed_area(points) > 0 else -1
    nx, ny = -sign * uy, sign * ux
    center = float(opening.get("position_on_wall", .5)) * length
    width = float(opening.get("width_m", 0))
    lo, hi = max(0, center - width / 2), min(length, center + width / 2)
    if hi <= lo:
        return None
    p, q = (a[0] + lo * ux, a[1] + lo * uy), (a[0] + hi * ux, a[1] + hi * uy)
    return Polygon([p, q, (q[0] + depth * nx, q[1] + depth * ny),
                    (p[0] + depth * nx, p[1] + depth * ny)])


@dataclass(frozen=True)
class Issue:
    code: str
    ids: tuple[str, ...]
    detail: str = ""

    def as_dict(self) -> dict:
        return {"code": self.code, "object_ids": list(self.ids), "detail": self.detail}


class Geometry:
    def __init__(self, condition: dict, parents: dict[str, str], relations: list[dict],
                 *, allow_thin_overlap: bool = False, hard_openings: bool = True):
        """``hard_openings=False`` (the draft workflow) scores door and window clearance as soft issues: the
        evaluator does not check openings, and ground-truth rooms often place furniture in those strips."""
        self.condition = condition
        self.room = condition["room"]
        self.floor = float(self.room["floor_z_m"])
        self.ceiling = self.floor + float(self.room["height_m"])
        self.polygon = Polygon(self.room["floor_polygon_xy_m"])
        self.requests = {o["id"]: o for o in condition["objects"]}
        self.fixed_raw = {o["id"]: o for o in self.room.get("fixed_objects", [])}
        self.fixed = {ident: as_box(obj) for ident, obj in self.fixed_raw.items()}
        self.parents = parents
        self.allow_thin_overlap = allow_thin_overlap
        self.hard_openings = hard_openings
        self.explicit = condition["constraints"]
        self.soft = relations + [r for r in self.explicit if not r.get("hard", True)]
        self.hard = [r for r in self.explicit if r.get("hard", True)]
        self.doors, self.windows = [], []
        for opening in self.room.get("openings", []):
            if opening.get("type") == "door":
                zone = _opening_strip(self.room["floor_polygon_xy_m"], opening, .8)
                if zone is not None:
                    self.doors.append((zone.intersection(self.polygon), opening))
            elif opening.get("type") == "window":
                strip = _opening_strip(self.room["floor_polygon_xy_m"], opening, .12)
                if strip is not None:
                    self.windows.append((strip.intersection(self.polygon), opening))

    def _support_polygon(self, child: dict, parent_id: str, parent: dict) -> Polygon:
        raw = self.fixed_raw.get(parent_id)
        surface_id = self.requests.get(child["id"], {}).get("support_surface_id")
        surfaces = raw.get("support_surfaces") if raw else None
        if surfaces:
            selected = next((s for s in surfaces if surface_id is None or s["surface_id"] == surface_id), None)
            if selected is None:
                return Polygon()
            x, y = parent["bottom_center_m"][:2]
            c, s = math.cos(parent["yaw_rad"]), math.sin(parent["yaw_rad"])
            return Polygon([(x + c * u - s * v, y + s * u + c * v)
                            for u, v in selected["local_polygon_xy_m"]])
        return footprint(parent)

    def _support_height(self, child: dict, parent_id: str, parent: dict) -> float:
        raw = self.fixed_raw.get(parent_id)
        surface_id = self.requests.get(child["id"], {}).get("support_surface_id")
        surfaces = raw.get("support_surfaces") if raw else None
        if surfaces:
            selected = next((s for s in surfaces if surface_id is None or s["surface_id"] == surface_id), None)
            if selected is not None:
                return parent["bottom_center_m"][2] + selected["local_z_m"]
        return parent["bottom_center_m"][2] + parent["target_size_local_m"][2]

    def check_box(self, box: dict, placed: dict[str, dict]) -> list[Issue]:
        ident = box["id"]
        shape = footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        issues = []
        if not self.polygon.buffer(BOUNDARY_TOL).covers(shape):
            issues.append(Issue("boundary", (ident,)))
        if z < self.floor - BOUNDARY_TOL or z + height > self.ceiling + BOUNDARY_TOL:
            issues.append(Issue("vertical_bounds", (ident,)))
        parent_id = self.parents[ident]
        if parent_id == "floor":
            if abs(z - self.floor) > EPS:
                issues.append(Issue("floor_support", (ident,)))
        elif parent_id == "wall":
            if not on_wall(box, self.room, tolerance=.1):
                issues.append(Issue("wall_support", (ident,)))
        else:
            parent = placed.get(parent_id, self.fixed.get(parent_id))
            if parent is None:
                issues.append(Issue("parent_not_placed", (ident, parent_id)))
            else:
                surface = self._support_polygon(box, parent_id, parent)
                if (abs(z - self._support_height(box, parent_id, parent)) > EPS or
                        not surface.buffer(SUPPORT_CENTRE_MARGIN).covers(shape.centroid)):
                    issues.append(Issue("object_support", (ident, parent_id)))

        if self.hard_openings:
            issues.extend(self.opening_issues(box))
        issues.extend(self._collisions_and_keepout(box, placed, shape, height))
        return issues

    def opening_issues(self, box: dict) -> list[Issue]:
        ident, shape = box["id"], footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        parent_id = self.parents.get(ident, "floor")
        issues = []
        for zone, opening in self.doors:
            if height <= .05 and parent_id == "floor":
                continue
            door_top = self.floor + float(opening.get("height_m", 2.1))
            if z < door_top - EPS and z + height > self.floor + EPS and shape.intersection(zone).area > EPS:
                issues.append(Issue("door_clearance", (ident,)))
                break
        for strip, opening in self.windows:
            sill = self.floor + float(opening.get("sill_height_m", 0))
            top = sill + float(opening.get("height_m", 1.2))
            if z < top - EPS and z + height > sill + EPS and shape.intersection(strip).area > EPS:
                issues.append(Issue("window_clearance", (ident,)))
                break
        return issues

    def _collisions_and_keepout(self, box: dict, placed: dict[str, dict], shape, height: float) -> list[Issue]:
        ident, issues = box["id"], []
        for other in list(placed.values()) + list(self.fixed.values()):
            if other["id"] == ident:
                continue
            if self.allow_thin_overlap and (height <= .05 or other["target_size_local_m"][2] <= .05):
                continue
            if _vertical_overlap(box, other) and shape.intersection(footprint(other)).area > EPS:
                issues.append(Issue("collision", (ident, other["id"])))
        for relation in self.hard:
            if relation["type"] == "keepout":
                target = relation.get("object_id")
                if (target is None or target == ident) and shape.intersection(
                        Polygon(relation["polygon_xy_m"])).area > EPS:
                    issues.append(Issue("keepout", (ident,)))
        return issues

    def relation_issue(self, relation: dict, placed: dict[str, dict]) -> Issue | None:
        kind = relation["type"]
        if kind == "keepout":
            return None
        ident = relation["object_id"]
        obj = placed.get(ident, self.fixed.get(ident))
        target_id = relation.get("target_id", relation.get("parent_id"))
        target = placed.get(target_id, self.fixed.get(target_id)) if target_id else None
        if obj is None or (kind in {"near", "clearance", "faces"} and target is None):
            return None
        if kind == "faces_direction":
            dx, dy = relation["direction_xy"]
            ok = _angle_error(obj["yaw_rad"], math.atan2(dy, dx)) <= relation.get("tolerance_rad", math.pi / 6) + EPS
        elif kind == "faces":
            dx = target["bottom_center_m"][0] - obj["bottom_center_m"][0]
            dy = target["bottom_center_m"][1] - obj["bottom_center_m"][1]
            ok = math.hypot(dx, dy) > EPS and _angle_error(
                obj["yaw_rad"], math.atan2(dy, dx)) <= relation.get("tolerance_rad", math.pi / 6) + EPS
        elif kind in {"near", "clearance"}:
            distance = footprint(obj).distance(footprint(target))
            ok = (distance <= relation.get("max_distance_m", 1.0) + EPS if kind == "near" else
                  distance + EPS >= relation["min_distance_m"])
        elif kind == "against_wall":
            ok = on_wall(obj, self.room, tolerance=relation.get("tolerance_m", .1))
        elif kind == "between":
            targets = relation["target_ids"]
            if any(t not in placed and t not in self.fixed for t in targets):
                return None
            a, b = [placed.get(t, self.fixed.get(t))["bottom_center_m"][:2] for t in targets]
            segment = LineString([a, b])
            ok = segment.intersects(footprint(obj)) and not segment.touches(footprint(obj))
        elif kind == "on":
            ok = self.parents.get(ident) == target_id
        else:
            raise ValueError(f"unsupported relation: {kind}")
        return None if ok else Issue("relation_" + kind, (ident,) + ((target_id,) if target_id else ()))

    def new_hard_issues(self, box: dict, placed: dict[str, dict]) -> list[Issue]:
        trial = {**placed, box["id"]: box}
        issues = self.check_box(box, placed)
        for relation in self.hard:
            if relation.get("object_id") == box["id"] or box["id"] in (
                    relation.get("target_id"), relation.get("parent_id"), *(relation.get("target_ids") or [])):
                issue = self.relation_issue(relation, trial)
                if issue is not None:
                    issues.append(issue)
        return issues

    def validate(self, placed: dict[str, dict]) -> list[Issue]:
        issues = []
        if set(placed) != set(self.requests):
            issues.append(Issue("missing_or_extra_ids", tuple(sorted(set(placed) ^ set(self.requests)))))
        for ident, box in placed.items():
            issues.extend(self.check_box(box, {k: v for k, v in placed.items() if k != ident}))
        for relation in self.hard:
            issue = self.relation_issue(relation, placed)
            if issue is not None:
                issues.append(issue)
        return issues

    def soft_score(self, placed: dict[str, dict]) -> tuple[float, list[Issue]]:
        issues = [issue for relation in self.soft
                  if (issue := self.relation_issue(relation, placed)) is not None]
        if not self.hard_openings:
            issues += [issue for box in placed.values() for issue in self.opening_issues(box)]
        return -float(len(issues)), issues
