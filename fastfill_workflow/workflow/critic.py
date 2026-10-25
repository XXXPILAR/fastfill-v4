"""The critic: one program-only assessment that every step proposing a change is accepted or rolled back by.

``Critic.assess(placed)`` returns
  H  hard violations (``Geometry.validate``: boundary, height, support, collisions, keep-out, the input's hard
     constraints, and door clearance when the geometry treats doors as hard), each with a continuous magnitude in
     metres (square root of an overlap area, a height or support gap, a relation's distance / angle gap), so a
     candidate that turns a 20 cm overlap into a 1 cm one counts as progress (critic_flowchart_v2 §3);
  S  soft costs, each ``weight * min(magnitude / scale, 1)`` so S is bounded per term:
       relation_user   the input's soft constraints (weight 1)
       relation_plan   step ①'s soft relations and group links, suggestions only (weight 0.5)
       template_near / template_faces   relation templates mined from the training rooms (``critic_templates.json``,
                       the even half; ``experiments/mine_split_templates.py``): for each floor obstacle of category A
                       and the nearest obstacle of category B, the gap beyond the template's 80th-percentile gap, or
                       the facing error beyond 45 degrees (weight 0.5); REL (fastfill.v2.ops.layout_quality) counts
                       the same pairs as pass / fail with the odd half's templates
       access          a floor obstacle at least 0.3 m tall whose usable sides (front; beds also the long sides,
                       tables every side; seats and tables do not block each other) all have less than half of the
                       0.5 m strip in front free (layout_quality.acc's rule), magnitude the missing share (weight 0.5)
       window          a soft window-clearance issue (weight 0.25), door clearance when doors are soft (weight 1)
Visual feedback is not part of either score (critic_flowchart_v2 §3: program scores and visual review are kept apart).

``better(candidate, current)`` is the commit rule: with hard violations left, fewer violations, or the same count
with a smaller total magnitude, and no new violation larger than ``PROTECT_M``; at equal H, a lower S; a hard-valid
layout only accepts hard-valid candidates. Both margins (``EPS_H``, ``EPS_S``) keep noise from counting as progress.

S uses the same definitions the evaluation reports (ACC, REL with another half's templates); a loop that optimises S
can inflate those numbers, so claims of better layouts need metrics the critic does not see (design rules, a VLM or
human comparison). This is recorded, not solved, here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import contract  # noqa: F401  (puts the fastfill package on sys.path)
from fastfill.v2.ops import layout_quality as lq
from fastfill.validate import SEAT, TABLE

from .geometry import Geometry, Issue, footprint

TEMPLATES = Path(__file__).resolve().parents[1] / "critic_templates.json"
EPS_H, EPS_S, PROTECT_M = .005, .01, .05
WEIGHTS = {"relation_user": 1., "relation_plan": .5, "template_near": .5, "template_faces": .5, "access": .5,
           "window": .25, "door": 1.}
SCALE = {"distance": 1., "angle": math.pi / 2, "share": .5}


@dataclass(frozen=True)
class Violation:
    code: str
    ids: tuple[str, ...]
    magnitude: float
    weight: float = 1.
    detail: str = ""

    @property
    def cost(self) -> float:
        return self.weight * min(self.magnitude, 1.)

    def as_dict(self) -> dict:
        return {"code": self.code, "object_ids": list(self.ids), "magnitude": round(self.magnitude, 4),
                "weight": self.weight, **({"detail": self.detail} if self.detail else {})}


@dataclass
class Assessment:
    hard: list[Violation] = field(default_factory=list)
    soft: list[Violation] = field(default_factory=list)

    @property
    def hard_count(self) -> int:
        return len(self.hard)

    @property
    def hard_magnitude(self) -> float:
        return sum(v.magnitude for v in self.hard)

    @property
    def soft_total(self) -> float:
        return sum(v.cost for v in self.soft)

    def key(self) -> tuple[int, float, float]:
        return self.hard_count, round(self.hard_magnitude, 4), round(self.soft_total, 4)

    def summary(self) -> dict:
        by_code: dict[str, list] = {}
        for v in self.soft:
            entry = by_code.setdefault(v.code, [0, 0.])
            entry[0] += 1
            entry[1] += v.cost
        return {"hard_count": self.hard_count, "hard_magnitude": round(self.hard_magnitude, 4),
                "soft_total": round(self.soft_total, 4),
                "soft_by_code": {k: {"count": n, "cost": round(c, 4)} for k, (n, c) in sorted(by_code.items())},
                "hard": [v.as_dict() for v in self.hard]}


def load_templates(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else TEMPLATES
    return json.loads(path.read_text())["templates"] if path.is_file() else []


def _sqrt_area(area: float) -> float:
    return math.sqrt(max(area, 0.))


class Critic:
    def __init__(self, geometry: Geometry, user_soft: list[dict], plan_soft: list[dict], templates: list[dict]):
        self.g = geometry
        self.user_soft, self.plan_soft = user_soft, plan_soft
        self.templates: dict[str, list[dict]] = {}
        for t in templates:
            self.templates.setdefault(t["a"], []).append(t)
        self.row = {"condition": geometry.condition}  # what layout_quality's helpers read
        self.fixed_polys = lq.fixed_obstacles(self.row)
        self.floor_poly = geometry.polygon.buffer(0)

    @classmethod
    def for_plan(cls, geometry: Geometry, plan, templates: list[dict] | None = None) -> "Critic":
        user = [r for r in geometry.explicit if not r.get("hard", True)]
        return cls(geometry, user, list(plan.relations), load_templates() if templates is None else templates)

    def with_geometry(self, geometry: Geometry) -> "Critic":
        critic = Critic.__new__(Critic)
        critic.__dict__.update(self.__dict__)
        critic.g = geometry
        return critic

    # ---------------------------------------------------------------- hard
    def _hard_magnitude(self, issue: Issue, placed: dict[str, dict]) -> float:
        g, code = self.g, issue.code
        box = placed.get(issue.ids[0]) if issue.ids else None
        if box is None:
            return 1.
        shape = footprint(box)
        z, height = box["bottom_center_m"][2], box["target_size_local_m"][2]
        if code == "collision":
            other = placed.get(issue.ids[1], g.fixed.get(issue.ids[1]))
            return max(_sqrt_area(shape.intersection(footprint(other)).area), .01) if other else 1.
        if code == "boundary":
            return max(_sqrt_area(shape.difference(g.polygon).area), .01)
        if code == "vertical_bounds":
            return max(g.floor - z, z + height - g.ceiling, .01)
        if code == "floor_support":
            return abs(z - g.floor)
        if code == "wall_support":
            return max(shape.distance(g.polygon.exterior), .01)
        if code == "object_support":
            parent = placed.get(issue.ids[1], g.fixed.get(issue.ids[1]))
            if parent is None:
                return 1.
            surface = g._support_polygon(box, issue.ids[1], parent)
            return max(abs(z - g._support_height(box, issue.ids[1], parent)) + surface.distance(shape.centroid), .01)
        if code == "door_clearance":
            return max((_sqrt_area(g.door_overlap(box, zone, opening)) for zone, opening in g.doors), default=.01)
        if code == "window_clearance":
            return max((_sqrt_area(shape.intersection(strip).area) for strip, _ in g.windows), default=.01)
        if code == "keepout":
            return max((_sqrt_area(shape.intersection(Polygon(r["polygon_xy_m"])).area) for r in g.hard
                        if r["type"] == "keepout"), default=.01)
        if code.startswith("relation_"):
            relation = next((r for r in g.hard if "relation_" + r["type"] == code and r.get("object_id") == issue.ids[0]),
                            None)
            gap = g.relation_gap(relation, placed) if relation else None
            return max(gap or 0., .01)
        return 1.

    def hard(self, placed: dict[str, dict]) -> list[Violation]:
        return [Violation(i.code, i.ids, self._hard_magnitude(i, placed), detail=i.detail) for i in self.g.validate(placed)]

    # ---------------------------------------------------------------- soft
    def _relations(self, relations: list[dict], code: str, placed: dict[str, dict]) -> list[Violation]:
        out = []
        for r in relations:
            gap = self.g.relation_gap(r, placed)
            if gap:
                scale = SCALE["angle"] if r["type"] in ("faces", "faces_direction") else SCALE["distance"]
                ids = (r["object_id"],) + tuple(t for t in [r.get("target_id")] + list(r.get("target_ids") or []) if t)
                out.append(Violation(code, ids, gap / scale, WEIGHTS[code], r["type"]))
        return out

    def _quality_boxes(self, placed: dict[str, dict]) -> list[dict]:
        layout = {"objects": [placed[i] for i in self.g.requests if i in placed]}
        return lq.boxes(self.row, layout)

    def _templates(self, bs: list[dict]) -> list[Violation]:
        by_cat: dict[str, list[dict]] = {}
        for b in bs:
            if lq.obstacle(b):
                by_cat.setdefault(b["cat"], []).append(b)
        out = []
        for a_cat, a_list in by_cat.items():
            for t in self.templates.get(a_cat, ()):
                b_list = by_cat.get(t["b"])
                if not b_list or t["b"] == a_cat:
                    continue
                for a in a_list:
                    nearest = min(b_list, key=lambda b: a["poly"].distance(b["poly"]))
                    if t["kind"] == "near":
                        excess = a["poly"].distance(nearest["poly"]) - t["max_gap_m"]
                        if excess > 0:
                            out.append(Violation("template_near", (a["id"], nearest["id"]), excess / SCALE["distance"],
                                                 WEIGHTS["template_near"], f"{a_cat} near {t['b']}"))
                    else:
                        to = nearest["centre"] - a["centre"]
                        norm = float(np.linalg.norm(to))
                        if norm <= 1e-6:
                            continue
                        angle = math.acos(max(-1., min(1., float(np.dot(a["front"], to / norm)))))
                        excess = angle - math.radians(lq.FACING_DEG)
                        if excess > 0:
                            out.append(Violation("template_faces", (a["id"], nearest["id"]), excess / SCALE["angle"],
                                                 WEIGHTS["template_faces"], f"{a_cat} faces {t['b']}"))
        return out

    def _access(self, bs: list[dict]) -> list[Violation]:
        out = []
        obstacles = [b for b in bs if lq.obstacle(b)]
        for b in obstacles:
            if b["size"][2] < lq.ACC_MIN_HEIGHT_M:
                continue
            blocked = unary_union([o["poly"] for o in obstacles if o is not b
                                   and not (b["head"] in SEAT and o["head"] in TABLE)
                                   and not (b["head"] in TABLE and o["head"] in SEAT)] + self.fixed_polys)
            sx, sy, _ = b["size"]
            sides = [(b["front"], sy / 2, sx / 2)]
            if b["head"] in lq.BEDS:
                sides += [(b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
            elif b["head"] in TABLE:
                sides += [(-b["front"], sy / 2, sx / 2), (b["side"], sx / 2, sy / 2), (-b["side"], sx / 2, sy / 2)]
            best = 0.
            for direction, half_width, half_depth in sides:
                s = lq.strip(b, direction, half_width, half_depth)
                if s.area > 0:
                    best = max(best, s.intersection(self.floor_poly).difference(blocked).area / s.area)
            if best < .5:
                out.append(Violation("access", (b["id"],), (.5 - best) / SCALE["share"], WEIGHTS["access"],
                                     f"best side {best:.0%} free"))
        return out

    def soft(self, placed: dict[str, dict]) -> list[Violation]:
        out = self._relations(self.user_soft, "relation_user", placed)
        out += self._relations(self.plan_soft, "relation_plan", placed)
        bs = self._quality_boxes(placed)
        out += self._templates(bs) + self._access(bs)
        for box in placed.values():
            for issue in self.g.opening_issues(box, doors=not self.g.hard_doors, windows=not self.g.hard_windows):
                code = "door" if issue.code == "door_clearance" else "window"
                out.append(Violation(code, issue.ids, self._hard_magnitude(issue, placed), WEIGHTS[code]))
        return out

    def assess(self, placed: dict[str, dict]) -> Assessment:
        return Assessment(self.hard(placed), self.soft(placed))

    # ---------------------------------------------------------------- commit rule
    @staticmethod
    def better(candidate: Assessment, current: Assessment) -> tuple[bool, str]:
        if current.hard_count == 0:
            if candidate.hard_count:
                return False, "new hard violation on a hard-valid layout"
            if candidate.soft_total < current.soft_total - EPS_S:
                return True, "soft cost lower"
            return False, "soft cost not lower"
        new = {(v.code, v.ids) for v in candidate.hard} - {(v.code, v.ids) for v in current.hard}
        worst_new = max((v.magnitude for v in candidate.hard if (v.code, v.ids) in new), default=0.)
        if worst_new > PROTECT_M:
            return False, f"new hard violation of {worst_new:.3f} m"
        if candidate.hard_count < current.hard_count:
            return True, "fewer hard violations"
        if candidate.hard_count == current.hard_count:
            if candidate.hard_magnitude < current.hard_magnitude - EPS_H:
                return True, "smaller hard violations"
            if abs(candidate.hard_magnitude - current.hard_magnitude) <= EPS_H and \
                    candidate.soft_total < current.soft_total - EPS_S:
                return True, "same hard violations, soft cost lower"
        return False, "hard violations not reduced"

    @staticmethod
    def not_worse(candidate: Assessment, current: Assessment) -> bool:
        """For moves that need no gain (polish's wall snap / axis alignment): nothing gets worse."""
        if current.hard_count == 0 and candidate.hard_count:
            return False
        return candidate.hard_count <= current.hard_count and \
            candidate.hard_magnitude <= current.hard_magnitude + 1e-9 and candidate.soft_total <= current.soft_total + 1e-9
