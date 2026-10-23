"""Step ⑤ of the draft workflow: small rule-based polish that never breaks a hard constraint.

Per floor object (priority order; whatever stands on it moves along rigidly):
  * wall snap: the side nearest a wall, within ``WALL_REACH_M``, is turned flush (yaw snapped to the wall's
    direction when within ``ALIGN_TOL``) and moved to ``SNAP_GAP_M`` from it;
  * axis alignment: a yaw within ``ALIGN_TOL`` of the room's wall directions is snapped to it;
  * soft relations: a violated soft ``near`` is closed by stepping towards the target (at most ``MAX_SHIFT_M``),
    a violated soft ``faces`` within ``MAX_TURN`` is turned to face the target.
A move is kept only when the moved boxes have no more hard issues than before (none, for a box that had none),
the soft score does not drop, and the object stays within ``MAX_SHIFT_M`` / ``MAX_TURN`` of where step ⑤ found it.
"""

from __future__ import annotations

import math

from shapely.geometry import LineString

from .geometry import Geometry, footprint
from .planning import Plan

MAX_SHIFT_M, MAX_TURN = .30, math.radians(30)
SNAP_GAP_M, WALL_REACH_M, ALIGN_TOL = .01, .15, math.radians(10)


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def _walls(geometry: Geometry) -> list[tuple[LineString, float]]:
    """(segment, direction of the wall's inward normal) per edge of the floor polygon."""
    points = geometry.room["floor_polygon_xy_m"]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))
    sign = 1 if area > 0 else -1
    walls = []
    for a, b in zip(points, points[1:] + points[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) > .1:
            walls.append((LineString([a, b]), math.atan2(sign * dx, -sign * dy)))
    return walls


def _moved(box: dict, dx: float, dy: float, turn: float = 0., pivot: tuple[float, float] | None = None) -> dict:
    x, y = box["bottom_center_m"][:2]
    if turn and pivot is not None:
        c, s = math.cos(turn), math.sin(turn)
        rx, ry = x - pivot[0], y - pivot[1]
        x, y = pivot[0] + c * rx - s * ry, pivot[1] + s * rx + c * ry
    return {**box, "bottom_center_m": [round(x + dx, 4), round(y + dy, 4), box["bottom_center_m"][2]],
            "yaw_rad": _wrap(box["yaw_rad"] + turn)}


class Refiner:
    def __init__(self, geometry: Geometry, plan: Plan, placed: dict[str, dict]):
        self.g, self.plan = geometry, plan
        self.placed = dict(placed)
        self.origin = {ident: (box["bottom_center_m"][:2], box["yaw_rad"]) for ident, box in placed.items()}
        self.walls = _walls(geometry)
        self.children: dict[str, list[str]] = {}
        for ident, parent in plan.parents.items():
            self.children.setdefault(parent, []).append(ident)
        self.counts = {"wall_snap": 0, "axis_align": 0, "near": 0, "faces": 0, "rejected": 0}

    def _carried(self, ident: str) -> list[str]:
        result, stack = [], [ident]
        while stack:
            for child in self.children.get(stack.pop(), []):
                result.append(child)
                stack.append(child)
        return result

    def _issues(self, ids: list[str], placed: dict[str, dict]) -> int:
        return sum(len(self.g.new_hard_issues(placed[i], placed)) for i in ids)

    def _within_bounds(self, ident: str, box: dict) -> bool:
        (x0, y0), yaw0 = self.origin[ident]
        x, y = box["bottom_center_m"][:2]
        return math.hypot(x - x0, y - y0) <= MAX_SHIFT_M + 1e-9 and \
            abs(_wrap(box["yaw_rad"] - yaw0)) <= MAX_TURN + 1e-9

    def _try(self, ident: str, dx: float, dy: float, turn: float, kind: str, need_soft_gain: bool) -> bool:
        """Moves ident (and what it carries) rigidly; keeps the move only if it is no worse."""
        box = self.placed[ident]
        pivot = tuple(box["bottom_center_m"][:2])
        moved = {ident: _moved(box, dx, dy, turn, pivot)}
        if not self._within_bounds(ident, moved[ident]):
            self.counts["rejected"] += 1
            return False
        for child in self._carried(ident):
            moved[child] = _moved(self.placed[child], dx, dy, turn, pivot)
        ids = list(moved)
        trial = {**self.placed, **moved}
        before_hard, after_hard = self._issues(ids, self.placed), self._issues(ids, trial)
        before_soft, after_soft = self.g.soft_score(self.placed)[0], self.g.soft_score(trial)[0]
        better_soft = after_soft > before_soft if need_soft_gain else after_soft >= before_soft
        if after_hard > before_hard or (before_hard == 0 and after_hard) or not better_soft:
            self.counts["rejected"] += 1
            return False
        self.placed = trial
        self.counts[kind] += 1
        return True

    def snap_to_wall(self, ident: str) -> None:
        box = self.placed[ident]
        shape = footprint(box)
        segment, normal = min(self.walls, key=lambda w: w[0].distance(shape))
        gap = segment.distance(shape)
        if gap > WALL_REACH_M:
            return
        # the four directions of the wall (normal, along, against, along back); turn flush if close to one
        options = [_wrap(normal + k * math.pi / 2) for k in range(4)]
        target = min(options, key=lambda a: abs(_wrap(box["yaw_rad"] - a)))
        turn = _wrap(target - box["yaw_rad"]) if abs(_wrap(target - box["yaw_rad"])) <= ALIGN_TOL else 0.
        turned = footprint(_moved(box, 0, 0, turn, tuple(box["bottom_center_m"][:2])))
        gap = segment.distance(turned)
        if gap > WALL_REACH_M:
            return
        step = gap - SNAP_GAP_M
        if abs(step) < 1e-3 and abs(turn) < 1e-4:
            return
        dx, dy = -math.cos(normal) * step, -math.sin(normal) * step  # towards the wall
        self._try(ident, dx, dy, turn, "wall_snap", need_soft_gain=False)

    def align_axis(self, ident: str) -> None:
        box = self.placed[ident]
        normal = min(self.walls, key=lambda w: w[0].distance(footprint(box)))[1]
        options = [_wrap(normal + k * math.pi / 2) for k in range(4)]
        target = min(options, key=lambda a: abs(_wrap(box["yaw_rad"] - a)))
        turn = _wrap(target - box["yaw_rad"])
        if 1e-4 < abs(turn) <= ALIGN_TOL:
            self._try(ident, 0., 0., turn, "axis_align", need_soft_gain=False)

    def relations(self, ident: str) -> None:
        for relation in self.g.soft:
            if relation.get("object_id") != ident or self.g.relation_issue(relation, self.placed) is None:
                continue
            target = self.placed.get(relation.get("target_id"), self.g.fixed.get(relation.get("target_id")))
            if target is None:
                continue
            box = self.placed[ident]
            x, y = box["bottom_center_m"][:2]
            tx, ty = target["bottom_center_m"][:2]
            if relation["type"] == "faces":
                turn = _wrap(math.atan2(ty - y, tx - x) - box["yaw_rad"])
                if abs(turn) <= MAX_TURN:
                    self._try(ident, 0., 0., turn, "faces", need_soft_gain=True)
            elif relation["type"] == "near":
                distance = math.hypot(tx - x, ty - y)
                if distance < 1e-6:
                    continue
                for step in (.05, .1, .15, .2, .25, .3):
                    if self._try(ident, (tx - x) / distance * step, (ty - y) / distance * step, 0., "near",
                                 need_soft_gain=True):
                        break


def refine(geometry: Geometry, plan: Plan, placed: dict[str, dict], *, rounds: int = 2) -> tuple[dict[str, dict], dict]:
    """Step ⑤: returns (placed, info) with counts of kept and rejected moves and soft issues before / after."""
    refiner = Refiner(geometry, plan, placed)
    soft_before = len(geometry.soft_score(placed)[1])
    hard_before = len(geometry.validate(placed))
    floor_objects = [ident for ident in placed if plan.parents.get(ident) == "floor"]
    floor_objects.sort(key=lambda i: plan.priority.get(i, 10 ** 6))
    for _ in range(rounds):
        for ident in floor_objects:
            refiner.snap_to_wall(ident)
            refiner.align_axis(ident)
            refiner.relations(ident)
    result = refiner.placed
    info = {**refiner.counts, "soft_issues_before": soft_before,
            "soft_issues_after": len(geometry.soft_score(result)[1]),
            "hard_issues_before": hard_before, "hard_issues_after": len(geometry.validate(result))}
    return result, info
