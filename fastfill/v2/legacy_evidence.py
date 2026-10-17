"""Apply recorded v3.2 source audits to new v2 copies, never to frozen IR.

These indices cover specific recorded defects and source-axis fallbacks. An
absent entry is not evidence that an object's mesh or semantic front is valid.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


OPENINGS_FILE = "designer/opening-target-join.json"
FRONT_FILE = "internscenes/k0_saved_object_exposure.jsonl"
FRONT_REASON = "source_local_x_vertical_geometric_heading_fallback"


def _text(record: Mapping[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing nonempty {key}")
    return value


def _read_openings(path: Path) -> Mapping[str, Mapping[str, float]]:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Malformed evidence file {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("saved_hits"), list):
        raise ValueError(f"{path}: expected saved_hits list")
    widths: dict[str, dict[str, float]] = {}
    for row in payload["saved_hits"]:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: invalid opening record")
        uid = _text(row, "uid", str(path))
        oid = _text(row, "source_object_id", str(path))
        span = row.get("along_wall_span")
        if (isinstance(span, bool) or not isinstance(span, (int, float))
                or not math.isfinite(span) or span <= 0):
            raise ValueError(f"{path}: opening width must be positive finite")
        if row.get("kind") != "windows":
            raise ValueError(f"{path}: only recorded window corrections are supported")
        existing = widths.get(uid, {}).get(oid)
        if existing is not None and existing != span:
            raise ValueError(f"{path}: conflicting width evidence for {uid}/{oid}")
        widths.setdefault(uid, {})[oid] = float(span)
    return MappingProxyType({uid: MappingProxyType(values) for uid, values in widths.items()})


def _read_front(path: Path) -> Mapping[str, frozenset[str]]:
    objects: dict[str, set[str]] = {}
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            context = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed evidence file {context}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{context}: invalid front evidence record")
            uid = _text(row, "uid", context)
            oid = _text(row, "object_id", context)
            axis = row.get("vertical_axis")
            if isinstance(axis, bool) or not isinstance(axis, int) or axis not in (0, 1, 2):
                raise ValueError(f"{context}: invalid vertical_axis")
            if axis == 0:
                objects.setdefault(uid, set()).add(oid)
    return MappingProxyType({uid: frozenset(values) for uid, values in objects.items()})


def _apply_object(obj: Mapping[str, Any], width: float | None, front_unknown: bool) -> dict[str, Any]:
    result = dict(obj)
    evidence = dict(obj.get("v2_evidence", {}))
    if width is not None:
        size = obj.get("size")
        if (str(obj.get("category", "")).lower() != "window" or not obj.get("structure")
                or not isinstance(size, (list, tuple)) or len(size) != 3):
            raise ValueError(f"Recorded window correction has invalid window geometry: {obj.get('id')}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or v <= 0 for v in size):
            raise ValueError(f"Recorded window geometry must have positive finite sizes: {obj.get('id')}")
        result = {**result, "size": [width, size[1], size[2]]}
        evidence = {**evidence, "original_width_m": evidence.get("original_width_m", size[0]),
                    "corrected_width_m": width,
                    "width_evidence_source": OPENINGS_FILE,
                    "width_evidence": "opening_profile_projected_along_source_wall"}
    if front_unknown:
        result = {**result, "front_known": False}
        evidence = {**evidence, "front_unknown_reason": FRONT_REASON,
                    "front_evidence_source": FRONT_FILE, "front_evidence": "unknown"}
    return {**result, "v2_evidence": evidence} if evidence else result


@dataclass(frozen=True)
class EvidenceIndex:
    """Read-only recorded corrections keyed by original room/object identifiers."""

    opening_widths: Mapping[str, Mapping[str, float]]
    front_unknown_objects: Mapping[str, frozenset[str]]
    loaded_files: tuple[str, ...]
    missing_files: tuple[str, ...]
    evidence_root: str | None

    @classmethod
    def from_root(cls, audit_root: str | Path | None, *, require: bool = False) -> "EvidenceIndex":
        """Load an explicit source-check audit root; optional absence is reported.

        Production migration can set require=True. Present but malformed files
        always fail, including when absence itself is permitted.
        """
        root = Path(audit_root) if audit_root is not None else None
        readers = ((OPENINGS_FILE, _read_openings), (FRONT_FILE, _read_front))
        loaded, missing, indices = [], [], []
        for relative, reader in readers:
            path = root / relative if root is not None else None
            if path is None or not path.is_file():
                missing.append(relative)
                indices.append(MappingProxyType({}))
            else:
                loaded.append(relative)
                indices.append(reader(path))
        if require and missing:
            raise FileNotFoundError(f"Required frozen source evidence missing under {root}: {', '.join(missing)}")
        return cls(indices[0], indices[1], tuple(loaded), tuple(missing), str(root) if root else None)

    def apply_evidence(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Return an isolated room with corrections and explicit evidence scope.

        Replay original v1 prep/selection BEFORE calling this method: front
        unknown markers intentionally affect v2 validity, not legacy selection.
        """
        room = deepcopy(dict(raw))
        uid = _text(room, "uid", "IR room")
        widths = self.opening_widths.get(uid, {})
        fronts = self.front_unknown_objects.get(uid, frozenset())
        objects = room.get("objects")
        if not isinstance(objects, list) or any(not isinstance(o, dict) for o in objects):
            raise ValueError(f"IR room {uid}: expected objects list")
        corrected = [_apply_object(o, widths.get(o.get("id")), o.get("id") in fronts) for o in objects]
        n_width = sum(o.get("id") in widths for o in objects)
        n_front = sum(o.get("id") in fronts for o in objects)
        status = "applied" if n_width or n_front else "applied_no_matching_records"
        if self.missing_files:
            status = "partial_evidence" if self.loaded_files else "not_applied"
        summary = {"status": status, "scope": "recorded_source_evidence_only",
                   "loaded_files": list(self.loaded_files), "missing_files": list(self.missing_files),
                   "evidence_root": self.evidence_root, "opening_width_corrections": n_width,
                   "front_unknown_objects": n_front}
        return {**room, "objects": corrected,
                "meta": {**room.get("meta", {}), "v2_evidence": summary}}
