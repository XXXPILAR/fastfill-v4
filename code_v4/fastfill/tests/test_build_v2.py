import math
from pathlib import Path

import pytest

from fastfill.build_v2 import keep_fixed, opening_type, strip_sizes, to_openings, untilt_fixed, with_v4_fronts

V2_ROOT = Path(__file__).resolve().parents[3] / "fastfill-main"


def box(i, category, size, pos, yaw=0.0, **kw):
    return {"id": i, "category": category, "size": list(size), "pos": list(pos), "yaw": yaw, **kw}


def test_v4_front_fixes_reach_the_raw_labels():
    raw = {"uid": "r", "objects": [box("b", "bed", [1.6, 2.0, 0.5], [2, 2, 0], yaw=0.1234),
                                   box("d", "desk", [0.6, 1.2, 0.75], [3.7, 1.5, 0], yaw=0.2),
                                   box("c", "chair", [0.5, 0.5, 0.9], [1, 1, 0])]}
    prepared = {"objects": [box("b", "bed", [2.0, 1.6, 0.5], [2, 2, 0], yaw=math.radians(277), yaw_src=0.1234,
                                size_src=[1.6, 2.0, 0.5], front_fix="bed_rot90"),
                            box("d", "desk", [0.6, 1.2, 0.75], [3.7, 1.5, 0], yaw=math.radians(191), yaw_src=0.2,
                                front_fix="rot180"),
                            box("c", "chair", [0.5, 0.5, 0.9], [1, 1, 0])]}
    out = {o["id"]: o for o in with_v4_fronts(raw, prepared)["objects"]}
    assert math.isclose(out["b"]["yaw"], 0.1234 - math.pi / 2 + 2 * math.pi) and out["b"]["size"] == [2.0, 1.6, 0.5]
    assert math.isclose(out["d"]["yaw"], 0.2 + math.pi) and out["d"]["size"] == [0.6, 1.2, 0.75]
    assert out["b"]["v2_evidence"] == {"v4_front_fix": "bed_rot90"} and "v2_evidence" not in out["c"]
    assert raw["objects"][0]["yaw"] == 0.1234                      # the IR room is not modified


def test_fixed_boxes_become_wall_openings_or_stay_fixed():
    room = {"floor_polygon_xy_m": [[0, 0], [4, 0], [4, 3], [4, 3], [0, 3]], "floor_z_m": 0.,
            "fixed_objects": [
                {"id": "f0", "category": "door", "size_local_m": [0.1, 0.9, 2.0], "bottom_center_m": [4.05, 1.0, 0],
                 "yaw_rad": 0.0},                                    # in the x=4 wall (edge 1), 1/3 along it
                {"id": "f1", "category": "window", "size_local_m": [1.2, 0.1, 1.0], "bottom_center_m": [1.0, 3.0, 0.9],
                 "yaw_rad": math.pi},                                # on edge 3 (edge 2 has zero length)
                {"id": "f2", "category": "column", "size_local_m": [0.3, 0.3, 2.5], "bottom_center_m": [2, 1.5, 0],
                 "yaw_rad": 0.0},
                {"id": "f3", "category": "door", "size_local_m": [0.1, 0.9, 2.0], "bottom_center_m": [2, 1.5, 0],
                 "yaw_rad": 0.0}]}                                   # 1.5 m from every wall
    kept, detached = to_openings(room)
    assert (kept, detached) == (1, 1) and [f["id"] for f in room["fixed_objects"]] == ["f2"]   # the column stays
    door, window = room["openings"]
    assert door == {"type": "door", "wall_index": 1, "position_on_wall": round(1 / 3, 6), "width_m": 0.9,
                    "height_m": 2.0}
    assert window["type"] == "window" and window["wall_index"] == 3 and window["position_on_wall"] == 0.75
    assert window["width_m"] == 1.2 and window["sill_height_m"] == 0.9


def test_opening_taxonomy():
    for category, kind in (("door", "door"), ("window1", "window"), ("garage door", "door"), ("doorframe", "door"),
                           ("doorway", "door"), ("sliding door", "door"), ("window frame", "window"),
                           ("shower door", None), ("cabinet door", None), ("bay window pad", None),
                           ("window seat", None), ("radiator", None), ("indoor plant", None)):
        assert opening_type(category) == kind, category


def test_only_unkept_fixed_kinds_are_removed():
    prepared = {"fixed": [{"id": i, "category": c, "fixed_kind": k} for i, c, k in
                          (("d", "door", "structure"), ("c", "column", "structure"), ("u", "unknown", "generic"),
                           ("w", "wardrobe", "no_front"), ("v", "doorway", "generic"))]}
    out, removed = keep_fixed(prepared, ("structure", "no_front"))
    assert [f["id"] for f in out["fixed"]] == ["d", "c", "w", "v"] and removed == {"generic": 1}
    assert len(prepared["fixed"]) == 5                                  # not modified


def test_size_phrases_are_cut_and_the_rest_of_the_description_kept():
    assert strip_sizes("large oak table, 120cm × 60cm") == "large oak table"
    assert strip_sizes("A soy candle with an 8\u202fcm diameter, unbleached wax") == "A soy candle, unbleached wax"
    assert strip_sizes("A ceramic vase, matte white finish, standing 20 cm tall") == "A ceramic vase, matte white finish"
    assert strip_sizes("A paper tag, rectangular with dimensions 2cm by 4cm, earth tone") == \
        "A paper tag, rectangular, earth tone"
    assert strip_sizes("A teak wood ruler 30cm long with straight edges") == "A teak wood ruler with straight edges"
    assert strip_sizes("120 x 60 x 75 cm") is None and strip_sizes("about 2 m tall") is None


def test_only_fixed_tilts_are_cleared():
    raw = {"objects": [box("f", "door", [1, 1, 1], [0, 0, 0], tilted=True), box("o", "lamp", [1, 1, 1], [0, 0, 0],
                                                                               tilted=True)]}
    out, n = untilt_fixed(raw, {"fixed": [{"id": "f"}]})
    assert n == 1 and [o["tilted"] for o in out["objects"]] == [False, True]


@pytest.mark.skipif(not (V2_ROOT / "fastfill" / "v2").is_dir(), reason="fastfill-main checkout not present")
def test_bridge_row_has_no_bbox_in_its_condition():
    from types import SimpleNamespace
    from fastfill.build import prep
    from fastfill.build_v2 import attach_v2, convert
    attach_v2(V2_ROOT)
    from fastfill.v2.schema import validate_layout
    obj = lambda i, cat, size, pos, yaw=0.0: {"id": i, "category": cat, "size": size, "pos": pos, "yaw": yaw,
                                              "anchor": "floor", "parent": None, "tilted": False}
    raw = {"uid": "S::r", "source": "S", "subset": None, "group": "g:1", "room_type": "bedroom",
           "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "boundary_type": "polygon", "height": 2.8,
           "objects": [obj("bed", "bed", [2.0, 1.6, 0.5], [1.05, 1.5, 0], math.pi),
                       obj("ns", "nightstand", [0.4, 0.4, 0.5], [0.25, 2.6, 0]),
                       obj("dr", "door", [0.1, 0.9, 2.0], [4.0, 1.0, 0])],
           "meta": {"front_known": True, "room_id": "r"}}
    a = SimpleNamespace(min_objects=1, max_vertices=0, reject_flagged=[], boundary_types=["polygon", "hull"],
                        anchors=["floor", "object"], oob_tol=0.1, hidden_max=0.3, source_anchors={})
    prepared, why = prep(raw, a)
    assert why is None
    row = convert(raw, prepared, "test", "axis")
    room = row["condition"]["room"]
    assert "fixed_objects" not in room and [o["type"] for o in room["openings"]] == ["door"]
    assert all(set(o) <= {"id", "category", "description", "support_parent"} for o in row["condition"]["objects"])
    validate_layout(row["target"], row["condition"])
    assert row["provenance"]["split"] == "test"
