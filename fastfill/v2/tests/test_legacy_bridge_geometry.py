"""Independent geometric-contract checks for the selected-corpus migration."""
from copy import deepcopy
import json
import math

import pytest

from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.legacy_evidence import EvidenceIndex


def obj(ident, category="chair", **extra):
    return {"id": ident, "category": category, "size": [0.5, 0.6, 0.8],
            "pos": [1.0, 1.0, 0.0], "yaw": 0.0, "tilted": False,
            "anchor": None, "parent": None, **extra}


def room(source, objects, **meta):
    return {"uid": "room", "source": source, "group": "house", "room_type": "office",
            "boundary": [[0., 0.], [5., 0.], [5., 4.], [0., 4.]],
            "boundary_type": "polygon", "height": 3.0, "objects": objects,
            "meta": {"front_known": True, **meta}}


def saved_row():
    return {"uid": "room", "source": "fixture", "flags": {}, "messages": [
        {"role": "system", "content": "frozen fixture"},
        {"role": "user", "content": json.dumps({"constraints": []})},
        {"role": "assistant", "content": json.dumps({"placements": []})}]}


def convert(raw, *, prepared=None, policy="strict"):
    if prepared is None:
        prepared = {**deepcopy(raw),
                    "objects": [deepcopy(o) for o in raw["objects"] if not o.get("structure")],
                    "fixed": [deepcopy(o) for o in raw["objects"] if o.get("structure")]}
    return convert_selected_room(raw, prepared, saved_row(), "train", front_policy=policy)


def by_source(result):
    return {source: (result["target"]["objects"][i],
                     {key: values[i] for key, values in result["validity"].items()})
            for i, source in enumerate(result["provenance"]["target_source_ids"])}


def index(tmp_path):
    (tmp_path / "designer").mkdir()
    (tmp_path / "internscenes").mkdir()
    (tmp_path / "designer/opening-target-join.json").write_text(json.dumps({
        "saved_hits": [{"uid": "room", "source_object_id": "window-src",
                        "kind": "windows", "along_wall_span": 1.4}]}))
    (tmp_path / "internscenes/k0_saved_object_exposure.jsonl").write_text(json.dumps({
        "uid": "room", "object_id": "k0-chair", "vertical_axis": 0}) + "\n")
    return EvidenceIndex.from_root(tmp_path, require=True)


def test_oblique_window_rejected_without_recorded_geometry_correction():
    raw = room("IL3D_3dfront", [obj("chair"), obj(
        "window-src", "window", structure=True, yaw=math.pi / 4,
        size=[1.0, 0.1, 1.5], pos=[3.0, 4.0, 0.9])])
    with pytest.raises(ValueError, match="oblique window"):
        convert(raw)


def test_recorded_oblique_window_correction_enters_fixed_condition(tmp_path):
    raw = room("IL3D_3dfront", [obj("chair"), obj(
        "window-src", "window", structure=True, yaw=math.pi / 4,
        size=[1.0, 0.1, 1.5], pos=[3.0, 4.0, 0.9])])
    prepared = {**deepcopy(raw), "objects": [deepcopy(raw["objects"][0])],
                "fixed": [deepcopy(raw["objects"][1])]}
    result = convert(index(tmp_path).apply_evidence(raw), prepared=prepared)
    fixed = result["condition"]["room"]["fixed_objects"][0]
    assert fixed["size_local_m"] == [1.4, 0.1, 1.5]
    assert fixed["yaw_rad"] == pytest.approx(math.pi / 4)
    assert prepared["fixed"][0]["size"][0] == 1.0
    assert raw["objects"][1]["size"][0] == 1.0


def test_per_object_k0_evidence_does_not_mask_unaffected_legacy_yaw(tmp_path):
    raw = room("InternScenes_mp3d", [obj("k0-chair"), obj("ordinary-chair")], n_permuted_axes=1)
    result = convert(index(tmp_path).apply_evidence(raw), policy="legacy-convention")
    byid = by_source(result)
    assert byid["k0-chair"][1]["yaw"] is False
    assert byid["ordinary-chair"][1]["yaw"] is True
    assert byid["k0-chair"][1]["position"] == [True] * 3
    assert byid["k0-chair"][1]["size"] == [True] * 3
    assert raw["objects"][0].get("front_known") is None


def test_strict_front_policy_masks_unverified_other_source_yaw():
    result = convert(room("InternScenes_mp3d", [obj("chair")]))
    assert result["validity"]["yaw"] == [False]


def test_multiscan_observed_floor_reframe_changes_each_z_and_height():
    raw = room("MultiScan", [obj("chair", pos=[1.0, 2.0, 0.07]), obj(
        "window", "window", structure=True, pos=[0.0, 2.0, 1.0])],
        floor_z=-1.28, floor_height=-1.23, height_reliable=True,
        n_floor_objects_outside_structure=0)
    raw["boundary_type"] = "hull"
    before = deepcopy(raw)
    result = convert(raw)
    assert raw == before
    condition_room = result["condition"]["room"]
    assert condition_room["floor_known"] is False
    assert condition_room["floor_z_m"] is None
    assert condition_room["height_m"] == pytest.approx(2.95)
    assert condition_room["boundary_known"] is False
    assert result["target"]["objects"][0]["bottom_center_m"][2] == pytest.approx(0.02)
    assert condition_room["fixed_objects"][0]["bottom_center_m"][2] == pytest.approx(0.95)
    assert result["provenance"]["vertical_reframe_m"] == pytest.approx(-0.05)
    assert result["validity"]["yaw"] == [True]


def test_multiscan_unreliable_ceiling_stays_unknown():
    raw = room("MultiScan", [obj("chair")], floor_z=-1.28, floor_height=-1.23,
               height_reliable=False, n_floor_objects_outside_structure=0)
    assert convert(raw)["condition"]["room"]["height_m"] is None


def test_multiscan_target_expanded_polygon_does_not_enter_condition():
    raw = room("MultiScan", [obj("chair")], floor_z=-1.28, floor_height=-1.23,
               height_reliable=True, n_floor_objects_outside_structure=1)
    with pytest.raises(ValueError, match="target-expanded"):
        convert(raw)


def test_holodeck_padded_box_is_not_full_local_size_supervision():
    result = convert(room("OptiScene_holodeck", [obj("chair")]), policy="legacy-convention")
    assert result["validity"]["position"] == [[True] * 3]
    assert result["validity"]["size"] == [[False] * 3]
    assert result["provenance"]["field_evidence"][0]["size_semantics"] == "padded_proxy"


@pytest.mark.parametrize("policy", ["strict", "legacy-convention"])
def test_mansion_annotation_footprint_proxy_does_not_supervise_local_size(policy):
    raw = room("MansionWorld", [obj("chair", anchor="floor")])
    result = convert(raw, policy=policy)
    assert result["validity"]["position"] == [[True] * 3]
    assert result["validity"]["size"] == [[False] * 3]
    assert result["provenance"]["field_evidence"][0]["size_semantics"] == "annotation_footprint_proxy"
    assert result["target"]["objects"][0]["target_size_local_m"] == [0.5, 0.6, 0.8]


def test_tilted_object_keeps_identity_but_has_no_yaw_only_geometry_supervision():
    result = convert(room("SAGE-10k", [obj("tilted", tilted=True, tilt_deg=14.0)]),
                     policy="legacy-convention")
    assert len(result["condition"]["objects"]) == 1
    assert result["validity"] == {"position": [[False] * 3], "size": [[False] * 3], "yaw": [False],
                                  "yaw_symmetry_order": [1], "exchangeable_group": [None],
                                  "size_axis_swap_allowed": [False]}


def test_tilted_fixed_proxy_is_not_admitted_as_verified_actual_geometry():
    raw = room("SpatialGen", [obj("chair"), obj(
        "fixed-tilted", "painting", structure=True, tilted=True)])
    with pytest.raises(ValueError, match="tilted fixed"):
        convert(raw)


def test_explicit_source_parent_is_remapped_without_inventing_support_surface():
    raw = room("SAGE-10k", [obj("table", "table", size=[1.0, 1.0, 0.75], anchor="floor"),
                            obj("book", "book", size=[0.2, 0.3, 0.04], pos=[1.0, 1.0, 0.75],
                                anchor="object", parent="table")])
    result = convert(raw, policy="legacy-convention")
    ids = dict(zip(result["provenance"]["target_source_ids"],
                   [o["id"] for o in result["target"]["objects"]]))
    requests = {o["id"]: o for o in result["condition"]["objects"]}
    assert requests[ids["book"]]["support_parent"] == ids["table"]
    assert requests[ids["table"]]["support_parent"] == "floor"
    assert "support_surface_id" not in requests[ids["book"]]


def test_bbox_inferred_parent_does_not_become_known_condition_support():
    raw = room("InteriorGS", [obj("parent"), obj("child", anchor="object", parent="parent",
                                                anchor_inferred=True)])
    result = convert(raw, policy="legacy-convention")
    assert all("support_parent" not in o for o in result["condition"]["objects"])


def test_legacy_height_drop_keeps_source_height_and_flags_conflict():
    raw = room("SpatialLM", [obj("tall", size=[0.5, 0.6, 3.4])])
    prepared = {**deepcopy(raw), "height": None, "fixed": [],
                "meta": {**raw["meta"], "height_dropped": True}}
    result = convert(raw, prepared=prepared)
    assert result["condition"]["room"]["height_m"] == 3.0  # the input never depends on the answer
    assert result["provenance"]["legacy_height_dropped"] is True
    assert result["provenance"]["height_conflict"] == {"objects": ["obj_0000"], "max_excess_m": pytest.approx(.4)}
    assert result["target"]["objects"][0]["target_size_local_m"] == [0.5, 0.6, 3.4]


def test_legacy_inferred_support_is_recorded_separately_from_raw_source_support():
    raw = room("InteriorGS", [obj("parent"), obj("child")])
    prepared = {**deepcopy(raw), "fixed": [], "objects": [
        {**deepcopy(raw["objects"][0]), "anchor": "floor", "anchor_inferred": True},
        {**deepcopy(raw["objects"][1]), "anchor": "object", "parent": "parent",
         "anchor_inferred": True}]}
    result = convert(raw, prepared=prepared, policy="legacy-convention")
    assert all("support_parent" not in o for o in result["condition"]["objects"])
    for evidence in result["provenance"]["field_evidence"]:
        assert evidence["source_support_inferred"] is False
        assert evidence["legacy_support_inferred"] is True
