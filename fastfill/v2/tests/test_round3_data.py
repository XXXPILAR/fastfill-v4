"""Round 3 data contract: unreferenced subsets of identical requests are exchangeable, and the room
height entering the condition never depends on the target boxes (K2 stays a provenance flag)."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.legacy_verify import _check_unreferenced_groups, _geometry, verify_selected_dataset
from fastfill.v2.matching import certify_group, group_labels
from fastfill.v2.multisource_verify import verify_full_pair
from fastfill.v2.qualified_data import qualify_sample
from fastfill.v2.tests.test_legacy_bridge import fixture, prepared, saved
from fastfill.v2.tests.test_legacy_bridge_geometry import by_source, convert, obj, room
from fastfill.v2.tests.test_legacy_verify import assert_failed, corpus, mutate_sample
from fastfill.v2.tests.test_qualified_data import sample as qualified_parent


def request(ident, category, **extra):
    return {"id": ident, "category": category, "description": category, **extra}


def complete(objects):
    return [[True] * 3 for _ in objects]


# --- subset exchangeable groups ---------------------------------------------

def test_referenced_member_no_longer_blocks_its_identical_unreferenced_peers():
    chairs = [request(i, "chair") for i in "ABC"] + [request("T", "table")]
    near = [{"type": "near", "object_id": "A", "target_id": "T", "max_distance_m": .5}]
    cushion = chairs + [request("cushion", "cushion", support_parent="A")]
    for objects, constraints in ((chairs, near), (cushion, [])):
        with pytest.raises(ValueError, match="exchange would change"):
            certify_group(objects, constraints, [0, 1, 2])
        labels = group_labels(objects, constraints, complete(objects))
        assert labels[:3] == [None, "anonymous_0", "anonymous_0"] and set(labels[3:]) == {None}
    incomplete = complete(chairs)
    incomplete[2] = [True, False, True]  # only B is left: no group
    assert group_labels(chairs, near, incomplete) == [None] * 4


def test_paired_nightstands_with_lamps_stay_ungrouped_and_symmetric_pairs_group_whole():
    stands = [request("s1", "nightstand"), request("s2", "nightstand")]
    lamps = [request("l1", "lamp", support_parent="s1"), request("l2", "lamp", support_parent="s2")]
    assert group_labels(stands + lamps, [], complete(stands + lamps)) == [None] * 4
    bed = [request("bed", "bed")]
    asymmetric = [{"type": "near", "object_id": "s1", "target_id": "bed", "max_distance_m": .3},
                  {"type": "against_wall", "object_id": "s2"}]
    assert group_labels(stands + bed, asymmetric, complete(stands + bed)) == [None] * 3
    symmetric = [{"type": "near", "object_id": s, "target_id": "bed", "max_distance_m": .3} for s in ("s1", "s2")]
    assert group_labels(stands + bed, symmetric, complete(stands + bed)) == ["anonymous_0", "anonymous_0", None]


def test_bridge_groups_unreferenced_chairs_and_verifier_recomputes_the_rule():
    raw = room("SAGE-10k", [obj("a"), obj("b", pos=[2., 1., 0.]), obj("c", pos=[3., 1., 0.]),
                            obj("cushion", "cushion", anchor="object", parent="a", pos=[1., 1., .45])])
    result = convert(raw, policy="axis")
    groups = {source: validity["exchangeable_group"] for source, (_, validity) in by_source(result).items()}
    assert groups["a"] is None and groups["cushion"] is None and groups["b"] == groups["c"] is not None
    assert _geometry(result)["exchangeable_members"] == 2
    _check_unreferenced_groups(result)
    result["validity"]["exchangeable_group"] = [None] * 4  # the round-2 whole-set-only labels
    with pytest.raises(ValueError, match="must share one exchangeable group"):
        _check_unreferenced_groups(result)


def test_verifier_requires_a_whole_certified_set_to_share_one_group():
    stands = [request("s1", "nightstand"), request("s2", "nightstand"), request("bed", "bed")]
    symmetric = [{"type": "near", "object_id": s, "target_id": "bed", "max_distance_m": .3} for s in ("s1", "s2")]
    row = {"condition": {"objects": stands, "constraints": symmetric}, "target": {"objects": [{"id": o["id"]} for o in stands]},
           "validity": {"exchangeable_group": group_labels(stands, symmetric, complete(stands)), "position": complete(stands)}}
    _check_unreferenced_groups(row)  # both stands are referenced, but their swap certifies: one group
    row["validity"]["exchangeable_group"] = [None] * 3
    with pytest.raises(ValueError, match="must share one exchangeable group"):
        _check_unreferenced_groups(row)


def test_streaming_verifier_rejects_a_dropped_legal_group(tmp_path):
    output, _, _ = corpus(tmp_path)
    mutate_sample(output, "test", lambda row: row["validity"].update(exchangeable_group=[None, None]))
    assert_failed(verify_selected_dataset(output), "must share one exchangeable group")


# --- target-independent room height -----------------------------------------

def test_legacy_height_drop_no_longer_makes_the_condition_depend_on_the_target():
    rows = {}
    for height in (.8, 3.1):  # source room height stays 3.0
        raw = fixture()
        raw["objects"][1]["size"][2] = height
        p = prepared(raw)
        assert bool(p["meta"].get("height_dropped")) is (height > 3)  # the frozen prep still drops it
        rows[height] = convert_selected_room(raw, p, saved(raw, p), "train")
    low, tall = rows[.8], rows[3.1]
    assert low["condition"] == tall["condition"] and tall["condition"]["room"]["height_m"] == 3.0
    assert (low["provenance"]["legacy_height_dropped"], tall["provenance"]["legacy_height_dropped"]) == (False, True)
    assert "height_conflict" not in low["provenance"]
    ident = tall["target"]["objects"][tall["provenance"]["target_source_ids"].index("source-a")]["id"]
    assert tall["provenance"]["height_conflict"] == {"objects": [ident], "max_excess_m": pytest.approx(.1)}
    assert _geometry(tall)  # labels untouched and still valid


def test_streaming_verifier_recomputes_the_bridge_height_flag(tmp_path):
    output, _, _ = corpus(tmp_path)
    mutate_sample(output, "train", lambda row: row["condition"]["room"].update(height_m=.5))  # exceeded, unflagged
    assert_failed(verify_selected_dataset(output), "provenance.height_conflict differs")


def test_streaming_verifier_rejects_the_round2_target_dependent_height_drop(tmp_path):
    output, _, _ = corpus(tmp_path)

    def drop(row):  # what the round-2 bridge wrote: height nulled because of a target, no K2 flag
        row["condition"]["room"]["height_m"] = None
        row["provenance"].pop("height_conflict", None)
        row["provenance"]["legacy_height_dropped"] = True
    mutate_sample(output, "train", drop)
    assert_failed(verify_selected_dataset(output), "room.height_m is null where the frozen prep dropped it")


def test_qualified_build_binds_the_bridge_height_rule(tmp_path):
    from fastfill.v2 import legacy_bridge
    from fastfill.v2.io import fingerprint
    from fastfill.v2.qualified_data import build_dataset
    parent = tmp_path / "parent"
    parent.mkdir()
    for split in ("train", "validation", "test"):
        (parent / f"{split}.jsonl").write_text(json.dumps(qualified_parent(split=split)) + "\n")
    (parent / "manifest.json").write_text('{"schema_version":"fastfill.v2"}\n')
    (parent / "rejections.jsonl").write_text("")
    manifest = build_dataset(parent, tmp_path / "new", expected_hashes={p.name: fingerprint(p) for p in parent.iterdir()})
    # qualify_sample computes K2 with legacy_bridge.height_conflict, so the release binds that file too.
    assert str(Path(legacy_bridge.__file__).resolve()) in manifest["implementation_sha256"]


def flagged_parent(source="SpatialLM"):
    parent = qualified_parent(source)
    parent["target"]["objects"][0]["target_size_local_m"][2] = 2.9
    parent["provenance"]["field_evidence"] = [{"tilted": False}]
    parent["provenance"]["height_conflict"] = {"objects": ["desk"], "max_excess_m": 2.9 - 2.8}
    return parent


def test_qualifier_keeps_an_equal_bridge_flag_without_a_journal_entry():
    parent = flagged_parent()
    row = qualify_sample(parent)
    assert row["provenance"]["height_conflict"] == parent["provenance"]["height_conflict"]
    assert row["provenance"]["qualification_changes"] == []
    verify_full_pair(parent, row, "train", holdout_groups=())


def test_qualifier_replaces_a_bridge_flag_its_masks_no_longer_support():
    parent = flagged_parent("Scan2CAD")  # every position is demoted
    row = qualify_sample(parent)
    assert "height_conflict" not in row["provenance"]
    assert {"field": "provenance.height_conflict", "before": parent["provenance"]["height_conflict"], "after": None,
            "reason": "height_conflict_follows_qualified_masks"} in row["provenance"]["qualification_changes"]
    verify_full_pair(parent, row, "train", holdout_groups=())
    stale = deepcopy(row)
    stale["provenance"]["height_conflict"] = parent["provenance"]["height_conflict"]
    with pytest.raises(ValueError, match="height_conflict"):
        verify_full_pair(parent, stale, "train", holdout_groups=())
