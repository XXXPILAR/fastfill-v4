"""Round 10: wall anchors and supported items the frozen prep dropped only by a selection rule become targets."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastfill.build import prep
from fastfill.scene import canonical, messages
from fastfill.v2.batch import AUGMENT_DEFAULTS, TinyTokenizer, augment_sample, collate_samples
from fastfill.v2.legacy_bridge import convert_selected_room
from fastfill.v2.legacy_verify import verify_selected_dataset
from fastfill.v2.multisource_data import ROOMGENBENCH_HOLDOUT_GROUPS
from fastfill.v2.multisource_verify import verify_full_pair
from fastfill.v2.qualified_data import qualify_sample
from fastfill.v2.review_data import review_sample
from fastfill.v2.tests.test_legacy_build import PACKAGE, V1_FILES, build, miniature_release, rows, sha256
from fastfill.v2.tests.test_legacy_verify import assert_failed

ROOT = Path(__file__).resolve().parents[3]
ARGS = {"sources": ["SAGE-10k"], "boundary_types": ["polygon", "hull"], "anchors": ["floor", "object"],
        "source_anchors": {}, "min_objects": 1, "max_vertices": 0, "oob_tol": .1, "hidden_max": .3, "reject_flagged": []}


def _obj(ident, category, pos, size, anchor, parent=None):
    return {"id": ident, "category": category, "size": size, "pos": pos, "yaw": 0., "anchor": anchor,
            "parent": parent, "tilted": False}


def wall_room(source="SAGE-10k", boundary_type="polygon", group="sage:wall-room"):
    """desk/bookcase pass the frozen prep; everything else it drops."""
    return {"uid": f"{source}::wall-room", "source": source, "group": group, "room_type": "study",
            "boundary_type": boundary_type, "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]], "height": 2.7,
            "meta": {"front_known": True}, "objects": [
        _obj("desk", "desk", [3, 2.5, 0], [1.2, .6, .75], "floor"),
        _obj("bookcase", "bookcase", [1, 4.6, 0], [.8, .35, 1.8], "floor"),
        _obj("trinket", "figurine", [1.1, 4.6, .28], [.05, .05, .1], "object", "tray"),  # 2 cm below its parent
        _obj("book_inner", "book", [1, 4.6, .9], [.2, .15, .25], "object", "bookcase"),  # inner shelf
        _obj("tray", "tray", [1.1, 4.6, .3], [.3, .2, .05], "object", "bookcase"),
        _obj("vase_float", "vase", [3, 2.5, 1.2], [.1, .1, .3], "object", "desk"),       # floats above the desk
        _obj("mug_off", "mug", [4.5, 2.5, .75], [.1, .1, .1], "object", "desk"),         # off the desk footprint
        _obj("painting", "painting", [3, 4.98, 1.4], [.8, .04, .6], "wall"),
        _obj("wall_shelf", "shelf", [5, 4.85, 1.2], [.6, .3, .03], "wall"),
        _obj("shelf_book", "book", [5, 4.85, 1.23], [.2, .15, .25], "object", "wall_shelf"),
        _obj("pendant", "pendant lamp", [3, 2.5, 2.2], [.4, .4, .4], "wall"),           # hanging -> ceiling
        _obj("poster", "poster", [1, .02, 1.5], [.6, .04, .8], None)]}                  # wall only by inference


def saved(raw):
    prepared, reason = prep(deepcopy(raw), SimpleNamespace(**ARGS))
    assert reason is None, reason
    assert [o["id"] for o in prepared["objects"]] == ["desk", "bookcase"]
    return prepared, {"uid": raw["uid"], "source": raw["source"], "flags": prepared["meta"]["flags"],
                      "messages": messages(canonical(prepared))}


def convert(raw, split="train"):
    return convert_selected_room(raw, *saved(raw), split)


def by_source(sample):
    """source id -> (selection rule, declared parent as source id / floor / wall / None, position, size, yaw valid)."""
    source = dict(zip([t["id"] for t in sample["target"]["objects"]], sample["provenance"]["target_source_ids"]))
    validity = sample["validity"]
    return {source[o["id"]]: (e["selection_rule"], source.get(o.get("support_parent"), o.get("support_parent")),
                              all(p), all(s), y)
            for o, e, p, s, y in zip(sample["condition"]["objects"], sample["provenance"]["field_evidence"],
                                     validity["position"], validity["size"], validity["yaw"])}


def test_wall_anchors_and_selection_dropped_supported_items_become_declared_targets():
    got = by_source(convert(wall_room()))
    assert got == {
        "desk": ("frozen_prep", "floor", True, True, True), "bookcase": ("frozen_prep", "floor", True, True, True),
        "painting": ("wall_anchor", "wall", True, True, True), "wall_shelf": ("wall_anchor", "wall", True, True, True),
        "book_inner": ("support_inside_parent", "bookcase", True, True, True),
        "tray": ("support_inside_parent", "bookcase", True, True, True),
        "trinket": ("support_on_added_parent", "tray", True, True, True),
        "shelf_book": ("support_on_added_parent", "wall_shelf", True, True, True)}
    # floating / off-footprint children, hanging (ceiling) items and inferred-only wall art stay dropped


def test_source_certified_labels_only_mansionworld_size_stays_masked():
    got = by_source(convert(wall_room("MansionWorld")))
    assert got["painting"] == ("wall_anchor", "wall", True, False, True)
    assert not any(size for _, _, _, size, _ in got.values())


def test_il3d_wall_flags_are_not_wall_supervision():
    got = by_source(convert(wall_room("IL3D_synthetic")))
    assert set(got) == {"desk", "bookcase", "book_inner", "tray", "trinket"}
    assert "wall" not in {parent for _, parent, *_ in got.values()}


def test_unknown_boundary_adds_the_wall_target_without_a_declaration():
    got = by_source(convert(wall_room(boundary_type="hull")))
    assert got["painting"][:2] == ("wall_anchor", None)


def test_wall_rows_survive_review_qualification_holdout_and_full_condition_verify():
    parent = review_sample(convert(wall_room(group=ROOMGENBENCH_HOLDOUT_GROUPS[0])))
    row = qualify_sample(parent, holdout_groups=ROOMGENBENCH_HOLDOUT_GROUPS)
    assert row["provenance"]["split"] == "test" and row["condition"] == parent["condition"]
    assert sum(o.get("support_parent") == "wall" for o in row["condition"]["objects"]) == 2
    assert verify_full_pair(parent, row, "train")["wall_declared_objects"] == 2
    rules = [e["selection_rule"] for e in parent["provenance"]["field_evidence"]]
    for tamper, message in (("off_wall", "WALL_GAP_M"), ("frozen_wall_anchor", "source wall/ceiling anchor"),
                            ("declared", "wall support declared")):
        bad_parent, bad_row = deepcopy(parent), deepcopy(row)
        for sample in (bad_parent, bad_row):
            if tamper == "off_wall":  # the painting moved to the room centre, still declared on the wall
                sample["target"]["objects"][rules.index("wall_anchor")]["bottom_center_m"][:2] = [3., 2.5]
            elif tamper == "frozen_wall_anchor":
                sample["provenance"]["field_evidence"][rules.index("frozen_prep")]["raw_anchor"] = "wall"
            else:  # a wall declaration without its source wall anchor
                sample["condition"]["objects"][rules.index("frozen_prep")]["support_parent"] = "wall"
        with pytest.raises(ValueError, match=message):
            verify_full_pair(bad_parent, bad_row, "train")


def test_training_batches_wall_declared_rows_under_default_augmentation():
    sample = convert(wall_room())
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        collate_samples([augment_sample(sample, AUGMENT_DEFAULTS, generator)], TinyTokenizer(), max_length=8192)
    batch = collate_samples([sample], TinyTokenizer(), max_length=8192)
    assert int(batch["slot_mask"].sum()) == 8


def wall_release(root):
    release, evidence = miniature_release(root)  # evidence files only; the SpatialLM release is replaced
    data, ir = release / "data/v3.2", release / "ir"
    for path in (*data.iterdir(), *ir.iterdir()):
        path.unlink()
    raw = wall_room()
    _, row = saved(raw)
    (data / "train.jsonl").write_text(json.dumps(row) + "\n")
    (data / "dev.jsonl").write_text("")
    (data / "test.jsonl").write_text("")
    (ir / "SAGE-10k.jsonl").write_text(json.dumps(raw) + "\n")
    manifest = {"args": ARGS, "files": {f"{s}.jsonl": {"sha256": sha256(data / f"{s}.jsonl")} for s in ("train", "dev", "test")},
                "ir_sha256": {"SAGE-10k.jsonl": sha256(ir / "SAGE-10k.jsonl")},
                "code_sha256": {name: sha256(PACKAGE / name) for name in V1_FILES}}
    (data / "MANIFEST.json").write_text(json.dumps(manifest))
    output = root / "dataset"
    build(release, evidence, output)
    return output


def test_build_counts_and_independent_verifier_accept_the_wall_rule(tmp_path):
    output = wall_release(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["wall_declarations_written"] == 2
    assert manifest["selection_rule_counts"] == {"frozen_prep": 2, "wall_anchor": 2, "support_inside_parent": 2,
                                                 "support_on_added_parent": 2}
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]
    assert report["wall_declarations_written"] == 2


def _index(row, source_id):
    return row["provenance"]["target_source_ids"].index(source_id)


def _tamper(name):
    def change(row):
        i = {k: _index(row, k) for k in ("desk", "painting", "book_inner")}
        objects, evidence, targets = row["condition"]["objects"], row["provenance"]["field_evidence"], row["target"]["objects"]
        if name == "wall_on_frozen":
            objects[i["desk"]]["support_parent"] = "wall"
        elif name == "wall_removed":
            objects[i["painting"]].pop("support_parent")
        elif name == "raw_anchor":
            evidence[i["painting"]]["raw_anchor"] = "floor"
        elif name == "source":
            row["provenance"]["source"] = "IL3D_synthetic"
        elif name == "unknown_rule":
            evidence[i["desk"]]["selection_rule"] = "nearest_wall"
        elif name == "child_undeclared":
            objects[i["book_inner"]].pop("support_parent")
        elif name == "child_rule":
            evidence[i["book_inner"]]["selection_rule"] = "support_on_added_parent"
        elif name == "child_off_parent":
            targets[i["book_inner"]]["bottom_center_m"][0] += 1.
        # review r10: self-consistent bad builds (rules, declarations and parents rewritten together)
        elif name == "wall_forgotten":  # the painting relabelled frozen_prep without its declaration
            evidence[i["painting"]]["selection_rule"] = "frozen_prep"
            objects[i["painting"]].pop("support_parent")
        elif name == "child_as_frozen":
            evidence[i["book_inner"]]["selection_rule"] = "frozen_prep"
        elif name == "frozen_wall_anchor":  # a source wall anchor passed off as frozen prep, floor declaration dropped too
            evidence[i["desk"]]["raw_anchor"] = "wall"
            objects[i["desk"]].pop("support_parent")
        elif name == "wall_to_centre":
            targets[i["painting"]]["bottom_center_m"][:2] = [3., 2.5]
    return change


@pytest.mark.parametrize("name,expected", [
    ("wall_on_frozen", "declared exactly on wall_anchor"), ("wall_removed", "declared exactly on wall_anchor"),
    ("raw_anchor", "without a source wall anchor"), ("source", "wall-anchor source"),
    ("unknown_rule", "unknown field_evidence.selection_rule"), ("child_undeclared", "must declare its source"),
    ("child_rule", "disagrees with its parent"), ("child_off_parent", "not on its parent's footprint"),
    ("wall_forgotten", "frozen_prep target with a source wall/ceiling anchor"),
    ("child_as_frozen", "frozen v3.2 row's placements"),
    ("frozen_wall_anchor", "frozen_prep target with a source wall/ceiling anchor"),
    ("wall_to_centre", "within WALL_GAP_M of the room boundary")])
def test_verifier_rejects_wall_and_support_rule_tampering(tmp_path, name, expected):
    output = wall_release(tmp_path)
    path = output / "train.jsonl"
    records = rows(path)
    _tamper(name)(records[0])
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert_failed(verify_selected_dataset(output), expected)


@pytest.mark.parametrize("key", ["wall_declarations_written", "selection_rule_counts"])
def test_verifier_recounts_the_new_manifest_fields(tmp_path, key):
    output = wall_release(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if key == "wall_declarations_written":
        manifest[key] += 1
    else:
        manifest[key]["wall_anchor"] -= 1
    path.write_text(json.dumps(manifest))
    assert_failed(verify_selected_dataset(output), key)


def test_include_uid_takes_a_smoke_room_past_the_per_source_cap(tmp_path):
    release, evidence = miniature_release(tmp_path)
    result = build(release, evidence, tmp_path / "dataset", max_scenes_per_source=1, include_uids=["SpatialLM:test-flagged"])
    assert result["split_samples"] == {"train": 1, "test": 1} and result["bounded_build"]
    assert rows(tmp_path / "dataset/test.jsonl")[0]["provenance"]["legacy_uid"] == "SpatialLM:test-flagged"
    for uid in ("SpatialLM:train-flagged", "SpatialLM:absent"):  # default-filtered or never selected
        with pytest.raises(ValueError, match="included UID"):
            build(release, evidence, tmp_path / uid.split(":")[1], include_uids=[uid])


BENCH = {"bathroom": "61ebde9f", "bedroom": "6b049b06", "gym": "fef15043", "livingroom": "60ee2ae3", "restaurant": "46cdcce3"}
RELEASE = ROOT / ".release/v3.2"
SCENES = ROOT / "RoomGenBench/bench/inputs/scenes"


@pytest.mark.skipif(not (RELEASE / "ir/SAGE-10k.jsonl").exists() or not SCENES.is_dir(),
                    reason="needs the frozen v3.2 release and RoomGenBench scenes")
def test_benchmark_rooms_now_carry_every_roomgenbench_object_with_its_place():
    from fastfill.v2.legacy_evidence import EvidenceIndex
    keys = [h.encode() for h in BENCH.values()]
    pick = lambda path: {(r := json.loads(line))["uid"]: r for line in path.open("rb") if any(k in line[:200] for k in keys)}
    saved_rows, raws = pick(RELEASE / "data/v3.2/train.jsonl"), pick(RELEASE / "ir/SAGE-10k.jsonl")
    args = SimpleNamespace(**json.loads((RELEASE / "data/v3.2/MANIFEST.json").read_text())["args"])
    evidence = EvidenceIndex.from_root(ROOT / ".release/audits/2026-09-28/source-check", require=True)
    places = {"floor": 0, "wall": 0, "object": 0}
    for name, key in BENCH.items():
        uid = f"SAGE-10k::layout_{key}::0"
        prepared, reason = prep(deepcopy(raws[uid]), args)
        assert reason is None
        sample = convert_selected_room(evidence.apply_evidence(raws[uid]), prepared, saved_rows[uid], "train")
        source = dict(zip([t["id"] for t in sample["target"]["objects"]], sample["provenance"]["target_source_ids"]))
        bench = {o["id"]: o["place_id"] for o in json.loads((SCENES / f"{name}.json").read_text())["objects"]}
        declared = {source[o["id"]]: source.get(o.get("support_parent"), o.get("support_parent"))
                    for o in sample["condition"]["objects"]}
        assert declared == bench, name
        for place in bench.values():
            places[place if place in ("floor", "wall") else "object"] += 1
    assert places == {"floor": 109, "wall": 58, "object": 146}


def test_a_wall_anchor_off_the_wall_stays_a_target_without_a_wall_declaration():
    """A source wall anchor whose footprint shows no side within WALL_GAP_M of the boundary (e.g. in a concave
    room) is kept as a target, but no wall support is declared: declarations must hold geometrically."""
    from fastfill.v2.validation import on_wall
    room = {"floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]]}
    touching = {"bottom_center_m": [0.05, 2.0, 1.2], "target_size_local_m": [0.04, 0.8, 0.6], "yaw_rad": 0.0}
    floating = {"bottom_center_m": [2.0, 2.0, 1.2], "target_size_local_m": [0.04, 0.8, 0.6], "yaw_rad": 0.0}
    assert on_wall(touching, room) and not on_wall(floating, room)


def test_an_off_wall_anchor_is_kept_undeclared_and_both_verifiers_agree():
    """The source says wall, but the footprint stands 0.28 m off the wall: kept as a wall_anchor target with
    field_evidence.wall_contact False and no wall declaration; legacy_verify and verify_full_pair both accept it."""
    from collections import Counter
    from fastfill.v2.legacy_verify import _selection_rules
    raw = wall_room()
    painting = next(o for o in raw["objects"] if o["id"] == "painting")
    painting["pos"] = [3, 4.7, 1.4]  # back face at y = 4.72, the wall at y = 5
    sample = convert(raw)
    got = by_source(sample)
    assert got["painting"][:2] == ("wall_anchor", None) and got["wall_shelf"][:2] == ("wall_anchor", "wall")
    evidence = {s: e for s, e in zip(sample["provenance"]["target_source_ids"], sample["provenance"]["field_evidence"])}
    assert evidence["painting"]["wall_contact"] is False and evidence["wall_shelf"]["wall_contact"] is True
    _selection_rules(sample, "SAGE-10k", Counter(), None)  # raises on any wall-rule inconsistency
    parent = review_sample(sample)
    assert verify_full_pair(parent, qualify_sample(parent), "train")["wall_declared_objects"] == 1
