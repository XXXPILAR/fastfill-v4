"""Round 6: requests declare support (floor / wall / another entry), RoomGenBench scenes become full requests,
and spread decoding hangs declared "wall" objects on the nearest wall."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples, load_tokenizer, tokenize_condition
from fastfill.v2.direct_layout import export_handoff, layout_to_roomgenbench, request_to_condition
from fastfill.v2.evaluate import serialize_predictions, spread_grid_xy
from fastfill.v2.roomgenbench import assemble_handoff, benchmark_request, main, write_benchmark_requests
from fastfill.v2.tests.test_spread_decode import ROOM, _logits
from fastfill.v2.validation import validate_scene

REFERENCE = Path(os.environ.get("FASTFILL_ROOMGENBENCH_ROOT") or Path(__file__).resolve().parents[3] / "RoomGenBench")
SCENES = REFERENCE / "bench" / "inputs" / "scenes"


def _request(*furniture):
    return {"room_type": "study", "room_size_m": [4., 3., 2.7], "furniture_list": list(furniture)}


# ---- 1. request schema ----------------------------------------------------------------------------------------

def test_declared_support_parents_reach_the_condition_and_undeclared_entries_are_unchanged():
    plain = [{"id": "desk", "category": "desk"}, {"id": "cup", "category": "cup"}, "chair", {"id": "clock", "category": "clock"}]
    declared = [{**plain[0], "support_parent": "floor"}, {**plain[1], "support_parent": "desk"}, plain[2],
                {**plain[3], "support_parent": "wall"}]
    condition = request_to_condition(_request(*declared))
    assert [o.get("support_parent") for o in condition["objects"]] == ["floor", "desk", None, "wall"]
    stripped = {**condition, "objects": [{k: v for k, v in o.items() if k != "support_parent"} for o in condition["objects"]]}
    assert json.dumps(stripped) == json.dumps(request_to_condition(_request(*plain)))
    assert all("support_parent" not in o for o in request_to_condition(_request(*plain))["objects"])
    # a declared floor fixes z in a known rectangle, as for full training conditions
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=1 << 15)
    assert batch["fixed_position_mask"][0, :, 2].tolist() == [True, False, False, False]


@pytest.mark.parametrize("furniture,message", [
    ([{"id": "cup", "category": "cup", "support_parent": "ghost"}], "'cup': support_parent 'ghost'"),
    ([{"id": "cup", "category": "cup", "support_parent": "cup"}], "'cup': support_parent 'cup'"),
    ([{"category": "cup", "support_parent": "obj_0000"}], "'obj_0000': support_parent 'obj_0000'"),
    ([{"id": "a", "category": "box", "support_parent": "b"}, {"id": "b", "category": "box", "support_parent": "a"}], "cycle"),
    ([{"id": "cup", "category": "cup", "support_parent": ""}], "support_parent must be a nonempty string"),
    ([{"id": "cup", "category": "cup", "support_parent": None}], "support_parent must be a nonempty string"),
])
def test_invalid_support_declarations_raise_clear_errors(furniture, message):
    with pytest.raises(ValueError, match=message):
        request_to_condition(_request(*furniture))


# ---- 2. RoomGenBench request builder ---------------------------------------------------------------------------

def _renamed(scene):
    return {o["id"]: f"obj_{index:04d}" for index, o in enumerate(scene["objects"])}


def test_benchmark_requests_fit_the_8192_token_training_context():
    # RoomGenBench's own ids (room_190f6f8e_table_e05ebbed, ...) put the restaurant at 9,362 Qwen3 tokens
    try:
        tokenizer = load_tokenizer("Qwen/Qwen3-8B", local_files_only=True)
    except (ImportError, OSError):
        pytest.skip("Qwen3-8B tokenizer is not cached locally")
    for path in sorted(SCENES.glob("*.json")):
        condition = request_to_condition(benchmark_request(json.loads(path.read_text())))
        assert len(tokenize_condition(condition, tokenizer)[0]) <= 8192, path.stem


def test_benchmark_requests_list_every_scene_object_with_its_place_id():
    totals = {"requested": 0, "floor": 0, "wall": 0, "on_object": 0}
    for path in sorted(SCENES.glob("*.json")):
        scene = json.loads(path.read_text())
        request = benchmark_request(scene)
        dims = scene["room"]["dimensions"]
        assert request["room_type"] == scene["room_type"]
        assert request["room_size_m"] == [dims["width"], dims["length"], dims["height"]]
        renamed = _renamed(scene)  # training-style ids: obj_0007 is scene["objects"][7]
        assert request["furniture_list"] == [{"id": renamed[o["id"]], "category": o["type"], "description": o["description"],
                                              "support_parent": renamed.get(o["place_id"], o["place_id"])}
                                             for o in scene["objects"]]
        condition = request_to_condition(request)
        assert [o["support_parent"] for o in condition["objects"]] == [renamed.get(o["place_id"], o["place_id"])
                                                                       for o in scene["objects"]]
        for o in scene["objects"]:
            totals["requested"] += 1
            totals[o["place_id"] if o["place_id"] in ("floor", "wall") else "on_object"] += 1
    assert totals == {"requested": 313, "floor": 109, "wall": 58, "on_object": 146}


def test_requests_cli_writes_one_request_per_scene_and_the_handoff_cli_is_unchanged(tmp_path, capsys):
    out = tmp_path / "requests"
    assert main(["--requests-from", str(SCENES), "--requests-out", str(out)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(p.name for p in out.iterdir()) == sorted(f'{r["scene_key"]}.json' for r in rows) and len(rows) == 5
    assert sum(r["requested"] for r in rows) == 313 and sum(r["wall"] for r in rows) == 58
    assert all(r["requested"] == r["floor"] + r["wall"] + r["on_object"] for r in rows)
    bathroom = json.loads((out / "bathroom.json").read_text())
    assert bathroom == benchmark_request(json.loads((SCENES / "bathroom.json").read_text()))
    with pytest.raises(FileExistsError):
        write_benchmark_requests(SCENES, out)
    with pytest.raises(ValueError, match="outside the reference repository"):
        write_benchmark_requests(SCENES, tmp_path / "ref" / "requests", roomgenbench_root=tmp_path / "ref")
    for argv in ([], ["--requests-from", str(SCENES)], ["--requests-from", str(SCENES), "--requests-out", str(tmp_path / "x"),
                                                         "--handoff", str(tmp_path)], ["--handoff", str(tmp_path)]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2


def test_requests_from_refuses_a_scene_over_predicts_default_max_objects(tmp_path):
    # before: checked with max_objects = len(objects), so the 129-object request passed here and failed in predict
    scene = {"scene_key": "hall", "room_type": "hall", "room": {"dimensions": {"width": 9., "length": 9., "height": 3.}},
             "objects": [{"id": f"o{i}", "type": "chair", "description": "chair", "place_id": "floor"} for i in range(129)]}
    (tmp_path / "scenes").mkdir()
    (tmp_path / "scenes" / "hall.json").write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="exceeds max_objects"):
        write_benchmark_requests(tmp_path / "scenes", tmp_path / "requests")
    assert not (tmp_path / "requests").exists()


def test_run_checkpoint_predicts_from_the_requests_of_the_scenes_and_builder_on_disk(tmp_path):
    # before: an existing runs/roomgenbench/requests was used as is (0 builder calls; the export predicted from OLD, exit 0)
    import hashlib
    import shutil
    import subprocess
    import sys
    repo, root = Path(__file__).resolve().parents[3], tmp_path / "checkout"
    for path in ("fastfill/v2/ops", "env/bin", "RoomGenBench/bench/inputs/scenes", "runs/roomgenbench/requests"):
        (root / path).mkdir(parents=True)
    shutil.copy(repo / "fastfill/v2/ops/run_checkpoint.sh", root / "fastfill/v2/ops")
    shutil.copy(SCENES / "gym.json", root / "RoomGenBench/bench/inputs/scenes")
    (root / "runs/roomgenbench/requests/gym.json").write_text('{"input_version": "OLD"}\n')  # e.g. an older builder's
    python = root / "env/bin/python"  # the real request builder; predict and the hand-off assembly are only logged
    python.write_text(f'#!/bin/bash\nprintf "%s\\n" "$*" >> calls.txt\n[ "$3" != --requests-from ] || exec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    for tag in ("a", "b"):
        subprocess.run(["bash", str(root / "fastfill/v2/ops/run_checkpoint.sh"), "/ck/step-1", tag], check=True,
                       env={**os.environ, "PYTHONPATH": str(repo)}, capture_output=True)
    fresh = json.dumps(benchmark_request(json.loads((SCENES / "gym.json").read_text())), indent=2) + "\n"
    assert (root / "runs/roomgenbench/requests/gym.json").read_text() == fresh
    aside = sorted(p.name for p in (root / "runs/roomgenbench").iterdir() if p.name.startswith("requests."))
    assert len(aside) == 1 and aside[0].startswith("requests.stale-")  # the second run kept the identical directory
    assert json.loads((root / "runs/roomgenbench" / aside[0] / "gym.json").read_text()) == {"input_version": "OLD"}
    calls = (root / "calls.txt").read_text().splitlines()
    assert [sum(f" {module} " in c for c in calls) for module in ("--requests-from", "fastfill.v2.predict")] == [2, 2]
    for tag in ("a", "b"):  # the hand-off records the request bytes it predicted from
        assert (root / "runs/roomgenbench" / tag / "requests.sha256").read_text().split() == [
            hashlib.sha256(fresh.encode()).hexdigest(), "runs/roomgenbench/requests/gym.json"]


# ---- 3. spread: declared "wall" objects --------------------------------------------------------------------------

def _wall_spread(cells, z, size=(.1, .8, .6), room=None, **kwargs):
    n = len(cells)
    return spread_grid_xy(torch.stack([_logits(*c) for c in cells]), torch.zeros(n, 16, 2), 4, np.array([size] * n),
                          np.zeros(n), np.array([[0., 0., v] for v in z]), room or {**ROOM, "height_m": 3.},
                          requests=[{"id": f"w{i}", "support_parent": "wall"} for i in range(n)], **kwargs)


def test_a_wall_object_hangs_back_flush_on_the_wall_nearest_its_cell_facing_in_with_z_inside_the_room():
    # cell 1 = (.5, 1.5): wall x = 0; cell 14 = (3.5, 2.5): wall x = 4; cell 7 = (1.5, 3.5): wall y = 4
    xy, yaw, z = _wall_spread([(1,), (14,), (7,)], [2.9, -.5, 1.])
    assert xy == pytest.approx(np.array([[.05, 1.5], [3.95, 2.5], [1.5, 3.95]]))
    assert yaw == pytest.approx([0., math.pi, -math.pi / 2])
    assert z == pytest.approx([2.4, 1.2, 1.])  # clamped to [floor, ceiling - .6]; -.5 is near the floor: centred at 1.5
    _, _, z = _wall_spread([(1,)], [5.], room=ROOM)  # unknown ceiling: only the floor bound
    assert z == pytest.approx([5.])


def test_wall_objects_avoid_each_other_only_on_a_shared_height_interval():
    # w1 shares w0's cell and height: it takes its next cell's projection; w2 hangs above w0 at the same spot
    xy, _, z = _wall_spread([(1, 2), (1, 2), (1, 2)], [1., 1., 2.])
    assert xy == pytest.approx(np.array([[.05, 1.5], [.05, 2.5], [.05, 1.5]])) and z == pytest.approx([1., 1., 2.])


def test_a_fixed_cabinet_against_the_wall_blocks_a_low_wall_object_but_not_a_high_one():
    cabinet = {"id": "cabinet", "size_local_m": [.4, 1., 1.], "bottom_center_m": [.2, 1.5, 0.], "yaw_rad": 0.}
    room = {**ROOM, "height_m": 3., "fixed_objects": [cabinet]}
    xy, _, _ = _wall_spread([(1, 2), (1, 2)], [.5, 1.5], room=room)
    assert xy == pytest.approx(np.array([[.05, 2.5], [.05, 1.5]]))


def test_a_keep_yaw_wall_object_keeps_its_yaw_but_still_sits_flush():
    xy, yaw, _ = spread_grid_xy(_logits(1)[None], torch.zeros(1, 16, 2), 4, np.array([[.1, .8, .6]]), np.array([math.pi / 2]),
                                np.array([[0., 0., 1.]]), ROOM, requests=[{"id": "w", "support_parent": "wall"}], keep_yaw=[True])
    assert yaw == pytest.approx([math.pi / 2]) and xy == pytest.approx(np.array([[.4, 1.5]]))  # turned: .8 deep


def test_a_wall_object_predicted_near_the_floor_hangs_at_eye_level_after_the_floor_furniture():
    # mirror (.6 m^2) and wardrobe (.54 m^2) both want cell 1 = (.5, 1.5); before: the mirror stayed on the floor
    # (z = 0), went first by area and pushed the wardrobe to (1.5, 2.5)
    xy, yaw, z = spread_grid_xy(torch.stack((_logits(1, 13), _logits(1, 6))), torch.zeros(2, 16, 2), 4,
                                np.array([[.3, 2., .6], [.9, .6, 2.]]), np.zeros(2), np.zeros((2, 3)), {**ROOM, "height_m": 3.},
                                requests=[{"id": "mirror", "support_parent": "wall"},
                                          {"id": "wardrobe", "support_parent": "floor"}])
    assert xy == pytest.approx(np.array([[3.85, 1.5], [.5, 1.5]])) and yaw == pytest.approx([math.pi, 0.])
    assert z == pytest.approx([1.2, 0.])  # centred at 1.5 m
    _, _, z = _wall_spread([(1,)], [.1], room={**ROOM, "height_m": 1.5})
    assert z == pytest.approx([.9])  # under a low ceiling


def test_a_wall_object_wider_or_deeper_than_the_room_is_centred_along_that_axis():
    xy, _, _ = _wall_spread([(1,)], [1.], size=(.1, 5., .6))  # 5 m wide on the x = 0 wall of a 4 m room
    assert xy == pytest.approx(np.array([[.05, 2.]]))
    xy, _, _ = _wall_spread([(1,)], [1.], size=(5., .5, .5))  # 5 m deep
    assert xy == pytest.approx(np.array([[2., 1.5]]))


def test_validate_scene_checks_a_wall_support_against_the_room_boundary():
    clock = {"id": "clock", "category": "clock", "support_parent": "wall"}
    condition = request_to_condition(_request(clock))

    def validated(cond, x):
        return validate_scene(cond, [{"id": "clock", "target_size_local_m": [.1, .4, .4], "bottom_center_m": [x, 1.5, 1.5],
                                      "yaw_rad": 0.}])

    def statuses(result):
        return [c["status"] for c in result["checks"] if c["code"] == "wall_support"]
    flush = validated(condition, .05)  # back on the x = 0 wall; before: support_parent_missing violation
    assert flush["ok"] and statuses(flush) == ["pass"]
    away = validated(condition, 1.)
    assert not away["ok"] and statuses(away) == ["violation"]
    unknown = validated(request_to_condition(_request(clock), room_size_semantics="reference_extent"), .05)
    assert "wall_unknown" in unknown["unknown_checks"] and statuses(unknown) == []
    # a hard on constraint to the wall is the same declaration (before: also a constraint_reference violation)
    on = {**condition, "objects": [{k: v for k, v in condition["objects"][0].items() if k != "support_parent"}],
          "constraints": [{"type": "on", "object_id": "clock", "target_id": "wall"}]}
    on = validated(on, .05)
    assert on["ok"] and statuses(on) == ["pass", "pass"]


# ---- 4. end to end: five benchmark rooms --------------------------------------------------------------------------

def _synthetic_layout(condition, seed):
    """serialize_predictions (spread) over seeded synthetic heads: the decode predict uses, without Qwen.
    Fixed z (declared floor) replaces the position head as in the model's forward."""
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=10**8, max_objects=10**6)
    n, g = len(condition["objects"]), torch.Generator().manual_seed(seed)
    position = torch.where(batch["fixed_position_mask"], batch["fixed_position_normalized"], torch.rand(1, n, 3, generator=g))
    predictions = {"position_normalized": position, "size": .05 + .6 * torch.rand(1, n, 3, generator=g),
                   "yaw_logits": torch.randn(1, n, 12, generator=g), "yaw_residuals": torch.zeros(1, n, 12),
                   "position_cell_logits": 3 * torch.randn(1, n, 256, generator=g),
                   "position_cell_residuals": torch.zeros(1, n, 256, 2)}
    return serialize_predictions(predictions, batch, grid_decode="spread")[0]


def test_five_benchmark_rooms_hand_off_every_object_with_its_declared_placement(tmp_path):
    seen = 0
    for seed, path in enumerate(sorted(SCENES.glob("*.json"))):
        scene = json.loads(path.read_text())
        condition = request_to_condition(benchmark_request(scene))
        layout = _synthetic_layout(condition, seed)
        downstream = layout_to_roomgenbench(condition, layout)
        renamed = _renamed(scene)
        expected = {renamed[o["id"]]: renamed.get(o["place_id"], o["place_id"]) for o in scene["objects"]}
        assert [o["id"] for o in downstream["objects"]] == list(expected)
        boxes = {o["id"]: o for o in layout["objects"]}
        width, length, height = condition["room"]["floor_polygon_xy_m"][2] + [condition["room"]["height_m"]]
        for obj in downstream["objects"]:
            assert obj["place_id"] == expected[obj["id"]] and obj["support_status"] == "declared"
            assert obj["place"] == (expected[obj["id"]] if expected[obj["id"]] in ("floor", "wall") else "on_object")
            box = boxes[obj["id"]]
            (x, y, z), (depth, _, h) = box["bottom_center_m"], box["target_size_local_m"]
            if obj["place"] == "floor":
                assert z == 0.
            elif obj["place"] == "on_object":  # on the parent's top, or (round 10) inside it more than 5 cm below the top
                parent = boxes[obj["place_id"]]
                top = parent["bottom_center_m"][2] + parent["target_size_local_m"][2]
                assert z == pytest.approx(top) or parent["bottom_center_m"][2] <= z < top - .05
            else:  # round 10: the axis across the wall may be local Y (a box predicted in the swapped form)
                c, s, side = abs(math.cos(box["yaw_rad"])), abs(math.sin(box["yaw_rad"])), box["target_size_local_m"][1]
                hx, hy = (c * depth + s * side) / 2, (s * depth + c * side) / 2
                gap = min(x - hx, width - x - hx, y - hy, length - y - hy)
                assert abs(gap) < 1e-6 and -1e-9 <= z <= height - h + 1e-9
        seen += len(downstream["objects"])
        if path.stem == "bathroom":  # the downstream explicit-placement policy accepts the handoff
            handoff = export_handoff(tmp_path / "handoff", condition, layout)
            receipt = assemble_handoff(handoff, tmp_path / "assembled", roomgenbench_root=REFERENCE, require_placement=True)
            assert receipt["requested_objects"] == 55 and receipt["counts"] == {"box": 55}
            assert sum(o["place"] == "wall" for o in receipt["objects"]) == 32
    assert seen == 313
