"""Results-first workflow: locked FastFill sizes, LLM placement, checked repair, fallback (scripted LLM, no network)."""
import json
import math

import pytest

from fastfill.v2 import llm_workflow as W
from fastfill.v2.evaluate import project_minimal, run_evaluation
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows


def env_file(tmp_path):
    path = tmp_path / "llm.env"
    path.write_text("OPENAI_BASE_URL=http://unused\nOPENAI_API_KEY=unused\nOPENAI_MODEL=scripted\n")
    return path


def data_file(tmp_path, rows):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def answer(placements, *, with_sizes=None):
    """A tagged answer; placements: id -> (x, y, z, facing_deg); with_sizes adds (ignored) size fields."""
    objects = [{"id": i, "x": x, "y": y, "z": z, "facing_deg": f, **(with_sizes or {})} for i, (x, y, z, f) in placements.items()]
    return "<reasoning>\n[Reason]\nbig things first {\"not\": \"json\"}\n[/Reason]\n</reasoning>\n<answer>\n[Design]\n" + \
        json.dumps({"objects": objects}) + "\n[/Design]\n</answer>"


def reference(row):
    return {t["id"]: (*t["bottom_center_m"], math.degrees(t["yaw_rad"])) for t in row["target"]["objects"]}


def scripted(*replies):
    seen = []

    def ask(env, model, messages):
        seen.append(messages)
        return replies[min(len(seen), len(replies)) - 1]  # the last reply repeats
    return ask, seen


def test_locked_sizes_are_shown_and_answers_cannot_change_them(tmp_path):
    row = project_minimal(real_rows()[0])
    ask, seen = scripted(answer(reference(row), with_sizes={"width_m": 9., "depth_m": 9., "height_m": 9.}))
    W.run(data_file(tmp_path, [real_rows()[0]]), tmp_path / "out", env_file(tmp_path), sizes="ground_truth", ask=ask, repairs=0)
    system, request = seen[0][0]["content"], seen[0][1]["content"]
    assert seen[0][0]["role"] == "system" and "given and fixed" in system and '"width_m"' not in system.split("[Design]")[1]
    first = row["target"]["objects"][0]
    assert f'"width_m": {round(first["target_size_local_m"][1], 3)}' in request
    layout = json.loads((tmp_path / "out/predictions.jsonl").read_text().splitlines()[0])["layout"]
    assert [o["target_size_local_m"] for o in layout["objects"]] == [t["target_size_local_m"] for t in row["target"]["objects"]]


def test_repair_feedback_lists_problems_and_the_fewest_problem_layout_is_kept(tmp_path):
    row = project_minimal(real_rows()[0])
    good = reference(row)
    stacked = {i: (good[next(iter(good))][0], good[next(iter(good))][1], p[2], p[3]) for i, p in good.items()}  # all at one spot
    ask, seen = scripted(answer(stacked), answer(good), answer(stacked))
    summary = W.run(data_file(tmp_path, [real_rows()[0]]), tmp_path / "out", env_file(tmp_path), sizes="ground_truth",
                    ask=ask, repairs=2)
    result = json.loads((tmp_path / "out/predictions.jsonl").read_text().splitlines()[0])
    assert "overlap" in seen[1][-1]["content"] and seen[1][-1]["role"] == "user"
    trace = result["problems_trace"]
    assert trace[0] > 0 and result["problems_final"] == min(t for t in trace if t is not None)
    assert result["source"] == "llm" and summary["llm_layouts"] == 1 and summary["api_calls"] == len(seen)


def test_unreadable_answers_fall_back_to_fastfill_and_are_counted(tmp_path, monkeypatch):
    row = project_minimal(real_rows()[0])
    monkeypatch.setattr(W, "_fastfill", lambda checkpoint, device: lambda condition: {"schema_version": "fastfill.v2", "objects": [
        {"id": t["id"], "target_size_local_m": t["target_size_local_m"], "bottom_center_m": t["bottom_center_m"], "yaw_rad": t["yaw_rad"]}
        for t in row["target"]["objects"]]})
    ask, seen = scripted("no json here")
    summary = W.run(data_file(tmp_path, [real_rows()[0]]), tmp_path / "out", env_file(tmp_path), sizes="checkpoint",
                    checkpoint="unused", ask=ask, retries=1)
    result = json.loads((tmp_path / "out/predictions.jsonl").read_text().splitlines()[0])
    assert result["source"] == "fastfill_fallback" and result["layout"] is not None and len(seen) == 2
    assert summary["fallback"] == 1 and summary["failed"] == 0


def test_without_sizes_the_llm_chooses_them_and_without_a_layout_the_row_fails(tmp_path):
    row = project_minimal(real_rows()[0])
    placements = reference(row)
    ask, _ = scripted(answer(placements, with_sizes={"width_m": .5, "depth_m": .4, "height_m": 1.}))
    W.run(data_file(tmp_path, [real_rows()[0]]), tmp_path / "a", env_file(tmp_path), sizes="none", ask=ask, repairs=0)
    layout = json.loads((tmp_path / "a/predictions.jsonl").read_text().splitlines()[0])["layout"]
    assert all(o["target_size_local_m"] == [.4, .5, 1.] for o in layout["objects"])
    ask, _ = scripted("still no json")
    summary = W.run(data_file(tmp_path, [real_rows()[0]]), tmp_path / "b", env_file(tmp_path), sizes="none", ask=ask)
    assert summary["failed"] == 1 and json.loads((tmp_path / "b/predictions.jsonl").read_text())["layout"] is None


def test_full_condition_renders_doors_and_flags_objects_blocking_them():
    row = real_rows()[0]
    room = {**row["condition"]["room"], "openings": [{"type": "door", "wall_index": 0, "position_on_wall": .5,
                                                     "width_m": .9, "height_m": 2.}]}
    condition = {**row["condition"], "room": room}
    (ax, ay), (bx, by) = room["floor_polygon_xy_m"][0], room["floor_polygon_xy_m"][1]
    text = W.request_text(condition, None)
    assert "[Task Doors and Windows]" in text and json.dumps([round((ax + bx) / 2, 3), round((ay + by) / 2, 3)]) in text
    assert "front of every door" in W.instructions(condition, True)
    first = row["target"]["objects"][0]
    inward = [(ax + bx) / 2, (ay + by) / 2]
    blocking = {"schema_version": "fastfill.v2", "objects": [
        {"id": t["id"], "target_size_local_m": t["target_size_local_m"], "yaw_rad": t["yaw_rad"],
         "bottom_center_m": t["bottom_center_m"] if t is not first else [inward[0], inward[1], room["floor_z_m"]]}
        for t in row["target"]["objects"]]}
    problems = W.workflow_problems(blocking, condition)
    assert any("in front of a door" in p and first["id"] in p for p in problems)


def test_workflow_predictions_score_with_evaluate(tmp_path):
    rows = real_rows()
    replies = [answer(reference(project_minimal(r))) for r in rows if project_minimal(r) is not None]
    queue = iter(replies)
    W.run(data_file(tmp_path, rows), tmp_path / "out", env_file(tmp_path), sizes="ground_truth",
          ask=lambda env, model, messages: next(queue), repairs=0, workers=1)
    report = run_evaluation(tmp_path / "out/rows.jsonl", tmp_path / "eval", predictions=tmp_path / "out/predictions.jsonl")
    assert report["baseline"] == "supplied_predictions" and report["requests"] == len(replies)


def test_sizes_and_checkpoint_must_agree(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        W.run(data_file(tmp_path, real_rows()), tmp_path / "x", env_file(tmp_path), sizes="checkpoint", ask=lambda *a: "")
    with pytest.raises(ValueError, match="checkpoint"):
        W.run(data_file(tmp_path, real_rows()), tmp_path / "y", env_file(tmp_path), sizes="none", checkpoint="m", ask=lambda *a: "")


def test_tidy_snaps_near_contacts_and_pushes_small_wall_crossings_back_inside():
    from fastfill.v2.validation import validate_scene
    condition = {"schema_version": "fastfill.v2", "constraints": [], "objects": [
        {"id": "shelf", "category": "shelf", "description": "shelf"}, {"id": "lamp", "category": "lamp", "description": "lamp"},
        {"id": "sofa", "category": "sofa", "description": "sofa"}, {"id": "bed", "category": "bed", "description": "bed"}],
        "room": {"frame": "right_handed_z_up", "floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]], "boundary_known": True,
                 "floor_known": True, "floor_z_m": 0., "height_m": 3.}}
    box = lambda i, xyz, size: {"id": i, "target_size_local_m": size, "bottom_center_m": xyz, "yaw_rad": 0.}
    layout = {"schema_version": "fastfill.v2", "objects": [
        box("shelf", [1., 1., 0.], [.4, .8, .711]),
        box("lamp", [1., 1., .71], [.2, .2, .5]),       # rounded 1 mm into the shelf
        box("sofa", [3.7, 2., .01], [1., 2., .9]),       # 0.2 m through the wall, 1 cm above the floor
        box("bed", [2., -.5, 0.], [2., 1.6, .6])]}       # 1.3 m through the wall: a real error, left to the LLM
    tidied, moved = W.tidy(layout, condition)
    by_id = {o["id"]: o for o in tidied["objects"]}
    assert by_id["lamp"]["bottom_center_m"][2] == pytest.approx(.711) and by_id["sofa"]["bottom_center_m"][2] == 0.
    assert by_id["sofa"]["bottom_center_m"][0] <= 3.5 and by_id["bed"]["bottom_center_m"] == [2., -.5, 0.]
    violations = {(c["code"], tuple(c["object_ids"])) for c in validate_scene(condition, tidied["objects"])["checks"]
                  if c["status"] == "violation"}
    assert ("boundary", ("bed",)) in violations and not {v for v in violations if v[1] != ("bed",)}
    assert moved == 2 and layout["objects"][1]["bottom_center_m"][2] == .71  # the input layout is not modified
    assert any("bed (bed) crosses a wall" in p for p in W.workflow_problems(tidied, condition))
