"""OptiScene-style LLM modes (llm_structured): protocol-only input, a train demonstration, tagged answers, input-only
harness. llm_baseline.py, which today's paid prompt/harness runs are bound to, stays byte-identical to commit 510f1e0."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request

import pytest

from fastfill.v2 import llm_baseline, llm_structured as S
from fastfill.v2.evaluate import project_minimal, run_evaluation
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows

DATA = Path(S.__file__).parents[2] / "outputs/fastfill_v2/rebuild-main-20261007b/main"
DEMO_SCENE = "il3d:b580392a-86d4-11f0-a478-60cf84ae2082"


def rows():
    return [p for p in map(project_minimal, real_rows()) if p is not None]


def design(row, room_type="...", exact=False):
    """The row's target as the [Design] JSON (mm / 0.1 deg unless exact), objects in request order."""
    q = (lambda value, digits: value) if exact else round
    targets = {t["id"]: t for t in row["target"]["objects"]}
    objects = []
    for o in row["condition"]["objects"]:
        t = targets[o["id"]]
        (depth, width, height), (x, y, z) = t["target_size_local_m"], t["bottom_center_m"]
        objects.append({"id": o["id"], "width_m": q(width, 3), "depth_m": q(depth, 3), "height_m": q(height, 3),
                        "x": q(x, 3), "y": q(y, 3), "z": q(z, 3), "facing_deg": q(math.degrees(t["yaw_rad"]) % 360, 1)})
    return '{"room type": ' + json.dumps(room_type) + ', "objects": [\n' + ",\n".join(json.dumps(o) for o in objects) + "\n]}"


def tagged(row, objects=None, exact=False):
    body = design(row, exact=exact) if objects is None else json.dumps({"objects": objects})
    return ('<reasoning>\n[Reason]\nBraces {"like": "these"} in the reasoning are ignored.\n[/Reason]\n</reasoning>\n\n'
            f"<answer>\n[Design]\n```json\n{body}\n```\n[/Design]\n</answer>")


def overlapping(row):
    objects = json.loads(design(row))["objects"]
    objects[1].update(x=objects[0]["x"], y=objects[0]["y"], z=objects[0]["z"])  # two objects in one place
    return objects


def env(tmp_path, rpm=""):
    path = tmp_path / "api.env"
    path.write_text(f"OPENAI_BASE_URL=https://relay/v1\nOPENAI_API_KEY=k\nOPENAI_MODEL=m\n{rpm}")
    return path


def write(tmp_path, data):
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in data))
    return path


def test_llm_baseline_is_byte_identical_to_510f1e0():
    """autorun reuses today's prompt/harness runs only while sha256(llm_baseline.py) matches their summary.json; any
    edit deletes them and pays for new, differently sampled answers. Back them up before changing this file."""
    path = Path(llm_baseline.__file__)
    try:
        old = subprocess.run(["git", "-C", str(path.parent), "show", "510f1e0:fastfill/v2/llm_baseline.py"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("commit 510f1e0 is not in this checkout")
    assert path.read_bytes() == old


def test_structured_input_is_the_protocol_fields_only(tmp_path):
    data, seen = rows(), []
    S.run(write(tmp_path, data), tmp_path / "s", env(tmp_path), ask=lambda env, model, messages: seen.append(messages) or tagged(data[0]))
    assert all(m[0] == {"role": "system", "content": S.STRUCTURED} for m in seen)
    for row in data:
        user = next(m[1]["content"] for m in seen if m[1]["content"] == S.sections(row["condition"]))
        room = row["condition"]["room"]
        listed = [{k: o[k] for k in ("id", "category", "description")} for o in row["condition"]["objects"]]
        protocol = {"room": {k: room[k] for k in ("room_type", "floor_polygon_xy_m", "floor_z_m", "height_m")}, "objects": listed}
        noisy = {**row["condition"], "room": {**room, "fixed_objects": [{"id": "f"}], "openings": [1]}, "constraints": [{"type": "near"}],
                 "objects": [{**o, "target_size_local_m": [9, 9, 9], "attributes": {}} for o in row["condition"]["objects"]]}
        assert user == S.sections(protocol) == S.sections(noisy)  # reads nothing else
        kind, size, objects = user.split("\n\n")  # the three sections and nothing else: no hint keyed on category
        assert kind == f"[Task Room Type]\n{room['room_type']}\n[/Task Room Type]"
        assert objects == "[Task Objects]\n[\n" + ",\n".join(json.dumps(o, ensure_ascii=False) for o in listed) + "\n]\n[/Task Objects]"
        head, extent, polygon, floor, tail = size.split("\n")
        assert (head, tail) == ("[Task Room Size]", "[/Task Room Size]")
        assert re.fullmatch(r"width \(x\) x length \(y\) x height \(z\): \d+\.\d{3} m x \d+\.\d{3} m x (\d+\.\d{3} m|unknown)", extent)
        assert polygon == "floor polygon (x, y): " + json.dumps([[round(x, 3), round(y, 3)] for x, y in room["floor_polygon_xy_m"]])
        assert re.fullmatch(r"floor z: (-?\d+\.\d{3} m|unknown)", floor)
        for t in row["target"]["objects"]:
            assert not any(f"{v:.3f}" in user for v in t["bottom_center_m"][:2] + t["target_size_local_m"])
            assert f"{math.degrees(t['yaw_rad']):.1f}" not in user
        assert row["provenance"]["split"] == "validation" and row["provenance"]["scene_id"] not in S.STRUCTURED
    declared = {"room": {"floor_polygon_xy_m": [[0, 0], [1, 0], [1, 1], [0, 1]], "floor_z_m": None, "height_m": None},
                "objects": [{"id": "a", "category": "lamp", "description": "", "support_parent": "b"}]}
    assert S.sections(declared) == (
        "[Task Room Type]\nunknown\n[/Task Room Type]\n\n[Task Room Size]\n"
        "width (x) x length (y) x height (z): 1.000 m x 1.000 m x unknown\nfloor polygon (x, y): [[0, 0], [1, 0], [1, 1], [0, 1]]\n"
        'floor z: unknown\n[/Task Room Size]\n\n[Task Objects]\n[\n{"id": "a", "category": "lamp", "description": "", "support_parent": "b"}\n]\n[/Task Objects]')
    # the template holds no size table: besides the demonstration, only the guidance numbers and "0 / 90 faces ..."
    assert set(re.findall(r"\d+(?:\.\d+)?", S.STRUCTURED.replace(S.DEMONSTRATION, ""))) == {"0", "90", *map(str, range(1, 10))}


@pytest.mark.skipif(not (DATA / "train.jsonl").is_file(), reason="rebuild-main-20261007b not on this machine")
def test_demonstration_is_rebuilt_from_its_train_row():
    needle = f'"scene_id":"{DEMO_SCENE}"'.encode()
    with (DATA / "train.jsonl").open("rb") as stream:
        row = json.loads(next(line for line in stream if needle in line))
    assert row["provenance"]["split"] == "train"
    for split in ("validation", "test"):
        assert DEMO_SCENE.encode() not in (DATA / f"{split}.jsonl").read_bytes()
    demo = project_minimal(row)
    text = S.sections(demo["condition"], "Example") + "\n\n[Example Reason]\n"
    assert S.DEMONSTRATION.startswith(text)
    shown = design(demo, "bedroom")
    assert S.DEMONSTRATION.endswith(f"\n[/Example Reason]\n\n[Example Design]\n{shown}\n[/Example Design]")
    reason = S.DEMONSTRATION[len(text):-len(f"\n[/Example Reason]\n\n[Example Design]\n{shown}\n[/Example Design]")]
    assert not re.search(r"\d", reason)  # the hand-written reason gives no sizes or positions
    layout, found = S.check_answer(f"<answer>[Design]{shown}[/Design]</answer>", demo["condition"])
    assert layout is not None and found == []


def test_same_rooms_as_the_prompt_modes_and_one_scorer(tmp_path):
    data = rows()
    answers = {S.sections(r["condition"]): tagged(r, exact=True) for r in data}
    summary = S.run(write(tmp_path, data), tmp_path / "s", env(tmp_path), ask=lambda env, model, messages: answers[messages[1]["content"]])
    assert (summary["failed"], summary["api_calls"], summary["mode"]) == (0, len(data), "structured")
    assert summary["prompt_sha256"] == hashlib.sha256(S.STRUCTURED.encode()).hexdigest()
    report = run_evaluation(tmp_path / "s/rows.jsonl", tmp_path / "eval", predictions=tmp_path / "s/predictions.jsonl")
    reference = report["model"]["reference"]
    assert reference["bottom_center_error_m"]["mean"] < 1e-6 and reference["log_size_error"]["mean"] < 1e-6
    assert report["collapse"]["predicted"] == report["collapse"]["ground_truth"]
    llm_baseline.run(write(tmp_path, data), tmp_path / "p", env(tmp_path), ask=lambda *a: "{}")
    rooms = (tmp_path / "p/rows.jsonl").read_bytes()
    assert (tmp_path / "s/rows.jsonl").read_bytes() == rooms  # the rows the prompt modes answered
    again = S.run(tmp_path / "p/rows.jsonl", tmp_path / "again", env(tmp_path), ask=lambda env, model, messages: answers[messages[1]["content"]])
    assert (tmp_path / "again/rows.jsonl").read_bytes() == rooms and again["data_sha256"] == hashlib.sha256(rooms).hexdigest()


def test_overlap_is_reported_and_the_repaired_answer_kept(tmp_path):
    row, seen = rows()[0], []
    bad = overlapping(row)
    def ask(env, model, messages):
        seen.append(messages)
        return tagged(row, bad) if len(seen) == 1 else tagged(row)
    summary = S.run(write(tmp_path, [row]), tmp_path / "h", env(tmp_path), mode="structured-harness", ask=ask)
    assert summary["api_calls"] == 2 and summary["repairs"] == 2 and summary["failed"] == 0
    assert "overlap" in seen[1][-1]["content"] and seen[1][-2] == {"role": "assistant", "content": tagged(row, bad)}
    prediction = json.loads((tmp_path / "h/predictions.jsonl").read_text())
    assert (prediction["repair_rounds"], prediction["checks_final"], prediction["first_answer_valid"]) == (1, 0, True)
    assert prediction["checks_first"] >= 1 and prediction["layout"] == S.check_answer(tagged(row), row["condition"])[0]
    once = S.run(write(tmp_path, [row]), tmp_path / "p", env(tmp_path), mode="structured", ask=lambda *a: tagged(row, bad))
    assert once["api_calls"] == 1 and once["repairs"] == 0 and once["mean_checks_final"] >= 1


@pytest.mark.parametrize("repair", ["unreadable", "request fails"])
def test_a_bad_repair_keeps_the_last_valid_layout(tmp_path, repair):
    row, seen = rows()[0], []
    bad = overlapping(row)
    def ask(env, model, messages):
        seen.append(messages)
        if len(seen) == 1:
            return tagged(row, bad)
        if repair == "request fails":
            raise ConnectionError("relay down")
        return "Sorry, no design this time."
    summary = S.run(write(tmp_path, [row]), tmp_path / "h", env(tmp_path), mode="structured-harness", ask=ask)
    prediction = json.loads((tmp_path / "h/predictions.jsonl").read_text())
    assert summary["failed"] == 0 and prediction["layout"] == S.check_answer(tagged(row, bad), row["condition"])[0]
    assert prediction["attempts"] == 1 and prediction["checks_final"] == prediction["checks_first"] >= 1
    assert (summary["api_calls"], prediction["repair_rounds"]) == ((2, 0) if repair == "request fails" else (3, 2))


def test_unparseable_first_answer_is_asked_again_with_the_error(tmp_path):
    row, seen = rows()[0], []
    def ask(env, model, messages):
        seen.append(messages)
        return "I would rather describe the room." if len(seen) == 1 else tagged(row)
    S.run(write(tmp_path, [row]), tmp_path / "h", env(tmp_path), mode="structured-harness", ask=ask)
    prediction = json.loads((tmp_path / "h/predictions.jsonl").read_text())
    assert prediction["layout"] is not None and (prediction["attempts"], prediction["repair_rounds"]) == (1, 1)
    assert prediction["first_answer_valid"] is False and "could not be read" in seen[1][-1]["content"]
    seen.clear()
    S.run(write(tmp_path, [row]), tmp_path / "s", env(tmp_path), mode="structured", ask=ask)
    prediction = json.loads((tmp_path / "s/predictions.jsonl").read_text())
    assert prediction["attempts"] == 2 and len(seen[1]) == 2  # without the harness: asked again from scratch


def test_checks_name_numbers_ids_floor_ceiling_and_supports():
    condition = {"room": {"floor_polygon_xy_m": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z_m": 0., "height_m": 3.},
                 "objects": [{"id": "t", "category": "table", "description": "", "support_parent": "floor"},
                             {"id": "l", "category": "lamp", "description": "", "support_parent": "t"}]}
    table = {"id": "t", "width_m": 1., "depth_m": 1., "height_m": 3.5, "x": 2., "y": 2., "z": 0., "facing_deg": 0}
    lamp = {"id": "l", "width_m": .2, "depth_m": .2, "height_m": .4, "x": 3.5, "y": 2., "z": 0., "facing_deg": 90}
    layout, found = S.check_answer(json.dumps({"objects": [{**table, "width_m": -1, "x": None}]}), condition)
    assert layout is None and found == ["answer must hold each requested id exactly once: duplicated [], unknown [], missing ['l']",
                                        "t: width_m must be a positive number, got -1", "t: x must be a finite number, got None"]
    layout, found = S.check_answer(json.dumps({"objects": [table, lamp]}), condition)
    assert layout is not None and any("above the 3.00 m ceiling" in f for f in found)
    assert any(f.startswith("l (lamp) must rest on t (table)") for f in found)
    on_top = {**lamp, "x": 2.2, "z": 3.5}
    assert not any("must rest" in f for f in S.check_answer(json.dumps({"objects": [table, on_top]}), condition)[1])
    for z, problem in ((-.5, "t (table) is 0.50 m below the floor"), (.3, "t (table) must stand on the floor (z = 0.00), not at z = 0.30")):
        low = {**table, "height_m": 1., "z": z}
        assert problem in S.check_answer(json.dumps({"objects": [low, {**on_top, "z": z + 1.}]}), condition)[1]
    crowded = {"room": condition["room"], "objects": [{"id": f"o{k}", "category": "box", "description": ""} for k in range(12)]}
    boxes = [{**table, "id": f"o{k}", "height_m": 3.5 if k == 11 else 1.} for k in range(12)]
    found = S.check_answer(json.dumps({"objects": boxes}), crowded)[1]
    assert len(found) == 1 + 66 and found[0] == "o11 (box) reaches 3.50 m, above the 3.00 m ceiling"  # 66 overlaps never hide it


class Relay:
    """urlopen stand-in: keyed by the request's first user text, the first answer is ``first[key]``, a repair ``later[key]``."""
    def __init__(self, first, later):
        self.first, self.later, self.bodies = first, later, []

    def __call__(self, request, timeout):
        self.bodies.append(request.data)
        messages = json.loads(request.data)["messages"]
        self.body = json.dumps({"choices": [{"message": {"content": (self.first if len(messages) == 2 else self.later)[messages[1]["content"]]}}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}).encode()
        return self

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self): return self.body


def test_structured_rows_record_token_usage(tmp_path, monkeypatch):
    data = rows()
    first = {S.sections(r["condition"]): tagged(r, [{"id": "nope"}]) for r in data}
    later = {S.sections(r["condition"]): tagged(r) for r in data}
    fake = Relay(first, later)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    summary = S.run(write(tmp_path, data), tmp_path / "s", env(tmp_path, "FASTFILL_LLM_RPM=600000\n"), mode="structured-harness", workers=1)
    body = json.loads(fake.bodies[0])
    assert body["messages"][0] == {"role": "system", "content": S.STRUCTURED} and len(body["messages"]) == 2
    assert {k: v for k, v in body.items() if k != "messages"} == llm_baseline.request_parameters({}, "m")
    assert summary["usage"] == {"calls_reported": 2 * len(data), "prompt_tokens": 200 * len(data),
                                "completion_tokens": 40 * len(data), "total_tokens": 240 * len(data)}
    predictions = [json.loads(line) for line in (tmp_path / "s/predictions.jsonl").read_text().splitlines()]
    assert all(p["usage"]["total_tokens"] == 240 and p["first_answer_valid"] is False for p in predictions)
