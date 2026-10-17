"""LLM baseline: three-field prompt carries no target geometry; answers convert to layouts exactly."""
import json
import math

from fastfill.v2 import llm_baseline
from fastfill.v2.evaluate import project_minimal, run_evaluation
from fastfill.v2.tests.test_evaluate_fix20261006 import real_rows


def _answer_from_targets(row):
    objects = []
    for t in row["target"]["objects"]:
        depth, width, height = t["target_size_local_m"]
        x, y, z = t["bottom_center_m"]
        objects.append({"id": t["id"], "width_m": width, "depth_m": depth, "height_m": height,
                        "x": x, "y": y, "z": z, "facing_deg": math.degrees(t["yaw_rad"])})
    return "```json\n" + json.dumps({"objects": objects}) + "\n```"


def test_prompt_is_three_field_and_answers_round_trip_through_evaluate(tmp_path):
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:2]
    assert rows
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for row in rows:
        text = llm_baseline.prompt(row["condition"])
        assert all(f"{t['bottom_center_m'][0]:.6f}" not in text for t in row["target"]["objects"])
    (tmp_path / "api.env").write_text("export OPENAI_BASE_URL=http://unused\nOPENAI_API_KEY='k'\nOPENAI_MODEL=m\n")
    by_prompt = {llm_baseline.prompt(r["condition"]): _answer_from_targets(r) for r in rows}
    summary = llm_baseline.run(data, tmp_path / "llm", tmp_path / "api.env", ask=lambda env, model, messages: by_prompt[messages[0]["content"]])
    assert summary["failed"] == 0 and summary["model"] == "m"
    report = run_evaluation(tmp_path / "llm/rows.jsonl", tmp_path / "eval", predictions=tmp_path / "llm/predictions.jsonl")
    reference = report["model"]["reference"]
    assert reference["bottom_center_error_m"]["mean"] < 1e-6 and reference["log_size_error"]["mean"] < 1e-6
    assert report["collapse"]["predicted"] == report["collapse"]["ground_truth"]


def test_unparseable_answers_are_recorded_failures_not_dropped(tmp_path):
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:1]
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(rows[0]) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\n")
    summary = llm_baseline.run(data, tmp_path / "llm", tmp_path / "api.env", model="m", ask=lambda *a: "no json here")
    assert summary == {**summary, "rows": 1, "failed": 1}
    assert json.loads((tmp_path / "llm/predictions.jsonl").read_text())["layout"] is None


def test_harness_reports_concrete_problems_and_keeps_the_repaired_layout(tmp_path):
    row = [p for p in map(project_minimal, real_rows()) if p is not None][0]
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(row) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\nOPENAI_MODEL=m\n")
    good = _answer_from_targets(row)
    bad = json.loads(good.strip("`\njson"))
    bad["objects"][1].update(x=bad["objects"][0]["x"], y=bad["objects"][0]["y"], z=bad["objects"][0]["z"])  # stack two objects
    seen = []
    def ask(env, model, messages):
        seen.append(messages)
        return json.dumps(bad) if len(seen) == 1 else good
    summary = llm_baseline.run(data, tmp_path / "h", tmp_path / "api.env", mode="harness", ask=ask)
    assert summary["api_calls"] == 2 and summary["mean_problems_first"] >= 1
    assert "overlap" in seen[1][-1]["content"] and seen[1][-2]["role"] == "assistant"
    prediction = json.loads((tmp_path / "h/predictions.jsonl").read_text())
    assert prediction["repair_rounds"] == 1 and prediction["problems_final"] <= summary["mean_problems_first"]
    once = llm_baseline.run(data, tmp_path / "p", tmp_path / "api.env", mode="prompt", ask=lambda *a: json.dumps(bad))
    assert once["api_calls"] == 1


def test_rate_limiter_spaces_requests():
    limiter = llm_baseline.RateLimiter(rpm=600)  # 0.1 s apart
    import time
    start = time.monotonic()
    for _ in range(4):
        limiter.wait()
    assert time.monotonic() - start >= .29


def test_reasoning_effort_defaults_to_medium_and_is_recorded(tmp_path):
    assert llm_baseline.request_parameters({}, "gpt-6.1-sol") == {"model": "gpt-6.1-sol", "reasoning_effort": "medium"}
    assert llm_baseline.request_parameters({"FASTFILL_LLM_REASONING_EFFORT": ""}, "m") == {"model": "m"}
    row = [p for p in map(project_minimal, real_rows()) if p is not None][0]
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=https://relay/v1\nOPENAI_API_KEY=secret-test-key\nOPENAI_MODEL=gpt-6.1-sol\n")
    summary = llm_baseline.run(tmp_path / "rows.jsonl", tmp_path / "o", tmp_path / "api.env", ask=lambda *a: _answer_from_targets(row))
    assert summary["request_parameters"]["reasoning_effort"] == "medium" and summary["endpoint"] == "https://relay/v1/chat/completions"
    assert "secret-test-key" not in json.dumps(summary) + (tmp_path / "o/summary.json").read_text()


def test_duplicate_unknown_or_missing_ids_make_the_answer_a_failure(tmp_path):
    row = [p for p in map(project_minimal, real_rows()) if p is not None][0]
    good = json.loads(_answer_from_targets(row).strip("`\njson"))
    for objects in (good["objects"] + good["objects"][:1], good["objects"][1:] + [{**good["objects"][0], "id": "nope"}]):
        try:
            llm_baseline.to_layout(json.dumps({"objects": objects}), row["condition"])
        except ValueError as error:
            assert "exactly once" in str(error)
        else:
            raise AssertionError("an answer with a duplicated or unknown id was accepted")


def test_first_answer_compliance_is_reported_apart_from_the_retried_success(tmp_path):
    from fastfill.v2.io import fingerprint
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:2]
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\nOPENAI_MODEL=m\n")
    good = {llm_baseline.prompt(r["condition"]): json.loads(_answer_from_targets(r).strip("`\njson")) for r in rows}
    duplicated = lambda answer: json.dumps({"objects": answer["objects"] + answer["objects"][:1]})
    asked = {}

    def ask(env, model, messages):  # row 0: a duplicated id first, then a valid layout; row 1: valid at once
        text = messages[0]["content"]
        asked[text] = asked.get(text, 0) + 1
        return duplicated(good[text]) if text == llm_baseline.prompt(rows[0]["condition"]) and asked[text] == 1 else json.dumps(good[text])
    summary = llm_baseline.run(data, tmp_path / "a", tmp_path / "api.env", ask=ask)
    predictions = [json.loads(line) for line in (tmp_path / "a/predictions.jsonl").read_text().splitlines()]
    assert [p["attempts"] for p in predictions] == [2, 1] and summary["failed"] == 0 and summary["first_answer_invalid"] == 1
    assert {"layout", "error", "latency_s", "repair_rounds", "problems_first", "problems_final"} <= set(predictions[0])
    assert summary["data_sha256"] == fingerprint(data) and summary["implementation_sha256"] == fingerprint(llm_baseline.__file__)
    summary = llm_baseline.run(data, tmp_path / "b", tmp_path / "api.env", ask=lambda env, model, messages: duplicated(good[messages[0]["content"]]))
    predictions = [json.loads(line) for line in (tmp_path / "b/predictions.jsonl").read_text().splitlines()]
    assert [p["attempts"] for p in predictions] == [2, 2] and summary["failed"] == 2 and summary["first_answer_invalid"] == 2


def test_failed_requests_are_no_invalid_first_answer(tmp_path):
    """Row 0's first request fails in transport and its first answer is valid; row 1 never gets an answer."""
    import urllib.error
    rows = [p for p in map(project_minimal, real_rows()) if p is not None][:2]
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "api.env").write_text("OPENAI_BASE_URL=x\nOPENAI_API_KEY=k\nOPENAI_MODEL=m\n")
    answers, asked = {llm_baseline.prompt(r["condition"]): _answer_from_targets(r) for r in rows}, []

    def ask(env, model, messages):
        text = messages[0]["content"]
        asked.append(text)
        if text == llm_baseline.prompt(rows[1]["condition"]) or asked.count(text) == 1:
            raise urllib.error.URLError("connection reset")
        return answers[text]
    summary = llm_baseline.run(data, tmp_path / "a", tmp_path / "api.env", ask=ask)
    assert (summary["failed"], summary["first_answer_invalid"]) == (1, 0) and summary["unanswered"] == 1
    predictions = [json.loads(line) for line in (tmp_path / "a/predictions.jsonl").read_text().splitlines()]
    assert [(p["attempts"], p["first_answer_valid"]) for p in predictions] == [(2, True), (2, None)]


def test_requests_name_their_own_user_agent(monkeypatch):
    seen = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
    def urlopen(request, timeout):
        seen.update(request.headers)
        return Response()
    monkeypatch.setattr(llm_baseline.urllib.request, "urlopen", urlopen)
    env = {"OPENAI_BASE_URL": "https://relay/v1", "OPENAI_API_KEY": "k"}
    assert llm_baseline.chat(env, "m", [{"role": "user", "content": "hi"}]) == "OK"
    assert seen["User-agent"] == llm_baseline.USER_AGENT  # not urllib's default Python-urllib, which the relay blocks


def test_a_truncated_relay_response_is_retried(monkeypatch):
    import http.client
    calls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"{}")
            return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode()
    monkeypatch.setattr(llm_baseline.urllib.request, "urlopen", lambda request, timeout: calls.append(1) or Response())
    monkeypatch.setattr(llm_baseline.time, "sleep", lambda s: None)
    env = {"OPENAI_BASE_URL": "https://relay/v1", "OPENAI_API_KEY": "k"}
    assert llm_baseline.chat(env, "m", [{"role": "user", "content": "hi"}]) == "OK" and len(calls) == 2
