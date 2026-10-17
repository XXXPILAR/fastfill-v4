"""Failed checkpoint generations remain in the measured latency population."""
from copy import deepcopy
import json
from types import SimpleNamespace
import time

import pytest

from fastfill.v2 import evaluate
from fastfill.v2.tests.test_runtime import condition, layout


def dataset(path):
    row = {"schema_version": "fastfill.v2", "condition": condition(), "target": layout(),
           "validity": {"position": [[True] * 3], "size": [[True] * 3], "yaw": [True]},
           "provenance": {"source": "synthetic", "house_id": "h", "scene_id": "one", "split": "test"}}
    rows = [row, {**deepcopy(row), "provenance": {**row["provenance"], "scene_id": "two"}}]
    path.write_text("".join(json.dumps(sample) + "\n" for sample in rows))


@pytest.mark.parametrize("failure", ["generation", "schema"])
def test_failed_generation_or_schema_time_is_retained_in_all_request_latency(tmp_path, monkeypatch, failure):
    data = tmp_path / "requests.jsonl"
    dataset(data)
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **k: SimpleNamespace(config=SimpleNamespace(backbone="tiny")))
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda *a, **k: object())
    calls = 0
    def generate(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(.025)
            if failure == "generation":
                raise ValueError("generation failed after work")
            return {"schema_version": "fastfill.v2", "objects": []}
        return layout()
    monkeypatch.setattr(evaluate, "predict_layout", generate)
    output = tmp_path / "evaluation"
    report = evaluate.run_evaluation(data, output, checkpoint=tmp_path / "mock.pt")
    outcomes = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    assert report["requests"] == 2 and report["inference_failed_requests"] == 1
    assert outcomes[1]["fastfill_latency_ms"] >= 20.
    assert outcomes[1]["fastfill_latency_status"] == "failure"
    assert outcomes[0]["fastfill_latency_status"] == "success"
    assert report["latency_ms"]["fastfill_observed_requests"] == 2
    assert report["latency_ms"]["fastfill_by_outcome"]["failure"]["requests"] == 1
    assert report["latency_ms"]["fastfill_by_outcome"]["success"]["requests"] == 1
    assert report["latency_ms"]["fastfill_p95"] >= 19.


def test_supplied_predictions_do_not_invent_generation_latency(tmp_path):
    data, predictions = tmp_path / "requests.jsonl", tmp_path / "predictions.jsonl"
    dataset(data)
    predictions.write_text(json.dumps(layout()) + "\n" + '{"bad":true}\n')
    output = tmp_path / "evaluation"
    report = evaluate.run_evaluation(data, output, predictions=predictions)
    outcomes = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
    assert all(row["fastfill_latency_ms"] is None for row in outcomes)
    assert all(row["fastfill_latency_status"] == "unobserved" for row in outcomes)
    assert report["latency_ms"]["fastfill_observed_requests"] == 0
    assert report["latency_ms"]["fastfill_p95"] is None


def test_requests_over_the_object_budget_are_counted_as_over_capacity(tmp_path, monkeypatch):
    from fastfill.v2.batch import collate_samples
    data = tmp_path / "requests.jsonl"
    dataset(data)
    monkeypatch.setattr(evaluate, "load_model", lambda *a, **k: SimpleNamespace(config=SimpleNamespace(backbone="tiny")))
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda *a, **k: object())
    def generate(model, tokenizer, condition, *args, **kwargs):  # the real budget check, here of a 0-object model
        collate_samples([{"condition": condition}], tokenizer, max_objects=0)
    monkeypatch.setattr(evaluate, "predict_layout", generate)
    report = evaluate.run_evaluation(data, tmp_path / "evaluation", checkpoint=tmp_path / "mock.pt")
    assert report["inference_failed_requests"] == report["over_capacity_requests"] == 2
