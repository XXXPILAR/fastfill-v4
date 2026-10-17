import copy

import torch

from fastfill.v2.batch import TinyTokenizer, collate_samples
from fastfill.v2.text_sft import collate_text_samples, build_text_model, text_loss, main, load_text_model, generate_text
from fastfill.v2.tests.test_batch import sample


def test_text_and_structured_condition_information_is_identical():
    tokenizer = TinyTokenizer()
    s = sample(1)
    structured = collate_samples([s], tokenizer)
    text = collate_text_samples([s], tokenizer)
    n = int(structured["attention_mask"].sum())
    assert torch.equal(text["input_ids"][0, :n], structured["input_ids"][0, :n])
    assert (text["labels"][0, :n] == -100).all()
    assert (text["labels"][0, n:] != -100).any()


def test_tiny_text_ce_is_differentiable_and_missing_labels_rejected():
    tokenizer = TinyTokenizer()
    s = sample(1)
    batch = collate_text_samples([s], tokenizer)
    model = build_text_model("tiny", hidden_size=32)
    loss = text_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.embedding.weight.grad.abs().sum() > 0
    s["validity"]["yaw"] = [False]
    import pytest
    with pytest.raises(ValueError, match="complete"):
        collate_text_samples([s], tokenizer)


def test_text_cli_offline_optimizer_checkpoint_and_generation(tmp_path):
    import json
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(sample(1)) + "\n")
    output = tmp_path / "text"
    main(["--data", str(manifest), "--output", str(output), "--backbone", "tiny",
          "--dry-run", "--hidden-size", "32", "--batch-size", "1"])
    config = json.loads((output / "text_config.json").read_text())
    assert config["offline_smoke"] is True
    assert len(json.loads((output / "text_training_log.json").read_text())) == 1
    model, tokenizer = load_text_model(output)
    assert isinstance(generate_text(model, tokenizer, sample(1)["condition"], max_new_tokens=5), str)


def test_text_training_rejects_validation_manifest_and_unsafe_dry_run(tmp_path):
    import json
    import pytest
    s = sample(1)
    s["provenance"]["split"] = "validation"
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(json.dumps(s) + "\n")
    with pytest.raises(ValueError, match="training.*split"):
        main(["--data", str(manifest), "--output", str(tmp_path / "out"), "--backbone", "tiny", "--dry-run"])
    with pytest.raises(SystemExit):
        main(["--data", str(manifest), "--output", str(tmp_path / "out"), "--dry-run"])


def test_text_training_rejects_existing_or_source_output_before_loading(tmp_path):
    import pytest
    from fastfill.v2.io import SOURCE_ROOT
    missing = tmp_path / "no-such-dataset.jsonl"
    with pytest.raises(FileExistsError):
        main(["--data", str(missing), "--output", str(tmp_path), "--backbone", "tiny", "--dry-run"])
    with pytest.raises(ValueError, match="read-only"):
        main(["--data", str(missing), "--output", str(SOURCE_ROOT / "never-write"), "--backbone", "tiny", "--dry-run"])


def test_text_preflights_entire_cohort_before_loading_model(tmp_path, monkeypatch):
    import json
    import pytest
    import fastfill.v2.text_sft as module
    good, incomplete = sample(1), copy.deepcopy(sample(1))
    incomplete["validity"]["yaw"] = [False]
    path = tmp_path / "mixed.jsonl"
    path.write_text(json.dumps(good) + "\n" + json.dumps(incomplete) + "\n")
    def forbidden_load(*args, **kwargs):
        raise AssertionError("model must not load before complete-label preflight")
    monkeypatch.setattr(module, "build_text_model", forbidden_load)
    with pytest.raises(ValueError, match="complete"):
        main(["--data", str(path), "--output", str(tmp_path / "out"), "--backbone", "tiny", "--dry-run"])
    assert not (tmp_path / "out").exists()
