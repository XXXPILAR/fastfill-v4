"""Bounded offline regressions for evaluation, merge artifacts, and download exits."""
import importlib.util
import json
import runpy
import socket
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def denied(*args, **kwargs):
        raise AssertionError("These tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", denied)


def load_merge():
    spec = importlib.util.spec_from_file_location("merge_under_test", ROOT / "fastfill/merge_lora.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hf(monkeypatch):
    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, prompts, **kwargs):
            assert kwargs.get("add_special_tokens") is False
            if isinstance(prompts, str):
                return {"input_ids": [ord(c) for c in prompts]}
            width = max(map(len, prompts))
            ids = np.array([[0] * (width - len(p)) + [ord(c) for c in p] for p in prompts])
            return Batch(input_ids=ids, attention_mask=(ids != 0).astype(int))

        def batch_decode(self, rows, **kwargs):
            return ["".join(chr(int(c)) for c in row) for row in rows]

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def generate(self, input_ids, attention_mask, max_new_tokens, **kwargs):
            calls.append((attention_mask.sum(axis=1).tolist(), max_new_tokens))
            assert kwargs["do_sample"] is False
            assert input_ids.shape[1] + max_new_tokens == 10
            return np.concatenate((input_ids, np.repeat(input_ids[:, -1:], max_new_tokens, axis=1)), axis=1)

    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model))
    return calls


@pytest.mark.parametrize("batch", [1, 2, 3, 20])
def test_hf_preserves_each_prompt_budget_and_order(hf, batch):
    from fastfill.evaluate import generate

    prompts = ["a", "bbbbbbbb", "cc", "d", "e" * 10, "f" * 11, "gg"]
    assert generate("mock", prompts, 10, batch) == ["a" * 9, "bb", "c" * 8, "d" * 9, None, None, "g" * 8]
    assert all(len(lengths) <= batch for lengths, _ in hf)
    if batch > 1:
        assert any(len(lengths) == 2 for lengths, _ in hf)


@pytest.mark.parametrize("prompts", [[], ["x" * 10, "y" * 11]])
def test_hf_does_not_generate_over_limit_prompts(hf, prompts):
    from fastfill.evaluate import generate

    assert generate("mock", prompts, 10, 2) == [None] * len(prompts)
    assert hf == []


@pytest.fixture
def merge_mock(monkeypatch):
    class Base:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text(json.dumps({"rope_parameters": {"rope_theta": 1000000}}))

    class Tokenizer:
        def __init__(self, source):
            self.config = json.loads((Path(source) / "tokenizer_config.json").read_text())

        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text(json.dumps(self.config))

    loader = Mock(side_effect=lambda path, **kwargs: Tokenizer(path))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bfloat16"))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=loader),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=Mock(return_value=Base()))))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(
        PeftModel=types.SimpleNamespace(from_pretrained=Mock(return_value=types.SimpleNamespace(
            merge_and_unload=lambda: Base())))))
    return load_merge(), loader


def merge_paths(tmp_path):
    base, adapter, merged = tmp_path / "base", tmp_path / "run/final", tmp_path / "merged"
    base.mkdir()
    adapter.mkdir(parents=True)
    (base / "tokenizer_config.json").write_text(json.dumps({"pad_token": None, "extra_special_tokens": ["base"]}))
    return base, adapter, merged


def test_merge_preserves_adapter_tokenizer_and_compatibility(merge_mock, tmp_path):
    merge, loader = merge_mock
    base, adapter, merged = merge_paths(tmp_path)
    (adapter / "tokenizer_config.json").write_text(json.dumps({
        "pad_token": "<eos>", "chat_template": "adapter template", "extra_special_tokens": ["trained"]}))
    merge.apply_lora(str(base), str(merged), str(adapter))
    tc = json.loads((merged / "tokenizer_config.json").read_text())
    assert tc == {"pad_token": "<eos>", "chat_template": "adapter template", "additional_special_tokens": ["trained"]}
    assert Path(loader.call_args.args[0]) == adapter
    cfg = json.loads((merged / "config.json").read_text())
    assert cfg["rope_theta"] == cfg["rope_parameters"]["rope_theta"] == 1000000


def test_merge_falls_back_only_when_adapter_tokenizer_absent(merge_mock, tmp_path):
    merge, loader = merge_mock
    base, adapter, merged = merge_paths(tmp_path)
    merge.apply_lora(str(base), str(merged), str(adapter))
    assert Path(loader.call_args.args[0]) == base
    (adapter / "tokenizer_config.json").write_text("invalid json")
    with pytest.raises(json.JSONDecodeError):
        merge.apply_lora(str(base), str(tmp_path / "bad"), str(adapter))
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("local", [False, True])
def test_merge_preserves_one_manifest_family(merge_mock, tmp_path, local):
    merge, _ = merge_mock
    base, adapter, merged = merge_paths(tmp_path)
    run = adapter if local else adapter.parent
    manifests = {"run_manifest.json": '{"data":"v3"}\n',
                 "run_manifest.resume1.json": '{"data":"v3.1"}\n',
                 "run_manifest.resume10.json": '{"args":{"resume":true}}\n'}
    for name, content in manifests.items():
        (run / name).write_text(content)
    (run / "run_manifest.resume-backup.json").write_text("not a run")
    if local:
        (adapter.parent / "run_manifest.json").write_text('{"unrelated":true}')
        (adapter.parent / "run_manifest.resume2.json").write_text('{"unrelated":true}')
    merge.apply_lora(str(base), str(merged), str(adapter))
    actual = {p.name: p.read_text() for p in merged.glob("run_manifest*.json")}
    assert actual == manifests


@pytest.fixture
def download_mock(monkeypatch, tmp_path):
    from fastfill import download
    from huggingface_hub.hf_api import RepoFile

    sources = {"first": ("owner/one", "first"), "second": ("owner/one", "second"),
               "third": ("owner/two", "third")}
    api = Mock()
    api.list_repo_tree.side_effect = lambda repo, **kwargs: [
        RepoFile(path=f"{folder}/data.json", size=2, oid="0" * 40)
        for owner, folder in sources.values() if owner == repo]
    monkeypatch.setattr(download, "SOURCES", sources)
    monkeypatch.setattr(download, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(sys, "argv", ["download", "--out", str(tmp_path), "--workers", "1"])
    return download, api


@pytest.mark.parametrize("failures", [{"first/data.json"}, {"first/data.json", "second/data.json", "third/data.json"}])
def test_download_failures_exit_nonzero_after_all_sources(download_mock, monkeypatch, failures):
    download, api = download_mock
    fetch = Mock(side_effect=lambda repo, path, local: path not in failures)
    monkeypatch.setattr(download, "fetch", fetch)
    with pytest.raises(SystemExit) as error:
        download.main()
    assert error.value.code == 1
    assert fetch.call_count == 3
    assert api.list_repo_tree.call_count == 2


def test_download_success_and_size_resume(download_mock, monkeypatch, tmp_path):
    download, _ = download_mock
    cached = tmp_path / "owner__one/first/data.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("{}")
    fetch = Mock(return_value=True)
    monkeypatch.setattr(download, "fetch", fetch)
    assert download.main() in (None, 0)
    assert [c.args[1] for c in fetch.call_args_list] == ["second/data.json", "third/data.json"]


def test_download_module_entrypoint_exits_after_retries(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.hf_api import RepoFile

    api = Mock()
    api.list_repo_tree.return_value = [RepoFile(path="SAGE10k/data.json", size=2, oid="0" * 40)]
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    fetch = Mock(side_effect=OSError("offline fixture failure"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetch)
    monkeypatch.setattr("time.sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["download", "--out", str(tmp_path), "--only", "SAGE-10k", "--workers", "1"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(ROOT / "fastfill/download.py"), run_name="__main__")
    assert error.value.code == 1
    assert fetch.call_count == 8


def test_tiny_real_cpu_merge(tmp_path):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3, "<user>": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tok.add_special_tokens({"additional_special_tokens": ["<user>"]})
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}<eos>{% endfor %}"
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
                      eos_token_id=1, rope_theta=1000000.)
    base, adapter, merged = tmp_path / "base", tmp_path / "run/final", tmp_path / "merged"
    model = Qwen3ForCausalLM(cfg).eval()
    model.save_pretrained(base)
    tok.save_pretrained(base)
    lora = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj"]))
    with torch.no_grad():
        for name, parameter in lora.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.125)
    lora.save_pretrained(adapter)
    tok.pad_token = tok.eos_token
    tok.save_pretrained(adapter)
    for name in ("run_manifest.json", "run_manifest.resume1.json"):
        (adapter.parent / name).write_text('{"fixture":true}\n')
    reference = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, local_files_only=True),
        adapter, torch_dtype=torch.bfloat16).eval()
    ids = torch.tensor([[2, 3, 1]])
    with torch.no_grad():
        before = reference(input_ids=ids).logits.float()
    load_merge().apply_lora(str(base), str(merged), str(adapter))
    reloaded = AutoModelForCausalLM.from_pretrained(merged, dtype=torch.bfloat16, local_files_only=True).eval()
    merged_tok = AutoTokenizer.from_pretrained(merged, local_files_only=True)
    with torch.no_grad():
        after = reloaded(input_ids=ids).logits.float()
    assert (before - after).abs().max().item() < 0.01
    assert merged_tok.pad_token_id == tok.pad_token_id == 1
    assert merged_tok.get_vocab() == tok.get_vocab()
    assert merged_tok.chat_template == tok.chat_template
    assert merged_tok(["hello", "hello world"], padding=True, return_tensors="pt")["input_ids"].shape == (2, 2)
    saved = json.loads((merged / "config.json").read_text())
    assert saved["rope_theta"] == saved["rope_parameters"]["rope_theta"] == 1000000.
    tc = json.loads((merged / "tokenizer_config.json").read_text())
    assert "additional_special_tokens" in tc and not isinstance(tc.get("extra_special_tokens"), list)
    assert (merged / "run_manifest.resume1.json").read_text() == '{"fixture":true}\n'
