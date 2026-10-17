"""Merge a FastFill adapter, retaining its tokenizer and run provenance.

    python -m fastfill.merge_lora --base_model_path BASE --lora_path RUN/final --output_path MERGED

Uses the local adapter's saved tokenizer when present, otherwise the base tokenizer.
The nearest run_manifest.json selects the run; its numbered resume manifests travel with it.
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def save_compatibility(output_path):
    """Keep the existing Transformers 5 -> 4 artifact compatibility rewrites."""
    cfg_path = Path(output_path) / "config.json"
    tok_path = Path(output_path) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    tc = json.loads(tok_path.read_text())
    # Transformers 4 needs top-level rope_theta and a list named additional_special_tokens.
    theta = (cfg.get("rope_parameters") or {}).get("rope_theta")
    compatible_cfg = {"rope_theta": theta, **cfg} if theta is not None else cfg
    compatible_tc = ({**{k: v for k, v in tc.items() if k != "extra_special_tokens"},
                      "additional_special_tokens": tc["extra_special_tokens"]}
                     if isinstance(tc.get("extra_special_tokens"), list) else tc)
    cfg_path.write_text(json.dumps(compatible_cfg, indent=2) + "\n")
    tok_path.write_text(json.dumps(compatible_tc, indent=2, ensure_ascii=False) + "\n")


def copy_run_manifests(lora_path, output_path):
    """Prefer an adapter-local run; do not mix in another run from its parent."""
    adapter = Path(lora_path)
    for run in (adapter, adapter.parent):
        if not (run / "run_manifest.json").is_file():
            continue
        for manifest in run.iterdir():
            if manifest.is_file() and re.fullmatch(r"run_manifest(?:\.resume[0-9]+)?\.json", manifest.name):
                shutil.copyfile(manifest, Path(output_path) / manifest.name)
        break


def apply_lora(model_name_or_path, output_path, lora_path):
    tokenizer_source = lora_path if (Path(lora_path) / "tokenizer_config.json").is_file() else model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch.bfloat16)
    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(base, lora_path, torch_dtype=torch.bfloat16)
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    save_compatibility(output_path)
    copy_run_manifests(lora_path, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    apply_lora(args.base_model_path, args.output_path, args.lora_path)


if __name__ == "__main__":
    main()
