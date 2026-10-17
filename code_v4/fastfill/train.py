"""LoRA SFT for FastFill v1 (adapted from scripts/sft_train.py).

Differences from the OptiScene script: pre-split files instead of a random split, loss on the
assistant answer only, over-length samples dropped (never truncated mid-JSON), EOS = <|im_end|>
supervised, Qwen3 thinking disabled in the chat template, final adapter saved; run_manifest.json (the build's
MANIFEST + these arguments) is written into --out so every checkpoint says which data it saw.
Hyper-parameters: LoRA r16/alpha32 as in the OptiScene paper, but lr 1e-4 x 2 epochs (~6k steps at batch 32) instead
of the paper's 5e-6 x 10 epochs, which is a full-fine-tuning rate, ~20x too low for a rank-16 adapter.

Effective batch = bs x grad_accum x n_gpus; keep it at 32 when changing the GPU count:
  1 GPU:   CUDA_VISIBLE_DEVICES=0 python -m fastfill.train --model Qwen/Qwen3-8B --data DATA --out outputs/ff-v1 --bs 4 --grad_accum 8
  8 GPUs:  torchrun --nproc_per_node 8 -m fastfill.train --model Qwen/Qwen3-8B --data DATA --out outputs/ff-v1 --bs 4 --grad_accum 1
Resume after an interruption: add --resume (continues from the last checkpoint in --out).
Sequence length: --max_len is the model's limit (Qwen3-8B: 40960), not a memory setting; a batch is padded to its
longest sample, so memory follows the data. Measure it before choosing --bs / --grad_accum:
  --dry_run                       token length and object count percentiles with the real tokenizer, the samples
                                  over --max_len listed (they are dropped from training, never truncated)
  --mem_test 8192,16384,32768,max one forward/backward/optimizer step per length on one GPU with a batch of --bs rows
                                  padded like a training batch (row 0 at that length, the others at half, see
                                  mem_batch); peak memory printed next to the static part (weights + LoRA params/grads
                                  + AdamW states, read after a warm-up step); 'max' = the longest train or dev row,
                                  because Trainer evaluates dev every --save_steps
  --eval_max_len N                leave dev rows longer than N tokens out of that evaluation (eval loss then covers
                                  the remaining subset); pick N from the mem_test table
"""
import argparse
import collections
import hashlib
import json
import math
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments, set_seed)

from fastfill.scene import prompt_text


def end_token(tok):
    return "<|im_end|>" if "<|im_end|>" in tok.get_vocab() else tok.eos_token


def encode(tok, rows, max_len, chunk=2000):
    """Tokenize (prompt, answer) pairs in batches (per-row tokenizer calls are ~300x slower); labels mask the
    prompt; rows longer than max_len are dropped, never truncated (a cut JSON answer teaches broken output).
    -> (samples, dropped row indices)."""
    out, dropped, end = [], [], end_token(tok)
    total = len(rows)
    for i in range(0, total, chunk):
        if total > 5000 and (i // chunk) % 10 == 0:
            print(f"  [Tokenize] {i}/{total} samples ({i*100//total}%)...", flush=True)
        part = rows[i:i + chunk]
        ps = tok([prompt_text(tok, m[:-1]) for m in part], add_special_tokens=False)["input_ids"]
        ans = tok([m[-1]["content"] + end for m in part], add_special_tokens=False)["input_ids"]
        for k, (p, a) in enumerate(zip(ps, ans)):
            if len(p) + len(a) > max_len:
                dropped.append(i + k)
            else:
                out.append({"input_ids": p + a, "attention_mask": [1] * (len(p) + len(a)), "labels": [-100] * len(p) + a})
    return out, dropped


def select(recs, exclude_flags=(), source_weight=None):
    """The sampling layer: rows with any flag in exclude_flags are left out; a row of a source with weight w is
    used int(w) times plus once more with probability frac(w) (stable per uid), so w=0.5 keeps half a source and
    w=2 doubles it. Default: every row once (the data's own proportions)."""
    out, gone = [], collections.Counter()
    for r in recs:
        hit = {k for k, v in (r.get("flags") or {}).items() if v} & set(exclude_flags)
        if hit:
            gone["flag:" + sorted(hit)[0]] += 1
            continue
        w = (source_weight or {}).get(r.get("source"), 1.0)
        u = int(hashlib.sha1(r["uid"].encode()).hexdigest(), 16) % 10 ** 6 / 10 ** 6
        n = int(w) + (u < w - int(w))
        gone["weight"] += n == 0
        out += [r] * n
    return out, gone


def load(tok, path, max_len, exclude_flags=(), source_weight=None):
    print(f"Loading and tokenizing {path}...", flush=True)
    recs, gone = select([json.loads(line) for line in open(path)], exclude_flags, source_weight)
    rows, dropped = encode(tok, [r["messages"] for r in recs], max_len)
    comp = collections.Counter(r.get("source") for r in recs)
    print(f"{path}: {len(rows)} samples, dropped {len(dropped)} longer than {max_len} tokens"
          + (f", left out {dict(gone)}" if +gone else "") + f"; by source {dict(comp.most_common())}", flush=True)
    return rows


def length_report(tok, path, max_len):
    """Real-tokenizer profile of one file: tokens per sample (total / answer) and objects per room, the samples over
    max_len listed by uid so they can be looked at instead of silently dropped."""
    recs = [json.loads(line) for line in open(path)]
    rows, over = encode(tok, [r["messages"] for r in recs], float("inf"))
    lens = [len(r["input_ids"]) for r in rows]
    ans = [sum(l != -100 for l in r["labels"]) for r in rows]
    nobj = [len(json.loads(r["messages"][1]["content"])["objects"]) for r in recs]
    nfix = [len(json.loads(r["messages"][1]["content"]).get("fixed", [])) for r in recs]
    pct = lambda v: {k: sorted(v)[min(int(f * len(v)), len(v) - 1)] for k, f in
                     (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.0))}
    rep = {"file": path, "samples": len(rows), "tokens_total": sum(lens), "tokens": pct(lens), "answer_tokens": pct(ans),
           "objects": pct(nobj), "fixed": pct(nfix),
           "over_max_len": [{"uid": recs[i]["uid"], "source": recs[i]["source"], "tokens": lens[i], "objects": nobj[i]}
                            for i in range(len(rows)) if lens[i] > max_len],
           "buckets": {f"<= {b}": sum(l <= b for l in lens) for b in (4096, 8192, 16384, 32768, max_len)}}
    return rep


def mem_batch(tok, longest, spec, bs):
    """The batch DataCollatorForSeq2Seq(padding=True) builds for a --bs step whose longest row has the requested length:
    'max' = the longest train/dev row itself, a number = that many tokens (the longest row's tokens repeated, every
    token supervised: an upper bound). The other bs-1 rows are the same tokens cut to half and right-padded (pad id,
    attention_mask 0, labels -100), so SDPA materializes its (bs,1,L,L) mask as in almost every real batch.
    -> {input_ids, attention_mask, labels}: bs lists of L ints each."""
    if spec == "max":
        ids, lab = longest["input_ids"], longest["labels"]
    else:
        ids = (longest["input_ids"] * (int(spec) // len(longest["input_ids"]) + 1))[:int(spec)]
        lab = ids
    L, h = len(ids), len(ids) // 2
    rows = [(ids, lab)] + [(ids[:h], lab[:h])] * (bs - 1)
    return {"input_ids": [i + [tok.pad_token_id] * (L - len(i)) for i, _ in rows],
            "attention_mask": [[1] * len(i) + [0] * (L - len(i)) for i, _ in rows],
            "labels": [l + [-100] * (L - len(l)) for _, l in rows]}


def mem_test(model, tok, train, dev, specs, bs):
    """One training step per requested length on the current GPU in the training configuration: a batch of bs rows
    padded like the collator's (mem_batch), gradient checkpointing on. 'max' = the longest train or dev row (Trainer
    evaluates dev every save_steps). A warm-up step (512 tokens) first allocates the LoRA grads and AdamW states, so
    static = weights + LoRA params/grads + AdamW states and the per-length column is what the step adds
    (activations, logits, attention mask). OOM at one length moves on to the next."""
    gpu = torch.device("cuda")
    model.to(gpu)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    longest = max(train + dev, key=lambda r: len(r["input_ids"]))
    print(f"longest train row {max(len(r['input_ids']) for r in train)} tokens, "
          f"longest dev row {max(len(r['input_ids']) for r in dev)} tokens (eval set)", flush=True)

    def step(spec):                       # -> peak bytes of one forward/backward/optimizer step, None on OOM
        b = None
        torch.cuda.reset_peak_memory_stats()
        try:                              # the batch's own allocation can OOM too
            b = {k: torch.tensor(v, device=gpu) for k, v in mem_batch(tok, longest, spec, bs).items()}
            model(**b).loss.backward()
            opt.step()
            return torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            opt.zero_grad(set_to_none=False)  # grads stay resident, as in training
            del b
            torch.cuda.empty_cache()

    if step("512") is None:               # warm-up: creates the LoRA grads and the AdamW moments
        print(f"OOM already at 512 tokens x bs {bs} on {torch.cuda.get_device_name()}: the model does not fit this card "
              f"in this configuration; no table", flush=True)
        return
    static = torch.cuda.memory_allocated()
    print(f"static (weights + LoRA params/grads + AdamW states) {static / 2 ** 30:.1f} GB", flush=True)
    for spec in specs:
        n = len(longest["input_ids"]) if spec == "max" else int(spec)
        peak = step(spec)
        if peak is None:
            print(f"len {n:6d}  bs {bs}  OOM on {torch.cuda.get_device_name()} "
                  f"({torch.cuda.get_device_properties(gpu).total_memory / 2 ** 30:.0f} GB)", flush=True)
        else:
            print(f"len {n:6d}  bs {bs}  peak {peak / 2 ** 30:5.1f} GB  = static {static / 2 ** 30:.1f} + "
                  f"activations/logits/mask {(peak - static) / 2 ** 30:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--data", required=True, help="dir with train.jsonl / dev.jsonl from fastfill.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=40960, help="model sequence limit; longer samples are dropped")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad_accum", type=int, default=8, help="per GPU; divide by the GPU count to keep batch 32")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--full_ft", action="store_true", help="full fine-tuning instead of LoRA (use with --deepspeed)")
    ap.add_argument("--deepspeed", default=None)
    ap.add_argument("--attn", default="sdpa", help="sdpa | flash_attention_2")
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--report_to", default="none")
    ap.add_argument("--dry_run", action="store_true", help="tokenize and print the length profile only (no GPU)")
    ap.add_argument("--mem_test", default=None, help="e.g. 8192,16384,32768,max: measure peak memory per length, then exit")
    ap.add_argument("--exclude_flags", nargs="*", default=["oob_objects", "overlapping_furniture", "fixed_collision"],
                    help="train rows carrying any of these quality flags are left out (see the row's 'flags'). Default: "
                         "the three whose reference answer serve.py would refuse (objects > 10 cm through the walls; "
                         "copies of one piece through each other; furniture through a door / column / stair box). "
                         "Pass --exclude_flags with no value to train on all")
    ap.add_argument("--source_weight", nargs="*", default=[],
                    help="SOURCE=w: use each train row of SOURCE w times on average (0.5 halves it, 2 doubles it)")
    ap.add_argument("--eval_max_len", type=int, default=0,
                    help="dev rows longer than this are left out of the eval at every save_steps (0 = keep all)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    a = ap.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    a.source_weight = {k: float(v) for k, v in (x.split("=") for x in a.source_weight)}
    known = {json.loads(line)["source"] for line in open(f"{a.data}/train.jsonl")} if a.source_weight else set()
    if set(a.source_weight) - known:
        raise SystemExit(f"--source_weight: unknown source(s) {sorted(set(a.source_weight) - known)}; known: {sorted(known)}")

    print(f"Loading tokenizer from {a.model}...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.dry_run:
        for split in ("train", "dev", "test"):
            rep = length_report(tok, f"{a.data}/{split}.jsonl", a.max_len)
            print(json.dumps(rep, indent=1))
        train = load(tok, f"{a.data}/train.jsonl", a.max_len, a.exclude_flags, a.source_weight)
        print(tok.decode(train[0]["input_ids"]))
        print("supervised part:", repr(tok.decode([t for t in train[0]["labels"] if t != -100])))
        return
    train = load(tok, f"{a.data}/train.jsonl", a.max_len, a.exclude_flags, a.source_weight)
    dev = load(tok, f"{a.data}/dev.jsonl", a.max_len)
    if a.eval_max_len:                      # protect the eval at every save_steps; eval loss then covers a subset
        kept = [r for r in dev if len(r["input_ids"]) <= a.eval_max_len]
        print(f"eval set: {len(dev) - len(kept)} dev rows longer than {a.eval_max_len} tokens left out, {len(kept)} kept", flush=True)
        dev = kept
    if int(os.environ.get("RANK", 0)) == 0 and not a.mem_test:   # provenance of this run (build MANIFEST + args)
        os.makedirs(a.out, exist_ok=True)
        man = f"{a.data}/MANIFEST.json"
        rec = {"args": vars(a), "data_manifest": json.load(open(man)) if os.path.exists(man) else None}
        path = f"{a.out}/run_manifest.json"
        if a.resume and os.path.exists(path):   # a resume is recorded beside the run it continues; a new run replaces
            k = 1
            while os.path.exists(f"{a.out}/run_manifest.resume{k}.json"):
                k += 1
            path = f"{a.out}/run_manifest.resume{k}.json"
        json.dump(rec, open(path, "w"), indent=1)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, attn_implementation=a.attn)
    model.config.use_cache = False
    set_seed(42)                            # before get_peft_model: same LoRA init every run (v1.0 vs v1.1 pairing)
    if not a.full_ft:
        model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()
    if a.mem_test:
        mem_test(model, tok, train, dev, a.mem_test.split(","), a.bs)
        return

    # 3% warmup as an integer step count: transformers 4.x rejects fractional warmup_steps, >=5.15 has no warmup_ratio
    # one process that sees several GPUs runs DataParallel: its batch is bs x n_gpu, like DDP's
    world = int(os.environ.get("WORLD_SIZE", 0)) or max(1, torch.cuda.device_count())
    total_steps = math.ceil(math.ceil(len(train) / (a.bs * world)) / a.grad_accum) * a.epochs
    # batch similar lengths together (~1/3 less padding); the argument was renamed in transformers 5.14
    by_length = ({"group_by_length": True} if "group_by_length" in TrainingArguments.__dataclass_fields__
                 else {"train_sampling_strategy": "group_by_length"})
    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bs, per_device_eval_batch_size=1,  # dev pads to 40k tokens
        gradient_accumulation_steps=a.grad_accum, learning_rate=a.lr, num_train_epochs=a.epochs, **by_length,
        lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * total_steps)), weight_decay=0.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10, eval_strategy="steps", eval_steps=a.save_steps, save_strategy="steps",
        save_steps=a.save_steps, save_total_limit=3, remove_unused_columns=False,
        report_to=a.report_to, deepspeed=a.deepspeed, ddp_find_unused_parameters=False)
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train(resume_from_checkpoint=True if a.resume else None)
    trainer.save_model(f"{a.out}/final")
    tok.save_pretrained(f"{a.out}/final")


if __name__ == "__main__":
    main()
