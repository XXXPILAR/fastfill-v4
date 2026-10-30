"""Joint structured geometry training. v1 entry points and snapshots are untouched.

Resume (``training.resume`` / ``--resume <state_dir>``) continues into a NEW
output directory. Restored: model, optimizer, LR schedule, GradScaler,
Python/NumPy/torch RNG, Accelerate's accumulation counter, completed-step and
skipped-window counters, earlier log records, and the data order (epoch,
consumed batches, shuffle generator; augmentation is seeded per
(seed, epoch, row) so replayed samples are identical). Required unchanged:
every config section except ``training.resume``, ``--max-samples`` and the
training data hash. Also required unchanged, unless ``--allow-resume-change``
accepts the change and both manifests record it as ``resumed_with_changes``
({field: {saved, current}}): the world size (it changes the global batch and
the rank-local data position) and the validation data hash (an accepted change
restarts ``selection_metric.best`` at the resumed step, since earlier scores were
taken on other data; states carry that step on, so later plain resumes keep it,
and the final manifest records it as ``selection_metric.best_after_step``). States
saved before that check lack both fields and resume with a warning; both
manifests list the unchecked fields in ``resume_unverified``. Not restored:
wall-clock ``elapsed_s`` and the earlier process's ``train.log``.

Run binding: ``run_manifest_start.json`` is written before the first update
(config, data/validation/implementation sha256, backbone path + HF snapshot
revision + config.json sha256, tokenizer sha256, augmentation, world size,
admitted train/validation/minimal-projection counts) and
``run_manifest.json`` when the run finishes. Every exported model directory gets
``checkpoint_manifest.json`` (``io.CHECKPOINT_KEYS``), which evaluate/predict
read through ``io.load_checkpoint_config``. ``--expect-data-sha256`` /
``--expect-validation-sha256`` abort before any output on a mismatch.

Validation reports the full condition and its three-field projection
(``evaluate.project_minimal``, boundary-known rectangular rooms only); the
projection's weighted geometry objective is the manifest's ``selection_metric``
(``SELECTION_METRIC``, lower is better). Collapse is logged beside it, over
every request (``collapse``) and over the predictions matched to label-complete
labels (``collapse_matched``); the labels' own collapse score on the same
projection is recorded at launch as ``selection_metric.ground_truth`` (both
manifests). Compare ``collapse_matched`` with it, which covers the same objects
(size-masked sources are only in ``collapse``): a ``collapse_matched`` score
below it means a layout more spread out than the data, not a more accurate one.
Validation and checkpointing also run at the final update.
Window and validation logs carry every term the criterion reports in
``term_sums``, e.g. position_cell / position_residual / position_z of the grid head.

Diagnostics (validation, its minimal projection, the baseline, the labels' collapse) are not prepared by
Accelerate: each rank reads its unpadded stride shard of the rows (``_diagnostic_loader``), so every row counts
once; ranks may run different numbers of batches (or none), and the reductions after each loop are the only
collectives. Their forward uses the unwrapped model and a rank-local criterion, so ``batches`` is the global
batch count and ``geometry_objective_mean_of_batches`` the mean of each batch's own objective over all of them.
Each diagnostic loader has its own generator, so diagnostics never draw from the global RNG: a resumed run
matches the uninterrupted one with dropout too. This changes the RNG stream relative to older runs; resuming
an older run's state stays exact for dropout 0, the only setting where the stream does not touch training.
States record this convention (``"diagnostics": DIAGNOSTICS``). A multi-rank state without it (saved before round
5) logged padded rank batches, so resuming it with validation restarts ``selection_metric.best`` at the resumed
step and records ``resumed_with_changes["diagnostics"]``; this needs no ``--allow-resume-change``, since training
itself is unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from functools import partial
import hashlib
import json
import shutil
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

from fastfill.v2.batch import AUGMENT_DEFAULTS, AUGMENT_MAGNITUDES, augment_sample, collate_samples, load_tokenizer, tokenize_condition
from fastfill.v2.data import filter_rows_by_flags
from fastfill.v2.evaluate import COLLAPSE_KEYS, batch_collapse_counts, collapse_score, project_minimal
from fastfill.v2.io import (CHECKPOINT_MANIFEST, backbone_provenance, ensure_disjoint, fingerprint, fingerprint_tree,
                            read_samples, run_metadata, safe_output, to_device)
from fastfill.v2.losses import GRID_TERMS, GeometryCriterion, LossConfig
from fastfill.v2.matching import match_batch
from fastfill.v2.model import ModelConfig, build_model, model_inputs
from fastfill.v2.objective import ObjectiveWindow, local_objective_count
from fastfill.v2.size_range import size_target_conflicts


TERMS = ("position", "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary")  # base criterion terms; windows log any key
# GeometryCriterion's term_sums keys in its order: diagnostic windows start from them, so an empty shard reduces the same shape.
DIAGNOSTIC_TERMS = ("position", *GRID_TERMS, "size", "yaw_cls", "yaw_reg", "box", "collision", "boundary")


def _diagnostic_terms(criterion):
    """DIAGNOSTIC_TERMS plus the optional terms this criterion's config reports."""
    return (DIAGNOSTIC_TERMS + (("expected_geometry",) if criterion.config.expected_geometry else ())
            + (("support", "support_accuracy") if criterion.config.support else ()))


def _cell_cost_options(model_config, loss_config):
    """Collate options for the expected-geometry loss, or None when it is off."""
    if not loss_config.expected_geometry:
        return None
    if model_config.position_head != "grid_residual":
        raise ValueError("loss.expected_geometry needs model.position_head grid_residual")
    return {"grid": model_config.position_grid, "door_depth_m": loss_config.door_clearance_m}
DIAGNOSTICS = "rank-shards"  # the diagnostics convention states record (round 5: unpadded rank shards, global batches)
KEEP_STATES = 2  # ponytail: state-step-* holds the frozen backbone (~29 GB); keep the newest two, model-step-* (~0.1 GB) all
SELECTION_METRIC = {"name": "validation.minimal.geometry_objective", "lower_is_better": True,
                    "definition": "weighted geometry objective on the minimal (three-field) projection of the validation rows; "
                                  "collapse (BEV overlap rate + central-quarter fraction) is reported beside it; ground_truth "
                                  "anchors collapse_matched (the same objects), never minimised on its own (a degenerate "
                                  "wall-hugging layout scores 0)"}


def _target_predictions(batch):
    """Finite stand-in predictions so preflight can certify every exchangeable group."""
    size = batch["targets"]["size"]
    return {"position_normalized": torch.nan_to_num(batch["targets"]["position_normalized"], nan=0.),
            "size": torch.where(torch.isfinite(size) & (size > 0), size, torch.ones_like(size)),
            "slot_mask": batch["slot_mask"]}


def _preflight(samples, tokenizer, model_config, training, loss_config=LossConfig()):
    kept, rejected = [], []
    for index, sample in enumerate(samples):
        condition = sample["condition"]
        tokens, _ = tokenize_condition(condition, tokenizer)
        if len(tokens) > training["max_length"] or len(condition["objects"]) > model_config.max_objects:
            rejected.append({"row": index, "reason": "context_or_object_budget", "tokens": len(tokens),
                             "objects": len(condition["objects"]), "provenance": sample["provenance"]})
            continue
        # Fail loudly on malformed geometry/conditions rather than filtering an unexplained error.
        b = collate_samples([sample], tokenizer, max_length=training["max_length"], max_objects=model_config.max_objects)
        try:
            match_batch(_target_predictions(b), b, loss_config.hungarian, loss_config.alpha_position, loss_config.alpha_size)
        except ValueError as error:
            raise ValueError(f"row {index} cannot be matched before training: {error}; provenance {sample['provenance']}") from error
        conflicts = size_target_conflicts(b, model_config, loss_config)
        if conflicts:
            rejected.append({"row": index, "reason": "size_target_outside_model_range",
                             "coordinates": conflicts, "provenance": sample["provenance"]})
            continue
        if not local_objective_count(b, loss_config):
            rejected.append({"row": index, "reason": "no_active_objective", "provenance": sample["provenance"]})
            continue
        kept.append(sample)
    if not kept:
        raise ValueError("no supervised samples have an active objective within context/object budgets; inspect enabled losses and label validity")
    return kept, rejected


def _gradients(model):
    names = ("backbone", "decoder", "position_head", "size_head", "yaw_logits_head", "yaw_residual_head")
    return {name: float(sum((p.grad.detach().float().square().sum() for key, p in model.named_parameters()
                            if name in key and p.grad is not None), torch.tensor(0., device=next(model.parameters()).device)).sqrt())
            for name in names}


def _nonfinite_gradients(model, accelerator):
    """bf16 has no GradScaler: detect inf/nan gradients on any rank before clipping or stepping."""
    local = any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    flag = torch.tensor(int(local), dtype=torch.int64, device=accelerator.device)
    return bool(accelerator.reduce(flag, reduction="sum").item())


@torch.no_grad()
def _rescale_flushed_window_gradients(model, configured_steps, microbatches):
    """Undo Accelerate's fixed-K divisor for an incomplete epoch-tail window.

    Backward and DDP synchronization have completed. Gradients may still be AMP
    scaled; multiplying before Accelerate unscale/clip preserves that scaling.
    Empty-label microbatches belong to the declared microbatch-mean denominator.
    """
    if not 1 <= microbatches <= configured_steps:
        raise RuntimeError("invalid synchronized accumulation-window length")
    if microbatches != configured_steps:
        factor = configured_steps / microbatches
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


class _Window:
    """Local per-term unweighted sums and valid counts over every microbatch of a window.

    The criterion exposes local ``term_sums``/``term_counts``; every key it
    reports is logged, in its order. One config gives every rank the same keys,
    so one summed collective at summary time gives the global count-weighted
    window means. ``loss`` keeps the microbatch mean of the weighted objective
    (local numerators over global counts, so its rank mean is the global value).
    ``summary(shards=True)`` (diagnostics: rank-local losses, rank shards of
    different lengths): ``loss`` is the mean over every rank's microbatches and
    ``microbatches`` their global count. ``keys`` (diagnostics) are summed even
    by a rank without batches.
    """

    def __init__(self, keys=()):
        self.loss, self.microbatches, self.rows, self.keys = 0., 0, [], tuple(keys)
        self.sums, self.counts = dict.fromkeys(self.keys, 0.), dict.fromkeys(self.keys, 0)

    def add(self, result, batch=None):
        self.loss += float(result["loss"].detach().float())
        self.microbatches += 1
        for key, value in result["term_sums"].items():
            self.sums[key] = self.sums.get(key, 0.) + float(value)
            self.counts[key] = self.counts.get(key, 0) + int(result["term_counts"][key])
        if batch is not None:
            self.rows.extend(p.get("scene_id") for p in batch["provenance"])
        return self

    def summary(self, accelerator, shards=False):
        # Every rank enters this collective with one dtype/shape, including ranks without valid terms (or batches).
        keys = list(self.sums)
        if self.keys and len(keys) != len(self.keys):
            raise RuntimeError(f"criterion terms {keys} outgrow the declared {list(self.keys)}; a rank without "
                               "batches would reduce another shape")
        local = torch.tensor([self.loss, float(self.microbatches), *(self.sums[key] for key in keys),
                              *(float(self.counts[key]) for key in keys)], dtype=torch.float64, device=_reduce_device(accelerator))
        values = accelerator.reduce(local, reduction="sum").tolist()
        batches, sums, counts = values[1], values[2:2 + len(keys)], values[2 + len(keys):]
        return {"loss": values[0] / max(1., batches) if shards else values[0] / accelerator.num_processes / max(1, self.microbatches),
                "unweighted": {key: total / count if count else None for key, total, count in zip(keys, sums, counts)},
                "counts": {key: int(count) for key, count in zip(keys, counts)},
                "microbatches": int(batches) if shards else self.microbatches}


def _reduce_device(accelerator):
    """Metal has no float64: mps runs (single-process under Accelerate, so no collective) reduce on the CPU."""
    return torch.device("cpu") if accelerator.device.type == "mps" else accelerator.device


def _record(window, model, optimizer, step, elapsed, count, accelerator):
    summary = window.summary(accelerator)
    return {"step": step, "elapsed_s": elapsed, "loss": summary["loss"], "unweighted": summary["unweighted"],
            "counts_window": summary["counts"], "gradient_norms": _gradients(model),
            "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "active_objective_count": count, "accumulation_microbatches": summary["microbatches"],
            "rows_sha256": hashlib.sha256(json.dumps(window.rows).encode()).hexdigest()[:16],
            "active_objective_scope": "global accumulated window",
            "logging_scope": "loss: microbatch mean; unweighted: count-weighted means over the window; rows digest: this rank's order"}


def _verify_input_fingerprints(metadata, accelerator=None):
    changed = []
    for path_key, hash_key in (("data_path", "data_sha256"),
                               ("validation_data_path", "validation_data_sha256")):
        path = metadata[path_key]
        if path is None:
            continue
        try:
            matches = fingerprint(path) == metadata[hash_key]
        except OSError:
            matches = False
        if not matches:
            changed.append(path)
    failed = bool(changed)
    if accelerator is not None:
        # All ranks must leave this boundary together if any input changed;
        # otherwise one process can fail while peers wait to finish the run.
        flags = torch.tensor(int(failed), dtype=torch.int64, device=accelerator.device)
        failed = bool(accelerator.reduce(flags, reduction="sum").item())
    if failed:
        detail = ", ".join(changed) if changed else "input observed by another process"
        raise RuntimeError(f"training input fingerprint changed during the run: {detail}")


DEFAULT_TRAINING = {"steps": 1000, "batch_size": 2, "learning_rate": 1e-4,
                    "gradient_accumulation_steps": 1, "max_length": 4096, "seed": 42,
                    "clip_grad_norm": 1., "mixed_precision": "no", "cpu": False, "cpu_threads": 2,
                    "checkpoint_every": 100, "validate_every": 100,
                    "resume": None, "export_model_every_checkpoint": True}
# warmup_steps None -> 3% of steps; decoder_lr None -> training.learning_rate (which trains LoRA/backbone).
DEFAULT_OPTIMIZER = {"warmup_steps": None, "schedule": "cosine", "decoder_lr": None}
DEFAULT_AUGMENTATION = AUGMENT_DEFAULTS
DEFAULT_VALIDATION = {"exclude_flags": ["oob_objects", "fixed_collision", "overlapping_furniture"]}
CONFIG_SECTIONS = ("model", "loss", "training", "optimizer", "augmentation", "validation")
# Fields added after runs were launched: left out of the resume digest at their defaults, so those runs still resume.
DIGEST_LATER_DEFAULTS = {"model": {"pose_conditioning": False, "support_head": False, "gradient_checkpointing": False},
                         "loss": {"expected_geometry": 0., "door_clearance_m": .8, "support": 0.},
                         "augmentation": {"given_pose_p": 0., "given_pose_corrupt_p": 0., "given_pose_noise_m": 0.,
                                          "given_pose_yaw_noise_deg": 0., "given_pose_size_noise": 0.}}


def _digest_view(resolved):
    return {name: {key: value for key, value in section.items()
                   if not (key in DIGEST_LATER_DEFAULTS.get(name, {}) and value == DIGEST_LATER_DEFAULTS[name][key])}
            if name in DIGEST_LATER_DEFAULTS else section for name, section in resolved.items()}


def _section(section, name, defaults):
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(section) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return {**defaults, **section}


def _positive_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and 0 < value < float("inf")


def _training_config(config):
    cfg = _section(config, "training", DEFAULT_TRAINING)
    for key in ("steps", "batch_size", "gradient_accumulation_steps", "max_length", "cpu_threads"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("checkpoint_every", "validate_every", "seed"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("cpu", "export_model_every_checkpoint"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("learning_rate", "clip_grad_norm"):
        if not _positive_number(cfg[key]):
            raise ValueError(f"{key} must be positive and finite")
    if cfg["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed precision must be no/fp16/bf16")
    if cfg["resume"] is not None and not isinstance(cfg["resume"], (str, Path)):
        raise ValueError("resume must be null or a saved state directory path")
    return cfg


def _optimizer_config(config, training):
    cfg = _section(config.get("optimizer", {}), "optimizer", DEFAULT_OPTIMIZER)
    if cfg["warmup_steps"] is None:
        cfg["warmup_steps"] = round(.03 * training["steps"])
    if cfg["decoder_lr"] is None:
        cfg["decoder_lr"] = training["learning_rate"]
    if (isinstance(cfg["warmup_steps"], bool) or not isinstance(cfg["warmup_steps"], int)
            or not 0 <= cfg["warmup_steps"] < training["steps"]):
        raise ValueError("warmup_steps must be an integer in [0, steps)")
    if cfg["schedule"] not in {"cosine", "constant"}:
        raise ValueError("schedule must be cosine or constant")
    if not _positive_number(cfg["decoder_lr"]):
        raise ValueError("decoder_lr must be positive and finite")
    return cfg


def _augmentation_config(config):
    cfg = _section(config.get("augmentation", {}), "augmentation", DEFAULT_AUGMENTATION)
    for key, value in cfg.items():
        if key in AUGMENT_MAGNITUDES:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"augmentation.{key} must be a finite nonnegative number")
        elif key.endswith("_p"):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"augmentation.{key} must be a probability in [0,1]")
        elif not isinstance(value, bool):
            raise ValueError(f"augmentation.{key} must be a boolean")
    return cfg


def _validation_config(config):
    cfg = _section(config.get("validation", {}), "validation", DEFAULT_VALIDATION)
    flags = cfg["exclude_flags"]
    if not isinstance(flags, list) or any(not isinstance(flag, str) or not flag for flag in flags):
        raise ValueError("validation.exclude_flags must be a list of flag names")
    return cfg


def _lr_factor(completed, warmup, total, schedule):
    """Linear warmup over `warmup` updates, then cosine from lr to 10% of lr at the final update."""
    if completed < warmup:
        return (completed + 1) / warmup
    if schedule == "constant":
        return 1.
    progress = (completed - warmup) / max(1, total - warmup)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def _parameter_groups(model, lr, decoder_lr):
    """Backbone (LoRA or full) parameters train at lr; projection, slots, decoder and heads at decoder_lr."""
    groups = {"backbone": [], "decoder": []}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups["backbone" if name.startswith("backbone.") else "decoder"].append(parameter)
    return [{"params": params, "lr": lr if name == "backbone" else decoder_lr, "name": name}
            for name, params in groups.items() if params]


class _Rows(torch.utils.data.Dataset):
    """Training rows; augmentation is seeded per (seed, epoch, row) so a resumed run replays identical samples.

    Preflight admits the original condition only. An augmented one that outgrows
    ``max_length`` (sign/digit growth, longer shuffled IDs) would make collate
    raise inside the DataLoader, so it falls back to the original row and is counted.
    """

    def __init__(self, rows, augmentation, seed, tokenizer=None, max_length=None):
        self.rows, self.augmentation, self.seed, self.epoch = rows, augmentation, seed, 0
        self.tokenizer, self.max_length, self.fallbacks = tokenizer, max_length, 0
        self.active = any(value > 0 if key.endswith("_p") else value for key, value in augmentation.items())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # ponytail: num_workers=0, so `epoch` set by the trainer and `fallbacks` are visible in-process.
        if not self.active:
            return self.rows[index]
        digest = hashlib.sha256(f"{self.seed}:{self.epoch}:{index}".encode()).digest()
        sample = augment_sample(self.rows[index], self.augmentation,
                                torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1))
        if self.tokenizer is not None and len(tokenize_condition(sample["condition"], self.tokenizer)[0]) > self.max_length:
            self.fallbacks += 1
            return self.rows[index]
        return sample


class _TrainerState:
    """Resume bookkeeping saved by Accelerate beside model/optimizer/scheduler/scaler/RNG state."""

    def __init__(self):
        self.values = {}

    def state_dict(self):
        return self.values

    def load_state_dict(self, values):
        self.values = values


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _diagnostic_loader(samples, accelerator, batch_size, collate, seed):
    """This rank's unpadded stride shard, not prepared: every row counts once globally (``prepare``'s even batches
    repeat leading rows on later ranks); ranks may hold different numbers of batches, an empty shard too. Its own
    generator keeps iteration (the loader's base seed draw) off the global RNG."""
    return DataLoader(samples[accelerator.process_index::accelerator.num_processes], batch_size=batch_size,
                      collate_fn=collate, generator=torch.Generator().manual_seed(seed))


def _rank_local(criterion, predictions, batch):
    """The criterion on one diagnostic batch with this rank's counts: shards differ in length, so its per-batch count
    all_reduce (``losses._mean``) would pair different batches or hang."""
    return criterion(predictions, batch, global_counts=False)


@torch.no_grad()
def _validation(model, loader, criterion, accelerator):
    """Count-weighted geometry terms plus the global predicted collapse counts and score of one projection:
    ``collapse`` over every request slot, ``collapse_matched`` over the slots the criterion's matching assigns to
    label-complete labels (the objects ``_label_collapse`` counts). ``loader`` yields this rank's shard
    (``_diagnostic_loader``); the unwrapped model runs no DDP collective per batch."""
    model = accelerator.unwrap_model(model)
    model.eval()
    window, collapse, matched = _Window(_diagnostic_terms(criterion)), dict.fromkeys(COLLAPSE_KEYS, 0.), dict.fromkeys(COLLAPSE_KEYS, 0.)
    for batch in loader:
        batch = to_device(batch, accelerator.device)
        predictions = model(**model_inputs(batch))
        result = _rank_local(criterion, predictions, batch)
        window.add(result)
        for counts, assignment in ((collapse, None), (matched, result["assignment"])):
            for key, value in batch_collapse_counts(predictions, batch, assignment).items():
                counts[key] += value
    model.train()
    summary = window.summary(accelerator, shards=True)
    return {"geometry_objective_mean_of_batches": summary["loss"], "batches": summary["microbatches"],
            "unweighted": summary["unweighted"], "counts": summary["counts"],
            "collapse": _reduce_collapse(collapse, accelerator), "collapse_matched": _reduce_collapse(matched, accelerator),
            "selection_scope": "geometry diagnostic; asset/runtime evaluation is separate"}


def _reduce_collapse(collapse, accelerator):
    totals = accelerator.reduce(torch.tensor(list(collapse.values()), dtype=torch.float64, device=_reduce_device(accelerator)),
                                reduction="sum").tolist()
    collapse = {key: value if key.endswith("_m") else int(value) for key, value in zip(collapse, totals)}
    return {**collapse, "score": collapse_score(collapse)}


@torch.no_grad()
def _label_collapse(loader, accelerator):
    """Global collapse counts and score of the labels themselves over one projection (the selection score's anchor).
    ``loader`` yields this rank's shard; counting runs on the CPU, so batches stay there."""
    collapse = dict.fromkeys(COLLAPSE_KEYS, 0.)
    for batch in loader:
        labels = {"position_normalized": batch["targets"]["position_normalized"], "size": batch["targets"]["size"],
                  "yaw": batch["targets"]["yaw"]}
        for key, value in batch_collapse_counts(labels, batch).items():
            collapse[key] += value
    return _reduce_collapse(collapse, accelerator)


def _category_median_sizes(rows):
    """Per-category per-axis medians of fully valid size labels; the global median backs unseen categories."""
    sizes = {}
    for row in rows:
        categories = {obj["id"]: obj.get("category") for obj in row["condition"]["objects"]}
        valid = row.get("validity", {}).get("size", [])
        for index, target in enumerate(row["target"]["objects"]):
            mask = valid[index] if index < len(valid) else False
            if mask is True or (isinstance(mask, list) and len(mask) == 3 and all(mask)):
                sizes.setdefault(categories.get(target["id"]), []).append(target["target_size_local_m"])
    medians = {category: [statistics.median(axis) for axis in zip(*values)] for category, values in sizes.items()}
    everything = [size for values in sizes.values() for size in values]
    medians[None] = [statistics.median(axis) for axis in zip(*everything)] if everything else [1., 1., 1.]
    return medians


def _baseline_predictions(batch, medians, yaw_bins):
    """Room-centre floor position, per-category median size, uniform yaw bins with zero residual."""
    mask = batch["slot_mask"]
    b, n = mask.shape
    size = torch.ones((b, n, 3), device=mask.device)
    for i, objects in enumerate(batch["objects"]):
        for j, obj in enumerate(objects):
            size[i, j] = torch.tensor(medians.get(obj.get("category"), medians[None]), device=mask.device)
    return {"position_normalized": torch.tensor([.5, .5, 0.], device=mask.device).expand(b, n, 3).clone(),
            "size": size, "yaw_logits": torch.zeros((b, n, yaw_bins), device=mask.device),
            "yaw_residuals": torch.zeros((b, n, yaw_bins), device=mask.device), "slot_mask": mask}


@torch.no_grad()
def _baseline(loader, criterion, accelerator, medians, yaw_bins):
    """``_validation``'s window over this rank's shard (``_diagnostic_loader``) of the baseline rows."""
    window = _Window(_diagnostic_terms(criterion))
    for batch in loader:
        batch = to_device(batch, accelerator.device)
        window.add(_rank_local(criterion, _baseline_predictions(batch, medians, yaw_bins), batch))
    return {**window.summary(accelerator, shards=True), "predictor": "room-centre floor position; per-category median size of the first "
            "20000 training rows (global median for unseen categories); uniform yaw bins with zero residual"}


def _due(step, every):
    return bool(every) and step % every == 0


def _verify_expected_hashes(metadata, expect_data_sha256, expect_validation_sha256):
    """Abort before any output when an input is not the exact file the operator pinned."""
    for label, expected, actual in (("training data", expect_data_sha256, metadata["data_sha256"]),
                                    ("validation data", expect_validation_sha256, metadata["validation_data_sha256"])):
        if expected is not None and expected != actual:
            raise ValueError(f"{label} sha256 {actual} does not match the expected {expected}")


def _export(model, directory, binding):
    """Deployable model plus the settings evaluate/predict bind to (``io.load_checkpoint_config``)."""
    model.save_pretrained(directory)
    (directory / CHECKPOINT_MANIFEST).write_text(json.dumps(binding, indent=2) + "\n")


def run_training(config, data, output, *, validation=None, dry_run=False, max_samples=None,
                 expect_data_sha256=None, expect_validation_sha256=None, allow_resume_change=False):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    unknown = set(config) - set(CONFIG_SECTIONS)
    if unknown:
        raise ValueError(f"config only accepts {'/'.join(CONFIG_SECTIONS)}; unknown sections {sorted(unknown)}")
    model_config, loss_config = ModelConfig(**config.get("model", {})), LossConfig(**config.get("loss", {}))
    training = _training_config(config.get("training", {}))
    optimizer_config = _optimizer_config(config, training)
    augmentation = _augmentation_config(config)
    validation_config = _validation_config(config)
    if dry_run and model_config.backbone != "tiny":
        raise ValueError("offline dry-run requires explicit tiny backbone")
    if augmentation["given_pose_p"] and not model_config.pose_conditioning:
        raise ValueError("augmentation.given_pose_p needs model.pose_conditioning")
    resume = training["resume"]
    resolved = {"model": asdict(model_config), "loss": asdict(loss_config), "optimizer": optimizer_config,
                "augmentation": augmentation, "validation": validation_config,
                "training": {key: value for key, value in training.items() if key != "resume"},
                "max_samples": max_samples, "dry_run": dry_run}
    config_digest = hashlib.sha256(json.dumps(_digest_view(resolved), sort_keys=True, default=str).encode()).hexdigest()
    target = safe_output(output)
    metadata = {**run_metadata(data),
                "validation_data_path": str(Path(validation).resolve()) if validation else None,
                "validation_data_sha256": fingerprint(validation) if validation else None}
    _verify_expected_hashes(metadata, expect_data_sha256, expect_validation_sha256)
    samples = read_samples(data, training=True, max_samples=max_samples)
    heldout = read_samples(validation) if validation else []
    _verify_input_fingerprints(metadata)
    validation_excluded = 0
    if heldout:
        if any(r["provenance"].get("split") != "validation" for r in heldout):
            raise ValueError("model selection accepts validation only; test is reserved for final evaluation")
        ensure_disjoint(samples, heldout)
        filtered = filter_rows_by_flags(heldout, validation_config["exclude_flags"])
        validation_excluded, heldout = len(heldout) - len(filtered), filtered
        if not heldout:
            raise ValueError("every validation row carries an excluded flag")
    if loss_config.support and not model_config.support_head:
        raise ValueError("loss.support needs model.support_head")
    torch.set_num_threads(training["cpu_threads"])
    accelerator = Accelerator(cpu=training["cpu"], mixed_precision=training["mixed_precision"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"], step_scheduler_with_optimizer=False)
    set_seed(training["seed"])
    tokenizer = load_tokenizer(model_config.backbone, local_files_only=model_config.local_files_only)
    samples, rejected = _preflight(samples, tokenizer, model_config, training, loss_config)
    validation_samples, validation_rejected = _preflight(heldout, tokenizer, model_config, training, loss_config) if heldout else ([], [])
    collate = partial(collate_samples, tokenizer=tokenizer, max_length=training["max_length"], max_objects=model_config.max_objects,
                      cell_costs=_cell_cost_options(model_config, loss_config))
    rows = _Rows(samples, augmentation, training["seed"], tokenizer, training["max_length"])
    generator = torch.Generator().manual_seed(training["seed"])
    loader = DataLoader(rows, batch_size=training["batch_size"], shuffle=True, collate_fn=collate, generator=generator)
    shard = partial(_diagnostic_loader, accelerator=accelerator, batch_size=training["batch_size"], collate=collate,
                    seed=training["seed"])
    val_loader = shard(validation_samples) if heldout else None
    minimal_samples = [row for row in map(project_minimal, validation_samples) if row is not None]
    minimal_loader = shard(minimal_samples) if minimal_samples else None
    baseline_rows = validation_samples[:2000]
    baseline_loader = shard(baseline_rows) if heldout else None
    steps = 1 if dry_run else training["steps"]
    model, criterion = build_model(model_config), GeometryCriterion(loss_config)
    optimizer = torch.optim.AdamW(_parameter_groups(model, training["learning_rate"], optimizer_config["decoder_lr"]),
                                  lr=training["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, partial(_lr_factor, warmup=optimizer_config["warmup_steps"],
                                                                     total=steps, schedule=optimizer_config["schedule"]))
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    state = _TrainerState()
    accelerator.register_for_checkpointing(state)
    logs, step, epoch, skip_batches = [], 0, 0, 0
    skipped_windows, skipped_gradient_overflow_windows, skipped_nonfinite_windows = 0, 0, 0
    # The labels' score anchors the selection score from launch on. Every rank joins the reduction.
    label_collapse = _label_collapse(minimal_loader, accelerator) if minimal_loader is not None else None
    bound = {"world_size": accelerator.num_processes, "validation_data_sha256": metadata["validation_data_sha256"]}
    resumed_with_changes, resume_unverified, selection_after = {}, [], 0
    if resume is not None:
        accelerator.load_state(str(resume))
        saved = state.values
        if saved.get("config_sha256") != config_digest or saved.get("data_sha256") != metadata["data_sha256"]:
            raise ValueError("resume requires the same config (except training.resume), --max-samples and training data")
        resume_unverified = [key for key in bound if key not in saved]
        if resume_unverified and accelerator.is_main_process:
            print(f"warning: {resume} predates the resume check of {resume_unverified}; resuming without verifying them",
                  file=sys.stderr, flush=True)
        resumed_with_changes = {key: {"saved": saved[key], "current": value} for key, value in bound.items()
                                if key in saved and saved[key] != value}
        if resumed_with_changes and not allow_resume_change:
            raise ValueError(f"resume changes {resumed_with_changes}; a new world size changes the global batch and the "
                             "rank-local data position, new validation data mixes old and new validation logs; "
                             "pass --allow-resume-change to accept and record it")
        # Recorded, never refused: a crashed run of older code is resumed by a plain --resume (autorun), and only
        # its logs change (multi-rank validation counted the padded rank batches before round 5).
        if heldout and saved.get("diagnostics") != DIAGNOSTICS and saved.get("world_size", accelerator.num_processes) > 1:
            resumed_with_changes["diagnostics"] = {"saved": saved.get("diagnostics"), "current": DIAGNOSTICS}
        step, epoch, skip_batches = saved["step"], saved["epoch"], saved["batches_done"]
        # Scores on validation data the run no longer uses, or under the older diagnostics, are not comparable: either
        # restarts selection, and the restart travels with the states (absent before round 3: every score counts).
        selection_after = (step if resumed_with_changes.keys() & {"validation_data_sha256", "diagnostics"}
                           else saved.get("selection_after_step", 0))
        generator.set_state(saved["generator_state"])
        skipped_windows, skipped_gradient_overflow_windows, skipped_nonfinite_windows = (
            saved["skipped_no_objective_windows"], saved["skipped_gradient_overflow_windows"], saved["skipped_nonfinite_windows"])
        logs = list(saved["logs"])
    resumed_at = step
    binding = None  # main process: what every exported model directory is bound to
    with contextlib.ExitStack() as stack:
        if accelerator.is_main_process:
            target.mkdir(parents=True, exist_ok=False)
            tokenizer.save_pretrained(target / "tokenizer")
            provenance = {"backbone": backbone_provenance(model_config.backbone, local_files_only=model_config.local_files_only),
                          "tokenizer_sha256": fingerprint_tree(target / "tokenizer")}
            binding = {"max_length": training["max_length"], "data_sha256": metadata["data_sha256"],
                       "validation_data_sha256": metadata["validation_data_sha256"], "config_sha256": config_digest,
                       "implementation_sha256": metadata["implementation_sha256"], **provenance}
            (target / "run_manifest_start.json").write_text(json.dumps(
                {**metadata, **provenance, "config": config, "config_sha256": config_digest, "resolved_config": resolved,
                 "augmentation": augmentation, "world_size": accelerator.num_processes,
                 # Admitted counts, so the step budget (e.g. 3 epochs of 124,375 scenes) can be checked at launch.
                 "supervised_samples": len(samples), "rejected_samples": len(rejected),
                 "validation_samples": len(validation_samples), "validation_minimal_samples": len(minimal_samples),
                 "resumed_from": str(resume) if resume is not None else None, "resumed_at_step": step,
                 "resumed_with_changes": resumed_with_changes, "resume_unverified": resume_unverified,
                 "selection_metric": {**SELECTION_METRIC, "ground_truth": label_collapse}}, indent=2, default=str) + "\n")
            stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, stack.enter_context((target / "train.log").open("x")))))
            if resume is not None:
                print(json.dumps({"resumed_from": str(resume), "step": step, "epoch": epoch, "batches_done": skip_batches}), flush=True)
        accelerator.wait_for_everyone()
        baseline = None
        if val_loader is not None:
            baseline = _baseline(baseline_loader, criterion, accelerator, _category_median_sizes(samples[:20000]), model_config.yaw_bins)
            baseline["rows"] = len(baseline_rows)
            if accelerator.is_main_process:
                print(json.dumps({"baseline": baseline}), flush=True)
        model.train()
        start = time.perf_counter()
        objective, window = ObjectiveWindow(), _Window()
        while step < steps:
            epoch_start_step = step
            epoch_start_skips = skipped_gradient_overflow_windows + skipped_nonfinite_windows
            rows.epoch = epoch
            epoch_generator_state = generator.get_state()
            epoch_loader = accelerator.skip_first_batches(loader, skip_batches) if skip_batches else loader
            batches_done, skip_batches = skip_batches, 0
            for batch in epoch_loader:
                batches_done += 1
                record = None
                with accelerator.accumulate(model):
                    result = criterion(model(**model_inputs(batch)), batch)
                    if not torch.isfinite(result["loss"]):
                        raise RuntimeError("nonfinite structured loss")
                    accelerator.backward(result["loss"])
                    objective = objective.add(result["active_objective_count_local"])
                    window.add(result, batch)
                    if accelerator.sync_gradients:
                        count = objective.global_count(accelerator)
                        objective = ObjectiveWindow()
                        if not count:
                            skipped_windows += 1
                        elif accelerator.scaler is None and _nonfinite_gradients(model, accelerator):
                            # No GradScaler (bf16/no): an inf/nan gradient would poison
                            # clipping and the update. Skip the window; not a training step.
                            skipped_nonfinite_windows += 1
                        else:
                            _rescale_flushed_window_gradients(model, training["gradient_accumulation_steps"], window.microbatches)
                            accelerator.clip_grad_norm_(model.parameters(), training["clip_grad_norm"])
                            optimizer.step()
                            if accelerator.optimizer_step_was_skipped:
                                # GradScaler found nonfinite gradients and deliberately
                                # suppressed the optimizer update. Do not advance the
                                # completed-update schedule, validate or checkpoint it.
                                skipped_gradient_overflow_windows += 1
                            else:
                                step += 1
                                record = _record(window, model, optimizer, step, time.perf_counter() - start, count, accelerator)
                                scheduler.step()
                        # Clear both completed and skipped windows. Non-sync
                        # microbatches keep their gradients until this boundary.
                        optimizer.zero_grad(set_to_none=True)
                        window = _Window()
                if record is not None:
                    # Capture training gradients before step/zero, then evaluate the
                    # updated model outside accumulation so step-k validation and
                    # state-step-k describe the same weights on every rank.
                    # The final update is validated and checkpointed too (unless that interval is 0), so
                    # selection can pick the last weights when steps is not a multiple of the interval.
                    final = step >= steps
                    checkpoint = _due(step, training["checkpoint_every"]) or (final and bool(training["checkpoint_every"]))
                    validate = checkpoint or _due(step, training["validate_every"]) or (final and bool(training["validate_every"]))
                    if val_loader is not None and validate:
                        full = _validation(model, val_loader, criterion, accelerator)
                        minimal = _validation(model, minimal_loader, criterion, accelerator) if minimal_loader is not None else None
                        record = {**record, "validation": {**full, "minimal": minimal,
                                                           "selection_metric": minimal["geometry_objective_mean_of_batches"] if minimal else None}}
                    logs.append(record)
                    if accelerator.is_main_process:
                        print(json.dumps(record), flush=True)
                    if checkpoint:
                        end = epoch_loader.end_of_dataloader
                        state.values = {"step": step, "epoch": epoch + int(end), "batches_done": 0 if end else batches_done,
                                        "generator_state": generator.get_state() if end else epoch_generator_state,
                                        "skipped_no_objective_windows": skipped_windows,
                                        "skipped_gradient_overflow_windows": skipped_gradient_overflow_windows,
                                        "skipped_nonfinite_windows": skipped_nonfinite_windows, "logs": logs,
                                        "config_sha256": config_digest, "data_sha256": metadata["data_sha256"], **bound,
                                        "selection_after_step": selection_after, "diagnostics": DIAGNOSTICS}
                        accelerator.save_state(str(target / f"state-step-{step}"))
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            for old in sorted(target.glob("state-step-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))[:-KEEP_STATES]:
                                shutil.rmtree(old)
                        if training["export_model_every_checkpoint"] and accelerator.is_main_process:
                            _export(accelerator.unwrap_model(model), target / f"model-step-{step}", {**binding, "step": step})
                if step >= steps:
                    break
            epoch += 1
            if step == epoch_start_step:
                if skipped_gradient_overflow_windows + skipped_nonfinite_windows > epoch_start_skips:
                    raise RuntimeError("epoch contains active objectives but every attempted optimizer update was skipped after nonfinite gradients")
                raise RuntimeError("epoch contains no active objective; no supervised optimizer update is possible")
        accelerator.wait_for_everyone()
        _verify_input_fingerprints(metadata, accelerator)
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            _export(unwrapped, target / "model", {**binding, "step": step})
            scored = [(r["validation"]["selection_metric"], r["step"]) for r in logs
                      if (r.get("validation") or {}).get("selection_metric") is not None and r["step"] > selection_after]
            best = min(scored) if scored else None
            manifest = {**metadata, **provenance, "config": config, "config_sha256": config_digest, "model": asdict(model_config),
                "loss": asdict(loss_config), "training": {**training, "resume": str(resume) if resume is not None else None},
                "optimizer": optimizer_config, "augmentation": augmentation, "validation": validation_config,
                "lr_schedule": "linear warmup over warmup_steps updates, then cosine from lr to 10% of lr at the final update; constant keeps lr",
                "parameter_groups": "backbone (LoRA/full) at learning_rate; projection/slots/decoder/heads at decoder_lr",
                "baselines": baseline, "offline_smoke": model_config.backbone == "tiny", "dry_run": dry_run,
                "steps_completed": step, "resumed_from": str(resume) if resume is not None else None, "resumed_at_step": resumed_at,
                "resumed_with_changes": resumed_with_changes, "resume_unverified": resume_unverified,
                "supervised_samples": len(samples), "validation_samples": len(validation_samples),
                "validation_minimal_samples": len(minimal_samples),
                "selection_metric": {**SELECTION_METRIC, "ground_truth": label_collapse, "best_after_step": selection_after,
                                     "best": {"value": best[0], "step": best[1]} if best else None},
                "validation_excluded_flagged": validation_excluded, "augmentation_fallbacks": rows.fallbacks,
                "skipped_no_objective_windows": skipped_windows,
                "skipped_gradient_overflow_windows": skipped_gradient_overflow_windows,
                "skipped_nonfinite_windows": skipped_nonfinite_windows,
                "rejected": rejected, "validation_rejected": validation_rejected,
                "subset_limit": max_samples, "world_size": accelerator.num_processes,
                "optimizer_step_policy": "update iff global accumulated-window eligible objective count is positive and every gradient is finite; zero numeric loss is eligible",
                "reduction": "complete-field valid instances; fixed coordinate sum divided by 3; global valid count per microbatch; mean over actual flushed-window microbatches",
                "elapsed_s": time.perf_counter() - start, "trainable_parameters": sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)}
            (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
            (target / "training_log.json").write_text(json.dumps(logs, indent=2) + "\n")
    accelerator.end_training()
    return logs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/structured.json")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", type=Path, help="state-step-<n> directory of an earlier run; output must be a new directory")
    parser.add_argument("--allow-resume-change", action="store_true",
                        help="accept a resume under a different world size or validation file; both manifests record it")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expect-data-sha256", help="abort unless --data has exactly this sha256")
    parser.add_argument("--expect-validation-sha256", help="abort unless --validation has exactly this sha256")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    config = {**config, "model": {**config.get("model", {})}, "training": {**config.get("training", {})}}
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    for key in ("steps", "max_length"):
        if getattr(args, key) is not None:
            config["training"][key] = getattr(args, key)
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    if args.cpu:
        config["training"]["cpu"] = True
    run_training(config, args.data, args.output, validation=args.validation, dry_run=args.dry_run, max_samples=args.max_samples,
                 expect_data_sha256=args.expect_data_sha256, expect_validation_sha256=args.expect_validation_sha256,
                 allow_resume_change=args.allow_resume_change)


if __name__ == "__main__":
    main()
