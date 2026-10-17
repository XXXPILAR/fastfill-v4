"""Configuration-specific target eligibility; never clamp or rewrite labels."""
from __future__ import annotations

import math

import torch

from .objective import supervision_masks


def size_target_conflicts(batch, model_config, loss_config):
    """Report learned, supervised axes outside reference * exp(+-limit).

    Complete-field masks follow the criterion. Fixed request dimensions bypass
    the learned size head. Missing labels are selected out before log arithmetic;
    disabled size/box objectives impose no size-target range requirement.
    A 1e-6 log tolerance only handles float32 rounding at the exact endpoints.
    """
    masks = supervision_masks(batch)
    slots = torch.zeros_like(batch["slot_mask"])
    if loss_config.size > 0:
        slots = slots | masks["size"]
    if loss_config.box > 0:
        slots = slots | masks["box"]
    fixed = batch.get("fixed_size_mask", torch.zeros_like(batch["validity"]["size"]))
    coordinates = slots.unsqueeze(-1) & ~fixed
    conflicts = []
    for b, i, q in coordinates.nonzero().tolist():
        target = float(batch["targets"]["size"][b, i, q])
        reference = model_config.size_reference[q]
        log_ratio = math.log(target) - math.log(reference)
        limit = model_config.size_log_limit
        if abs(log_ratio) > limit + 1e-6:
            conflicts = conflicts + [{"batch_index": b, "object_id": batch["objects"][b][i]["id"],
                                      "dimension": ("w", "d", "h")[q], "target_m": target,
                                      "minimum_m": reference * math.exp(-limit),
                                      "maximum_m": reference * math.exp(limit)}]
    return conflicts
