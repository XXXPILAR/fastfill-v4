"""Remasking decoding (Stage 2 of masked layout training): place a request's objects over several rounds.

MaskGIT / SceneNAT confidence-first iterative decoding at the object level, for a ``pose_conditioning`` model:

1. ``model.encode`` once (backbone + slot pooling); every round only ``model.decode`` runs.
2. Each round decodes every slot with the poses committed so far given (``given_pose`` / ``given_pose_mask``,
   ``masked_layout.pose_features`` of the committed pose, as in training). Each open slot picks its argmax cell;
   its confidence is the softmax probability of that cell under the raw logits (SceneNAT's score of the sampled
   token). The most confident open slots are committed so that ``floor(N * cos(pi/2 * (r + 1) / R))`` slots stay
   open after round ``r`` of ``R`` (at least one commit per round; none open after the last round).
3. With ``legal`` the cell choice is restricted, per slot, to its least-cost cells under ``cell_costs`` (the
   expected-geometry costs: footprint outside the floor, a door's clearance zone for floor objects that can block,
   the room's fixed objects, and, round by round, the footprints of the objects already committed whose height interval overlaps the
   slot's, as the scorer's collision check counts them). Costs use the slot's own predicted size, yaw and z. Slots of one round are committed in confidence order and each re-checks its cell against the
   footprints committed just before it. A slot whose every cell costs something takes its cheapest cells. The
   bottom centre decoded in the chosen cell is kept when it is as cheap as the cell's best candidate position
   (``cell_costs.cell_positions``), else it snaps to the nearest least-cost candidate.

4. With ``retract`` (SceneNAT's sampler) a committed slot can be taken back: each round ranks the open slots by
   their confidence now and the committed ones by their confidence when committed, keeps the most confident so
   that the schedule's number stays open, and masks the others again (pose, footprint and score dropped), to be
   re-predicted with the poses kept so far. Slots given by the request are never retracted.
5. A ``support_head`` model's support prediction (floor or not) stands in for an undeclared ``support_parent``
   in deciding which objects block doors, and its ``support_logits`` ride along with the committed rows.

The result is a predictions dict like the model's: every slot carries the rows (cell logits restricted to its
legal cells, residuals, z, size, yaw logits and residuals) of the round that committed it, so
``evaluate.serialize_predictions(..., grid_decode="argmax")`` writes exactly the committed poses. ``rounds=1``
without ``legal`` is the plain argmax decode.

Pose features need each slot's yaw symmetry order and K1 flag; batches collated from labelled rows carry them
(``yaw_symmetry_order`` / ``size_axis_swap_allowed``), a bare condition gets the collate defaults (order 1, no swap).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from fastfill.v2.cell_costs import (THIN_HEIGHT_M, RoomRaster, box_corners, cell_cost_maps, cell_cost_options, cell_positions,
                                   footprint_offsets, vertical_extent)
from fastfill.v2.geometry import decode_grid_position, decode_yaw
from fastfill.v2.masked_layout import pose_features

FLOOR_PREDICTION_TOLERANCE_M = .15  # a predicted bottom this close to the floor stands on it
ILLEGAL_LOGIT = -1e4
ROW_KEYS = ("position_normalized", "size", "yaw_logits", "yaw_residuals", "position_cell_logits", "position_cell_residuals")


def open_after(total: int, round_index: int, rounds: int) -> int:
    """Slots left open after round ``round_index`` (0-based) of ``rounds``: MaskGIT's cosine schedule."""
    return math.floor(total * math.cos(math.pi / 2 * (round_index + 1) / rounds))


def _point_cost(raster: RoomRaster, points: np.ndarray, size, yaw: float, blocks: bool, z_range, options: dict) -> np.ndarray:
    """Cost (as in ``cell_cost_maps``) of the footprint with its bottom centre at each of ``points`` (k x 2)."""
    samples = points[:, None, :] + footprint_offsets(size[:2], yaw, options)[None]
    outside, door = raster.lookup(samples)
    cost = outside.mean(-1) + door.mean(-1) * blocks
    occupied = raster.occupied(z_range)
    if occupied is not None:
        cost = cost + raster.lookup_occupied(samples, occupied).mean(-1)
    return cost


def _snap(raster, candidates, decoded, size, yaw, blocks, z_range, options) -> np.ndarray:
    """The decoded bottom centre if it costs no more than the cell's best candidate, else the nearest of the
    least-cost candidates: the cell was chosen for some legal position in it, the residual may miss it."""
    points = np.vstack((decoded[None], candidates))
    cost = _point_cost(raster, points, size, yaw, blocks, z_range, options)
    best = cost.min()
    if cost[0] <= best + 1e-6:
        return decoded
    legal = np.flatnonzero(cost <= best + 1e-6)
    return points[legal[np.argmin(np.linalg.norm(points[legal] - decoded, axis=1))]]


def _blocks(condition_object: dict, z_world: float, floor_z, height: float, predicted_floor: bool | None = None) -> bool:
    """Floor object that can block a door or collide with other floor objects: a declared support wins, then the
    support head's prediction (floor or not), else a predicted bottom near the floor; thin objects (carpets) never
    block."""
    declared = condition_object.get("support_parent")
    if declared is not None:
        on_floor = declared == "floor"
    elif predicted_floor is not None:
        on_floor = predicted_floor
    else:
        on_floor = isinstance(floor_z, (int, float)) and abs(z_world - floor_z) <= FLOOR_PREDICTION_TOLERANCE_M
    return on_floor and height > THIN_HEIGHT_M


@torch.no_grad()
def remask_predictions(model, batch: dict, *, rounds: int = 4, legal: bool = True, retract: bool = False,
                       door_depth_m: float = .8, stats: dict | None = None) -> dict:
    """Decode a collated batch over ``rounds`` rounds (see the module docstring). ``stats`` (a dict) receives
    ``decode_calls``, ``commits_per_round`` and ``retractions_per_round``."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if model.config.position_head != "grid_residual":
        raise ValueError("remasking decoding needs the grid_residual position head")
    if rounds > 1 and not model.config.pose_conditioning:
        raise ValueError("remasking decoding over several rounds needs a model built with pose_conditioning")
    model.eval()
    grid = model.config.position_grid
    slot_mask = batch["slot_mask"].bool()
    b, n = slot_mask.shape
    encoded = model.encode(batch["input_ids"], batch["attention_mask"], batch["object_spans"], batch["slot_mask"])
    fixed = {key: batch[key] for key in ("fixed_position_normalized", "fixed_position_mask", "fixed_size", "fixed_size_mask")
             if key in batch}
    device = slot_mask.device
    pinned = batch.get("given_pose_mask", torch.zeros_like(slot_mask)).bool() & slot_mask  # given by the request
    committed = pinned.clone()
    features = batch.get("given_pose", torch.zeros((b, n, 8), device=device)).clone().float()
    out = None
    options = cell_cost_options({"grid": grid, "door_depth_m": door_depth_m})
    rasters = [RoomRaster(c["room"], options, .1) if legal else None for c in batch["conditions"]]
    base_blockers = [list(r.blockers) if legal else [] for r in rasters]  # the room's fixed objects
    footprints = [{} for _ in range(b)]  # committed slot -> its blocker
    candidates = [cell_positions(c["room"], options) if legal else None for c in batch["conditions"]]
    origin, scale = batch["origin"].cpu().double(), batch["scale"].cpu().double()
    symmetry = batch.get("yaw_symmetry_order", torch.ones((b, n), dtype=torch.long)).to(device)
    swap = batch.get("size_axis_swap_allowed", torch.zeros((b, n), dtype=torch.bool)).to(device)
    total = (slot_mask & ~pinned).sum(1).tolist()
    score = [{} for _ in range(b)]  # committed slot -> its confidence when committed (SceneNAT keeps it)
    log = {"decode_calls": 0, "commits_per_round": [], "retractions_per_round": []}
    for r in range(rounds):
        kwargs = dict(fixed)
        if model.config.pose_conditioning:
            kwargs.update(given_pose=features.to(encoded["slots"].dtype), given_pose_mask=committed)
        pred = model.decode(encoded, **kwargs)
        log["decode_calls"] += 1
        if out is None:
            out = {key: pred[key].clone() for key in (*ROW_KEYS, *(("support_logits",) if "support_logits" in pred else ()))}
            out["slot_mask"] = pred["slot_mask"]
        raw = pred["position_cell_logits"].float()
        yaw = decode_yaw(pred["yaw_logits"].float(), pred["yaw_residuals"].float())
        support = pred.get("support_logits")
        commits = retractions = 0
        for row in range(b):
            open_slots = [i for i in range(n) if slot_mask[row, i] and not committed[row, i]]
            if not open_slots:
                continue
            objects, room = batch["conditions"][row]["objects"], batch["conditions"][row]["room"]
            size = pred["size"][row].float().cpu().double().numpy()
            yaw_row = yaw[row].cpu().double().numpy()
            z_world = (pred["position_normalized"][row, :, 2].float().cpu().double() * scale[row, 2] + origin[row, 2]).numpy()
            predicted_floor = (support[row].argmax(-1) == 0).tolist() if support is not None else [None] * n
            blocks = {i: _blocks(objects[i], z_world[i], room.get("floor_z_m"), size[i, 2], predicted_floor[i])
                      for i in open_slots}
            z_range = {i: vertical_extent(z_world[i], size[i, 2], options) for i in open_slots}
            version = [0]  # bumped whenever the occupied footprints change

            def restricted(i):
                if not legal:
                    return raw[row, i]
                costs, _ = cell_cost_maps(room, [(size[i, :2].tolist(), float(yaw_row[i]), blocks[i], z_range[i])], options,
                                          raster=rasters[row])
                cost = torch.from_numpy(costs[0]).to(raw.device)
                return torch.where(cost <= cost.min() + 1e-6, raw[row, i], torch.full_like(raw[row, i], ILLEGAL_LOGIT))

            if retract and legal and footprints[row]:
                # rank the open slots against the room and its fixed objects only: committed footprints may be
                # retracted below, the slots that commit are re-checked against the ones kept
                rasters[row].blockers = list(base_blockers[row])
                version[0] += 1
            logits = {i: restricted(i) for i in open_slots}
            seen = version[0]
            probability = raw[row].softmax(-1)
            confidence = {i: float(probability[i, int(logits[i].argmax())]) for i in open_slots}
            keep_open = 0 if r == rounds - 1 else open_after(total[row], r, rounds)
            if retract:
                # SceneNAT: the kept set is the most confident of the open slots (scored now) and the committed ones
                # (scored when committed); a committed slot outside it is masked again and re-predicted later
                pool = {**score[row], **confidence}
                keep = set(sorted(pool, key=lambda k: -pool[k])[:max(0, len(pool) - keep_open)])
                for i in [i for i in score[row] if i not in keep]:
                    committed[row, i] = False
                    features[row, i] = 0.
                    del score[row][i]
                    footprints[row].pop(i, None)
                    retractions += 1
                if legal and footprints[row]:
                    rasters[row].blockers = base_blockers[row] + list(footprints[row].values())
                    version[0] += 1
                chosen = sorted((i for i in open_slots if i in keep), key=lambda k: -confidence[k])
            else:
                count = max(1, len(open_slots) - keep_open)
                chosen = sorted(open_slots, key=lambda k: -confidence[k])[:count]
            for i in chosen:
                if legal and version[0] != seen:
                    logits[i] = restricted(i)  # footprints committed (or retracted) since the logits were computed
                cell_logits = logits[i]
                xy = decode_grid_position(cell_logits[None], pred["position_cell_residuals"][row, i][None].float(), grid)[0]
                if legal:
                    lo, sc = origin[row, :2].numpy(), scale[row, :2].numpy()
                    world = _snap(rasters[row], candidates[row][int(cell_logits.argmax())], xy.cpu().double().numpy() * sc + lo,
                                  size[i], float(yaw_row[i]), blocks[i], z_range[i], options)
                    xy = torch.from_numpy((world - lo) / sc).to(xy)
                position = torch.cat((xy.to(pred["position_normalized"].dtype), pred["position_normalized"][row, i, 2:]))
                for key in out:
                    if key != "slot_mask":
                        out[key][row, i] = pred[key][row, i]
                out["position_cell_logits"][row, i] = cell_logits.to(out["position_cell_logits"].dtype)
                out["position_normalized"][row, i] = position
                features[row, i] = pose_features(position[None].float(), pred["size"][row, i][None].float(),
                                                 yaw[row, i][None], symmetry[row, i][None], swap[row, i][None])[0]
                committed[row, i] = True
                score[row][i] = confidence[i]
                commits += 1
                if legal:
                    world = position[:2].cpu().double().numpy() * scale[row, :2].numpy() + origin[row, :2].numpy()
                    rasters[row].block(box_corners(world, size[i, :2], float(yaw_row[i])), z_range[i])
                    footprints[row][i] = rasters[row].blockers[-1]
                    version[0] += 1
        log["commits_per_round"].append(commits)
        log["retractions_per_round"].append(retractions)
        if not (slot_mask & ~committed).any():
            break
    if stats is not None:
        stats.update(log)
    return out
