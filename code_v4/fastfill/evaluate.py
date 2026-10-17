"""Generate layouts for held-out rooms and score them.

    python -m fastfill.evaluate --model outputs/ff-v1-merged --rooms data/fastfill_v1/test_rooms.jsonl --out eval/ff-v1
    python -m fastfill.evaluate --gt-only --rooms data/fastfill_v1/test_rooms.jsonl --out eval/gt   # reference layouts
    python -m fastfill.evaluate                     # self-check

Rooms come from fastfill.build (already canonical: ids and order as the model saw them in training).
Every request is counted: unparsable or incomplete outputs, and prompts of --max_len tokens or more (prompt_too_long,
never generated), are invalid and fail all their constraints (no retries, no filtering); identical requests are scored
once. Metrics are reported per source and pooled separately for
in-distribution sources and held-out (eval-only) sources.
Validity, 1 mm containment, acceptance, per-object and collision rates and OOR are reported next to their GT value
(`*_gt`): the reference layout of the same rooms scored exactly like the model's answer (repair and serve.py's
acceptance included), over the same rows (all requests, or the complete answers for the per-object and collision
rates). Constraints hold on the reference by construction, and it always parses.
Uses vLLM when installed, else HF generate (up to --batch rows of equal token length). Greedy decoding; each answer may use the context left after its prompt
(--max_len, the model's sequence limit), so nothing is truncated below what training saw.
"""
import argparse
import collections
import copy
import itertools
import json
import os
import sys

from shapely.geometry import Polygon

from fastfill.interface import snap_inside
from fastfill.scene import apply, footprint, gt_placements, messages, parse, prompt_text
from fastfill.validate import check, holds

BUCKETS = ("main", "severe", "tuck", "contained", "fixed")


def generate(model, prompts, max_len, batch):
    """One text per prompt; None for a prompt of max_len tokens or more (nothing fits after it and vLLM raises on it),
    which score() records as prompt_too_long."""
    res = [None] * len(prompts)
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model, max_model_len=max_len)
        tok = llm.get_tokenizer()
        fit = [(i, max_len - len(tok(p)["input_ids"])) for i, p in enumerate(prompts)]
        fit = [(i, n) for i, n in fit if n > 0]
        outs = llm.generate([prompts[i] for i, _ in fit], [SamplingParams(temperature=0, max_tokens=n) for _, n in fit])
        for (i, _), o in zip(fit, outs):
            res[i] = o.outputs[0].text
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
        lengths = sorted((len(tok(p, add_special_tokens=False)["input_ids"]), i) for i, p in enumerate(prompts))
        # A shared generation limit is safe only for equal-length prompts; padding otherwise steals short rows' budget.
        for length, group in itertools.groupby(lengths, key=lambda row: row[0]):
            if length >= max_len:
                continue
            fit = [i for _, i in group]
            for k in range(0, len(fit), batch):
                idx = fit[k:k + batch]
                enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(m.device)
                out = m.generate(**enc, max_new_tokens=max_len - length, do_sample=False)
                for i, t in zip(idx, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                    res[i] = t
    return res


def geometry(room):
    c = check(room)
    return {"valid": c["valid"], "oob_objects": len(c["oob"]), "support_fail": len(c["support_fail"]),
            "ceiling_objects": len(c["ceiling"]), "oor": c["oor"], "coll_fixed": len(c["fixed_collisions"]),
            **{f"coll_{k}": len(c["collisions"][k]) for k in BUCKETS[:4]}}


def score(room, text, error=None):
    """`text` None scores the reference layout (--gt-only, qa_report); `error` (prompt_too_long) marks a request the
    model never answered, counted as a failed request like an unparsable answer. r["gt"] = judge() of the reference."""
    ids = [o["id"] for o in room["objects"]]
    gt = judge(room, gt_placements(room))
    r = {"uid": room["uid"], "source": room["source"], "group": room["group"], "n": len(ids),
         "held_out": bool(room.get("meta", {}).get("eval_only")), "gt": gt}
    pl, err = ({}, error) if error else parse(text, ids) if text is not None else (None, None)
    r["error"] = err
    r["parsed"] = not error and err not in ("truncated_or_bad_json", "bad_json")
    cons = room.get("constraints") or []
    if err is not None:        # a failed request fails every constraint it was asked to satisfy
        return {**r, "valid": False, "constraints": [[c[0], False] for c in cons],
                "constraints_repaired": [[c[0], False] for c in cons]}
    return {**r, **(gt if text is None else judge(room, pl))}


def judge(room, pl):
    """One complete answer (parse() form) scored raw, and after the interface's <= 10 cm push back inside (reported
    apart: the raw proposal is the model's result)."""
    cons = room.get("constraints") or []
    pred = apply(room, pl)
    r = geometry(pred)
    r["constraints"] = [[c[0], holds(c, pred["objects"], room["boundary"])] for c in cons]
    floor = Polygon(room["boundary"])
    rep = copy.deepcopy(pl)
    snap_inside(rep, room, floor, cons, room["boundary"])     # never breaks a constraint that held (all are hard)
    fixed = apply(room, rep)
    c = check(fixed)
    r["inside_1mm"], r["inside_1mm_repaired"] = inside(pred, floor), inside(fixed, floor)
    r["valid_repaired"] = c["valid"]
    r["constraints_repaired"] = [[k[0], holds(k, fixed["objects"], room["boundary"])] for k in cons]
    r["repaired_objects"] = sum(rep[i]["pos"] != pl[i]["pos"] for i in pl)
    # what serve.py accepts: valid after repair, every object within 1 mm of the walls, nothing through a non-window
    # fixed box, every constraint holding
    r["accepted"] = c["valid"] and r["inside_1mm_repaired"] and not c["fixed_blocking"] \
        and all(v for _, v in r["constraints_repaired"])
    return r


def inside(room, floor, tol=1e-3):
    """Every object, on others too, inside the boundary within 1 mm (EmbodiedGen's containment rule)."""
    grown = floor.buffer(tol)
    return all(grown.covers(Polygon(footprint(o))) for o in room["objects"])


def summarize(rows):
    def agg(rs):
        n = len(rs)
        if not n:
            return None
        ok = [r for r in rs if r["error"] is None]
        objs = sum(r["n"] for r in ok)
        s = {"rooms": n, "groups": len({r["group"] for r in rs}),
             "parse_rate": sum(r["parsed"] for r in rs) / n, "complete_rate": len(ok) / n,
             "errors": dict(collections.Counter(r["error"] for r in rs if r["error"]))}
        # over every request (a failed one counts as failing); *_repaired after the interface's deterministic repair
        # (not model skill); accepted = serve.py's 200, repair included; GT: the reference layouts of the same rooms
        for k, key in (("valid_rate", "valid"), ("valid_rate_repaired", "valid_repaired"),
                       ("inside_1mm_rate", "inside_1mm"), ("inside_1mm_rate_repaired", "inside_1mm_repaired"),
                       ("accepted_rate", "accepted")):
            s[k] = sum(r.get(key, False) for r in rs) / n
            s[f"{k}_gt"] = sum(r["gt"][key] for r in rs) / n
        # geometry is measured on complete answers only; with none it is unknown (None), not a perfect 0
        rate = lambda num, den: num / den if den else None
        for k, key in (("repaired_object_rate", "repaired_objects"), ("oob_object_rate", "oob_objects"),
                       ("support_fail_rate", "support_fail"), ("ceiling_object_rate", "ceiling_objects")):
            s[k] = rate(sum(r[key] for r in ok), objs)
            s[f"{k}_gt"] = rate(sum(r["gt"][key] for r in ok), objs)
        for k in BUCKETS:   # share of rooms with a collision of this kind: model vs GT on the same (complete) rooms
            name = "fixed_collision" if k == "fixed" else k
            s[f"rooms_with_{name}"] = rate(sum(r[f"coll_{k}"] > 0 for r in ok), len(ok))
            s[f"rooms_with_{name}_gt"] = rate(sum(r["gt"][f"coll_{k}"] > 0 for r in ok), len(ok))
        s["mean_oor"] = rate(sum(r["oor"] for r in ok), len(ok))
        s["mean_oor_gt"] = rate(sum(r["gt"]["oor"] for r in ok), len(ok))
        by_c, by_r = collections.defaultdict(list), collections.defaultdict(list)
        for r in rs:                   # every request, failed ones included
            for t, v in r.get("constraints", []):
                by_c[t].append(v)
            for t, v in r.get("constraints_repaired", []):
                by_r[t].append(v)
        s["constraint_satisfaction"] = {t: sum(v) / len(v) for t, v in sorted(by_c.items())}
        s["constraint_satisfaction_repaired"] = {t: sum(v) / len(v) for t, v in sorted(by_r.items())}
        return s
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    # eval-only sources (SpatialGen, SceneSmith) are out of distribution: never pooled with the rest
    return {"in_dist": agg([r for r in rows if not r["held_out"]]), "held_out": agg([r for r in rows if r["held_out"]]),
            **{k: agg(v) for k, v in sorted(by.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", required=True, help="dev_rooms.jsonl / test_rooms.jsonl from fastfill.build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="merged checkpoint dir or HF id")
    ap.add_argument("--gt-only", action="store_true", help="score the reference layouts themselves")
    ap.add_argument("--max_len", type=int, default=40960, help="model sequence limit; an answer may use what its prompt leaves")
    ap.add_argument("--batch", type=int, default=1, help="HF generate only: prompts per forward pass (at 40,960 context "
                    "the KV cache of more than one row does not fit on one GPU)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    uniq = {}                          # identical requests (same room text and constraints) are scored once
    for line in open(a.rooms):
        r = json.loads(line)
        uniq.setdefault(messages(r, r.get("constraints"), with_target=False)[1]["content"], r)
    rooms = list(uniq.values())[:a.limit]
    if a.gt_only:
        texts = [None] * len(rooms)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = [prompt_text(tok, messages(r, r.get("constraints"), with_target=False)) for r in rooms]
        texts = generate(a.model, prompts, a.max_len, a.batch)
    # a None text after generation is a prompt the model could not take (>= --max_len tokens): a failed request
    rows = [score(r, t, "prompt_too_long" if t is None and not a.gt_only else None) for r, t in zip(rooms, texts)]
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/predictions.jsonl", "w") as f:
        for r, t in zip(rows, texts):
            f.write(json.dumps({**r, "output": t}, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    json.dump(summary, open(f"{a.out}/metrics.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("in_dist", "held_out")}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:]:
        main()
    else:
        from fastfill.scene import target_json
        # the reference chair stands 3 cm through the south wall facing a box 28.5 deg off its front: not inside 1 mm,
        # repaired by (-4, +3) cm (the shortest push would break the faces constraint), then accepted
        room = {"uid": "u", "source": "S", "group": "g", "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "boundary_type": "polygon", "height": 2.8, "fixed": [], "constraints": [["faces", "chair_1", "box_1"]],
                "objects": [{"id": "chair_1", "category": "chair", "size": [.5, .5, .9], "pos": [1, .22, 0], "yaw": 0.0},
                            {"id": "box_1", "category": "box", "size": [.06, .06, .3], "pos": [1.35, .03, 0], "yaw": 0.0}]}
        gt = score(room, None)
        assert not gt["inside_1mm"] and gt["inside_1mm_repaired"] and gt["repaired_objects"] == 1 and gt["accepted"], gt
        assert all(gt[k] == v for k, v in gt["gt"].items()) and score(room, target_json(room))["accepted"]
        # a model answer 20 cm through the west wall (oob, not repaired, faces lost) and a prompt that did not fit
        off = target_json({**room, "objects": [{**room["objects"][0], "pos": [.05, 1.5, 0]}, room["objects"][1]]})
        rows = [gt, score(room, off), score(room, None, "prompt_too_long")]
        assert rows[1]["oob_objects"] == 1 and rows[1]["gt"] == gt["gt"] and rows[2]["gt"] == gt["gt"]
        s = summarize(rows)["in_dist"]
        want = {"valid_rate": (1 / 3, 1), "accepted_rate": (1 / 3, 1), "inside_1mm_rate": (0, 0),
                "inside_1mm_rate_repaired": (1 / 3, 1), "valid_rate_repaired": (1 / 3, 1),
                "oob_object_rate": (1 / 4, 0), "repaired_object_rate": (1 / 4, 2 / 4), "support_fail_rate": (0, 0)}
        assert all((s[k], s[f"{k}_gt"]) == v for k, v in want.items()), s
        assert s["constraint_satisfaction"] == {"faces": 1 / 3} and s["rooms_with_main_gt"] == 0, s
        # --gt-only: every rate equals its GT
        s = summarize([gt])["in_dist"]
        assert all(s[k] == s[f"{k}_gt"] for k in s if f"{k}_gt" in s), s
        print("evaluate.py self-check ok")
