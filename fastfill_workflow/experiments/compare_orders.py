"""Three orders of checking and repairing the same one-shot draft, no API: whole room, object by object, layer by layer.

Scope: every variant starts from the SAME one-shot draft (step ③ placed every object in one LLM call, seeing all of
them), so this compares only the order and scope of repair, not placing objects one by one or group by group; that
comparison needs new drafts per mode (``run_orders.py``, with the API).

Every variant starts from a saved run's step-③ drafts (``run_draft.py``'s room_XXX.json; plan rebuilt as in
``replay_critic.py``: supports, groups and priority, the groups' near relations, no LLM soft relations), uses the same
critic (``workflow.critic``: doors hard, templates from the even half), the same program proposals (``workflow.loop``,
no GPT, sizes fixed), the same total program-search budget per room, and the same critic-judged polish at the end;
only the order and the scope of checking and repair differ:

  A  whole room       the critic loop on every object at once (what ``run_draft_workflow(critic=True)`` does)
  B  object by object objects enter in the solver's order (supports first, then the plan's priority); each enters at
                      its draft pose (an object on a moved parent at its drafted offset on the parent), is checked
                      against what is already placed, and only it is repaired (hard issues, then its soft costs);
                      it is then frozen
  C  layer by layer   layer 1: the objects on the floor or a wall, checked and repaired together (scopes as A)
                      without the objects standing on them, then frozen; layer 2: one supporting object at a time
                      (in the solver's order), its children enter at their drafted offsets and only they are
                      repaired, then frozen

The partial rooms of B and C are checked as rooms holding only the objects placed so far (the input's constraints
and step ①'s relations whose objects are all there). The final layout of every variant is assessed by the full
critic. Writes <out>/<variant>/predictions.jsonl (in --rooms order), <out>/rooms.json (per room and variant: critic key
draft -> final, seconds) and <out>/summary.md (overall and per size bin: hard-valid rooms under the old and the new
door rule, hard violations left, REL / ACC / OPC / COL with --templates, seconds).

    python experiments/compare_orders.py --run experiments/test50_v4_server --rooms experiments/test50_rooms.json \\
        --rows ../test300/rows.jsonl --templates experiments/quality50/eval_templates_odd.json \\
        --out experiments/compare_orders --workers 8
"""
import argparse, json, math, statistics, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow.catalog import SizeCatalog  # noqa: E402
from workflow.contract import check_request, fill_room_defaults  # noqa: E402
from workflow.critic import Critic, load_templates  # noqa: E402
from workflow.geometry import Geometry  # noqa: E402
from workflow.local_repair import FixedSizes  # noqa: E402
from workflow.loop import FocusedLoop, Loop, LoopBudget  # noqa: E402
from workflow.orders import critic_geometry, sub_room  # noqa: E402
from workflow.planning import Plan, _group_relations, _required_parents  # noqa: E402
from workflow.refine import refine  # noqa: E402
from workflow.solver import _order, _pose, _wrap  # noqa: E402

CATALOG = HERE / "size_catalog.json"
VARIANTS = ("A", "B", "C")
NAMES = {"A": "A whole room", "B": "B object by object", "C": "C layer by layer"}
BINS = ((1, 20, "1-20 objects"), (21, 10 ** 6, "21+ objects"))


class Room:
    def __init__(self, row, record, templates):
        condition, _ = fill_room_defaults(row["condition"])
        self.condition = check_request(condition)
        d = record["diagnostics"]
        self.plan = Plan(parents=dict(d["support_parents"]), groups=d.get("functional_groups") or [],
                         priority={k: int(v) for k, v in (d.get("priority") or {}).items()})
        _group_relations(self.plan)
        self.sizes = d["sizes"]
        self.draft = {b["id"]: b for b in d["draft_layout"]["objects"]}
        self.catalog = SizeCatalog(CATALOG)
        self.templates = templates
        self.locked = _required_parents(self.condition)
        self.geometry = critic_geometry(self.condition, self.plan)
        self.critic = Critic.for_plan(self.geometry, self.plan, templates)

    def sub(self, ids):
        """Geometry and critic of a room holding only ``ids`` (orders.sub_room)."""
        return sub_room(self.condition, self.plan, ids, self.templates)

    def entering(self, ident, placed):
        """``ident``'s draft box, carried to its parent's current pose when the parent has moved since the draft."""
        box, parent = self.draft[ident], self.plan.parents[ident]
        if parent not in placed or parent not in self.draft:
            return box
        old, new = self.draft[parent], placed[parent]
        turn = new["yaw_rad"] - old["yaw_rad"]
        dx = box["bottom_center_m"][0] - old["bottom_center_m"][0]
        dy = box["bottom_center_m"][1] - old["bottom_center_m"][1]
        c, s = math.cos(turn), math.sin(turn)
        x, y = new["bottom_center_m"][0] + c * dx - s * dy, new["bottom_center_m"][1] + s * dx + c * dy
        trial = _pose(ident, box["target_size_local_m"], x, y, 0, box["yaw_rad"] + turn, parent)
        z = self.geometry._support_height(trial, parent, new)
        return _pose(ident, box["target_size_local_m"], x, y, z, _wrap(box["yaw_rad"] + turn), parent)

    def loop(self, geometry, critic, movable, seconds, max_attempts):
        cls = Loop if movable is None else FocusedLoop
        kwargs = {} if movable is None else {"movable": movable}
        return cls(geometry, self.plan, self.sizes, self.catalog, critic, locked_parents=self.locked, ask=None,
                   budget=LoopBudget(gpt_repairs=0, max_attempts=max_attempts, search_seconds=max(seconds, 0.)),
                   **kwargs)


def run_variant(room, variant, budget_s):
    spent, attempts, accepted = 0., 0, 0

    def run(loop, placed):
        nonlocal spent, attempts, accepted
        placed, info = loop.run(placed)
        spent += loop.search_s
        attempts += len(info["attempts"])
        accepted += sum(a["accepted"] for a in info["attempts"])
        return placed

    order = _order(room.geometry, FixedSizes(room.sizes, room.catalog), room.plan, set(room.geometry.requests))
    if variant == "A":
        placed = run(room.loop(room.geometry, room.critic, None, budget_s, 24), dict(room.draft))
    elif variant == "B":
        placed = {}
        for k, ident in enumerate(order):
            placed[ident] = room.entering(ident, placed)
            geometry, critic = room.sub(placed)
            share = (budget_s - spent) / (len(order) - k)  # the rest of the budget, shared by the objects left
            placed = run(room.loop(geometry, critic, {ident}, share, 4), placed)
    else:
        layer1 = [i for i in order if room.plan.parents[i] in ("floor", "wall")]
        placed = {i: room.draft[i] for i in layer1}
        geometry, critic = room.sub(placed)
        groups = [p for p in order if any(room.plan.parents[c] == p for c in order)] + \
            sorted({room.plan.parents[c] for c in order} & set(room.geometry.fixed))
        placed = run(room.loop(geometry, critic, set(layer1), budget_s / (1 + len(groups) * .25), 24), placed)
        for k, parent in enumerate(groups):
            children = [c for c in order if room.plan.parents[c] == parent]
            for c in children:
                placed[c] = room.entering(c, placed)
            geometry, critic = room.sub(placed)
            share = (budget_s - spent) / (len(groups) - k)
            placed = run(room.loop(geometry, critic, set(children), share, 3 * len(children) + 4), placed)
    looped = room.critic.assess(placed)
    placed, _polish = refine(room.geometry, room.plan, placed, critic=room.critic)
    return placed, {"loop": looped.key(), "attempts": attempts, "accepted": accepted, "search_s": round(spent, 2)}


def job(args):
    index, row, record, variant, budget_s, templates = args
    start = time.perf_counter()
    try:
        room = Room(row, record, templates)
        draft = room.critic.assess(room.draft)
        placed, info = run_variant(room, variant, budget_s)
        final = room.critic.assess(placed)
        old_rule = Geometry(room.condition, room.plan.parents, room.plan.relations, hard_openings=False)
        out = {"index": index, "variant": variant, "objects": len(room.draft), "draft": draft.key(),
               "final": final.key(), "hard_valid_new_rule": final.hard_count == 0,
               "hard_valid_old_rule": not old_rule.validate(placed), **info,
               "layout": {"schema_version": room.condition["schema_version"],
                          "objects": [placed[o["id"]] for o in room.condition["objects"]]}}
    except Exception as exc:  # noqa: BLE001
        out = {"index": index, "variant": variant, "layout": None, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-1500:]}
    out["seconds"] = round(time.perf_counter() - start, 2)
    return out


def summarise(rows_by_index, results, eval_templates, indices):
    from fastfill.v2.ops import layout_quality as lq
    from fastfill.v2.schema import migrate_legacy_row
    lines = []
    for lo, hi, label in ((1, 10 ** 6, "all rooms"),) + BINS:
        chosen = [i for i in indices if lo <= len(rows_by_index[i]["condition"]["objects"]) <= hi]
        lines += [f"\n### {label} ({len(chosen)} rooms)\n",
                  "| variant | hard-valid (doors hard) | hard-valid (old rule) | hard violations left | REL ↑ | ACC ↑ | "
                  "OPC ↓ | COL ↓ | soft cost | s/room median | s/room max |", "|" + "---|" * 11]
        rows = [migrate_legacy_row(rows_by_index[i]) for i in chosen]
        for v in VARIANTS:
            rs = [results[v][i] for i in chosen]
            ok = [r for r in rs if r.get("layout")]
            q = lq.score_method(rows, [results[v][i].get("layout") for i in chosen], eval_templates)
            f = lambda x: "–" if x is None else f"{100 * x:.1f}"  # noqa: E731
            secs = [r["seconds"] for r in rs]
            lines.append(f"| {NAMES[v]} | {sum(r['hard_valid_new_rule'] for r in ok)}/{len(rs)} | "
                         f"{sum(r['hard_valid_old_rule'] for r in ok)}/{len(rs)} | {sum(r['final'][0] for r in ok)} | "
                         f"{f(q['REL'])} | {f(q['ACC'])} | {f(q['OPC'])} | {f(q['COL'])} | "
                         f"{sum(r['final'][2] for r in ok):.2f} | {statistics.median(secs):.1f} | {max(secs):.1f} |")
        draft = [results["A"][i] for i in chosen if results["A"][i].get("layout")]
        lines.append(f"| draft (step ③) | {sum(r['draft'][0] == 0 for r in draft)}/{len(chosen)} | – | "
                     f"{sum(r['draft'][0] for r in draft)} | | | | | {sum(r['draft'][2] for r in draft):.2f} | | |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rooms", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--templates", required=True, help="evaluation templates (REL), not the critic's")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--search-seconds", type=float, default=180., help="program-search budget per room and variant")
    a = ap.parse_args()
    indices = json.loads(Path(a.rooms).read_text())["indices"]
    lines = Path(a.rows).read_text().splitlines()
    rows_by_index = {i: json.loads(lines[i]) for i in indices}
    critic_templates = load_templates()
    jobs = [(i, rows_by_index[i], json.loads((Path(a.run) / f"room_{i:03d}.json").read_text()), v, a.search_seconds,
             critic_templates) for v in a.variants for i in indices]
    results = {v: {} for v in VARIANTS}
    with ProcessPoolExecutor(a.workers) as pool:
        for out in pool.map(job, jobs):
            results[out["variant"]][out["index"]] = out
            print(f"{out['variant']} room {out['index']:3d} ({out.get('objects', '?')} obj): {out.get('draft', '')} -> "
                  f"{out.get('final', '')} {out['seconds']}s {out.get('error', '')}", flush=True)
    out_dir = Path(a.out)
    for v in a.variants:
        (out_dir / v).mkdir(parents=True, exist_ok=True)
        with open(out_dir / v / "predictions.jsonl", "w") as f:
            for i in indices:
                f.write(json.dumps({"layout": results[v][i].get("layout")}) + "\n")
    (out_dir / "rooms.json").write_text(json.dumps({v: {str(i): {k: x for k, x in r.items() if k != "layout"}
                                                        for i, r in results[v].items()} for v in a.variants}, indent=1))
    if set(a.variants) == set(VARIANTS):
        table = summarise(rows_by_index, results, json.loads(Path(a.templates).read_text())["templates"], indices)
        text = (f"# Check-and-repair order: whole room / object by object / layer by layer\n\nDrafts: {a.run}; "
                f"{len(indices)} rooms; program proposals only (no GPT), sizes fixed, {a.search_seconds:.0f} s search "
                "budget per room and variant, critic-judged polish at the end. REL with the evaluation templates "
                f"{a.templates}.\n" + table + "\n")
        (out_dir / "summary.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
