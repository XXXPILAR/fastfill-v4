"""Replay steps ④–⑦ on saved drafts, no API: the v4 path against the critic loop (program proposals only).

A saved run (``run_draft.py``'s room_XXX.json) holds each room's draft, sizes, supports, groups and priority, but not
step ①'s LLM soft relations (runs before the critic recorded ``plan_relations``), so the replay's plan has the groups'
near relations only: both variants see the same plan. Variants:
  v4      soft doors (old rule), ``local_repair`` (shrinking allowed), no polish: what test50_v4 ran
  critic  critic geometry (hard doors, evaluator rule), ``critic_loop`` with ``ask=None`` (no GPT), polish judged
          by the critic, sizes fixed
Writes <out>/<variant>/predictions.jsonl (aligned with --rows' subset in the run's order) and <out>/replay.json
(per room: status, critic keys draft -> final, seconds).

    python experiments/replay_critic.py --run experiments/test50_v4_server --rooms experiments/test50_rooms.json \\
        --rows ../test300/rows.jsonl --out experiments/replay_v4 --workers 8
"""
import argparse, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from workflow.catalog import SizeCatalog  # noqa: E402
from workflow.contract import check_request, fill_room_defaults  # noqa: E402
from workflow.critic import Critic, load_templates  # noqa: E402
from workflow.geometry import Geometry  # noqa: E402
from workflow.local_repair import local_repair  # noqa: E402
from workflow.loop import LoopBudget, critic_loop  # noqa: E402
from workflow.planning import Plan, _group_relations, _required_parents  # noqa: E402
from workflow.refine import refine  # noqa: E402

CATALOG = HERE / "size_catalog.json"


def replay(job):
    index, row, record, variant, budget = job
    start = time.perf_counter()
    try:
        condition, _ = fill_room_defaults(row["condition"])
        condition = check_request(condition)
        d = record["diagnostics"]
        plan = Plan(parents=dict(d["support_parents"]), groups=d.get("functional_groups") or [],
                    priority={k: int(v) for k, v in (d.get("priority") or {}).items()})
        _group_relations(plan)
        sizes = d["sizes"]
        placed = {b["id"]: b for b in d["draft_layout"]["objects"]}
        catalog = SizeCatalog(CATALOG)
        out = {"index": index}
        if variant == "v4":
            geometry = Geometry(condition, plan.parents, plan.relations, hard_openings=False)
            if geometry.validate(placed):
                plan.draft_poses = {i: {"x": b["bottom_center_m"][0], "y": b["bottom_center_m"][1],
                                        "yaw_rad": b["yaw_rad"], "alt_yaw_rad": None} for i, b in placed.items()}
                placed, _ = local_repair(geometry, plan, sizes, placed, catalog)
            out["hard"] = len(geometry.validate(placed))
        else:
            geometry = Geometry(condition, plan.parents, plan.relations, hard_doors=True, hard_windows=False,
                                door_rule="evaluator")
            critic = Critic.for_plan(geometry, plan, load_templates())
            draft = critic.assess(placed)
            placed, info = critic_loop(geometry, plan, sizes, placed, catalog, critic,
                                       locked_parents=_required_parents(condition), ask=None,
                                       budget=LoopBudget(**budget))
            looped = critic.assess(placed)
            placed, polish = refine(geometry, plan, placed, critic=critic)
            final = critic.assess(placed)
            out.update(draft=draft.key(), loop=looped.key(), final=final.key(), hard=final.hard_count,
                       termination=info["termination"], attempts=len(info["attempts"]),
                       accepted=sum(a["accepted"] for a in info["attempts"]),
                       polish={k: polish[k] for k in ("wall_snap", "axis_align", "near", "faces", "rejected")},
                       soft_by_code_draft=draft.summary()["soft_by_code"], soft_by_code_final=final.summary()["soft_by_code"])
        out["layout"] = {"schema_version": condition["schema_version"],
                         "objects": [placed[o["id"]] for o in condition["objects"]]}
    except Exception as exc:  # noqa: BLE001
        out = {"index": index, "layout": None, "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-1500:]}
    out["seconds"] = round(time.perf_counter() - start, 2)
    return variant, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rooms", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=["v4", "critic"])
    ap.add_argument("--search-seconds", type=float, default=180.)
    a = ap.parse_args()
    indices = json.loads(Path(a.rooms).read_text())["indices"]
    rows = Path(a.rows).read_text().splitlines()
    budget = {"gpt_repairs": 0, "search_seconds": a.search_seconds}
    jobs = [(i, json.loads(rows[i]), json.loads((Path(a.run) / f"room_{i:03d}.json").read_text()), v, budget)
            for v in a.variants for i in indices]
    results = {v: {} for v in a.variants}
    with ProcessPoolExecutor(a.workers) as pool:
        for variant, out in pool.map(replay, jobs):
            results[variant][out["index"]] = out
            print(f"{variant:6s} room {out['index']:3d} hard={out.get('hard')} {out.get('draft', '')} -> "
                  f"{out.get('final', '')} {out['seconds']}s {out.get('error', '')}", flush=True)
    out_dir = Path(a.out)
    for v in a.variants:
        (out_dir / v).mkdir(parents=True, exist_ok=True)
        with open(out_dir / v / "predictions.jsonl", "w") as f:
            for i in indices:
                f.write(json.dumps({"layout": results[v][i].get("layout")}) + "\n")
    (out_dir / "replay.json").write_text(json.dumps({v: {str(i): {k: x for k, x in r.items() if k != "layout"}
                                                         for i, r in results[v].items()} for v in a.variants}, indent=1))


if __name__ == "__main__":
    main()
