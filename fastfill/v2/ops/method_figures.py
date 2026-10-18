"""Results table and room comparisons of several methods on the same rows (no model, no network).

    python -m fastfill.v2.ops.method_figures --config methods.json --out DIR [--rooms 3,40,77]

``methods.json``: {"rows": rows.jsonl, "methods": [{"name", "predictions", "report", "llm_runs" (optional: a
directory of room_<k>.json with token usage, llm_probe / llm_hybrid), "latency_s" (optional: seconds per room when
the room files do not say, e.g. the FastFill decode)}]}. ``report`` is ``fastfill.v2.evaluate``'s report.json of the
predictions. Writes DIR/results_table.md, results_table.png (the table), results_bars.png (one panel per headline
metric) and rooms/row<k>.png: each method's top-down plan of one room (vlm_judge.draw_plan) with its problem count
(``llm_workflow.workflow_problems``: the strict scorer's checks) and the ground truth first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fastfill.v2.ops.remask_ablation import summarize  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402

HEADLINE = (("clean_room_rate", "Clean rooms (%) ↑", 100, "{:.1f}"), ("constraint_pass", "Constraints satisfied (%) ↑", 100, "{:.1f}"),
            ("collision_pairs_per_room", "Collision pairs / room ↓", 1, "{:.2f}"),
            ("footprint_outside", "Objects outside the room (%) ↓", 100, "{:.1f}"), ("door_blocking", "Blocking a door (%) ↓", 100, "{:.1f}"),
            ("log_size", "Size error (log) ↓", 1, "{:.3f}"), ("success", "Rooms with a layout (%) ↑", 100, "{:.1f}"),
            ("seconds", "Seconds per room ↓ (log scale)", 1, "{:.3g}"), ("tokens", "LLM tokens per room ↓", 1, "{:,.0f}"))
COLORS = ("#7f8c8d", "#8e44ad", "#e67e22", "#16a085", "#2e86de", "#c0392b", "#f1c40f")


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def usage(run):
    rooms = [json.loads(p.read_text()) for p in sorted(Path(run).glob("room_*.json"))]
    tokens = sum((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0) for r in rooms for u in r.get("usage") or [])
    seconds = [r["latency_s"] for r in rooms if not r.get("error") and r.get("latency_s") is not None]
    return tokens / max(len(rooms), 1), statistics.median(seconds) if seconds else None


def metrics(method):
    report = json.loads(Path(method["report"]).read_text())
    m = summarize(report)
    m["success"] = 1 - report["inference_failed_requests"] / report["requests"]
    m["tokens"], m["seconds"] = 0., method.get("latency_s")
    if method.get("llm_runs"):
        m["tokens"], m["seconds"] = usage(method["llm_runs"])
    return m


def table(names, rows, out):
    lines = ["| Method | " + " | ".join(h[1] for h in HEADLINE) + " |", "|---" * (len(HEADLINE) + 1) + "|"]
    cells = []
    for name, m in zip(names, rows):
        cell = [("–" if m.get(k) is None else f.format(scale * m[k])) for k, _, scale, f in HEADLINE]
        cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cell) + " |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(2 + 1.55 * len(HEADLINE), .55 * len(names) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=names, colLabels=[textwrap.fill(h[1], 16) for h in HEADLINE], loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.1)
    for c in range(len(HEADLINE)):
        t[0, c].set_height(t[0, c].get_height() * 1.9)
    for c, (k, _, _, _) in enumerate(HEADLINE):  # best value of each column in bold
        values = [m.get(k) for m in rows]
        known = [v for v in values if v is not None]
        if not known or (k == "tokens" and not any(known)):
            continue
        lower = "↓" in HEADLINE[c][1]
        best = min(known) if lower else max(known)
        for r, v in enumerate(values):
            if v == best:
                t[r + 1, c].set_text_props(weight="bold")
                t[r + 1, c].set_facecolor("#e8f6ef")
    fig.savefig(out / "results_table.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def bars(names, rows, out, title):
    fig, axes = plt.subplots(3, 3, figsize=(17, 11.5))
    for ax, (k, label, scale, f) in zip(axes.flat, HEADLINE):
        values = [(m.get(k) or 0) * scale for m in rows]
        b = ax.bar(range(len(names)), values, color=COLORS[:len(names)])
        for x, v, m in zip(b, values, rows):
            ax.text(x.get_x() + x.get_width() / 2, v, "–" if m.get(k) is None else f.format(v), ha="center", va="bottom", fontsize=9)
        if k == "seconds":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max(values + [1e-9]) * 1.2)
        ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=.6)
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "results_bars.png", dpi=160)
    plt.close(fig)


def rooms(config, names, out, chosen):
    from fastfill.v2.llm_workflow import workflow_problems
    from fastfill.v2.ops.vlm_judge import draw_plan, items
    rows = [migrate_legacy_row(r) for r in jsonl(config["rows"])]
    layouts = [[p.get("layout") for p in jsonl(m["predictions"])] for m in config["methods"]]
    if not chosen:  # every method answered; 8-20 objects; one room per source, in test order
        seen, chosen = set(), []
        for k, r in enumerate(rows):
            if 8 <= len(r["condition"]["objects"]) <= 20 and all(l[k] is not None for l in layouts) \
                    and r["provenance"]["source"] not in seen:
                seen.add(r["provenance"]["source"])
                chosen.append(k)
        chosen = chosen[:6]
    (out / "rooms").mkdir(parents=True, exist_ok=True)
    panels = [("Ground truth", None)] + list(zip(names, layouts))
    for k in chosen:
        row = rows[k]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.2))
        for ax, (name, ls) in zip(axes, panels):
            layout = row["target"] if ls is None else ls[k]
            if layout is None:
                ax.axis("off")
                ax.set_title(f"{name}\nno layout", fontsize=12)
                continue
            its, _ = items(row, layout)
            draw_plan(ax, row, its)
            problems = workflow_problems(layout, row["condition"])
            ax.set_title(f"{name}\n{len(problems)} problem{'s' if len(problems) != 1 else ''}", fontsize=12,
                         color="#1e8449" if not problems else "#b03a2e", weight="bold")
        room = row["condition"]["room"]
        fig.suptitle(f"test300 room {k}: {row['provenance']['source']}, {room.get('room_type') or 'room'}, "
                     f"{len(row['condition']['objects'])} objects", fontsize=14, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(out / "rooms" / f"row{k:03d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote rooms/row{k:03d}.png", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rooms", type=lambda s: [int(x) for x in s.split(",")])
    p.add_argument("--title", default="Same 300 test rooms, same evaluator")
    a = p.parse_args(argv)
    config = json.loads(a.config.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    names = [m["name"] for m in config["methods"]]
    rows = [metrics(m) for m in config["methods"]]
    table(names, rows, a.out)
    bars(names, rows, a.out, a.title)
    print((a.out / "results_table.md").read_text())
    rooms(config, names, a.out, a.rooms)


if __name__ == "__main__":
    main()
