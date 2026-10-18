"""Results table, trade-off scatter, metric bars and room comparisons of several methods on the same rows (no model,
no network).

    python -m fastfill.v2.ops.method_figures --config methods.json --out DIR [--rooms 3,40,77]

``methods.json``: {"rows": rows.jsonl, "templates": relation templates (``layout_quality mine``), "methods": [{"name",
"predictions", "report", "llm_runs" (optional: a directory of room_<k>.json with token usage, llm_probe / llm_hybrid),
"latency_s" (optional: seconds per room when the room files do not say, e.g. the FastFill decode), "rooms" (optional,
default true: drawn in the room comparisons)}]}. ``report`` is ``fastfill.v2.evaluate``'s report.json of the
predictions. Legality comes from the report, plausibility (REL, ACC, NAV) and SceneEval-style COL / OPC from
``layout_quality``; a ground-truth row (the labels scored the same way) is added for reference.

Writes DIR/results_table.md and .png, tradeoff.png (clean rooms against relation plausibility, one point per method,
the ground truth and a shuffled ground truth as references), results_bars.png (one panel per metric) and
rooms/row<k>.png: each method's top-down plan of one room (vlm_judge.draw_plan) with its problem count
(``llm_workflow.workflow_problems``) and relation score, the ground truth first. Rooms are chosen by a fixed rule,
not by any method's outcome: in test order, the first room of each common room type (bedroom, living, dining,
study / office, kitchen, bathroom) with 6-25 objects, at least three relation checks on its ground truth, and a layout from every drawn method.
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

from fastfill.v2.ops import layout_quality as lq  # noqa: E402
from fastfill.v2.ops.remask_ablation import summarize  # noqa: E402
from fastfill.v2.schema import migrate_legacy_row  # noqa: E402

HEADLINE = (("clean_room_rate", "Clean rooms (%) ↑", 100, "{:.1f}"), ("constraint_pass", "Constraints satisfied (%) ↑", 100, "{:.1f}"),
            ("REL", "Relation plausibility (%) ↑", 100, "{:.1f}"), ("ACC", "Accessible objects (%) ↑", 100, "{:.1f}"),
            ("COL", "Objects in a collision (%) ↓", 100, "{:.1f}"), ("footprint_outside", "Objects outside the room (%) ↓", 100, "{:.1f}"),
            ("OPC", "Doors blocked (%) ↓", 100, "{:.1f}"), ("log_size", "Size error (log) ↓", 1, "{:.3f}"),
            ("success", "Rooms with a layout (%) ↑", 100, "{:.1f}"), ("seconds", "Seconds per room ↓", 1, "{:.3g}"),
            ("tokens", "LLM tokens per room ↓", 1, "{:,.0f}"))
COLORS = ("#7f8c8d", "#8e44ad", "#e67e22", "#16a085", "#2e86de", "#c0392b", "#f1c40f")
REFERENCE = "Ground truth (reference)"
ROOM_GROUPS = (("bedroom", ("bed",)), ("living room", ("living",)), ("dining room", ("dining",)),
               ("study / office", ("study", "office", "library")), ("kitchen", ("kitchen",)), ("bathroom", ("bath",)))


def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def usage(run):
    rooms = [json.loads(p.read_text()) for p in sorted(Path(run).glob("room_*.json"))]
    tokens = sum((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0) for r in rooms for u in r.get("usage") or [])
    seconds = [r["latency_s"] for r in rooms if not r.get("error") and r.get("latency_s") is not None]
    return tokens / max(len(rooms), 1), statistics.median(seconds) if seconds else None


def metrics(method, rows, templates):
    report = json.loads(Path(method["report"]).read_text())
    m = summarize(report)
    m["success"] = 1 - report["inference_failed_requests"] / report["requests"]
    m["tokens"], m["seconds"] = 0., method.get("latency_s")
    if method.get("llm_runs"):
        m["tokens"], m["seconds"] = usage(method["llm_runs"])
    m.update(lq.score_method(rows, [p.get("layout") for p in jsonl(method["predictions"])], templates))
    return m


def reference(rows, templates, report):
    """The labels scored like the methods: legality from a method's report (its ground-truth side), the rest computed."""
    m = {k: None for k, *_ in HEADLINE}
    m.update(lq.score_method(rows, [r["target"] for r in rows], templates))
    m["clean_room_rate"] = (report["validation"].get("ground_truth") or {}).get("clean_room_rate")
    gt = report["collapse"]["ground_truth"]
    m["footprint_outside"] = gt["footprint_outside_fraction"]
    m["success"], m["tokens"] = 1., None
    return m


def table(names, rows, out):
    lines = ["| Method | " + " | ".join(h[1] for h in HEADLINE) + " |", "|---" * (len(HEADLINE) + 1) + "|"]
    cells = []
    for name, m in zip(names, rows):
        cell = [("–" if m.get(k) is None else f.format(scale * m[k])) for k, _, scale, f in HEADLINE]
        cells.append(cell)
        lines.append(f"| {name} | " + " | ".join(cell) + " |")
    (out / "results_table.md").write_text("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(2.4 + 1.3 * len(HEADLINE), .55 * len(names) + 1.4))
    ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=names, colLabels=[textwrap.fill(h[1], 14) for h in HEADLINE], loc="center",
                 cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 2.1)
    for c in range(len(HEADLINE)):
        t[0, c].set_height(t[0, c].get_height() * 2.2)
    methods = [r for r, n in enumerate(names) if n != REFERENCE]
    for c, (k, label, _, _) in enumerate(HEADLINE):  # best method value of each column in bold (the reference excluded)
        values = {r: rows[r].get(k) for r in methods if rows[r].get(k) is not None}
        if not values or (k == "tokens" and not any(values.values())):
            continue
        best = (min if "↓" in label else max)(values.values())
        for r, v in values.items():
            if v == best:
                t[r + 1, c].set_text_props(weight="bold")
                t[r + 1, c].set_facecolor("#e8f6ef")
    for r, n in enumerate(names):
        if n == REFERENCE:
            for c in range(-1, len(HEADLINE)):
                t[r + 1, c].set_facecolor("#f2f3f4")
                t[r + 1, c].set_text_props(style="italic", color="#555555")
    fig.savefig(out / "results_table.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def tradeoff(names, rows, shuffled, out, title):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for i, (name, m) in enumerate(zip(names, rows)):
        if m.get("clean_room_rate") is None or m.get("REL") is None:
            continue
        ref = name == REFERENCE
        ax.scatter(100 * m["clean_room_rate"], 100 * m["REL"], s=260 if not ref else 330, marker="*" if ref else "o",
                   color="#555555" if ref else COLORS[i % len(COLORS)], edgecolor="black", zorder=3)
        note = "" if ref or m.get("log_size") is None else f"\nsize err {m['log_size']:.3f}, {m['tokens']:,.0f} tok/room"
        x, y = 100 * m["clean_room_rate"], 100 * m["REL"]
        crowded = any(0 < 100 * o["clean_room_rate"] - x < 35 and abs(100 * o["REL"] - y) < 10 for o in rows
                      if o is not m and o.get("clean_room_rate") is not None and o.get("REL") is not None)
        ax.annotate(name + note, (x, y), xytext=(-12, 8) if crowded else (10, -4), textcoords="offset points",
                    ha="right" if crowded else "left", fontsize=9, color="#333333", style="italic" if ref else "normal")
    ax.axhline(100 * shuffled, color="#b03a2e", ls="--", lw=1)
    ax.text(1, 100 * shuffled + 1, "shuffled ground truth (arrangement at random)", color="#b03a2e", fontsize=9)
    ax.set(xlim=(0, 105), ylim=(0, 100), xlabel="Clean rooms (%): no collision, nothing outside, no door blocked ↑",
           ylabel="Relation plausibility (%): mined layout rules satisfied ↑")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(ls=":", alpha=.6)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=170)
    plt.close(fig)


def bars(names, rows, out, title):
    fig, axes = plt.subplots(3, 4, figsize=(21, 13.5))
    for ax, (k, label, scale, f) in zip(axes.flat, HEADLINE):
        values = [None if m.get(k) is None else m[k] * scale for m in rows]
        colors = ["#d5d8dc" if n == REFERENCE else COLORS[i % len(COLORS)] for i, n in enumerate(names)]
        b = ax.bar(range(len(names)), [v or 0 for v in values], color=colors, hatch=[("//" if n == REFERENCE else "") for n in names])
        for x, v in zip(b, values):
            ax.text(x.get_x() + x.get_width() / 2, v or 0, "–" if v is None else f.format(v), ha="center", va="bottom", fontsize=9)
        if k == "seconds":
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max([v for v in values if v is not None] + [1e-9]) * 1.2)
        ax.set_xticks(range(len(names)), [textwrap.fill(n, 18) for n in names], rotation=25, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", ls=":", alpha=.6)
    for ax in list(axes.flat)[len(HEADLINE):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.subplots_adjust(left=.04, right=.99, top=.92, bottom=.09, hspace=.75, wspace=.22)
    fig.savefig(out / "results_bars.png", dpi=150)
    plt.close(fig)


def rooms(config, out, chosen, templates):
    from fastfill.v2.llm_workflow import workflow_problems
    from fastfill.v2.ops.vlm_judge import draw_plan, items
    rows = [migrate_legacy_row(r) for r in jsonl(config["rows"])]
    drawn = [m for m in config["methods"] if m.get("rooms", True)]
    layouts = [[p.get("layout") for p in jsonl(m["predictions"])] for m in drawn]
    if not chosen:
        chosen, seen = [], set()
        for k, r in enumerate(rows):
            kind = next((g for g, words in ROOM_GROUPS
                         if any(w in (r["condition"]["room"].get("room_type") or "").lower() for w in words)), None)
            if kind and kind not in seen and 6 <= len(r["condition"]["objects"]) <= 25 \
                    and all(l[k] is not None for l in layouts) and lq.rel(lq.boxes(r, r["target"]), templates)[1] >= 3:
                seen.add(kind)
                chosen.append(k)
    (out / "rooms").mkdir(parents=True, exist_ok=True)
    panels = [("Ground truth", None)] + [(m["name"], l) for m, l in zip(drawn, layouts)]
    for k in chosen:
        row = rows[k]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 6.4))
        for ax, (name, ls) in zip(axes, panels):
            layout = row["target"] if ls is None else ls[k]
            if layout is None:
                ax.axis("off")
                ax.set_title(f"{name}\nno layout", fontsize=12)
                continue
            its, _ = items(row, layout)
            draw_plan(ax, row, its)
            problems = workflow_problems(layout, row["condition"])
            hits, total = lq.rel(lq.boxes(row, layout), templates)
            ax.set_title(f"{name}\n{len(problems)} problem{'s' if len(problems) != 1 else ''} · relations {hits}/{total}",
                         fontsize=12, color="#1e8449" if not problems else "#b03a2e", weight="bold")
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
    rows = [migrate_legacy_row(r) for r in jsonl(config["rows"])]
    templates = json.loads(Path(config["templates"]).read_text())["templates"]
    names = [m["name"] for m in config["methods"]]
    results = [metrics(m, rows, templates) for m in config["methods"]]
    ref = reference(rows, templates, json.loads(Path(config["methods"][0]["report"]).read_text()))
    shuffled = lq.score_method(rows, [lq.shuffled(r, k) for k, r in enumerate(rows)], templates)["REL"]
    names, results = names + [REFERENCE], results + [ref]
    table(names, results, a.out)
    tradeoff(names, results, shuffled, a.out, a.title)
    bars(names, results, a.out, a.title)
    (a.out / "results.json").write_text(json.dumps(dict(zip(names, results)), indent=1, default=float) + "\n")
    print((a.out / "results_table.md").read_text())
    rooms(config, a.out, a.rooms, templates)


if __name__ == "__main__":
    main()
