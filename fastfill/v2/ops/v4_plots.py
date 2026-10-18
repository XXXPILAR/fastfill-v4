"""Figures for the v4 runs, from files the runs already write (no model, no GPU, no network):

    curves   training_log.json of each arm: train loss (smoothed) with the validation points, every loss term
             (position cell / residual / z, size, yaw, support), the validation quality checks, learning rate and
             gradient norms
    results  runs/ablation-v4: every arm x decoding mode next to GPT-5.6 on the 300 test rooms, one panel per metric
             with the trivial baselines as dashed lines; the validation sample; per-source position and size error;
             the support head's accuracy
    rooms    a few test rooms drawn side by side: ground truth, GPT-5.6 and each arm (vlm_judge's top-down plan and
             two corner views)

    python -m fastfill.v2.ops.v4_plots all --out runs/plots      # every figure whose inputs exist
    python -m fastfill.v2.ops.v4_plots curves --runs base=runs/main4-...-base D=... --out runs/plots
    python -m fastfill.v2.ops.v4_plots rooms --rows 3,40,77 --out runs/plots

Arms default to the newest runs/main4-*-e5-{base,D,E}* directory with a training_log.json. Needs matplotlib.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ARMS = ("base", "D", "E")
COLOR = {"base": "#7f8c8d", "D": "#2e86de", "E": "#e67e22", "E_nosupport": "#f5b041", "GPT-5.6": "#8e44ad"}
TERMS = (("position_cell", "position cell (CE)"), ("position_residual", "position residual"), ("position_z", "height z"),
         ("size", "size"), ("yaw_cls", "yaw class (CE)"), ("support", "support (CE)"))
METRICS = (("position_m", "Position error (m) ↓", 1, "room_center_position", "bottom_center_error_m", "room centre"),
           ("log_size", "Size error (log) ↓", 1, "category_median_size", "log_size_error", "category median"),
           ("yaw_rad", "Yaw error (rad) ↓", 1, "uniform_yaw", "yaw_error_rad", "uniform yaw"),
           ("footprint_outside", "Outside room (%) ↓", 100, None, None, None),
           ("door_blocking", "Blocking a door (%) ↓", 100, None, None, None),
           ("collision_pairs_per_room", "Collisions per room ↓", 1, None, None, None),
           ("constraint_pass", "Constraints satisfied (%) ↑", 100, None, None, None),
           ("clean_room_rate", "Clean rooms (%) ↑", 100, None, None, None))


def color(name):
    return COLOR.get(name.split(" ")[0], "#34495e")


def smooth(values, alpha=.98):
    out, level = [], None
    for v in values:
        level = v if level is None else alpha * level + (1 - alpha) * v
        out.append(level)
    return out


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def find_runs(specs):
    if specs:
        return {k: Path(v) for k, v in (s.split("=", 1) for s in specs)}
    runs = {}
    for arm in ARMS:
        found = sorted((Path(p).parent for p in glob.glob(str(REPO / f"runs/main4-*-e5-{arm}*/training_log.json"))),
                       key=lambda p: p.stat().st_mtime)
        if found:
            runs[arm] = found[-1]
    return runs


# ---------------------------------------------------------------------------------------------------------- curves
def curves(runs, out):
    logs = {arm: json.loads((path / "training_log.json").read_text()) for arm, path in runs.items()
            if (path / "training_log.json").is_file()}
    if not logs:
        print("curves: no training_log.json found")
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    for arm, log in logs.items():
        steps, loss = [r["step"] for r in log], [r["loss"] for r in log]
        ax.plot(steps, loss, color=color(arm), alpha=.15, lw=.6)
        ax.plot(steps, smooth(loss), color=color(arm), lw=2, label=f"{arm} train (final {smooth(loss)[-1]:.3f})")
        val = [(r["step"], r["validation"]["geometry_objective_mean_of_batches"]) for r in log if r.get("validation")]
        if val:
            ax.plot(*zip(*val), "o--", color=color(arm), ms=5, lw=1, label=f"{arm} validation")
    ax.set(xlabel="step", ylabel="loss", title="Training loss (each arm's own objective: D/E add masked and support terms)")
    ax.grid(ls=":", alpha=.6)
    ax.legend(ncol=2, fontsize=9)
    save(fig, out / "loss_curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (key, title) in zip(axes.flat, TERMS):
        for arm, log in logs.items():
            pts = [(r["step"], r["unweighted"][key]) for r in log if (r.get("unweighted") or {}).get(key) is not None]
            if pts:
                ax.plot([p[0] for p in pts], smooth([p[1] for p in pts]), color=color(arm), lw=1.6, label=f"{arm} train")
            val = [(r["step"], r["validation"]["unweighted"].get(key)) for r in log
                   if r.get("validation") and r["validation"]["unweighted"].get(key) is not None]
            if val:
                ax.plot(*zip(*val), "o--", color=color(arm), ms=4, lw=1, label=f"{arm} val")
        ax.set_title(title)
        ax.grid(ls=":", alpha=.6)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Loss terms (unweighted; lines: training, smoothed; dots: validation)")
    fig.tight_layout()
    save(fig, out / "loss_terms.png")

    checks = (("footprint_outside_objects", "objects", "Outside room (%)"),
              ("door_blocking_objects", "floor_objects", "Blocking a door (%)"),
              ("bev_overlap_pairs", "pairs", "Overlapping pairs (%)"))
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for arm, log in logs.items():
        val = [r for r in log if r.get("validation")]
        if not val:
            continue
        steps = [r["step"] for r in val]
        sel = [(r["step"], r["validation"]["selection_metric"]) for r in val
               if isinstance(r["validation"].get("selection_metric"), (int, float))]
        if sel:
            axes[0].plot(*zip(*sel), "o-", color=color(arm), label=arm)
        for ax, (num, den, _) in zip(axes[1:], checks):
            c = [r["validation"].get("collapse_matched") or r["validation"]["collapse"] for r in val]
            ax.plot(steps, [100 * x[num] / max(x[den], 1) for x in c], "o-", color=color(arm), label=arm)
    for ax, title in zip(axes, ["Selection metric ↓ (best checkpoint)"] + [c[2] for c in checks]):
        ax.set(title=title, xlabel="step")
        ax.grid(ls=":", alpha=.6)
    axes[0].legend()
    fig.suptitle("Validation during training")
    fig.tight_layout()
    save(fig, out / "validation_during_training.png")

    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.2))
    for arm, log in logs.items():
        steps = [r["step"] for r in log]
        for group, ls in (("backbone", "-"), ("decoder", "--")):
            lr = [(r.get("learning_rates") or {}).get(group) for r in log]
            if any(v is not None for v in lr):
                a.plot(steps, lr, ls, color=color(arm), label=f"{arm} {group}")
        for part, ls in (("backbone", "-"), ("decoder", "--"), ("position_head", ":")):
            g = [(s, (r.get("gradient_norms") or {}).get(part)) for s, r in zip(steps, log)]
            g = [(s, v) for s, v in g if v is not None]
            if g:
                b.plot([p[0] for p in g], smooth([p[1] for p in g]), ls, color=color(arm), label=f"{arm} {part}")
    a.set(title="Learning rate", xlabel="step")
    b.set(title="Gradient norm (smoothed)", xlabel="step", yscale="log")
    for ax in (a, b):
        ax.grid(ls=":", alpha=.6)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out / "lr_and_gradients.png")


# --------------------------------------------------------------------------------------------------------- results
def summarize(report):
    from fastfill.v2.ops.remask_ablation import summarize as base
    return base(report)


def results_table(ablation, prefix):
    """[(label, metrics, report)] for every arm x mode under ablation/<prefix><arm>, plus GPT-5.6 when scored."""
    table = []
    for arm in (*ARMS, "E_nosupport"):
        path = ablation / f"{prefix}{arm}" / "ablation.json"
        if not path.is_file():
            continue
        for mode, metrics in json.loads(path.read_text()).items():
            if mode == "ground_truth":
                continue
            report = ablation / f"{prefix}{arm}" / f"eval_{mode}" / "report.json"
            table.append((f"{arm} {mode}", metrics, json.loads(report.read_text()) if report.is_file() else None))
    gpt = ablation / f"{prefix}gpt56" / "report.json"
    if gpt.is_file():
        report = json.loads(gpt.read_text())
        table.append(("GPT-5.6", summarize(report), report))
    return table


def metric_panels(table, title, path):
    fig, axes = plt.subplots(2, 4, figsize=(20, max(6, .55 * len(table) + 3)))
    labels = [t[0] for t in table]
    reference = next((r for _, _, r in table if r and r.get("baselines_paired")), None)
    for ax, (key, name, scale, base, base_key, base_name) in zip(axes.flat, METRICS):
        values = [m.get(key) for _, m, _ in table]
        shown = [scale * v if v is not None else 0 for v in values]
        bars = ax.barh(range(len(table)), shown, color=[color(l) for l in labels])
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    " –" if v is None else f" {scale * v:.{1 if scale == 100 else 3}f}", va="center", fontsize=8)
        b = reference["baselines_paired"].get(base, {}).get(base_key, {}).get("mean") if base and reference else None
        if b is not None:
            ax.axvline(b, color="k", ls="--", lw=1)
            name += f"\n(dashed: {base_name} {b:.3f})"
        ax.set_yticks(range(len(table)), labels if ax in axes[:, 0] else [""] * len(table), fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=11)
        ax.margins(x=.18)
        ax.grid(axis="x", ls=":", alpha=.6)
    fig.suptitle(title, fontsize=14, weight="bold")
    fig.tight_layout()
    save(fig, path)


def by_source(table, path):
    picked = [(l, r) for l, _, r in table if r and r.get("by_source")]
    if not picked:
        return
    sources = sorted({s for _, r in picked for s in r["by_source"]})
    n = {s: max(r["by_source"].get(s, {}).get("requests", 0) for _, r in picked) for s in sources}
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, .5 * len(sources) * max(1, len(picked) / 3) + 2)))
    height = .8 / len(picked)
    for ax, (key, name) in zip(axes, (("bottom_center_error_m", "Position error (m) ↓"), ("log_size_error", "Size error (log) ↓"))):
        for k, (label, report) in enumerate(picked):
            vals = [(((report["by_source"].get(s) or {}).get("model") or {}).get("reference") or {}).get(key, {}).get("mean")
                    for s in sources]
            ax.barh([i + k * height for i in range(len(sources))], [v or 0 for v in vals], height, color=color(label),
                    alpha=.6 + .4 * (k % 2), label=label)
        ax.set_yticks([i + .4 - height / 2 for i in range(len(sources))], [f"{s} (N={n[s]})" for s in sources], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name)
        ax.grid(axis="x", ls=":", alpha=.6)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Test 300 by data source", fontsize=14, weight="bold")
    fig.tight_layout()
    save(fig, path)


def support_figure(table, path):
    rows = [(l, m["support_accuracy"]) for l, m, _ in table if (m.get("support_accuracy") or {}).get("all") is not None]
    if not rows:
        return
    keys = ("all", "floor", "wall", "object")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = .8 / len(rows)
    for k, (label, s) in enumerate(rows):
        ax.bar([i + k * w for i in range(len(keys))], [100 * (s.get(x) or 0) for x in keys], w, label=label,
               color=color(label), alpha=.5 + .5 * (k % 2))
    ax.set_xticks([i + .4 - w / 2 for i in range(len(keys))], ["all", "floor", "wall", "on furniture"])
    ax.set(ylabel="accuracy (%)", ylim=(0, 105), title="Support prediction accuracy (no support declared in the input)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", ls=":", alpha=.6)
    save(fig, path)


def results(ablation, out):
    test = results_table(ablation, "test300_")
    if test:
        metric_panels(test, "Test 300 rooms: FastFill v4 arms vs GPT-5.6 (same rooms, same evaluator)", out / "test300_metrics.png")
        best = [t for t in test if t[0] == "GPT-5.6"] + [max((t for t in test if t[0].startswith(arm + " ") and t[2]),
                                                              key=lambda t: -t[1]["position_m"], default=None) for arm in ARMS]
        by_source([t for t in best if t], out / "test300_by_source.png")
    val = results_table(ablation, "")
    if val:
        metric_panels(val, "Validation sample: every arm x decoding mode", out / "validation_metrics.png")
        support_figure(val, out / "support_accuracy.png")
    probe = results_table(ablation, "probe20_")
    if probe:
        metric_panels(probe, "The 20 probe rooms", out / "probe20_metrics.png")
    if not (test or val or probe):
        print(f"results: nothing under {ablation}")


# ----------------------------------------------------------------------------------------------------------- rooms
def jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def room_layouts(ablation, rows_path, gpt_path, mode):
    rows = jsonl(rows_path)
    methods = {"Ground truth": [{"schema_version": r["target"]["schema_version"], "objects": r["target"]["objects"]} for r in rows]}
    if Path(gpt_path).is_file():
        methods["GPT-5.6"] = [p.get("layout") for p in jsonl(gpt_path)]
    for arm in ARMS:
        path = ablation / f"test300_{arm}" / f"predictions_{mode}.jsonl"
        if not path.is_file() and arm == "base":
            path = ablation / f"test300_{arm}" / "predictions_spread.jsonl"
        if path.is_file():
            methods[f"{arm} {path.stem.split('_', 1)[1]}"] = [p.get("layout") for p in jsonl(path)]
    return rows, methods


def default_rows(rows, methods, count=6):
    """Rooms every method answered, 6-30 objects, one per source while possible, in test order."""
    ok = [i for i, r in enumerate(rows) if 6 <= len(r["condition"]["objects"]) <= 30
          and all(m[i] is not None for m in methods.values())]
    picked, seen = [], set()
    for i in ok:
        if rows[i]["provenance"]["source"] not in seen:
            picked.append(i)
            seen.add(rows[i]["provenance"]["source"])
    return (picked + [i for i in ok if i not in picked])[:count]


def rooms(ablation, rows_path, gpt_path, mode, chosen, out):
    from fastfill.v2.ops.vlm_judge import items, render_panel
    rows, methods = room_layouts(ablation, rows_path, gpt_path, mode)
    chosen = chosen or default_rows(rows, methods)
    with tempfile.TemporaryDirectory() as tmp:
        for r in chosen:
            row = rows[r]
            cols = min(2, len(methods))
            fig, axes = plt.subplots(math.ceil(len(methods) / cols), cols, figsize=(12 * cols, 8.2 * math.ceil(len(methods) / cols)),
                                     squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, (name, layouts) in zip(axes.flat, methods.items()):
                ax.axis("off")
                layout = layouts[r]
                try:
                    if layout is None:
                        raise ValueError("no layout")
                    its, _ = items(row, layout)
                except (ValueError, KeyError) as error:
                    ax.set_title(f"{name}: {error}", fontsize=14)
                    continue
                png = Path(tmp) / f"{r}-{len(name)}.png"
                render_panel(png, row, its)
                ax.imshow(plt.imread(png))
                ax.set_title(name, fontsize=16, weight="bold", color=color(name))
            room = row["condition"]["room"]
            fig.suptitle(f"test300 row {r}: {row['provenance']['source']}, {room.get('room_type') or 'room'}, "
                         f"{len(row['condition']['objects'])} objects", fontsize=18, weight="bold")
            fig.tight_layout()
            save(fig, out / "rooms" / f"row{r:03d}.png")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("all", "curves", "results", "rooms"))
    p.add_argument("--out", type=Path, default=REPO / "runs/plots")
    p.add_argument("--runs", nargs="*", help="arm=run_dir (default: newest runs/main4-*-e5-<arm>*)")
    p.add_argument("--ablation", type=Path, default=REPO / "runs/ablation-v4")
    p.add_argument("--test-rows", type=Path, default=REPO / "fastfill/v2/ops/test300/rows.jsonl")
    p.add_argument("--gpt", type=Path, default=REPO / "fastfill/v2/ops/test300/gpt56_predictions.jsonl")
    p.add_argument("--mode", default="remask4LX", help="decoding mode drawn in rooms (base falls back to spread)")
    p.add_argument("--rows", type=lambda s: [int(x) for x in s.split(",")], help="test300 row numbers for rooms")
    a = p.parse_args(argv)
    if a.command in ("all", "curves"):
        curves(find_runs(a.runs), a.out)
    if a.command in ("all", "results"):
        results(a.ablation, a.out)
    if a.command in ("all", "rooms"):
        rooms(a.ablation, a.test_rows, a.gpt, a.mode, a.rows, a.out)


if __name__ == "__main__":
    main()
