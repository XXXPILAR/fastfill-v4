"""FastFill 2D Floor Plan & Layout Visualizer.

Renders room boundaries, fixed elements (doors/windows/stairs), and object placements
with bounding boxes, IDs, labels, and front orientation arrows.

Usage:
    # 1. Visualize a sample from test.jsonl:
    python -m fastfill.tools.visualize --data data/v3.1/test.jsonl --index 0 --out room_0.png

    # 2. Render the built-in synthetic demonstration:
    python -m fastfill.tools.visualize --out demo.png

Requires matplotlib. --data accepts SFT message rows; evaluator predictions are not accepted by this CLI.
"""
import argparse
import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
import numpy as np


def draw_layout(room, placements, save_path="layout.png"):
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#f8f9fa")
    ax.set_facecolor("#ffffff")
    
    # 1. Draw boundary
    b = room.get("boundary", [])
    if b:
        bx, by = zip(*(b + [b[0]]))
        ax.plot(bx, by, color="#1a1a1a", lw=3.5, label="Boundary / Wall", zorder=2)
        ax.fill(bx, by, facecolor="#f1f3f5", alpha=0.5, zorder=1)

    # 2. Draw fixed elements (doors, windows, columns, stairs)
    for f in room.get("fixed", []):
        x, y = f["pos"][:2]
        sx, sy = f["size"][:2]
        yaw = math.radians(f.get("yaw", 0))
        corners = np.array([
            [-sx/2, -sy/2],
            [ sx/2, -sy/2],
            [ sx/2,  sy/2],
            [-sx/2,  sy/2]
        ])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rot = corners @ R.T + np.array([x, y])
        ax.add_patch(MPoly(rot, facecolor="#dee2e6", edgecolor="#495057", lw=1.5, hatch="//", zorder=3))
        ax.text(x, y, f.get("id", "fixed"), fontsize=7, ha="center", va="center", color="#212529", zorder=4)

    # 3. Draw object placements
    pmap = {p["id"]: p for p in placements}
    colors = ["#4ea8de", "#56cfe1", "#72efdd", "#80ffdb", "#48bfe3", "#64dfdf"]
    
    for idx, obj in enumerate(room.get("objects", [])):
        oid = obj["id"]
        if oid not in pmap:
            continue
        p = pmap[oid]
        x, y = p["pos"][:2]
        sx, sy = obj["size"][:2]
        yaw = math.radians(p.get("yaw", 0))
        
        corners = np.array([
            [-sx/2, -sy/2],
            [ sx/2, -sy/2],
            [ sx/2,  sy/2],
            [-sx/2,  sy/2]
        ])
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        rot = corners @ R.T + np.array([x, y])
        
        c = colors[idx % len(colors)]
        ax.add_patch(MPoly(rot, facecolor=c, edgecolor="#1d3557", lw=2, alpha=0.85, zorder=5))
        
        # Front arrow pointing along local +X
        arrow_len = min(sx, sy) * 0.45
        dx = math.cos(yaw) * arrow_len
        dy = math.sin(yaw) * arrow_len
        ax.arrow(x, y, dx, dy, head_width=0.08, head_length=0.08, fc="#e63946", ec="#e63946", zorder=6)
        
        # Label
        label = f"{oid}\n{p.get('yaw', 0)}°"
        ax.text(x, y, label, fontsize=8, ha="center", va="center", fontweight="bold", color="#1d3557", zorder=7)

    room_type = room.get("room_type", "Room Layout")
    ax.set_title(f"FastFill Scene Layout: {room_type} ({len(placements)} objects)", fontsize=13, fontweight="bold", pad=12)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5, zorder=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Visualizer] Layout saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="FastFill Layout Visualizer")
    parser.add_argument("--data", default=None, help="Path to jsonl file (train/dev/test)")
    parser.add_argument("--index", type=int, default=0, help="Line index to visualize")
    parser.add_argument("--out", default="layout.png", help="Output PNG path")
    args = parser.parse_args()

    if not args.data:
        # Create a synthetic demo room if no data passed
        demo_room = {
            "room_type": "bedroom",
            "boundary": [[0, 0], [4.5, 0], [4.5, 3.8], [0, 3.8]],
            "fixed": [
                {"id": "door_1", "size": [0.9, 0.15, 2.0], "pos": [0.8, 0, 0], "yaw": 0},
                {"id": "window_1", "size": [1.5, 0.2, 1.2], "pos": [4.5, 2.0, 1.0], "yaw": 90}
            ],
            "objects": [
                {"id": "double_bed_1", "size": [2.0, 1.8, 0.8]},
                {"id": "nightstand_1", "size": [0.5, 0.5, 0.5]},
                {"id": "wardrobe_1", "size": [0.6, 1.6, 2.0]},
                {"id": "desk_1", "size": [0.7, 1.2, 0.75]},
                {"id": "chair_1", "size": [0.5, 0.5, 0.8]}
            ]
        }
        demo_placements = [
            {"id": "double_bed_1", "pos": [2.0, 2.7, 0], "yaw": 270},
            {"id": "nightstand_1", "pos": [3.3, 2.7, 0], "yaw": 270},
            {"id": "wardrobe_1", "pos": [0.4, 2.8, 0], "yaw": 0},
            {"id": "desk_1", "pos": [3.8, 0.5, 0], "yaw": 180},
            {"id": "chair_1", "pos": [3.2, 0.5, 0], "yaw": 0}
        ]
        draw_layout(demo_room, demo_placements, args.out)
        return

    with open(args.data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.index:
                rec = json.loads(line)
                user_msg = json.loads(rec["messages"][1]["content"])
                asst_msg = json.loads(rec["messages"][2]["content"])
                draw_layout(user_msg, asst_msg.get("placements", []), args.out)
                break


if __name__ == "__main__":
    main()
