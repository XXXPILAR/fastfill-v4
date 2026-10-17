"""Recreate the CPU-only RoomGenBench integration fixture in a new output path.

Run from the FastFill repository root:
  python outputs/fastfill_v2/audit-four-20261006/roomgenbench-dynamic/smoke.py \
    --output-dir /tmp/fastfill-roomgenbench-new-smoke
Synthetic box meshes exercise GLB consumption; they are not learned assets.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import trimesh

from fastfill.v2.direct_layout import export_handoff, request_to_condition
from fastfill.v2.io import safe_output
from fastfill.v2.roomgenbench import assemble_handoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roomgenbench-root", default="RoomGenBench")
    args = parser.parse_args()
    root = safe_output(args.output_dir, create=True)
    condition = request_to_condition({"room_type": "study", "room_size_m": [5., 4.], "furniture_list": [
        {"id": "desk", "category": "desk"}, {"id": "cup", "category": "cup"}]}, room_size_semantics="reference_extent")
    condition = {**condition, "room": {**condition["room"], "fixed_objects": [
        {"id": "fixed_shelf", "category": "shelf", "size_local_m": [1., .4, 2.],
         "bottom_center_m": [4., 2., 0.], "yaw_rad": .2}]},
        "objects": [{**condition["objects"][0], "support_parent": "floor"},
                    {**condition["objects"][1], "support_parent": "desk"}],
        "constraints": [{"type": "near", "object_id": "desk", "target_id": "fixed_shelf", "max_distance_m": 2.}]}
    layout = {"schema_version": "fastfill.v2", "objects": [
        {"id": "desk", "target_size_local_m": [1.4, .7, .75], "bottom_center_m": [2., 1.5, .3], "yaw_rad": .7},
        {"id": "cup", "target_size_local_m": [.1, .08, .15], "bottom_center_m": [2., 1.5, 1.05], "yaw_rad": -.4}]}
    handoff = export_handoff(root / "handoff", condition, layout)
    reference = Path(args.roomgenbench_root).resolve()
    proxy = assemble_handoff(handoff, root / "layout_boxes", roomgenbench_root=reference)
    scene = json.loads((handoff / "roomgenbench_scene.json").read_text())
    assets = root / "synthetic_fixture_assets"
    assets.mkdir()
    for obj in scene["objects"]:
        mesh = trimesh.creation.box(extents=[.9, .8, .6])
        mesh.apply_translation([.2, .8, -.1])
        mesh.export(assets / f'{obj["asset_key"]}.glb')
        (assets / f'{obj["asset_key"]}.json').write_text(json.dumps({
            "asset_key": obj["asset_key"], "method": "synthetic_fixture", "status": "ok",
            "prompt": obj["description"], "seconds": 0.,
            "notes": "CPU-created geometric integration fixture; no learned generation or real asset lookup"}))
    mesh = assemble_handoff(handoff, root / "generated_mesh_fixture", method="synthetic_fixture",
                            assets_dir=assets, roomgenbench_root=reference)
    empty = root / "empty_assets"
    empty.mkdir()
    failure = assemble_handoff(handoff, root / "missing_assets", method="synthetic_fixture",
                               assets_dir=empty, roomgenbench_root=reference)
    report = {"scope": "synthetic_dynamic_mesh_integration_fixture_only", "trained_model": False,
              "real_asset_generation": False, "proxy_counts": proxy["counts"], "fixture_mesh_counts": mesh["counts"],
              "missing_asset_counts": failure["counts"], "fixed_bbox_proxies": proxy["fixed_bbox_proxies"],
              "reference_assembler": proxy["reference_assembler"],
              "not_executed": ["learned asset generation", "Blender", "mesh collision validation", "physics", "Solver", "persistent Host"]}
    (root / "smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
