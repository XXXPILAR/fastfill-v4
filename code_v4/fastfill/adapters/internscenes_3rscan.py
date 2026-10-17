"""InternScenes Real2Sim re-layout of 3RScan scans (conventions: see internscenes.py). Group = 3RScan reference scan."""
from fastfill.adapters.internscenes import load_real, rscan_reference

SOURCE = "InternScenes_3rscan"


def load(root):
    ref = rscan_reference(root)
    yield from load_real(root, "3rscan_exported.json", SOURCE,
                         lambda s: f"3rscan:{ref.get(s['source_scene_name'], s['source_scene_name'])}",
                         lambda s: {"scan_id": s["source_scene_name"],
                                    "is_reference_scan": ref.get(s["source_scene_name"]) == s["source_scene_name"]})
