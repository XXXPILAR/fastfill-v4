"""InternScenes Real2Sim re-layout of ARKitScenes videos (conventions: see internscenes.py).
Group = arkit:<video_id>; several videos of one visit (same venue) are listed in meta.group_aliases."""
from fastfill.adapters.internscenes import arkit_visits, load_real

SOURCE = "InternScenes_arkit"


def load(root):
    visit = arkit_visits(root)
    yield from load_real(root, "arkitscenes_exported.json", SOURCE,
                         lambda s: "arkit:" + s["source_scene_name"],
                         lambda s: {"video_id": s["source_scene_name"], "visit_id": visit.get(s["source_scene_name"]),
                                    "group_aliases": [f"arkit_visit:{visit[s['source_scene_name']]}"]
                                    if visit.get(s["source_scene_name"]) else []})
