"""InternScenes Real2Sim re-layout of ScanNet v2 scans (conventions: see internscenes.py).
room_type = ScanNet sceneType. Group = scannet:sceneXXXX (rescans _NN share it)."""
from fastfill.adapters.internscenes import load_real

SOURCE = "InternScenes_scannet"


def load(root):
    yield from load_real(root, "scannet_exported.json", SOURCE,
                         lambda s: "scannet:" + s["source_scene_name"].rsplit("_", 1)[0])
