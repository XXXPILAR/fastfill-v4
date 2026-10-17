"""InternScenes Real2Sim re-layout of Matterport3D regions (conventions: see internscenes.py).
One record = one Matterport region (a room-level unit). Group = mp3d:<house hash>.
InternScenes ships 96 regions twice within a house (e.g. 1LXtFkjw3qL region2 == region29: identical boundary,
same object ids/categories, boxes within 1 cm); only the first copy is emitted (1696 -> 1600 records)."""
import numpy as np

from fastfill.adapters.internscenes import load_real

SOURCE = "InternScenes_mp3d"


def load(root):
    seen = {}
    for ir in load_real(root, "matterport3d_exported.json", SOURCE,
                        lambda s: "mp3d:" + s["source_layout"].split("/")[2],
                        lambda s: {"region": s["source_scene_name"]}):
        key = (ir["group"], tuple((o["id"].rsplit(":", 1)[1], o["category"]) for o in ir["objects"]))
        arr = np.array([o["size"] + o["pos"] for o in ir["objects"]], dtype=float)
        bnd = np.array(ir["boundary"], dtype=float)
        if any(b.shape == bnd.shape and a.shape == arr.shape and np.allclose(b, bnd, atol=0.01)
               and np.allclose(a, arr, atol=0.01) for a, b in seen.get(key, [])):
            continue
        seen.setdefault(key, []).append((arr, bnd))
        yield ir
