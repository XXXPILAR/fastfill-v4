"""IL3D "Synthetic Data" subset: 6,954 generated single rooms furnished with HSSD assets (IL3D's own supplement).

Conventions (checked against IL3D's USDZ meshes, see il3d_3dfront.py for the shared parts):
- size = assets.json mesh extents x |layout scale| (the layout bbox is the unscaled mesh extent).
- position = the asset origin, which for HSSD meshes is NOT always the bottom center (hanging lamps, wall shelves).
  The mesh bounds come from the HSSD-200 asset catalog (XXXpilar__hssd_clean/asset_catalog.csv, same GLB frame as
  IL3D's USDZ, checked on 79 assets) plus EXTRA_BOUNDS for the 54 assets it lacks.
- boundary = the generated rectangular floor (exact for these rooms); no room height in the source.
- category = the asset's category (furniture_category = asset_category = layout category for all 47,837 objects); the
  box is the asset mesh the generator substituted, so its category is what the box is (a "bed" slot filled with a
  3.0 x 0.11 x 1.18 m railing mesh stays a railing). The layout's own label (object_name, e.g. "dryer-0") is kept
  as layout_name. Only when the asset category is the generic "furniture" (988 objects, HSSD label "Others": washers,
  dryers, dishwashers, shelving) the layout label names the object better, so it becomes the category (instance
  suffix dropped) and the asset category is kept as asset_category. Of those, 1 ("floor_drain_cover" on a washer
  mesh) is thereby shown as a fixed structure box instead of placed. Against the mesh's HSSD synset (catalog
  wnsynsetkey) the name agrees for 449, differs for 453 and the mesh has none for 86; the differences are mostly
  look-alike swaps ("dryer" on 209 washing-machine placements, "washing_machine"/"dryer" on 78 of a "DISHWASHER HOBART"
  mesh, shelving names on a bathroom rack filed as handcart.n.01). The name is used anyway: it is the layout's label
  of the slot whose pose is the target, the size already is the mesh's, and a name-agrees-with-synset rule would put
  539 back to "furniture".
"""
import csv
import json
import math
import os
import re

from fastfill.adapters.il3d_3dfront import IL3D, il3d_extra, load_assets, placed_size
from fastfill.adapters.unified import convert_room, iter_records

SOURCE = "IL3D_synthetic"
FRONT_OFFSET_DEG = -90   # HSSD mesh +Z -> IR local -Y (front test)
GENERIC_ASSET = "furniture"   # an asset category that names nothing: the layout's object_name is used instead

# mesh bounds (m, y-up asset frame) of the 54 IL3D HSSD assets absent from the HSSD-200 catalog, read from
# IL3D HSSD.zip USDZ (extents match assets.json meta for all 54)
EXTRA_BOUNDS = {
    "08d386c25702f0613120b54e2b0615b4e4d198f7": ([-0.828425, 0.000115, -0.215669], [0.828425, 1.464155, 0.215669]),
    "0a3e0a10ec97c41a36939e89b0591d2cfe5e41e1": ([-0.6, -0.0, -0.651672], [0.6, 0.882244, 0.651672]),
    "103a70c6e8d54ac0cfc0fe28d66a8bea6b7c9144": ([-0.762062, -0.000101, -0.766263], [0.762062, 0.939839, 0.766263]),
    "1b583238d6351379093ac7ac1df38699568bb4ec": ([-0.60996, -0.0, -0.2159], [0.609961, 0.787413, 0.2159]),
    "1cdcc88935c2babba66677fa1d95aee0e6a410f3": ([-0.395, -0.0, -0.165], [0.395, 0.7, 0.165]),
    "1e4e58bf53e51df27beeb774b5d70818de124068": ([-0.323064, 0.0, -0.397458], [0.323064, 0.924784, 0.397458]),
    "252ea1f9106ada162cf68bb61c71efd4a62d4c87": ([-0.701118, -0.0, -0.16], [0.701119, 1.474944, 0.16]),
    "28f1e0bae95e70346996067a45e4e32c2e1b7d75": ([-1.106139, -0.002024, -0.792466], [1.106139, 0.854505, 0.792466]),
    "2a6fa282275e6ddb1fc35ebdc6a9ef91ae93a8ae": ([-0.45, 1.21079, -0.153132], [0.45, 2.274057, 0.153132]),
    "31a2b3d52aafa438f763b1d4a6b60297b95fa4e4": ([-0.250502, -3e-06, -0.089994], [0.250502, 0.161684, 0.089994]),
    "36b072f5c0db91a6b77fb0bfeaef686ee23a37f9": ([-1.338968, -0.008765, -1.339009], [1.339019, 2.507386, 1.338978]),
    "381f06fc43fe78e832a882769242e022dea8c81e": ([-0.751624, -0.000244, -0.799559], [0.747077, 1.023986, 0.798994]),
    "38da5efc3b4c3c250718e555cf68439746c71c65": ([-0.8, -0.0, -0.578804], [0.8, 0.75, 0.578804]),
    "3a886310f67f7cca9d6ea34b5b1b28bc43f282f5": ([-0.69831, 0.0, -0.521203], [0.69831, 1.040641, 0.521203]),
    "3a8afefa6b4a735293520f5bd14f3ec9d456a004": ([-1.82, -0.0, -0.251171], [1.819989, 1.97001, 0.251171]),
    "3dca32a89b68041131bf3b0db5a07b80ad26c0c7": ([-0.53, 0.0, -0.225], [0.53, 1.9, 0.225]),
    "3e8fb8993fed7051cb6da2ee1001d1cba046aaf8": ([-0.466968, -6.3e-05, -0.629036], [0.659181, 0.798143, 0.629036]),
    "46e08ada1af32862e55de18a583e8f60e7fa7917": ([-0.803414, -0.00243, -0.299899], [0.713549, 0.850902, 0.30151]),
    "47a95d3baa393487ba90ce9c90509cff0c9bb4e6": ([-0.43042, -0.0, -0.1524], [0.430419, 1.067658, 0.1524]),
    "4d36e3731f127abfb07f1d970239d6d9f21e1b3e": ([-0.703974, -0.0, -0.173289], [0.703974, 0.741946, 0.17329]),
    "4d6310a9ffa5b50e7dc45940cf724ba1fa3ffddd": ([-1.850866, -0.015017, -0.288937], [1.850866, 2.696201, 0.288936]),
    "51824f2ecd931fbed1389e715e8184354c6f62d4": ([-0.55, -0.004554, -0.127143], [0.55, 0.452, 0.065389]),
    "520a96262a7843c84a8cbebc172e70830d71c2db": ([-0.185151, -0.0, -0.158055], [0.185151, 1.602131, 0.158055]),
    "5be064296d6117e1151e58b8f858b9147b6279a9": ([-0.64419, -9e-05, -0.175091], [0.644901, 1.667616, 0.175091]),
    "605f233dd70c9a66d02914f1420d964642bdba81": ([-1.07618, -0.0, -1.382074], [1.07618, 0.836965, 1.382071]),
    "62fd5aea80235288d1896ddab73cdcef97a6a585": ([-0.630063, -0.0, -0.536025], [0.630063, 0.859829, 0.536025]),
    "6cf17da9a4062b73cafe548996b92e167d023a9a": ([-0.562498, -0.0, -0.510002], [0.562497, 0.9, 0.510002]),
    "7718968f80499672922d0434c8c617b0902f0bad": ([-0.810003, 0.0, -0.6125], [0.810003, 0.717001, 0.612501]),
    "777d1748a60d9b3ae52ad799aa0e215fd22715c6": ([-0.461928, -1.1e-05, -0.207056], [0.452472, 0.7493, 0.199345]),
    "799439ef082a442613ddcadd2603aeeeb477b13e": ([-0.7, 0.00325, -0.7], [0.7, 0.45887, 0.7]),
    "79d4fe8642358909adaa07a8b05793bcdd7f4450": ([-0.701118, -0.0, -0.16], [0.701119, 1.474944, 0.16]),
    "8597e0af462124aef9ec06dde13f45bb2b223a74": ([-0.5, -7.7e-05, -0.614043], [0.5, 1.09571, 0.611347]),
    "88ed586afa70ba6619fb122f205253a4a9d74aaa": ([-0.305, -0.0, -0.165], [0.305, 1.830585, 0.165]),
    "92bcbbdf57f3668c4b6192b9de9dfd95640de0d1": ([-0.95, 0.000194, -0.225], [0.95, 1.060194, 0.225001]),
    "9a1c224cd11be35a5c28347de7868674f56789da": ([-0.15, -0.002598, -0.45], [0.15, 0.3, 0.45]),
    "a375e44843d9ca2e2dda1bae8b65edbc8b84bbf0": ([-0.776006, 3.5e-05, -0.775123], [0.775905, 0.978471, 0.775655]),
    "a59f443c419c8514ecface071b4fd67bd44b4471": ([-0.415, -0.0, -0.1925], [0.415, 1.45, 0.192501]),
    "a60150783b79e0fb4cb3d938483f718585c52bda": ([-0.51, -0.0, -0.215], [0.51, 1.83, 0.215]),
    "a7323daeac0ded5ebafdb6231be3932e88a5b7f3": ([-0.45085, -0.0, -0.333766], [0.45085, 1.638814, 0.333766]),
    "b827ad39b820d4908a4222dc95dd4b63ecf6c0a6": ([-0.45, 1.21079, -0.153132], [0.45, 2.274053, 0.153132]),
    "ba28803b05660ca87ad0650276988f02dce1081e": ([-0.774645, -0.0, -0.774677], [0.774645, 0.914617, 0.774677]),
    "bc58cc78e7268f25da1dbee0f989c25c7e3d1108": ([-1.82, -0.0, -0.2975], [1.820001, 1.898, 0.2975]),
    "bf70431f51b8e06f2d7ef70916682dcd943553b8": ([-0.71163, -0.0, -0.580984], [0.716089, 0.99339, 0.3]),
    "c184f9647c5788e29d55dbf622bae24513ec48ec": ([-0.1775, -0.0, -0.145], [0.1775, 1.310004, 0.145]),
    "c4492609d027c557c5dc1ab8fc83bdd2e7617e12": ([-0.449618, -0.0, -0.183844], [0.449618, 1.52999, 0.150187]),
    "ca6bc7678efa7a97f59ee6a650217a5492b3a0f8": ([-0.770001, -0.0, -0.764999], [0.770001, 0.7, 0.764999]),
    "cee999e8b57589a4862425c588838f28bcc56f3b": ([-0.702015, -1e-06, -1.076095], [0.702015, 0.890897, 1.076095]),
    "d724fb718897f8200e6203a094369dc5e6c45ca2": ([-0.550002, 0.0, -0.45], [0.550002, 0.77, 0.45]),
    "dd56f602c6c29a97bbff71f4e48f07454ec97b62": ([-0.521833, 0.001956, -0.773037], [0.521833, 0.385286, 0.773038]),
    "efb3a025089c14264fde9b03c67d7353fb6759c5": ([-0.5334, -0.0, -0.1524], [0.5334, 1.57988, 0.167796]),
    "f8261de437a1f1f23263e4df0deb0a493ed52387": ([-0.530916, -0.0, -0.3048], [0.530916, 0.662856, 0.3048]),
    "ff0d0a7fe6e2a685cf03bf6ef2e32ae44d79372d": ([-0.45, 1.21079, -0.153132], [0.45, 2.274057, 0.153132]),
    "ff587a04fe3412b83b8fc870a7216374e70506c4": ([-0.828425, 0.000114, -0.215669], [0.828425, 1.473051, 0.215669]),
    "xxxxbc6e2459xae45x49d0x86c4xe27c2b0767a7": ([-0.399916, 0.0, -0.2032], [0.399916, 1.942959, 0.2032]),
}


def load_catalog(root):
    with open(os.path.join(root, "XXXpilar__hssd_clean/asset_catalog.csv")) as f:
        return {r["template_name"]: (json.loads(r["local_bbox_min"]), json.loads(r["local_bbox_max"]))
                for r in csv.DictReader(f) if r["local_bbox_min"]} | EXTRA_BOUNDS


def bottom_center(f, bounds):
    """Asset-origin position -> bottom center of the placed box (IR frame), or None."""
    p, rot = f.get("furniture_position"), f.get("furniture_rotation") or {}
    if not (p and bounds and rot.get("z") is not None):
        return None
    (x0, y0, z0), (x1, y1, z1) = bounds
    s = f["source_fields"]["layout_object"]["scale"]
    cx, cz, by = s[0] * (x0 + x1) / 2, s[2] * (z0 + z1) / 2, min(s[1] * y0, s[1] * y1)
    yaw = math.radians(rot["z"])                 # yaw of mesh +X; mesh +Z is IR local -Y
    c, n = math.cos(yaw), math.sin(yaw)
    return {"x": p["x"] + c * cx + n * cz, "y": p["y"] + n * cx - c * cz, "z": p["z"] + by}


def labels(f):
    """-> (IR category, extra IR fields): the layout label is always kept; it is the category only for GENERIC_ASSET."""
    cat, name = f.get("furniture_category"), f.get("object_name")
    if cat == GENERIC_ASSET and name:
        return re.sub(r"-\d+$", "", name), {"layout_name": name, "asset_category": cat}
    return cat, {"layout_name": name} if name else {}


def load(root):
    assets, catalog = load_assets(root), load_catalog(root)
    for scene in iter_records(os.path.join(root, IL3D, "synthetic_data.json")):
        for room in scene["rooms"]:
            furniture = []
            for f in room["furniture"]:
                f = dict(f)
                f["furniture_size"] = placed_size(f, assets.get(f.get("furniture_asset_id")))
                f["furniture_position"] = bottom_center(f, catalog.get(f.get("furniture_asset_id")))
                f["furniture_category"], f["_labels"] = labels(f)
                furniture.append(f)
            yield convert_room({**room, "furniture": furniture}, source=SOURCE, uid=f"il3d:{scene['scene_id']}",
                               group=f"il3d:{scene['scene_id']}", subset="Synthetic Data", boundary_type="polygon",
                               front_offset_deg=FRONT_OFFSET_DEG, extra=lambda f: {**il3d_extra(f), **f["_labels"]},
                               meta={"il3d_scene_id": scene["scene_id"], "front_known": True,
                                     "position_from": "asset origin + HSSD-200 catalog mesh bounds",
                                     "size_from": "assets.json mesh extents x layout scale"})


if __name__ == "__main__":
    f = {"furniture_position": {"x": 1.0, "y": 1.0, "z": 2.0}, "furniture_rotation": {"z": 90.0},
         "source_fields": {"layout_object": {"scale": [1, 2, 1]}}}
    b = bottom_center(f, ([-0.5, -0.5, 0.0], [0.5, 0.0, 0.4]))   # hanging: origin at top, box centered at z=+0.2
    assert all(abs(b[k] - v) < 1e-9 for k, v in {"x": 1.2, "y": 1.0, "z": 1.0}.items()), b
    assert labels({"furniture_category": "furniture", "object_name": "clothes_dryer-2"}) == (
        "clothes_dryer", {"layout_name": "clothes_dryer-2", "asset_category": "furniture"})
    assert labels({"furniture_category": "balcony_railing", "object_name": "bed-0"}) == ("balcony_railing", {"layout_name": "bed-0"})
    assert labels({"furniture_category": "furniture"}) == ("furniture", {})
    print("il3d_synthetic self-check ok")
