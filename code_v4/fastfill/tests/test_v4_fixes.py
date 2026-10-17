import math
from types import SimpleNamespace

from fastfill.build import prep
from fastfill.v4fixes import fix_bed_fronts, swallowing_box, wall_side
from shapely.geometry import Polygon

ROOM = [[0, 0], [4, 0], [4, 3], [0, 3]]


def bed(size, pos, yaw=0.0, category="bed"):
    return {"id": "b", "category": category, "size": list(size), "pos": list(pos), "yaw": yaw}


def test_bed_front_on_long_side_is_turned_to_the_foot_end():
    b = bed([1.6, 2.0, 0.5], [2.0, 2.0, 0.0])           # long along Y, +Y short end at the y=3 wall
    assert fix_bed_fronts([b], ROOM) == 1
    assert math.isclose(b["yaw"], math.radians(270)) and b["size"] == [2.0, 1.6, 0.5]
    assert wall_side(b, Polygon(ROOM).exterior) == "-X"      # head end now at the wall, front into the room
    assert b["front_fix"] == "bed_rot90" and b["size_src"] == [1.6, 2.0, 0.5] and b["yaw_src"] == 0.0

    b = bed([1.6, 2.0, 0.5], [2.0, 1.0, 0.0])           # -Y short end at the y=0 wall
    assert fix_bed_fronts([b], ROOM) == 1 and math.isclose(b["yaw"], math.radians(90))
    assert wall_side(b, Polygon(ROOM).exterior) == "-X"


def test_beds_left_alone():
    long_side = bed([2.0, 1.6, 0.5], [2.0, 2.2, 0.0])   # long along X, a long side at the wall: dorm-style, fine
    square = bed([1.9, 2.0, 0.5], [2.0, 2.0, 0.0])      # ratio < 1.15: front ambiguous
    middle = bed([1.6, 2.0, 0.5], [2.0, 1.5, 0.0])      # touches no wall
    sofa_bed = bed([1.6, 2.0, 0.5], [2.0, 2.0, 0.0], category="sofa bed")
    bunk = bed([1.0, 2.0, 1.6], [2.0, 2.0, 0.0], category="bunk bed")
    objs = [long_side, square, middle, sofa_bed, bunk]
    before = [dict(o) for o in objs]
    assert fix_bed_fronts(objs, ROOM) == 0 and objs == before


def test_flipped_front_is_turned_only_with_open_floor_behind():
    from fastfill.v4fixes import fix_flipped_fronts
    desk = {"id": "d", "category": "desk", "size": [0.6, 1.2, 0.75], "pos": [3.7, 1.5, 0], "yaw": 0.0}  # +X at x=4 wall
    assert fix_flipped_fronts([desk], ROOM) == 1 and math.isclose(desk["yaw"], math.pi) and desk["front_fix"] == "rot180"
    ok = {"id": "d", "category": "desk", "size": [0.6, 1.2, 0.75], "pos": [3.7, 1.5, 0], "yaw": math.pi}  # back at wall
    # +X side at the x=4 wall, -X side at x=1.0 with 1 m of floor behind it: facing the wall -> flipped
    deep = {"id": "w", "category": "wardrobe", "size": [3.0, 1.2, 2.0], "pos": [2.5, 1.5, 0], "yaw": 0.0}
    corner = {"id": "c", "category": "cabinet", "size": [0.4, 0.4, 0.8], "pos": [3.8, 0.2, 0], "yaw": 0.0}  # in a corner
    sofa_l = {"id": "s", "category": "l shaped sofa", "size": [0.9, 2.0, 0.8], "pos": [3.55, 1.5, 0], "yaw": 0.0}
    assert fix_flipped_fronts([ok], ROOM) == 0 and fix_flipped_fronts([deep], ROOM) == 1
    assert fix_flipped_fronts([corner], ROOM) == 0 and fix_flipped_fronts([sofa_l], ROOM) == 0


def test_swallowing_box():
    big = {"id": "bed", "category": "bed", "size": [3.0, 2.5, 0.6], "pos": [2, 1.5, 0], "yaw": 0}
    wardrobe = {"id": "w", "category": "wardrobe", "size": [0.6, 1.0, 2.0], "pos": [2.8, 1.5, 0], "yaw": 0}
    lamp = {"id": "l", "category": "floor lamp", "size": [0.3, 0.3, 1.5], "pos": [1.2, 1.5, 0], "yaw": 0}
    assert swallowing_box([big, wardrobe, lamp])
    stands = [{"id": f"n{i}", "category": "nightstand", "size": [0.4, 0.4, 0.5], "pos": [x, 1.5, 0], "yaw": 0}
              for i, x in enumerate((1.0, 3.0))]
    assert not swallowing_box([big] + stands)            # nightstands tucked against a bed are normal
    assert not swallowing_box([big, wardrobe])           # a single overlap is left to the collision metrics
    for name in ("l shaped sofa", "combination sofa", "kids bed", "loft bed", "corner sofa"):   # enclosing by design
        assert not swallowing_box([{**big, "category": name}, wardrobe, lamp]), name


def args(**kw):
    return SimpleNamespace(boundary_types=["polygon", "hull"], anchors=["floor", "object"], source_anchors={},
                           min_objects=1, max_vertices=0, oob_tol=0.1, hidden_max=0.3, reject_flagged=[], **kw)


def test_prep_accounts_for_every_source_object():
    objs = [
        {"id": "desk", "category": "desk", "size": [1.2, .6, .75], "pos": [1, 1, 0], "yaw": 0},
        {"id": "lamp", "category": "table lamp", "size": [.2, .2, .4], "pos": [1.2, 1, .75], "yaw": 0},
        {"id": "case", "category": "bookcase", "size": [.4, 1.0, 2.0], "pos": [3.7, 1.5, 0], "yaw": 0},
        {"id": "book", "category": "book", "size": [.2, .15, .25], "pos": [3.7, 1.5, 1.1], "yaw": 0},
        {"id": "pic", "category": "painting", "size": [.8, .05, .6], "pos": [2, 2.97, 1.4], "yaw": 0},
        {"id": "pend", "category": "pendant lamp", "size": [.5, .5, .3], "pos": [2, 1.5, 2.3], "yaw": 0},
        {"id": "door", "category": "door", "size": [.9, .1, 2.1], "pos": [3, 0, 0], "yaw": 0},
        {"id": "unk", "category": "unknown", "size": [.5, .5, .5], "pos": [0.5, 2.5, 0], "yaw": 0},
        {"id": "bed", "category": "bed", "size": [1.6, 2.0, .5], "pos": [2.0, 1.95, 0], "yaw": 0},
    ]
    room = {"uid": "t::1", "source": "T", "group": "g", "room_type": "bedroom", "boundary": ROOM,
            "boundary_type": "polygon", "height": 2.8, "objects": objs, "meta": {"front_known": True}}
    out, why = prep(room, args())
    assert why is None
    dropped = out["meta"]["dropped"]
    assert sum(dropped.values()) + len(out["objects"]) == len(objs)
    assert dropped["inside_furniture"] == 1 and dropped["wall"] == 1 and dropped["ceiling"] == 1
    assert dropped["shown_fixed"] == 2                    # the door and the unlabelled floor box
    bedo = next(o for o in out["objects"] if o["category"] == "bed")
    assert bedo["front_fix"] == "bed_rot90" and out["meta"]["flags"]["bed_front_fixed"] == 1


def test_ase_openings_follow_the_nearest_wall():
    from fastfill.adapters.ase import openings
    from fastfill.adapters.unified import convert_room
    room = {"room_id": "r", "room_boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "doors": [{"position": {"x": 2, "y": 0.05, "z": 1.0}, "width": 0.9, "height": 2.0}],
            "windows": [{"position": {"x": 3.98, "y": 1.5, "z": 1.5}, "width": 1.2, "height": 1.0}]}
    ir = convert_room({**room, "furniture": openings(room, "r")}, source="ASE", uid="t", group="g",
                      boundary_type="polygon", center_z=True, front_offset_deg=-90)
    door, window = ir["objects"]
    assert door["size"] == [0.9, 0.1, 2.0] and abs(math.sin(door["yaw"])) < 1e-9      # along the y=0 wall
    assert window["size"] == [1.2, 0.1, 1.0] and abs(math.cos(window["yaw"])) < 1e-9  # along the x=4 wall
    assert door["pos"][2] == 0.0 and window["pos"][2] == 1.0                           # centre -> bottom


def test_contrast_pairs_are_feasible_and_exclusive():
    from fastfill.tools.contrast_set import build_pairs
    from fastfill.validate import check, holds
    room = {"uid": "r", "source": "T", "group": "g", "boundary_type": "polygon", "height": 2.8, "fixed": [],
            "boundary": [[0, 0], [6, 0], [6, 5], [0, 5]], "meta": {}, "objects": [
                {"id": "table_1", "category": "table", "size": [1.2, 0.8, 0.75], "pos": [2, 2.5, 0], "yaw": 0},
                {"id": "chair_1", "category": "chair", "size": [0.5, 0.5, 0.9], "pos": [1.2, 2.5, 0], "yaw": 0},
                {"id": "sofa_1", "category": "sofa", "size": [0.9, 2.0, 0.8], "pos": [5.5, 2.5, 0], "yaw": math.pi},
                {"id": "tv_stand_1", "category": "tv stand", "size": [0.4, 1.5, 0.5], "pos": [0.2, 4.0, 0], "yaw": 0}]}
    out, _ = build_pairs([room])
    pairs = {}
    for r in out:
        pairs.setdefault(r["meta"]["pair_id"], {})[r["meta"]["variant"]] = r
    assert pairs
    for p in pairs.values():
        a, b = p["A"], p["B"]
        assert holds(a["constraints"][0], a["objects"], a["boundary"]) and check(b)["valid"]
        assert holds(b["constraints"][0], b["objects"], b["boundary"])
        assert not holds(b["constraints"][0], a["objects"], a["boundary"])     # the reference layout fails B
        assert not holds(a["constraints"][0], b["objects"], b["boundary"])     # the witness fails A
        assert [o["id"] for o in a["objects"]] == [o["id"] for o in b["objects"]]
