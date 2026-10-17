"""V4 data repairs and audit helpers used by build.prep (on top of the v3.2 pipeline).

fix_bed_fronts   A bed's semantic front is its foot end, a short end. A bed that is clearly elongated along its
                 local Y with one short end (+Y or -Y side) against the boundary was annotated with its front on a
                 long side: turn the front 90 degrees so the head end is the side at the wall. Both signals are
                 required; near-square beds and beds touching the wall with a long side are left as they are.
                 Measured on the v3.2 IR (inspect_ir front test, short-end variant): ~11-12% of elongated beds in
                 IL3D_3dfront and Structured3D, ~15-20% in the ScanNet/3RScan re-layouts, ~0-2% elsewhere.
swallowing_box   A seat/bed box whose footprint holds the centres of two or more other floor objects it is not
                 tucked against (nightstands, chairs at tables are normal): an annotation box that covers its
                 neighbours (a "bed" box over a wardrobe and lamps). Quality flag only.
inside_furniture Whether an object classified "wall" (elevated, nothing directly below) actually sits inside a piece
                 of furniture (a book on a shelf level, food in a fridge): used to record why it was dropped.
"""
import math
import re

from shapely.geometry import Point, Polygon

from fastfill.anchors import FLAT
from fastfill.scene import footprint, head, norm_cat
from fastfill.validate import collision_kind

# reference flags a training run leaves out by default (train.py --exclude_flags); the clean eval subsets drop them too
DEFAULT_EXCLUDED_FLAGS = ("oob_objects", "overlapping_furniture", "fixed_collision")

# beds whose front is not the foot end, or which do not stand against a wall head first
BED_EXCLUDE = {"sofa", "day", "bunk", "loft", "dog", "pet", "cat", "treatment", "massage", "hospital", "baby", "crib",
               "murphy", "folding", "water", "canopy", "trundle"}
BED_MIN_RATIO = 1.15     # long / short footprint side; below this the long axis (and so the front) is ambiguous
BED_WALL_TOL = 0.15      # a side counts as against the wall when its midpoint is this close to the boundary

SWALLOWERS = {"bed", "sofa", "couch", "loveseat"}
# boxes that enclose other furniture by design: an L/U/corner/sectional sofa's box spans its coffee table, a loft or
# kids bed stands over its desk, a "combination" is several pieces in one box (build.COMPOSITE plus these)
ENCLOSING = re.compile(r"\b(combination|l shaped|u shaped|composite|kids|children|loft|bunk|sectional|corner|modular)\b")


def wall_side(o, ring, tol=BED_WALL_TOL):
    """Which side (+X, -X, +Y, -Y) of o's footprint has its midpoint closest to `ring` within tol, or None."""
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    x, y = o["pos"][0], o["pos"][1]
    best, name = tol, None
    for lab, (dx, dy) in (("+X", (hx, 0)), ("-X", (-hx, 0)), ("+Y", (0, hy)), ("-Y", (0, -hy))):
        d = ring.distance(Point(x + c * dx - s * dy, y + s * dx + c * dy))
        if d < best:
            best, name = d, lab
    return name


def is_plain_bed(o):
    return head(o) == "bed" and not set(norm_cat(o["category"]).split()) & BED_EXCLUDE


def fix_bed_fronts(objs, boundary):
    """In place: re-front beds annotated with the front on a long side (see module doc). Returns the number fixed;
    each fixed bed carries front_fix = "bed_rot90" and its source yaw/size in yaw_src / size_src."""
    ring = Polygon(boundary).exterior
    n = 0
    for o in objs:
        if o.get("parent") or not is_plain_bed(o):
            continue
        sx, sy = o["size"][0], o["size"][1]
        if sy < BED_MIN_RATIO * sx:
            continue
        side = wall_side(o, ring)
        if side not in ("+Y", "-Y"):
            continue
        # head end at local +Y -> the new front (+X) is old -Y: yaw - 90; head at -Y -> yaw + 90
        o["yaw_src"], o["size_src"] = o["yaw"], list(o["size"])
        o["yaw"] = (o["yaw"] + (-math.pi / 2 if side == "+Y" else math.pi / 2)) % (2 * math.pi)
        o["size"] = [sy, sx, o["size"][2]]
        o["front_fix"] = "bed_rot90"
        n += 1
    return n


# furniture whose back stands against a wall: a front pressed to the wall with open floor behind is a flipped label
BACK_TO_WALL = {"bed", "sofa", "couch", "loveseat", "wardrobe", "closet", "armoire", "bookshelf", "bookcase", "dresser",
                "sideboard", "desk", "piano", "stand", "console", "cabinet"}
NOT_BACK_TO_WALL = re.compile(r"\b(combination|l shaped|u shaped|corner|sectional|modular|wall|kitchen|bathroom|bath|"
                              r"night|island)\b")
FLIP_OPEN_M = 0.5        # the back side must be at least this far from the boundary (a corner or alcove is ambiguous)


def side_distance(o, ring, side):
    c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
    hx, hy = o["size"][0] / 2, o["size"][1] / 2
    dx, dy = {"+X": (hx, 0), "-X": (-hx, 0), "+Y": (0, hy), "-Y": (0, -hy)}[side]
    return ring.distance(Point(o["pos"][0] + c * dx - s * dy, o["pos"][1] + s * dx + c * dy))


def fix_flipped_fronts(objs, boundary):
    """In place: turn by 180 degrees back-to-wall furniture whose FRONT (+X) side is the side against the wall while
    its back (-X) side has open floor behind it (>= FLIP_OPEN_M). Measured on the v3.2 IR after excluding source
    no-front objects: 0-0.1% in the clean synthetic sources (the rule's false-positive floor), 0.5-2.6% in
    3D-FRONT/SpatialLM/Structured3D/HSSD/scans; a visual check of 5 random hits found 4 desks/cabinets/wardrobes
    facing the wall. (The 90-degree variant of this test hits mostly small cabinets in corners and is not applied.)
    Returns the number turned; each carries front_fix = "rot180" and yaw_src."""
    ring = Polygon(boundary).exterior
    n = 0
    for o in objs:
        if o.get("parent") or o.get("front_fix") or head(o) not in BACK_TO_WALL or \
                NOT_BACK_TO_WALL.search(norm_cat(o["category"])) or (head(o) == "bed" and not is_plain_bed(o)):
            continue
        if wall_side(o, ring) != "+X" or side_distance(o, ring, "-X") < FLIP_OPEN_M:
            continue
        o["yaw_src"] = o["yaw"]
        o["yaw"] = (o["yaw"] + math.pi) % (2 * math.pi)
        o["front_fix"] = "rot180"
        n += 1
    return n


def swallowing_box(objs):
    """True if a bed/sofa box contains the centres of >= 2 other floor objects it is not tucked against."""
    floor = [o for o in objs if not o.get("parent") and head(o) not in FLAT and o["size"][2] >= 0.06]
    fps = [Polygon(footprint(o)) for o in floor]
    for i, a in enumerate(floor):
        if head(a) not in SWALLOWERS or ENCLOSING.search(norm_cat(a["category"])):
            continue
        a0, a1 = a["pos"][2], a["pos"][2] + a["size"][2]
        inside = 0
        for j, b in enumerate(floor):
            if i == j or fps[j].area >= fps[i].area:
                continue
            if min(a1, b["pos"][2] + b["size"][2]) - max(a0, b["pos"][2]) <= 0.05:
                continue
            if fps[i].contains(Point(b["pos"][:2])) and collision_kind(a, b, fps[i], fps[j]) != "tuck":
                inside += 1
        if inside >= 2:
            return True
    return False


def inside_furniture(o, furniture):
    """o's centre lies on a footprint of `furniture` (dicts) and its bottom between that piece's bottom and top."""
    pt, z = Point(o["pos"][:2]), o["pos"][2]
    return any(f is not o and f["pos"][2] - 0.02 <= z < f["pos"][2] + f["size"][2] - 0.02
               and Polygon(footprint(f)).contains(pt) for f in furniture)
