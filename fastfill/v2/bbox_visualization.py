"""Self-contained colored bbox GLB/SVG; no assets, model weights or network I/O."""
from __future__ import annotations

import hashlib
from html import escape
import json
import math
import struct


PALETTE = ((.35, .57, .78), (.77, .53, .33), (.46, .68, .48), (.61, .52, .74), (.76, .67, .39))
FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def _color(category):
    index = int.from_bytes(hashlib.sha256(category.encode()).digest()[:2], "little") % len(PALETTE)
    return PALETTE[index]


def _label(value):
    """XML 1.0 allows fewer control characters than JSON; sanitize display only."""
    cleaned = "".join(c if (ord(c) in (9, 10, 13) or 0x20 <= ord(c) <= 0xD7FF
                           or 0xE000 <= ord(c) <= 0xFFFD or 0x10000 <= ord(c) <= 0x10FFFF)
                      else "\ufffd" for c in value)
    return escape(cleaned)


def glb_bytes(scene):
    """Bake world coordinates into Y-up glTF vertices: (x,y,z)->(x,z,-y).

    Every object is a separate named mesh node. Full OBB corners are used as-is,
    so neither dimensions nor position are adjusted to make a scene look valid.
    """
    document = {"asset": {"version": "2.0", "generator": "FastFill v2 bbox handoff"},
                "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [],
                "materials": [], "buffers": [], "bufferViews": [], "accessors": [],
                "extras": {"geometry_kind": "predicted_bbox_proxy", "source_frame": "right_handed_z_up",
                           "world_to_gltf": "(x,y,z) -> (x,z,-y)", "asset_retrieval": "not_attempted"}}
    binary = bytearray()
    def accessor(values, component, kind, target):
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        flat = [x for row in values for x in row] if kind == "VEC3" else values
        fmt = "f" if component == 5126 else "H"
        if any(not math.isfinite(v) or abs(v) > 3.4e38 for v in flat):
            raise ValueError("bbox geometry cannot be represented by finite float32 GLB coordinates")
        packed = struct.pack("<" + fmt * len(flat), *flat)
        binary.extend(packed)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(packed), "target": target}
        index = len(document["accessors"])
        result = {"bufferView": len(document["bufferViews"]), "componentType": component,
                  "count": len(values), "type": kind}
        if kind == "VEC3":
            result = {**result, "min": [min(row[q] for row in values) for q in range(3)],
                      "max": [max(row[q] for row in values) for q in range(3)]}
        document["bufferViews"].append(view)
        document["accessors"].append(result)
        return index
    def mesh(name, corners, faces, color):
        vertices = [[x, z, -y] for x, y, z in corners]
        positions = accessor(vertices, 5126, "VEC3", 34962)
        triangles = [v for a, b, c, d in faces for v in (a, b, c, a, c, d)]
        indices = accessor(triangles, 5123, "SCALAR", 34963)
        material = {"name": name, "doubleSided": True,
                    "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.], "metallicFactor": 0., "roughnessFactor": 1.}}
        document["materials"].append(material)
        document["meshes"].append({"name": name, "primitives": [{"attributes": {"POSITION": positions},
                                    "indices": indices, "material": len(document["materials"]) - 1, "mode": 4}]})
        document["nodes"].append({"name": name, "mesh": len(document["meshes"]) - 1})
        document["scenes"][0]["nodes"].append(len(document["nodes"]) - 1)
    room = scene["room"]
    floor = [[x, y, room.get("floor_z_m") or 0.] for x, y in room["floor_polygon_xy_m"]]
    # Fan triangles are valid for the rectangular direct-request room. General
    # non-convex room conditions get a bbox-outline proxy instead of fake floor.
    low = [min(p[q] for p in floor) for q in (0, 1)]
    high = [max(p[q] for p in floor) for q in (0, 1)]
    z = floor[0][2]
    room_node = "room_extent_proxy"
    while room_node in {obj["id"] for obj in scene["objects"]}:
        room_node = "_" + room_node
    mesh(room_node, [[low[0], low[1], z], [high[0], low[1], z],
                              [high[0], high[1], z], [low[0], high[1], z]], ((0, 1, 2, 3),), (.88, .86, .82))
    for obj in scene["objects"]:
        mesh(obj["id"], obj["bbox"]["corners_m"], FACES, _color(obj["category"]))
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    return (struct.pack("<4sII", b"glTF", 2, length) + struct.pack("<I4s", len(encoded), b"JSON") + encoded
            + struct.pack("<I4s", len(binary), b"BIN\0") + bytes(binary))


def preview_svg(scene):
    """A portable isometric proxy preview; labels are escaped as XML text."""
    room, objects = scene["room"], scene["objects"]
    floor = [[x, y, room.get("floor_z_m") or 0.] for x, y in room["floor_polygon_xy_m"]]
    all_points = floor + [p for obj in objects for p in obj["bbox"]["corners_m"]]
    def project(p):
        return [(p[0] - p[1]) * math.sqrt(3) / 2, (p[0] + p[1]) / 2 - p[2]]
    projected = [project(p) for p in all_points]
    low = [min(p[q] for p in projected) for q in (0, 1)]
    high = [max(p[q] for p in projected) for q in (0, 1)]
    scale = min(800 / max(high[0] - low[0], 1e-9), 460 / max(high[1] - low[1], 1e-9))
    def points(vertices):
        return " ".join(f"{50 + (project(p)[0] - low[0]) * scale:.3f},{90 + (project(p)[1] - low[1]) * scale:.3f}" for p in vertices)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="620" viewBox="0 0 900 620">',
             '<rect width="900" height="620" fill="#f4f6f6"/>',
             f'<text x="28" y="35" font-family="sans-serif" font-size="22">{_label(room.get("room_type") or "Room")} · {len(objects)} bbox envelopes</text>',
             '<text x="28" y="60" font-family="sans-serif" font-size="14">Bbox proxy preview · category colors · no assets or support certification</text>',
             f'<polygon points="{points(floor)}" fill="#e4e0d8" stroke="#a5aaa8"/>']
    faces = []
    for obj in objects:
        for face in FACES:
            vertices = [obj["bbox"]["corners_m"][index] for index in face]
            a, b = [[vertices[n][q] - vertices[0][q] for q in range(3)] for n in (1, 2)]
            normal = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
            if sum(normal) > 1e-12:
                depth = sum(p[0] + p[1] + p[2] for p in vertices) / 4
                faces = faces + [(depth, obj, face)]
    for _, obj, face in sorted(faces, key=lambda entry: entry[0]):
        shade = .84 if face == FACES[1] else .68
        rgb = [round(255 * (shade * channel + 1 - shade)) for channel in _color(obj["category"])]
        color = "#" + "".join(f"{channel:02x}" for channel in rgb)
        polygon = points([obj["bbox"]["corners_m"][index] for index in face])
        parts.append(f'<polygon points="{polygon}" fill="{color}" stroke="#526675" stroke-width="0.7"><title>{_label(obj["id"] + ": " + obj["category"])}</title></polygon>')
    return "\n".join(parts + ["</svg>\n"])
