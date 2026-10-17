"""Direct room/list -> bbox handoff, without requiring an asset resolver.

GLB tests decode the published binary convention independently of the exporter.
They intentionally do not use trimesh or treat a proxy box as a furniture mesh.
"""
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import pytest

from fastfill.v2.direct_layout import (
    export_handoff,
    layout_to_roomgenbench,
    layout_to_scene,
    request_to_condition,
    write_bbox_glb,
)
from fastfill.v2.schema import validate_condition


def request():
    return {"room_type": "living_room", "room_size_m": [5., 4., 2.8],
            "furniture_list": ["sofa", {"category": "chair", "count": 2},
                               {"id": "lamp", "category": "lamp", "description": "reading lamp"}]}


def layout(condition, *, yaw=0.):
    return {"schema_version": "fastfill.v2", "objects": [
        {"id": obj["id"], "target_size_local_m": [1.4, .7, .5 + index],
         "bottom_center_m": [2., 1.5, .2 * index], "yaw_rad": yaw}
        for index, obj in enumerate(condition["objects"])]}


def corners(position, size, yaw):
    """Independent full-extent, bottom-center OBB reference."""
    c, s = math.cos(yaw), math.sin(yaw)
    w, d, h = size
    return [[position[0] + c * x - s * y,
             position[1] + s * x + c * y, position[2] + z]
            for x, y, z in itertools.product((-w / 2, w / 2), (-d / 2, d / 2), (0, h))]


def corner_set(values):
    return sorted(tuple(round(float(coordinate), 6) for coordinate in point) for point in values)


def read_glb(path):
    data = Path(path).read_bytes()
    assert len(data) >= 28
    magic, version, length = struct.unpack_from("<III", data)
    assert (magic, version, length) == (0x46546C67, 2, len(data))
    offset, chunks = 12, []
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        assert chunk_length % 4 == 0
        end = offset + 8 + chunk_length
        assert end <= len(data)
        chunks.append((chunk_type, data[offset + 8:end]))
        offset = end
    assert offset == len(data)
    assert chunks[0][0] == 0x4E4F534A
    document = json.loads(chunks[0][1])
    binary = next(value for kind, value in chunks if kind == 0x004E4942)
    assert document["asset"]["version"] == "2.0"
    assert len(document["buffers"]) == 1
    assert "uri" not in document["buffers"][0]
    assert document["buffers"][0]["byteLength"] <= len(binary)
    return document, binary


def node_vertices(document, binary, name):
    """Apply a named node and its ancestors to its own mesh positions."""
    indices = [index for index, node in enumerate(document["nodes"]) if node.get("name") == name]
    assert len(indices) == 1
    node_index = indices[0]
    node = document["nodes"][node_index]
    parents = {child: index for index, parent in enumerate(document["nodes"]) for child in parent.get("children", [])}
    matrix = node_matrix(node)
    while node_index in parents:
        node_index = parents[node_index]
        matrix = multiply_matrices(node_matrix(document["nodes"][node_index]), matrix)
    values = []
    for primitive in document["meshes"][node["mesh"]]["primitives"]:
        accessor = document["accessors"][primitive["attributes"]["POSITION"]]
        assert accessor["componentType"] == 5126 and accessor["type"] == "VEC3"
        view = document["bufferViews"][accessor["bufferView"]]
        offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        stride = view.get("byteStride", 12)
        for index in range(accessor["count"]):
            point = struct.unpack_from("<3f", binary, offset + stride * index)
            values.append([sum(matrix[4 * j + q] * point[j] for j in range(3)) + matrix[12 + q]
                           for q in range(3)])
    return values


def multiply_matrices(left, right):
    return [sum(left[4 * k + row] * right[4 * column + k] for k in range(4))
            for column in range(4) for row in range(4)]


def node_matrix(node):
    if "matrix" in node:
        return node["matrix"]
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = node.get("scale", [1, 1, 1])
    tx, ty, tz = node.get("translation", [0, 0, 0])
    return [
        (1 - 2 * (y * y + z * z)) * sx, 2 * (x * y + z * w) * sx,
        2 * (x * z - y * w) * sx, 0,
        2 * (x * y - z * w) * sy, (1 - 2 * (x * x + z * z)) * sy,
        2 * (y * z + x * w) * sy, 0,
        2 * (x * z + y * w) * sz, 2 * (y * z - x * w) * sz,
        (1 - 2 * (x * x + y * y)) * sz, 0, tx, ty, tz, 1]


def test_request_expansion_preserves_ids_order_and_only_available_information():
    source = request()
    saved = deepcopy(source)
    condition = request_to_condition(source)
    validate_condition(condition)
    assert source == saved
    assert condition == request_to_condition(source)
    assert condition["room"]["room_type"] == "living_room"
    assert condition["room"]["floor_polygon_xy_m"] == [[0, 0], [5, 0], [5, 4], [0, 4]]
    assert condition["room"]["height_m"] == 2.8
    assert [obj["id"] for obj in condition["objects"]] == ["obj_0000", "obj_0001", "obj_0002", "lamp"]
    assert [obj["category"] for obj in condition["objects"]] == ["sofa", "chair", "chair", "lamp"]
    assert condition["objects"][-1]["description"] == "reading lamp"
    assert condition["constraints"] == []
    for obj in condition["objects"]:
        assert "support_parent" not in obj
        assert "fixed_size_local_m" not in obj and "asset_id" not in obj
    assert "openings" not in condition["room"] or condition["room"]["openings"] == []


def test_two_dimensional_room_does_not_invent_ceiling_or_force_tabletop_objects_to_floor():
    condition = request_to_condition({"room_type": "living_room", "room_size_m": [5, 4],
                                     "furniture_list": ["table", "cup", "wall mirror"]})
    validate_condition(condition)
    assert condition["room"].get("height_m") is None
    assert all("support_parent" not in obj for obj in condition["objects"])
    from fastfill.v2.batch import TinyTokenizer, collate_samples
    batch = collate_samples([{"condition": condition}], TinyTokenizer(), max_length=4096)
    assert not batch["fixed_position_mask"].any()
    assert batch["scale"][0, 2] == 3  # Protocol reference; no claim of an observed ceiling.


@pytest.mark.parametrize("update", [
    {"room_size_m": [0, 4]}, {"room_size_m": [-1, 4]},
    {"room_size_m": [True, 4]}, {"room_size_m": [float("nan"), 4]},
    {"room_size_m": [5, float("inf")]}, {"room_size_m": [5]},
    {"room_size_m": [5, 4, 0]}, {"room_size_m": [5, 4, 3, 2]},
    {"room_size_m": "5x4"}, {"room_type": ""}, {"room_type": 4},
    {"furniture_list": "chair"}, {"furniture_list": [True]},
    {"furniture_list": [""]}, {"furniture_list": [{"category": ""}]},
    {"furniture_list": [{"category": "chair", "description": 1}]},
    {"furniture_list": [{"category": "chair", "count": 0}]},
    {"furniture_list": [{"category": "chair", "count": -1}]},
    {"furniture_list": [{"category": "chair", "count": True}]},
    {"furniture_list": [{"category": "chair", "count": 1.5}]},
    {"furniture_list": [{"category": "chair", "count": 2.0}]},
    {"furniture_list": [{"category": "chair", "count": "2"}]},
    {"furniture_list": [{"id": "chairs", "category": "chair", "count": 2}]},
    {"furniture_list": [{"id": "", "category": "chair"}]},
    {"furniture_list": [{"id": "floor", "category": "chair"}]},
    {"furniture_list": [{"category": "chair", "actual_size_local_m": [1, 1, 1]}]},
    {"assets": []},
])
def test_invalid_minimal_requests_are_rejected(update):
    with pytest.raises(ValueError):
        request_to_condition({**request(), **update})


def test_missing_fields_duplicate_and_generated_id_collisions_are_rejected():
    for key in request():
        source = request()
        source.pop(key)
        with pytest.raises(ValueError):
            request_to_condition(source)
    for furniture in ([{"id": "same", "category": "chair"}, {"id": "same", "category": "table"}],
                      ["chair", {"id": "obj_0000", "category": "table"}]):
        with pytest.raises(ValueError):
            request_to_condition({**request(), "furniture_list": furniture})


@pytest.mark.parametrize("yaw", [0., math.pi / 2, math.radians(97), -math.pi])
def test_canonical_scene_uses_each_object_height_full_extents_and_rotation_invariant_local_size(yaw):
    condition = request_to_condition(request())
    prediction = layout(condition, yaw=yaw)
    saved_condition, saved_layout = deepcopy(condition), deepcopy(prediction)
    scene = layout_to_scene(condition, prediction)
    assert scene["schema_version"] == "fastfill.bbox-scene.v1"
    assert scene["frame"] == "right_handed_z_up"
    assert scene["room"] == condition["room"]
    assert len(scene["objects"]) == len(condition["objects"])
    for predicted, actual, requested in zip(prediction["objects"], scene["objects"], condition["objects"]):
        assert actual["id"] == requested["id"]
        assert actual["category"] == requested["category"]
        assert actual["description"] == requested["description"]
        assert actual["target_size_local_m"] == predicted["target_size_local_m"]
        assert actual["bottom_center_m"] == predicted["bottom_center_m"]
        bbox = actual["bbox"]
        assert bbox["size_local_m"] == predicted["target_size_local_m"]
        assert bbox["yaw_rad"] == yaw
        x, y, z = predicted["bottom_center_m"]
        assert bbox["center_m"] == pytest.approx([x, y, z + predicted["target_size_local_m"][2] / 2])
        assert corner_set(bbox["corners_m"]) == corner_set(corners(
            predicted["bottom_center_m"], predicted["target_size_local_m"], yaw))
    assert condition == saved_condition and prediction == saved_layout
    scene["room"]["floor_polygon_xy_m"][0][0] = -100
    scene["objects"][0]["target_size_local_m"][0] = 100
    assert condition == saved_condition and prediction == saved_layout


def test_scene_export_requires_all_requested_ids_once_without_padding_or_extra_instances():
    condition = request_to_condition(request())
    prediction = layout(condition)
    variants = [[], prediction["objects"][:-1], prediction["objects"] * 2,
                prediction["objects"] + [{**prediction["objects"][0], "id": "padding_0"}]]
    for objects in variants:
        with pytest.raises(ValueError):
            layout_to_scene(condition, {**prediction, "objects": objects})


def test_scene_export_joins_by_stable_identity_when_prediction_order_differs():
    condition = request_to_condition(request())
    prediction = layout(condition)
    shuffled = {**prediction, "objects": list(reversed(prediction["objects"]))}
    scene = layout_to_scene(condition, shuffled)
    by_id = {obj["id"]: obj for obj in prediction["objects"]}
    for obj in scene["objects"]:
        assert obj["bottom_center_m"] == by_id[obj["id"]]["bottom_center_m"]
        assert obj["target_size_local_m"] == by_id[obj["id"]]["target_size_local_m"]


@pytest.mark.parametrize("yaw", [0., math.pi / 2, math.radians(97), -math.pi])
def test_roomgenbench_converts_canonical_axis_and_preserves_world_bbox(yaw):
    condition = request_to_condition(request())
    prediction = layout(condition, yaw=yaw)
    saved_condition, saved_prediction = deepcopy(condition), deepcopy(prediction)
    downstream = layout_to_roomgenbench(condition, prediction)
    assert downstream["room_type"] == condition["room"]["room_type"]
    assert downstream["room"]["dimensions"] == {"width": 5., "length": 4., "height": 2.8}
    assert downstream["room"]["position"] == {"x": 0., "y": 0., "z": 0.}
    assert downstream["room"]["doors"] == [] and downstream["room"]["windows"] == []
    assert [(w["id"], w["start_point"], w["end_point"], w["height"], w["thickness"]) for w in downstream["room"]["walls"]] == [
        (f"wall_{i:02d}", dict(zip("xyz", a + [0.])), dict(zip("xyz", b + [0.])), 2.8, .1)
        for i, (a, b) in enumerate([([0., 0.], [5., 0.]), ([5., 0.], [5., 4.]), ([5., 4.], [0., 4.]), ([0., 4.], [0., 0.])])]
    for actual, predicted in zip(downstream["objects"], prediction["objects"]):
        assert {"id", "type", "description", "asset_key", "position", "rotation", "dimensions", "place_id"} <= set(actual)
        assert actual["id"] == predicted["id"]
        # No support was declared: only the object resting on the known floor is placed, and marked inferred.
        on_floor = predicted["bottom_center_m"][2] == 0.
        assert actual["place_id"] == ("floor" if on_floor else None)
        assert actual["support_status"] == ("inferred" if on_floor else "unknown")
        w, d, h = predicted["target_size_local_m"]
        assert actual["dimensions"] == {"width": d, "length": w, "height": h}
        position = [actual["position"][axis] for axis in "xyz"]
        assert position == predicted["bottom_center_m"]
        rotation = actual["rotation"]
        assert rotation["x"] == rotation["y"] == 0
        converted_yaw = math.radians(rotation["z"])
        assert corner_set(corners(position, [d, w, h], converted_yaw)) == corner_set(corners(position, [w, d, h], yaw))
        # SAGE's local +Y axis equals FastFill's +X axis in world coordinates.
        assert [-math.sin(converted_yaw), math.cos(converted_yaw)] == pytest.approx([math.cos(yaw), math.sin(yaw)])
    assert condition == saved_condition and prediction == saved_prediction


def test_roomgenbench_does_not_invent_unknown_room_height_or_support():
    condition = request_to_condition({**request(), "room_size_m": [5, 4]})
    downstream = layout_to_roomgenbench(condition, layout(condition))
    assert downstream["room"]["dimensions"].get("height") is None
    assert downstream["room"].get("ceiling_height") is None
    assert {w["height"] for w in downstream["room"]["walls"]} == {2.7}  # shell display default, not a claim
    assert all(obj["place_id"] is None for obj in downstream["objects"] if obj["position"]["z"] > .02)
    assert all(obj["support_status"] != "declared" for obj in downstream["objects"])


@pytest.mark.parametrize("yaw", [0., math.pi / 2, math.radians(97), -math.pi])
def test_glb_is_self_contained_and_applies_full_extent_bottom_center_z_up_to_y_up(tmp_path, yaw):
    condition = request_to_condition(request())
    prediction = layout(condition, yaw=yaw)
    scene = layout_to_scene(condition, prediction)
    output = tmp_path / "layout.glb"
    write_bbox_glb(output, scene)
    document, binary = read_glb(output)
    assert not document.get("images")
    assert not document.get("textures")
    for obj in prediction["objects"]:
        actual = node_vertices(document, binary, obj["id"])
        expected = [[x, z, -y] for x, y, z in corners(obj["bottom_center_m"], obj["target_size_local_m"], yaw)]
        # Vertices may be duplicated for flat face normals; compare unique corners.
        assert sorted(set(corner_set(actual))) == sorted(set(corner_set(expected)))
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_bbox_glb(output, scene)
    assert output.read_bytes() == saved


def test_handoff_exports_without_catalog_or_support_validation_and_preserves_raw_prediction(tmp_path):
    condition = request_to_condition(request())
    prediction = layout(condition)
    # Deliberately outside the room: diagnostics may reject it, but never shrink/delete it.
    prediction["objects"][0]["bottom_center_m"] = [-4., 1., .4]
    saved_condition, saved_prediction = deepcopy(condition), deepcopy(prediction)
    directory = tmp_path / "handoff"
    export_handoff(directory, condition, prediction)
    assert {"scene.json", "roomgenbench_scene.json", "layout.glb", "preview.svg", "diagnostics.json"} <= {p.name for p in directory.iterdir()}
    scene = json.loads((directory / "scene.json").read_text())
    assert scene == layout_to_scene(condition, prediction)
    assert scene["objects"][0]["bottom_center_m"] == [-4., 1., .4]
    assert len(scene["objects"]) == len(prediction["objects"])
    json.loads((directory / "diagnostics.json").read_text())
    assert ET.parse(directory / "preview.svg").getroot().tag == "{http://www.w3.org/2000/svg}svg"
    read_glb(directory / "layout.glb")
    assert condition == saved_condition and prediction == saved_prediction
    snapshots = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        export_handoff(directory, condition, prediction)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == snapshots


def test_invalid_handoff_is_rejected_before_any_directory_is_created(tmp_path):
    condition = request_to_condition(request())
    prediction = layout(condition)
    prediction["objects"][0]["target_size_local_m"] = [0., 1., 1.]
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError):
        export_handoff(directory, condition, prediction)
    assert not directory.exists()


def test_source_data_paths_are_protected_for_direct_exports():
    condition = request_to_condition(request())
    prediction = layout(condition)
    with pytest.raises(ValueError):
        export_handoff(Path("/Volumes/harddisk/3D_Room_Collections/new-handoff"), condition, prediction)
    with pytest.raises(ValueError):
        write_bbox_glb(Path("/Volumes/harddisk/3D_Room_Collections/new-layout.glb"), layout_to_scene(condition, prediction))


def test_prediction_cli_accepts_minimal_request_and_direct_handoff_without_assets(tmp_path):
    from fastfill.v2.batch import TinyTokenizer
    from fastfill.v2.model import ModelConfig, build_model
    from fastfill.v2.predict import main
    checkpoint = tmp_path / "model"
    build_model(ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16,
                           decoder_heads=2, decoder_layers=1, tiny_hidden_size=16)).save_pretrained(checkpoint)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request()))
    output = tmp_path / "prediction.json"
    handoff = tmp_path / "handoff"
    main(["--checkpoint", str(checkpoint), "--request", str(request_path), "--output", str(output),
          "--export-dir", str(handoff), "--device", "cpu"])
    prediction = json.loads(output.read_text())
    assert {obj["id"] for obj in prediction["objects"]} == {"obj_0000", "obj_0001", "obj_0002", "lamp"}
    scene = json.loads((handoff / "scene.json").read_text())
    assert scene == layout_to_scene(request_to_condition(request()), prediction)
    read_glb(handoff / "layout.glb")


def test_svg_escapes_markup_and_filters_xml_controls_without_changing_json_semantics(tmp_path):
    source = {"room_type": "living_room\u0001 <script>& room", "room_size_m": [5, 4],
              "furniture_list": [{"id": "chair\u0001 </title><script>bad</script>",
                                  "category": "chair\u0001 <>&", "description": "oak <>& chair"}]}
    condition = request_to_condition(source)
    prediction = layout(condition)
    handoff = tmp_path / "xml-safe"
    export_handoff(handoff, condition, prediction)
    scene = json.loads((handoff / "scene.json").read_text())
    assert scene["room"]["room_type"] == source["room_type"]
    assert scene["objects"][0]["id"] == source["furniture_list"][0]["id"]
    assert scene["objects"][0]["category"] == source["furniture_list"][0]["category"]
    assert scene["objects"][0]["description"] == source["furniture_list"][0]["description"]
    svg = (handoff / "preview.svg").read_text()
    tree = ET.fromstring(svg)
    assert "\u0001" not in svg
    assert {element.tag.rsplit("}", 1)[-1] for element in tree.iter()} <= {"svg", "rect", "text", "polygon", "title"}
    text = " ".join(element.text or "" for element in tree.iter())
    assert "<script>" in text  # Displayed text, never an executable SVG element.
    assert "<>&" in text


def test_prediction_refuses_output_file_as_ancestor_of_export_directory_before_model_load(tmp_path, monkeypatch):
    from fastfill.v2 import predict
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request()))
    output = tmp_path / "prediction.json"
    handoff = output / "handoff"
    def forbidden_load(*args, **kwargs):
        raise AssertionError("invalid output paths must be rejected before loading model weights")
    monkeypatch.setattr(predict, "load_model", forbidden_load)
    with pytest.raises(ValueError, match="overlap|contain"):
        predict.main(["--checkpoint", str(tmp_path / "missing-model"), "--request", str(request_path),
                      "--output", str(output), "--export-dir", str(handoff)])
    assert not output.exists() and not handoff.exists()


def test_generation_registry_follows_benchmark_type_description_key_with_first_instance_dimensions(tmp_path):
    from fastfill.v2.direct_layout import asset_key
    source = {**request(), "furniture_list": [{"category": "chair", "description": "oak chair", "count": 3},
                                              {"category": "chair", "description": "metal chair"}]}
    condition = request_to_condition(source)
    prediction = layout(condition)
    for obj in prediction["objects"]:
        obj["target_size_local_m"] = [1., .7, .9]
    prediction["objects"][1]["yaw_rad"] = math.pi / 2
    prediction["objects"][1]["bottom_center_m"] = [3., 2., 0.]
    prediction["objects"][2]["target_size_local_m"] = [1.2, .7, .9]
    handoff = tmp_path / "registry"
    export_handoff(handoff, condition, prediction)
    downstream = json.loads((handoff / "roomgenbench_scene.json").read_text())
    objects = downstream["objects"]
    # RoomGenBench prepare_inputs: slug(type)[:24] + "_" + sha1(description)[:8], shared across instances.
    assert objects[0]["asset_key"] == objects[1]["asset_key"] == objects[2]["asset_key"] == asset_key("chair", "oak chair")
    assert objects[0]["asset_key"] == "chair_" + hashlib.sha1(b"oak chair").hexdigest()[:8]
    assert objects[3]["asset_key"] != objects[0]["asset_key"]
    assert objects[2]["dimensions"]["length"] == 1.2  # per-instance dimensions remain on the scene object
    registry = [json.loads(line) for line in (handoff / "assets.jsonl").read_text().splitlines()]
    assert len(registry) == 2
    by_key = {entry["asset_key"]: entry for entry in registry}
    oak = by_key[objects[0]["asset_key"]]
    assert oak["n_instances"] == 3 and oak["dimensions"] == {"width": .7, "length": 1., "height": .9}
    assert oak["place"] == "floor" and oak["support_status"] == "inferred"  # first instance, z=0
    assert by_key[objects[3]["asset_key"]]["place"] == "unknown"
    assert {obj["id"] for obj in objects} == {obj["id"] for obj in prediction["objects"]}
    assert len(objects) == 4
    assert all(entry["asset_key_kind"] == "downstream_generation_key_only" for entry in registry)


def test_object_id_equal_to_room_proxy_name_remains_unique_and_selectable_in_glb(tmp_path):
    source = {**request(), "furniture_list": [{"id": "room_extent_proxy", "category": "table"}]}
    condition = request_to_condition(source)
    prediction = layout(condition, yaw=math.radians(97))
    output = tmp_path / "collision.glb"
    write_bbox_glb(output, layout_to_scene(condition, prediction))
    document, binary = read_glb(output)
    names = [node.get("name") for node in document["nodes"]]
    assert len(names) == len(set(names))
    actual = node_vertices(document, binary, "room_extent_proxy")
    obj = prediction["objects"][0]
    expected = [[x, z, -y] for x, y, z in corners(obj["bottom_center_m"], obj["target_size_local_m"], obj["yaw_rad"])]
    assert sorted(set(corner_set(actual))) == sorted(set(corner_set(expected)))


def test_direct_inventory_budget_rejects_overflow_before_expansion_and_never_truncates():
    source = {**request(), "furniture_list": [{"category": "chair", "count": 3}]}
    condition = request_to_condition(source, max_objects=3)
    assert len(condition["objects"]) == 3
    with pytest.raises(ValueError, match="max_objects"):
        request_to_condition(source, max_objects=2)
    with pytest.raises(ValueError, match="max_objects"):
        request_to_condition({**source, "furniture_list": [{"category": "chair", "count": 10**12}]})
    assert source["furniture_list"][0]["count"] == 3
    with pytest.raises(ValueError, match="nonempty"):
        request_to_condition({**source, "furniture_list": []})


@pytest.mark.parametrize("budget", [0, -1, True, 3., "3", None])
def test_object_budget_itself_must_be_a_positive_integer(budget):
    with pytest.raises(ValueError, match="max_objects"):
        request_to_condition(request(), max_objects=budget)
