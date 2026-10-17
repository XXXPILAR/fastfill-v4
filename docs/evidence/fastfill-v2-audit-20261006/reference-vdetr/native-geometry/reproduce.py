"""Read-only CPU checks of pinned V-DETR geometry and Python PointNet helpers.

No compilation, CUDA execution, dependency installation, or upstream writes.
Package initializers are bypassed only to avoid the optional plyfile import.
PointNet's unavailable native extension is a sentinel for Python-only checks;
no extension function is invoked or emulated in those checks.
"""

from __future__ import annotations

import importlib
import json
import math
from itertools import product
from pathlib import Path
import sys
import types
import warnings
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon
import torch


ROOT = Path(__file__).resolve().parents[5] / "V-DETR"
RESULT_PATH = Path(__file__).with_name("reproduction-results.json")


def exception_result(fn):
    try:
        value = fn()
        return {"completed": True, "value": value}
    except Exception as exc:
        return {"completed": False, "exception": type(exc).__name__, "message": str(exc)}


def setup_package():
    package = types.ModuleType("utils")
    package.__path__ = [str(ROOT / "utils")]
    sys.modules["utils"] = package
    return importlib.import_module("utils.box_util")


def make_box(module, size, angle, center, dtype=torch.float32):
    np_box = module.get_3d_box(np.asarray(size), angle, np.asarray(center))
    return torch.tensor(np_box, dtype=dtype)[None, None], np_box


def cpu_geometry(module):
    counts = torch.tensor([1])
    rows = []
    for angle in (0.0, 0.3, math.pi / 4, math.pi / 2, math.pi, -math.pi / 2):
        box, corners = make_box(module, [2.0, 1.0, 1.0], angle, [0.0, 0.0, 0.0])
        polygon = Polygon(corners[[3, 2, 1, 0]][:, [0, 2]])
        reference = polygon.intersection(polygon).area
        row = {"angle_rad": angle, "reference_self_intersection_volume": reference}
        for dtype, label in ((torch.float32, "float32"), (torch.float64, "float64")):
            box = torch.tensor(corners, dtype=dtype)[None, None]
            value = module.generalized_box3d_iou(
                box, box, counts, rotated_boxes=True,
                return_inter_vols_only=True, needs_grad=True,
            )
            row[label + "_self_intersection_volume"] = value.item()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            row["numpy_self_iou"] = exception_result(
                lambda: float(module.box3d_iou(corners, corners)[0])
            )
            row["numpy_warnings"] = [str(w.message) for w in observed]
        rows.append(row)

    # Axis-aligned control: exact scalar formula with different centers and sizes.
    rng = np.random.default_rng(20261006)
    controls = []
    grad_finite = []
    for _ in range(200):
        size1, size2 = rng.uniform(0.1, 4.0, (2, 3))
        center1, center2 = rng.uniform(-2.0, 2.0, (2, 3))
        box1, _ = make_box(module, size1, 0.0, center1)
        box2, _ = make_box(module, size2, 0.0, center2)
        box1.requires_grad_()
        actual = module.generalized_box3d_iou(
            box1, box2, counts, rotated_boxes=False, needs_grad=True,
        )
        # get_3d_box size axes are (local x, local z, height y).
        camera_size1, camera_size2 = size1[[0, 2, 1]], size2[[0, 2, 1]]
        min1, max1 = center1 - camera_size1 / 2, center1 + camera_size1 / 2
        min2, max2 = center2 - camera_size2 / 2, center2 + camera_size2 / 2
        inter = np.maximum(0.0, np.minimum(max1, max2) - np.maximum(min1, min2)).prod()
        union = size1.prod() + size2.prod() - inter
        enclosure = (np.maximum(max1, max2) - np.minimum(min1, min2)).prod()
        expected = inter / union - (enclosure - union) / enclosure
        controls.append(abs(actual.item() - expected))
        actual.sum().backward()
        grad_finite.append(bool(torch.isfinite(box1.grad).all()))

    zero = torch.zeros((1, 1, 8, 3))
    zero_giou = module.generalized_box3d_iou(
        zero, zero, counts, rotated_boxes=False, needs_grad=True,
    )
    optional_counts = exception_result(lambda: module.generalized_box3d_iou(
        zero + 1.0, zero + 1.0, None, rotated_boxes=True, needs_grad=True,
    ).tolist())

    # The criterion's actual predicate tests positive angles, not nonzero angles.
    all_negative_angles = torch.tensor([[-0.3]])
    criterion_flag = torch.any(all_negative_angles > 0).item()
    neg_box, _ = make_box(module, [2.0, 1.0, 1.0], -0.3, [0.0, 0.0, 0.0])
    chosen_inter = module.generalized_box3d_iou(
        neg_box, neg_box, counts, rotated_boxes=criterion_flag,
        return_inter_vols_only=True, needs_grad=True,
    ).item()
    chosen_giou = module.generalized_box3d_iou(
        neg_box, neg_box, counts, rotated_boxes=criterion_flag, needs_grad=True,
    ).item()
    thin_cases = []
    for thin_axis in (0.001, 0.0001, 0.00001):
        size = [2.0, thin_axis, 1.0]
        box, _ = make_box(module, size, 0.0, [0.0, 0.0, 0.0])
        production_volume = module.box3d_vol_tensor(box).item()
        production_inter = module.generalized_box3d_iou(
            box, box, counts, rotated_boxes=False,
            return_inter_vols_only=True, needs_grad=True,
        ).item()
        production_enclosure = module.enclosing_box3d_vol(box, box).item()
        production_union = 2 * production_volume - production_inter
        thin_cases.append({
            "size_m": size, "reference_self_volume": float(np.prod(size)),
            "camera_xyz_min": box.amin(dim=2).tolist(),
            "camera_xyz_max": box.amax(dim=2).tolist(),
            "production_box3d_vol_tensor": production_volume,
            "production_intersection": production_inter,
            "production_enclosure": production_enclosure,
            "production_union": production_union,
            "computed_iou": production_inter / production_union,
            "computed_giou_second_term": -(1 - production_union / production_enclosure),
            "production_self_giou": module.generalized_box3d_iou(
                box, box, counts, rotated_boxes=False, needs_grad=True,
            ).item(),
            "expected_self_giou": 1.0,
        })
    return {
        "rotated_self_cases": rows,
        "axis_aligned_controls": {
            "pairs": len(controls), "max_giou_absolute_error": max(controls),
            "all_gradients_finite": all(grad_finite),
            "sizes_each_axis_m": [0.1, 4.0], "seed": 20261006,
        },
        "zero_volume": {"all_finite": bool(torch.isfinite(zero_giou).all()),
                        "value_repr": str(zero_giou.tolist())},
        "rotated_nums_k2_none": optional_counts,
        "criterion_negative_only": {
            "angles": all_negative_angles.tolist(), "predicate_result": criterion_flag,
            "expected_self_intersection_volume": 2.0,
            "chosen_branch_intersection_volume": chosen_inter,
            "chosen_branch_giou": chosen_giou,
        },
        "sub_millimeter_axis_cases": thin_cases,
    }


def python_pointnet_checks():
    native = types.ModuleType("pointnet2._ext")
    pointnet_package = types.ModuleType("pointnet2")
    pointnet_package.__path__ = []
    pointnet_package._ext = native
    sys.modules["pointnet2"] = pointnet_package
    sys.modules["pointnet2._ext"] = native
    sys.path.insert(0, str(ROOT / "third_party" / "pointnet2"))
    pu = importlib.import_module("pointnet2_utils")
    pm = importlib.import_module("pointnet2_modules")
    checks = {}
    for flag in (False, True):
        checks["GroupAll_ret_grouped_xyz_" + str(flag)] = exception_result(
            lambda: pu.GroupAll(ret_grouped_xyz=flag)(torch.ones((1, 2, 3)), None)
        )
    checks["RandomDropout_forward"] = exception_result(
        lambda: pu.RandomDropout()(torch.ones((1, 4)))
    )
    specification = [[0, 8]]
    _ = pm.PointnetSAModuleMSG(npoint=1, radii=[1.0], nsamples=[2], mlps=specification)
    after_first = [list(s) for s in specification]
    _ = pm.PointnetSAModuleMSG(npoint=1, radii=[1.0], nsamples=[2], mlps=specification)
    checks["caller_mlp_spec_mutation"] = {
        "original": [[0, 8]], "after_first": after_first,
        "after_second": specification,
    }
    checks["extension_execution"] = "none; import sentinel only"
    return checks


def serial_fps_policy(points, count):
    """Scalar illustration of the source's origin-exclusion policy, not CUDA proof."""
    distances = np.full(len(points), 1e10)
    old, result = 0, [0]
    for _ in range(1, count):
        best, best_idx = -1.0, 0
        for idx, point in enumerate(points):
            if np.dot(point, point) <= 1e-3:
                continue
            candidate = min(np.dot(point - points[old], point - points[old]), distances[idx])
            distances[idx] = candidate
            if candidate > best:
                best, best_idx = candidate, idx
        old = best_idx
        result.append(old)
    return result


def cuboid_negative_coordinates():
    module = importlib.import_module("utils.random_cuboid")
    cloud = np.array([[-10.0, -10.0, -10.0], [-9.9, -9.9, -9.9], [-9.0, -9.0, -9.0]])
    boxes = np.array([[-9.0, -9.0, -9.0, 1.0, 1.0, 1.0]])
    with patch.object(np.random, "rand", return_value=np.zeros(3)), patch.object(
        np.random, "choice", return_value=0,
    ):
        cropped_cloud, returned_boxes, _ = module.RandomCuboid(min_points=1)(cloud, boxes)
    center_inside = np.all(
        (returned_boxes[:, :3] >= cropped_cloud[:, :3].min(axis=0))
        & (returned_boxes[:, :3] <= cropped_cloud[:, :3].max(axis=0)), axis=1,
    )
    return {
        "randomness": "controlled crop size 0.5 and first point center; original production function",
        "target_box_sum": float(boxes.sum()),
        "cropped_cloud": cropped_cloud.tolist(),
        "returned_boxes": returned_boxes.tolist(),
        "returned_box_center_inside_cropped_cloud": center_inside.tolist(),
    }


def coordinate_convention(module):
    size = np.array([4.0, 2.0, 1.0])
    bottom_center = np.array([1.0, 2.0, 3.0])
    center = bottom_center + [0.0, 0.0, size[2] / 2]
    camera_center = module.flip_axis_to_camera_np(center)
    theta = math.pi / 2
    camera = module.get_3d_box(size, theta, camera_center)
    depth = camera[:, [0, 2, 1]].copy()
    depth[:, 2] *= -1
    positive_local_x_axis = (depth[0] - depth[3]) / size[0]

    yaw = 0.7
    local_corners = np.asarray(list(product((-1.0, 1.0), repeat=3))) * size / 2
    c, s = math.cos(yaw), math.sin(yaw)
    rotation_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    canonical = local_corners @ rotation_z.T + center
    errors = {}
    for label, angle in (("box_angle_equal_yaw", yaw), ("box_angle_negative_yaw", -yaw)):
        camera = module.get_3d_box(size, angle, camera_center)
        converted = camera[:, [0, 2, 1]].copy()
        converted[:, 2] *= -1
        pair_distances = np.linalg.norm(canonical[:, None] - converted[None], axis=-1)
        errors[label] = float(pair_distances.min(axis=1).max())
    return {
        "size_m": size.tolist(), "canonical_bottom_center_m": bottom_center.tolist(),
        "geometric_center_m": center.tolist(), "camera_center_m": camera_center.tolist(),
        "camera_box_angle_positive_pi_over_two": theta,
        "depth_world_direction_of_positive_local_x": positive_local_x_axis.tolist(),
        "canonical_positive_yaw_point_seven": yaw,
        "corner_set_max_nearest_point_error_m": errors,
        "note": "At ±π/2 cuboid corner sets alone have π symmetry; local +X direction disambiguates sign. Non-axis yaw 0.7 corner sets test the actual sign conversion.",
    }


def main():
    module = setup_package()
    points = np.array([[0.01, 0.0, 0.0], [0.015, 0.0, 0.0], [0.025, 0.0, 0.0]])
    result = {
        "upstream_commit": "9062d75fe2c91e5d4a771b5325483fc330a3e827",
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "scope": "CPU only; no upstream changes; no CUDA build or execution",
        "initialization_bypass": "utils.__init__ skipped to avoid local missing optional plyfile; original box_util and misc functions imported unchanged",
        "geometry": cpu_geometry(module),
        "coordinate_convention": coordinate_convention(module),
        "box_ops3d_import": exception_result(lambda: importlib.import_module("utils.box_ops3d").__name__),
        "python_pointnet": python_pointnet_checks(),
        "ancillary_random_cuboid_negative_coordinate_case": cuboid_negative_coordinates(),
        "native_fps_origin_policy_illustration": {
            "cuda_execution": False,
            "points": points.tolist(), "indices_original": serial_fps_policy(points, 2),
            "indices_translated_one_meter": serial_fps_policy(points + [1, 0, 0], 2),
            "line_evidence": "sampling_gpu.cu:103-104",
        },
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
