"""Read-only AST-isolated V-DETR counterexamples; no native imports/builds.

Run from repository root: python3 outputs/fastfill_v2/audit-four-20261006/reference-vdetr/python_repros.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4] / "V-DETR"
OUT = Path(__file__).resolve().parent


def extract(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def error_call(fn):
    try:
        fn()
    except Exception as exc:
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": None}


def main():
    make_parser = extract("main.py", "make_args_parser", {"argparse": argparse})
    args = make_parser().parse_args(["--dataset_name", "scannet"])
    model_class = extract("models/model_vdetr.py", "ModelVDETR", {"nn": nn})
    random_fps_error = error_call(lambda: model_class(None, None, None, None, args=args))
    assert random_fps_error["type"] == "AttributeError"
    assert "random_fps" in random_fps_error["message"]

    # Supplying missing args.random_fps allows a later branch to be tested in isolation.
    # The upstream function is unchanged; the namespace only bypasses native import.
    run_encoder = extract("models/model_vdetr.py", "run_encoder", {
        "ME": SimpleNamespace(utils=SimpleNamespace(batch_sparse_collate=lambda pairs: pairs)),
        "torch": torch,
    }, class_name="ModelVDETR")
    no_color_error = error_call(lambda: run_encoder(
        SimpleNamespace(use_color=False, voxel_size=0.01), [torch.zeros(3, 3)]
    ))
    assert no_color_error["type"] == "UnboundLocalError"
    assert "xyz" in no_color_error["message"]

    share_class = extract("models/vdetr_transformer.py", "ShareSelfAttention", {"nn": nn})
    attention = share_class(dim=2, num_heads=1, dropout=0).double().eval()
    with torch.no_grad():
        attention.q.weight.zero_(); attention.q.bias.zero_()
        attention.k.weight.zero_(); attention.k.bias.zero_()
        attention.v.weight.copy_(torch.eye(2, dtype=torch.float64)); attention.v.bias.zero_()
        attention.proj.weight.copy_(torch.eye(2, dtype=torch.float64)); attention.proj.bias.zero_()
    # N=3, B=2. Each batch has constant channel values 0,10,20 vs 100,110,120.
    values = torch.tensor([[[0, 0], [100, 100]], [[10, 10], [110, 110]],
                           [[20, 20], [120, 120]]], dtype=torch.float64)
    actual_attention, _ = attention(torch.zeros_like(values), torch.zeros_like(values), value=values)
    expected_attention = values.mean(dim=0).unsqueeze(0).expand_as(values)
    assert not torch.allclose(actual_attention, expected_attention)

    all_gather = extract("utils/dist.py", "all_gather_dict", {
        "torch": torch, "is_distributed": lambda: False,
    })
    sample_clouds = [torch.ones(4, 3)]
    actual_gather = all_gather({"scan_idx": torch.tensor([7]), "point_clouds": sample_clouds})
    list_first_error = error_call(lambda: all_gather({"point_clouds": sample_clouds}))
    assert isinstance(actual_gather["point_clouds"], torch.Tensor)
    assert actual_gather["point_clouds"].tolist() == [7]
    assert list_first_error["type"] == "UnboundLocalError"

    # Actual AST-selected Dataset __getitem__ on a tiny local fixture. The native
    # modules are not imported; this branch fails before any native invocation.
    tree = ast.parse((ROOT / "datasets/scannet.py").read_text())
    method = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "ScannetDetectionDataset"
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "__getitem__")
    choices_line = next(n.lineno for n in ast.walk(method) if isinstance(n, ast.Name) and n.id == "choices" and isinstance(n.ctx, ast.Load))
    dataset = extract("datasets/scannet.py", "ScannetDetectionDataset", {
        "Dataset": torch.utils.data.Dataset, "np": np, "torch": torch, "os": os,
        "json": json, "MEAN_COLOR_RGB": np.array([109.8, 97.2, 83.8]), "IGNORE_LABEL": -100,
    })
    fixture = OUT / "tiny-superpoint-fixture"
    detection = fixture / "detection"
    scan = "scene0000_00"
    scan_dir = fixture / "scans" / scan
    detection.mkdir(parents=True, exist_ok=True); scan_dir.mkdir(parents=True, exist_ok=True)
    np.save(detection / (scan + "_vert.npy"), np.arange(24, dtype=np.float32).reshape(4, 6))
    np.save(detection / (scan + "_ins_label.npy"), np.ones(4, dtype=np.int64))
    np.save(detection / (scan + "_sem_label.npy"), np.ones(4, dtype=np.int64) * 3)
    np.save(detection / (scan + "_bbox.npy"), np.array([[0, 0, 1, 1, 1, 1, 3]], dtype=np.float32))
    (scan_dir / (scan + "_vh_clean_2.0.010000.segs.json")).write_text(json.dumps({"segIndices": [0, 0, 1, 1]}))
    example = object.__new__(dataset)
    example.scan_names = [scan]; example.data_path = str(detection) + "/"
    example.use_superpoint = True; example.use_normals = False; example.use_color = False
    example.use_height = False; example.augment = False; example.use_random_cuboid = True
    example.dataset_config = SimpleNamespace(max_num_obj=64)
    superpoint_error = error_call(lambda: example[0])
    assert superpoint_error["type"] == "UnboundLocalError"
    assert "choices" in superpoint_error["message"]

    forward = extract("models/model_vdetr.py", "forward", {"torch": torch}, class_name="ModelVDETR")
    cls_logits = torch.zeros(1, 19, 3); cls_logits[:, 18, :] = 10
    ce_self = SimpleNamespace(
        run_encoder=lambda clouds: (torch.zeros(1, 3, 3), torch.zeros(3, 1, 2), None),
        encoder_to_decoder_projection=lambda features: features,
        decoder=SimpleNamespace(pointcls_heads=lambda features: cls_logits),
        hard_anchor=False, dataset_config=SimpleNamespace(mean_size_arr=np.ones((18, 3))),
    )
    ce_anchor_error = error_call(lambda: forward(ce_self, {
        "point_clouds": [torch.zeros(3, 3)], "point_cloud_dims_min": torch.zeros(1, 3),
        "point_cloud_dims_max": torch.ones(1, 3),
    }))
    assert ce_anchor_error["type"] == "IndexError"

    # DIoU penalty, already confirmed by the earlier review; repeat exact AST slice.
    diou_tree = ast.parse((ROOT / "criterion.py").read_text())
    diou = next(n for n in diou_tree.body if isinstance(n, ast.FunctionDef) and n.name == "diff_diou_rotated_3d")
    r2_assignment = next(n for n in diou.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "r2")
    penalty_cases = []
    for name, second in [("z_only", [0, 0, 2, 1, 1, 1, 0]), ("width_only", [0, 0, 0, 3, 1, 1, 0])]:
        a = torch.tensor([0, 0, 0, 1, 1, 1, 0], dtype=torch.float64)
        b = torch.tensor(second, dtype=torch.float64)
        env = {"box1": a[[0, 1, 3, 4, 6]], "box2": b[[0, 1, 3, 4, 6]]}
        exec(compile(ast.Module(body=[r2_assignment], type_ignores=[]), str(ROOT / "criterion.py"), "exec"), env)
        expected = ((a[:3] - b[:3]) ** 2).sum().item()
        assert env["r2"].item() != expected
        penalty_cases.append({"case": name, "actual_r2": env["r2"].item(), "expected_xyz_r2": expected})

    tree = ast.parse((ROOT / "datasets/__init__.py").read_text())
    dataset_keys = [k.value for n in tree.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name) and t.id == "DATASET_FUNCTIONS"
                    for k in n.value.keys]
    assert dataset_keys == ["scannet"]

    chromatic = extract("datasets/scannet.py", "ChromaticAutoContrast", {"np": np})
    with np.errstate(divide="ignore", invalid="ignore"):
        contrast = chromatic(p=1, blend_factor=1)(np.ones((3, 3)))
    assert not np.isfinite(contrast).all()

    results = {
        "source_head": "9062d75fe2c91e5d4a771b5325483fc330a3e827",
        "scope": "Exact AST-selected definitions / expressions executed on CPU; no MMCV, MinkowskiEngine, PointNet2 native imports; no original training/eval/compile or assets",
        "torch_version": torch.__version__,
        "fresh_parser": {"use_color": args.use_color, "random_fps_declared": hasattr(args, "random_fps"),
                         "iou_type": args.iou_type, "share_selfattn": args.share_selfattn,
                         "repeat_num": args.repeat_num, "matching_angle_weights": [args.matcher_anglecls_cost, args.matcher_anglereg_cost]},
        "constructor_missing_random_fps": random_fps_error,
        "run_encoder_no_color": no_color_error,
        "share_selfattn_value_layout": {"actual_first_query_by_batch": actual_attention[0].tolist(),
                                       "expected_first_query_by_batch": expected_attention[0].tolist(),
                                       "max_absolute_error": (actual_attention - expected_attention).abs().max().item(),
                                       "scope": "Optional --share_selfattn; B>1. Default disabled and README batch/GPU1 does not expose cross-batch corruption"},
        "all_gather_dict_nontensor": {"actual_point_clouds_type": type(actual_gather["point_clouds"]).__name__,
                                     "actual_point_clouds": actual_gather["point_clouds"].tolist(),
                                     "list_first_error": list_first_error,
                                     "scope": "Default evaluate currently remove_empty_box=False so this corrupt field is unused by default AP path"},
        "superpoint_choices": {"status": "AST-selected Dataset __getitem__ executed with tiny fixture",
                               "error": superpoint_error, "first_read_line": choices_line,
                               "scope": "Validation augment=False leaves choices unassigned; separate engine evaluate also omits superpoint input"},
        "celoss_background_anchor": {"error": ce_anchor_error,
                                     "scope": "AST-selected model forward with deterministic upstream shaped point logits; CE has background class18 but mean-size table only18rows; focal default avoids this"},
        "diou_penalty_previously_confirmed": penalty_cases,
        "rotated_giou_flag": {"expression": "torch.any(targets['gt_box_angles'] > 0).item()",
                              "negative_only_example_flag": bool(torch.any(torch.tensor([-0.4, 0.0]) > 0).item()),
                              "scope": "Static and predicate execution; ScanNet gt_box_angles all0"},
        "sunrgbd": {"parser_accepts": make_parser().parse_args(["--dataset_name", "sunrgbd"]).dataset_name,
                    "implemented_dataset_keys": dataset_keys},
        "chromatic_constant_channel": {"all_finite": bool(np.isfinite(contrast).all()), "default_probability": args.color_contrastp},
        "missing_readme_build_files": [str(p.relative_to(ROOT)) for p in [ROOT / "scannet/scannet_utils.py", ROOT / "scannet/meta_data", ROOT / "utils/box_intersection.pyx"] if not p.exists()],
        "sources_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["main.py", "models/model_vdetr.py", "models/vdetr_transformer.py", "utils/dist.py", "criterion.py", "datasets/scannet.py", "datasets/__init__.py"]},
    }
    (OUT / "python_repros.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
