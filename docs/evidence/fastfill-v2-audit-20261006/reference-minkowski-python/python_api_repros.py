#!/usr/bin/env python3
"""Read-only AST execution of exact upstream Python nodes; no ME import/build.

Coordinate/native containers are mocked. Torch operations are real CPU operations.
This verifies Python API counterexamples, not Minkowski native/CUDA integration.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import inspect
import json
import math
import platform
import sys
import types
import warnings
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package = args.repo / 'MinkowskiEngine'
    sources = {}
    results = {}

    def load_nodes(path, names, env):
        source_path = args.repo / path
        source = source_path.read_text()
        tree = ast.parse(source, filename=str(source_path))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert set(n.name for n in nodes) == set(names)
        sources[path] = {'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'nodes': sorted(set(names) | set(sources.get(path, {}).get('nodes', [])))}
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, str(source_path), 'exec'), env)
        return env

    def load_method(path, cls, method, env):
        source_path = args.repo / path
        source = source_path.read_text()
        tree = ast.parse(source, filename=str(source_path))
        class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        node = next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == method)
        sources[path] = {'sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'nodes': sorted(set(sources.get(path, {}).get('nodes', [])) | {f'{cls}.{method}'})}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), env)
        return env[method]

    def capture(name, fn):
        try:
            results[name] = {'returned': fn()}
        except Exception as exc:
            results[name] = {'exception_type': type(exc).__name__, 'message': str(exc)}

    env = {'torch': torch, 'nn': torch.nn, 'Module': torch.nn.Module, 'MinkowskiModuleBase': torch.nn.Module}
    load_nodes('MinkowskiEngine/MinkowskiNormalization.py', ['MinkowskiBatchNorm', 'MinkowskiSyncBatchNorm'], env)
    original = env['MinkowskiBatchNorm'](3)
    converted = env['MinkowskiSyncBatchNorm'].convert_sync_batchnorm(original)
    results['sync_batchnorm_conversion'] = {
        'outer_type': type(converted).__name__,
        'inner_type': type(converted.bn).__name__,
        'inner_is_sync_batchnorm': isinstance(converted.bn, torch.nn.SyncBatchNorm),
        'same_original_inner_object': converted.bn is original.bn,
    }
    assert not results['sync_batchnorm_conversion']['inner_is_sync_batchnorm']
    assert results['sync_batchnorm_conversion']['same_original_inner_object']

    class Field:
        def __init__(self, features, *, coordinate_field_map_key='field_key', coordinate_manager='manager'):
            self._F = features
            self.F = features
            self.coordinate_field_map_key = coordinate_field_map_key
            self._manager = coordinate_manager

    env = {'torch': torch, 'COORDINATE_MANAGER_DIFFERENT_ERROR': 'manager', 'COORDINATE_KEY_DIFFERENT_ERROR': 'key'}
    Field._is_same_key = load_method('MinkowskiEngine/MinkowskiTensorField.py', 'TensorField', '_is_same_key', env)
    Field._binary_functor = load_method('MinkowskiEngine/MinkowskiTensorField.py', 'TensorField', '_binary_functor', env)
    field = Field(torch.tensor([[1.0], [2.0]]))
    capture('tensor_field_plus_field', lambda: field._binary_functor(field, torch.add))
    capture('tensor_field_plus_torch', lambda: field._binary_functor(torch.ones(2, 1), torch.add))

    env = {'torch': torch, 'TensorField': Field, 'Union': Union, 'MinkowskiModuleBase': torch.nn.Module,
           'to_sparse': lambda x: {'route': 'to_sparse', 'coordinates_supplied': False},
           'to_sparse_all': lambda x, c: {'route': 'to_sparse_all', 'coordinates_supplied': c is not None}}
    load_nodes('MinkowskiEngine/MinkowskiOps.py', ['MinkowskiToSparseTensor'], env)
    layer = env['MinkowskiToSparseTensor']
    capture('dense_to_sparse_default', lambda: layer()(torch.zeros(1, 1, 2)))
    capture('dense_to_sparse_custom_coords', lambda: layer(coordinates=torch.tensor([[0, 7], [0, 8]]))(torch.ones(1, 1, 2)))
    assert results['dense_to_sparse_default']['returned']['route'] == 'to_sparse_all'
    assert not results['dense_to_sparse_custom_coords']['returned']['coordinates_supplied']

    class RegionType(Enum):
        HYPER_CUBE = 0
        HYPER_CROSS = 1
        CUSTOM = 2
        HYBRID = 3  # supplied only to bypass enum presence; native enum not validated here

    env = {'torch': torch, 'np': np, 'Union': Union, 'Sequence': Sequence, 'reduce': reduce, 'math': math, 'RegionType': RegionType}
    load_nodes('MinkowskiEngine/MinkowskiCommon.py', ['convert_to_int_list'], env)
    load_nodes('MinkowskiEngine/MinkowskiKernelGenerator.py', ['get_kernel_volume', 'convert_region_type', 'KernelGenerator'], env)
    kernel = env['KernelGenerator']
    capture('custom_kernel_constructor', lambda: kernel(kernel_size=3, region_type=RegionType.CUSTOM, region_offsets=torch.IntTensor([[0, 0], [1, 0]]), dimension=2))
    capture('cross_kernel_get', lambda: kernel(kernel_size=3, region_type=RegionType.HYPER_CROSS, dimension=2).get_kernel([1, 1], False))
    capture('transpose_kernel_get', lambda: kernel(kernel_size=3, dimension=2).get_kernel([1, 1], True))
    capture('custom_region_convert', lambda: env['convert_region_type'](RegionType.CUSTOM, [1, 1], [3, 3], [1, 1], [1, 1], torch.IntTensor([[0, 0]]), None, 2))

    class Sparse:
        def __init__(self):
            self._D = 1
            self.tensor_stride = [1]
            self.C = torch.IntTensor([[0, 5], [0, 6]])
            self.F = torch.tensor([[1.0], [2.0]])
            self.coordinate_map_key = 'sparse_key'
            self.coordinate_manager = 'manager'
            self.inverse_mapping = torch.tensor([0, 1])
            self.quantization_mode = 'average'

    env = {'torch': torch}
    Sparse.sparse = load_method('MinkowskiEngine/MinkowskiSparseTensor.py', 'SparseTensor', 'sparse', env)
    x = Sparse()
    before = x.C.clone()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        converted, minimum, stride = x.sparse()
    results['sparse_conversion_mutates_coordinates'] = {'before': before.tolist(), 'after': x.C.tolist(), 'returned_minimum': minimum.tolist(), 'changed': not torch.equal(before, x.C)}
    assert results['sparse_conversion_mutates_coordinates']['changed']
    capture('sparse_max_without_min', lambda: Sparse().sparse(max_coords=torch.IntTensor([9])))

    class Quantization:
        RANDOM_SUBSAMPLE = 'random'
        UNWEIGHTED_AVERAGE = 'average'

    fake_field_module = types.ModuleType('MinkowskiTensorField')
    fake_field_module.TensorField = Field
    sys.modules['MinkowskiTensorField'] = fake_field_module
    env = {'torch': torch, 'SparseTensorQuantizationMode': Quantization, 'SparseTensor': Sparse}
    Sparse.cat_slice = load_method('MinkowskiEngine/MinkowskiSparseTensor.py', 'SparseTensor', 'cat_slice', env)
    capture('cat_slice_sparse_input', lambda: Sparse().cat_slice(Sparse()))
    del sys.modules['MinkowskiTensorField']

    env = {'torch': torch, 'collections': collections, 'np': np}
    load_nodes('MinkowskiEngine/utils/collation.py', ['sparse_collate'], env)
    coords = [torch.IntTensor([[10]]), torch.IntTensor([[20], [21]])]
    feats = [torch.tensor([[100.0], [101.0]]), torch.tensor([[200.0]])]
    bc, bf = env['sparse_collate'](coords, feats)
    results['collate_per_sample_mismatch'] = {'input_coordinate_lengths': [1, 2], 'input_feature_lengths': [2, 1], 'accepted': True, 'coordinate_feature_pairs': list(zip(bc.tolist(), bf.tolist()))}
    assert bc[1, 0] == 1 and bf[1, 0] == 101

    env = {'torch': torch, 'Callable': Callable, 'Union': Union, 'Optional': Optional,
           '_TensorOrTensors': torch.Tensor, '_gradcheck': torch.autograd.gradcheck}
    load_nodes('MinkowskiEngine/utils/gradcheck.py', ['gradcheck'], env)
    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x*x
        @staticmethod
        def backward(ctx, grad):
            return 2*ctx.saved_tensors[0]*grad
    capture('modern_torch_gradcheck', lambda: env['gradcheck'](Square, (torch.tensor([1.0], dtype=torch.float64, requires_grad=True),)))
    results['modern_torch_gradcheck_signature'] = str(inspect.signature(torch.autograd.gradcheck))

    env = {'torch': torch, 'SparseTensor': Sparse}
    stable_forward = load_method('MinkowskiEngine/MinkowskiNormalization.py', 'MinkowskiStableInstanceNorm', 'forward', env)
    class Norm:
        mean_in = staticmethod(lambda x: x)
    capture('stable_instance_norm_current_sparse', lambda: stable_forward(Norm(), Sparse()))

    env = {'nn': torch.nn, 'ME': types.SimpleNamespace()}
    load_nodes('MinkowskiEngine/modules/resnet_block.py', ['BasicBlock', 'Bottleneck'], env)
    load_nodes('MinkowskiEngine/modules/senet_block.py', ['SEBasicBlock', 'SEBottleneck'], env)
    capture('se_basic_block_constructor', lambda: env['SEBasicBlock'](16, 16, D=3))
    capture('se_bottleneck_constructor', lambda: env['SEBottleneck'](16, 16, D=3))

    loader = torch.utils.data.DataLoader(torch.tensor([1]), num_workers=0)
    iterator = iter(loader)
    capture('example_training_iterator_next', lambda: iterator.next())
    env = {'np': np}
    load_nodes('examples/common.py', ['Timer'], env)
    capture('example_timer_numpy', lambda: env['Timer']().min_time)

    expected = {
        'tensor_field_plus_field': 'AttributeError', 'tensor_field_plus_torch': 'AttributeError',
        'custom_kernel_constructor': 'RuntimeError', 'cross_kernel_get': 'TypeError',
        'transpose_kernel_get': 'AttributeError', 'custom_region_convert': 'AssertionError',
        'sparse_max_without_min': 'AttributeError', 'cat_slice_sparse_input': 'TypeError',
        'modern_torch_gradcheck': 'TypeError', 'stable_instance_norm_current_sparse': 'AttributeError',
        'se_basic_block_constructor': 'TypeError', 'se_bottleneck_constructor': 'TypeError',
        'example_training_iterator_next': 'AttributeError',
    }
    for name, kind in expected.items():
        assert results[name].get('exception_type') == kind, (name, results[name])

    report = {'scope': 'exact upstream AST Python nodes, real CPU torch, mocked coordinate/native containers; no MinkowskiEngine native import/build/GPU/train',
              'repo': str(args.repo.resolve()), 'python': sys.version, 'platform': platform.platform(),
              'torch': torch.__version__, 'numpy': np.__version__, 'sources': sources, 'results': results,
              'expected_exception_cases_passed': len(expected), 'native_integration_executed': False}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'expected_exception_cases_passed': len(expected), 'additional_deterministic_wrong_results': 5, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
