"""Isolated source-mechanism illustrations; NOT a native/CUDA execution test.

Run from any directory. This script uses only Python's standard library and
writes results next to itself. It does not import/build/install MinkowskiEngine.
Source hashes bind the examples to the files actually reviewed.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3] / "MinkowskiEngine"


def source_hash(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def label_conflict():
    coords = ["A", "A", "B", "B"]
    labels = [1, 1, 2, 3]
    invalid = -1
    voxels = {}
    inverse = [0] * len(coords)
    colabels = []
    writes = []
    for row, (coord, label) in enumerate(zip(coords, labels)):
        if coord not in voxels:
            unique_index = len(colabels)
            voxels[coord] = [unique_index, label]
            colabels.append(label)
            inverse[row] = unique_index
        else:
            value = voxels[coord]
            if value[1] != label and value[1] != invalid:
                value[1] = invalid
                wrong_index = inverse[value[0]]
                colabels[wrong_index] = invalid
                writes.append({"voxel": coord, "unique_index": value[0],
                               "written_unique_index": wrong_index})
            inverse[row] = value[0]
    expected = [1, -1]
    assert colabels == [-1, 2] and colabels != expected
    return {"coords": coords, "labels": labels, "inverse": inverse,
            "actual_mechanism": colabels, "expected": expected, "writes": writes}


def packed_counts(rows):
    counts = Counter(rows)
    keys = sorted(counts)
    nnz = len(rows)
    packed = [counts[key] for key in keys] + [None] * (nnz - len(keys))
    probes = [{"row_id": row, "count_index": row,
               "state": "out_of_allocation" if row >= nnz else
                        "unwritten" if packed[row] is None else "written",
               "value": packed[row] if row < nnz else None}
              for row in sorted(rows)]
    return {"rows": rows, "allocation_elements": nnz, "packed_keys": keys,
            "packed_counts": packed, "source_mechanism_probes": probes}


def copy_gemm_shared_map():
    channels, map_count, block_threads = 3, 342, 512
    nthreads = channels * map_count
    allocated = (block_threads + channels - 1) // channels
    block, tx = 1, 511
    global_i = block * block_threads + tx
    remainder = (block * block_threads) % channels
    index = (tx + remainder) // channels
    assert global_i < nthreads and index >= allocated
    return {"channels": channels, "map_count": map_count,
            "nthreads": nthreads, "shared_allocation_elements": allocated,
            "block": block, "thread": tx, "active": global_i < nthreads,
            "shared_index": index, "last_legal_index": allocated - 1}


def direct_max_cpu():
    features = [-3.0, -2.0]
    out, mask = 0.0, 0
    for index, feature in enumerate(features):
        if out < feature:
            out, mask = feature, index
    assert out == 0.0 and max(features) == -2.0
    return {"features": features, "actual_mechanism": out, "mask": mask,
            "expected": max(features), "expected_mask": 1}


def direct_max_gpu_gap():
    output_ids = [0, 2]
    actual_out_nrows = 3
    kernel_out_nrows = len(set(output_ids))
    written_ids = [row for row in output_ids if row < kernel_out_nrows]
    assert written_ids == [0]
    return {"output_ids": output_ids, "allocated_out_nrows": actual_out_nrows,
            "kernel_out_nrows": kernel_out_nrows, "written_ids": written_ids,
            "incorrectly_skipped_ids": [2]}


def global_pool_gap():
    batch_ids = [0, 2]
    maps_by_original_id = [[0, 1], [], [2, 3]]
    out_nrows = len(batch_ids)
    return {"batch_indices": batch_ids, "vec_maps": maps_by_original_id,
            "out_nrows": out_nrows,
            "cpu_loop": [{"b": b, "input_map": maps_by_original_id[b],
                          "output_index": batch_ids[b],
                          "output_index_in_bounds": batch_ids[b] < out_nrows}
                         for b in range(out_nrows)],
            "gpu_loop_maps": maps_by_original_id[:out_nrows],
            "expected_compact_maps": [maps_by_original_id[bid] for bid in batch_ids]}


def float_stride_example():
    value = 16777219
    f32 = struct.unpack("f", struct.pack("f", value))[0]
    stride = 2
    actual = int(f32 // stride) * stride
    expected = (value // stride) * stride
    assert actual != expected
    return {"int32_coordinate": value, "float32_coordinate": f32,
            "stride": stride, "actual_float_mechanism": actual,
            "expected_integer_floor": expected}


def main():
    sources = ["src/quantization.cpp", "src/spmm.cu", "src/math_functions.cuh",
               "src/direct_max_pool.cpp", "src/pooling_max_kernel.hpp",
               "src/pooling_max_kernel.cu", "src/global_pooling_cpu.cpp",
               "src/global_pooling_gpu.cu", "src/storage.cuh",
               "src/coordinate_map.hpp", "src/coordinate_map_gpu.cu"]
    result = {
        "execution_kind": "Python standard-library simulation of reviewed source arithmetic",
        "not_native_execution": True,
        "cuda_executed": False,
        "native_binary_imported": False,
        "source_sha256": {p: source_hash(p) for p in sources},
        "label_conflict": label_conflict(),
        "spmm_average_single_gap": packed_counts([5]),
        "spmm_average_packed_gap": packed_counts([0, 0, 2]),
        "copy_gemm_shared_map": copy_gemm_shared_map(),
        "direct_max_cpu": direct_max_cpu(),
        "direct_max_gpu_gap": direct_max_gpu_gap(),
        "global_pool_gap": global_pool_gap(),
        "gpu_storage_grow": {"old_allocated_elements": 4, "new_elements": 8,
                             "source_copy_elements": 8, "overread_elements": 4},
        "large_coordinate_float_stride": float_stride_example(),
        "noncontiguous_quantize_pointer": {
            "backing_storage": [0, 1, 0, 0, 2, 0],
            "shape": [2, 2], "stride": [3, 2],
            "logical_rows": [[0, 0], [0, 0]], "expected_unique": 1,
            "dense_pointer_rows": [[0, 1], [0, 0]], "dense_pointer_unique": 2,
            "note": "Address-mechanism illustration, no tensor/native invocation",
        },
    }
    (OUT / "mechanism_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "mechanism_results.json"),
                      "assertions_passed": True,
                      "native_execution": False, "cuda_execution": False}))


if __name__ == "__main__":
    main()
