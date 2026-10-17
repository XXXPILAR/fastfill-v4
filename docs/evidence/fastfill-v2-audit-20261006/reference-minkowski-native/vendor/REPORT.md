# MinkowskiEngine vendor and small-vector static review

Audit date: 2026-10-06. Scope: all native `.h/.hpp/.cuh/.cpp/.cu` files under `MinkowskiEngine/src/3rdparty/`, plus `src/primitives/small_vector.hpp`. All **15 files / 7,754 lines** were read in full. This is a reference-source audit, not a verification that FastFill v2 uses MinkowskiEngine or that these libraries are part of the actual Qwen pilot.

No upstream source, training package, data, configuration or checkpoint was modified. No compiler, dependency installation, C++/CUDA runtime, GPU, training or Host operation was run for this subtask. The only writes are this report and its inventory/hash evidence. Confirmed below means a source-level control-flow/lifetime defect with a concrete trigger; it does not mean an executed reproducer or confirmed default production failure.

The range receipts and SHA256 values are in [full-read-inventory.json](full-read-inventory.json); [SHA256SUMS](SHA256SUMS) binds all reviewed files. Every final read block was complete. One aggregate output exceeded its combined budget; the affected `device_atomics.cuh` second half and the complete `wrapper_types.hpp` were subsequently read again individually. The machine inventory checks contiguous coverage from line 1 through EOF for each file.

## Conclusion and production reachability

This source subset cannot be described as defect-free: there are seven grouped, confirmed interface/boundary defects below. **No CRITICAL/HIGH defect in the default MinkowskiEngine production path was established by this sub-review.** Priority denotes the conditional interface problem, not an assertion that the actual FastFill training currently exercises it.

The coordinator's full native review supplied the following call-site evidence, which limits the practical conclusions:

- `CoordinateMapGPU` overrides the vendor global allocator template with `TemplatedAllocator<pair>` backed by `src/allocators.cuh`. V1's unthrown CUDA error is not proved reachable through the current coordinate-map allocator.
- Coordinate-map hashtable occupancy defaults to 50 (`coordinate_map_gpu.cuh:91`); reserve uses `compute_hash_table_size` (`:140`). Manager algorithm configurations use 25/50/25 (`coordinate_map_manager.hpp:142,147,152`), preserving spare buckets under normal input sizes. V4 requires a fully occupied table or over-capacity insertion, so it is not established on these normal paths.
- Native consumers mainly compare/dereference the cycle iterators returned by map `find`; no postfix increment or const increment use was found. V2 is an interface defect in those unused operations.
- No non-vendor native call to the vendor `genericAtomicOperation`, floating `atomicMin`/`atomicMax` was found. V3 is not established on the coordinate-map's native integer CAS path.
- Current NVTX integration is `CUDF_FUNC_RANGE` → `domain_thread_range`, not `domain_process_range`. V7 is an unused profiling API defect.
- No include or use of `small_vector.hpp` was found outside its own file in `src/`/`pybind/`. In particular, the self-element `push_back` in `CoordinateMap::expand_tensor_stride` operates on `default_types::stride_type = std::vector<uint_type>` (`src/types.hpp:47`), not `small_vector`. V5–V6 therefore do not establish a coordinate-stride defect.

These reachability conclusions are static and scoped to the inspected native checkout; they are not GPU execution or build validation.

## Confirmed interface defects

### V1 · P2 · CUDA allocator error is constructed but never thrown

`src/3rdparty/hash/hash_allocator.cuh:35–44,77–87`: both allocator `allocate` implementations call `cudaMalloc`; the failure branch clears the CUDA error and evaluates `std::runtime_error(...)` without `throw`, then returns `d_tmp`. The caller receives a return value despite allocation failure, and the original error is discarded. The report does not assume CUDA leaves an arbitrary specific pointer value; the confirmed problem is that this branch does not fail as intended and the pointer is unusable for the requested allocation.

Trigger: any failed nonzero allocation through either vendor allocator, such as GPU OOM. Correction: propagate the CUDA error with a thrown exception or checked error-return protocol; initialize the pointer and check overflow in the requested byte count if these standalone allocator APIs are retained. Current coordinate maps use an overriding allocator; no default-production occurrence is claimed.

### V2 · P2 · Cycle iterator operations have invalid lifetime and const behavior

`src/3rdparty/hash/helper_functions.cuh:194–211`: postfix increment returns a reference to local `old`, which is destroyed on function exit. A consumer copying/dereferencing the returned old iterator uses a dangling reference. Return the old iterator by value.

Related concrete defects in the same interface: the `const` increment overloads (`:185–191,204–211`) assign/increment the non-mutable `m_current` member, making those template operations invalid when instantiated on a const iterator. The const `operator->` (`:223`) calls `m_current.operator->()`, even though the map instantiates this adapter with a raw `value_type*`; a raw pointer has no member `operator->`. Consistent iterator type design should distinguish a mutable iterator over const elements from a const iterator object and should return the raw pointer directly in the pointer specialization. These unused operations were not compiled here.

### V3 · P2 · Floating CAS retry loop can never finish for NaN

`src/3rdparty/cudf/detail/utilities/device_atomics.cuh:112–158`: generic 4- and 8-byte CAS loops test retry with `assumed != old_value` in the original value type. For a floating target initialized to NaN and an operator that preserves NaN, even successful CAS returns the same NaN bits but `NaN != NaN` stays true, so the loop does not terminate. Floating `DeviceMin`/`DeviceMax` route through these generic loops; the native `atomicAdd` specializations do not all use this generic path.

Correction: compare CAS return bits with the assumed integer bits when deciding success, as required for floating atomics. A CUDA regression should cover NaN, signed zero, and normal concurrent updates before using this helper. No real GPU hang was induced, and current coordinate-map integer CAS usage does not demonstrate this trigger.

### V4 · P2 · Full hashtable probes are unbounded

`src/3rdparty/concurrent_unordered_map.cuh:341–354,371–419,438–463`: insertion probes until success/duplicate, and each `find` probes until a matching key or unused sentinel. Once all capacity buckets contain distinct keys, `find` for an absent key loops forever instead of returning `end`; inserting another distinct key likewise cannot stop. The implementation does not track a completed cycle or produce a table-full result.

Correction: bound probes to capacity and return an explicit failure/end result. An exact-capacity boundary test is meaningful for the standalone map. Normal coordinate maps deliberately keep spare buckets at 25–50% occupancy, so this is not counted as a demonstrated failure of the current default reserve policy. Sentinel-key insertion is separately documented as undefined by the vendor and is not treated as a new finding.

### V5 · P2 · Small-vector self-append writes beyond its storage

`src/primitives/small_vector.hpp:254–259`: `push_back(small_vector const &values)` computes an old destination index, then `resize(size_ + values.size_)`. If `values` is `*this`, resize doubles `values.size_` as well. The loop then copies this doubled count into an array sized only for twice the original count. Example: `small_vector<int> v{1,2,3}; v.push_back(v);` allocates 6 elements, then writes destination indices 3 through 8.

Correction: preserve the original source count before resize and handle source aliasing deliberately, or reject self-append explicitly. No production consumer of this class was identified.

### V6 · P2 · Small-vector element insertion does not preserve aliased values

`src/primitives/small_vector.hpp:238–250`: `insert`/single-value `push_back` retain a `T const&` argument across `resize` and shifting. Static-storage example: `v={1,2,3}; v.insert(0, v[1]);` should insert the original value 2, but `move_backward` first changes the referenced slot at index 1 to 1, so the implementation inserts 1. No allocation or undefined behavior is needed for this deterministic wrong-value example.

For an already dynamic vector, `push_back(v[i])` or `insert(..., v[i])` also retains a reference into the old heap block across `realloc`; when it moves that block, the final assignment reads freed storage. Crossing from static storage to heap for the first time alone does not establish this dangling-reference trigger, because the embedded static array remains alive.

Correction: copy the input value before any resize/shift. No production consumer of this class was identified. The static/static `swap` path also swaps up to the larger logical size (`:127`), potentially reading uninitialized scalar entries on the shorter side; this adjacent concern was not executed and is kept as a review follow-up rather than an additional grouped acceptance defect.

### V7 · P3 · Process NVTX range ownership is not correctly ended/transferred

`src/3rdparty/cudf/detail/nvtx/nvtx3.hpp:1792–1844`: `domain_process_range<D>` starts with `nvtxDomainRangeStartEx(domain::get<D>(), ...)` but destroys with global-domain `nvtxRangeEnd`, omitting the custom domain. Its move-assignment body neither returns `*this` despite returning a reference, nor ends an already-owned destination range; it also does not reset the destination `moved_from_` state when reusing a moved-from object. Thus the API can lose profiling-range ownership or fail to end the intended domain range.

Correction: end with the matching domain API, manage move state and existing ownership, return `*this`, and test moves between active and moved-from ranges. Current integration uses the separate thread-range RAII implementation, so no numerical/model-training effect is established.

## Further review candidates, not confirmed default failures

`robin_hood.h` is a modified vendored robin-hood-hashing 3.6.0 header. It was read in full, including allocator, hash, table, iterator, clone, insert, erase and resize logic. Its added iterator `steps` counter is not copied by mutable→const conversion/assignment (`:1203–1223`), so diagnostics may reset or retain a stale count if those conversions are used. Compile legality of cross-specialization private-member access and actual conversion use were not verified; neither is reported as a confirmed build failure. Exception safety during very large/allocation-failing rehashes would merit isolated fault-injection testing if this old header becomes a maintained dependency. No such tests or normal-path failure claim were produced here.

`MurmurHash3_32` uses reinterpretation/alignment assumptions and floating canonicalization; helper vectorized loads/stores and subword atomics likewise rely on type/alignment/storage contracts. Static reading alone does not establish a defect in supported integral coordinate-key instantiations. Do not convert these general C++ portability concerns into a count of actual corrupted coordinates without a supported-call reproducer.

No additional source-level defect was confirmed in `managed.cuh`, `ranges.hpp`, `device_operators.cuh`, the type declarations, `error.hpp`, or the legacy wrapper operators within their documented contracts. That statement is a bounded review result, not a proof that every possible template instantiation or CUDA version is correct.

## Complete file inventory

| File relative to MinkowskiEngine | Lines | Reading coverage |
|---|---:|---|
| `src/3rdparty/concurrent_unordered_map.cuh` | 599 | Entire file |
| `src/3rdparty/cudf/detail/nvtx/nvtx3.hpp` | 1,945 | Entire file |
| `src/3rdparty/cudf/detail/nvtx/ranges.hpp` | 54 | Entire file |
| `src/3rdparty/cudf/detail/utilities/device_atomics.cuh` | 634 | Entire file |
| `src/3rdparty/cudf/detail/utilities/device_operators.cuh` | 147 | Entire file |
| `src/3rdparty/cudf/detail/utilities/hash_functions.cuh` | 235 | Entire file |
| `src/3rdparty/cudf/types.h` | 232 | Entire file |
| `src/3rdparty/cudf/types.hpp` | 252 | Entire file |
| `src/3rdparty/cudf/utilities/error.hpp` | 134 | Entire file |
| `src/3rdparty/cudf/utilities/legacy/wrapper_types.hpp` | 515 | Entire file |
| `src/3rdparty/hash/hash_allocator.cuh` | 108 | Entire file |
| `src/3rdparty/hash/helper_functions.cuh` | 249 | Entire file |
| `src/3rdparty/hash/managed.cuh` | 48 | Entire file |
| `src/3rdparty/robin_hood.h` | 2,265 | Entire file |
| `src/primitives/small_vector.hpp` | 337 | Entire file |
| **Total** | **7,754** | **15/15** |

This sub-review excludes all other native files, C++ tests, setup/build scripts, Python API code and upstream repositories. Their evidence belongs to the coordinator's separate review. Vendor/helper defects should be recorded and tested before future use or upstream replacement; they do not alone decide whether the separate FastFill Qwen full-training launch is qualified.
