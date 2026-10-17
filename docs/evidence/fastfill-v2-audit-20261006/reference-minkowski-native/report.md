# MinkowskiEngine native 源码只读审查

审查日期：2026-10-06。对象为本地 `MinkowskiEngine/` 引用仓库；本报告不把该引用仓库的缺陷等同于 FastFill 当前训练入口正在调用这些功能。

## 范围、执行与证据

原生范围为 `src/`、`pybind/` 和 `tests/cpp/` 的全部 `.h/.hpp/.cuh/.cpp/.cu`：**91 文件、29,532 行**，包括 vendored headers。主审查者完整读生产原生代码及 11 个 C++/CUDA 测试，76 文件、21,778 行；协作审查者完整读 14 个 `src/3rdparty/` 文件和 `src/primitives/small_vector.hpp`，15 文件、7,754 行。文件级状态、原始 SHA256 与最终 hash 核对见 [native-coverage.json](native-coverage.json)；vendor 详细意见见 [vendor/REPORT.md](vendor/REPORT.md)。最终 **91/91 hash 不变**，native inventory 无新增或删除，见 [native-integrity-result.json](native-integrity-result.json)；这是该审查范围内源文件不变的证据。

本轮没有安装依赖、编译扩展、导入已编译 MinkowskiEngine、执行原生 CPU/CUDA 算子或运行服务器训练。阅读覆盖并非执行分支覆盖。下面的“确证”指源代码本身的索引、生命周期或接口契约错误；对于运行时后果，明确区分可从源码推导的错误与尚需实际 CUDA/native 检验的部分。

[mechanism_repros.py](mechanism_repros.py) 已在本地运行，产生 [mechanism_results.json](mechanism_results.json)。它只用 Python 标准库模拟源码中的整数索引和比较机制，绑定源文件 hash；**不是原生或 GPU 复现**。它验证了 label 索引、packed-count 索引、COPY_GEMM shared index、直接 max-pool、gapped batch 等反例的算术，不证明二进制编译或具体 CUDA 故障表现。

## 可从源码确证、公开或生产接口可达的问题

### N1 · P2 · 冲突 label 写入了另一个 voxel

`src/quantization.cpp:192` 使用 `colabels[inverse_mapping[val.first]]`。`val.first` 是 unique voxel index；`inverse_mapping` 的下标却是原始输入行号。应直接用 unique index 更新 `colabels`。

输入坐标 `A,A,B,B`、标签 `1,1,2,3`、invalid label=-1，B 发生冲突时 `val.first=1` 而 `inverse_mapping[1]=0`，得到 `[-1,2]`，期望 `[1,-1]`。Python 审查者确认公开 `sparse_quantize(..., labels=...)` 会到这个实现。本轮没有执行编译后的该接口。

### N2 · P1 · GPU spmm_average 用原 row ID 下标访问 packed counts

`src/spmm.cu:501–509` 的 `thrust::reduce_by_key` 把 counts 紧密写在 `[0,num_unique_keys)`；`:53` 却按 `sorted_row[x]` 原始 row ID 读取 `reduced_val`，没有用产出的 `unique_row_ptr` 建立对应关系。

合法 sparse row `rows=[5]`、`cols=[0]`、`size=(6,1)` 的 `nnz=1` 分配仅一个 count，源机制访问 count[5]，越过分配；`rows=[0,0,2]` 会读取未写入的 count[2]。公开 Python `spmm_average` 只核验 row/col 长度，不要求 row IDs 从零连续。CPU 实现以 unique inverse 正确取 counts，因此不是已声明的 dense-row 协议。实际 GPU sanitizer 与错误输出尚未执行。

### N3 · P1 · 空 kernel map 返回悬空 const 引用

`src/coordinate_map_manager.cpp:661` 返回 `kernel_map_type const&`，`:716–718` 空输入/输出分支却返回 `empty_map_functor()()` 临时对象。CPU functor `:619`、GPU `src/coordinate_map_manager.cu:246` 均按值返回空 map；临时对象在 return 表达式结束后销毁。

公开 manager `kernel_map_th` 与一些 pooling/convolution 路径消费该引用。能确证引用生命周期无效；不能从阅读声称每个空 convolution 都崩溃，因为部分调用者另有零行分支。

### N4 · P1/P2 · GPU TensorField quantization 多读一个 host stride，并丢失 device allocation

`src/coordinate_map_gpu.cu:159–165` 为 `coordinate_size` 个元素分配并复制 stride，输入资格实际是 `tensor_stride.size()==coordinate_size-1`。对正常 3D field，三项 stride 却复制四项，读取 host vector 有效范围之外。该函数到 `:185` 返回前没有释放 `d_tensor_stride`；函数内既没有 owner，也没有交给缓存。

`field_to_sparse_insert_and_map` 在 `src/coordinate_map_manager.cpp:244` 调用它。长度和 ownership 错误成立；实际复制内容、单次 device 内存增长量和 GPU 表现未测。

### N5 · P2 · GPU interpolation/field mapping 的 copy direction 与两端指针不一致

`src/coordinate_map_gpu.cu:2182–2192` 与 `:2262–2268` 从 device scratch `d_in_map/d_out_map/d_weight` 拷到 CUDA tensor，却声明 `cudaMemcpyHostToDevice`。这两端都是 device，方向应为 DeviceToDevice 或正确使用 Default/UVA 路径。

存在有效 mapping 时可达。这里确证的是 CUDA 指针方向契约不匹配；没有在本轮硬件/运行时执行，不能指定它必然返回哪种 CUDA error 或是否某种环境看似成功。

### N6 · P2 · CPU direct max pooling 把全负组输出为零

`src/direct_max_pool.cpp:83–86` 初始化 `out_feat/max_index` 为零；CPU `:118` 直接调用 `max_pooling_forward_pointer_kernel_cpu`，后者 `src/pooling_max_kernel.hpp:55` 只在当前值小于输入时更新。一个组的 features `[-3,-2]` 得到 0、mask=0，期望 -2、mask=1，前向和选中的反传来源均错误。

正常 `MaxPoolingForwardKernelCPU` 在 `pooling_max_kernel.hpp:76–78` 先填 mask=-1、output=-max，故本项限定 direct 路径。Python 审查者确认 `TensorField.sparse(MAX_POOL)` 的公开路径直接调用该函数，可由重复 field coordinates 与负特征触发。

### N7 · P2 · GPU direct max pooling 丢掉带空洞的输出 row ID

`src/pooling_max_kernel.cu:219–226` 传给 reduced kernel 的 `out_nrows` 是 `num_unique_out_map`，而 `:91` 把真正的 `out_map_row` 与该值比较。`out_map=[0,2]`、实际 `out_nrows=3` 时两组 distinct outputs 令传入值为 2，row 2 被跳过，虽然对应输出空间已经分配。

公开 `MinkowskiDirectMaxPoolingFunction` 接受独立 `out_nrows`，没有连续 row qualification；测试也使用可随机出现空洞的 output maps。正常 TensorField quantization 的 inverse mapping 通常紧密，此项不表示每次 field max 都失败。GPU 后果未实际执行。

### N8 · P2 · CPU origin map 拒绝每 batch 恰好一个 coordinate 的合法场景

`src/coordinate_map_cpu.hpp:737` 与 `:1081` 要求 `in_size > out_size`，但每 batch 一个输入时二者相等，仍是合法 origin reduction。两 batch、各一个 coordinate 的 Global Sum/Avg 路径会触发拒绝；单 batch 的部分 pooling 有单独 shortcut，不能用单 batch 证明问题。

Python 审查者确认 default Sum/Avg 的 PYTORCH_INDEX 枚举原样送入 native，公开 origin-map API 同样可达。本轮未执行编译后的 native 调用。

### N9 · P2 · Global pooling 的 PYTORCH_INDEX 路径混用 compact index 与原 batch ID

manager 的 `origin_map_th` 产出的 `batch_indices` 是实际 batch IDs，`vec_maps` 却按原 batch ID 索引，保留 gap 空条目（`src/coordinate_map_manager.cpp:869–873`）。CPU `src/global_pooling_cpu.cpp:145–153` 以 compact `b` 读 `vec_maps[b]`，再向只有 unique batch 数行的 `out_feat[batch_index[b]]` 写；GPU `src/global_pooling_gpu.cu:131–138` 也按 compact `b` 读 maps，并丢掉实际 IDs。

用 `[0,2]` 两 batch、各至少两点避开 N8：maps 为 `[batch0,empty,batch2]`，输出只有 2 行。CPU 尝试 output[2] 且拿错 input map；GPU 将第二组读为 empty，sum 丢失该 batch，avg 可产生 empty-mean NaN。noncontiguous batch 是项目 CHANGELOG 宣称支持且 Python tests 明确使用的输入。这里仍是静态控制流/索引反例，未运行 native。

### N10 · P1/P2 · Field map API 与 GPU global pooling 错用离散 map 前提

公开 manager `origin_field_map` 在 `src/coordinate_map_manager.cpp:926–927` 先调用 `origin_map_key`；该 helper `src/coordinate_map_manager.hpp:510` 不检查是否存在 discrete map，直接解引用 `m_coordinate_maps.begin()`。仅插入 TensorField 的 manager 可以有 field map 而没有 discrete map，此时直接调用 `origin_field_map(key)` 的引用无效；该 API 在 `pybind/extern.hpp:801` 导出，Python manager wrapper 也直接转发。正常 GlobalPoolingForward 在初始化未设置的 output key 时先调用 `origin_field()` 创建离散 origin map，所以不能把这个 empty-container 反例推广到所有正常 field pooling。

另外，GPU PYTORCH_INDEX `src/global_pooling_gpu.cu:132` 无论 `is_field` 都调用 `origin_map_th`，不像 CPU 使用 `origin_field_map_th`。公开 Global pooling 支持 TensorField，Python wrapper 用 field key；创建 origin map 并不会把输入 field key 变成 discrete key，因此多 batch 的这条 GPU 分支仍错用 API。实际错误/崩溃未执行；这两个触发条件分别记录，不能全称为一个已实测的 GPU crash。

### N11 · P1 · COPY_GEMM shared map 的 allocation 少一个可访问 entry

`src/math_functions.cuh:78/100` 用 `(threadIdx.x + block_rem)/length` 索引 shared map，`:120/:141` 却仅分配 `ceil(512/length)`。合法 `length=3`、342 个 mappings 令 `nthreads=1026`；block 1、thread 511 仍 active，其 `block_rem=2`，访问 index 171；仅分配 171 entries，最后合法 index=170。copy 与 accumulate 两个 helper 都有此机制。

`src/convolution_kernel.cu:451,485,613,658` 的 convolution COPY_GEMM forward/backward 调用这些 helpers；COPY_GEMM 是公开模式，channel 数并无整除 512 的限制。该例为整数机制确证，尚未运行 CUDA sanitizer。相邻 stream 问题另列“未解决的运行时验证”，不与本项混为一个已实测 race。

### N12 · P2 · 每次创建的新 cuSPARSE handle 未销毁

`src/gpu.cu:98–101` 的 `getCurrentCUDASparseHandle` 每调用一次执行 `cusparseCreate`，返回裸 handle；`src/` 中没有对应 `cusparseDestroy(handle)`。spmm、broadcast forward/backward、local/transpose pooling、global pooling 的实际 native 入口调用此 factory。

这是 handle ownership 丢失，区别于 Tensor descriptors 已被销毁的情况。未测每次资源损耗或累积失败阈值。`quantization.cpp` 使用 PyTorch 自己的 handle factory，不应连带把该调用标成此泄漏。

### N13 · P2 · 较大的合法 int32 coordinate 在 stride 中先丢失整数精度

CPU `src/coordinate_map.hpp:64/74` 显式转换 float 再 floor/stride；GPU `src/coordinate_map_gpu.cu:389–391` 使用 float 除法。合法 int32 coordinate 16,777,219、stride=2，转 float32 后成为 16,777,220，得到 strided coordinate 16,777,220，整数 floor 期望 16,777,218。

这明确限定为超出 float32 精确整数区间的输入；常规室内 voxel coordinate 通常较小，不能据此声称正常全部 geometry 错误。协议目前接受 int32 而未对上述值设资格上界，完整 int32 范围与实现精度不一致。CPU mechanism 已模拟；实际 GPU rounding/编译尚未验收。

### N14 · P1/P2 · Tensor quantization 的原生边界未验证 CPU 与 contiguous

`src/quantization.cpp` 的 `quantize_th`（`:103`）与 `quantize_label_th`（`:248`）用 data pointer 加 `row*ncols` 遍历，却不检查 tensor backend 或 strides。公开 `quantize(coords)` 的 `.int()` 对已是 int32 的非连续 tensor 可以保留原 storage/stride；`quantize_label` 的 direct wrapper 也缺少这些保护。`sparse_quantize` 的 label 路径有 CPU assertion，不能把该保护推广到所有 direct APIs；该路径的已有 int32 非连续输入仍需处理 contiguous。

反例地址机制：`base=[[0,1,0],[0,2,0]]`，`coords=base[:,::2]` 的逻辑内容是两行 `[0,0]`，shape 2×2、stride(3,2)，应有一个 unique voxel；dense pointer 读取首四个元素得到 `[0,1]` 和 `[0,0]`，两 unique。若把 CUDA tensor送到CPU解引用函数则 pointer/device 前提也不成立。本轮不执行此危险 native 调用；companion 审查者实际以 PyTorch 验证 `.int()` is same、stride 保留、非 contiguous，receipt 见 [quantize_stride_receipt.json](../reference-minkowski-python/quantize_stride_receipt.json)，该验证同样没有执行 MinkowskiEngine binary。

## helper、未使用代码与测试 harness：不能一律上升为当前训练故障

| 位置 | 确认或限制 | 当前可达性边界 |
|---|---|---|
| `src/storage.cuh:144–146` | `gpu_storage::resize` 从非空 old allocation 增长时复制 new length，old=4/new=8 会多读 4 元素 | helper 本身确证；当前生产调用主要从空分配到 N 后缩小，未找到明确的非零增长 default 路径 |
| `src/coordinate.hpp:125–136` | postfix ++/-- 返回下一步但不修改 iterator 本身，且返回值不是旧迭代器 | 当前循环主要用 prefix，未证明 postfix 活跃调用 |
| `src/math_functions_cpu.cpp:94` | `cpu_div<double>` 调了 `vdMul` | 唯一找到的 broadcast division caller 在注释中，不能称当前启用的 broadcast division 失败 |
| `src/coordinate_map_manager.cpp` 的 `origin/origin_field` | 搜索最小 map 时 `min_size` 未随选择更新，可能选择最后 qualifying map | 未构造证明输出损坏的合法 manager 序列，保留算法审查意见 |
| `src/allocators.cuh` 的 cached allocator | T!=char 时 cache key 的 bytes/elements 口径不一致 | 生产 pybind 未实例化此 backend，不能称当前 allocator 普遍失败 |
| `src/common.hpp` | 包含已不存在的旧 headers | 未找到 active include，不是生产安装失败证据 |

vendor 的 allocator、cycle iterator、small_vector、generic floating CAS、NVTX process range 等问题详见独立报告。特别是 `CoordinateMapGPU` 为 vendored concurrent map 显式传入项目 allocator，不能把 vendored 默认 allocator 的缺 throw 写成所有 GPU maps 的实际失败；项目的 `stride_type` 是 `std::vector`，`small_vector` 当前没有生产调用证据。正常 hash table 默认 occupancy 为 25%/50%，full-table absent-find 无限循环条件不属于正常 occupancy。

11 个 native test 文件也已读完。有些 extension helpers 保留旧 manager 三个 template parameters，而当前 class 要四个；convolution helpers 的 signature 与 stride binding 也落后，`type_test.cpp` 使用旧 `cpuInMap/cpuOutMap` 名称。GPU manager test 的 to_torch 还声明 DeviceToHost 但 destination 是 CUDA tensor。CPU batch-find test 使用 coordinate 行数代替 query 行数，查询数不等时其测试 helper 可能越界。这些是测试 harness 源兼容性/边界问题，**不是本轮实际编译结果**，也不能据此把正常 production build 直接判失败。

## 仍需真实环境解答的事项

- CUDA kernels/Thrust 多处在默认 stream 发射，而 PyTorch native wrapper/BLAS handle 使用 current stream，包括 COPY_GEMM helpers 未接受 stream 参数。源码 stream 使用不一致已观察；nonblocking stream 的真实 ordering/race 需要带 event/sanitizer 的运行验证，不能只靠阅读断言每次都有 race。
- cuSPARSE pooling 的 workspace / selected algorithm 假设、CUDA toolkit 与 PyTorch 兼容性、新旧内部 `torch::Backend/CheckedFrom` API、BLAS 链接、实际 native tests 的可编译性尚未验收。
- CPU direct map indices/device/contiguity 等边界也应做非法输入隔离验证；未把每个缺 assert 均算一个已发生的 native crash。
- CUSTOM region manager 明确 `ASSERT(region_type != CUSTOM, "Not implemented yet.")`。其 coordinate/cache 代码不完整属于当前不支持功能，不能称合法支持的 CUSTOM map 算错。

审查证明文件均被覆盖并指出了具体源代码缺陷；它不能证明 native 库已可安装、CUDA 分支已正确执行，也不构成 FastFill 训练正确性的替代证明。若后续真正采用这些算子，需先修复所采用路径的缺陷，使用对应真实环境补 native 反例回归，并保留输入资格与依赖版本。
