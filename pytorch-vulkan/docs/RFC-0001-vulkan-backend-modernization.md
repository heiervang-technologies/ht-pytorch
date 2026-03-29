# RFC-0001: Vulkan Backend Modernization

**Authors:** Markus Heiervang, Claude (Anthropic)
**Status:** Draft
**Created:** 2026-03-29
**Tracking:** heiervang-technologies/ht-pytorch#3

## Abstract

We propose replacing PyTorch's abandoned native C++ Vulkan backend with a modern Rust-based implementation that uses storage buffers (SSBOs), cooperative matrix operations, and supports both inference and training. The Vulkan backend has been unmaintained since PyTorch 2 and uses an architecture (image3D textures, no autograd, no torch.compile) that cannot reach competitive performance. Our Rust-based backend, currently running out-of-tree via PrivateUse1, has already demonstrated training, flash attention, and torch.compile support. This RFC describes promoting the Vulkan dispatch key to a first-class backend and wiring it to the Rust implementation.

## Motivation

### The case for Vulkan

CUDA locks machine learning to NVIDIA hardware. Vulkan 1.3 runs on NVIDIA, AMD, Intel, Qualcomm, Apple (via MoltenVK), and ARM GPUs. A competitive Vulkan backend makes every GPU a potential ML accelerator, reducing hardware dependency and e-waste from GPUs that are perfectly capable but lack CUDA support.

### Why the native backend can't get there

PyTorch's native Vulkan backend was built for mobile inference on Android. Its architecture has 6 fundamental limitations:

1. **image3D texture storage.** All tensors are stored as 3D image textures with a hardware-enforced 16384 dimension limit. Dimensions beyond this silently corrupt data. Weight matrices for 7B+ models exceed this. Memory bandwidth utilization is 3-12% of theoretical because texture sampling is optimized for spatial locality, not linear access patterns.

2. **No tensor core acceleration.** No VK_KHR_cooperative_matrix support. The matmul shader is a scalar 4x4 tiled loop. On an RTX 3090, CUDA tensor cores deliver 142 TFLOPS INT8 vs ~2 TFLOPS from scalar shaders -- a 70x gap on the most critical operation.

3. **No training support.** Zero backward kernels. No autograd integration beyond `requires_grad()` guard checks. The Vulkan dispatch key maps to `AutogradOther` (shared with FPGA, Metal, Sparse) rather than having a dedicated autograd key.

4. **No torch.compile.** Everything runs eagerly. No graph capture, no operator fusion, no equivalent to CUDA graphs. Each dispatch pays full Python-to-C++-to-Vulkan overhead.

5. **No async compute.** Single command queue. Submit info has `waitSemaphoreCount=0` and `signalSemaphoreCount=0`. No timeline semaphores. Cannot overlap compute and data transfer.

6. **No subgroup operations.** All 147 GLSL shaders use naive scalar loops for reductions. No `subgroupAdd`, `subgroupBallot`, or `subgroupShuffle` -- hardware-level primitives that are free in terms of latency.

### What the Rust backend already provides

Our Rust-based backend (`pytorch-vulkan/`) addresses most of these:

| Capability | Native C++ | Rust backend |
|-----------|-----------|-------------|
| Storage | image3D (16384 limit) | SSBOs (no limit) |
| Tensor cores | None | VK_KHR_cooperative_matrix |
| Data types | f32 only (f16 storage) | f32, f16, bf16 compute |
| Training | None | Flash attn bwd, softmax bwd |
| torch.compile | None | Inductor + AOT backends |
| Memory | Basic VMA | Fence-tracked buffer pool |
| Descriptors | Pool (32768 limit) | Push descriptors |
| Shaders | 147 (image3D) | 81 (SSBOs, multi-dtype) |

Benchmarked on RTX 3090: TinyLlama inference + training + generation at 274 tok/s.

The Rust backend currently runs via `PrivateUse1` dispatch, which is a hack intended for prototyping. Moving to the real Vulkan dispatch key makes it a proper PyTorch backend.

## Design

### Principle: own the dispatch key, not the workaround

The `PrivateUse1` mechanism requires runtime renaming (`rename_privateuse1_backend`), custom hooks registration, and breaks assumptions in PyTorch internals that check device type. A real backend with its own dispatch key gets:

- Native `torch.device("vulkan")` without hacks
- Dedicated `AutogradVulkan` for clean backward pass registration
- Proper device guards, allocator integration, and serialization
- Compatibility with all PyTorch tooling that dispatches by device type

### Architecture

```
                    PyTorch Dispatcher
                          |
                   DispatchKey::Vulkan
                          |
              +-----------+-----------+
              |                       |
    TORCH_LIBRARY_IMPL         AutogradVulkan
       (aten, Vulkan)          (backward passes)
              |                       |
         C++ Shim (adapted from shim.cpp)
              |
         extern "C" FFI
              |
    Rust vulkan-compute crate
    (ash, gpu-allocator, pipelines)
              |
         Vulkan 1.3 API
              |
         GPU Hardware
```

The C++ shim is a thin dispatch layer (~2600 lines) that converts PyTorch tensors to raw pointers and calls into the Rust crate via `extern "C"` functions. The Rust crate owns all GPU state: device initialization, memory allocation, pipeline management, and compute dispatch.

### Phase 1: Promote Vulkan to a real BackendComponent

The Vulkan dispatch key is currently a "fake backend" in `c10/core/DispatchKey.h`:

```cpp
// These are fake backends that map to the AutogradOther key
Vulkan, // TODO: put this in BackendComponents
```

We add `Vulkan` to `C10_FORALL_BACKEND_COMPONENTS`, which auto-generates `DispatchKey::Vulkan`, `AutogradVulkan`, `SparseVulkan`, and `QuantizedVulkan`.

**Files changed:**
- `c10/core/DispatchKey.h` -- move Vulkan into BackendComponents
- `c10/core/DispatchKeySet.h` -- remove from `autogradother_backends`
- `aten/src/ATen/core/VariableFallbackKernel.cpp` -- add AutogradVulkan fallthrough

This is a contained change (~20 lines) with no runtime impact when Vulkan is not in use.

### Phase 2: Remove the old native backend

Delete `aten/src/ATen/native/vulkan/` entirely:
- `api/` -- Context, Adapter, Command, Descriptor, Resource, Allocator (replaced by Rust crate)
- `glsl/` -- 147 shaders designed for image3D layout (replaced by 81 SSBO shaders)
- `ops/` -- 70+ op implementations (replaced by shim dispatch)
- `impl/` -- internal utilities
- `VulkanGuardImpl.cpp` -- replaced by new guard
- `VulkanOpaqueTensorImpl.h` -- not needed (Rust backend uses standard TensorImpl with real storage, not opaque wrappers)

The two architectures are incompatible at the storage layer (opaque image3D textures vs. host-mapped SSBOs with `data_ptr()`). There is no incremental migration path -- a clean replacement is simpler and safer.

### Phase 3: Wire the Rust backend to the Vulkan dispatch key

Adapt `pytorch-vulkan/csrc/shim.cpp`:

```cpp
// Before (PrivateUse1 hack):
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", &vulkan_add);
}
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    m.impl("scaled_dot_product_attention", &sdpa_autograd);
}

// After (real backend):
TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
    m.impl("add.Tensor", &vulkan_add);
}
TORCH_LIBRARY_IMPL(aten, AutogradVulkan, m) {
    m.impl("scaled_dot_product_attention", &sdpa_autograd);
}
```

Move the adapted shim into the PyTorch build tree. The Rust crate compiles to `libvulkan_compute.a` and links statically, gated behind the existing `USE_VULKAN` CMake flag.

### Phase 4: Python device layer

```python
# Before (hack):
torch.utils.rename_privateuse1_backend("vkgpu")
x = torch.randn(4, 4, device="vkgpu")

# After (native):
x = torch.randn(4, 4, device="vulkan")
```

No registration hacks, no custom hooks. `torch.device("vulkan")` works like `torch.device("cuda")`.

### Phase 5: Performance and feature completion

With the architectural foundation in place:

1. **Port INT8/INT4 quantized matvec** from the native ops work (#2) to SSBO shaders
2. **Build-time SPIR-V compilation** (currently compiled at Python init)
3. **Subgroup operations** in reduction shaders
4. **Timeline semaphores** for async compute/transfer overlap
5. **Expand op coverage** toward full aten parity

## Build system

The Rust crate adds a build dependency on `cargo`. This is gated behind `USE_VULKAN`:

```cmake
if(USE_VULKAN)
    # Compile Rust crate to static library
    add_custom_command(
        OUTPUT ${VULKAN_COMPUTE_LIB}
        COMMAND cargo build --release --manifest-path ${VULKAN_COMPUTE_DIR}/Cargo.toml
        COMMENT "Building vulkan-compute Rust crate"
    )
    # Link into PyTorch
    target_link_libraries(torch_cpu PRIVATE ${VULKAN_COMPUTE_LIB})
endif()
```

When `USE_VULKAN=OFF` (the default), the Rust toolchain is not required and the Vulkan backend is simply not compiled.

## Compatibility

### What breaks

- Code using `torch.device("vulkan")` with the old native backend on Android/mobile. The old backend's tensor format (opaque image3D) is incompatible. Saved Vulkan tensors from the old backend cannot be loaded.
- Any downstream code that registers ops at `AutogradOther` expecting to catch Vulkan tensors. After promotion, Vulkan tensors dispatch to `AutogradVulkan` instead.

### What doesn't break

- All non-Vulkan PyTorch functionality (CUDA, CPU, MPS, XPU) is completely unaffected.
- Standard PyTorch builds with `USE_VULKAN=OFF` see no changes.
- The Vulkan dispatch key name and device string remain `"vulkan"`.

### Migration path

For the small number of users of the old native backend (primarily Android):
- The old backend can be preserved as a separate build target if needed, but we do not expect demand.
- All existing Vulkan ops are re-implemented in the Rust backend with better performance.

## Alternatives considered

### Keep both backends

Run the native C++ backend for inference/mobile and the Rust backend for training/desktop. Rejected because it doubles maintenance burden, fragments the shader library, and the native backend's image3D architecture fundamentally limits performance even for inference.

### Port Rust backend features into the C++ backend

Incrementally add SSBOs, cooperative matrix, training support to the existing C++ code. Rejected because the storage layer incompatibility (image3D vs SSBO) means every op would need to be rewritten anyway, and the Rust crate provides better memory safety, build-time shader validation, and a cleaner FFI boundary.

### Upstream to PyTorch

Contribute directly to `pytorch/pytorch`. Not feasible in the short term -- upstream requires extensive review, CI infrastructure, and backward compatibility guarantees. Working in our fork lets us move fast. Upstreaming individual improvements (like the BackendComponent promotion) can happen later once the approach is proven.

## Open questions

1. **Android/mobile support.** Do we care about maintaining Vulkan on Android? The old backend was the primary user there. If yes, the Rust crate needs `aarch64-linux-android` cross-compilation support.

2. **Shader compilation strategy.** Currently shaders compile to SPIR-V at Python init via `glslangValidator`. Should we embed pre-compiled SPIR-V at build time? This improves startup latency but increases binary size and complicates the build.

3. **Memory allocator design.** The Rust backend uses a fence-tracked buffer pool. Should we add a caching allocator similar to CUDA's `CudaCachingAllocator` for better allocation performance under training workloads?

4. **Op coverage prioritization.** Which missing aten ops should we implement first for maximum impact? Current coverage is ~80 ops out of ~2000 in the aten namespace.

## References

- heiervang-technologies/ht-pytorch#2 -- Native backend inference ops PR (INT8/INT4 quantization)
- heiervang-technologies/ht-pytorch#3 -- Tracking issue
- `pytorch-vulkan/ROADMAP.md` -- Rust backend roadmap
- [Vulkan 1.3 spec](https://registry.khronos.org/vulkan/specs/1.3/html/)
- [VK_KHR_cooperative_matrix](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_cooperative_matrix.html)
