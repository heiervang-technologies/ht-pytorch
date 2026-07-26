# pytorch-vulkan

`pytorch-vulkan` is an experimental, out-of-tree PyTorch backend for desktop
Vulkan compute. It uses `PrivateUse1` for eager execution and keeps PyTorch's
existing in-tree Vulkan backend untouched.

This package is an alpha research backend. Its checked support scope is
deliberately narrower than "all Vulkan GPUs" or "all PyTorch 2.x releases."
See [compatibility.md](docs/compatibility.md) for the release gates and current
evidence.

## Current design

- Rust Vulkan runtime using `ash` and `gpu-allocator`
- C++ `TORCH_LIBRARY_IMPL` kernels for `PrivateUse1`
- Build-time GLSL-to-SPIR-V compilation
- Capability-gated FP16, atomic-float, push-descriptor, and NVIDIA
  cooperative-matrix paths
- Strict fallback accounting for training and validation
- Eager training kernels for core optimizer and transformer operations
- Capability-partitioned FX executor registered as the `vulkan` compiler
  backend

The FX backend is not TorchInductor. It keeps unsupported partitions on
PrivateUse1 eager dispatch, but it does not yet provide Inductor scheduling,
general fusion, or graph-wide memory planning.

## Build

The supported build entry point is:

```bash
pip install -e . -v --no-build-isolation
```

The build requires:

- a compatible PyTorch 2.12 development or release build;
- Python 3.10 through 3.14;
- Vulkan headers and loader;
- `glslangValidator`;
- Rust 1.77 or newer with the locked Cargo dependencies available.

Shaders are compiled into package data during the build. Runtime
initialization never invokes a shader compiler.

## Usage

```python
import torch
import pytorch_vulkan

if not pytorch_vulkan.init():
    raise RuntimeError("No compatible Vulkan device is available")

print(pytorch_vulkan.device_info())

x = torch.arange(16, dtype=torch.float32).to("vkgpu:0")
y = (x + x).relu()
print(y.cpu())
```

Use strict fallback mode when validating a supported workload:

```python
pytorch_vulkan.reset_fallback_stats()
with pytorch_vulkan.strict_fallbacks():
    output = model(inputs)

assert pytorch_vulkan.fallback_stats()["count"] == 0
```

Unsupported operations are otherwise counted and may use PyTorch's CPU
fallback. CPU fallback is never evidence that an operation is Vulkan-native.

## Lifecycle

```python
pytorch_vulkan.memory_stats()
pytorch_vulkan.empty_cache()
pytorch_vulkan.shutdown()
pytorch_vulkan.init()
```

`shutdown()` refuses to destroy the runtime while tensor allocations are live.
Pipeline handles are explicitly destroyed before the Vulkan device.
`memory_stats()` reports allocator and pipeline counts as well as total,
pending, and flush-generation command statistics. Pending commands stay below
the `PYTORCH_VULKAN_MAX_PENDING_DISPATCHES` threshold.

## Testing

The Issue #4 acceptance suite uses PyTorch's device-generic test harness:

```bash
python tests/test_issue4_acceptance.py
```

CI compiles and validates every shader and runs correctness tests with
Lavapipe. NVIDIA, AMD, and Intel jobs are defined for explicitly provisioned
self-hosted runners; a vendor is not considered supported until its lane has a
recorded green run.

## Documentation

- [Compatibility and release gates](docs/compatibility.md)
- [Architecture and migration direction](docs/architecture.md)
- [ExecuTorch Vulkan reuse evaluation](docs/executorch-evaluation.md)
- [Benchmark protocol](docs/benchmarking.md)
- [Maintainer policy](MAINTAINERS.md)

## Provenance

The package entered this fork in commit `01f3484b11c` as a flattened snapshot
described as originating from `area51/pytorch-vulkan` with 214 commits. That
GitHub repository is no longer publicly resolvable. The snapshot commit and
its author metadata are retained, but the missing upstream objects cannot be
reconstructed from this checkout. A future history import must use an
authoritative bundle or repository supplied by the original maintainers.

## License

The standalone package is distributed under the [MIT License](LICENSE).
