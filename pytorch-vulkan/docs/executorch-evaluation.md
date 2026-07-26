# ExecuTorch Vulkan reuse evaluation

## Decision

Do not vendor or fork the ExecuTorch Vulkan backend into this package. Reuse
its design and pursue shared upstream components for inference, while keeping
PyTorch eager dispatch, autograd, optimizer kernels, and desktop lifecycle
ownership in this project.

This decision reflects different execution contracts:

- ExecuTorch lowers tagged Edge-dialect partitions into delegate-owned
  programs and executes serialized inference artifacts.
- This package implements live PyTorch `PrivateUse1` tensors, eager ATen
  dispatch, autograd, and training.
- ExecuTorch Vulkan is primarily documented as an Android inference backend,
  while this package currently targets Linux desktop development.

The boundaries do not justify duplicating capability, layout, shader, or
testing infrastructure.

## Components to reuse

| ExecuTorch component | Direction | Reason |
| --- | --- | --- |
| Declarative operator registry | Align concepts and field names | ExecuTorch uses its registry as partitioner source-of-truth; this package now does the same for shader loading |
| Capability and adapter selection | Extract or consume an upstream library | Device variation should not be rediscovered in two Vulkan runtimes |
| Layout transitions and tensor representation | Reuse design and tests | ExecuTorch explicitly manages optimal representations between operators |
| Graph partitioning and fusion passes | Adapt to FX/AOT Autograd | The partition/preprocess boundary is a better model than per-node Python dispatch |
| Dynamic-shape and symbolic metadata tests | Port applicable cases | Both projects need guarded shape-specialized execution |
| Inference shaders for matmul, SDPA, KV cache, RoPE, and quantized linear | Share only through reviewed upstream interfaces | These overlap directly with desktop inference |
| Validation and operator test generation | Port or invoke upstream tooling | Shader correctness and cross-device coverage benefit from one methodology |

Primary references:

- [ExecuTorch Vulkan backend](https://docs.pytorch.org/executorch/stable/android-vulkan.html)
- [Vulkan operator registry and support](https://docs.pytorch.org/executorch/stable/backends/vulkan/vulkan-op-support.html)
- [Delegate partition and preprocess model](https://docs.pytorch.org/executorch/stable/compiler-delegate-and-partitioner.html)

## Components that remain local

- `PrivateUse1` allocator, device guard, stream-facing API, and ATen kernels;
- `AutogradPrivateUse1`, AOT Autograd, gradient, and optimizer behavior;
- strict PyTorch CPU-fallback accounting;
- desktop package lifecycle and Python wheel integration;
- training-specific kernels and numerical acceptance suites.

## Integration constraints

Shader source cannot be copied opportunistically. Reuse requires a pinned
ExecuTorch revision, preserved license and attribution, an explicit mapping of
descriptor layouts and push constants, and differential tests on the same
hardware. Runtime code can be shared only after its ownership, threading,
allocator, and error contracts are compatible with eager PyTorch.

ExecuTorch portable fallback is also not a substitute for this package's
strict training mode. Unsupported partitions must remain observable and must
not cross a CPU boundary when zero fallback is required.

## Road to shared infrastructure

1. Pin a released ExecuTorch revision in a design PR and inventory overlapping
   kernels by schema, dtype, layout, and capability.
2. Replace the experimental FX interpreter with capability-based partitioning
   and a typed executable-plan boundary modeled on delegate preprocessing.
3. Prototype one shared inference kernel family, beginning with SDPA and KV
   cache, behind an adapter that preserves each runtime's tensor contract.
4. Run shader-level and end-to-end differential tests on Lavapipe, NVIDIA,
   AMD, Intel, and the Android devices supported by ExecuTorch.
5. Propose generally reusable capability, shader-registry, and validation
   pieces upstream rather than maintaining a downstream copy.
6. Expand sharing only when performance, binary size, ownership, and release
   cadence are demonstrably better than separate implementations.

The evaluation is complete; implementation of shared components remains a
separate, evidence-gated roadmap item.
