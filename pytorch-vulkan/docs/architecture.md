# Architecture and migration direction

## Boundary

The backend remains a standalone `PrivateUse1` extension. This keeps desktop
eager execution and training experiments isolated from PyTorch's existing
mobile-oriented Vulkan implementation. Upstream PyTorch changes should be
small, backend-agnostic enabling changes with independent tests.

The package should ultimately move to its own repository. Until then,
`pytorch-vulkan/` is treated as a standalone distribution with its own license,
metadata, CI, and ownership policy.

## Runtime

The Rust runtime owns one Vulkan device, compute queue, bounded command batch,
allocator cache, and shader pipelines. The C++ layer stores PyTorch tensor
metadata and registers eager kernels. Python owns initialization, capability
gating, packaged shader loading, fallback policy, and public lifecycle APIs.

Buffer lookup is range-aware so aliases can resolve a base allocation plus
offset. Generic shader dispatch rejects unsupported layouts instead of
silently treating a view as contiguous.

The current runtime exposes one default stream. The dispatch batch is bounded
by `PYTORCH_VULKAN_MAX_PENDING_DISPATCHES` and synchronizes when the threshold
is reached. Additional streams and events require queue ownership, dependency,
and allocator-lifetime designs; they must not be represented as existing
features.

## Memory policy

The current allocator uses host-visible `CpuToGpu` memory. This is correct for
the present CPU-transfer and fallback bridge but is not the desired discrete
GPU policy.

The production design is:

1. device-local tensor buffers for normal execution;
2. pooled host-visible staging buffers for CPU transfers;
3. explicit transfer commands and synchronization;
4. direct host-visible tensor allocation only when a measured UMA or ReBAR
   policy selects it;
5. no CPU aliasing of device-local storage.

That change requires replacing all direct host-pointer fallback aliases first.
Until implemented and validated, discrete-GPU performance claims must disclose
the host-visible allocation policy.

## Operator registry

`pytorch_vulkan/operator_registry.py` is the source of shader variants,
dtype/layout/rank scope, capability requirements, FX names, dispatch shape,
autograd policy, exact descriptor count, push-constant schema, and numerical
tolerances. Native eager shader registration and the FX backend consume this
registry.

C++ contains kernel implementations but no second shader-file table. New
kernels must add their contract to the registry and acceptance coverage in the
same change.

## Graph compilation

`fx_backend.py` is an experimental FX executor, not TorchInductor. Mixed
graphs are split into capability-supported and eager submodules. Unsupported
submodules retain their device and execute through normal PrivateUse1 dispatch,
so the strict fallback policy remains authoritative. Supported partitions
still dispatch one shader per node and are not yet the production compiler.

The next compiler milestone is:

1. lower each supported partition to a typed internal graph;
2. fuse compatible pointwise producer/consumer regions;
3. plan intermediate lifetimes and reuse allocations;
4. emit one executable plan per guarded shape/dtype/capability signature.

The `inductor_backend.py` module remains only as a deprecation shim.

## ExecuTorch

The [reuse evaluation](executorch-evaluation.md) selects upstream sharing for
inference registries, capability handling, layouts, graph passes, validation,
and overlapping kernels. It rejects wholesale vendoring because ExecuTorch's
serialized inference delegate and this package's eager/autograd runtime have
different tensor, lifecycle, and execution contracts.

## Coexistence

No in-tree `DispatchKey::Vulkan` replacement is proposed. A future migration
requires:

- multi-vendor evidence and versioned wheels;
- an ownership agreement with both implementations;
- a mapping of layouts, operator contracts, and mobile constraints;
- a staged compatibility period with no ambiguous `torch.device` behavior;
- narrowly reviewed upstream changes rather than another snapshot import.
