# Roadmap

The roadmap is ordered by release risk. Checked items indicate implementation,
not production evidence; release gates remain in
[docs/compatibility.md](docs/compatibility.md).

## Correctness baseline

- [x] Dtype-safe scalar optimizer kernels
- [x] Exact `empty_strided` metadata and range-aware alias lookup
- [x] Strict, per-operation CPU fallback accounting
- [x] Queried Vulkan capability bits
- [x] Matching NVIDIA cooperative-matrix extension and shader dialect
- [x] Bounded command submission
- [x] Explicit pipeline destruction and allocator statistics
- [x] Build-time SPIR-V packaging
- [x] KV-cache attention shader
- [ ] Eliminate direct CPU aliases in preparation for device-local memory
- [x] Boolean and additive broadcast-mask coverage
- [ ] Native generator-aware Vulkan RNG and dropout

## Training acceptance

- [x] Native embedding gradient path
- [x] Multi-step SGD, Adam, and AdamW parity tests
- [x] Real shifted causal language-model loss
- [x] Forward, gradient, parameter, and optimizer-state comparisons
- [x] Strict zero-fallback assertion for the supported reference model
- [x] Empty, dynamic, non-aligned, strided, masked, causal, and dropout tests
- [ ] Add larger published reference-model suites after hardware CI is green

## Compiler and memory

- [x] Rename the per-node executor from `inductor_backend.py`
- [x] Make the declarative operator registry the shader-loading source
- [x] Partition supported FX regions
- [ ] Add typed lowering and pointwise fusion
- [ ] Add graph-wide lifetime planning and allocation reuse
- [ ] Add device-local storage with pooled staging transfers
- [ ] Select host-visible tensors only under an explicit UMA/ReBAR policy

## Portability and release

- [x] Lavapipe shader, validation, and correctness workflow
- [x] Capability-gated NVIDIA, AMD, and Intel runner definitions
- [ ] Resolve package licensing from authoritative provenance
- [x] Compatibility, benchmarking, lifecycle, and coexistence documentation
- [ ] Provision and record green runs on all three hardware vendors
- [ ] Produce and verify reproducible release wheels
- [ ] Restore the original 214-commit history from an authoritative source
- [ ] Confirm two active human maintainers
- [x] Complete the ExecuTorch Vulkan reuse evaluation
- [ ] Prototype shared ExecuTorch inference components upstream
