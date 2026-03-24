# PyTorch Vulkan Backend Roadmap

**Primary Objective:** Support full training loops (forward and backward passes, optimizers) on Vulkan-capable devices (AMD, Intel, etc.) using PyTorch 2.10 and `torch.compile`.

## Phase 1: Foundation (Current)
- [x] Establish out-of-tree extension architecture (PrivateUse1).
- [x] Scaffold Rust (`ash`, `gpu-allocator`) and C++ shim integration.
- [x] Register basic tensor factory ops (`empty`, `copy_`).
- [x] Set up Python `torch.compile` (`inductor`) backend registration.
- [x] Compile and validate GLSL shaders to SPIR-V via glslangValidator in CI.
- [x] Implement build system and CI pipeline for PyTorch 2.10.
- [x] End-to-end test of a simple pointwise FX graph compilation to SPIR-V.

## Phase 2: Core Operations
- [x] Map Inductor pointwise operations to Vulkan compute shaders.
- [x] Element-wise unary ops (neg, exp, log, tanh, sigmoid).
- [x] Map Inductor reduction operations to Vulkan compute shaders.
- [x] Implement naive GEMM via compute shaders for correctness.
- [x] Implement tiled shared-memory GEMM for performance.
- [x] Memory pool optimization and buffer reuse strategy.

## Phase 3: Training Support
- [x] Ensure autograd compatibility via `torch.compile` AOT Autograd / functorch.
- [x] Implement backward pass shader generation.
- [x] Gradient accumulation and zero_grad support.
- [x] Mixed precision support via float16 storage buffers.
- [ ] Investigate bf16 emulation for Vulkan (no native SPIR-V support; pack as uint16 with manual f32 conversion in shader, or use VK_KHR_shader_float16_int8 + software bf16<->f16 conversion).
- [ ] Optimize memory consumption for training graphs.
- [x] Validate optimizer steps (SGD, Adam, AdamW) on Vulkan backend.

## Phase 4: Performance & Optimization
- [x] Implement fused SDPA (Scaled Dot-Product Attention) kernel for Transformers.
- [x] Fix broadcast support in binary ops (required for correct SDPA backward).
- [x] Native SDPA dispatch on PrivateUse1 (encoder layer 125ms -> 17ms).
- [x] Push descriptors (VK_KHR_push_descriptor) for zero-alloc dispatch.
- [x] Cooperative matrix (NV) Tensor Core matmul for FP16.
- [x] Native addmm for nn.Linear layers.
- [x] **CRITICAL: Per-buffer fence tracking in allocator** - enabled async copy_ (encoder 200ms -> 51ms).
- [x] gpu-allocator integration for sub-microsecond buffer allocation (dispatch overhead 90us -> 2us).
- [x] Async copy_ via vkCmdCopyBuffer.
- [ ] Native _transformer_encoder_layer_fwd (implemented but disabled, needs async copy).
- [x] Implement true Flash-Attention 2 kernel for arbitrary sequence lengths (f32 & f16 forward and backward).
- [ ] Integrate [KernelAgent](https://github.com/meta-pytorch/KernelAgent.git) for automated GLSL template optimization.
- [x] Benchmark against CPU and CUDA baselines (LlamaBlock H=256 at 2.9ms).
- [ ] Vendor-specific tuning for AMD RDNA and Intel Arc architectures.
- [ ] Test functionality on Intel GPUs (discrete and integrated).

## Phase 4b: Op Coverage (reduce CPU fallbacks)
- [x] Native sqrt, abs, div, pow, rsqrt, threshold_backward, _local_scalar_dense.
- [x] Native cat (2-input, 18x faster than CPU fallback).
- [x] Native addcdiv/addcdiv_ for Adam optimizer.
- [x] Native lerp/lerp_ for Adam EMA updates.
- [x] Native addcmul/addcmul_ for Adam second moment.
- [x] In-place variants: add_.Scalar, mul_.Scalar, div_.Scalar, add_.Tensor for optimizer param updates.
- [x] Native sum/mean reduction for loss computation (non-dim and dim variants).
- [x] Native sub.out, _softmax_backward_data for training backward pass.
- [x] Native mse_loss/mse_loss_backward decomposition.
- [x] Scalar ops: add.Scalar, mul.Scalar, div.Scalar for optimizer internals.

## Phase 5: Model Support & New Architectures
- [x] RoPE (Rotary Position Embeddings) via native shader.
- [x] LayerNorm and RMSNorm fused kernels.
- [x] GELU and SiLU activations (fused, not decomposed).
- [ ] Experimental: Run Qwen 3.5 inference on Vulkan (gated delta nets architecture).
- [x] **Experimental: Tiny LLaMA (2-layer) end-to-end inference AND training on Vulkan in f16.**
- [ ] FP16/BF16 compute pipeline: full dtype support across all ops for half-precision training and inference.
- [ ] KV-cache management for autoregressive inference.

## Phase 6: Production Readiness (see issue #56)
- [ ] Autograd-preserving CPU fallback (upstream PR to PyTorch).
- [ ] Multi-vendor testing (AMD RDNA, Intel Arc/iGPU).
- [ ] CI pipeline with automated shader validation.
- [ ] Package as pip-installable wheel.
- [ ] Documentation: architecture guide, op support matrix, benchmarking guide.
