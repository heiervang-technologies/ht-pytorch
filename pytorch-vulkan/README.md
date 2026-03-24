# pytorch-vulkan

An out-of-tree PyTorch backend that runs training and inference on any Vulkan-capable GPU (NVIDIA, AMD, Intel). No CUDA required.

## What it does

- Registers a `vkgpu` device via PyTorch's PrivateUse1 extension mechanism
- 55 GLSL compute shaders (f32 + f16 variants) compiled to SPIR-V at init
- Native C++ eager-mode dispatch for all core ops (bypasses Python overhead)
- FP16 Tensor Core acceleration via NV cooperative matrix (`VK_NV_cooperative_matrix`)
- Flash Attention 2 forward + backward with atomic float accumulation
- Full training support: autograd, Adam/AdamW optimizers, `torch.compile` backend
- LLaMA architecture: RMSNorm, RoPE, SiLU/GELU, fused SDPA

## Quick start

```bash
# Requirements: Vulkan SDK, glslangValidator, Rust toolchain, PyTorch 2.10+
cd pytorch-vulkan
python -m venv venv && source venv/bin/activate
pip install torch  # PyTorch 2.10+
python setup.py build_ext --inplace

# Verify
python -c "
import pytorch_vulkan
pytorch_vulkan.init()
print(pytorch_vulkan.device_name())  # e.g. 'NVIDIA GeForce RTX 3090'
"
```

## Usage

```python
import torch
import pytorch_vulkan
pytorch_vulkan.init()

# Tensors on Vulkan
x = torch.randn(1024, device="vkgpu:0")
y = torch.randn(1024, device="vkgpu:0")
z = x + y  # runs on GPU via Vulkan compute shader

# FP16 with Tensor Core matmul
a = torch.randn(64, 128, dtype=torch.float16, device="vkgpu:0")
b = torch.randn(128, 64, dtype=torch.float16, device="vkgpu:0")
c = torch.mm(a, b)  # NV cooperative matrix on aligned dims

# Move a model to Vulkan
model = torch.nn.Linear(256, 128).to("vkgpu:0")
out = model(torch.randn(8, 256, device="vkgpu:0"))

# Training with Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = out.sum()
loss.backward()
optimizer.step()
```

### Flash Attention

```python
from pytorch_vulkan import flash_attention_vulkan

q = torch.randn(1, 8, 64, 64, device="vkgpu:0")
k = torch.randn(1, 8, 64, 64, device="vkgpu:0")
v = torch.randn(1, 8, 64, 64, device="vkgpu:0")
out = flash_attention_vulkan(q, k, v)  # O(N) memory, supports backward
```

### LLaMA block

```python
from pytorch_vulkan import vulkan_sdpa

# RMSNorm, RoPE, SDPA, SiLU MLP all run natively on Vulkan
# See tests/test_llama.py for a full LlamaBlock example
```

### torch.compile

```python
from pytorch_vulkan import register
register()

@torch.compile(backend="vulkan")
def fn(x, y):
    return torch.sigmoid(x + y)

result = fn(
    torch.randn(1024, device="vkgpu:0"),
    torch.randn(1024, device="vkgpu:0"),
)
```

## Architecture

```
pytorch-vulkan/
  rust/vulkan-compute/     Rust core: Vulkan device, allocator, pipeline, dispatch
    src/device.rs           VkDevice singleton, async command queue, push descriptors
    src/allocator.rs        Fence-tracked buffer pool with generation safety
    src/pipeline.rs         SPIR-V loading, push descriptor dispatch
  csrc/shim.cpp            C++ shim: PyTorch PrivateUse1 op registration
  pytorch_vulkan/           Python package
    device.py               Device init, shader pre-compilation
    inductor_backend.py     torch.compile backend (FX graph -> SPIR-V)
    flash_attention.py      Flash Attention 2 (fwd + bwd)
    sdpa.py                 Fused SDPA with autograd
    aot_backend.py          AOT Autograd training backend
  shaders/                 55 GLSL compute shaders (f32 + f16 variants)
  tests/                   120+ tests: correctness, training, f16, LLaMA
```

### Key design decisions

- **Host-mapped memory (ReBAR)**: All buffers use `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` when available, enabling zero-copy CPU fallbacks
- **Async command batching**: GPU commands are recorded into a single command buffer and submitted on flush, minimizing API overhead
- **Push descriptors**: `VK_KHR_push_descriptor` writes buffer bindings directly into the command buffer, bypassing descriptor pool allocation
- **Fence-tracked pool**: Freed buffers are tagged with their flush generation and only reused after the GPU work that used them has completed
- **C++ eager dispatch**: Core ops dispatch directly from C++ via `TORCH_LIBRARY_IMPL`, bypassing Python and torch.compile overhead

## Supported ops

| Category | Ops |
|----------|-----|
| Factory | `empty`, `empty_strided`, `zeros`, `fill_`, `normal_`, `uniform_` |
| Copy | `copy_`, `_to_copy`, `contiguous` (async GPU shader) |
| Binary | `add`, `mul`, `sub` (with broadcast) |
| Unary | `relu`, `neg`, `sigmoid`, `tanh`, `exp`, `log`, `silu`, `gelu` |
| Reduction | `sum`, `mean`, `sum.dim`, `mean.dim`, `softmax` |
| Matmul | `mm`, `bmm`, `addmm` (tiled + cooperative matrix) |
| Attention | `scaled_dot_product_attention`, Flash Attention 2 |
| Norm | `layer_norm`, `rms_norm` |
| Position | `rope` (rotary position embeddings) |
| Shape | `view`, `reshape`, `expand`, `as_strided`, `transpose`, `detach` |
| Other | `embedding`, `_copy_from_and_resize`, `resize_` |

All ops have FP16 variants.

## Running tests

```bash
python -m pytest tests/ -v
# 116 passed, 2 skipped, 2 xfailed
```

## Requirements

- Vulkan 1.2+ capable GPU with `HOST_VISIBLE` memory
- `glslangValidator` (Vulkan SDK or `pacman -S glslang`)
- Rust toolchain (`rustup`)
- PyTorch 2.10+
- Linux (tested on Arch Linux with NVIDIA RTX 3090)

## Performance

RTX 3090, PyTorch 2.10, Arch Linux:

| Operation | Vulkan | Notes |
|-----------|--------|-------|
| add 1M f32 | 0.080ms | Native compute shader |
| mm 256x256 f32 | 0.069ms | Tiled shared-memory matmul |
| mm 32x256 f16 | 0.073ms | Tiled f16 shader (no dtype conversion) |
| softmax 32x256 | 0.050ms | Fused workgroup reduction |
| Flash Attn v2 | 0.88ms | 256-thread optimized, 8x V parallelism |
| SDPA (B=4,H=8,S=64,D=64) | 0.016ms | 8.6x faster than CPU |
| Adam optimizer step | Zero fallback | All ops native on Vulkan |

### LlamaBlock f16 scaling (RMSNorm + RoPE + FlashAttn + SiLU MLP)

| Hidden size | Seq len | Latency | Notes |
|-------------|---------|---------|-------|
| 64 | 8 | 0.8-1.2ms | Small model, dispatch-bound |
| 128 | 16 | 0.8-1.3ms | Sweet spot |
| 256 | 32 | 1.4-2.9ms | Scales well |
| 512 | 64 | ~80ms | Attention dominates (O(S^2)) |

### Autoregressive generation (KV-cache + Flash Attention)

| Model | Tokens/sec | ms/token | Notes |
|-------|-----------|----------|-------|
| TinyLlama H=128 2-layer | 274 | 3.65 | Greedy decoding, f16 |

Custom models using `F.scaled_dot_product_attention` run fully on Vulkan.
`nn.TransformerEncoderLayer` falls back for the fused MHA kernel.

## License

MIT
