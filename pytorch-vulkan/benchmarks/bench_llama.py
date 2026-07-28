import time

import pytorch_vulkan
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_vulkan.flash_attention import flash_attention_vulkan


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin):
        B, S, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, S, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if hidden_states.device.type == "cpu":
            # Very slow CPU fallback
            pass
        elif hidden_states.device.type == "cuda":
            # Skip RoPE for raw benchmark speed equivalent
            attn_output = F.scaled_dot_product_attention(q, k, v)
        else:
            # We don't have torch.ops.vulkan_ops.rope registered in python namespace yet,
            # so we'll skip the RoPE call in the tight benchmark loop to strictly measure
            # the matmuls, norms, and attention.
            attn_output = flash_attention_vulkan(q, k, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.self_attn = LlamaAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, cos, sin):
        normed = self.input_layernorm(hidden_states)
        hidden_states = hidden_states + self.self_attn(normed, cos, sin)
        normed = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.mlp(normed)
        return hidden_states


import copy


def bench_llama():
    pytorch_vulkan.init()

    B, S, H, D = 1, 256, 16, 64
    hidden_size = H * D
    intermediate_size = hidden_size * 4

    block = LlamaBlock(hidden_size, H, intermediate_size).to(torch.float16)

    print("Benchmarking LLaMA Block (B=1, S=256, D=1024)")
    print("-" * 50)

    # CUDA
    if torch.cuda.is_available():
        block_cuda = copy.deepcopy(block).to("cuda")
        x_cuda = torch.randn(B, S, hidden_size, dtype=torch.float16, device="cuda")
        cos = torch.randn(S, D // 2, dtype=torch.float16, device="cuda")
        sin = torch.randn(S, D // 2, dtype=torch.float16, device="cuda")

        for _ in range(10):
            block_cuda(x_cuda, cos, sin)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            block_cuda(x_cuda, cos, sin)
        torch.cuda.synchronize()
        print(f"CUDA:   {(time.perf_counter() - start) * 1000 / 100:.3f} ms")

    # Vulkan
    block_vk = copy.deepcopy(block).to("vkgpu:0")
    x_vk = torch.randn(B, S, hidden_size, dtype=torch.float16, device="vkgpu:0")
    cos_vk = torch.randn(S, D // 2, dtype=torch.float16, device="vkgpu:0")
    sin_vk = torch.randn(S, D // 2, dtype=torch.float16, device="vkgpu:0")
    for _ in range(10):
        block_vk(x_vk, cos_vk, sin_vk)
    pytorch_vulkan._C.flush()

    start = time.perf_counter()
    for _ in range(100):
        block_vk(x_vk, cos_vk, sin_vk)
    pytorch_vulkan._C.flush()
    print(f"Vulkan: {(time.perf_counter() - start) * 1000 / 100:.3f} ms")


if __name__ == "__main__":
    bench_llama()
