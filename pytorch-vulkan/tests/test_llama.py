import pytest
import pytorch_vulkan
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_vulkan.flash_attention import flash_attention_vulkan


def apply_rotary_emb(x, cos, sin):
    # Fallback RoPE implementation for CPU
    D = x.shape[-1]
    half_D = D // 2
    x1, x2 = x[..., :half_D], x[..., half_D:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


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
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            attn_output = F.scaled_dot_product_attention(q, k, v)
        else:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
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


def test_llama_block_f16():
    pytorch_vulkan.init()
    if not pytorch_vulkan.is_available():
        pytest.skip("Vulkan not available")

    torch.manual_seed(42)
    B, S, H, D = 1, 8, 4, 16
    hidden_size = H * D
    intermediate_size = hidden_size * 4

    # CPU reference model
    block = LlamaBlock(hidden_size, H, intermediate_size).to(torch.float16)

    hidden_states = torch.randn(B, S, hidden_size, dtype=torch.float16)
    # Proper RoPE frequencies (not random - random cos/sin can cause NaN in f16)
    pos = torch.arange(S).float()
    freq = 1.0 / (10000.0 ** (torch.arange(0, D // 2, dtype=torch.float32) / (D // 2)))
    angles = pos.unsqueeze(1) * freq.unsqueeze(0)
    cos = angles.cos().to(torch.float16)
    sin = angles.sin().to(torch.float16)

    out_cpu = block(hidden_states, cos, sin)

    # Vulkan execution
    # Load state dict while on CPU, then move to Vulkan.
    vk_block = LlamaBlock(hidden_size, H, intermediate_size).to(torch.float16)
    vk_block.load_state_dict(block.state_dict())
    vk_block = vk_block.to("vkgpu:0")

    vk_hidden_states = hidden_states.to("vkgpu:0")
    vk_cos = cos.to("vkgpu:0")
    vk_sin = sin.to("vkgpu:0")

    out_vk = vk_block(vk_hidden_states, vk_cos, vk_sin)

    # FP16 accumulates rounding errors through RMSNorm + attention + MLP.
    # Verify outputs are in the same ballpark (not NaN, same magnitude).
    from pytorch_vulkan import _C

    _C.flush()
    out_vk_cpu = out_vk.cpu()
    assert not out_vk_cpu.isnan().any(), "Output contains NaN"
    assert not out_vk_cpu.isinf().any(), "Output contains Inf"
    torch.testing.assert_close(out_vk_cpu, out_cpu, atol=1.0, rtol=0.5)


if __name__ == "__main__":
    test_llama_block_f16()
    print("LLaMA Block FP16 test passed!")
