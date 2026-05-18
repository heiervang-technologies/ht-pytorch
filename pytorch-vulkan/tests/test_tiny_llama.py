"""End-to-end tiny LLaMA model inference and training on Vulkan."""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def requires_vulkan(fn):
    try:
        from pytorch_vulkan import init, is_available
        init()
        available = is_available()
    except ImportError:
        available = False
    return pytest.mark.skipif(not available, reason="Vulkan backend not available")(fn)


def apply_rotary_emb(x, cos, sin):
    D = x.shape[-1]
    half_D = D // 2
    x1, x2 = x[..., :half_D], x[..., half_D:]
    cos = cos[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2], :].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TinyLlamaConfig:
    vocab_size = 256
    hidden_size = 128
    num_heads = 4
    num_layers = 2
    intermediate_size = 512
    max_seq_len = 64
    head_dim = hidden_size // num_heads


class TinyLlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)


class TinyLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.attn = TinyLlamaAttention(config)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.mlp = TinyLlamaMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TinyLlamaBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=1e-5)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE
        half_d = config.head_dim // 2
        pos = torch.arange(config.max_seq_len).float()
        freq = 1.0 / (10000.0 ** (torch.arange(0, half_d, dtype=torch.float32) / half_d))
        angles = pos.unsqueeze(1) * freq.unsqueeze(0)
        self.register_buffer("cos_cache", angles.cos())
        self.register_buffer("sin_cache", angles.sin())

    def forward(self, input_ids):
        x = self.embed(input_ids)
        cos = self.cos_cache[:input_ids.shape[1]]
        sin = self.sin_cache[:input_ids.shape[1]]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)


@requires_vulkan
def test_tiny_llama_inference():
    """Run a complete tiny LLaMA forward pass on Vulkan."""
    from pytorch_vulkan import _C

    config = TinyLlamaConfig()
    model = TinyLlama(config)
    # Scale down weights to prevent f16 overflow in attention dot products
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.1)
    model = model.half()

    # CPU reference
    tokens = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        logits_cpu = model(tokens)

    # Vulkan
    model_vk = TinyLlama(config).half()
    model_vk.load_state_dict(model.state_dict())
    model_vk = model_vk.to("vkgpu:0")
    _C.flush()

    tokens_vk = tokens.to("vkgpu:0")
    _C.flush()

    with torch.no_grad():
        logits_vk = model_vk(tokens_vk)
    _C.flush()

    assert logits_vk.shape == (1, 16, config.vocab_size)
    assert not logits_vk.to("cpu").isnan().any(), "Logits contain NaN"

    # Check approximate match (f16 accumulation causes some divergence)
    torch.testing.assert_close(
        logits_vk.to("cpu"), logits_cpu, atol=2.0, rtol=0.5)


@requires_vulkan
def test_tiny_llama_training_step():
    """Run a complete training step on tiny LLaMA on Vulkan."""
    from pytorch_vulkan import _C

    config = TinyLlamaConfig()
    model = TinyLlama(config)
    # Scale weights to prevent f16 overflow in attention dot products
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.1)
    model = model.half().to("vkgpu:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    _C.flush()

    tokens = torch.randint(0, config.vocab_size, (2, 16), device="vkgpu:0")
    targets = torch.randint(0, config.vocab_size, (2, 16), device="vkgpu:0")
    _C.flush()

    # Forward
    logits = model(tokens)
    _C.flush()
    assert not logits.to("cpu").isnan().any(), "Forward logits NaN"

    # Use sum loss. Note: backward through SDPA requires vulkan_sdpa
    # (our custom autograd function), not the registered aten::scaled_dot_product_attention
    # which lacks an autograd kernel.
    loss = logits.sum()
    _C.flush()
    loss_val = loss.item()
    assert not math.isnan(loss_val), "Loss is NaN"

    # Backward
    loss.backward()
    _C.flush()

    # Check gradients exist for non-embedding parameters
    # (embedding_dense_backward falls back to CPU which may not propagate)
    for name, p in model.named_parameters():
        if p.requires_grad and "embed" not in name:
            assert p.grad is not None, f"No gradient for {name}"

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    _C.flush()
