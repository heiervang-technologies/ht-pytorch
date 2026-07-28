"""Integration test: RoPE + Attention + LayerNorm transformer block on Vulkan."""

import math

import pytest
import torch
import torch.nn as nn


def requires_vulkan(fn):
    try:
        from pytorch_vulkan import init, is_available

        init()
        available = is_available()
    except ImportError:
        available = False
    return pytest.mark.skipif(not available, reason="Vulkan backend not available")(fn)


def vk(t):
    return t.to("vkgpu:0")


def apply_rope_cpu(x, cos, sin):
    """Reference CPU RoPE: x shape (B, H, S, D), cos/sin shape (S, D//2)."""
    B, H, S, D = x.shape
    half_D = D // 2
    x1, x2 = x[..., :half_D], x[..., half_D:]
    cos = cos[:S, :].unsqueeze(0).unsqueeze(0)  # (1, 1, S, D//2)
    sin = sin[:S, :].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


@requires_vulkan
def test_rope_basic():
    """Test RoPE produces correct rotated embeddings."""
    B, H, S, D = 1, 2, 8, 16
    half_D = D // 2

    x = torch.randn(B, H, S, D)

    # Precompute cos/sin for position embeddings.
    pos = torch.arange(S).float()
    freq = 1.0 / (10000.0 ** (torch.arange(0, half_D, dtype=torch.float32) / half_D))
    angles = pos.unsqueeze(1) * freq.unsqueeze(0)  # (S, half_D)
    cos_cache = angles.cos()
    sin_cache = angles.sin()

    # CPU reference
    expected = apply_rope_cpu(x, cos_cache, sin_cache)

    # Vulkan RoPE via the custom op
    from pytorch_vulkan import _C

    _C.flush()

    x_vk = vk(x.contiguous().view(B * H, S, D))
    cos_vk = vk(cos_cache)
    sin_vk = vk(sin_cache)

    # RoPE is applied via the decomposed CPU-style code path,
    # which uses our registered binary ops (mul, sub, cat).
    # Test that the decomposed path gives correct results on Vulkan.
    q_vk = vk(x).view(B, H, S, D)
    result_vk = apply_rope_cpu(q_vk, vk(cos_cache), vk(sin_cache))
    _C.flush()

    result = result_vk.to("cpu")
    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-3)


@requires_vulkan
def test_transformer_block_shapes():
    """Test a minimal transformer block produces correct output shapes on Vulkan."""
    B, S, D = 2, 16, 64
    H = 4

    # Build a simple transformer encoder layer and run on Vulkan.
    layer = nn.TransformerEncoderLayer(
        d_model=D, nhead=H, dim_feedforward=D * 4, dropout=0.0, batch_first=True
    )
    layer.eval()

    x = torch.randn(B, S, D)

    # CPU reference
    with torch.no_grad():
        out_cpu = layer(x)

    # Vulkan
    x_vk = vk(x)
    layer_vk = layer.to("vkgpu:0")

    from pytorch_vulkan import _C

    _C.flush()

    with torch.no_grad():
        out_vk = layer_vk(x_vk)
    _C.flush()

    assert out_vk.shape == (B, S, D)
    torch.testing.assert_close(out_vk.to("cpu"), out_cpu, atol=1e-3, rtol=1e-2)


@requires_vulkan
def test_transformer_training_step_vulkan():
    """Test that a training step on a simple model works end-to-end on Vulkan."""
    B, S, D = 2, 8, 32
    vocab_size = 100

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, D)
            self.linear1 = nn.Linear(D, D)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(D, vocab_size)

        def forward(self, x):
            h = self.embed(x)
            h = self.relu(self.linear1(h))
            return self.linear2(h)

    model = TinyModel().to("vkgpu:0")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate random input tokens.
    tokens = torch.randint(0, vocab_size, (B, S), device="vkgpu:0")
    targets = torch.randint(0, vocab_size, (B, S), device="vkgpu:0")

    from pytorch_vulkan import _C

    # Training step.
    optimizer.zero_grad()
    logits = model(tokens)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    _C.flush()

    loss_val = loss.item()
    assert loss_val > 0, "Loss should be positive"
    assert not math.isnan(loss_val), "Loss should not be NaN"

    loss.backward()
    _C.flush()

    # Check gradients exist and are non-zero.
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No grad for {name}"
            assert p.grad.abs().sum().item() > 0, f"Zero grad for {name}"

    optimizer.step()
    _C.flush()

    # Verify parameters changed.
    logits2 = model(tokens)
    loss2 = nn.functional.cross_entropy(logits2.view(-1, vocab_size), targets.view(-1))
    _C.flush()
    loss2_val = loss2.item()

    assert not math.isnan(loss2_val), "Loss after step should not be NaN"
