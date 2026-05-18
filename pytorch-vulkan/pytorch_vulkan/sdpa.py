"""Fused Scaled Dot-Product Attention for the Vulkan backend.

Registers a custom SDPA op that uses the fused sdpa.comp kernel for forward
and decomposes into existing ops (bmm, softmax, transpose) for backward.
"""

import math
import struct
import torch
import torch.nn.functional as F
import logging
from typing import Optional

log = logging.getLogger(__name__)

_sdpa_pipeline = None


def _ext():
    try:
        from pytorch_vulkan import _C
        return _C
    except ImportError:
        return None


def _get_sdpa_pipeline():
    """Lazily compile and cache the SDPA pipeline."""
    global _sdpa_pipeline
    if _sdpa_pipeline is not None:
        return _sdpa_pipeline

    ext = _ext()
    if ext is None:
        return None

    from pytorch_vulkan.inductor_backend import compile_glsl_to_spirv, SHADER_DIR

    shader_path = SHADER_DIR / "sdpa.comp"
    if not shader_path.exists():
        log.warning("sdpa.comp not found")
        return None

    spirv = compile_glsl_to_spirv(shader_path.read_text())
    handle = ext.load_shader(spirv)
    if handle == 0:
        log.error("Failed to load SDPA pipeline")
        return None

    _sdpa_pipeline = handle
    log.info("Loaded fused SDPA pipeline (handle=%d)", handle)
    return handle


class FusedSDPA(torch.autograd.Function):
    """Custom autograd function for fused SDPA.

    Forward uses the fused Vulkan kernel (sdpa.comp).
    Backward decomposes into standard ops that our backend handles.
    """

    @staticmethod
    def forward(ctx, query, key, value, scale=None):
        """
        Args:
            query: (batch, heads, seq_len, d_k)
            key:   (batch, heads, seq_len, d_k)
            value: (batch, heads, seq_len, d_v)
            scale: optional float, defaults to 1/sqrt(d_k)
        """
        B, H, S, D_K = query.shape
        D_V = value.shape[-1]

        if scale is None:
            scale = 1.0 / math.sqrt(D_K)

        pipeline = _get_sdpa_pipeline()
        ext = _ext()

        if pipeline is not None and ext is not None and S <= 256:
            # Use fused kernel.
            # Reshape to (B*H, S, D) for the shader.
            q_flat = query.contiguous().view(B * H, S, D_K)
            k_flat = key.contiguous().view(B * H, S, D_K)
            v_flat = value.contiguous().view(B * H, S, D_V)
            output = torch.empty(B * H, S, D_V, dtype=query.dtype, device=query.device)

            push_data = struct.pack("<IIIf", S, D_K, D_V, scale)

            # Dispatch: one workgroup per (query_row, batch_head).
            success = ext.dispatch(
                pipeline,
                [q_flat, k_flat, v_flat, output],
                (S, B * H, 1),
                push_data,
            )

            if success:
                output = output.view(B, H, S, D_V)
                # Save for backward.
                ctx.save_for_backward(query, key, value, output)
                ctx.scale = scale
                return output

        # Fallback: decomposed SDPA.
        log.debug("SDPA fallback: seq_len=%d > 256 or kernel unavailable", S)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, value)
        ctx.save_for_backward(query, key, value, output)
        ctx.scale = scale
        ctx.attn_weights = attn_weights
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass using decomposed ops on the same device (GPU-native)."""
        query, key, value, output = ctx.saved_tensors
        scale = ctx.scale

        # Recompute attention weights (avoids storing the large attn matrix).
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # grad_value = attn_weights^T @ grad_output
        grad_value = torch.matmul(attn_weights.transpose(-2, -1), grad_output)

        # grad_attn = grad_output @ value^T
        grad_attn = torch.matmul(grad_output, value.transpose(-2, -1))

        # Softmax backward: grad_scores = attn * (grad_attn - sum(grad_attn * attn, dim=-1, keepdim=True))
        dot = (grad_attn * attn_weights).sum(dim=-1, keepdim=True)
        grad_scores = attn_weights * (grad_attn - dot) * scale

        # grad_query = grad_scores @ key
        grad_query = torch.matmul(grad_scores, key)

        # grad_key = grad_scores^T @ query
        grad_key = torch.matmul(grad_scores.transpose(-2, -1), query)

        return grad_query, grad_key, grad_value, None


def vulkan_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Drop-in replacement for F.scaled_dot_product_attention using fused Vulkan kernel."""
    if attn_mask is not None or dropout_p > 0.0 or is_causal:
        # Fall back to PyTorch's implementation for masked/causal/dropout attention.
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask,
            dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    return FusedSDPA.apply(query, key, value, scale)
