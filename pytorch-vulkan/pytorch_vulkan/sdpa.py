"""Fused Scaled Dot-Product Attention for the Vulkan backend.

Registers a custom SDPA op that uses the fused sdpa.comp kernel for forward
and decomposes into existing ops (bmm, softmax, transpose) for backward.
"""

import logging
import math
import struct

import torch
import torch.nn.functional as F

from pytorch_vulkan.device import load_shader, shader_is_live
from pytorch_vulkan.operator_registry import shader_variant


log = logging.getLogger(__name__)

_sdpa_pipeline = None


def _ext():
    try:
        from pytorch_vulkan import _C

        return _C
    except ImportError:
        return None


def _get_sdpa_pipeline():
    """Lazily load and cache the packaged SDPA pipeline."""
    global _sdpa_pipeline
    if shader_is_live(_sdpa_pipeline):
        return _sdpa_pipeline

    _sdpa_pipeline = load_shader(shader_variant("fused_sdpa", "float32"))
    return _sdpa_pipeline


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
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Fused SDPA requires rank-4 tensors")
        if query.device != key.device or query.device != value.device:
            raise ValueError("Fused SDPA tensors must use the same device")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("Fused SDPA tensors must use the same dtype")
        B, H, S, D_K = query.shape
        D_V = value.shape[-1]
        if key.shape != query.shape:
            raise ValueError("Fused SDPA requires query and key shapes to match")
        if value.shape[:3] != query.shape[:3]:
            raise ValueError(
                "Fused SDPA requires matching batch, head, and sequence dimensions"
            )
        if D_K == 0:
            raise ValueError("Fused SDPA requires a non-empty query head")

        if scale is None:
            scale = 1.0 / math.sqrt(D_K)

        if B == 0 or H == 0 or S == 0 or D_V == 0:
            output = torch.empty(
                (B, H, S, D_V),
                dtype=query.dtype,
                device=query.device,
            )
            ctx.save_for_backward(query, key, value, output)
            ctx.scale = scale
            return output

        pipeline = _get_sdpa_pipeline()
        ext = _ext()

        if (
            pipeline is not None
            and ext is not None
            and query.dtype == torch.float32
            and S <= 256
            and D_V <= 256
        ):
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

        # Apply the softmax Jacobian without materializing it.
        dot = (grad_attn * attn_weights).sum(dim=-1, keepdim=True)
        grad_scores = attn_weights * (grad_attn - dot) * scale

        # grad_query = grad_scores @ key
        grad_query = torch.matmul(grad_scores, key)

        # grad_key = grad_scores^T @ query
        grad_key = torch.matmul(grad_scores.transpose(-2, -1), query)

        return grad_query, grad_key, grad_value, None


def vulkan_sdpa(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    """Drop-in replacement for F.scaled_dot_product_attention using fused Vulkan kernel."""
    if attn_mask is not None or dropout_p > 0.0 or is_causal:
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    return FusedSDPA.apply(query, key, value, scale)
