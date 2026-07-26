"""Flash Attention 2 for Vulkan with training support.

Implements the tiled online softmax algorithm with log-sum-exp saving
for memory-efficient attention. Forward and backward are fused Vulkan
compute shaders that avoid materializing the N*N attention matrix.
"""

import logging
import math
import struct

import torch

from pytorch_vulkan.device import device_info, load_shader, shader_is_live
from pytorch_vulkan.operator_registry import shader_variant


log = logging.getLogger(__name__)

_fa_fwd_pipeline = None
_fa_fwd_f16_pipeline = None
_fa_kvcache_f16_pipeline = None
_fa_bwd_pipeline = None
_fa_bwd_f16_pipeline = None

TILE_Q = 32
_MAX_SHADER_INDEX = (1 << 32) - 1
_MAX_SHADER_FLOAT = 3.4028235e38


def _fits_shader_launch(*values):
    return all(0 <= int(value) <= _MAX_SHADER_INDEX for value in values)


def _fits_shader_scale(scale):
    return not math.isfinite(scale) or abs(scale) <= _MAX_SHADER_FLOAT


def _ext():
    try:
        from pytorch_vulkan import _C

        return _C
    except ImportError:
        return None


def _get_fa_pipelines():
    """Lazily load and cache packaged FA2 pipelines."""
    global \
        _fa_fwd_pipeline, \
        _fa_fwd_f16_pipeline, \
        _fa_bwd_pipeline, \
        _fa_bwd_f16_pipeline

    capabilities = device_info()["capabilities"]
    if not shader_is_live(_fa_fwd_pipeline):
        _fa_fwd_pipeline = load_shader(
            shader_variant("flash_attention_forward", "float32"),
            capabilities,
        )
    if not shader_is_live(_fa_fwd_f16_pipeline):
        _fa_fwd_f16_pipeline = load_shader(
            shader_variant("flash_attention_forward", "float16"),
            capabilities,
        )
    if not shader_is_live(_fa_bwd_pipeline):
        _fa_bwd_pipeline = load_shader(
            shader_variant("flash_attention_backward", "float32"),
            capabilities,
        )
    if not shader_is_live(_fa_bwd_f16_pipeline):
        _fa_bwd_f16_pipeline = load_shader(
            shader_variant("flash_attention_backward", "float16"),
            capabilities,
        )
    return (
        _fa_fwd_pipeline,
        _fa_fwd_f16_pipeline,
        _fa_bwd_pipeline,
        _fa_bwd_f16_pipeline,
    )


class FlashAttentionVulkan(torch.autograd.Function):
    """Flash Attention 2 with forward and backward on Vulkan.

    Forward: Tiled online softmax, saves LSE for backward.
    Backward: Recomputes attention using LSE, atomic gradient accumulation.
    """

    @staticmethod
    def forward(ctx, query, key, value, scale=None):
        """
        Args:
            query: (B, H, S, D_k)
            key:   (B, H, S, D_k)
            value: (B, H, S, D_v)
            scale: float, defaults to 1/sqrt(D_k)
        """
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("Flash Attention requires rank-4 tensors")
        if query.device != key.device or query.device != value.device:
            raise ValueError("Flash Attention tensors must use the same device")
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError("Flash Attention tensors must use the same dtype")
        if query.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise ValueError("Flash Attention requires a floating-point dtype")
        B, H, S, D_K = query.shape
        D_V = value.shape[-1]
        if key.shape != query.shape:
            raise ValueError("Flash Attention requires query and key shapes to match")
        if value.shape[:3] != query.shape[:3]:
            raise ValueError(
                "Flash Attention requires matching batch, head, and sequence dimensions"
            )
        if D_K == 0:
            raise ValueError("Flash Attention requires a non-empty query head")

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
            ctx.bwd_pipeline = None
            return output

        fwd_pipeline, fwd_f16_pipeline, bwd_pipeline, bwd_f16_pipeline = (
            _get_fa_pipelines()
        )
        ext = _ext()
        selected_pipeline = (
            fwd_f16_pipeline if query.dtype == torch.float16 else fwd_pipeline
        )

        if (
            selected_pipeline is not None
            and ext is not None
            and D_K <= 64
            and D_V <= 64
            and _fits_shader_scale(scale)
            and _fits_shader_launch(
                B,
                H,
                S,
                D_K,
                D_V,
                B * H,
                B * H * S * D_K,
                B * H * S * D_V,
            )
        ):
            orig_dtype = query.dtype

            if orig_dtype == torch.float16:
                q_flat = query.contiguous().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().view(B * H, S, D_V).contiguous()
                output = torch.empty(
                    B * H, S, D_V, dtype=torch.float16, device=query.device
                )
                lse = torch.empty(B * H, S, dtype=torch.float32, device=query.device)
                pipeline_to_use = selected_pipeline
                ctx.bwd_pipeline = bwd_f16_pipeline
                ctx.pipeline_dtype = torch.float16
            else:
                q_flat = query.contiguous().float().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().float().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().float().view(B * H, S, D_V).contiguous()
                output = torch.empty(
                    B * H, S, D_V, dtype=torch.float32, device=query.device
                )
                lse = torch.empty(B * H, S, dtype=torch.float32, device=query.device)
                pipeline_to_use = selected_pipeline
                ctx.bwd_pipeline = bwd_pipeline
                ctx.pipeline_dtype = torch.float32

            push_data = struct.pack("<IIIf", S, D_K, D_V, scale)
            groups_x = (S + TILE_Q - 1) // TILE_Q

            success = ext.dispatch(
                pipeline_to_use,
                [q_flat, k_flat, v_flat, output, lse],
                (groups_x, B * H, 1),
                push_data,
            )

            if success:
                compute_output = output.view(B, H, S, D_V)
                output = compute_output.to(orig_dtype)
                lse = lse.view(B, H, S)
                ctx.save_for_backward(query, key, value, compute_output, lse)
                ctx.scale = scale
                return output

        # Fallback to decomposed attention.
        log.debug("FA2 pipeline unavailable, falling back to decomposed SDPA")
        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, value)
        ctx.save_for_backward(query, key, value, output)
        ctx.scale = scale
        ctx.bwd_pipeline = None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        ext = _ext()

        if (
            ctx.bwd_pipeline is not None
            and ext is not None
            and len(ctx.saved_tensors) == 5
        ):
            # Use fused FA2 backward shader.
            query, key, value, output, lse = ctx.saved_tensors
            B, H, S, D_K = query.shape
            D_V = value.shape[-1]

            if ctx.pipeline_dtype == torch.float16:
                q_flat = query.contiguous().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().view(B * H, S, D_V).contiguous()
                o_flat = output.contiguous().view(B * H, S, D_V).contiguous()
                do_flat = grad_output.contiguous().view(B * H, S, D_V).contiguous()
                lse_flat = lse.contiguous().view(B * H, S).contiguous()

                dq = torch.zeros(
                    B * H, S, D_K, dtype=torch.float32, device=query.device
                )
                dk = torch.zeros(
                    B * H, S, D_K, dtype=torch.float32, device=query.device
                )
                dv = torch.zeros(
                    B * H, S, D_V, dtype=torch.float32, device=query.device
                )
            else:
                q_flat = query.float().contiguous().view(B * H, S, D_K)
                k_flat = key.float().contiguous().view(B * H, S, D_K)
                v_flat = value.float().contiguous().view(B * H, S, D_V)
                o_flat = output.float().contiguous().view(B * H, S, D_V)
                do_flat = grad_output.float().contiguous().view(B * H, S, D_V)
                lse_flat = lse.contiguous().view(B * H, S).contiguous()

                dq = torch.zeros_like(q_flat)
                dk = torch.zeros_like(k_flat)
                dv = torch.zeros_like(v_flat)

            push_data = struct.pack("<IIIf", S, D_K, D_V, scale)
            groups_x = (S + TILE_Q - 1) // TILE_Q

            success = ext.dispatch(
                ctx.bwd_pipeline,
                [q_flat, k_flat, v_flat, o_flat, do_flat, lse_flat, dq, dk, dv],
                (groups_x, B * H, 1),
                push_data,
            )

            if success:
                return (
                    dq.view(B, H, S, D_K).to(query.dtype),
                    dk.view(B, H, S, D_K).to(key.dtype),
                    dv.view(B, H, S, D_V).to(value.dtype),
                    None,
                )

        # Fallback: decomposed backward on the input device.
        log.debug("FA2 backward using decomposed device operations")
        if len(ctx.saved_tensors) == 5:
            query, key, value, output, lse = ctx.saved_tensors
        else:
            query, key, value, output = ctx.saved_tensors

        attn = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        gv = torch.matmul(attn.transpose(-2, -1), grad_output)
        ga = torch.matmul(grad_output, value.transpose(-2, -1))
        dot = (ga * attn).sum(dim=-1, keepdim=True)
        gs = attn * (ga - dot) * scale

        gq = torch.matmul(gs, key)
        gk = torch.matmul(gs.transpose(-2, -1), query)

        return gq, gk, gv, None


def flash_attention_vulkan(query, key, value, scale=None):
    """Drop-in replacement for F.scaled_dot_product_attention using Flash Attention 2.

    Uses fused Vulkan compute shaders for O(N) memory attention.
    Supports training with backward pass.
    """
    return FlashAttentionVulkan.apply(query, key, value, scale)


def _get_kvcache_pipeline():
    """Lazily load the packaged KV-cache flash attention shader."""
    global _fa_kvcache_f16_pipeline
    if shader_is_live(_fa_kvcache_f16_pipeline):
        return _fa_kvcache_f16_pipeline
    _fa_kvcache_f16_pipeline = load_shader(
        shader_variant("flash_attention_kvcache", "float16")
    )
    return _fa_kvcache_f16_pipeline


def flash_attention_kvcache(query, key, value, scale=None):
    """Flash Attention with KV-cache support for autoregressive generation.

    Handles asymmetric Q/KV sequence lengths correctly using f32 accumulation
    in the shader to prevent f16 overflow.

    Args:
        query: (B, H, q_len, D) - typically q_len=1 during generation
        key:   (B, H, kv_len, D) - full cached key sequence
        value: (B, H, kv_len, D) - full cached value sequence
        scale: optional float scaling factor

    Returns:
        output: (B, H, q_len, D)
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("KV-cache attention requires rank-4 tensors")
    if query.device != key.device or query.device != value.device:
        raise ValueError("KV-cache attention tensors must use the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("KV-cache attention tensors must use the same dtype")
    if query.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise ValueError("KV-cache attention requires a floating-point dtype")
    B, H, Q_S, D_K = query.shape
    KV_S = key.shape[2]
    D_V = value.shape[-1]
    if key.shape[:2] != query.shape[:2] or key.shape[-1] != D_K:
        raise ValueError("KV-cache key shape is incompatible with query")
    if value.shape[:3] != key.shape[:3]:
        raise ValueError("KV-cache value shape is incompatible with key")
    if D_K == 0:
        raise ValueError("KV-cache attention requires a non-empty query head")
    if B == 0 or H == 0 or Q_S == 0 or D_V == 0:
        return torch.empty(
            (B, H, Q_S, D_V),
            dtype=query.dtype,
            device=query.device,
        )
    if KV_S == 0:
        return torch.zeros(
            (B, H, Q_S, D_V),
            dtype=query.dtype,
            device=query.device,
        )

    if scale is None:
        scale = 1.0 / math.sqrt(D_K)

    pipeline = _get_kvcache_pipeline()
    ext = _ext()

    if (
        pipeline is not None
        and ext is not None
        and query.dtype == torch.float16
        and D_K <= 64
        and D_V <= 64
        and _fits_shader_scale(scale)
        and _fits_shader_launch(
            B,
            H,
            Q_S,
            KV_S,
            D_K,
            D_V,
            B * H,
            B * H * Q_S * D_K,
            B * H * KV_S * D_K,
            B * H * KV_S * D_V,
            B * H * Q_S * D_V,
        )
    ):
        q_flat = query.contiguous().view(B * H, Q_S, D_K).contiguous()
        k_flat = key.contiguous().view(B * H, KV_S, D_K).contiguous()
        v_flat = value.contiguous().view(B * H, KV_S, D_V).contiguous()
        output = torch.empty(B * H, Q_S, D_V, dtype=torch.float16, device=query.device)
        lse = torch.empty(B * H, Q_S, dtype=torch.float32, device=query.device)

        # Push constants: q_seq_len, kv_seq_len, d_k, d_v, scale
        push_data = struct.pack("<IIIIf", Q_S, KV_S, D_K, D_V, scale)
        groups_x = (Q_S + TILE_Q - 1) // TILE_Q

        success = ext.dispatch(
            pipeline,
            [q_flat, k_flat, v_flat, output, lse],
            (groups_x, B * H, 1),
            push_data,
        )

        if success:
            return output.view(B, H, Q_S, D_V)

    # Fallback: decomposed attention (use f32 to avoid overflow)
    q_f32 = query.float()
    k_f32 = key.float()
    v_f32 = value.float()
    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v_f32)
    return output.to(query.dtype)
