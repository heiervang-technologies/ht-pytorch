"""Flash Attention 2 for Vulkan with training support.

Implements the tiled online softmax algorithm with log-sum-exp saving
for memory-efficient attention. Forward and backward are fused Vulkan
compute shaders that avoid materializing the N*N attention matrix.
"""

import math
import struct
import torch
import logging
from typing import Optional

from pytorch_vulkan.device import load_shader_bytes

log = logging.getLogger(__name__)

_fa_fwd_pipeline = None
_fa_fwd_f16_pipeline = None
_fa_kvcache_f16_pipeline = None
_fa_bwd_pipeline = None
_fa_bwd_f16_pipeline = None

TILE_Q = 32


def _ext():
    try:
        from pytorch_vulkan import _C
        return _C
    except ImportError:
        return None


def _get_fa_pipelines():
    """Lazily compile and cache FA2 pipelines."""
    global _fa_fwd_pipeline, _fa_fwd_f16_pipeline, _fa_bwd_pipeline, _fa_bwd_f16_pipeline

    if _fa_fwd_pipeline is not None and _fa_fwd_f16_pipeline is not None and _fa_bwd_pipeline is not None and _fa_bwd_f16_pipeline is not None:
        return _fa_fwd_pipeline, _fa_fwd_f16_pipeline, _fa_bwd_pipeline, _fa_bwd_f16_pipeline

    ext = _ext()
    if ext is None:
        return None, None, None, None

    from pytorch_vulkan.inductor_backend import compile_glsl_to_spirv, SHADER_DIR

    for name, attr in [("flash_attn_fwd_v2.comp", "_fa_fwd_pipeline"),
                       ("flash_attn_fwd_v2_f16.comp", "_fa_fwd_f16_pipeline"),
                       ("flash_attn_bwd.comp", "_fa_bwd_pipeline"),
                       ("flash_attn_bwd_f16.comp", "_fa_bwd_f16_pipeline")]:
        path = SHADER_DIR / name
        if not path.exists():
            log.warning("FA2 shader not found: %s", path)
            return None, None, None, None
        spirv = compile_glsl_to_spirv(path.read_text())
        handle = load_shader_bytes(ext, spirv)
        if handle == 0:
            log.error("Failed to load FA2 pipeline: %s", name)
            return None, None, None, None
        globals()[attr] = handle
        log.info("Loaded FA2 pipeline %s (handle=%d)", name, handle)

    _fa_fwd_pipeline = globals()["_fa_fwd_pipeline"]
    _fa_fwd_f16_pipeline = globals()["_fa_fwd_f16_pipeline"]
    _fa_bwd_pipeline = globals()["_fa_bwd_pipeline"]
    _fa_bwd_f16_pipeline = globals()["_fa_bwd_f16_pipeline"]
    return _fa_fwd_pipeline, _fa_fwd_f16_pipeline, _fa_bwd_pipeline, _fa_bwd_f16_pipeline


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
        B, H, S, D_K = query.shape
        D_V = value.shape[-1]

        if scale is None:
            scale = 1.0 / math.sqrt(D_K)

        fwd_pipeline, fwd_f16_pipeline, bwd_pipeline, bwd_f16_pipeline = _get_fa_pipelines()
        ext = _ext()

        if fwd_pipeline is not None and ext is not None:
            orig_dtype = query.dtype
            
            if orig_dtype == torch.float16 and fwd_f16_pipeline is not None:
                q_flat = query.contiguous().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().view(B * H, S, D_V).contiguous()
                output = torch.empty(B * H, S, D_V, dtype=torch.float16, device=query.device)
                lse = torch.empty(B * H, S, dtype=torch.float32, device=query.device)
                pipeline_to_use = fwd_f16_pipeline
                ctx.bwd_pipeline = bwd_f16_pipeline
            else:
                q_flat = query.contiguous().float().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().float().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().float().view(B * H, S, D_V).contiguous()
                output = torch.empty(B * H, S, D_V, dtype=torch.float32, device=query.device)
                lse = torch.empty(B * H, S, dtype=torch.float32, device=query.device)
                pipeline_to_use = fwd_pipeline
                ctx.bwd_pipeline = bwd_pipeline

            push_data = struct.pack("<IIIf", S, D_K, D_V, scale)
            groups_x = (S + TILE_Q - 1) // TILE_Q

            success = ext.dispatch(
                pipeline_to_use,
                [q_flat, k_flat, v_flat, output, lse],
                (groups_x, B * H, 1),
                push_data,
            )

            if success:
                output = output.view(B, H, S, D_V).to(orig_dtype)
                lse = lse.view(B, H, S)
                ctx.save_for_backward(query, key, value, output, lse)
                ctx.orig_dtype = orig_dtype
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
        ctx.lse = None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        ext = _ext()

        # GPU backward via FA2 shader is validated. 
        # C++ shim supports 16 bindings and async zero-initialization.
        use_gpu_backward = True
        if use_gpu_backward and ctx.bwd_pipeline is not None and ext is not None and len(ctx.saved_tensors) == 5:
            # Use fused FA2 backward shader.
            query, key, value, output, lse = ctx.saved_tensors
            B, H, S, D_K = query.shape
            D_V = value.shape[-1]

            if query.dtype == torch.float16:
                q_flat = query.contiguous().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().view(B * H, S, D_V).contiguous()
                o_flat = output.contiguous().view(B * H, S, D_V).contiguous()
                do_flat = grad_output.contiguous().view(B * H, S, D_V).contiguous()
                lse_flat = lse.contiguous().view(B * H, S).contiguous()

                # f16 atomic adds might be unsupported, but we can safely allocate dQ, dK, dV in f32
                # and return them cast to f16. The shader handles the accumulation safely.
                dq = torch.zeros(B * H, S, D_K, dtype=torch.float32, device=query.device)
                dk = torch.zeros(B * H, S, D_K, dtype=torch.float32, device=query.device)
                dv = torch.zeros(B * H, S, D_V, dtype=torch.float32, device=query.device)
            else:
                q_flat = query.contiguous().view(B * H, S, D_K).contiguous()
                k_flat = key.contiguous().view(B * H, S, D_K).contiguous()
                v_flat = value.contiguous().view(B * H, S, D_V).contiguous()
                o_flat = output.contiguous().view(B * H, S, D_V).contiguous()
                do_flat = grad_output.contiguous().view(B * H, S, D_V).contiguous()
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
                if query.dtype == torch.float16:
                    dq = dq.to(torch.float16)
                    dk = dk.to(torch.float16)
                    dv = dv.to(torch.float16)
                return (
                    dq.view(B, H, S, D_K),
                    dk.view(B, H, S, D_K),
                    dv.view(B, H, S, D_V),
                    None,
                )

        # Fallback: decomposed backward on CPU.
        log.debug("FA2 backward fallback to CPU")
        if len(ctx.saved_tensors) == 5:
            query, key, value, output, lse = ctx.saved_tensors
        else:
            query, key, value, output = ctx.saved_tensors

        q, k, v = query.to("cpu"), key.to("cpu"), value.to("cpu")
        go = grad_output.to("cpu")
        device = grad_output.device

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        gv = torch.matmul(attn.transpose(-2, -1), go)
        ga = torch.matmul(go, v.transpose(-2, -1))
        dot = (ga * attn).sum(dim=-1, keepdim=True)
        gs = attn * (ga - dot) * scale

        gq = torch.matmul(gs, k)
        gk = torch.matmul(gs.transpose(-2, -1), q)

        return gq.to(device), gk.to(device), gv.to(device), None


def flash_attention_vulkan(query, key, value, scale=None):
    """Drop-in replacement for F.scaled_dot_product_attention using Flash Attention 2.

    Uses fused Vulkan compute shaders for O(N) memory attention.
    Supports training with backward pass.
    """
    return FlashAttentionVulkan.apply(query, key, value, scale)


def _get_kvcache_pipeline():
    """Lazily compile the KV-cache flash attention shader."""
    global _fa_kvcache_f16_pipeline
    if _fa_kvcache_f16_pipeline is not None:
        return _fa_kvcache_f16_pipeline

    ext = _ext()
    if ext is None:
        return None

    from pytorch_vulkan.inductor_backend import compile_glsl_to_spirv, SHADER_DIR
    path = SHADER_DIR / "flash_attn_fwd_v2_kvcache_f16.comp"
    if not path.exists():
        return None
    spirv = compile_glsl_to_spirv(path.read_text())
    handle = load_shader_bytes(ext, spirv)
    if handle == 0:
        return None
    _fa_kvcache_f16_pipeline = handle
    log.info("Loaded FA2 KV-cache pipeline (handle=%d)", handle)
    return handle


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
    B, H, Q_S, D_K = query.shape
    KV_S = key.shape[2]
    D_V = value.shape[-1]

    if scale is None:
        scale = 1.0 / math.sqrt(D_K)

    pipeline = _get_kvcache_pipeline()
    ext = _ext()

    if pipeline is not None and ext is not None and query.dtype == torch.float16:
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
