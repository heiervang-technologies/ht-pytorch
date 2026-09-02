"""Compile-friendly Qwen 3.5 building blocks used by the XPU workload."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def qwen_rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Qwen 3.5's one-centered RMSNorm with FP32 accumulation."""

    output = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (output * (1.0 + weight.float())).to(x.dtype)


def qwen_rms_norm_gated(
    hidden_states: Tensor,
    gate: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Qwen 3.5's DeltaNet output norm followed by a SiLU gate."""

    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.square().mean(-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = weight * normalized.to(input_dtype)
    return (normalized * F.silu(gate.float())).to(input_dtype)


def swiglu(x: Tensor, gate_weight: Tensor, up_weight: Tensor) -> Tensor:
    return F.silu(F.linear(x, gate_weight)) * F.linear(x, up_weight)


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rope(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    query_output = query * cos + rotate_half(query) * sin
    key_output = key * cos + rotate_half(key) * sin
    return query_output, key_output


def gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    beta: Tensor,
    decay: Tensor,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Differentiable fixed-shape DeltaNet recurrence without state mutation.

    Inputs use ``[batch, heads, sequence, feature]`` layout. ``beta`` and
    ``decay`` use ``[batch, heads, sequence]``. The explicit recurrence avoids
    custom CUDA-only kernels and is capturable by AOTAutograd on Intel XPU.
    """

    if query.shape[:-1] != key.shape[:-1] or query.shape[:-1] != value.shape[:-1]:
        raise ValueError("query, key, and value must share batch/head/sequence shape")
    if beta.shape != query.shape[:-1] or decay.shape != query.shape[:-1]:
        raise ValueError("beta and decay must match batch/head/sequence dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key feature dimensions must match")

    state_shape = (*query.shape[:2], key.shape[-1], value.shape[-1])
    state = query.new_zeros(state_shape) if initial_state is None else initial_state
    outputs = []
    for index in range(query.shape[2]):
        q_t = query[:, :, index]
        k_t = key[:, :, index]
        v_t = value[:, :, index]
        state = state * decay[:, :, index, None, None]
        prediction = torch.einsum("bhkv,bhk->bhv", state, k_t)
        residual = (v_t - prediction) * beta[:, :, index, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, residual)
        outputs.append(torch.einsum("bhkv,bhk->bhv", state, q_t))
    return torch.stack(outputs, dim=2), state


def _rms_norm_forward(module: nn.Module, x: Tensor) -> Tensor:
    return qwen_rms_norm(x, module.weight, module.eps)


def _mlp_forward(module: nn.Module, x: Tensor) -> Tensor:
    return module.down_proj(swiglu(x, module.gate_proj.weight, module.up_proj.weight))


def _rms_norm_gated_forward(
    module: nn.Module, hidden_states: Tensor, gate: Tensor
) -> Tensor:
    return qwen_rms_norm_gated(
        hidden_states,
        gate,
        module.weight,
        module.variance_epsilon,
    )


def enable_qwen35_compilation_paths(model: nn.Module) -> dict[str, int]:
    """Select equivalent, compiler-visible paths for supported Qwen modules.

    This is deliberately opt-in for the HT variant. The stock compiled variant
    is left untouched, making benchmark comparisons honest. Modules are selected
    by their Transformers class names so importing this file does not require
    Transformers or a model download.
    """

    replaced = {"rms_norm": 0, "rms_norm_gated": 0, "swiglu": 0}
    for module in model.modules():
        name = type(module).__name__
        if name in ("Qwen3_5RMSNorm", "Qwen3_5MoeRMSNorm"):
            module.forward = _rms_norm_forward.__get__(module, type(module))
            replaced["rms_norm"] += 1
        elif name in ("Qwen3_5RMSNormGated", "Qwen3_5MoeRMSNormGated"):
            if getattr(module, "activation", "silu") not in ("silu", "swish"):
                continue
            module.forward = _rms_norm_gated_forward.__get__(module, type(module))
            replaced["rms_norm_gated"] += 1
        elif name in ("Qwen3_5MLP", "Qwen3_5MoeMLP"):
            hidden_act = getattr(getattr(module, "config", None), "hidden_act", "silu")
            if hidden_act not in ("silu", "swish"):
                continue
            # PEFT wraps these projections. Reading a wrapper's base weight
            # directly would silently bypass its LoRA branch and gradients.
            if any(
                type(projection) is not nn.Linear
                for projection in (module.gate_proj, module.up_proj, module.down_proj)
            ):
                continue
            if module.gate_proj.bias is not None or module.up_proj.bias is not None:
                continue
            module.forward = _mlp_forward.__get__(module, type(module))
            replaced["swiglu"] += 1
    return replaced
