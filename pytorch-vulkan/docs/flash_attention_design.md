# Flash Attention 2 for Vulkan - Architectural Design

## Overview

Implement Flash Attention 2 (Dao, 2023) forward and backward passes as fused
Vulkan compute shaders. This is the key to competitive transformer performance
on Vulkan, enabling O(N) memory for attention instead of O(N²).

## Algorithm Summary

### Forward Pass
```
For each block of queries Q_i (tile_q rows):
  For each block of keys K_j (tile_k rows):
    S_ij = Q_i @ K_j^T / sqrt(d)         # local scores
    m_ij = rowmax(S_ij)                    # local max
    P_ij = exp(S_ij - m_ij)               # local softmax numerator
    l_ij = rowsum(P_ij)                    # local softmax denominator

    # Online softmax update (no materialization of full N×N matrix)
    m_new = max(m_old, m_ij)
    l_new = exp(m_old - m_new) * l_old + exp(m_ij - m_new) * l_ij
    O_i = diag(exp(m_old - m_new) / l_new) * O_i_old
        + diag(exp(m_ij - m_new) / l_new) * P_ij @ V_j

    m_old, l_old = m_new, l_new

Save: O (output), L = m + log(l) (log-sum-exp per row)
```

### Backward Pass
The backward recomputes attention on-the-fly using saved LSE:
```
For each block of queries Q_i:
  Load O_i, dO_i, L_i (saved log-sum-exp)
  D_i = rowsum(dO_i * O_i)    # diagonal correction term

  For each block of keys K_j:
    S_ij = Q_i @ K_j^T / sqrt(d)
    P_ij = exp(S_ij - L_i)     # recompute softmax using saved LSE

    dV_j += P_ij^T @ dO_i
    dP_ij = dO_i @ V_j^T
    dS_ij = P_ij * (dP_ij - D_i)   # softmax backward

    dQ_i += dS_ij @ K_j / sqrt(d)
    dK_j += dS_ij^T @ Q_i / sqrt(d)
```

## Vulkan Shader Architecture

### Tile Sizes
- TILE_Q = 32 (queries per tile)
- TILE_K = 32 (keys per tile)
- Head dim D = up to 128 (fits in registers/shared memory)
- Workgroup size: (TILE_Q, 1, 1) = 32 threads

### Shared Memory Layout
```glsl
shared float s_Q[TILE_Q][D];      // query tile
shared float s_K[TILE_K][D];      // key tile (loaded per inner loop)
shared float s_V[TILE_K][D];      // value tile (loaded per inner loop)
shared float s_S[TILE_Q][TILE_K]; // score tile
shared float s_O[TILE_Q][D];      // output accumulator
shared float s_m[TILE_Q];         // running max
shared float s_l[TILE_Q];         // running sum
```

### Forward Shader (flash_attn_fwd.comp)
- Dispatch: (ceil(seq_len/TILE_Q), batch*heads, 1) workgroups
- Each workgroup processes TILE_Q query rows
- Inner loop over K/V tiles
- Saves O and LSE = m + log(l) per row

### Backward Shader (flash_attn_bwd.comp)
- Dispatch: (ceil(seq_len/TILE_Q), batch*heads, 1) workgroups
- Each workgroup computes dQ for TILE_Q rows
- Needs atomic adds for dK, dV (accumulated across Q tiles)
- Or: separate pass for dK/dV with transposed tile iteration

### Push Constants
```glsl
layout(push_constant) uniform PushConstants {
    uint seq_len;
    uint d_k;
    uint d_v;
    float scale;        // 1/sqrt(d_k)
    uint num_tiles_k;   // ceil(seq_len / TILE_K)
};
```

### Buffer Bindings
Forward:
- binding 0: Q (batch*heads, seq_len, d_k)
- binding 1: K (batch*heads, seq_len, d_k)
- binding 2: V (batch*heads, seq_len, d_v)
- binding 3: O (output, same shape as Q with d_v)
- binding 4: LSE (batch*heads, seq_len) - log-sum-exp per row

Backward:
- binding 0: Q, K, V (read)
- binding 1: O, dO (read)
- binding 2: LSE (read)
- binding 3: dQ, dK, dV (write)

## Integration with PyTorch

### Autograd Function
```python
class FlashAttentionVulkan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        O, LSE = _dispatch_flash_attn_fwd(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, O, LSE)
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors
        dQ, dK, dV = _dispatch_flash_attn_bwd(Q, K, V, O, dO, LSE, ctx.scale)
        return dQ, dK, dV, None
```

### Key Difference from Inference-Only
The critical difference for training is saving and using LSE:
- Forward: compute LSE = m + log(l) for each query row
- Backward: use LSE to recompute P = exp(S - LSE) without storing full N×N

## References
- Flash Attention 2: https://arxiv.org/abs/2307.09288
- GGML Vulkan FA: ggml/src/ggml-vulkan/vulkan-shaders/flash_attn*.comp
- VulkanCooperativeMatrixAttention: github.com/etasnadi/VulkanCooperativeMatrixAttention
- Aule-Attention: github.com/AuleTechnologies/Aule-Attention
