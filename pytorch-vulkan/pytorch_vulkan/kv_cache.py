"""KV-Cache for autoregressive inference on Vulkan.

Pre-allocates key/value buffers and appends new tokens each generation
step, avoiding recomputation of attention over the full sequence.
"""

import torch
from typing import Optional, Tuple


class KVCache:
    """Static KV-cache for a single attention layer.

    Pre-allocates (batch, heads, max_seq_len, head_dim) buffers for K and V.
    Each generation step appends new K/V at the current position.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_seq_len = max_seq_len
        self.pos = 0  # current fill position

        # Use empty + zero_ to ensure the async fill_buffer zeros the data.
        # torch.zeros on Vulkan may get a pooled buffer with stale data
        # if the zero fill hasn't flushed before reuse.
        self.k_cache = torch.empty(
            batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        ).zero_()
        self.v_cache = torch.empty(
            batch_size, num_heads, max_seq_len, head_dim,
            dtype=dtype, device=device,
        ).zero_()

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V tokens and return the full cached K/V.

        Args:
            k_new: (batch, heads, new_tokens, head_dim)
            v_new: (batch, heads, new_tokens, head_dim)

        Returns:
            (k_cached, v_cached): full K/V up to current position
        """
        new_len = k_new.shape[2]
        end_pos = self.pos + new_len

        assert end_pos <= self.max_seq_len, (
            f"KV-cache overflow: {end_pos} > {self.max_seq_len}"
        )

        self.k_cache[:, :, self.pos:end_pos, :] = k_new
        self.v_cache[:, :, self.pos:end_pos, :] = v_new
        self.pos = end_pos

        return (
            self.k_cache[:, :, :end_pos, :].contiguous(),
            self.v_cache[:, :, :end_pos, :].contiguous(),
        )

    def reset(self):
        """Reset cache for a new sequence."""
        self.pos = 0
        self.k_cache.zero_()
        self.v_cache.zero_()


class LayerKVCache:
    """KV-cache for all layers in a model."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ):
        self.caches = [
            KVCache(batch_size, num_heads, max_seq_len, head_dim, dtype, device)
            for _ in range(num_layers)
        ]

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.caches[layer_idx]

    def reset(self):
        for cache in self.caches:
            cache.reset()
