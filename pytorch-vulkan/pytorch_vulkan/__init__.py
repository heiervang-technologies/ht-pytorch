"""PyTorch Vulkan backend - out-of-tree PrivateUse1 device extension (Rust + C++ shim)."""

from pytorch_vulkan.aot_backend import get_backward_op_report, register_training
from pytorch_vulkan.device import (
    device_count,
    device_info,
    device_name,
    empty_cache,
    fallback_stats,
    init,
    is_available,
    memory_stats,
    reset_fallback_stats,
    shutdown,
    strict_fallbacks,
)
from pytorch_vulkan.flash_attention import (
    flash_attention_kvcache,
    flash_attention_vulkan,
)
from pytorch_vulkan.fx_backend import register
from pytorch_vulkan.kv_cache import KVCache, LayerKVCache
from pytorch_vulkan.sdpa import vulkan_sdpa


__all__ = [
    "init",
    "shutdown",
    "is_available",
    "device_count",
    "device_name",
    "device_info",
    "memory_stats",
    "empty_cache",
    "fallback_stats",
    "reset_fallback_stats",
    "strict_fallbacks",
    "register",
    "register_training",
    "get_backward_op_report",
    "vulkan_sdpa",
    "flash_attention_vulkan",
    "flash_attention_kvcache",
    "KVCache",
    "LayerKVCache",
]
