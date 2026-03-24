"""PyTorch Vulkan backend - out-of-tree PrivateUse1 device extension (Rust + C++ shim)."""

from pytorch_vulkan.device import is_available, device_count, device_name, init
from pytorch_vulkan.inductor_backend import register
from pytorch_vulkan.aot_backend import register_training, get_backward_op_report
from pytorch_vulkan.sdpa import vulkan_sdpa
from pytorch_vulkan.flash_attention import flash_attention_vulkan, flash_attention_kvcache
from pytorch_vulkan.kv_cache import KVCache, LayerKVCache

__all__ = [
    "init", "is_available", "device_count", "device_name",
    "register", "register_training", "get_backward_op_report",
    "vulkan_sdpa", "flash_attention_vulkan", "flash_attention_kvcache",
    "KVCache", "LayerKVCache",
]
