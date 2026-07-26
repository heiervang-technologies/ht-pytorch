"""Compatibility imports for the renamed Vulkan FX backend."""

import warnings

from pytorch_vulkan.fx_backend import register, vulkan_compiler, vulkan_fx_compiler


warnings.warn(
    "pytorch_vulkan.inductor_backend was renamed to pytorch_vulkan.fx_backend; "
    "the backend does not use TorchInductor scheduling or code generation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["register", "vulkan_compiler", "vulkan_fx_compiler"]
