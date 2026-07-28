"""Vulkan device lifecycle, capabilities, and packaged shader loading."""

from __future__ import annotations

import contextlib
import logging
import sys
import threading
import types
from importlib import resources
from typing import Iterator

import torch

from pytorch_vulkan.operator_registry import (
    native_shaders,
    shader_binding_count,
    ShaderVariant,
)


log = logging.getLogger(__name__)
_initialized = False
_torch_registered = False
_owned_shader_handles: set[int] = set()
_lifecycle_lock = threading.RLock()


def _ext():
    try:
        from pytorch_vulkan import _C

        return _C
    except ImportError:
        return None


def _register_torch_backend(ext) -> None:
    global _torch_registered
    if _torch_registered:
        return
    current_name = torch._C._get_privateuse1_backend_name()
    if current_name == "privateuseone":
        torch.utils.rename_privateuse1_backend("vkgpu")
    elif current_name != "vkgpu":
        raise RuntimeError(f"PrivateUse1 is already registered as {current_name!r}")

    vkgpu_mod = types.ModuleType("torch.vkgpu")
    vkgpu_mod.is_available = lambda: ext.is_available()
    vkgpu_mod.device_count = lambda: ext.device_count()
    vkgpu_mod.current_device = lambda: 0
    vkgpu_mod.set_device = lambda device=None: None
    vkgpu_mod._is_in_bad_fork = lambda: False
    vkgpu_mod.manual_seed_all = lambda seed: None
    vkgpu_mod.synchronize = lambda device=None: ext.flush()
    sys.modules["torch.vkgpu"] = vkgpu_mod
    setattr(torch, "vkgpu", vkgpu_mod)
    torch.utils.generate_methods_for_privateuse1_backend("vkgpu")
    _torch_registered = True


def _shader_bytes(shader_file: str) -> bytes:
    shader = resources.files("pytorch_vulkan").joinpath(
        "_shaders", f"{shader_file}.spv"
    )
    if not shader.is_file():
        raise RuntimeError(
            f"packaged SPIR-V shader is missing: {shader_file}.spv; "
            "rebuild pytorch-vulkan to compile its shaders"
        )
    return shader.read_bytes()


def load_shader(
    shader: ShaderVariant | str,
    capabilities: dict[str, bool] | None = None,
) -> int | None:
    with _lifecycle_lock:
        ext = _ext()
        if ext is None:
            return None
        if isinstance(shader, ShaderVariant):
            variant = shader
        else:
            required = set()
            if "_f16" in shader:
                required.update({"shader_float16", "storage_buffer16_bit_access"})
            if "coopmat" in shader:
                required.add("cooperative_matrix_nv")
            if "attn_bwd" in shader:
                required.add("shader_buffer_float32_atomic_add")
            variant = ShaderVariant(shader, frozenset(required))
        if capabilities is None:
            capabilities = device_info()["capabilities"]
        if any(not capabilities.get(name, False) for name in variant.capabilities):
            return None
        handle = int(
            ext.load_shader(
                _shader_bytes(variant.file),
                shader_binding_count(variant),
            )
        )
        if handle == 0:
            return None
        _owned_shader_handles.add(handle)
        return handle


def shader_is_live(handle: int | None) -> bool:
    with _lifecycle_lock:
        return handle is not None and handle in _owned_shader_handles


def _record_cpu_fallback(op_name: str) -> None:
    ext = _ext()
    if ext is not None and ext.is_available():
        ext.record_fallback(op_name)


def init() -> bool:
    global _initialized
    with _lifecycle_lock:
        if _initialized:
            return True
        ext = _ext()
        if ext is None or not ext.init():
            return False
        _register_torch_backend(ext)
        capabilities = dict(ext.device_info()["capabilities"])
        try:
            for op_name, variant in native_shaders(capabilities):
                handle = load_shader(variant, capabilities)
                if handle is not None:
                    ext.register_shader_handle(op_name, handle)
                    log.debug(
                        "registered Vulkan shader %s as %d",
                        op_name,
                        handle,
                    )
        except Exception:
            shutdown()
            raise
        _initialized = True
        return True


def shutdown() -> bool:
    global _initialized
    with _lifecycle_lock:
        ext = _ext()
        if ext is None or not ext.is_available():
            _initialized = False
            return True
        stats = dict(ext.memory_stats())
        if stats["active_allocations"]:
            log.error(
                "cannot shut down Vulkan with %d live tensor allocations",
                stats["active_allocations"],
            )
            return False
        for handle in tuple(_owned_shader_handles):
            if not ext.destroy_shader(handle):
                return False
            _owned_shader_handles.remove(handle)
        ext.clear_shader_handles()
        if not ext.shutdown():
            return False
        _initialized = False
        return True


def is_available() -> bool:
    return init()


def device_count() -> int:
    ext = _ext()
    return ext.device_count() if ext is not None and init() else 0


def device_name() -> str:
    ext = _ext()
    return ext.device_name() if ext is not None and init() else "N/A"


def device_info() -> dict:
    ext = _ext()
    if ext is None or not init():
        return {}
    return dict(ext.device_info())


def memory_stats() -> dict:
    ext = _ext()
    if ext is None or not ext.is_available():
        return {
            "active_bytes": 0,
            "cached_bytes": 0,
            "active_allocations": 0,
            "cached_allocations": 0,
            "active_pipelines": 0,
            "total_dispatches": 0,
            "pending_dispatches": 0,
            "flush_generation": 0,
            "auto_flush_threshold": 0,
        }
    return dict(ext.memory_stats())


def empty_cache() -> None:
    ext = _ext()
    if ext is not None and ext.is_available():
        ext.empty_cache()


def reset_fallback_stats() -> None:
    ext = _ext()
    if ext is None or not init():
        raise RuntimeError("Vulkan backend is unavailable")
    ext.reset_fallback_stats()


def fallback_stats() -> dict:
    ext = _ext()
    if ext is None or not init():
        return {"count": 0, "operations": {}, "strict": False}
    return dict(ext.fallback_stats())


@contextlib.contextmanager
def strict_fallbacks() -> Iterator[None]:
    ext = _ext()
    if ext is None or not init():
        raise RuntimeError("Vulkan backend is unavailable")
    previous = bool(ext.fallback_stats()["strict"])
    ext.set_strict_fallback(True)
    try:
        yield
    finally:
        ext.set_strict_fallback(previous)
