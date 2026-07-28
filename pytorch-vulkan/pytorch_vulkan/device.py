"""Vulkan device helpers wrapping the Rust/C++ extension."""

import struct

import torch


def _ext():
    try:
        from pytorch_vulkan import _C
        return _C
    except ImportError:
        return None


_initialized = False


def _shader_binding_count(spirv: bytes) -> int:
    """Read descriptor binding decorations from a SPIR-V module."""
    if len(spirv) < 20 or len(spirv) % 4:
        raise ValueError("invalid SPIR-V bytecode")
    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    bindings = set()
    offset = 5
    while offset < len(words):
        word_count = words[offset] >> 16
        opcode = words[offset] & 0xFFFF
        if word_count == 0 or offset + word_count > len(words):
            raise ValueError("invalid SPIR-V instruction stream")
        # OpDecorate %target Binding binding_number
        if opcode == 71 and word_count >= 4 and words[offset + 2] == 33:
            bindings.add(words[offset + 3])
        offset += word_count
    if not bindings:
        raise ValueError("SPIR-V shader has no descriptor bindings")
    return max(bindings) + 1


def load_shader_bytes(ext, spirv: bytes) -> int:
    """Load SPIR-V through the binding-count-aware extension ABI."""
    return int(ext.load_shader(spirv, _shader_binding_count(spirv)))


def init() -> bool:
    global _initialized
    if _initialized:
        return True
    ext = _ext()
    if ext is None:
        return False
    result = ext.init()
    if result:
        import sys
        import types
        # Use 'vkgpu' to avoid conflict with PyTorch's built-in vulkan backend
        torch.utils.rename_privateuse1_backend("vkgpu")
        vkgpu_mod = types.ModuleType("torch.vkgpu")
        vkgpu_mod.is_available = lambda: ext.is_available()
        vkgpu_mod.device_count = lambda: ext.device_count()
        sys.modules["torch.vkgpu"] = vkgpu_mod
        setattr(torch, "vkgpu", vkgpu_mod)
        torch.utils.generate_methods_for_privateuse1_backend("vkgpu")
        # Pre-compile core shaders and register handles with C++ for native dispatch.
        _register_native_shaders(ext)
        _initialized = True
    return result


def _register_native_shaders(ext):
    """Pre-compile GLSL shaders and register pipeline handles with C++.

    This enables native C++ dispatch for core ops (add, mul, etc.)
    bypassing Python/torch.compile entirely.
    """
    from pytorch_vulkan.inductor_backend import compile_glsl_to_spirv, SHADER_DIR
    import logging
    log = logging.getLogger(__name__)

    # Map of op_name -> (shader_file, push_constant_format)
    native_ops = {
        # FP32 ops
        "add": "add.comp",
        "mul": "mul.comp",
        "sub": "sub.comp",
        "relu": "relu.comp",
        "neg": "neg.comp",
        "sigmoid": "sigmoid.comp",
        "tanh": "tanh.comp",
        "exp": "exp.comp",
        "log": "log.comp",
        "gelu": "gelu.comp",
        "layer_norm": "layer_norm.comp",
        "rmsnorm": "rmsnorm.comp",
        "rope": "rope.comp",
        "matmul_tiled": "matmul_tiled.comp",
        "bmm": "bmm.comp",
        "softmax": "softmax.comp",
        "softmax_backward": "softmax_backward.comp",
        "silu": "silu.comp",
        "sqrt": "sqrt.comp",
        "abs": "abs.comp",
        "rsqrt": "rsqrt.comp",
        "embedding": "embedding.comp",
        "embedding_f16": "embedding_f16.comp",
        "div": "div.comp",
        "threshold_backward": "threshold_backward.comp",
        "pow": "pow.comp",
        "cat2": "cat2.comp",
        "copy": "copy.comp",
        "addcdiv": "addcdiv.comp",
        # FP16 variants
        "add_f16": "add_f16.comp",
        "add_bf16": "add_bf16.comp",
        "copy_bf16": "copy_bf16.comp",
        "mul_bf16": "mul_bf16.comp",
        "sub_bf16": "sub_bf16.comp",
        "div_bf16": "div_bf16.comp",
        "silu_bf16": "silu_bf16.comp",
        "softmax_bf16": "softmax_bf16.comp",
        "mul_f16": "mul_f16.comp",
        "sub_f16": "sub_f16.comp",
        "softmax_f16": "softmax_f16.comp",
        "relu_f16": "relu_f16.comp",
        "neg_f16": "neg_f16.comp",
        "sigmoid_f16": "sigmoid_f16.comp",
        "tanh_f16": "tanh_f16.comp",
        "exp_f16": "exp_f16.comp",
        "log_f16": "log_f16.comp",
        "silu_f16": "silu_f16.comp",
        "cat2_f16": "cat2_f16.comp",
        # Cooperative matrix (Tensor Core) - requires f16 inputs
        "bmm_coopmat": "bmm_coopmat.comp",
        "gelu_f16": "gelu_f16.comp",
        "layer_norm_f16": "layer_norm_f16.comp",
        "rmsnorm_f16": "rmsnorm_f16.comp",
        "rope_f16": "rope_f16.comp",
        "matmul_tiled_f16": "matmul_tiled_f16.comp",
        "copy_f16": "copy_f16.comp",
        "copy_f16_to_f32": "copy_f16_to_f32.comp",
        "copy_f32_to_f16": "copy_f32_to_f16.comp",
        "addcdiv_f16": "addcdiv_f16.comp",
        "addcmul": "addcmul.comp",
        "addcmul_f16": "addcmul_f16.comp",
        "lerp": "lerp.comp",
        "lerp_f16": "lerp_f16.comp",
        "lerp_tensor": "lerp_tensor.comp",
        "lerp_tensor_f16": "lerp_tensor_f16.comp",
        }
    if not hasattr(ext, "register_shader_handle"):
        log.debug("C++ extension does not support register_shader_handle yet")
        return

    for op_name, shader_file in native_ops.items():
        path = SHADER_DIR / shader_file
        if not path.exists():
            continue
        try:
            spirv = compile_glsl_to_spirv(path.read_text())
            handle = load_shader_bytes(ext, spirv)
            if handle != 0:
                ext.register_shader_handle(op_name, handle)
                log.debug("Registered native shader: %s (handle=%d)", op_name, handle)
        except Exception as e:
            log.warning("Failed to register native shader %s: %s", op_name, e)


def is_available() -> bool:
    ext = _ext()
    if ext is None:
        return False
    if not ext.is_available():
        ext.init()
    return ext.is_available()


def device_count() -> int:
    ext = _ext()
    return ext.device_count() if ext else 0


def device_name() -> str:
    ext = _ext()
    return ext.device_name() if ext else "N/A"
