"""Inductor backend for torch.compile -- compiles FX graphs to SPIR-V compute shaders
and dispatches them via the Rust/C++ Vulkan backend."""

import struct
import torch
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SHADER_DIR = Path(__file__).parent.parent / "shaders"

# Cache: op_name -> (pipeline_handle, num_inputs)
_pipeline_cache: dict[str, tuple[int, int]] = {}


def _ext():
    try:
        from pytorch_vulkan import _C
        return _C
    except ImportError:
        return None


_registered = False

def register():
    """Register the Vulkan backend with torch.compile's Inductor."""
    global _registered
    if _registered:
        return
    torch._dynamo.register_backend(vulkan_compiler, name="vulkan")
    _registered = True
    log.info("Registered Vulkan Inductor backend")


# ---------------------------------------------------------------------------
# Supported pointwise ops and their shader / dispatch metadata
# ---------------------------------------------------------------------------

# Float16 shader variants. If a tensor is f16 and an f16 variant exists,
# we use it instead of the f32 version. The f16 shaders use
# GL_EXT_shader_explicit_arithmetic_types_float16 for native half precision.
_F16_VARIANTS = {
    "add": "add_f16.comp",
    "mul": "mul_f16.comp",
    "_softmax": "softmax_f16.comp",
}

# Maps FX op name -> (shader_file, num_inputs, num_outputs)
_POINTWISE_OPS = {
    # Binary ops
    "add": ("add.comp", 2, 1),       # a + b -> result
    "mul": ("mul.comp", 2, 1),       # a * b -> result
    # Unary ops
    "relu": ("relu.comp", 1, 1),     # max(a, 0) -> result
    "neg": ("neg.comp", 1, 1),       # -a -> result
    "exp": ("exp.comp", 1, 1),       # exp(a) -> result
    "log": ("log.comp", 1, 1),       # log(a) -> result
    "tanh": ("tanh.comp", 1, 1),     # tanh(a) -> result
    "sigmoid": ("sigmoid.comp", 1, 1),  # 1/(1+exp(-a)) -> result
    "sub": ("sub.comp", 2, 1),           # a - b -> result
    # Comparison / mask ops (backward support)
    "le": ("le.comp", 2, 1),             # (a <= b) ? 1.0 : 0.0
    "where": ("where_.comp", 3, 1),      # cond ? a : b
    # Fused backward ops
    "threshold_backward": ("threshold_backward.comp", 2, 1),  # relu backward
}

_MATMUL_OPS = {
    "mm": ("matmul_tiled.comp", 2, 1),
}

_BMM_OPS = {
    "bmm": ("bmm.comp", 2, 1),
}

_TRANSPOSE_OPS = {
    "t": ("transpose.comp", 1, 1),
    "transpose": ("transpose.comp", 1, 1),
}

# Reduction ops need special multi-pass dispatch logic.
# Maps FX op name -> (shader_file, finalizer)
# "finalizer" is None for sum, or a callable that post-processes the scalar result.
_REDUCTION_OPS = {
    "sum": ("sum.comp", None),
    "mean": ("mean.comp", "mean"),  # divide by num_elements after reduction
}

# Fused ops with special dispatch (one workgroup per row).
_FUSED_OPS = {
    "_softmax": "softmax.comp",
    "_softmax_backward_data": "softmax_backward.comp",
}

# Dim-aware reductions for backward pass shape preservation.
_DIM_REDUCTION_OPS = {
    "sum": "sum_dim.comp",
    "mean": "mean_dim.comp",
}


def vulkan_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    """Entry point called by torch.compile(backend='vulkan').

    Receives an FX GraphModule from Dynamo, returns a callable that
    executes the graph on Vulkan via compiled SPIR-V shaders.
    """
    ext = _ext()
    if ext is None:
        log.warning("C extension not available, falling back to eager")
        return gm

    log.info("Compiling FX graph with %d nodes for Vulkan", len(gm.graph.nodes))

    # Pre-compile all supported ops in the graph.
    for node in gm.graph.nodes:
        if node.op == "call_function":
            op_name = _get_op_name(node)
            if op_name in _POINTWISE_OPS and op_name not in _pipeline_cache:
                shader_file, num_inputs, _ = _POINTWISE_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _MATMUL_OPS and op_name not in _pipeline_cache:
                shader_file, num_inputs, _ = _MATMUL_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _BMM_OPS and op_name not in _pipeline_cache:
                shader_file, num_inputs, _ = _BMM_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            # Pre-compile dim-aware reduction variants.
            if op_name in _DIM_REDUCTION_OPS:
                dim_key = f"{op_name}_dim"
                if dim_key not in _pipeline_cache:
                    pipeline = _compile_and_load(ext, _DIM_REDUCTION_OPS[op_name])
                    if pipeline is not None:
                        _pipeline_cache[dim_key] = (pipeline, 1)
            # Pre-compile fused ops (softmax etc).
            if op_name in _FUSED_OPS and op_name not in _pipeline_cache:
                pipeline = _compile_and_load(ext, _FUSED_OPS[op_name])
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, -1)  # special dispatch
            # Pre-compile f16 shader variants.
            if op_name in _F16_VARIANTS:
                f16_key = f"{op_name}_f16"
                if f16_key not in _pipeline_cache:
                    pipeline = _compile_and_load(ext, _F16_VARIANTS[op_name])
                    if pipeline is not None:
                        _pipeline_cache[f16_key] = (pipeline, _POINTWISE_OPS.get(op_name, (None, -1))[1])

    # Build the compiled execution function.
    # Walk the graph and replace supported ops with Vulkan dispatches.
    node_list = list(gm.graph.nodes)

    def run(*args):
        env: dict[str, torch.Tensor] = {}

        # Populate env with placeholder (input) values.
        arg_idx = 0
        for node in node_list:
            if node.op == "placeholder":
                env[node.name] = args[arg_idx]
                arg_idx += 1

        for node in node_list:
            if node.op == "call_function":
                op_name = _get_op_name(node)

                if op_name in _REDUCTION_OPS and op_name in _pipeline_cache:
                    input_tensor = env[node.args[0].name] if isinstance(node.args[0], torch.fx.Node) else node.args[0]

                    # Check if this is a dim-aware reduction (has dim argument).
                    has_dim = len(node.args) > 1 and node.args[1] is not None
                    dim_key = f"{op_name}_dim"

                    if has_dim and dim_key in _pipeline_cache:
                        # Dim-aware reduction.
                        result = _dispatch_dim_reduction(
                            ext, _pipeline_cache[dim_key][0],
                            input_tensor, node.args, op_name)
                    else:
                        # Full reduction to scalar.
                        pipeline_handle, _ = _pipeline_cache[op_name]
                        orig_numel = input_tensor.numel()
                        result = _dispatch_reduction(ext, pipeline_handle, input_tensor)

                    if result is not None:
                        env[node.name] = result
                    else:
                        log.warning("Vulkan reduction failed for %s, falling back", op_name)
                        env[node.name] = _eager_call(node, env)

                elif op_name in _pipeline_cache:
                    pipeline_handle, num_inputs = _pipeline_cache[op_name]

                    if num_inputs != -1:
                        # If any input is a scalar, fall back to eager since our shaders expect buffers
                        if any(not isinstance(a, torch.fx.Node) for a in node.args[:num_inputs]):
                            log.warning("Vulkan dispatch failed for %s due to scalar input, falling back", op_name)
                            env[node.name] = _eager_call(node, env)
                            continue

                    input_tensors = [env[a.name] for a in node.args[:num_inputs] if isinstance(a, torch.fx.Node)]

                    # Select f16 shader variant if available and input is float16.
                    if input_tensors and input_tensors[0].dtype == torch.float16:
                        f16_key = f"{op_name}_f16"
                        if f16_key in _pipeline_cache:
                            pipeline_handle, _ = _pipeline_cache[f16_key]
                            log.debug("Using f16 shader for %s", op_name)

                    if op_name in _MATMUL_OPS:
                        a, b = input_tensors[0], input_tensors[1]
                        m, k = a.shape
                        k2, n = b.shape
                        output = torch.empty((m, n), dtype=a.dtype, device=a.device)
                        push_data = struct.pack("<III", m, n, k)
                        groups_x = (n + 15) // 16
                        groups_y = (m + 15) // 16
                        success = ext.dispatch(pipeline_handle, [a, b, output], (groups_x, groups_y, 1), push_data)
                    elif op_name in _BMM_OPS:
                        a, b = input_tensors[0], input_tensors[1]
                        batch, m, k = a.shape
                        batch2, k2, n = b.shape
                        output = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
                        push_data = struct.pack("<IIII", batch, m, n, k)
                        groups_x = (n + 15) // 16
                        groups_y = (m + 15) // 16
                        success = ext.dispatch(pipeline_handle, [a, b, output], (groups_x, groups_y, batch), push_data)
                    elif op_name in _TRANSPOSE_OPS:
                        a = input_tensors[0]
                        if a.dim() == 2:
                            m, n = a.shape
                            output = torch.empty((n, m), dtype=a.dtype, device=a.device)
                            push_data = struct.pack("<II", m, n)
                            groups_x = (n + 15) // 16
                            groups_y = (m + 15) // 16
                            success = ext.dispatch(pipeline_handle, [a, output], (groups_x, groups_y, 1), push_data)
                        else:
                            # Fallback for non-2D transpose
                            success = False
                    elif op_name == "threshold_backward":
                        # threshold_backward(grad_output, input, threshold)
                        grad_out = input_tensors[0]
                        inp = input_tensors[1]
                        threshold = node.args[2] if len(node.args) > 2 else 0.0
                        if isinstance(threshold, torch.fx.Node):
                            threshold = 0.0
                        output = torch.empty_like(grad_out)
                        num_elements = grad_out.numel()
                        push_data = struct.pack("<If", num_elements, float(threshold))
                        groups_x = (num_elements + 255) // 256
                        success = ext.dispatch(
                            pipeline_handle,
                            [grad_out, inp, output],
                            (groups_x, 1, 1),
                            push_data,
                        )
                    elif op_name == "_softmax":
                        # Fused softmax: one workgroup per row.
                        # _softmax(input, dim, half_to_float)
                        inp = input_tensors[0].contiguous()
                        dim_arg = node.args[1] if len(node.args) > 1 else -1
                        if isinstance(dim_arg, torch.fx.Node):
                            dim_arg = -1
                        dim_arg = dim_arg % inp.dim()
                        output = torch.empty_like(inp)
                        if dim_arg == inp.dim() - 1:
                            outer = inp.numel() // inp.shape[-1]
                            dim_size = inp.shape[-1]
                            push_data = struct.pack("<II", outer, dim_size)
                            success = ext.dispatch(
                                pipeline_handle,
                                [inp, output],
                                (outer, 1, 1),
                                push_data,
                            )
                        else:
                            success = False
                    elif op_name == "_softmax_backward_data":
                        # Fused softmax backward: one workgroup per row.
                        # _softmax_backward_data(grad_output, output, dim, input_dtype)
                        grad_out = input_tensors[0].contiguous()
                        softmax_out = input_tensors[1].contiguous()
                        dim_arg = node.args[2] if len(node.args) > 2 else -1
                        if isinstance(dim_arg, torch.fx.Node):
                            dim_arg = -1
                        if isinstance(dim_arg, int):
                            dim_arg = dim_arg % grad_out.dim()
                        else:
                            dim_arg = grad_out.dim() - 1
                        if dim_arg == grad_out.dim() - 1:
                            outer = grad_out.numel() // grad_out.shape[-1]
                            dim_size = grad_out.shape[-1]
                            output = torch.empty_like(grad_out)
                            push_data = struct.pack("<II", outer, dim_size)
                            success = ext.dispatch(
                                pipeline_handle,
                                [grad_out, softmax_out, output],
                                (outer, 1, 1),
                                push_data,
                            )
                        else:
                            success = False
                            output = torch.empty_like(grad_out)
                    else:
                        # Generic pointwise dispatch.
                        ref = input_tensors[0]
                        output = torch.empty_like(ref)

                        # Push constants: num_elements as uint32.
                        num_elements = ref.numel()
                        push_data = struct.pack("<I", num_elements)

                        # Workgroup dispatch: ceil(num_elements / 256).
                        groups_x = (num_elements + 255) // 256

                        success = ext.dispatch(
                            pipeline_handle,
                            input_tensors + [output],
                            (groups_x, 1, 1),
                            push_data,
                        )

                    if not success:
                        log.warning("Vulkan dispatch failed for %s, falling back", op_name)
                        env[node.name] = _eager_call(node, env)
                    else:
                        env[node.name] = output
                else:
                    # Unsupported op - execute eagerly.
                    env[node.name] = _eager_call(node, env)
            elif node.op == "call_method":
                env[node.name] = _eager_call(node, env)

            elif node.op == "output":
                # Return the output value(s).
                out_args = node.args[0]
                if isinstance(out_args, (tuple, list)):
                    return tuple(
                        env[a.name] if isinstance(a, torch.fx.Node) else a
                        for a in out_args
                    )
                elif isinstance(out_args, torch.fx.Node):
                    return env[out_args.name]
                return out_args

    return run


def _resolve_arg(a, env, device_ref):
    """Resolve a single FX graph argument, moving tensors to CPU."""
    if isinstance(a, torch.fx.Node):
        t = env[a.name]
        if isinstance(t, torch.Tensor):
            if device_ref[0] is None:
                device_ref[0] = t.device
            return t.to("cpu")
        return t
    elif isinstance(a, torch.Tensor):
        if device_ref[0] is None:
            device_ref[0] = a.device
        return a.to("cpu")
    elif isinstance(a, (list, tuple)):
        # Handle list-of-nodes (e.g. tensor list for cat/stack).
        resolved = [_resolve_arg(item, env, device_ref) for item in a]
        return type(a)(resolved) if isinstance(a, tuple) else resolved
    return a


def _eager_call(node: torch.fx.Node, env: dict) -> torch.Tensor:
    """Fall back to eager execution for an unsupported or failed op by using CPU."""
    device_ref = [None]  # mutable ref for nested resolution
    resolved_args = [_resolve_arg(a, env, device_ref) for a in node.args]
    device = device_ref[0]
            
    resolved_kwargs = {k: _resolve_arg(v, env, device_ref) for k, v in node.kwargs.items()}
    device = device_ref[0]
            
    if node.op == "call_method":
        method = getattr(resolved_args[0], node.target)
        result = method(*resolved_args[1:], **resolved_kwargs)
    else:
        result = node.target(*resolved_args, **resolved_kwargs)
    if isinstance(result, torch.Tensor) and device is not None:
        result = result.to(device)
    elif isinstance(result, (tuple, list)):
        result = type(result)(r.to(device) if isinstance(r, torch.Tensor) and device is not None else r for r in result)
    return result


def _compile_and_load(ext, shader_file: str) -> Optional[int]:
    """Compile a GLSL shader to SPIR-V and load it as a pipeline."""
    glsl_path = SHADER_DIR / shader_file
    if not glsl_path.exists():
        log.warning("Shader template not found: %s", glsl_path)
        return None

    spirv = compile_glsl_to_spirv(glsl_path.read_text())
    handle = ext.load_shader(spirv)
    if handle == 0:
        log.error("Failed to load shader pipeline for %s", shader_file)
        return None

    log.info("Loaded Vulkan pipeline for %s (handle=%d)", shader_file, handle)
    return handle


def _get_op_name(node: torch.fx.Node) -> str:
    target = str(node.target)
    parts = target.split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


def compile_glsl_to_spirv(glsl_source: str) -> bytes:
    """Compile GLSL compute shader source to SPIR-V using glslangValidator."""
    with tempfile.NamedTemporaryFile(suffix=".comp", mode="w", delete=False) as f:
        f.write(glsl_source)
        f.flush()
        glsl_path = f.name

    spirv_path = glsl_path + ".spv"
    result = subprocess.run(
        ["glslangValidator", "-V", glsl_path, "-o", spirv_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GLSL compilation failed:\n{result.stderr}")

    spirv_bytes = Path(spirv_path).read_bytes()
    Path(glsl_path).unlink(missing_ok=True)
    Path(spirv_path).unlink(missing_ok=True)

    return spirv_bytes


# ---------------------------------------------------------------------------
# Multi-pass reduction dispatch
# ---------------------------------------------------------------------------

WORKGROUP_SIZE = 256


def _dispatch_reduction(ext, pipeline_handle: int, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
    """Dispatch a reduction shader with multi-pass support.

    The reduction shader produces one partial sum per workgroup. If there are
    multiple workgroups, we recursively reduce the partials until we have a
    single scalar.
    """
    num_elements = input_tensor.numel()
    current = input_tensor  # Vulkan backend just reads the flat buffer via data_ptr anyway

    while num_elements > 1:
        num_groups = max(1, (num_elements + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE)
        output = torch.empty(num_groups, dtype=current.dtype, device=current.device)
        push_data = struct.pack("<I", num_elements)

        success = ext.dispatch(
            pipeline_handle,
            [current, output],
            (num_groups, 1, 1),
            push_data,
        )
        if not success:
            return None

        current = output
        num_elements = num_groups

    return current.view(())  # scalar tensor


def _dispatch_dim_reduction(
    ext, pipeline_handle: int, input_tensor: torch.Tensor,
    node_args: tuple, op_name: str,
) -> Optional[torch.Tensor]:
    """Dispatch a dim-aware reduction shader.

    Handles aten.sum.dim_IntList and aten.mean.dim by computing
    outer_size, reduce_size, inner_size from the tensor shape and
    reduction dimensions.
    """
    # Parse dim argument (could be int or list of ints).
    dims_arg = node_args[1]
    if isinstance(dims_arg, torch.fx.Node):
        # Can't resolve dynamic dims, fall back.
        return None
    if isinstance(dims_arg, int):
        dims = [dims_arg]
    elif isinstance(dims_arg, (list, tuple)):
        dims = list(dims_arg)
    else:
        return None

    shape = list(input_tensor.shape)
    ndim = len(shape)

    # Normalize negative dims.
    dims = [d % ndim for d in dims]
    dims.sort()

    # For simplicity, handle single-dim reduction (most common case).
    if len(dims) != 1:
        # Multi-dim reduction: fall back for now.
        log.debug("Multi-dim reduction not yet supported on Vulkan, falling back")
        return None

    reduce_dim = dims[0]
    reduce_size = shape[reduce_dim]
    outer_size = 1
    for i in range(reduce_dim):
        outer_size *= shape[i]
    inner_size = 1
    for i in range(reduce_dim + 1, ndim):
        inner_size *= shape[i]

    # Output shape: input shape with reduce_dim removed.
    # Check keepdim argument.
    keepdim = False
    if len(node_args) > 2:
        kd = node_args[2]
        if isinstance(kd, bool):
            keepdim = kd

    out_shape = list(shape)
    if keepdim:
        out_shape[reduce_dim] = 1
    else:
        out_shape.pop(reduce_dim)

    total_outputs = outer_size * inner_size
    output = torch.empty(out_shape, dtype=input_tensor.dtype, device=input_tensor.device)

    push_data = struct.pack("<III", outer_size, reduce_size, inner_size)
    groups_x = (total_outputs + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE

    success = ext.dispatch(
        pipeline_handle,
        [input_tensor.contiguous(), output],
        (max(1, groups_x), 1, 1),
        push_data,
    )

    if not success:
        return None

    return output
