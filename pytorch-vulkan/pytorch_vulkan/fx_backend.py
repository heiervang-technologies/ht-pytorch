"""FX graph backend for torch.compile using packaged Vulkan shaders."""

import logging
import struct
from typing import Optional

import torch

from pytorch_vulkan.device import (
    load_shader,
    shader_is_live,
)
from pytorch_vulkan.operator_registry import fx_operators, native_operators

log = logging.getLogger(__name__)

# Cache: op_name -> (pipeline_handle, num_inputs)
_pipeline_cache: dict[str, tuple[int, int]] = {}


def _ext():
    try:
        from pytorch_vulkan import _C
        return _C
    except ImportError:
        return None


_registered = False
_MAX_SHADER_INDEX = (1 << 32) - 1


def _fits_shader_u32(*values: int) -> bool:
    return all(0 <= int(value) <= _MAX_SHADER_INDEX for value in values)


def register():
    """Register the Vulkan FX backend with torch.compile."""
    global _registered
    if _registered:
        return
    torch._dynamo.register_backend(vulkan_fx_compiler, name="vulkan")
    _registered = True
    log.info("Registered Vulkan FX backend")


# ---------------------------------------------------------------------------
# Supported pointwise ops and their shader / dispatch metadata
# ---------------------------------------------------------------------------

_FX_OPERATORS = fx_operators()
_NATIVE_OPERATORS = native_operators()
_F16_VARIANTS = {
    name: spec.shaders["float16"].file
    for name, spec in _FX_OPERATORS.items()
    if "float16" in spec.shaders
}

# Maps FX op name -> (shader_file, num_inputs, num_outputs)
_POINTWISE_OPS = {
    name: (spec.shaders["float32"].file, spec.inputs, spec.outputs)
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch == "pointwise" and "float32" in spec.shaders
}

_MATMUL_OPS = {
    name: (spec.shaders["float32"].file, spec.inputs, spec.outputs)
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch == "mm" and "float32" in spec.shaders
}

_BMM_OPS = {
    name: (spec.shaders["float32"].file, spec.inputs, spec.outputs)
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch == "bmm" and "float32" in spec.shaders
}

_TRANSPOSE_OPS = {
    name: (spec.shaders["float32"].file, spec.inputs, spec.outputs)
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch == "transpose" and "float32" in spec.shaders
}

# Reduction ops need special multi-pass dispatch logic.
# Maps FX op name -> (shader_file, finalizer)
# "finalizer" is None for sum, or a callable that post-processes the scalar result.
_REDUCTION_OPS = {
    name: (spec.shaders["float32"].file, name if name == "mean" else None)
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch == "reduction" and "float32" in spec.shaders
}

# Fused ops with special dispatch (one workgroup per row).
_FUSED_OPS = {
    name: spec.shaders["float32"].file
    for name, spec in _FX_OPERATORS.items()
    if spec.dispatch in {"softmax", "softmax_backward"}
    and "float32" in spec.shaders
}

# Dim-aware reductions for backward pass shape preservation.
_DIM_REDUCTION_OPS = {
    "sum": _NATIVE_OPERATORS["sum_dim"].shaders["float32"].file,
    "mean": _NATIVE_OPERATORS["mean_dim"].shaders["float32"].file,
}


class _CompiledPartition(torch.nn.Module):
    def __init__(self, callable_):
        super().__init__()
        self.callable = callable_

    def forward(self, *args):
        return self.callable(*args)


def _is_compilable_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False
    op_name = _get_op_name(node)
    return (
        op_name in _POINTWISE_OPS
        or op_name in _MATMUL_OPS
        or op_name in _BMM_OPS
        or op_name in _TRANSPOSE_OPS
        or op_name in _REDUCTION_OPS
        or op_name in _FUSED_OPS
    )


def _partition_mixed_graph(gm: torch.fx.GraphModule):
    executable_nodes = [
        node
        for node in gm.graph.nodes
        if node.op not in {"placeholder", "get_attr", "output"}
    ]
    kinds = [_is_compilable_node(node) for node in executable_nodes]
    if not any(kinds) or all(kinds):
        return None

    partition_ids = {}
    partition_kinds = {}
    partition_id = -1
    previous_kind = None
    for node, supported in zip(executable_nodes, kinds):
        if supported != previous_kind:
            partition_id += 1
            previous_kind = supported
        partition_ids[node.name] = partition_id
        partition_kinds[partition_id] = supported

    from torch.fx.passes.split_module import split_module

    split = split_module(
        gm,
        gm,
        lambda node: partition_ids[node.name],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    for current_id, supported in partition_kinds.items():
        if not supported:
            continue
        name = f"submod_{current_id}"
        submodule = split.get_submodule(name)
        compiled = _compile_fx_graph(submodule, [])
        setattr(split, name, _CompiledPartition(compiled))
    split._vulkan_partition_kinds = tuple(partition_kinds.values())
    return split


def vulkan_fx_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    """Entry point called by torch.compile(backend='vulkan').

    Receives an FX GraphModule from Dynamo, returns a callable that
    executes the graph on Vulkan via compiled SPIR-V shaders.
    """
    partitioned = _partition_mixed_graph(gm)
    if partitioned is not None:
        return partitioned
    return _compile_fx_graph(gm, example_inputs)


def _compile_fx_graph(gm: torch.fx.GraphModule, example_inputs: list):
    ext = _ext()
    if ext is None:
        log.warning("C extension not available, falling back to eager")
        return gm

    log.info("Compiling FX graph with %d nodes for Vulkan", len(gm.graph.nodes))

    # Pre-compile all supported ops in the graph.
    for node in gm.graph.nodes:
        if node.op == "call_function":
            op_name = _get_op_name(node)
            if (
                op_name in _POINTWISE_OPS
                and (
                    op_name not in _pipeline_cache
                    or not shader_is_live(_pipeline_cache[op_name][0])
                )
            ):
                shader_file, num_inputs, _ = _POINTWISE_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _MATMUL_OPS and (
                op_name not in _pipeline_cache
                or not shader_is_live(_pipeline_cache[op_name][0])
            ):
                shader_file, num_inputs, _ = _MATMUL_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _BMM_OPS and (
                op_name not in _pipeline_cache
                or not shader_is_live(_pipeline_cache[op_name][0])
            ):
                shader_file, num_inputs, _ = _BMM_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _TRANSPOSE_OPS and (
                op_name not in _pipeline_cache
                or not shader_is_live(_pipeline_cache[op_name][0])
            ):
                shader_file, num_inputs, _ = _TRANSPOSE_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, num_inputs)
            elif op_name in _REDUCTION_OPS and (
                op_name not in _pipeline_cache
                or not shader_is_live(_pipeline_cache[op_name][0])
            ):
                shader_file, _ = _REDUCTION_OPS[op_name]
                pipeline = _compile_and_load(ext, shader_file)
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, 1)
            # Pre-compile dim-aware reduction variants.
            if op_name in _DIM_REDUCTION_OPS:
                dim_key = f"{op_name}_dim"
                if (
                    dim_key not in _pipeline_cache
                    or not shader_is_live(_pipeline_cache[dim_key][0])
                ):
                    pipeline = _compile_and_load(ext, _DIM_REDUCTION_OPS[op_name])
                    if pipeline is not None:
                        _pipeline_cache[dim_key] = (pipeline, 1)
            # Pre-compile fused ops (softmax etc).
            if op_name in _FUSED_OPS and (
                op_name not in _pipeline_cache
                or not shader_is_live(_pipeline_cache[op_name][0])
            ):
                pipeline = _compile_and_load(ext, _FUSED_OPS[op_name])
                if pipeline is not None:
                    _pipeline_cache[op_name] = (pipeline, -1)  # special dispatch
            # Pre-compile f16 shader variants.
            if op_name in _F16_VARIANTS:
                f16_key = f"{op_name}_f16"
                if (
                    f16_key not in _pipeline_cache
                    or not shader_is_live(_pipeline_cache[f16_key][0])
                ):
                    pipeline = _compile_and_load(ext, _F16_VARIANTS[op_name])
                    if pipeline is not None:
                        inputs = _POINTWISE_OPS.get(
                            op_name, (None, -1)
                        )[1]
                        _pipeline_cache[f16_key] = (pipeline, inputs)

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
                    input_arg = node.args[0]
                    input_tensor = (
                        env[input_arg.name]
                        if isinstance(input_arg, torch.fx.Node)
                        else input_arg
                    )
                    if (
                        node.kwargs
                        or not input_tensor.is_contiguous()
                        or input_tensor.storage_offset() != 0
                        or not _fits_shader_u32(input_tensor.numel())
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue

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
                        if result is not None and op_name == "mean":
                            result = result / orig_numel

                    if result is not None:
                        env[node.name] = result
                    else:
                        log.warning("Vulkan reduction failed for %s, falling back", op_name)
                        env[node.name] = _eager_call(node, env)

                elif op_name in _pipeline_cache:
                    pipeline_handle, num_inputs = _pipeline_cache[op_name]

                    if num_inputs != -1:
                        if any(
                            not isinstance(arg, torch.fx.Node)
                            for arg in node.args[:num_inputs]
                        ):
                            log.warning(
                                "Vulkan dispatch for %s requires tensor "
                                "inputs; using eager dispatch",
                                op_name,
                            )
                            env[node.name] = _eager_call(node, env)
                            continue

                    input_tensors = [
                        env[arg.name]
                        for arg in node.args[:num_inputs]
                        if isinstance(arg, torch.fx.Node)
                    ]

                    if node.kwargs:
                        env[node.name] = _eager_call(node, env)
                        continue
                    if input_tensors and any(
                        not tensor.is_contiguous()
                        or tensor.storage_offset() != 0
                        for tensor in input_tensors
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue
                    if input_tensors and any(
                        tensor.dtype != input_tensors[0].dtype
                        for tensor in input_tensors[1:]
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue
                    if input_tensors and any(
                        not _fits_shader_u32(tensor.numel())
                        for tensor in input_tensors
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue
                    if (
                        op_name in _POINTWISE_OPS
                        and input_tensors
                        and any(
                            tensor.shape != input_tensors[0].shape
                            for tensor in input_tensors[1:]
                        )
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue

                    if input_tensors and input_tensors[0].dtype == torch.float16:
                        f16_key = f"{op_name}_f16"
                        if f16_key in _pipeline_cache:
                            pipeline_handle, _ = _pipeline_cache[f16_key]
                            log.debug("Using f16 shader for %s", op_name)
                        else:
                            env[node.name] = _eager_call(node, env)
                            continue
                    elif (
                        input_tensors
                        and input_tensors[0].dtype != torch.float32
                    ):
                        env[node.name] = _eager_call(node, env)
                        continue

                    if op_name in _MATMUL_OPS:
                        a, b = input_tensors[0], input_tensors[1]
                        if (
                            a.ndim != 2
                            or b.ndim != 2
                            or a.shape[1] != b.shape[0]
                            or a.numel() == 0
                            or b.numel() == 0
                        ):
                            env[node.name] = _eager_call(node, env)
                            continue
                        m, k = a.shape
                        n = b.shape[1]
                        if not _fits_shader_u32(m, n, k, m * n):
                            env[node.name] = _eager_call(node, env)
                            continue
                        output = torch.empty((m, n), dtype=a.dtype, device=a.device)
                        push_data = struct.pack("<III", m, n, k)
                        groups_x = (n + 15) // 16
                        groups_y = (m + 15) // 16
                        success = ext.dispatch(
                            pipeline_handle,
                            [a, b, output],
                            (groups_x, groups_y, 1),
                            push_data,
                        )
                    elif op_name in _BMM_OPS:
                        a, b = input_tensors[0], input_tensors[1]
                        if (
                            a.ndim != 3
                            or b.ndim != 3
                            or a.shape[0] != b.shape[0]
                            or a.shape[2] != b.shape[1]
                            or a.numel() == 0
                            or b.numel() == 0
                        ):
                            env[node.name] = _eager_call(node, env)
                            continue
                        batch, m, k = a.shape
                        n = b.shape[2]
                        if not _fits_shader_u32(
                            batch,
                            m,
                            n,
                            k,
                            batch * m * n,
                        ):
                            env[node.name] = _eager_call(node, env)
                            continue
                        output = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
                        push_data = struct.pack("<IIII", batch, m, n, k)
                        groups_x = (n + 15) // 16
                        groups_y = (m + 15) // 16
                        success = ext.dispatch(
                            pipeline_handle,
                            [a, b, output],
                            (groups_x, groups_y, batch),
                            push_data,
                        )
                    elif op_name in _TRANSPOSE_OPS:
                        a = input_tensors[0]
                        dimensions = node.args[1:]
                        if (
                            a.dim() == 2
                            and (
                                not dimensions
                                or (
                                    len(dimensions) == 2
                                    and all(
                                        isinstance(dimension, int)
                                        for dimension in dimensions
                                    )
                                    and {
                                        dimensions[0] % 2,
                                        dimensions[1] % 2,
                                    }
                                    == {0, 1}
                                )
                            )
                        ):
                            m, n = a.shape
                            if not _fits_shader_u32(m, n):
                                env[node.name] = _eager_call(node, env)
                                continue
                            output = torch.empty((n, m), dtype=a.dtype, device=a.device)
                            push_data = struct.pack("<II", m, n)
                            groups_x = (n + 15) // 16
                            groups_y = (m + 15) // 16
                            success = ext.dispatch(
                                pipeline_handle,
                                [a, output],
                                (groups_x, groups_y, 1),
                                push_data,
                            )
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
                        if (
                            inp.numel() > 0
                            and inp.shape[-1] > 0
                            and dim_arg == inp.dim() - 1
                        ):
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
                        if (
                            grad_out.numel() > 0
                            and grad_out.shape[-1] > 0
                            and dim_arg == grad_out.dim() - 1
                        ):
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
                        if ref.numel() == 0:
                            env[node.name] = output
                            continue

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
            elif node.op == "call_module":
                args = _resolve_arg(node.args, env)
                kwargs = _resolve_arg(node.kwargs, env)
                env[node.name] = gm.get_submodule(node.target)(
                    *args, **kwargs
                )
            elif node.op == "get_attr":
                value = gm
                for field in node.target.split("."):
                    value = getattr(value, field)
                env[node.name] = value

            elif node.op == "output":
                return _resolve_arg(node.args[0], env)

    return run


def _resolve_arg(a, env):
    """Resolve a nested FX graph argument without changing its device."""
    if isinstance(a, torch.fx.Node):
        return env[a.name]
    if isinstance(a, (list, tuple)):
        resolved = [_resolve_arg(item, env) for item in a]
        return type(a)(resolved) if isinstance(a, tuple) else resolved
    if isinstance(a, dict):
        return {key: _resolve_arg(value, env) for key, value in a.items()}
    return a


def _eager_call(node: torch.fx.Node, env: dict) -> torch.Tensor:
    """Execute an uncompiled node through normal eager dispatch."""
    resolved_args = [_resolve_arg(a, env) for a in node.args]
    resolved_kwargs = {
        key: _resolve_arg(value, env)
        for key, value in node.kwargs.items()
    }
    if node.op == "call_method":
        method = getattr(resolved_args[0], node.target)
        return method(*resolved_args[1:], **resolved_kwargs)
    return node.target(*resolved_args, **resolved_kwargs)


def _compile_and_load(ext, shader_file: str) -> Optional[int]:
    """Load a packaged SPIR-V shader."""
    del ext
    return load_shader(shader_file)


def _get_op_name(node: torch.fx.Node) -> str:
    target = node.target
    if isinstance(target, str):
        return target
    schema = getattr(target, "_schema", None)
    if schema is not None:
        return schema.name.rsplit("::", 1)[-1]
    name = getattr(target, "__name__", None)
    if name:
        return name.split(".", 1)[0]
    target_text = str(target)
    parts = target_text.split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


vulkan_compiler = vulkan_fx_compiler


# ---------------------------------------------------------------------------
# Multi-pass reduction dispatch
# ---------------------------------------------------------------------------

WORKGROUP_SIZE = 256


def _dispatch_reduction(
    ext,
    pipeline_handle: int,
    input_tensor: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Dispatch a reduction shader with multi-pass support.

    The reduction shader produces one partial sum per workgroup. If there are
    multiple workgroups, we recursively reduce the partials until we have a
    single scalar.
    """
    num_elements = input_tensor.numel()
    if not _fits_shader_u32(num_elements):
        return None
    if num_elements == 0:
        return torch.zeros((), dtype=input_tensor.dtype, device=input_tensor.device)
    current = input_tensor  # Vulkan backend just reads the flat buffer via data_ptr anyway

    while num_elements > 1:
        num_groups = max(
            1,
            min(
                256,
                (num_elements + WORKGROUP_SIZE - 1)
                // WORKGROUP_SIZE,
            ),
        )
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
    if ndim == 0 or any(not isinstance(dim, int) for dim in dims):
        return None
    if any(dim < -ndim or dim >= ndim for dim in dims):
        return None

    # Normalize negative dims.
    dims = [dim % ndim for dim in dims]
    if len(dims) != len(set(dims)):
        return None
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
    if total_outputs == 0:
        return torch.empty(
            out_shape,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
    if reduce_size == 0 or not _fits_shader_u32(
        outer_size,
        reduce_size,
        inner_size,
        total_outputs,
    ):
        return None
    output = torch.empty(
        out_shape,
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )

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
