"""AOT Autograd integration for the experimental Vulkan FX backend."""

import logging
from collections import Counter

import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

from pytorch_vulkan.fx_backend import vulkan_fx_compiler


log = logging.getLogger(__name__)

# Accumulates backward op statistics across compilations for analysis.
_backward_op_counts: Counter = Counter()


_training_registered = False


def register_training():
    """Register the training-aware Vulkan backend with torch.compile."""
    global _training_registered
    if _training_registered:
        return
    torch._dynamo.register_backend(vulkan_train_compiler, name="vulkan_train")
    _training_registered = True
    log.info("Registered Vulkan training backend (vulkan_train)")


def vulkan_train_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    """Split a graph with AOT Autograd and compile both FX graphs."""
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=_forward_compiler,
        bw_compiler=_backward_compiler,
        decompositions=_get_decompositions(),
    )


def _forward_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    """Compile the forward graph using our Vulkan backend."""
    log.info("=== AOT Forward Graph ===")
    _log_graph_ops(gm, "forward")
    return vulkan_fx_compiler(gm, example_inputs)


def _backward_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    """Compile the backward graph.

    Logs all ops and compiles using Vulkan backend.
    """
    log.info("=== AOT Backward Graph ===")
    ops = _log_graph_ops(gm, "backward")

    # Accumulate stats for analysis.
    _backward_op_counts.update(ops)

    return vulkan_fx_compiler(gm, example_inputs)


def _log_graph_ops(gm: torch.fx.GraphModule, label: str) -> list[str]:
    """Log all ops in an FX graph and return the op names."""
    ops = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            op_name = str(node.target)
            ops.append(op_name)
            log.info(
                "  [%s] %s: %s (args=%d)", label, node.name, op_name, len(node.args)
            )
    log.info("  [%s] Total ops: %d", label, len(ops))
    return ops


def _get_decompositions():
    """Get the decomposition table for AOT Autograd.

    This tells aot_autograd how to decompose higher-level ops into primitives
    that our backend can handle. We use the core ATen decompositions which
    break things down into basic arithmetic.
    """
    from torch._decomp import get_decompositions

    # Decompose these higher-level ops into primitives we have shaders for.
    decomp_ops = [
        torch.ops.aten.relu_.default,
        torch.ops.aten.sigmoid_.default,
        torch.ops.aten.tanh_.default,
        torch.ops.aten.t.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.native_batch_norm.default,
        torch.ops.aten.native_layer_norm.default,
    ]
    return get_decompositions(decomp_ops)


def get_backward_op_report() -> str:
    """Return a human-readable report of backward ops seen so far.

    Call this after running some training steps to see which ops need
    backward SPIR-V shaders.
    """
    if not _backward_op_counts:
        return "No backward ops recorded yet. Run some training steps first."

    lines = ["Backward Op Frequency Report:", "=" * 50]
    for op, count in _backward_op_counts.most_common():
        # Simplify op names: aten.mul.Tensor -> mul
        short = op.split(".")[-2] if "." in op else op
        lines.append(f"  {short:30s}  x{count}")
    lines.append(f"\nTotal unique ops: {len(_backward_op_counts)}")
    lines.append(f"Total op calls:   {sum(_backward_op_counts.values())}")
    return "\n".join(lines)
