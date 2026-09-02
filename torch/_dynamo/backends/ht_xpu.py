"""Inductor backend tuned for fixed-shape Intel XPU training workloads."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from torch import fx
from torch._dynamo import register_backend
from torch._logging import trace_structured


@dataclass(frozen=True)
class GraphProfile:
    nodes: int
    call_function: int
    call_method: int
    call_module: int
    operators: dict[str, int]


_GRAPH_PROFILES: list[GraphProfile] = []


def _target_name(target: Any) -> str:
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", str(target))
    return f"{module}.{name}" if module else name


def profile_graph(graph_module: fx.GraphModule) -> GraphProfile:
    """Return a stable operator inventory for benchmark diagnostics."""

    counts = Counter(
        _target_name(node.target)
        for node in graph_module.graph.nodes
        if node.op == "call_function"
    )
    operations = Counter(node.op for node in graph_module.graph.nodes)
    return GraphProfile(
        nodes=len(list(graph_module.graph.nodes)),
        call_function=operations["call_function"],
        call_method=operations["call_method"],
        call_module=operations["call_module"],
        operators=dict(sorted(counts.items())),
    )


def clear_graph_profiles() -> None:
    _GRAPH_PROFILES.clear()


def get_graph_profiles() -> tuple[GraphProfile, ...]:
    return tuple(_GRAPH_PROFILES)


@register_backend(name="ht_xpu", tags=("experimental",))
def ht_xpu(
    graph_module: fx.GraphModule,
    example_inputs: list[Any],
    **kwargs: Any,
) -> Any:
    """Compile with explicit XPU autotuning while retaining Inductor semantics."""

    profile = profile_graph(graph_module)
    _GRAPH_PROFILES.append(profile)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "ht_xpu_graph_profile",
            "encoding": "json",
        },
        payload_fn=lambda: json.dumps(asdict(profile), sort_keys=True),
    )

    from torch._inductor.compile_fx import compile_fx

    options = kwargs.pop("options", None) or {}
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported ht_xpu backend arguments: {unexpected}")
    config_patches = {
        "max_autotune": True,
        "max_autotune_gemm": True,
        "coordinate_descent_tuning": True,
        "epilogue_fusion": True,
        "shape_padding": True,
    }
    config_patches.update(options)
    graph_module.graph.lint()
    graph_module.recompile()
    return compile_fx(
        graph_module,
        example_inputs,
        config_patches=config_patches,
    )
