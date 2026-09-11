from __future__ import annotations

import platform
import subprocess
from collections.abc import Iterable
from typing import Any

import torch


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_metadata() -> dict[str, Any]:
    xpu = getattr(torch, "xpu", None)
    available = bool(xpu and xpu.is_available())
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_revision": _git_revision(),
        "xpu_available": available,
        "xpu_device_count": xpu.device_count() if available else 0,
        "xpu_name": xpu.get_device_name(0) if available else None,
    }


def synchronize(device: str) -> None:
    if device.startswith("xpu"):
        if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
            raise RuntimeError("XPU requested but torch.xpu is unavailable")
        torch.xpu.synchronize()


def reset_peak_memory(device: str) -> None:
    if device.startswith("xpu") and torch.xpu.is_available():
        torch.xpu.reset_peak_memory_stats()


def memory_metrics(device: str) -> dict[str, int]:
    if device.startswith("xpu") and torch.xpu.is_available():
        return {
            "allocated_bytes": torch.xpu.memory_allocated(),
            "reserved_bytes": torch.xpu.memory_reserved(),
            "max_allocated_bytes": torch.xpu.max_memory_allocated(),
            "max_reserved_bytes": torch.xpu.max_memory_reserved(),
        }
    return {}


def assert_tensor_residency(objects: Iterable[Any], expected_device: str) -> None:
    expected = torch.device(expected_device).type
    failures: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != expected:
                failures.append(f"{path}: {value.device}")
            gradient = getattr(value, "grad", None)
            if gradient is not None:
                visit(gradient, f"{path}.grad")
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, torch.nn.Module):
            for name, item in value.named_parameters():
                visit(item, f"{path}.parameter.{name}")
            for name, item in value.named_buffers():
                visit(item, f"{path}.buffer.{name}")
        elif isinstance(value, torch.optim.Optimizer):
            visit(value.state, f"{path}.state")

    for index, obj in enumerate(objects):
        visit(obj, f"object[{index}]")
    if failures:
        raise RuntimeError("tensor residency check failed: " + "; ".join(failures[:20]))


class CompileDiagnostics:
    """Capture Dynamo/Inductor counters without depending on their presence."""

    def __enter__(self) -> "CompileDiagnostics":
        self._before = self._snapshot()
        return self

    def __exit__(self, *_: object) -> None:
        after = self._snapshot()
        self.delta = {key: after.get(key, 0) - self._before.get(key, 0) for key in after}

    @staticmethod
    def _snapshot() -> dict[str, int]:
        try:
            from torch._dynamo.utils import counters
            return {
                "unique_graphs": int(counters["stats"]["unique_graphs"]),
                "graph_breaks": int(sum(counters["graph_break"].values())),
                "recompiles": int(sum(counters["recompiles"].values())),
            }
        except (ImportError, KeyError, TypeError):
            return {"unique_graphs": 0, "graph_breaks": 0, "recompiles": 0}
