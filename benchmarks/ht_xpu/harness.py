from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .diagnostics import (
    CompileDiagnostics, assert_tensor_residency, machine_metadata, memory_metrics,
    reset_peak_memory, synchronize,
)
from .schema import BenchmarkResult
from .stats import summarize


VARIANTS = ("cpu", "xpu-eager", "xpu-compiled", "ht")


class Workload(Protocol):
    """Minimal plug-in contract. A step must include forward/backward/optimizer work."""

    name: str
    units_per_step: float

    def step(self) -> Any: ...
    def residency_objects(self) -> list[Any]: ...


WorkloadFactory = Callable[[str, str], Workload]
_WORKLOADS: dict[str, WorkloadFactory] = {}


def register_workload(name: str, factory: WorkloadFactory) -> None:
    if name in _WORKLOADS:
        raise ValueError(f"workload already registered: {name}")
    _WORKLOADS[name] = factory


def create_workload(name: str, variant: str, device: str) -> Workload:
    try:
        return _WORKLOADS[name](variant, device)
    except KeyError as exc:
        raise ValueError(f"unknown workload {name!r}; registered: {sorted(_WORKLOADS)}") from exc


@dataclass(frozen=True)
class BenchmarkConfig:
    variant: str
    device: str
    warmup_steps: int = 20
    measured_steps: int = 100
    repeats: int = 5
    check_residency: bool = True

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if min(self.warmup_steps, self.measured_steps, self.repeats) < 1:
            raise ValueError("warmup_steps, measured_steps and repeats must be positive")


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run(self, workload: Workload) -> BenchmarkResult:
        cfg = self.config
        if cfg.check_residency:
            assert_tensor_residency(workload.residency_objects(), cfg.device)
        for _ in range(cfg.warmup_steps):
            workload.step()
        synchronize(cfg.device)
        # Optimizer state and gradients are commonly allocated lazily on step one.
        if cfg.check_residency:
            assert_tensor_residency(workload.residency_objects(), cfg.device)
        reset_peak_memory(cfg.device)
        samples: list[float] = []
        with CompileDiagnostics() as compile_diagnostics:
            for _ in range(cfg.repeats):
                synchronize(cfg.device)
                start = time.perf_counter()
                for _ in range(cfg.measured_steps):
                    workload.step()
                synchronize(cfg.device)
                samples.append((time.perf_counter() - start) / cfg.measured_steps)
        throughput = [workload.units_per_step / value for value in samples]
        return BenchmarkResult(
            workload=workload.name, variant=cfg.variant, device=cfg.device,
            warmup_steps=cfg.warmup_steps, measured_steps=cfg.measured_steps,
            repeats=cfg.repeats, samples_seconds=samples, throughput=throughput,
            metadata=machine_metadata(),
            metrics={"latency_seconds": summarize(samples), "throughput": summarize(throughput), **memory_metrics(cfg.device)},
            diagnostics=compile_diagnostics.delta,
        )
