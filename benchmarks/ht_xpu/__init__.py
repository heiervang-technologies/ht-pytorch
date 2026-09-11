"""Shared benchmark infrastructure for HT's Intel XPU training workloads."""

from .harness import BenchmarkConfig, BenchmarkRunner, Workload, WorkloadFactory
from .schema import BenchmarkResult

__all__ = ["BenchmarkConfig", "BenchmarkResult", "BenchmarkRunner", "Workload", "WorkloadFactory"]
