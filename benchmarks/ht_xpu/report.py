from __future__ import annotations

import argparse
from pathlib import Path

from .schema import BenchmarkResult
from .stats import compare


def render(results: list[BenchmarkResult], baseline: str = "xpu-compiled", gate: float = 1.10) -> str:
    lines = ["# HT XPU benchmark report", "", "| Workload | Variant | Median throughput | p95 latency (s) | Speedup | Gate |", "|---|---:|---:|---:|---:|---:|"]
    by_workload = {r.workload: r for r in results if r.variant == baseline}
    for result in sorted(results, key=lambda item: (item.workload, item.variant)):
        speedup = gate_text = "—"
        if result.variant != baseline and result.workload in by_workload:
            comparison = compare(by_workload[result.workload].throughput, result.throughput, gate=gate)
            speedup = f"{comparison.speedup:.3f}× [{comparison.ci_low:.3f}, {comparison.ci_high:.3f}]"
            gate_text = "PASS" if comparison.passes else "FAIL"
        lines.append(f"| {result.workload} | {result.variant} | {result.metrics['throughput']['median']:.3f} | {result.metrics['latency_seconds']['p95']:.6f} | {speedup} | {gate_text} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render HT XPU benchmark JSON files")
    parser.add_argument("path", type=Path)
    parser.add_argument("--baseline", default="xpu-compiled")
    parser.add_argument("--gate", type=float, default=1.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
    report = render([BenchmarkResult.read(path) for path in paths], args.baseline, args.gate)
    if args.output:
        args.output.write_text(report)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
