from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .schema import BenchmarkResult
from .stats import compare


def _result_key(result: BenchmarkResult) -> tuple[str, str, str]:
    config = json.dumps(
        result.metadata.get("benchmark_config", {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return result.workload, result.device.split(":", 1)[0], config


def validate_results(
    results: list[BenchmarkResult], baseline: str = "xpu-compiled"
) -> list[str]:
    """Return pairing/acceptance errors that would invalidate speedup claims."""

    errors: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    baselines: dict[tuple[str, str, str], list[BenchmarkResult]] = {}
    for result in results:
        identity = (*_result_key(result), result.variant)
        if identity in seen:
            errors.append(f"duplicate result for {result.workload}/{result.device}/{result.variant}")
        seen.add(identity)
        if result.variant == baseline:
            key = _result_key(result)
            baselines.setdefault(key, []).append(result)
        if len(result.samples_seconds) != result.repeats or len(result.throughput) != result.repeats:
            errors.append(f"{result.workload}/{result.variant}: sample count does not match repeats")
        if result.warmup_steps < 1 or result.measured_steps < 1:
            errors.append(f"{result.workload}/{result.variant}: invalid step counts")
        if any(not math.isfinite(value) or value <= 0 for value in result.throughput):
            errors.append(f"{result.workload}/{result.variant}: throughput must be finite and positive")
    for result in results:
        matches = baselines.get(_result_key(result), [])
        if result.variant == "ht" and len(matches) != 1:
            errors.append(
                f"{result.workload}/{result.device}: HT result requires exactly one {baseline} baseline"
            )
        elif result.variant == "ht":
            baseline_result = matches[0]
            protocol = ("warmup_steps", "measured_steps", "repeats")
            if any(getattr(result, item) != getattr(baseline_result, item) for item in protocol):
                errors.append(
                    f"{result.workload}/{result.device}: HT and baseline protocols differ"
                )
    return errors


def render(results: list[BenchmarkResult], baseline: str = "xpu-compiled", gate: float = 1.10) -> str:
    lines = ["# HT XPU benchmark report", "", "| Workload | Variant | Median throughput | p95 latency (s) | Speedup | Gate |", "|---|---:|---:|---:|---:|---:|"]
    errors = validate_results(results, baseline)
    if errors:
        raise ValueError("invalid benchmark result set: " + "; ".join(errors))
    by_workload = {_result_key(r): r for r in results if r.variant == baseline}
    acceptance: list[bool] = []
    for result in sorted(results, key=lambda item: (item.workload, item.variant)):
        speedup = gate_text = "—"
        if result.variant == "ht" and _result_key(result) in by_workload:
            comparison = compare(by_workload[_result_key(result)].throughput, result.throughput, gate=gate)
            speedup = f"{comparison.speedup:.3f}× [{comparison.ci_low:.3f}, {comparison.ci_high:.3f}]"
            gate_text = "PASS" if comparison.passes else "FAIL"
            acceptance.append(comparison.passes)
        lines.append(f"| {result.workload} | {result.variant} | {result.metrics['throughput']['median']:.3f} | {result.metrics['latency_seconds']['p95']:.6f} | {speedup} | {gate_text} |")
    verdict = "NOT EVALUATED" if not acceptance else ("PASS" if all(acceptance) else "FAIL")
    lines.extend(["", f"Overall HT acceptance: **{verdict}**"])
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
