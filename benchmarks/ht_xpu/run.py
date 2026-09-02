from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .harness import BenchmarkConfig, BenchmarkRunner, VARIANTS, create_workload


def _workload_options(items: list[str]) -> dict[str, str]:
    options = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"workload config must be KEY=VALUE, got {item!r}")
        options[key.replace("-", "_")] = value
    return options


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an HT XPU training benchmark")
    parser.add_argument("--workload", required=True)
    parser.add_argument(
        "--workload-config", action="append", default=[], metavar="KEY=VALUE",
        help="workload-specific setting (repeatable)",
    )
    parser.add_argument(
        "--workload-module", action="append", default=[],
        help="module to import for workload registration (repeatable)",
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    builtins = {
        "convnext": "benchmarks.ht_xpu.workloads.convnext",
        "qwen35": "benchmarks.ht_xpu.qwen_sft",
    }
    if args.workload in builtins:
        importlib.import_module(builtins[args.workload])
    for module in args.workload_module:
        importlib.import_module(module)
    device = args.device or ("cpu" if args.variant == "cpu" else "xpu")
    workload = create_workload(
        args.workload, args.variant, device, _workload_options(args.workload_config)
    )
    result = BenchmarkRunner(BenchmarkConfig(
        args.variant, device, args.warmup, args.steps, args.repeats
    )).run(workload)
    result.write(args.output)
    finalize = getattr(workload, "finalize", None)
    if finalize is not None:
        finalize()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
