import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

import pytorch_vulkan


def synchronize():
    torch.vkgpu.synchronize()


def git_commit():
    for directory in Path(__file__).resolve().parents:
        if (directory / ".git").exists():
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() if result.returncode == 0 else None
    return None


def measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    synchronize()
    samples = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter_ns()
        result = operation()
        synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return result, samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not pytorch_vulkan.init():
        raise RuntimeError("Vulkan backend is unavailable")
    torch.manual_seed(0)
    left_cpu = torch.randn(args.size, dtype=torch.float32)
    right_cpu = torch.randn(args.size, dtype=torch.float32)
    left = left_cpu.to("vkgpu:0")
    right = right_cpu.to("vkgpu:0")

    pytorch_vulkan.reset_fallback_stats()
    result, samples = measure(
        lambda: left + right,
        args.warmup,
        args.iterations,
    )
    expected = left_cpu + right_cpu
    actual = result.cpu()
    absolute_error = (actual - expected).abs()
    relative_error = absolute_error / expected.abs().clamp_min(1e-12)

    report = {
        "operation": "aten.add.Tensor",
        "shape": list(left.shape),
        "strides": list(left.stride()),
        "dtype": str(left.dtype),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "latency_ms": {
            "median": statistics.median(samples),
            "minimum": min(samples),
            "maximum": max(samples),
        },
        "error": {
            "maximum_absolute": absolute_error.max().item(),
            "maximum_relative": relative_error.max().item(),
        },
        "fallback": pytorch_vulkan.fallback_stats(),
        "device": pytorch_vulkan.device_info(),
        "software": {
            "git_commit": git_commit(),
            "pytorch_vulkan": importlib.metadata.version("pytorch-vulkan"),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
