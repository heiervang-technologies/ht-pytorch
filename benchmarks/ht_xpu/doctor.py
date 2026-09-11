#!/usr/bin/env python3
"""Report Intel XPU readiness and exercise training-critical operations."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _command(*args: str) -> dict[str, Any]:
    if shutil.which(args[0]) is None:
        return {"available": False}
    try:
        result = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}
    return {
        "available": True,
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def _packages() -> dict[str, str]:
    names = ("torch", "pytorch-triton-xpu", "triton", "numpy", "cmake", "ninja")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _run_check(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        detail = operation()
        return {
            "name": name,
            "ok": True,
            "seconds": time.perf_counter() - started,
            "detail": detail,
        }
    except Exception as error:
        return {
            "name": name,
            "ok": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def _torch_report(run_smoke: bool) -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"importable": False, "error": f"{type(error).__name__}: {error}"}

    xpu = getattr(torch, "xpu", None)
    available = xpu is not None and xpu.is_available()
    report: dict[str, Any] = {
        "importable": True,
        "version": torch.__version__,
        "git_version": getattr(torch.version, "git_version", None),
        "xpu_runtime_version": getattr(torch.version, "xpu", None),
        "xpu_available": available,
        "xpu_device_count": xpu.device_count() if available else 0,
    }
    if not available:
        report["checks"] = []
        return report

    report["devices"] = [
        {
            "index": index,
            "name": xpu.get_device_name(index),
            "properties": str(xpu.get_device_properties(index)),
        }
        for index in range(xpu.device_count())
    ]
    if not run_smoke:
        return report

    device = torch.device("xpu")

    def bf16_gemm() -> dict[str, Any]:
        left = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
        right = torch.randn(256, 256, device=device, dtype=torch.bfloat16)
        output = left @ right
        xpu.synchronize()
        assert output.device.type == "xpu" and torch.isfinite(output).all().item()
        return {"shape": list(output.shape), "dtype": str(output.dtype)}

    def convolution_backward() -> dict[str, Any]:
        module = torch.nn.Conv2d(8, 16, 3, padding=1).to(device, torch.bfloat16)
        inputs = torch.randn(
            2, 8, 32, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        module(inputs).float().square().mean().backward()
        xpu.synchronize()
        assert inputs.grad is not None and inputs.grad.device.type == "xpu"
        assert module.weight.grad is not None and module.weight.grad.device.type == "xpu"
        return {"input_gradient": True, "weight_gradient": True}

    def sdpa_backward() -> dict[str, Any]:
        tensors = [
            torch.randn(
                2,
                4,
                64,
                32,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        output = torch.nn.functional.scaled_dot_product_attention(*tensors)
        output.float().sum().backward()
        xpu.synchronize()
        assert all(tensor.grad is not None for tensor in tensors)
        assert all(tensor.grad.device.type == "xpu" for tensor in tensors)
        return {"shape": list(output.shape), "all_gradients": True}

    def adamw_step() -> dict[str, Any]:
        module = torch.nn.Linear(64, 64).to(device, torch.bfloat16)
        # Lunar Lake must avoid AdamW's foreach path: it can allocate or execute
        # state off-device. Capturable single-tensor AdamW keeps step state on XPU.
        optimizer = torch.optim.AdamW(
            module.parameters(), lr=1e-3, foreach=False, capturable=True
        )
        inputs = torch.randn(8, 64, device=device, dtype=torch.bfloat16)
        module(inputs).float().square().mean().backward()
        optimizer.step()
        xpu.synchronize()
        state_devices = {
            value.device.type
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        }
        assert state_devices == {"xpu"}
        return {
            "optimizer_state_devices": sorted(state_devices),
            "foreach": optimizer.defaults["foreach"],
            "capturable": optimizer.defaults["capturable"],
        }

    report["checks"] = [
        _run_check("bf16_gemm", bf16_gemm),
        _run_check("convolution_backward", convolution_backward),
        _run_check("sdpa_backward", sdpa_backward),
        _run_check("adamw_step", adamw_step),
    ]
    return report


def collect(run_smoke: bool = True) -> dict[str, Any]:
    render_nodes = sorted(str(path) for path in Path("/dev/dri").glob("renderD*"))
    return {
        "schema_version": 1,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "kernel": platform.release(),
            "machine": platform.machine(),
            "container": Path("/.dockerenv").exists(),
            "render_nodes": render_nodes,
            "environment": {
                name: os.environ[name]
                for name in (
                    "ONEAPI_ROOT",
                    "TORCH_XPU_ARCH_LIST",
                    "USE_XPU",
                    "USE_CUDA",
                    "USE_ROCM",
                    "USE_XCCL",
                )
                if name in os.environ
            },
        },
        "packages": _packages(),
        "commands": {
            "lspci": _command("lspci", "-nnk"),
            "clinfo": _command("clinfo", "--list"),
            "sycl-ls": _command("sycl-ls"),
            "xpu-smi": _command("xpu-smi", "discovery", "-j"),
            "icpx": _command("icpx", "--version"),
            "intel_packages": _command(
                "dpkg-query",
                "-W",
                "-f=${Package}=${Version}\\n",
                "intel-opencl-icd",
                "libze-intel-gpu1",
                "libze1",
            ),
            "power_profile": _command("powerprofilesctl", "get"),
        },
        "torch": _torch_report(run_smoke),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    parser.add_argument("--no-smoke", action="store_true", help="only enumerate XPU")
    parser.add_argument(
        "--require-xpu",
        action="store_true",
        help="fail unless XPU exists and all requested smoke checks pass",
    )
    args = parser.parse_args()
    report = collect(run_smoke=not args.no_smoke)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.require_xpu:
        torch_report = report["torch"]
        if not torch_report.get("xpu_available"):
            return 1
        if any(not check["ok"] for check in torch_report.get("checks", ())):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
