#!/usr/bin/env bash
set -euo pipefail

mode=${1:-official}
torch_version=${HT_XPU_TORCH_VERSION:-2.13.0}
python=/opt/venv/bin/python

source /opt/intel/oneapi/setvars.sh --force

case "${mode}" in
    official)
        "${python}" -m pip install --no-cache-dir \
            --index-url https://download.pytorch.org/whl/xpu \
            "torch==${torch_version}"
        "${python}" -m benchmarks.ht_xpu.doctor --require-xpu \
            --output=benchmarks/ht_xpu/results/official-wheel-doctor.json
        ;;
    fork)
        "${python}" -m pip uninstall -y torch
        "${python}" -m pip install --no-cache-dir -r requirements-build.txt
        git submodule sync --recursive
        git submodule update --init --recursive
        USE_XPU=1 USE_CUDA=0 USE_ROCM=0 USE_XCCL=0 \
            TORCH_XPU_ARCH_LIST=lnl-m \
            CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}" \
            "${python}" -m pip install -e . -v --no-build-isolation
        "${python}" -m benchmarks.ht_xpu.doctor --require-xpu \
            --output=benchmarks/ht_xpu/results/fork-doctor.json
        ;;
    *)
        echo "usage: $0 {official|fork}" >&2
        exit 2
        ;;
esac
