#!/usr/bin/env bash
set -euo pipefail

python=${HT_XPU_PYTHON:-/opt/venv/bin/python}
vision_commit=499ca5103b5c6abdf1973651d6eb3db9dfecdfbd
vision_sha256=5303130a7a4c13bb349eb6046686a0f7bc739bf3b6b59935ca0b6568022e1470

source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
"${python}" -m pip install --no-deps \
    -r benchmarks/ht_xpu/requirements-vision.txt

temporary_directory=
cleanup() {
    if [[ -n ${temporary_directory} ]]; then
        rm -rf -- "${temporary_directory}"
    fi
}
trap cleanup EXIT

if [[ -n ${HT_XPU_TORCHVISION_SOURCE:-} ]]; then
    vision_source=${HT_XPU_TORCHVISION_SOURCE}
else
    temporary_directory=$(mktemp -d)
    archive=${temporary_directory}/vision.tar.gz
    curl --fail --location --retry 3 --output "${archive}" \
        "https://codeload.github.com/pytorch/vision/tar.gz/${vision_commit}"
    echo "${vision_sha256}  ${archive}" | sha256sum --check
    tar --extract --gzip --file "${archive}" --directory "${temporary_directory}"
    vision_source=${temporary_directory}/vision-${vision_commit}
fi

MAX_JOBS=${MAX_JOBS:-4} "${python}" -m pip install \
    --no-deps --no-build-isolation --force-reinstall "${vision_source}"
"${python}" -c 'import torch, torchvision; assert torchvision.extension._has_ops(); assert hasattr(torch.ops.torchvision, "nms")'
