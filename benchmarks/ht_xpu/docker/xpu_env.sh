#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
image=${HT_XPU_IMAGE:-ht-pytorch-xpu:ubuntu24.04-oneapi2025.3}
venv_volume=${HT_XPU_VENV_VOLUME:-ht-pytorch-xpu-venv}
pip_cache_volume=${HT_XPU_PIP_CACHE_VOLUME:-ht-pytorch-xpu-pip-cache}
render_node=$(find /dev/dri -maxdepth 1 -name 'renderD*' -print -quit 2>/dev/null || true)
hf_cache=${HF_HOME:-${HOME}/.cache/huggingface}
mkdir -p "${hf_cache}"

cache_args=(
    --env=HF_HOME=/hf-cache
    --env=HF_HUB_DISABLE_XET=1
    --env=HF_HUB_DOWNLOAD_TIMEOUT=300
    --volume="${hf_cache}:/hf-cache"
)
if [[ -n ${HF_TOKEN:-} ]]; then
    cache_args+=(--env=HF_TOKEN)
fi

if [[ ${1:-shell} != build-image && -z ${render_node} ]]; then
    echo "no /dev/dri render node found" >&2
    exit 1
fi

case ${1:-shell} in
    build-image)
        docker build --tag "${image}" "${repo_root}/benchmarks/ht_xpu/docker"
        ;;
    shell)
        exec docker run --rm -it \
            --device=/dev/dri \
            --group-add="$(stat -c '%g' "${render_node}")" \
            --ipc=host \
            --user="$(id -u):$(id -g)" \
            --env=HOME=/tmp \
            --env=PIP_CACHE_DIR=/tmp/.cache/pip \
            "${cache_args[@]}" \
            --volume="${venv_volume}:/opt/venv" \
            --volume="${pip_cache_volume}:/tmp/.cache/pip" \
            --volume="${repo_root}:/workspace" \
            --workdir=/workspace \
            "${image}"
        ;;
    *)
        exec docker run --rm \
            --device=/dev/dri \
            --group-add="$(stat -c '%g' "${render_node}")" \
            --ipc=host \
            --user="$(id -u):$(id -g)" \
            --env=HOME=/tmp \
            --env=PIP_CACHE_DIR=/tmp/.cache/pip \
            "${cache_args[@]}" \
            --volume="${venv_volume}:/opt/venv" \
            --volume="${pip_cache_volume}:/tmp/.cache/pip" \
            --volume="${repo_root}:/workspace" \
            --workdir=/workspace \
            "${image}" "${@:1}"
        ;;
esac
