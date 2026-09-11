#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
image=${HT_XPU_IMAGE:-ht-pytorch-xpu:ubuntu24.04-oneapi2025.3}
render_node=$(find /dev/dri -maxdepth 1 -name 'renderD*' -print -quit 2>/dev/null || true)

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
            --volume="${repo_root}:/workspace" \
            --workdir=/workspace \
            "${image}" "${@:1}"
        ;;
esac
