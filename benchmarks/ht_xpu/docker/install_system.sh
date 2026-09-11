#!/usr/bin/env bash
set -euo pipefail

toolkit_url=${1:?Intel Deep Learning Essentials URL is required}

apt-get update
apt-get install --no-install-recommends -y \
    ca-certificates curl gpg-agent software-properties-common \
    build-essential cmake git libopenblas-dev libssl-dev libuv1-dev \
    ninja-build numactl pciutils pkg-config python3.12 python3.12-dev \
    python3.12-venv wget

add-apt-repository -y ppa:kobuk-team/intel-graphics
apt-get update
apt-get install --no-install-recommends -y \
    clinfo intel-gsc intel-ocloc intel-opencl-icd \
    libigc-dev libigdfcl-dev libigfxcmrt-dev libze-dev libze-intel-gpu1 libze1

wget --quiet --output-document=/tmp/intel-deep-learning-essentials.sh "${toolkit_url}"
chmod +x /tmp/intel-deep-learning-essentials.sh
/tmp/intel-deep-learning-essentials.sh -a --silent --eula accept

python3.12 -m venv /opt/venv
/opt/venv/bin/python -m pip install --no-cache-dir --upgrade \
    pip==25.2 setuptools==80.9.0 wheel==0.45.1
chmod -R a+rwX /opt/venv

rm -f /tmp/intel-deep-learning-essentials.sh
apt-get clean
rm -rf /var/lib/apt/lists/*
