FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG XPU_TOOLKIT_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/b3e6c1bf-a6d5-4580-8b1d-80cbfd38c8af/intel-deep-learning-essentials-2025.3.2.36_offline.sh

COPY install_system.sh /opt/ht-xpu/install_system.sh
RUN /opt/ht-xpu/install_system.sh "${XPU_TOOLKIT_URL}"

ENV PATH=/opt/venv/bin:/opt/intel/oneapi/compiler/latest/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV TORCH_XPU_ARCH_LIST=lnl-m
ENV USE_XPU=1
ENV USE_CUDA=0
ENV USE_ROCM=0
ENV USE_XCCL=0

WORKDIR /workspace
CMD ["bash"]
