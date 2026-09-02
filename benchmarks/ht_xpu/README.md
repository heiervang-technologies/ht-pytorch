# HT XPU benchmark harness

Workloads register a factory with `register_workload(name, factory)`. The factory
receives a variant (`cpu`, `xpu-eager`, `xpu-compiled`, or `ht`) and device, and
returns an object implementing `step()`, `residency_objects()`, `name`, and
`units_per_step`. A step must cover the complete training operation being measured.

Pass the module containing registrations using `--workload-module`. Run with
`python -m benchmarks.ht_xpu.run` and combine JSON outputs with
`python -m benchmarks.ht_xpu.report RESULTS_DIR`. XPU timing is bracketed by device
synchronization; initial compilation is excluded by warm-up and Dynamo graph-break
and recompile counters are emitted as diagnostics.

The Qwen benchmark keeps the comparison variants separate. `xpu-compiled` uses
stock `backend="inductor"`; `ht` selects `backend="ht_xpu"`, compiler-visible
Qwen normalization paths, and fixed-shape GEMM autotuning. The HT backend emits
an `ht_xpu_graph_profile` structured-trace artifact for every forward and backward
graph. This inventory is diagnostic evidence only: report speedups exclusively
from measured benchmark JSON, never from the presence of an optimization.

## Intel XPU environment

This environment targets the Intel Arc GPU in Lunar Lake (`lnl-m`). It does not
configure the Intel NPU. The host supplies the kernel driver and `/dev/dri`; the
Ubuntu 24.04 container supplies a matching Level Zero userspace and the oneAPI
compiler stack.

## Build and validate

```bash
benchmarks/ht_xpu/docker/xpu_env.sh build-image
benchmarks/ht_xpu/docker/xpu_env.sh \
  benchmarks/ht_xpu/docker/setup_python.sh official
benchmarks/ht_xpu/docker/xpu_env.sh \
  benchmarks/ht_xpu/docker/setup_python.sh fork
```

The official-wheel step is intentionally separate and must pass before building
the fork. It installs the pinned official PyTorch 2.13 XPU wheel and exercises
device enumeration, BF16 matrix multiplication, convolution backward, SDPA
backward, and an AdamW step. The fork step removes that wheel and builds this
checkout with:

```text
USE_XPU=1 USE_CUDA=0 USE_ROCM=0 USE_XCCL=0 TORCH_XPU_ARCH_LIST=lnl-m
```

Both steps write versioned JSON diagnostics under the ignored `results/`
directory. Override `HT_XPU_TORCH_VERSION` to test another official release and
`HT_XPU_IMAGE` to use another local image tag.

The wrapper mounts the host `${HF_HOME:-$HOME/.cache/huggingface}` at
`/hf-cache` and disables the Xet transport, which can stall on large anonymous
downloads. Cache and validate the real benchmark checkpoint and Dolly dataset
before a hardware run:

```bash
benchmarks/ht_xpu/docker/xpu_env.sh \
  python -m benchmarks.ht_xpu.prepare_assets
benchmarks/ht_xpu/docker/xpu_env.sh \
  python -m benchmarks.ht_xpu.prepare_assets --local-files-only
```

The second command is the offline acceptance check. Randomly initialized model
weights are not a benchmark substitute. Set `HF_TOKEN` in the invoking
environment if Hub rate limits require authentication.

For an interactive container:

```bash
benchmarks/ht_xpu/docker/xpu_env.sh shell
python -m benchmarks.ht_xpu.doctor --require-xpu
```

The invoking user needs permission to access `/dev/dri/renderD128`. The wrapper
passes its numeric group into the container. Do not install an out-of-tree
kernel driver in the container.

## ConvNeXt-Tiny

ConvNeXt-Tiny is built in. A hardware-free smoke run (with a deliberately small
batch) is:

```bash
python -m benchmarks.ht_xpu.run --workload convnext --variant cpu \
  --workload-config synthetic=true --workload-config pretrained=false \
  --workload-config batch-size=2 --warmup 1 --steps 1 --repeats 1 \
  --output results/convnext-cpu.json
```

For the benchmark-of-record, omit the workload overrides to train pretrained
ConvNeXt-Tiny on CIFAR-100 at 224px with batch 32. Use
`--workload-config checkpoint=PATH` to save after measurement and
`--workload-config resume=PATH` to restore model, AdamW state, and step count.
