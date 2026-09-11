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
