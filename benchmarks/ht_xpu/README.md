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
