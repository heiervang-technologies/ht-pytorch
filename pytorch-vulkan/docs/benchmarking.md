# Benchmark protocol

Performance results are publishable only when they can be reproduced and
audited for fallback and numerical correctness.

Every result must record:

- git commit and package version;
- PyTorch and Python versions;
- GPU name, vendor/device IDs, driver version, and Vulkan API version;
- queried capability dictionary;
- dtype, complete input shapes, strides, and relevant operator arguments;
- warm-up count, measured iteration count, and statistic reported;
- explicit synchronization immediately before and after the measured region;
- CPU fallback count and operation histogram;
- maximum absolute and relative error against the reference.

Example measurement structure:

```python
pytorch_vulkan.reset_fallback_stats()
for _ in range(warmup):
    operation()
torch.vkgpu.synchronize()

start = time.perf_counter()
for _ in range(iterations):
    result = operation()
torch.vkgpu.synchronize()
elapsed = time.perf_counter() - start

metadata = {
    "device": pytorch_vulkan.device_info(),
    "fallback": pytorch_vulkan.fallback_stats(),
    "iterations": iterations,
    "seconds": elapsed,
}
```

CPU and other accelerator baselines must use equivalent dtypes, shapes,
threading settings, warm-up, and synchronization. A result with CPU fallback
is labeled hybrid and must not be presented as Vulkan kernel performance.
