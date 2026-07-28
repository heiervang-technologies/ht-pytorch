# Compatibility and release gates

## Version policy

The first versioned release targets one PyTorch minor line at a time because
the extension consumes version-sensitive C++ and Python APIs. The current
package metadata accepts `torch>=2.12.0a0,<2.13`; this is a development target
until the complete CI matrix below is green against a released build.

| Component | Current target | Evidence required for release |
| --- | --- | --- |
| PyTorch | 2.12.x | Lavapipe and all enabled hardware lanes |
| Python | 3.10-3.14 | Lavapipe matrix |
| Vulkan | 1.2 | `vulkaninfo`, shader validation, acceptance suite |
| Linux x86-64 | Alpha target | Wheel install and runtime suite |
| Other operating systems | Unsupported | Dedicated build and hardware CI |

An untested PyTorch minor must not be added to the dependency range. Patch
releases remain in range only while CI stays green.

## Device capabilities

The runtime queries feature structures with
`vkGetPhysicalDeviceFeatures2`. Tests and shader loading are gated on the
returned bits, not extension names alone.

| Capability | Use |
| --- | --- |
| `shaderFloat16` and `storageBuffer16BitAccess` | FP16 storage and kernels |
| buffer/shared float atomic add | Flash Attention backward |
| `VK_KHR_push_descriptor` and descriptor limits | Optional descriptor fast path |
| NV cooperative-matrix feature, 16x16x16 FP16 subgroup tile, subgroup size 32 | Optional NVIDIA BMM path |

Every capability has a non-specialized path or a reported fallback. Missing
optional features must not cause a mismatched shader to load.
Pipeline layouts use each shader's exact descriptor count and are rejected
when that count exceeds the device limits.

## Vendor status

| Vendor/path | Status | Production gate |
| --- | --- | --- |
| Mesa Lavapipe | Required PR correctness lane | Green validation and tests |
| NVIDIA | Provisioned-runner lane defined | Repeated green runs on named GPU/driver |
| AMD | Provisioned-runner lane defined | Repeated green runs on named GPU/driver |
| Intel | Provisioned-runner lane defined | Repeated green runs on named GPU/driver |

Defining a job is not hardware evidence. Release notes must name the exact
device, driver, Vulkan API version, capability set, and CI run.

## Numerical policy

The declarative operator registry carries per-dtype tolerances. Default
acceptance limits are:

| dtype | absolute | relative |
| --- | ---: | ---: |
| float32 | `1e-5` | `1e-5` |
| float16 | `2e-3` | `2e-3` |
| bfloat16 | `2e-2` | `2e-2` |

Matrix multiplication and full-model tests may use documented, kernel-specific
limits. A tolerance change requires an error-distribution report rather than a
single worst-case example.

## Release checklist

A release is blocked until:

1. GLSL compilation and `spirv-val` pass for every packaged shader.
2. Vulkan validation reports no errors.
3. Forward results, all gradients, updated parameters, and optimizer states
   match CPU references across multiple steps.
4. Supported reference models report zero CPU fallback.
5. Lavapipe and each advertised hardware lane are green.
6. Reproducible wheels exist for every advertised Python/platform pair.
7. Two active human maintainers approve the release.
