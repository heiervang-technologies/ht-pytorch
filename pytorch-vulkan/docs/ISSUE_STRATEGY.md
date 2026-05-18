# Discussion: Long-Term Strategy for PyTorch Vulkan Backend (Out-of-Tree vs. Fork)

As we have successfully achieved end-to-end training and inference (e.g., Tiny LLaMA) using our custom Vulkan compute backend, we are hitting the ceiling of what is possible as a pure out-of-tree `PrivateUse1` extension. We need to decide on our long-term architectural strategy.

## Option 1: Stay Purely Out-of-Tree (Current Approach)
**Pros:**
* **Velocity:** Extremely fast iteration cycle. We own the repo and don't have to wait on upstream PR reviews.
* **Maintainability:** Easier to maintain across PyTorch versions since we only rely on the public C++ extension API.
* **Independence:** No upstream merge politics.

**Cons:**
* **Autograd Limitations:** We cannot easily register custom autograd kernels for fused operations (like our optimized SDPA or RMSNorm) because `PrivateUse1` does not cleanly expose the full autograd dispatch keys without extensive hacks.
* **Graph Breaks:** CPU fallbacks silently break the autograd graph for unregistered operations, causing frustrating bugs (like our recent `native_rms_norm` tracing issues).
* **Internals Access:** We are blocked from utilizing PyTorch's internal memory management (caching allocators) and graph optimization passes (like inductor fusions specific to our hardware).

## Option 2: Hard Fork PyTorch
**Pros:**
* **First-Class Citizen:** We can add Vulkan as a proper backend alongside CUDA and MPS.
* **Full Dispatch & Autograd:** Complete access to register at all dispatch keys (including `AutogradVulkan`), entirely bypassing the `PrivateUse1` limitations.
* **Deep Integration:** Direct access to internal graph optimizers, memory pools, and the `torch.compile` stack.

**Cons:**
* **Maintenance Nightmare:** Keeping a hard fork synced with PyTorch `main` is a massive, ongoing engineering burden.
* **Adoption Friction:** Users are highly unlikely to install a custom fork of PyTorch just to get Vulkan support.
* **Historical Precedent:** PyTorch already attempted an in-tree Vulkan backend around version 1.7, which was eventually abandoned due to maintenance costs and shifting priorities.

## Option 3: The Hybrid Approach (Recommended)
**Stay out-of-tree, but contribute strategic upstream patches.**

Instead of forking, we maintain our `pytorch-vulkan` repository as an out-of-tree extension but actively submit targeted PRs to PyTorch upstream to improve the `PrivateUse1` API surface. 

**Why this is the best path forward:**
1. **Targeted Upstream Fixes:** We can submit PRs to fix the specific blockers we face, such as allowing out-of-tree backends to properly register custom autograd nodes or improving the fallback mechanism so it doesn't shatter the computation graph.
2. **Preserve Velocity:** We keep our fast development speed for the core shader generation and Rust dispatch pipeline.
3. **Community Adoption:** Users can simply `pip install pytorch-vulkan` alongside their official PyTorch installation. 

### Next Steps
If we agree on the Hybrid approach, our immediate action items should be:
1. Identify the exact PyTorch C++ dispatch internals that are currently blocking our custom autograd kernels (like the fused Flash Attention backward pass).
2. Draft a minimal, non-intrusive PR to PyTorch upstream to expose these necessary hooks for `PrivateUse1`.
3. Continue optimizing our core shaders (e.g., BF16 emulation, FA2 v3) out-of-tree.

Thoughts?
