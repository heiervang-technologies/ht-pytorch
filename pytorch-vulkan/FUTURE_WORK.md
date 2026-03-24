# Future Work: Kernel Optimization

Once the base PyTorch Inductor to Vulkan SPIR-V compiler backend is functional, we plan to integrate automated kernel optimization.

## KernelAgent
Repository: [https://github.com/meta-pytorch/KernelAgent.git](https://github.com/meta-pytorch/KernelAgent.git)

**Purpose:**
We aim to use KernelAgent to run agentic optimizations on our generated Vulkan kernels. Instead of hand-tuning the SPIR-V or GLSL templates for every hardware vendor, we can leverage this agent to automatically explore and benchmark kernel optimizations specific to older consumer GPUs (AMD/Intel) that this backend targets.
