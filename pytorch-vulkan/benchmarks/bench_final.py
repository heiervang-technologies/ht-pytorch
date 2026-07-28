import copy
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# Import our backend
try:
    import pytorch_vulkan

    pytorch_vulkan.init()
    HAS_VULKAN = pytorch_vulkan.is_available()
except ImportError:
    HAS_VULKAN = False

# Import LlamaBlock from our existing benchmark
try:
    from benchmarks.bench_llama import LlamaBlock
except ImportError:
    LlamaBlock = None


def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "vkgpu":
        if HAS_VULKAN and hasattr(pytorch_vulkan, "_C"):
            pytorch_vulkan._C.flush()


def bench(name, fn, device_type, warmup=10, iters=50):
    try:
        for _ in range(warmup):
            fn()
        sync_device(device_type)

        timings = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            sync_device(device_type)
            timings.append((time.perf_counter() - start) * 1000)

        return statistics.median(timings)
    except Exception as e:
        return float("nan")


def run_suite():
    devices = [("CPU", "cpu")]
    if torch.cuda.is_available():
        devices.append(("CUDA", "cuda"))
    if HAS_VULKAN:
        devices.append(("Vulkan", "vkgpu:0"))

    results = {name: {} for name, _ in devices}

    print("=" * 90)
    print("PyTorch Vulkan vs CUDA vs CPU - Final Benchmarks (Median of 50 iters)")
    print("=" * 90)

    for dev_name, dev_str in devices:
        print(f"Running on {dev_name}...")
        dtype = torch.float32

        # 1. Pointwise Add (1M)
        a = torch.randn(1_000_000, device=dev_str, dtype=dtype)
        b = torch.randn(1_000_000, device=dev_str, dtype=dtype)
        results[dev_name]["Add (1M)"] = bench(
            "Add", lambda: torch.add(a, b), dev_str.split(":")[0]
        )

        # 2. Matmul (256, 512, 1024)
        m256_1 = torch.randn(256, 256, device=dev_str, dtype=dtype)
        m256_2 = torch.randn(256, 256, device=dev_str, dtype=dtype)
        results[dev_name]["MM (256x256)"] = bench(
            "MM 256", lambda: torch.mm(m256_1, m256_2), dev_str.split(":")[0]
        )

        m512_1 = torch.randn(512, 512, device=dev_str, dtype=dtype)
        m512_2 = torch.randn(512, 512, device=dev_str, dtype=dtype)
        results[dev_name]["MM (512x512)"] = bench(
            "MM 512", lambda: torch.mm(m512_1, m512_2), dev_str.split(":")[0]
        )

        m1024_1 = torch.randn(1024, 1024, device=dev_str, dtype=dtype)
        m1024_2 = torch.randn(1024, 1024, device=dev_str, dtype=dtype)
        results[dev_name]["MM (1024x1024)"] = bench(
            "MM 1024", lambda: torch.mm(m1024_1, m1024_2), dev_str.split(":")[0]
        )

        # 3. Softmax (32x256)
        s1 = torch.randn(32, 256, device=dev_str, dtype=dtype)
        results[dev_name]["Softmax (32x256)"] = bench(
            "Softmax", lambda: F.softmax(s1, dim=-1), dev_str.split(":")[0]
        )

        # SDPA is not explicitly requested in the prompt, but it's a key op, I'll add it anyway just in case.
        # Prompt said: "1) pointwise add 1M elements, 2) mm 256x256, 512x512, 1024x1024, 3) softmax 32x256, 4) LlamaBlock f16, 5) training step with Adam."

        # 4. LlamaBlock f16
        if LlamaBlock is not None:
            B, S, H, D = 1, 256, 16, 64
            hidden_size = H * D
            intermediate_size = hidden_size * 4

            block = LlamaBlock(hidden_size, H, intermediate_size).to(torch.float16)
            block_dev = copy.deepcopy(block).to(dev_str)

            x_f16 = torch.randn(B, S, hidden_size, dtype=torch.float16, device=dev_str)
            cos_f16 = torch.randn(S, D // 2, dtype=torch.float16, device=dev_str)
            sin_f16 = torch.randn(S, D // 2, dtype=torch.float16, device=dev_str)

            # Use smaller iteration count for LLama to avoid timeouts
            results[dev_name]["LLaMA Block (f16)"] = bench(
                "LLaMA",
                lambda: block_dev(x_f16, cos_f16, sin_f16),
                dev_str.split(":")[0],
                warmup=2,
                iters=10,
            )

        # 5. Adam Training Step
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).to(
            dev_str
        )
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        x_tr = torch.randn(32, 128, device=dev_str)
        y_tr = torch.randn(32, 128, device=dev_str)

        def train_step():
            opt.zero_grad()
            out = model(x_tr)
            loss = F.mse_loss(out, y_tr)
            loss.backward()
            opt.step()

        results[dev_name]["Adam Train Step"] = bench(
            "Adam", train_step, dev_str.split(":")[0], warmup=2, iters=10
        )

    # Print Table
    print("\n" + "=" * 90)
    header = f"{'Operation':<25} | {'CPU (ms)':<15} | {'CUDA (ms)':<15} | {'Vulkan (ms)':<15}"
    print(header)
    print("-" * 90)

    ops = list(results["CPU"].keys())
    for op in ops:
        cpu_time = results.get("CPU", {}).get(op, float("nan"))
        cuda_time = results.get("CUDA", {}).get(op, float("nan"))
        vk_time = results.get("Vulkan", {}).get(op, float("nan"))

        cpu_str = f"{cpu_time:.3f}" if cpu_time == cpu_time else "FAIL"
        cuda_str = f"{cuda_time:.3f}" if cuda_time == cuda_time else "FAIL"
        vk_str = f"{vk_time:.3f}" if vk_time == vk_time else "FAIL"

        print(f"{op:<25} | {cpu_str:<15} | {cuda_str:<15} | {vk_str:<15}")
    print("=" * 90)


if __name__ == "__main__":
    run_suite()
