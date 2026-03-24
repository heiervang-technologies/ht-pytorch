"""Comprehensive benchmark: CPU vs CUDA vs Vulkan (RTX 3090) vs Vulkan (AMD iGPU).

Run with: python benchmarks/bench_comprehensive.py
For AMD iGPU: VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json python benchmarks/bench_comprehensive.py --igpu
"""

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def bench(name, fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1000
    return ms


def run_suite(device_name, device):
    """Run all benchmarks on a given device, return dict of results."""
    results = {}

    # 1. Element-wise add
    try:
        N = 1_000_000
        a = torch.randn(N, device=device)
        b = torch.randn(N, device=device)
        results["add_1M"] = bench(f"add({N})", lambda: a + b)
    except Exception as e:
        results["add_1M"] = f"FAIL"

    # 2. Matrix multiply
    try:
        M, K, N_ = 512, 512, 512
        a = torch.randn(M, K, device=device)
        b = torch.randn(K, N_, device=device)
        results["mm_512"] = bench("mm(512x512)", lambda: torch.mm(a, b))
    except Exception as e:
        results["mm_512"] = f"FAIL"

    # 3. Batched matmul
    try:
        a = torch.randn(8, 64, 64, device=device)
        b = torch.randn(8, 64, 64, device=device)
        results["bmm_8x64"] = bench("bmm(8,64,64)", lambda: torch.bmm(a, b))
    except Exception as e:
        results["bmm_8x64"] = f"FAIL"

    # 4. Softmax
    try:
        x = torch.randn(32, 256, device=device)
        results["softmax_32x256"] = bench("softmax(32x256)", lambda: F.softmax(x, dim=-1))
    except Exception as e:
        results["softmax_32x256"] = f"FAIL"

    # 5. SDPA (fused)
    try:
        B, H, S, D = 4, 8, 64, 64
        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        if device_name.startswith("Vulkan"):
            from pytorch_vulkan import vulkan_sdpa
            results["sdpa_4x8x64"] = bench("sdpa(4,8,64,64)", lambda: vulkan_sdpa(q, k, v))
        else:
            results["sdpa_4x8x64"] = bench("sdpa(4,8,64,64)", lambda: F.scaled_dot_product_attention(q, k, v))
    except Exception as e:
        results["sdpa_4x8x64"] = f"FAIL"

    # 6. Flash Attention 2
    try:
        B, H, S, D = 2, 4, 32, 32
        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        v = torch.randn(B, H, S, D, device=device)
        if device_name.startswith("Vulkan"):
            from pytorch_vulkan import flash_attention_vulkan
            results["fa2_2x4x32"] = bench("FA2(2,4,32,32)", lambda: flash_attention_vulkan(q, k, v))
        else:
            results["fa2_2x4x32"] = bench("FA2(2,4,32,32)", lambda: F.scaled_dot_product_attention(q, k, v))
    except Exception as e:
        results["fa2_2x4x32"] = f"FAIL"

    # 7. Sigmoid
    try:
        x = torch.randn(100_000, device=device)
        results["sigmoid_100K"] = bench("sigmoid(100K)", lambda: torch.sigmoid(x))
    except Exception as e:
        results["sigmoid_100K"] = f"FAIL"

    # 8. ReLU
    try:
        x = torch.randn(100_000, device=device)
        results["relu_100K"] = bench("relu(100K)", lambda: torch.relu(x))
    except Exception as e:
        results["relu_100K"] = f"FAIL"

    # 9. Transformer Encoder Layer (forward only)
    try:
        layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.0, batch_first=True).to(device)
        x = torch.randn(4, 32, 128, device=device)
        results["encoder_4x32x128"] = bench("encoder(4,32,128)", lambda: layer(x))
    except Exception as e:
        results["encoder_4x32x128"] = f"FAIL"

    # 10. MLP training step
    try:
        model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64)).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(16, 64, device=device)

        def train_step():
            out = model(x)
            loss = out.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

        results["mlp_train_step"] = bench("MLP train step", train_step)
    except Exception as e:
        results["mlp_train_step"] = f"FAIL"

    return results


def main():
    from pytorch_vulkan import init
    init()

    igpu_mode = "--igpu" in sys.argv
    vk_device = "vkgpu:0"

    # Determine which Vulkan GPU
    from pytorch_vulkan import device_name
    vk_name = device_name()
    is_igpu = "AMD" in vk_name or "Raphael" in vk_name or "RADV" in vk_name

    print("=" * 90)
    print("PyTorch Vulkan Backend - Comprehensive Benchmark")
    print("=" * 90)
    print(f"Vulkan GPU: {vk_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Run benchmarks
    devices = {"CPU": "cpu"}
    if torch.cuda.is_available():
        devices["CUDA RTX 3090"] = "cuda:0"
    devices[f"Vulkan ({vk_name[:30]})"] = vk_device

    all_results = {}
    for name, dev in devices.items():
        print(f"Running benchmarks on {name}...")
        all_results[name] = run_suite(name, dev)

    # Print table
    ops = list(next(iter(all_results.values())).keys())
    op_labels = {
        "add_1M": "Add (1M elem)",
        "mm_512": "MatMul (512x512)",
        "bmm_8x64": "BMM (8x64x64)",
        "softmax_32x256": "Softmax (32x256)",
        "sdpa_4x8x64": "SDPA (4,8,64,64)",
        "fa2_2x4x32": "FA2 (2,4,32,32)",
        "sigmoid_100K": "Sigmoid (100K)",
        "relu_100K": "ReLU (100K)",
        "encoder_4x32x128": "Encoder Layer",
        "mlp_train_step": "MLP Train Step",
    }

    print()
    print("=" * 90)

    # Header
    col_names = list(all_results.keys())
    header = f"{'Operation':<22s}"
    for c in col_names:
        header += f" | {c:>20s}"
    header += " | Vulkan/CUDA"
    print(header)
    print("-" * len(header))

    for op in ops:
        label = op_labels.get(op, op)
        row = f"{label:<22s}"
        vals = {}
        for name in col_names:
            v = all_results[name].get(op)
            if isinstance(v, str):
                row += f" | {'FAIL':>20s}"
            else:
                row += f" | {v:>17.3f} ms"
                vals[name] = v

        # Vulkan/CUDA ratio
        vk_key = [k for k in col_names if k.startswith("Vulkan")][0]
        cuda_key = [k for k in col_names if k.startswith("CUDA")]
        if cuda_key and vk_key in vals and cuda_key[0] in vals:
            ratio = vals[vk_key] / vals[cuda_key[0]]
            row += f" | {ratio:>8.1f}x"
        else:
            row += f" | {'N/A':>8s}"

        print(row)

    print("=" * 90)
    print()
    print("Note: Vulkan dispatch overhead dominates for small ops.")
    print("SDPA and FA2 use fused kernels that bypass per-op dispatch.")


if __name__ == "__main__":
    main()
