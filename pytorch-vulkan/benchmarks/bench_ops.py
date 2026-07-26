"""Benchmarks comparing Vulkan backend vs CPU vs CUDA for key operations.

Run with: python benchmarks/bench_ops.py
"""

import time

import torch
import torch.nn.functional as F


def bench(name, fn, warmup=5, iters=50):
    """Benchmark a function with warmup and timing."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / iters * 1000  # ms
    print(f"  {name:40s} {elapsed:8.3f} ms")
    return elapsed


def bench_matmul(device, M=512, K=512, N=512):
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    fn = lambda: torch.mm(a, b)
    if device == "vkgpu:0":
        fn = torch.compile(fn, backend="vulkan")
    return bench(f"mm({M}x{K} @ {K}x{N}) [{device}]", fn)


def bench_softmax(device, B=32, S=128):
    x = torch.randn(B, S, device=device)
    fn = lambda: F.softmax(x, dim=-1)
    if device == "vkgpu:0":
        fn = torch.compile(fn, backend="vulkan")
    return bench(f"softmax({B}x{S}) [{device}]", fn)


def bench_add(device, N=1_000_000):
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    fn = lambda: a + b
    if device == "vkgpu:0":
        fn = torch.compile(fn, backend="vulkan")
    return bench(f"add({N}) [{device}]", fn)


def bench_sdpa(device, B=4, H=8, S=64, D=64):
    q = torch.randn(B, H, S, D, device=device)
    k = torch.randn(B, H, S, D, device=device)
    v = torch.randn(B, H, S, D, device=device)
    if device == "vkgpu:0":
        from pytorch_vulkan import vulkan_sdpa

        return bench(
            f"sdpa(B={B},H={H},S={S},D={D}) [{device}]", lambda: vulkan_sdpa(q, k, v)
        )
    else:
        return bench(
            f"sdpa(B={B},H={H},S={S},D={D}) [{device}]",
            lambda: F.scaled_dot_product_attention(q, k, v),
        )


def bench_transformer_step(device, d_model=128, nhead=4, seq_len=32, batch=8):
    import torch.nn as nn

    model = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.0, batch_first=True
    ).to(device)
    x = torch.randn(batch, seq_len, d_model, device=device)

    def step():
        out = model(x)
        return out

    return bench(f"encoder_layer(B={batch},S={seq_len},d={d_model}) [{device}]", step)


def main():
    from pytorch_vulkan import init

    init()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    devices.append("vkgpu:0")

    print("=" * 60)
    print("PyTorch Vulkan Backend Benchmarks")
    print("=" * 60)

    for category, bench_fn in [
        ("Element-wise Add (1M elements)", lambda d: bench_add(d)),
        ("Matrix Multiply (512x512)", lambda d: bench_matmul(d)),
        ("Softmax (32x128)", lambda d: bench_softmax(d)),
        ("SDPA (B=4, H=8, S=64, D=64)", lambda d: bench_sdpa(d)),
        ("Transformer Encoder Layer", lambda d: bench_transformer_step(d)),
    ]:
        print(f"\n--- {category} ---")
        results = {}
        for device in devices:
            try:
                results[device] = bench_fn(device)
            except Exception as e:
                print(f"  {'[' + device + ']':>42s} FAILED: {e}")

        if "cpu" in results and "vkgpu:0" in results:
            speedup = results["cpu"] / results["vkgpu:0"]
            print(f"  {'Vulkan vs CPU speedup:':>42s} {speedup:.2f}x")
        if "cuda:0" in results and "vkgpu:0" in results:
            ratio = results["vkgpu:0"] / results["cuda:0"]
            print(f"  {'Vulkan/CUDA ratio:':>42s} {ratio:.2f}x slower")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
