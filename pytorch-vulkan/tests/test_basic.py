"""Basic smoke tests for the Vulkan backend."""

import pytest
import torch


def requires_vulkan(fn):
    try:
        from pytorch_vulkan import init, is_available
        init()
        available = is_available()
    except ImportError:
        available = False
    return pytest.mark.skipif(not available, reason="Vulkan backend not available")(fn)


@requires_vulkan
def test_device_available():
    from pytorch_vulkan import is_available, device_count, device_name
    assert is_available()
    assert device_count() >= 1
    assert len(device_name()) > 0


@requires_vulkan
def test_empty_tensor():
    t = torch.empty(4, 4, device="vkgpu:0")
    assert t.device.type == "vkgpu"
    assert t.shape == (4, 4)


@requires_vulkan
def test_cpu_to_vulkan_roundtrip():
    cpu_tensor = torch.randn(3, 3)
    vulkan_tensor = cpu_tensor.to("vkgpu:0")
    back = vulkan_tensor.to("cpu")
    assert torch.allclose(cpu_tensor, back)


@requires_vulkan
def test_torch_compile_basic():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x, y):
        return x + y

    x = torch.randn(4, device="vkgpu:0")
    y = torch.randn(4, device="vkgpu:0")
    result = fn(x, y)
    assert result.shape == (4,)

@requires_vulkan
def test_torch_compile_unary_ops():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.neg(x)

    x = torch.randn(64, device="vkgpu:0")
    result = fn(x)
    expected = -x.to("cpu")
    assert torch.allclose(result.to("cpu"), expected, atol=1e-5)


@requires_vulkan
def test_torch_compile_sigmoid():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.sigmoid(x)

    x = torch.randn(128, device="vkgpu:0")
    result = fn(x)
    expected = torch.sigmoid(x.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, atol=1e-5)


@requires_vulkan
def test_torch_compile_sum():
    """Test parallel reduction sum."""
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.sum(x)

    x = torch.randn(1024, device="vkgpu:0")
    result = fn(x)
    expected = torch.sum(x.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, atol=1e-3)


@requires_vulkan
def test_torch_compile_sum_large():
    """Test multi-pass reduction with >256 workgroups."""
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.sum(x)

    # 100K elements = ~391 workgroups, needs 2 passes.
    x = torch.randn(100_000, device="vkgpu:0")
    result = fn(x)
    expected = torch.sum(x.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, atol=1e-1)


@requires_vulkan
def test_torch_compile_mean():
    """Test parallel reduction mean."""
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.mean(x)

    x = torch.randn(2048, device="vkgpu:0")
    result = fn(x)
    expected = torch.mean(x.to("cpu"))
    assert torch.allclose(result.to("cpu"), expected, atol=1e-3)


@requires_vulkan
def test_torch_compile_matmul():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(a, b):
        return torch.mm(a, b)

    a = torch.randn(16, 32, device="vkgpu:0")
    b = torch.randn(32, 16, device="vkgpu:0")
    result = fn(a, b)
    assert result.shape == (16, 16)

@requires_vulkan
def test_torch_compile_transpose():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return x.t()

    x = torch.randn(10, 20, device="vkgpu:0")
    result = fn(x)
    assert result.shape == (20, 10)

@requires_vulkan
def test_torch_compile_ones_like():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(x):
        return torch.ones_like(x)

    x = torch.randn(15, device="vkgpu:0")
    result = fn(x)
    assert result.shape == (15,)

@requires_vulkan
def test_torch_compile_bmm():
    from pytorch_vulkan import register
    register()

    @torch.compile(backend="vulkan")
    def fn(a, b):
        return torch.bmm(a, b)

    a = torch.randn(4, 16, 32, device="vkgpu:0")
    b = torch.randn(4, 32, 16, device="vkgpu:0")
    result = fn(a, b)
    assert result.shape == (4, 16, 16)

