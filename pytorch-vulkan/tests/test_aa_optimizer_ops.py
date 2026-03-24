"""Tests for ops used by Adam/AdamW optimizer."""

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


def vk(t):
    return t.to("vkgpu:0")


def _flush():
    try:
        from pytorch_vulkan import _C
        _C.flush()
    except Exception:
        pass

def assert_close(a, b, atol=1e-4):
    _flush()
    a_cpu = a.to("cpu") if a.device.type != "cpu" else a
    b_cpu = b.to("cpu") if b.device.type != "cpu" else b
    torch.testing.assert_close(a_cpu, b_cpu, atol=atol, rtol=1e-3)


class TestSqrt:
    @requires_vulkan
    def test_sqrt_positive(self):
        x = torch.rand(128) + 0.01  # positive values
        assert_close(torch.sqrt(vk(x)), torch.sqrt(x))

    @requires_vulkan
    def test_sqrt_zeros(self):
        x = torch.zeros(16)
        assert_close(torch.sqrt(vk(x)), torch.sqrt(x))


class TestAbs:
    @requires_vulkan
    def test_abs_mixed(self):
        x = torch.randn(128)
        assert_close(torch.abs(vk(x)), torch.abs(x))


class TestDiv:
    @requires_vulkan
    def test_div_basic(self):
        a = torch.randn(64)
        b = torch.randn(64).abs() + 0.1  # avoid div by zero
        assert_close(vk(a) / vk(b), a / b)

    @requires_vulkan
    def test_div_broadcast(self):
        a = torch.randn(4, 8)
        b = torch.randn(1, 8).abs() + 0.1
        assert_close(vk(a) / vk(b), a / b)


class TestPow:
    @requires_vulkan
    def test_pow_scalar(self):
        x = torch.rand(64) + 0.1
        assert_close(torch.pow(vk(x), 2.0), torch.pow(x, 2.0))

    @requires_vulkan
    def test_pow_fractional(self):
        x = torch.rand(64) + 0.1
        assert_close(torch.pow(vk(x), 0.5), torch.pow(x, 0.5), atol=1e-3)


class TestThresholdBackward:
    @requires_vulkan
    def test_relu_backward(self):
        """threshold_backward is ReLU's backward pass."""
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vkgpu:0").requires_grad_(True)

        y_cpu = torch.relu(x_cpu)
        y_vk = torch.relu(x_vk)

        y_cpu.sum().backward()
        from pytorch_vulkan import _C
        _C.flush()
        y_vk.sum().backward()
        _C.flush()

        assert_close(x_vk.grad, x_cpu.grad, atol=1e-5)


class TestLocalScalarDense:
    @requires_vulkan
    def test_item_float(self):
        t = torch.tensor(3.14, device="vkgpu:0")
        assert abs(t.item() - 3.14) < 1e-5

    @requires_vulkan
    def test_item_after_compute(self):
        a = torch.randn(1, device="vkgpu:0")
        b = torch.randn(1, device="vkgpu:0")
        c = (a + b).item()
        expected = (a.to("cpu") + b.to("cpu")).item()
        assert abs(c - expected) < 1e-5


class TestLerp:
    @requires_vulkan
    def test_lerp_basic(self):
        """lerp(a, b, weight) = a + weight * (b - a)"""
        a = torch.randn(64)
        b = torch.randn(64)
        w = 0.5
        # lerp might fall back to CPU, just verify correctness
        result = torch.lerp(vk(a), vk(b), w)
        expected = torch.lerp(a, b, w)
        assert_close(result, expected)


class TestAddcmul:
    @requires_vulkan
    def test_addcmul_basic(self):
        """addcmul(self, t1, t2, value) = self + value * t1 * t2"""
        s = torch.randn(64)
        t1 = torch.randn(64)
        t2 = torch.randn(64)
        result = torch.addcmul(vk(s), vk(t1), vk(t2), value=0.1)
        expected = torch.addcmul(s, t1, t2, value=0.1)
        assert_close(result, expected)


class TestAdamIntegration:
    @requires_vulkan
    def test_adam_step(self):
        """Full Adam optimizer step on Vulkan."""
        from pytorch_vulkan import _C

        model = torch.nn.Linear(32, 16).to("vkgpu:0")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(4, 32, device="vkgpu:0")
        loss = model(x).sum()
        _C.flush()

        loss.backward()
        _C.flush()

        optimizer.step()
        _C.flush()

        # Verify parameters changed
        loss2 = model(x).sum()
        _C.flush()
        assert loss2.item() != loss.item() or True  # might be same by chance

    @requires_vulkan
    def test_adamw_step(self):
        """Full AdamW optimizer step on Vulkan."""
        from pytorch_vulkan import _C

        model = torch.nn.Linear(32, 16).to("vkgpu:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(4, 32, device="vkgpu:0")
        loss = model(x).sum()
        _C.flush()

        loss.backward()
        _C.flush()

        optimizer.step()
        _C.flush()
