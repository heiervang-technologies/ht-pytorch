"""Comprehensive FP16 (half-precision) correctness tests."""

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


def assert_close(a, b, atol=1e-3):
    a_cpu = a.to("cpu") if a.device.type != "cpu" else a
    b_cpu = b.to("cpu") if b.device.type != "cpu" else b
    torch.testing.assert_close(a_cpu, b_cpu, atol=atol, rtol=1e-2)


class TestFP16Lifecycle:
    @requires_vulkan
    def test_empty_f16(self):
        t = torch.empty(8, 8, dtype=torch.float16, device="vkgpu:0")
        assert t.dtype == torch.float16
        assert t.shape == (8, 8)

    @requires_vulkan
    def test_roundtrip_f16(self):
        cpu = torch.randn(32, dtype=torch.float16)
        back = vk(cpu).to("cpu")
        assert torch.equal(cpu, back)

    @requires_vulkan
    def test_zeros_f16(self):
        t = torch.zeros(16, dtype=torch.float16, device="vkgpu:0")
        assert t.to("cpu").sum().item() == 0.0

    @requires_vulkan
    @pytest.mark.skip(reason="fill_ uses f32 memset pattern, needs f16-aware fill")
    def test_fill_f16(self):
        t = torch.empty(16, dtype=torch.float16, device="vkgpu:0")
        t.fill_(3.14)
        assert_close(t, torch.full((16,), 3.14, dtype=torch.float16), atol=0.01)


def _flush():
    try:
        from pytorch_vulkan import _C
        _C.flush()
    except Exception:
        pass


class TestFP16BinaryOps:
    @requires_vulkan
    def test_add_f16_1d(self):
        _flush()
        a = torch.randn(65, dtype=torch.float16)
        b = torch.randn(65, dtype=torch.float16)
        assert_close(vk(a) + vk(b), a + b)

    @requires_vulkan
    def test_add_f16_2d(self):
        _flush()
        a = torch.randn(9, 9, dtype=torch.float16)
        b = torch.randn(9, 9, dtype=torch.float16)
        assert_close(vk(a) + vk(b), a + b)

    @requires_vulkan
    def test_add_f16_3d(self):
        _flush()
        a = torch.randn(3, 5, 9, dtype=torch.float16)
        b = torch.randn(3, 5, 9, dtype=torch.float16)
        assert_close(vk(a) + vk(b), a + b)

    @requires_vulkan
    def test_mul_f16_1d(self):
        _flush()
        a = torch.randn(65, dtype=torch.float16)
        b = torch.randn(65, dtype=torch.float16)
        assert_close(vk(a) * vk(b), a * b)

    @requires_vulkan
    def test_mul_f16_2d(self):
        _flush()
        a = torch.randn(9, 9, dtype=torch.float16)
        b = torch.randn(9, 9, dtype=torch.float16)
        assert_close(vk(a) * vk(b), a * b)

    @requires_vulkan
    def test_sub_f16_1d(self):
        _flush()
        a = torch.randn(65, dtype=torch.float16)
        b = torch.randn(65, dtype=torch.float16)
        assert_close(vk(a) - vk(b), a - b)

    @requires_vulkan
    def test_add_broadcast_f16(self):
        _flush()
        a = torch.randn(4, 8, dtype=torch.float16)
        b = torch.randn(1, 8, dtype=torch.float16)
        assert_close(vk(a) + vk(b), a + b)

    @requires_vulkan
    def test_mul_broadcast_f16(self):
        _flush()
        a = torch.randn(4, 8, dtype=torch.float16)
        b = torch.randn(4, 1, dtype=torch.float16)
        assert_close(vk(a) * vk(b), a * b)


class TestFP16UnaryOps:
    @requires_vulkan
    def test_relu_f16(self):
        x = torch.randn(128, dtype=torch.float16)
        assert_close(torch.relu(vk(x)), torch.relu(x))

    @requires_vulkan
    def test_neg_f16(self):
        x = torch.randn(128, dtype=torch.float16)
        assert_close(torch.neg(vk(x)), torch.neg(x))

    @requires_vulkan
    def test_sigmoid_f16(self):
        x = torch.randn(128, dtype=torch.float16)
        assert_close(torch.sigmoid(vk(x)), torch.sigmoid(x), atol=1e-2)

    @requires_vulkan
    def test_tanh_f16(self):
        x = torch.randn(128, dtype=torch.float16)
        assert_close(torch.tanh(vk(x)), torch.tanh(x), atol=1e-2)

    @requires_vulkan
    def test_exp_f16(self):
        x = torch.randn(128, dtype=torch.float16).clamp(-5, 5)
        assert_close(torch.exp(vk(x)), torch.exp(x), atol=0.05)

    @requires_vulkan
    def test_log_f16(self):
        x = (torch.rand(128, dtype=torch.float16) + 0.01)
        assert_close(torch.log(vk(x)), torch.log(x), atol=0.01)

class TestFP16Matmul:
    @requires_vulkan
    def test_mm_f16_aligned(self):
        """F16 mm via cooperative matrix (Tensor Core) - requires 16-aligned dims."""
        a = torch.randn(32, 64, dtype=torch.float16)
        b = torch.randn(64, 32, dtype=torch.float16)
        result = torch.mm(vk(a), vk(b))
        expected = torch.mm(a, b)
        assert_close(result, expected, atol=0.5)

    @requires_vulkan
    def test_mm_f16_unaligned(self):
        """F16 mm with non-aligned dims falls back to CPU."""
        a = torch.randn(8, 16, dtype=torch.float16)
        b = torch.randn(16, 8, dtype=torch.float16)
        result = torch.mm(vk(a), vk(b))
        expected = torch.mm(a, b)
        assert_close(result, expected, atol=0.1)

    @requires_vulkan
    def test_bmm_f16_aligned(self):
        """F16 bmm via cooperative matrix."""
        a = torch.randn(4, 16, 32, dtype=torch.float16)
        b = torch.randn(4, 32, 16, dtype=torch.float16)
        result = torch.bmm(vk(a), vk(b))
        expected = torch.bmm(a, b)
        assert_close(result, expected, atol=0.5)

    @requires_vulkan
    @pytest.mark.skip(reason="dtype casting between f16 and f32 on Vulkan needs _to_copy dtype support")
    def test_f16_to_f32_compute(self):
        """Test that f16 tensors can be upcasted and computed in f32."""
        x = torch.randn(32, dtype=torch.float16, device="vkgpu:0")
        y = x.float()
        assert y.dtype == torch.float32
        z = y + y
        assert_close(z, (x.to("cpu").float() + x.to("cpu").float()))
