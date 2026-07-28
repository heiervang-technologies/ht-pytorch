"""Comprehensive numerical correctness tests for all Vulkan ops.

Every test compares Vulkan output against PyTorch CPU reference to validate
that our SPIR-V compute shaders produce mathematically correct results.
Tolerances are set per-op based on expected floating point behavior.
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def init_vulkan():
    from pytorch_vulkan import init

    init()


@pytest.fixture(autouse=True)
def register_backend():
    from pytorch_vulkan import register

    register()


def vk(t):
    return t.to("vkgpu:0")


def assert_close(result, expected, atol=1e-5, rtol=1e-5):
    """Assert tensors are close with detailed error on failure."""
    r = result.to("cpu") if result.device.type != "cpu" else result
    e = expected.to("cpu") if expected.device.type != "cpu" else expected
    assert r.shape == e.shape, f"Shape mismatch: {r.shape} vs {e.shape}"
    if not torch.allclose(r, e, atol=atol, rtol=rtol):
        diff = (r - e).abs()
        pytest.fail(
            f"Max abs diff: {diff.max().item():.8f}, "
            f"mean diff: {diff.mean().item():.8f}, "
            f"atol={atol}, rtol={rtol}"
        )


# =========================================================================
# Tensor lifecycle ops
# =========================================================================


class TestTensorLifecycle:
    def test_empty_creates_correct_shape(self):
        for shape in [(4,), (3, 5), (2, 3, 4), (2, 3, 4, 5)]:
            t = torch.empty(*shape, device="vkgpu:0")
            assert t.shape == shape

    def test_empty_correct_dtype(self):
        for dtype in [torch.float32, torch.float16, torch.int64, torch.int32]:
            t = torch.empty(4, device="vkgpu:0", dtype=dtype)
            assert t.dtype == dtype

    def test_copy_roundtrip_exact(self):
        cpu = torch.randn(64)
        back = vk(cpu).to("cpu")
        assert torch.equal(cpu, back), "Copy roundtrip must be bit-exact"

    def test_copy_roundtrip_2d(self):
        cpu = torch.randn(8, 16)
        back = vk(cpu).to("cpu")
        assert torch.equal(cpu, back)

    def test_copy_roundtrip_4d(self):
        cpu = torch.randn(2, 3, 4, 5)
        back = vk(cpu).to("cpu")
        assert torch.equal(cpu, back)

    def test_copy_roundtrip_f16(self):
        cpu = torch.randn(32).half()
        back = vk(cpu).to("cpu")
        assert torch.equal(cpu, back)

    def test_fill(self):
        t = torch.empty(16, device="vkgpu:0")
        t.fill_(3.14)
        assert_close(t, torch.full((16,), 3.14))

    def test_zero(self):
        t = torch.randn(16, device="vkgpu:0")
        t.zero_()
        assert_close(t, torch.zeros(16), atol=0)

    def test_normal_distribution(self):
        t = torch.empty(10000, device="vkgpu:0")
        t.normal_(0.0, 1.0)
        t_cpu = t.to("cpu")
        assert abs(t_cpu.mean().item()) < 0.1, "Mean should be near 0"
        assert abs(t_cpu.std().item() - 1.0) < 0.1, "Std should be near 1"


# =========================================================================
# View / reshape ops
# =========================================================================


class TestViewReshape:
    @pytest.mark.parametrize(
        "src,dst",
        [
            ((4, 16, 256), (-1, 256)),
            ((4, 16, 256), (64, 256)),
            ((8, 32), (-1,)),
            ((2, 3, 4), (6, 4)),
            ((2, 3, 4, 5), (2, 3, 20)),
            ((2, 3, 4, 5), (6, 20)),
        ],
    )
    def test_view_shapes(self, src, dst):
        cpu = torch.randn(*src)
        result = vk(cpu).view(*dst)
        expected = cpu.view(*dst)
        assert result.shape == expected.shape
        assert_close(result, expected, atol=0)

    def test_reshape_preserves_data(self):
        cpu = torch.randn(3, 4, 5)
        result = vk(cpu).reshape(12, 5)
        assert_close(result, cpu.reshape(12, 5), atol=0)


# =========================================================================
# Pointwise binary ops
# =========================================================================


class TestPointwiseBinary:
    @pytest.mark.parametrize("shape", [(64,), (8, 16), (4, 8, 16)])
    def test_add(self, shape):
        a, b = torch.randn(*shape), torch.randn(*shape)

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return x + y

        assert_close(fn(vk(a), vk(b)), a + b)

    @pytest.mark.parametrize("shape", [(64,), (8, 16), (4, 8, 16)])
    def test_mul(self, shape):
        a, b = torch.randn(*shape), torch.randn(*shape)

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return x * y

        assert_close(fn(vk(a), vk(b)), a * b)

    @pytest.mark.parametrize("shape", [(64,), (8, 16)])
    def test_sub(self, shape):
        a, b = torch.randn(*shape), torch.randn(*shape)

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return x - y

        assert_close(fn(vk(a), vk(b)), a - b)


# =========================================================================
# Pointwise unary ops
# =========================================================================


class TestPointwiseUnary:
    @pytest.mark.parametrize("shape", [(128,), (8, 16)])
    def test_neg(self, shape):
        a = torch.randn(*shape)

        @torch.compile(backend="vulkan")
        def fn(x):
            return -x

        assert_close(fn(vk(a)), -a)

    def test_relu(self):
        a = torch.randn(256)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.relu(x)

        assert_close(fn(vk(a)), torch.relu(a))

    def test_relu_zeros_negatives(self):
        a = torch.randn(1000)
        result = torch.relu(vk(a)).to("cpu")  # eager relu via CPU fallback
        assert (result >= 0).all()

    def test_sigmoid(self):
        a = torch.randn(256)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sigmoid(x)

        assert_close(fn(vk(a)), torch.sigmoid(a))

    def test_sigmoid_range(self):
        a = torch.randn(1000)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sigmoid(x)

        result = fn(vk(a)).to("cpu")
        assert (result >= 0).all() and (result <= 1).all()

    def test_tanh(self):
        a = torch.randn(256)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.tanh(x)

        assert_close(fn(vk(a)), torch.tanh(a))

    def test_tanh_range(self):
        a = torch.randn(1000)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.tanh(x)

        result = fn(vk(a)).to("cpu")
        assert (result >= -1).all() and (result <= 1).all()

    def test_exp(self):
        a = torch.randn(256) * 0.5  # small values to avoid overflow

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.exp(x)

        assert_close(fn(vk(a)), torch.exp(a), atol=1e-4)

    def test_log(self):
        a = torch.rand(256) + 0.01  # positive values only

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.log(x)

        assert_close(fn(vk(a)), torch.log(a), atol=1e-4)


# =========================================================================
# Reductions
# =========================================================================


class TestReductions:
    def test_sum_scalar(self):
        a = torch.randn(1024)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sum(x)

        assert_close(fn(vk(a)), torch.sum(a), atol=1e-2)

    def test_sum_large_multipass(self):
        a = torch.randn(100_000)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sum(x)

        assert_close(fn(vk(a)), torch.sum(a), atol=1.0)

    def test_mean_scalar(self):
        a = torch.randn(2048)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.mean(x)

        assert_close(fn(vk(a)), torch.mean(a), atol=1e-2)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_sum_dim(self, dim):
        a = torch.randn(8, 16)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sum(x, dim=dim)

        expected = torch.sum(a, dim=dim)
        result = fn(vk(a))
        assert result.shape == expected.shape
        assert_close(result, expected, atol=1e-3)

    def test_mean_dim(self):
        a = torch.randn(8, 16)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.mean(x, dim=-1)

        assert_close(fn(vk(a)), torch.mean(a, dim=-1), atol=1e-3)


# =========================================================================
# Matmul
# =========================================================================


class TestMatmul:
    @pytest.mark.parametrize("m,k,n", [(4, 8, 4), (16, 32, 16), (32, 64, 32)])
    def test_mm(self, m, k, n):
        a, b = torch.randn(m, k), torch.randn(k, n)

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return torch.mm(x, y)

        assert_close(fn(vk(a), vk(b)), torch.mm(a, b), atol=1e-3)

    def test_bmm(self):
        a = torch.randn(4, 16, 32)
        b = torch.randn(4, 32, 16)

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return torch.bmm(x, y)

        assert_close(fn(vk(a), vk(b)), torch.bmm(a, b), atol=1e-3)


# =========================================================================
# Softmax
# =========================================================================


class TestSoftmax:
    @pytest.mark.parametrize("shape", [(4, 8), (8, 32), (2, 4, 16)])
    def test_softmax_values(self, shape):
        a = torch.randn(*shape)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        assert_close(fn(vk(a)), torch.softmax(a, dim=-1), atol=1e-5)

    def test_softmax_properties(self):
        a = torch.randn(8, 32)

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        result = fn(vk(a)).to("cpu")
        # All positive
        assert (result > 0).all()
        # Rows sum to 1
        assert_close(result.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_softmax_numerical_stability(self):
        # Large values that would overflow without max subtraction
        a = torch.tensor([[1000.0, 1001.0, 1002.0]], device="vkgpu:0")

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        result = fn(a).to("cpu")
        assert not torch.isnan(result).any(), "Softmax should not produce NaN"
        assert not torch.isinf(result).any(), "Softmax should not produce Inf"
        assert_close(result.sum(dim=-1), torch.ones(1), atol=1e-5)


# =========================================================================
# SDPA
# =========================================================================


class TestSDPA:
    def test_sdpa_correctness(self):
        from pytorch_vulkan import vulkan_sdpa

        B, H, S, D = 2, 4, 16, 32
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        result = vulkan_sdpa(vk(q), vk(k), vk(v))
        expected = F.scaled_dot_product_attention(q, k, v)

        assert_close(result, expected, atol=1e-4)

    # @pytest.mark.skip(reason="SDPA backward produces empty gradients")
    def test_sdpa_backward_gradients(self):
        from pytorch_vulkan import vulkan_sdpa

        B, H, S, D = 1, 2, 8, 16
        q = torch.randn(B, H, S, D, device="vkgpu:0", requires_grad=True)
        k = torch.randn(B, H, S, D, device="vkgpu:0", requires_grad=True)
        v = torch.randn(B, H, S, D, device="vkgpu:0", requires_grad=True)

        result = vulkan_sdpa(q, k, v)
        result.sum().backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
        assert q.grad.shape == q.shape
        # Gradients should be nonzero
        assert q.grad.abs().sum().item() > 0

    # @pytest.mark.skip(reason="SDPA backward produces empty gradients")
    def test_sdpa_matches_cpu_backward(self):
        from pytorch_vulkan import vulkan_sdpa

        B, H, S, D = 1, 2, 8, 16
        q_cpu = torch.randn(B, H, S, D, requires_grad=True)
        k_cpu = torch.randn(B, H, S, D, requires_grad=True)
        v_cpu = torch.randn(B, H, S, D, requires_grad=True)

        # CPU reference
        out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        out_cpu.sum().backward()

        # Vulkan
        q_vk = q_cpu.detach().clone().to("vkgpu:0").requires_grad_(True)
        k_vk = k_cpu.detach().clone().to("vkgpu:0").requires_grad_(True)
        v_vk = v_cpu.detach().clone().to("vkgpu:0").requires_grad_(True)

        out_vk = vulkan_sdpa(q_vk, k_vk, v_vk)
        out_vk.sum().backward()

        assert_close(q_vk.grad, q_cpu.grad, atol=1e-3)
        assert_close(k_vk.grad, k_cpu.grad, atol=1e-3)
        assert_close(v_vk.grad, v_cpu.grad, atol=1e-3)


# =========================================================================
# Float16
# =========================================================================


class TestFloat16:
    def test_f16_roundtrip(self):
        cpu = torch.randn(64).half()
        assert torch.equal(vk(cpu).to("cpu"), cpu)

    def test_f16_add(self):
        a, b = torch.randn(64).half(), torch.randn(64).half()

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return x + y

        assert_close(fn(vk(a), vk(b)), a + b, atol=1e-2)

    def test_f16_mul(self):
        a, b = torch.randn(64).half(), torch.randn(64).half()

        @torch.compile(backend="vulkan")
        def fn(x, y):
            return x * y

        assert_close(fn(vk(a), vk(b)), a * b, atol=1e-2)


# =========================================================================
# Embedding
# =========================================================================


class TestEmbedding:
    def test_embedding_correctness(self):
        import torch.nn as nn

        embed_cpu = nn.Embedding(100, 32)
        embed_vk = nn.Embedding(100, 32).to("vkgpu:0")
        # Copy weights
        embed_vk.weight.data.copy_(embed_cpu.weight.data.to("vkgpu:0"))

        idx = torch.tensor([0, 5, 50, 99])
        expected = embed_cpu(idx)
        result = embed_vk(idx.to("vkgpu:0"))
        assert_close(result, expected, atol=0)


# =========================================================================
# Backward ops
# =========================================================================


class TestBackwardOps:
    def test_threshold_backward(self):
        """Test relu backward produces correct gradients."""
        x_cpu = torch.randn(64, requires_grad=True)
        x_vk = x_cpu.detach().clone().to("vkgpu:0").requires_grad_(True)

        # CPU reference
        y_cpu = torch.relu(x_cpu)
        y_cpu.sum().backward()

        # Vulkan (via compiled)
        y_vk = torch.relu(x_vk)
        y_vk.sum().backward()

        assert_close(x_vk.grad, x_cpu.grad, atol=1e-5)
