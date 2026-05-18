"""Correctness tests for addmm (used by nn.Linear)."""

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


@requires_vulkan
@pytest.mark.parametrize("M,K,N", [(8, 16, 8), (16, 32, 16), (1, 64, 32)])
def test_addmm_correctness(M, K, N):
    """Test addmm(bias, input, weight) matches CPU."""
    from pytorch_vulkan import _C
    _C.flush()

    bias = torch.randn(N)
    input_t = torch.randn(M, K)
    weight = torch.randn(K, N)

    expected = torch.addmm(bias, input_t, weight)
    result = torch.addmm(vk(bias), vk(input_t), vk(weight))
    _C.flush()

    torch.testing.assert_close(result.to("cpu"), expected, atol=1e-3, rtol=1e-3)


@requires_vulkan
def test_addmm_via_linear():
    """Test that nn.Linear (which uses addmm internally) works on Vulkan."""
    from pytorch_vulkan import _C
    _C.flush()

    linear = torch.nn.Linear(32, 16)
    x = torch.randn(4, 32)

    expected = linear(x)

    linear_vk = torch.nn.Linear(32, 16).to("vkgpu:0")
    linear_vk.load_state_dict(linear.state_dict())
    _C.flush()

    result = linear_vk(vk(x))
    _C.flush()

    torch.testing.assert_close(result.to("cpu"), expected, atol=1e-3, rtol=1e-3)


@requires_vulkan
def test_addmm_alpha_beta():
    """Test addmm with non-default alpha and beta."""
    from pytorch_vulkan import _C
    _C.flush()

    bias = torch.randn(8)
    input_t = torch.randn(4, 16)
    weight = torch.randn(16, 8)

    # alpha=2.0, beta=0.5: result = 0.5 * bias + 2.0 * (input @ weight)
    expected = torch.addmm(bias, input_t, weight, alpha=2.0, beta=0.5)
    result = torch.addmm(vk(bias), vk(input_t), vk(weight), alpha=2.0, beta=0.5)
    _C.flush()

    torch.testing.assert_close(result.to("cpu"), expected, atol=1e-2, rtol=1e-2)
