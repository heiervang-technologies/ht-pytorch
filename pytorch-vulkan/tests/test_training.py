"""Training / AOT Autograd integration tests."""

import logging

import pytest
import pytorch_vulkan
import torch
import torch.nn as nn


pytorch_vulkan.init()


def requires_vulkan(fn):
    try:
        from pytorch_vulkan import is_available

        available = is_available()
    except ImportError:
        available = False
    return pytest.mark.skipif(not available, reason="Vulkan backend not available")(fn)


@requires_vulkan
def test_aot_autograd_pointwise():
    """Test that AOT Autograd traces forward+backward for pointwise ops."""
    from pytorch_vulkan import get_backward_op_report, register_training

    logging.basicConfig(level=logging.INFO)
    register_training()

    @torch.compile(backend="vulkan_train")
    def fn(x, y):
        return (x * y + x).relu()

    x = torch.randn(64, requires_grad=True, device="vkgpu:0")
    y = torch.randn(64, requires_grad=True, device="vkgpu:0")
    out = fn(x, y)
    out.sum().backward()

    # Check gradients exist.
    assert x.grad is not None
    assert y.grad is not None
    assert x.grad.shape == (64,)

    report = get_backward_op_report()
    print(report)


@requires_vulkan
def test_aot_autograd_matmul():
    """Test AOT Autograd with matmul to discover backward ops."""
    from pytorch_vulkan import get_backward_op_report, register_training

    logging.basicConfig(level=logging.INFO)
    register_training()

    @torch.compile(backend="vulkan_train")
    def fn(a, b):
        return torch.mm(a, b)

    a = torch.randn(16, 32, requires_grad=True, device="vkgpu:0")
    b = torch.randn(32, 16, requires_grad=True, device="vkgpu:0")
    out = fn(a, b)
    out.sum().backward()

    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.shape == (16, 32)

    report = get_backward_op_report()
    print(report)


@requires_vulkan
def test_aot_autograd_simple_mlp():
    """Test AOT Autograd with a simple MLP to discover the full op set
    needed for real training."""
    from pytorch_vulkan import get_backward_op_report, register_training

    logging.basicConfig(level=logging.INFO)
    register_training()

    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleMLP().to("vkgpu:0")
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    compiled_model = torch.compile(model, backend="vulkan_train")

    # Single training step.
    x = torch.randn(8, 32, device="vkgpu:0")
    target = torch.randn(8, 10, device="vkgpu:0")

    out = compiled_model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    report = get_backward_op_report()
    print("\n" + report)
    # This report tells us exactly which backward ops need SPIR-V shaders.


@requires_vulkan
def test_aot_autograd_adam():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().to("vkgpu:0")
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    compiled_model = torch.compile(model, backend="vulkan_train")

    x = torch.randn(4, 16, device="vkgpu:0")
    target = torch.randn(4, 8, device="vkgpu:0")

    out = compiled_model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()


@requires_vulkan
def test_aot_autograd_adamw():
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 8)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel().to("vkgpu:0")
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    compiled_model = torch.compile(model, backend="vulkan_train")

    x = torch.randn(4, 16, device="vkgpu:0")
    target = torch.randn(4, 8, device="vkgpu:0")

    out = compiled_model(x)
    loss = (out - target).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
