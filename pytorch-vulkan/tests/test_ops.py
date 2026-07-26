"""Comprehensive op-level tests for the Vulkan backend."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture(autouse=True)
def init_vulkan():
    from pytorch_vulkan import init

    init()


def vk(t):
    """Move tensor to Vulkan device."""
    return t.to("vkgpu:0")


def close(a, b, atol=1e-4):
    """Check tensors are close, handling device transfer."""
    a_cpu = a.to("cpu") if a.device.type != "cpu" else a
    b_cpu = b.to("cpu") if b.device.type != "cpu" else b
    return torch.allclose(a_cpu, b_cpu, atol=atol)


# --- View / Reshape ---


class TestViewReshape:
    def test_view_3d_to_2d(self):
        t = vk(torch.randn(4, 16, 256))
        r = t.view(-1, 256)
        assert r.shape == (64, 256)
        assert r.device.type == "vkgpu"

    def test_view_2d_to_1d(self):
        t = vk(torch.randn(8, 32))
        r = t.view(-1)
        assert r.shape == (256,)

    def test_reshape_preserves_data(self):
        cpu = torch.randn(3, 4)
        v = vk(cpu)
        r = v.reshape(12)
        assert close(r, cpu.reshape(12))

    def test_view_4d(self):
        t = vk(torch.randn(2, 4, 8, 16))
        r = t.view(2, 4, 128)
        assert r.shape == (2, 4, 128)


# --- Embedding ---


class TestEmbedding:
    def test_basic_embedding(self):
        embed = nn.Embedding(100, 32).to("vkgpu:0")
        idx = torch.tensor([0, 5, 10, 99], device="vkgpu:0")
        out = embed(idx)
        assert out.shape == (4, 32)
        assert out.device.type == "vkgpu"

    def test_embedding_2d_indices(self):
        embed = nn.Embedding(50, 16).to("vkgpu:0")
        idx = torch.randint(0, 50, (3, 8), device="vkgpu:0")
        out = embed(idx)
        assert out.shape == (3, 8, 16)


# --- Softmax ---


class TestSoftmax:
    def test_softmax_2d_last_dim(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        cpu = torch.randn(8, 32)
        v = vk(cpu)
        result = fn(v)
        expected = torch.softmax(cpu, dim=-1)
        assert close(result, expected, atol=1e-5)

    def test_softmax_row_sums(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        v = vk(torch.randn(4, 16))
        result = fn(v).to("cpu")
        sums = result.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_softmax_all_positive(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.softmax(x, dim=-1)

        v = vk(torch.randn(2, 8))
        result = fn(v).to("cpu")
        assert (result > 0).all()


# --- Dim-aware Reductions ---


class TestDimReductions:
    def test_sum_dim0(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sum(x, dim=0)

        cpu = torch.randn(8, 10)
        result = fn(vk(cpu))
        expected = torch.sum(cpu, dim=0)
        assert result.shape == (10,)
        assert close(result, expected, atol=1e-3)

    def test_sum_dim1(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.sum(x, dim=1)

        cpu = torch.randn(4, 32)
        result = fn(vk(cpu))
        expected = torch.sum(cpu, dim=1)
        assert result.shape == (4,)
        assert close(result, expected, atol=1e-3)

    def test_mean_dim(self):
        from pytorch_vulkan import register

        register()

        @torch.compile(backend="vulkan")
        def fn(x):
            return torch.mean(x, dim=-1)

        cpu = torch.randn(4, 16)
        result = fn(vk(cpu))
        expected = torch.mean(cpu, dim=-1)
        assert result.shape == (4,)
        assert close(result, expected, atol=1e-3)


# --- Transformer Integration ---


class TestTransformer:
    def test_encoder_layer_shapes(self):
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True
        ).to("vkgpu:0")
        x = vk(torch.randn(2, 8, 32))
        out = layer(x)
        assert out.shape == (2, 8, 32)
        assert out.device.type == "vkgpu"

    def test_transformer_training_step(self):
        from pytorch_vulkan import register_training

        register_training()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(64, 32)
                self.encoder = nn.TransformerEncoderLayer(
                    d_model=32,
                    nhead=4,
                    dim_feedforward=64,
                    dropout=0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(32, 64)

            def forward(self, x):
                return self.head(self.encoder(self.embed(x)))

        model = TinyModel().to("vkgpu:0")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        compiled = torch.compile(model, backend="vulkan_train")

        x = torch.randint(0, 64, (2, 4), device="vkgpu:0")
        target = torch.randint(0, 64, (2, 4), device="vkgpu:0")

        logits = compiled(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 64), target.reshape(-1))
        loss.backward()
        opt.step()

        assert loss.item() > 0  # sanity check
