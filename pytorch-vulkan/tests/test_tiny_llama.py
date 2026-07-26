import copy
import unittest

import pytorch_vulkan
import torch
import torch.nn.functional as F


VULKAN_AVAILABLE = pytorch_vulkan.init()

from torch.testing._internal.common_device_type import (  # noqa: E402
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase  # noqa: E402


def apply_rotary_embedding(tensor, cosine, sine):
    half = tensor.shape[-1] // 2
    first, second = tensor[..., :half], tensor[..., half:]
    cosine = cosine[: tensor.shape[-2]].unsqueeze(0).unsqueeze(0)
    sine = sine[: tensor.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat(
        [first * cosine - second * sine, first * sine + second * cosine],
        dim=-1,
    )


class TinyLlamaAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden, cosine, sine):
        batch, sequence, width = hidden.shape

        def split_heads(projection):
            return (
                projection(hidden)
                .view(batch, sequence, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        query = apply_rotary_embedding(split_heads(self.query), cosine, sine)
        key = apply_rotary_embedding(split_heads(self.key), cosine, sine)
        value = split_heads(self.value)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )
        return self.output(
            attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        )


class TinyLlamaBlock(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.attention_norm = torch.nn.RMSNorm(hidden_size, eps=1e-5)
        self.attention = TinyLlamaAttention(hidden_size, num_heads)
        self.mlp_norm = torch.nn.RMSNorm(hidden_size, eps=1e-5)
        self.gate = torch.nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up = torch.nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, hidden, cosine, sine):
        hidden = hidden + self.attention(self.attention_norm(hidden), cosine, sine)
        normalized = self.mlp_norm(hidden)
        return hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))


class TinyLlama(torch.nn.Module):
    def __init__(self, vocab_size=31, hidden_size=16, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.block = TinyLlamaBlock(hidden_size, num_heads)
        self.norm = torch.nn.RMSNorm(hidden_size, eps=1e-5)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        half_head = hidden_size // num_heads // 2
        positions = torch.arange(16, dtype=torch.float32)
        frequencies = 1.0 / (
            10000.0
            ** (torch.arange(half_head, dtype=torch.float32) / max(half_head, 1))
        )
        angles = positions[:, None] * frequencies[None, :]
        self.register_buffer("cosine", angles.cos())
        self.register_buffer("sine", angles.sin())

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        hidden = self.block(hidden, self.cosine, self.sine)
        return self.lm_head(self.norm(hidden))


def causal_language_model_loss(logits, tokens):
    predictions = logits[:, :-1]
    targets = tokens[:, 1:]
    one_hot = F.one_hot(targets, logits.shape[-1]).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    return -(predictions.softmax(-1).log() * one_hot).sum() / targets.numel()


@unittest.skipUnless(VULKAN_AVAILABLE, "Vulkan backend is unavailable")
class TestTinyLlama(TestCase):
    def test_forward_backward_and_adamw_parity(self, device):
        torch.manual_seed(11)
        reference = TinyLlama()
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.mul_(0.1)
        actual = copy.deepcopy(reference).to(device)
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=1e-3, weight_decay=0.01
        )
        actual_optimizer = torch.optim.AdamW(
            actual.parameters(), lr=1e-3, weight_decay=0.01
        )
        batches = [
            torch.tensor([[1, 5, 2, 8, 3, 9, 4, 7]]),
            torch.tensor([[4, 3, 7, 1, 8, 2, 6, 5]]),
            torch.tensor([[2, 9, 1, 4, 8, 3, 7, 6]]),
        ]

        pytorch_vulkan.reset_fallback_stats()
        for tokens_cpu in batches:
            reference_optimizer.zero_grad(set_to_none=True)
            expected_logits = reference(tokens_cpu)
            expected_loss = causal_language_model_loss(expected_logits, tokens_cpu)
            expected_loss.backward()
            reference_optimizer.step()

            actual_optimizer.zero_grad(set_to_none=True)
            tokens = tokens_cpu.to(device)
            with pytorch_vulkan.strict_fallbacks():
                actual_logits = actual(tokens)
                actual_loss = causal_language_model_loss(actual_logits, tokens)
                actual_loss.backward()
                actual_optimizer.step()

            self.assertEqual(
                actual_logits.cpu(),
                expected_logits.detach(),
                atol=8e-4,
                rtol=8e-4,
            )
            self.assertEqual(
                actual_loss.cpu(),
                expected_loss.detach(),
                atol=8e-4,
                rtol=8e-4,
            )
            for expected_parameter, actual_parameter in zip(
                reference.parameters(), actual.parameters()
            ):
                self.assertIsNotNone(actual_parameter.grad)
                self.assertEqual(
                    actual_parameter.grad.cpu(),
                    expected_parameter.grad,
                    atol=1e-3,
                    rtol=1e-3,
                )
                self.assertEqual(
                    actual_parameter.detach().cpu(),
                    expected_parameter.detach(),
                    atol=1e-3,
                    rtol=1e-3,
                )

        self.assertEqual(pytorch_vulkan.fallback_stats()["count"], 0)
        embedding_grad = actual.embedding.weight.grad
        self.assertIsNotNone(embedding_grad)
        self.assertGreater(embedding_grad.abs().sum().item(), 0)

        for expected_parameter, actual_parameter in zip(
            reference.parameters(), actual.parameters()
        ):
            expected_state = reference_optimizer.state[expected_parameter]
            actual_state = actual_optimizer.state[actual_parameter]
            self.assertEqual(set(actual_state), set(expected_state))
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if isinstance(expected_value, torch.Tensor):
                    self.assertEqual(
                        actual_value.cpu(),
                        expected_value,
                        atol=1e-3,
                        rtol=1e-3,
                    )
                else:
                    self.assertEqual(actual_value, expected_value)


if not VULKAN_AVAILABLE:

    class TestVulkanUnavailable(TestCase):
        def test_vulkan_backend_unavailable(self):
            self.skipTest("Vulkan backend is unavailable")


instantiate_device_type_tests(
    TestTinyLlama,
    globals(),
    only_for="privateuse1",
)


if __name__ == "__main__":
    run_tests()
