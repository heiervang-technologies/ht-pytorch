import copy
import unittest

import torch
from torch import nn

from benchmarks.ht_xpu.qwen_ops import (
    apply_rope,
    enable_qwen35_compilation_paths,
    gated_delta_rule,
    qwen_rms_norm,
    swiglu,
)


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size))
        self.eps = eps

    def forward(self, x):
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        return (output * (1.0 + self.weight.float())).type_as(x)


class Qwen3_5MLP(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=13):
        super().__init__()
        self.config = type("Config", (), {"hidden_act": "silu"})()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gated = torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class TestQwenOps(unittest.TestCase):
    def test_rms_norm_forward_backward(self):
        x = torch.randn(2, 4, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        actual = qwen_rms_norm(x, weight)
        reference = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, True) + 1e-6)
        reference = (reference * (1 + weight.float())).to(x.dtype)
        torch.testing.assert_close(actual, reference)
        actual_grads = torch.autograd.grad(actual.sum(), (x, weight))
        reference_grads = torch.autograd.grad(reference.sum(), (x, weight))
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad)

    def test_swiglu_and_rope_backward(self):
        x = torch.randn(2, 3, 8, requires_grad=True)
        gate = torch.randn(13, 8, requires_grad=True)
        up = torch.randn(13, 8, requires_grad=True)
        output = swiglu(x, gate, up)
        self.assertEqual(output.shape, (2, 3, 13))
        q = torch.randn(2, 4, 3, 8, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        cos = torch.randn(1, 1, 3, 8)
        sin = torch.randn(1, 1, 3, 8)
        q_out, k_out = apply_rope(q, k, cos, sin)
        (output.sum() + q_out.sum() + k_out.sum()).backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)

    def test_delta_rule_full_backward(self):
        inputs = [torch.randn(1, 2, 4, 3, requires_grad=True) for _ in range(3)]
        beta_logits = torch.randn(1, 2, 4, requires_grad=True)
        decay_logits = torch.randn(1, 2, 4, requires_grad=True)
        initial_state = torch.randn(1, 2, 3, 3, requires_grad=True)
        output, state = gated_delta_rule(
            *inputs,
            torch.sigmoid(beta_logits),
            torch.sigmoid(decay_logits),
            initial_state,
        )
        (output.square().mean() + state.square().mean()).backward()
        for value in (*inputs, beta_logits, decay_logits, initial_state):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_module_replacement_preserves_results_and_gradients(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = Qwen3_5RMSNorm(8)
                self.mlp = Qwen3_5MLP()

            def forward(self, x):
                return self.mlp(self.norm(x))

        reference = Block()
        optimized = copy.deepcopy(reference)
        replaced = enable_qwen35_compilation_paths(optimized)
        self.assertEqual(replaced, {"rms_norm": 1, "swiglu": 1})
        reference_input = torch.randn(2, 4, 8, requires_grad=True)
        optimized_input = reference_input.detach().clone().requires_grad_()
        reference_output = reference(reference_input)
        optimized_output = optimized(optimized_input)
        torch.testing.assert_close(reference_output, optimized_output)
        reference_output.sum().backward()
        optimized_output.sum().backward()
        torch.testing.assert_close(reference_input.grad, optimized_input.grad)
        for reference_parameter, optimized_parameter in zip(
            reference.parameters(), optimized.parameters()
        ):
            torch.testing.assert_close(
                reference_parameter.grad, optimized_parameter.grad
            )

    def test_module_replacement_does_not_bypass_peft_wrappers(self):
        class AdapterLinear(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(features, features))
                self.adapter = nn.Parameter(torch.randn(features, features))
                self.bias = None

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight + self.adapter)

        module = Qwen3_5MLP(hidden_size=8, intermediate_size=8)
        module.gate_proj = AdapterLinear(8)
        replaced = enable_qwen35_compilation_paths(module)
        self.assertEqual(replaced["swiglu"], 0)
        module(torch.randn(2, 3, 8)).sum().backward()
        self.assertIsNotNone(module.gate_proj.adapter.grad)


if __name__ == "__main__":
    unittest.main()
