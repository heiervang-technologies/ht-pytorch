import copy

import torch
from torch import nn
from torch.testing._internal.common_utils import run_tests, TestCase

from benchmarks.ht_xpu.qwen_ops import (
    apply_rope,
    enable_qwen35_compilation_paths,
    gated_delta_rule,
    qwen_rms_norm,
    qwen_rms_norm_gated,
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


class TestQwenOps(TestCase):
    def test_rms_norm_forward_backward(self):
        x = torch.randn(2, 4, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        actual = qwen_rms_norm(x, weight)
        reference = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, True) + 1e-6)
        reference = (reference * (1 + weight.float())).to(x.dtype)
        self.assertEqual(actual, reference)
        self.assertEqual(
            torch.autograd.grad(actual.sum(), (x, weight)),
            torch.autograd.grad(reference.sum(), (x, weight)),
        )

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

    def test_gated_rms_norm_backward(self):
        hidden = torch.randn(2, 8, requires_grad=True)
        gate = torch.randn(2, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        output = qwen_rms_norm_gated(hidden, gate, weight)
        normalized = hidden.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + 1e-6)
        reference = weight * normalized.to(hidden.dtype)
        reference = (reference * torch.nn.functional.silu(gate.float())).to(
            hidden.dtype
        )
        self.assertEqual(output, reference)
        self.assertEqual(
            torch.autograd.grad(output.sum(), (hidden, gate, weight)),
            torch.autograd.grad(reference.sum(), (hidden, gate, weight)),
        )

    def test_delta_rule_full_backward(self):
        inputs = [torch.randn(1, 2, 4, 3, requires_grad=True) for _ in range(3)]
        beta = torch.sigmoid(torch.randn(1, 2, 4, requires_grad=True))
        decay = torch.sigmoid(torch.randn(1, 2, 4, requires_grad=True))
        output, state = gated_delta_rule(*inputs, beta, decay)
        (output.square().mean() + state.square().mean()).backward()
        for value in inputs:
            self.assertIsNotNone(value.grad)

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
        self.assertEqual(replaced, {"rms_norm": 1, "rms_norm_gated": 0})
        reference_input = torch.randn(2, 4, 8, requires_grad=True)
        optimized_input = reference_input.detach().clone().requires_grad_()
        reference_output = reference(reference_input)
        optimized_output = optimized(optimized_input)
        self.assertEqual(reference_output, optimized_output)
        reference_output.sum().backward()
        optimized_output.sum().backward()
        self.assertEqual(reference_input.grad, optimized_input.grad)
        for reference_parameter, optimized_parameter in zip(
            reference.parameters(), optimized.parameters()
        ):
            self.assertEqual(reference_parameter.grad, optimized_parameter.grad)

    def test_module_replacement_does_not_bypass_lora_projections(self):
        class AdapterLinear(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.adapter = nn.Linear(base.in_features, base.out_features, bias=False)

            def forward(self, x):
                return self.base(x) + self.adapter(x)

        mlp = Qwen3_5MLP()
        enable_qwen35_compilation_paths(mlp)
        mlp.gate_proj = AdapterLinear(mlp.gate_proj)
        mlp(torch.randn(2, 8)).sum().backward()
        self.assertIsNotNone(mlp.gate_proj.adapter.weight.grad)


if __name__ == "__main__":
    run_tests()
