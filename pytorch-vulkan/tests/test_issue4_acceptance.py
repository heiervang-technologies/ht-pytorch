import copy
import gc
import math
import re
import threading
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import pytorch_vulkan
from pytorch_vulkan.operator_registry import (
    OPERATOR_REGISTRY,
    native_shaders,
    shader_binding_count,
)


VULKAN_AVAILABLE = pytorch_vulkan.init()

from torch.testing._internal.common_device_type import (  # noqa: E402
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (  # noqa: E402
    TestCase,
    parametrize,
    run_tests,
)


def causal_lm_loss(logits, tokens):
    predictions = logits[:, :-1]
    targets = tokens[:, 1:]
    one_hot = F.one_hot(targets, logits.size(-1)).to(
        device=logits.device, dtype=logits.dtype
    )
    log_probabilities = predictions.softmax(-1).log()
    return -(log_probabilities * one_hot).sum() / targets.numel()


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(23, 12)
        self.projection = torch.nn.Linear(12, 23, bias=False)

    def forward(self, tokens):
        return self.projection(self.embedding(tokens))


class TestOperatorRegistry(TestCase):
    def test_registry_is_unique_and_complete(self):
        shader_directory = Path(__file__).parents[1] / "shaders"
        native_keys = [operator.native_key for operator in OPERATOR_REGISTRY]
        self.assertEqual(len(native_keys), len(set(native_keys)))
        all_capabilities = {
            capability: True
            for capability in (
                "shader_float16",
                "storage_buffer16_bit_access",
                "shader_buffer_float32_atomic_add",
                "shader_shared_float32_atomic_add",
                "cooperative_matrix_nv",
                "push_descriptor",
            )
        }
        registration_keys = [
            key for key, _ in native_shaders(all_capabilities)
        ]
        self.assertEqual(
            len(registration_keys), len(set(registration_keys))
        )
        shim_source = (
            Path(__file__).parents[1] / "csrc" / "shim.cpp"
        ).read_text()
        for operator in OPERATOR_REGISTRY:
            if operator.autoload:
                self.assertIn(
                    f'"{operator.native_key}"',
                    shim_source,
                )
        dtype_dispatch_keys = set(
            re.findall(
                r'find_shader_for_dtype\(\s*"([^"]+)"',
                shim_source,
            )
        )
        direct_dispatch_keys = set(
            re.findall(
                r'g_shader_handles\.find\(\s*"([^"]+)"',
                shim_source,
            )
        )
        direct_dispatch_keys.update(
            re.findall(
                r'shader_name\s*=\s*"([^"]+)"',
                shim_source,
            )
        )
        self.assertTrue(dtype_dispatch_keys.issubset(set(native_keys)))
        self.assertTrue(
            direct_dispatch_keys.issubset(set(registration_keys))
        )
        registered_shader_files = set()
        scalar_types = {"uint": "u32", "int": "i32", "float": "f32"}
        for operator in OPERATOR_REGISTRY:
            self.assertTrue(operator.schema)
            self.assertTrue(operator.layouts)
            self.assertTrue(operator.ranks)
            self.assertTrue(operator.push_constants)
            self.assertTrue(operator.autograd)
            self.assertTrue(operator.tolerances)
            for variant in operator.shaders.values():
                shader_path = shader_directory / variant.file
                self.assertTrue(shader_path.is_file())
                shader_source = shader_path.read_text()
                descriptor_bindings = tuple(
                    map(
                        int,
                        re.findall(
                            r"\bbinding\s*=\s*(\d+)",
                            shader_source,
                        ),
                    )
                )
                binding_count = shader_binding_count(variant)
                self.assertGreaterEqual(binding_count, 1)
                self.assertLessEqual(binding_count, 16)
                self.assertEqual(
                    tuple(sorted(descriptor_bindings)),
                    tuple(range(binding_count)),
                )
                push_block = re.search(
                    r"layout\s*\(push_constant\)\s*uniform\s+\w+\s*"
                    r"\{(.*?)\}",
                    shader_source,
                    re.DOTALL,
                )
                self.assertIsNotNone(push_block)
                declared_types = tuple(
                    scalar_types[scalar_type]
                    for scalar_type, _ in re.findall(
                        r"\b(uint|int|float)\s+(\w+)\s*;",
                        push_block.group(1),
                    )
                )
                registry_types = tuple(
                    field.rsplit(":", 1)[1]
                    for field in operator.push_constants
                )
                self.assertEqual(declared_types, registry_types)
                self.assertLessEqual(len(registry_types) * 4, 128)
                registered_shader_files.add(variant.file)
        shader_files = {path.name for path in shader_directory.glob("*.comp")}
        self.assertEqual(registered_shader_files, shader_files)


@unittest.skipUnless(VULKAN_AVAILABLE, "Vulkan backend is unavailable")
class TestVulkanIssue4(TestCase):
    @parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.bfloat16],
    )
    def test_dtype_aware_scalar(self, device, dtype):
        capabilities = pytorch_vulkan.device_info()["capabilities"]
        if dtype == torch.float16 and not (
            capabilities["shader_float16"]
            and capabilities["storage_buffer16_bit_access"]
        ):
            self.skipTest("FP16 shader and storage features are required")
        expected = torch.full((7,), 3.0, dtype=dtype)
        actual = torch.ones(7, dtype=dtype, device=device)
        with pytorch_vulkan.strict_fallbacks():
            actual.add_(1.0).mul_(2.0).div_(2.0)
        self.assertEqual(actual.cpu(), expected)

    def test_empty_strided(self, device):
        tensor = torch.empty_strided(
            (2, 3), (5, 1), dtype=torch.float32, device=device
        )
        self.assertEqual(tensor.stride(), (5, 1))
        source = torch.arange(6, dtype=torch.float32).view(2, 3)
        tensor.copy_(source)
        self.assertEqual(tensor.cpu(), source)

    def test_out_resize_and_mixed_dtype_fallback(self, device):
        left = torch.arange(6, dtype=torch.float32, device=device).view(2, 3)
        right = torch.ones_like(left)
        output = torch.empty(0, dtype=torch.float32, device=device)
        with pytorch_vulkan.strict_fallbacks():
            torch.add(left, right, out=output)
        self.assertEqual(output.size(), (2, 3))
        self.assertEqual(
            output.cpu(),
            torch.arange(6, dtype=torch.float32).view(2, 3) + 1,
        )

        pytorch_vulkan.reset_fallback_stats()
        promoted = left + right.to(torch.float64)
        self.assertEqual(promoted.dtype, torch.float64)
        self.assertEqual(
            promoted.cpu(),
            torch.arange(6, dtype=torch.float64).view(2, 3) + 1,
        )
        self.assertGreater(pytorch_vulkan.fallback_stats()["count"], 0)

    @parametrize(
        "shape",
        [
            (0,),
            (2, 0, 3),
            (257,),
        ],
    )
    def test_empty_and_dynamic_pointwise(self, device, shape):
        left_cpu = torch.empty(shape).normal_()
        right_cpu = torch.empty(shape).normal_()
        left = left_cpu.to(device)
        right = right_cpu.to(device)
        with pytorch_vulkan.strict_fallbacks():
            result = (left + right).relu()
        self.assertEqual(result.cpu(), (left_cpu + right_cpu).relu())

    def test_nonaligned_matmul_and_noncontiguous_copy(self, device):
        left_cpu = torch.randn(17, 19)
        right_cpu = torch.randn(19, 13)
        left = left_cpu.to(device)
        right = right_cpu.to(device)
        with pytorch_vulkan.strict_fallbacks():
            result = left @ right
        self.assertEqual(
            result.cpu(), left_cpu @ right_cpu, atol=2e-4, rtol=2e-4
        )

        source_cpu = torch.arange(35, dtype=torch.float32).view(5, 7).t()
        source = source_cpu.to(device)
        destination = torch.empty_strided(
            source_cpu.shape,
            source_cpu.stride(),
            dtype=source_cpu.dtype,
            device=device,
        )
        destination.copy_(source)
        self.assertEqual(destination.cpu(), source_cpu)

    def test_storage_offset_and_noncontiguous_pointwise(self, device):
        left_base_cpu = torch.randn(4, 7)
        right_base_cpu = torch.randn(4, 7)
        left_cpu = left_base_cpu[:, 1:6].t()
        right_cpu = right_base_cpu[:, 1:6].t()
        left = left_base_cpu.to(device)[:, 1:6].t()
        right = right_base_cpu.to(device)[:, 1:6].t()
        with pytorch_vulkan.strict_fallbacks():
            actual = -(left + right)
        self.assertEqual(actual.cpu(), -(left_cpu + right_cpu))

    @parametrize("operation", ["sum", "mean"])
    def test_reduction_dimension_semantics(self, device, operation):
        values_cpu = torch.randn(2, 3, 5)
        values = values_cpu.to(device)
        expected = getattr(values_cpu, operation)(
            dim=[], keepdim=True
        )
        with pytorch_vulkan.strict_fallbacks():
            actual = getattr(values, operation)(dim=[], keepdim=True)
        self.assertEqual(
            actual.cpu(), expected, atol=2e-4, rtol=2e-4
        )

        with self.assertRaisesRegex(RuntimeError, "multiple times"):
            getattr(values, operation)(dim=[1, -2])
        with self.assertRaises(IndexError):
            getattr(values, operation)(dim=[3])

        empty_cpu = torch.empty(0, 3, 5)
        with pytorch_vulkan.strict_fallbacks():
            empty = getattr(empty_cpu.to(device), operation)(
                dim=1
            )
        self.assertEqual(
            empty.cpu(),
            getattr(empty_cpu, operation)(dim=1),
        )

    def test_reduction_explicit_dtype_is_correctly_reported(self, device):
        values_cpu = torch.randn(4, 7)
        values = values_cpu.to(device)
        pytorch_vulkan.reset_fallback_stats()
        actual = values.sum(dim=1, dtype=torch.float64)
        self.assertEqual(
            actual.cpu(),
            values_cpu.sum(dim=1, dtype=torch.float64),
        )
        self.assertGreater(pytorch_vulkan.fallback_stats()["count"], 0)

    def test_fx_eager_boundary_stays_on_device(self, device):
        pytorch_vulkan.register()

        @torch.compile(backend="vulkan", fullgraph=True)
        def compiled(left, right):
            return torch.mean((left + right).view(-1))

        @torch.compile(backend="vulkan", fullgraph=True)
        def compiled_strided(left, right):
            return left[:, 1:] + right[:, 1:]

        left_cpu = torch.randn(3, 5)
        right_cpu = torch.randn(1, 5)
        pytorch_vulkan.reset_fallback_stats()
        with pytorch_vulkan.strict_fallbacks():
            actual = compiled(
                left_cpu.to(device),
                right_cpu.to(device),
            )
            actual_strided = compiled_strided(
                left_cpu.to(device),
                left_cpu.to(device),
            )

        self.assertEqual(
            actual.cpu(),
            (left_cpu + right_cpu).mean(),
            atol=2e-4,
            rtol=2e-4,
        )
        self.assertEqual(
            actual_strided.cpu(),
            left_cpu[:, 1:] + left_cpu[:, 1:],
        )
        self.assertEqual(pytorch_vulkan.fallback_stats()["count"], 0)

    def test_addmm_scalars(self, device):
        matrix1_cpu = torch.randn(5, 7)
        matrix2_cpu = torch.randn(7, 3)
        bias_cpu = torch.randn(3)
        expected = torch.addmm(
            bias_cpu, matrix1_cpu, matrix2_cpu, beta=0.25, alpha=1.75
        )
        with pytorch_vulkan.strict_fallbacks():
            actual = torch.addmm(
                bias_cpu.to(device),
                matrix1_cpu.to(device),
                matrix2_cpu.to(device),
                beta=0.25,
                alpha=1.75,
            )
        self.assertEqual(actual.cpu(), expected, atol=3e-4, rtol=3e-4)

    @parametrize("operation", ["addcdiv", "addcmul"])
    def test_optimizer_kernel_broadcasting(self, device, operation):
        base_cpu = torch.randn(2, 3)
        first_cpu = torch.randn(1, 3)
        second_cpu = torch.randn(2, 1).abs().add_(0.5)
        reference = getattr(torch, operation)(
            base_cpu, first_cpu, second_cpu, value=0.25
        )
        base = base_cpu.to(device)
        first = first_cpu.to(device)
        second = second_cpu.to(device)
        with pytorch_vulkan.strict_fallbacks():
            actual = getattr(torch, operation)(
                base, first, second, value=0.25
            )
        self.assertEqual(actual.cpu(), reference, atol=2e-4, rtol=2e-4)

    @parametrize("tensor_weight", [False, True])
    def test_lerp_empty_and_storage_offset(self, device, tensor_weight):
        empty = torch.empty(0, device=device)
        with pytorch_vulkan.strict_fallbacks():
            empty_result = torch.lerp(
                empty,
                empty,
                torch.empty(0, device=device)
                if tensor_weight
                else 0.25,
            )
        self.assertEqual(empty_result.numel(), 0)

        base_cpu = torch.randn(11)
        end_cpu = torch.randn(7)
        expected = base_cpu.clone()
        expected[2:9].lerp_(
            end_cpu,
            torch.full((7,), 0.25) if tensor_weight else 0.25,
        )
        base = base_cpu.to(device)
        end = end_cpu.to(device)
        weight = (
            torch.full((7,), 0.25, device=device)
            if tensor_weight
            else 0.25
        )
        pytorch_vulkan.reset_fallback_stats()
        base[2:9].lerp_(end, weight)
        self.assertEqual(base.cpu(), expected)
        self.assertGreater(pytorch_vulkan.fallback_stats()["count"], 0)

    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_embedding_gradient(self, device, index_dtype):
        weight_cpu = torch.randn(11, 7, requires_grad=True)
        weight = weight_cpu.detach().to(device).requires_grad_()
        indices_cpu = torch.tensor(
            [[1, 3, 1], [0, 10, 3]], dtype=index_dtype
        )
        indices = indices_cpu.to(device)

        expected = F.embedding(indices_cpu, weight_cpu).square().sum()
        expected.backward()
        with pytorch_vulkan.strict_fallbacks():
            actual = F.embedding(indices, weight).square().sum()
            actual.backward()
        self.assertEqual(actual.cpu(), expected.detach())
        self.assertEqual(weight.grad.cpu(), weight_cpu.grad)

    def test_index_bounds_are_checked(self, device):
        weight = torch.randn(5, 3, device=device)
        invalid = torch.tensor([0, 5], device=device)
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            F.embedding(invalid, weight)

        values = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)
        sliced = values[:, 1:]
        with self.assertRaisesRegex(RuntimeError, "out of bounds"):
            F.one_hot(sliced, 5)

    @parametrize(
        "mask_kind",
        ["causal", "additive", "boolean", "fully_masked"],
    )
    def test_causal_and_masked_sdpa(self, device, mask_kind):
        torch.manual_seed(17)
        query_cpu = torch.randn(2, 3, 7, 11)
        key_cpu = torch.randn(2, 3, 7, 11)
        value_cpu = torch.randn(2, 3, 7, 13)
        causal = mask_kind == "causal"
        mask_cpu = None
        if mask_kind == "additive":
            mask_cpu = torch.zeros(1, 1, 7, 7)
            mask_cpu[..., -1] = -math.inf
        elif mask_kind == "boolean":
            mask_cpu = torch.ones(1, 1, 1, 7, dtype=torch.bool)
            mask_cpu[..., -1] = False
        elif mask_kind == "fully_masked":
            mask_cpu = torch.zeros(1, 1, 1, 7, dtype=torch.bool)

        expected = F.scaled_dot_product_attention(
            query_cpu,
            key_cpu,
            value_cpu,
            attn_mask=mask_cpu,
            is_causal=causal,
        )
        query = query_cpu.to(device)
        key = key_cpu.to(device)
        value = value_cpu.to(device)
        mask = mask_cpu.to(device) if mask_cpu is not None else None
        with pytorch_vulkan.strict_fallbacks():
            actual = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                is_causal=causal,
            )
        self.assertEqual(
            actual.cpu(), expected, atol=3e-4, rtol=3e-4
        )

    def test_kvcache_attention_is_one_fused_dispatch(self, device):
        capabilities = pytorch_vulkan.device_info()["capabilities"]
        if not (
            capabilities["shader_float16"]
            and capabilities["storage_buffer16_bit_access"]
        ):
            self.skipTest("FP16 shader and storage features are required")

        torch.manual_seed(19)
        query_cpu = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        key_cpu = torch.randn(1, 2, 7, 8, dtype=torch.float16)
        value_cpu = torch.randn(1, 2, 7, 8, dtype=torch.float16)
        scores = (
            query_cpu.float() @ key_cpu.float().transpose(-2, -1)
        ) / math.sqrt(query_cpu.size(-1))
        expected = (scores.softmax(-1) @ value_cpu.float()).to(torch.float16)
        query = query_cpu.to(device)
        key = key_cpu.to(device)
        value = value_cpu.to(device)
        baseline = pytorch_vulkan.memory_stats()["total_dispatches"]

        with pytorch_vulkan.strict_fallbacks():
            actual = pytorch_vulkan.flash_attention_kvcache(
                query, key, value
            )

        final = pytorch_vulkan.memory_stats()["total_dispatches"]
        self.assertEqual(final - baseline, 1)
        self.assertEqual(actual.cpu(), expected, atol=1e-2, rtol=1e-2)

    @parametrize("dtype", [torch.float32, torch.float16])
    def test_flash_attention_forward_and_backward(self, device, dtype):
        if dtype == torch.float16:
            capabilities = pytorch_vulkan.device_info()["capabilities"]
            if not (
                capabilities["shader_float16"]
                and capabilities["storage_buffer16_bit_access"]
                and capabilities[
                    "shader_buffer_float32_atomic_add"
                ]
            ):
                self.skipTest(
                    "FP16 storage and float atomic add are required"
                )
        torch.manual_seed(29)
        query_cpu = torch.randn(
            1, 2, 9, 8, dtype=dtype, requires_grad=True
        )
        key_cpu = torch.randn(
            1, 2, 9, 8, dtype=dtype, requires_grad=True
        )
        value_cpu = torch.randn(
            1, 2, 9, 7, dtype=dtype, requires_grad=True
        )
        expected = F.scaled_dot_product_attention(
            query_cpu, key_cpu, value_cpu
        )
        expected.square().sum().backward()

        query = query_cpu.detach().to(device).requires_grad_()
        key = key_cpu.detach().to(device).requires_grad_()
        value = value_cpu.detach().to(device).requires_grad_()
        pytorch_vulkan.reset_fallback_stats()
        with pytorch_vulkan.strict_fallbacks():
            actual = pytorch_vulkan.flash_attention_vulkan(
                query, key, value
            )
            actual.square().sum().backward()

        output_tolerance = 4e-4 if dtype == torch.float32 else 1e-2
        gradient_tolerance = 6e-4 if dtype == torch.float32 else 2e-2
        self.assertEqual(
            actual.cpu(),
            expected.detach(),
            atol=output_tolerance,
            rtol=output_tolerance,
        )
        for actual_grad, expected_tensor in (
            (query.grad, query_cpu),
            (key.grad, key_cpu),
            (value.grad, value_cpu),
        ):
            self.assertEqual(
                actual_grad.cpu(),
                expected_tensor.grad,
                atol=gradient_tolerance,
                rtol=gradient_tolerance,
            )
        self.assertEqual(pytorch_vulkan.fallback_stats()["count"], 0)

    def test_bfloat16_softmax_backward(self, device):
        values_cpu = torch.randn(3, 9, dtype=torch.bfloat16, requires_grad=True)
        expected_output = values_cpu.softmax(-1)
        expected_loss = (expected_output * expected_output).sum()
        expected_loss.backward()

        values = values_cpu.detach().to(device).requires_grad_()
        with pytorch_vulkan.strict_fallbacks():
            output = values.softmax(-1)
            loss = (output * output).sum()
            loss.backward()
        self.assertEqual(
            output.cpu(), expected_output.detach(), atol=2e-2, rtol=2e-2
        )
        self.assertEqual(
            values.grad.cpu(), values_cpu.grad, atol=2e-2, rtol=2e-2
        )

    def test_float16_relu_backward(self, device):
        capabilities = pytorch_vulkan.device_info()["capabilities"]
        if not (
            capabilities["shader_float16"]
            and capabilities["storage_buffer16_bit_access"]
        ):
            self.skipTest("FP16 shader and storage features are required")
        values_cpu = torch.randn(4, 11, dtype=torch.float16, requires_grad=True)
        values = values_cpu.detach().to(device).requires_grad_()
        expected = values_cpu.relu().square().sum()
        expected.backward()
        with pytorch_vulkan.strict_fallbacks():
            actual = values.relu().square().sum()
            actual.backward()
        self.assertEqual(actual.cpu(), expected.detach())
        self.assertEqual(values.grad.cpu(), values_cpu.grad)

    @parametrize("dtype", [torch.float32, torch.float16])
    def test_layer_norm_forward_and_backward(self, device, dtype):
        if dtype == torch.float16:
            capabilities = pytorch_vulkan.device_info()["capabilities"]
            if not (
                capabilities["shader_float16"]
                and capabilities["storage_buffer16_bit_access"]
            ):
                self.skipTest("FP16 shader and storage features are required")
        torch.manual_seed(23)
        values_cpu = torch.randn(2, 3, 7, dtype=dtype, requires_grad=True)
        weight_cpu = torch.randn(3, 7, dtype=dtype, requires_grad=True)
        bias_cpu = torch.randn(3, 7, dtype=dtype, requires_grad=True)
        expected, expected_mean, expected_rstd = torch.native_layer_norm(
            values_cpu, (3, 7), weight_cpu, bias_cpu, 1e-5
        )
        expected.square().sum().backward()

        values = values_cpu.detach().to(device).requires_grad_()
        weight = weight_cpu.detach().to(device).requires_grad_()
        bias = bias_cpu.detach().to(device).requires_grad_()
        with pytorch_vulkan.strict_fallbacks():
            actual, actual_mean, actual_rstd = torch.native_layer_norm(
                values, (3, 7), weight, bias, 1e-5
            )
            actual.square().sum().backward()
        tolerance = 3e-4 if dtype == torch.float32 else 1e-2
        self.assertEqual(actual_mean.dtype, dtype)
        self.assertEqual(actual_rstd.dtype, dtype)
        self.assertEqual(
            actual.cpu(), expected.detach(), atol=tolerance, rtol=tolerance
        )
        self.assertEqual(
            actual_mean.cpu(),
            expected_mean.detach(),
            atol=tolerance,
            rtol=tolerance,
        )
        self.assertEqual(
            actual_rstd.cpu(),
            expected_rstd.detach(),
            atol=tolerance,
            rtol=tolerance,
        )
        self.assertEqual(
            values.grad.cpu(), values_cpu.grad, atol=tolerance, rtol=tolerance
        )
        self.assertEqual(
            weight.grad.cpu(), weight_cpu.grad, atol=tolerance, rtol=tolerance
        )
        self.assertEqual(
            bias.grad.cpu(), bias_cpu.grad, atol=tolerance, rtol=tolerance
        )

    def test_mixed_dtype_layer_norm_reports_fallback(self, device):
        capabilities = pytorch_vulkan.device_info()["capabilities"]
        if not (
            capabilities["shader_float16"]
            and capabilities["storage_buffer16_bit_access"]
        ):
            self.skipTest("FP16 shader and storage features are required")
        values = torch.randn(2, 5, dtype=torch.float16, device=device)
        weight = torch.ones(5, dtype=torch.float32, device=device)
        bias = torch.zeros(5, dtype=torch.float32, device=device)
        pytorch_vulkan.reset_fallback_stats()
        output, mean, rstd = torch.native_layer_norm(
            values, (5,), weight, bias, 1e-5
        )
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(mean.dtype, torch.float32)
        self.assertEqual(rstd.dtype, torch.float32)
        self.assertGreater(pytorch_vulkan.fallback_stats()["count"], 0)

    def test_dropout_is_reproducible_and_reported(self, device):
        data = torch.ones(4096, device=device)
        pytorch_vulkan.reset_fallback_stats()
        torch.manual_seed(31)
        first = F.dropout(data, p=0.25, training=True)
        torch.manual_seed(31)
        second = F.dropout(data, p=0.25, training=True)
        self.assertEqual(first.cpu(), second.cpu())
        self.assertGreater(pytorch_vulkan.fallback_stats()["count"], 0)

    @parametrize("optimizer_name", ["sgd", "adam", "adamw"])
    def test_multistep_optimizer_and_lm_parity(self, device, optimizer_name):
        torch.manual_seed(3)
        reference = TinyLanguageModel()
        actual = copy.deepcopy(reference).to(device)
        optimizer_types = {
            "sgd": torch.optim.SGD,
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }
        options = {"lr": 2e-3}
        if optimizer_name == "adamw":
            options["weight_decay"] = 0.01
        reference_optimizer = optimizer_types[optimizer_name](
            reference.parameters(), **options
        )
        actual_optimizer = optimizer_types[optimizer_name](
            actual.parameters(), **options
        )
        batches = [
            torch.tensor([[1, 2, 3, 4, 5], [5, 3, 8, 2, 1]]),
            torch.tensor([[4, 4, 9, 2, 0], [3, 7, 1, 6, 2]]),
            torch.tensor([[9, 8, 7, 6, 5], [2, 1, 3, 5, 8]]),
        ]

        pytorch_vulkan.reset_fallback_stats()
        for tokens_cpu in batches:
            reference_optimizer.zero_grad(set_to_none=True)
            expected_logits = reference(tokens_cpu)
            expected_loss = causal_lm_loss(expected_logits, tokens_cpu)
            expected_loss.backward()
            reference_optimizer.step()

            actual_optimizer.zero_grad(set_to_none=True)
            tokens = tokens_cpu.to(device)
            with pytorch_vulkan.strict_fallbacks():
                actual_logits = actual(tokens)
                actual_loss = causal_lm_loss(actual_logits, tokens)
                actual_loss.backward()
                actual_optimizer.step()

            self.assertEqual(
                actual_logits.cpu(),
                expected_logits.detach(),
                atol=4e-4,
                rtol=4e-4,
            )
            self.assertEqual(
                actual_loss.cpu(), expected_loss.detach(), atol=4e-4, rtol=4e-4
            )
            for expected_parameter, actual_parameter in zip(
                reference.parameters(), actual.parameters()
            ):
                self.assertIsNotNone(actual_parameter.grad)
                self.assertEqual(
                    actual_parameter.grad.cpu(),
                    expected_parameter.grad,
                    atol=5e-4,
                    rtol=5e-4,
                )
                self.assertEqual(
                    actual_parameter.detach().cpu(),
                    expected_parameter.detach(),
                    atol=5e-4,
                    rtol=5e-4,
                )

        self.assertEqual(pytorch_vulkan.fallback_stats()["count"], 0)
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
                        atol=5e-4,
                        rtol=5e-4,
                    )
                else:
                    self.assertEqual(actual_value, expected_value)

    def test_lifecycle_and_memory_stats(self, device):
        info = pytorch_vulkan.device_info()
        self.assertEqual(info["extensions"], sorted(info["extensions"]))
        self.assertEqual(len(info["api_version_components"]), 3)
        self.assertGreaterEqual(info["max_storage_buffer_bindings"], 1)
        gc.collect()
        pytorch_vulkan.empty_cache()
        baseline = pytorch_vulkan.memory_stats()
        self.assertEqual(baseline["active_allocations"], 0)
        self.assertGreater(baseline["active_pipelines"], 0)
        tensor = torch.ones(128, device=device)
        active = pytorch_vulkan.memory_stats()
        self.assertGreaterEqual(
            active["active_allocations"], baseline["active_allocations"] + 1
        )
        del tensor
        gc.collect()
        pytorch_vulkan.empty_cache()
        final = pytorch_vulkan.memory_stats()
        self.assertLessEqual(
            final["active_allocations"], baseline["active_allocations"]
        )
        self.assertTrue(pytorch_vulkan.shutdown())
        self.assertEqual(pytorch_vulkan.memory_stats()["active_pipelines"], 0)
        self.assertTrue(pytorch_vulkan.init())

    def test_pending_dispatches_are_bounded(self, device):
        baseline = pytorch_vulkan.memory_stats()
        threshold = baseline["auto_flush_threshold"]
        self.assertGreater(threshold, 0)
        if threshold > 4096:
            self.skipTest(
                "set PYTORCH_VULKAN_MAX_PENDING_DISPATCHES to at most 4096"
            )

        tensor = torch.zeros(1, device=device)
        baseline = pytorch_vulkan.memory_stats()
        for _ in range(threshold + 1):
            tensor = tensor + 1.0
        final = pytorch_vulkan.memory_stats()

        self.assertGreater(
            final["flush_generation"], baseline["flush_generation"]
        )
        self.assertGreaterEqual(
            final["total_dispatches"] - baseline["total_dispatches"],
            threshold + 1,
        )
        self.assertLess(final["pending_dispatches"], threshold)
        self.assertEqual(tensor.cpu(), torch.tensor([threshold + 1.0]))

    def test_strict_fallback_mode_is_thread_local(self, device):
        del device
        ready = threading.Event()
        release = threading.Event()
        worker_observations = []

        def worker():
            with pytorch_vulkan.strict_fallbacks():
                worker_observations.append(
                    pytorch_vulkan.fallback_stats()["strict"]
                )
                ready.set()
                release.wait(timeout=10)
            worker_observations.append(
                pytorch_vulkan.fallback_stats()["strict"]
            )

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(ready.wait(timeout=10))
        self.assertFalse(pytorch_vulkan.fallback_stats()["strict"])
        release.set()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_observations, [True, False])

    def test_immediately_reusable_allocator_entry(self, device):
        pytorch_vulkan.empty_cache()
        tensor = torch.empty(1024, dtype=torch.float32, device=device)
        del tensor
        gc.collect()
        cached = pytorch_vulkan.memory_stats()
        self.assertGreaterEqual(cached["cached_allocations"], 1)

        replacement = torch.empty(1024, dtype=torch.float32, device=device)
        reused = pytorch_vulkan.memory_stats()
        self.assertLess(
            reused["cached_allocations"], cached["cached_allocations"]
        )
        del replacement


instantiate_device_type_tests(
    TestVulkanIssue4,
    globals(),
    only_for="privateuse1",
)


if __name__ == "__main__":
    run_tests()
