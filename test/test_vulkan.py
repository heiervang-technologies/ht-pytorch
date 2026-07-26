# Owner(s): ["oncall: mobile"]

import unittest
import torch
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io

@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanRewritePass(TestCase):
    @staticmethod
    def validate_transformed_module(
            # To please flake
            self,
            pattern_count_map,
            data_shape,
            prepack_removal=False,
            fuse_clamping_ops=False):
        module_instance = self
        scripted_model = torch.jit.script(module_instance)
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        scripted_model(input_data)
        torch._C._jit_pass_vulkan_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_vulkan_fold_prepacking_ops(scripted_model._c)

        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        deserialized_scripted_model = torch.jit.load(buffer)
        for pattern, v in pattern_count_map.items():
            if (v == 0):
                FileCheck().check(pattern).run(deserialized_scripted_model.graph)
            elif (v == -1):
                FileCheck().check_not(pattern).run(deserialized_scripted_model.graph)
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)

    def test_conv(self):
        # Conv params
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        class Conv2D(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv2d(x, self.weight, self.bias,
                                self.strides, self.paddings, self.dilations, self.groups)

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2D(), pattern_count_map, data_shape)

        class Conv2DRelu(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(), pattern_count_map, data_shape)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::relu"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


        class Conv2DHardtanh(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.hardtanh(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2DHardtanh(), pattern_count_map, data_shape)
        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DHardtanh(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::hardtanh"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanNativeOps(TestCase):
    def test_review_regressions(self):
        gather_input = torch.arange(24, dtype=torch.float).reshape(2, 3, 4)
        gather_index = torch.tensor(
            [[[3, 1], [0, 2], [1, 1]],
             [[2, 0], [3, 1], [0, 2]]],
            dtype=torch.long)
        expected_gather = torch.gather(gather_input, -1, gather_index)
        actual_gather = torch.gather(
            gather_input.vulkan(), -1, gather_index).cpu()
        self.assertEqual(actual_gather, expected_gather)

        scatter_src = torch.arange(12, dtype=torch.float).reshape(2, 3, 2)
        expected_scatter = gather_input.scatter(
            -1, gather_index, scatter_src)
        actual_scatter = gather_input.vulkan().scatter(
            -1, gather_index, scatter_src.vulkan()).cpu()
        self.assertEqual(actual_scatter, expected_scatter)

        topk_input = torch.randn(3, 2, 8)
        expected_values, expected_indices = torch.topk(topk_input, 3)
        actual_values, actual_indices = torch.topk(topk_input.vulkan(), 3)
        self.assertEqual(actual_values.cpu(), expected_values)
        self.assertEqual(actual_indices, expected_indices)

        norm_input = torch.randn(3, 2, 8)
        weight = torch.randn(8)
        bias = torch.randn(8)
        rms_expected = (
            norm_input *
            torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + 1e-5) *
            weight)
        rms_actual = torch.ops.vulkan_prepack.rms_norm(
            norm_input.vulkan(), weight.vulkan()).cpu()
        self.assertEqual(rms_actual, rms_expected, atol=1e-4, rtol=1e-4)

        layer_expected = torch.nn.functional.layer_norm(
            norm_input, (8,), weight, bias, 1e-5)
        layer_actual = torch.ops.vulkan_prepack.layer_norm(
            norm_input.vulkan(), weight.vulkan(), bias.vulkan()).cpu()
        self.assertEqual(layer_actual, layer_expected, atol=1e-4, rtol=1e-4)

        silu_input = torch.randn(3, 2, 16)
        gate, up = silu_input.chunk(2, dim=-1)
        silu_expected = torch.nn.functional.silu(gate) * up
        silu_actual = torch.ops.vulkan_prepack.silu_mul(
            silu_input.vulkan()).cpu()
        self.assertEqual(silu_actual, silu_expected, atol=1e-4, rtol=1e-4)

        condition = torch.tensor([[[True], [False]]])
        where_self = torch.randn(3, 2, 4, dtype=torch.float)
        where_other = torch.randn(1, 1, 4, dtype=torch.half)
        where_expected = torch.where(condition, where_self, where_other)
        where_actual = torch.where(
            condition, where_self.vulkan(), where_other).cpu()
        self.assertEqual(where_actual, where_expected)

        q8_weight = torch.randn(8, 4)
        q8_weight_vk, q8_bias_vk, q8_scale_vk, q8_zero_point = (
            torch.ops.vulkan_prepack.create_q8_linear(q8_weight, None))
        with self.assertRaisesRegex(RuntimeError, "exactly one input row"):
            torch.ops.vulkan_prepack.run_q8_linear(
                torch.randn(2, 8).vulkan(),
                q8_weight_vk,
                q8_bias_vk,
                q8_scale_vk,
                q8_zero_point.item())

        q8g_weight_vk, q8g_bias_vk, q8g_scale_vk, q8g_group_size = (
            torch.ops.vulkan_prepack.create_q8g_linear(q8_weight, None, 8))
        with self.assertRaisesRegex(RuntimeError, "exactly one input row"):
            torch.ops.vulkan_prepack.run_q8g_linear(
                torch.randn(2, 8).vulkan(),
                q8g_weight_vk,
                q8g_bias_vk,
                q8g_scale_vk,
                q8g_group_size.item())

        q4g_weight_vk, q4g_bias_vk, q4g_scale_vk, q4g_group_size = (
            torch.ops.vulkan_prepack.create_q4g_linear(q8_weight, None, 8))
        with self.assertRaisesRegex(RuntimeError, "exactly one input row"):
            torch.ops.vulkan_prepack.run_q4g_linear(
                torch.randn(2, 8).vulkan(),
                q4g_weight_vk,
                q4g_bias_vk,
                q4g_scale_vk,
                q4g_group_size.item())

        with self.assertRaisesRegex(
                RuntimeError, "divisible by 8 for int4 packing"):
            torch.ops.vulkan_prepack.create_q4g_linear(
                torch.randn(8, 4), None, 4)


if __name__ == "__main__":
    run_tests()
