import unittest

import torch
from torch._inductor import config
from torch._inductor.decomposition import decompositions
from torch._subclasses.fake_tensor import FakeTensorMode


class TestXPUNativeLayerNormConfig(unittest.TestCase):
    def setUp(self):
        mode = FakeTensorMode()
        with mode:
            self.input = torch.randn(2, 4, 8, device="xpu")
            self.grad = torch.randn_like(self.input)
            self.weight = torch.randn(8, device="xpu")
            self.bias = torch.randn(8, device="xpu")

    def test_forward_decomposition_is_preserved_by_default(self):
        decomposition = decompositions[torch.ops.aten.native_layer_norm.default]
        result = decomposition(
            self.input, [8], self.weight, self.bias, 1e-6
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        with config.patch({"xpu.use_native_layer_norm": True}):
            result = decomposition(
                self.input, [8], self.weight, self.bias, 1e-6
            )
        self.assertIs(result, NotImplemented)

    def test_backward_decomposition_is_opted_out_with_forward(self):
        forward = decompositions[torch.ops.aten.native_layer_norm.default]
        output, mean, rstd = forward(
            self.input, [8], self.weight, self.bias, 1e-6
        )
        self.assertEqual(output.shape, self.input.shape)
        backward = decompositions[
            torch.ops.aten.native_layer_norm_backward.default
        ]
        result = backward(
            self.grad,
            self.input,
            [8],
            mean,
            rstd,
            self.weight,
            self.bias,
            [True, True, True],
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        with config.patch({"xpu.use_native_layer_norm": True}):
            result = backward(
                self.grad,
                self.input,
                [8],
                mean,
                rstd,
                self.weight,
                self.bias,
                [True, True, True],
            )
        self.assertIs(result, NotImplemented)


if __name__ == "__main__":
    unittest.main()
