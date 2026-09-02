import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.ht_xpu.harness import create_workload, register_workload
from benchmarks.ht_xpu.run import _workload_options
from benchmarks.ht_xpu.workloads.convnext import ConvNeXtConfig, ConvNeXtWorkload


def _tiny_model(config: ConvNeXtConfig) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, config.num_classes),
    )


class TestConvNeXtConfig(unittest.TestCase):
    def test_typed_options(self):
        options = _workload_options(["batch-size=2", "synthetic=true", "pretrained=false"])
        config = ConvNeXtConfig.from_options(options)
        self.assertEqual(config.batch_size, 2)
        self.assertTrue(config.synthetic)
        self.assertFalse(config.pretrained)

    def test_unknown_option_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown ConvNeXt"):
            ConvNeXtConfig.from_options({"not_real": "1"})

    def test_legacy_factory_contract(self):
        name = "test-convnext-legacy"
        register_workload(name, lambda variant, device: (variant, device))
        self.assertEqual(
            create_workload(name, "cpu", "cpu", {"ignored": "1"}), ("cpu", "cpu")
        )


class TestConvNeXtCPU(unittest.TestCase):
    def _workload(self, **overrides):
        values = dict(
            batch_size=2, image_size=8, num_classes=4, workers=0,
            synthetic=True, pretrained=False, seed=7,
        )
        values.update(overrides)
        return ConvNeXtWorkload("cpu", "cpu", ConvNeXtConfig(**values), model_builder=_tiny_model)

    def test_full_training_step_updates_parameters_and_optimizer(self):
        workload = self._workload()
        self.assertFalse(workload.optimizer.defaults["foreach"])
        self.assertFalse(workload.optimizer.defaults["capturable"])
        before = [parameter.detach().clone() for parameter in workload.model.parameters()]
        loss = workload.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, workload.model.parameters())
            )
        )
        self.assertTrue(workload.optimizer.state)
        self.assertTrue(all(parameter.grad is not None for parameter in workload.model.parameters()))

    def test_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "convnext.pt"
            first = self._workload(checkpoint=str(path))
            first.step()
            first.finalize()
            expected = [parameter.detach().clone() for parameter in first.model.parameters()]

            resumed = self._workload(resume=str(path))
            self.assertEqual(resumed.completed_steps, 1)
            for wanted, actual in zip(expected, resumed.model.parameters()):
                torch.testing.assert_close(wanted, actual)
            self.assertTrue(resumed.optimizer.state)


@unittest.skipUnless(
    getattr(torch, "xpu", None) is not None and torch.xpu.is_available(),
    "requires Intel XPU",
)
class TestConvNeXtXPU(unittest.TestCase):
    def test_adamw_state_remains_on_xpu(self):
        config = ConvNeXtConfig(
            batch_size=2, image_size=8, num_classes=4, workers=0,
            synthetic=True, pretrained=False, seed=7,
        )
        workload = ConvNeXtWorkload(
            "xpu-eager", "xpu", config, model_builder=_tiny_model
        )
        workload.step()
        self.assertFalse(workload.optimizer.defaults["foreach"])
        self.assertTrue(workload.optimizer.defaults["capturable"])
        state_tensors = [
            value
            for state in workload.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertTrue(state_tensors)
        self.assertEqual({tensor.device.type for tensor in state_tensors}, {"xpu"})


if __name__ == "__main__":
    unittest.main()
