import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.ht_xpu.harness import create_workload, register_workload
from benchmarks.ht_xpu.diagnostics import assert_tensor_residency
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
    def _workload(self, variant="cpu", **overrides):
        values = dict(
            batch_size=2, image_size=8, num_classes=4, workers=0,
            synthetic=True, pretrained=False, seed=7,
        )
        values.update(overrides)
        return ConvNeXtWorkload(
            variant, "cpu", ConvNeXtConfig(**values), model_builder=_tiny_model
        )

    def test_full_training_step_updates_parameters_and_optimizer(self):
        workload = self._workload()
        self.assertFalse(workload.optimizer.defaults["foreach"])
        self.assertFalse(workload.optimizer.defaults["capturable"])
        self.assertFalse(workload.optimizer.defaults["fused"])
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

    def test_ht_selects_fused_adamw_and_preserves_full_step(self):
        workload = self._workload(variant="ht")
        self.assertFalse(workload.optimizer.defaults["foreach"])
        self.assertFalse(workload.optimizer.defaults["capturable"])
        self.assertTrue(workload.optimizer.defaults["fused"])
        before = [parameter.detach().clone() for parameter in workload.model.parameters()]
        images, targets = next(workload.batches)

        loss = workload._train_step(images, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, workload.model.parameters())
            )
        )
        self.assertTrue(workload.optimizer.state)
        self.assertTrue(
            all(parameter.grad is not None for parameter in workload.model.parameters())
        )
        assert_tensor_residency(workload.residency_objects(), "cpu")


@unittest.skipUnless(
    getattr(torch, "xpu", None) is not None and torch.xpu.is_available(),
    "requires Intel XPU",
)
class TestConvNeXtXPU(unittest.TestCase):
    def _workload(self, variant):
        config = ConvNeXtConfig(
            batch_size=2, image_size=8, num_classes=4, workers=0,
            synthetic=True, pretrained=False, seed=7,
        )
        return ConvNeXtWorkload(
            variant, "xpu", config, model_builder=_tiny_model
        )

    def test_adamw_state_remains_on_xpu(self):
        workload = self._workload("xpu-eager")
        workload.step()
        self.assertFalse(workload.optimizer.defaults["foreach"])
        self.assertTrue(workload.optimizer.defaults["capturable"])
        self.assertFalse(workload.optimizer.defaults["fused"])
        state_tensors = [
            value
            for state in workload.optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ]
        self.assertTrue(state_tensors)
        self.assertEqual({tensor.device.type for tensor in state_tensors}, {"xpu"})

    def test_ht_fused_adamw_full_step_remains_on_xpu(self):
        workload = self._workload("ht")
        images, targets = next(workload.batches)
        images = images.to("xpu", memory_format=torch.channels_last)
        targets = targets.to("xpu")
        loss = workload._train_step(images, targets)
        torch.xpu.synchronize()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(workload.optimizer.defaults["fused"])
        assert_tensor_residency(workload.residency_objects(), "xpu")


if __name__ == "__main__":
    unittest.main()
