import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.ht_xpu.diagnostics import assert_tensor_residency
from benchmarks.ht_xpu.harness import BenchmarkConfig, BenchmarkRunner
from benchmarks.ht_xpu.report import render
from benchmarks.ht_xpu.schema import BenchmarkResult
from benchmarks.ht_xpu.stats import compare, percentile, summarize


class TinyTrainingWorkload:
    name = "tiny"
    units_per_step = 4

    def __init__(self):
        self.model = torch.nn.Linear(4, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.inputs = torch.ones(4, 4)

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model(self.inputs).sum().backward()
        self.optimizer.step()

    def residency_objects(self):
        return [self.model, self.optimizer, self.inputs]


class TestStats(unittest.TestCase):
    def test_summary_and_percentile(self):
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        self.assertEqual(summarize([1, 2, 3])["median"], 2)

    def test_comparison_gate(self):
        result = compare([10] * 5, [12] * 5, bootstrap_samples=100)
        self.assertTrue(result.passes)
        self.assertAlmostEqual(result.speedup, 1.2)


class TestHarness(unittest.TestCase):
    def test_cpu_run_round_trip_and_report(self):
        result = BenchmarkRunner(BenchmarkConfig("cpu", "cpu", 1, 2, 3)).run(TinyTrainingWorkload())
        self.assertEqual(len(result.samples_seconds), 3)
        self.assertGreater(result.metrics["throughput"]["median"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = result.write(Path(directory) / "result.json")
            restored = BenchmarkResult.read(path)
        self.assertEqual(restored.workload, "tiny")
        self.assertIn("tiny", render([restored], baseline="cpu"))

    def test_residency_failure(self):
        with self.assertRaisesRegex(RuntimeError, "residency"):
            assert_tensor_residency([torch.ones(1)], "meta")


if __name__ == "__main__":
    unittest.main()
