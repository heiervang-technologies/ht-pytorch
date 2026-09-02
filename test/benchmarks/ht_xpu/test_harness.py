import tempfile
import unittest
from pathlib import Path

import torch

from benchmarks.ht_xpu.diagnostics import assert_tensor_residency
from benchmarks.ht_xpu.harness import BenchmarkConfig, BenchmarkRunner
from benchmarks.ht_xpu.report import render, validate_results
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

    def test_report_only_gates_ht_against_matching_configuration(self):
        def result(variant, throughput, config):
            samples = [4 / value for value in throughput]
            return BenchmarkResult(
                workload="tiny", variant=variant, device="xpu", warmup_steps=20,
                measured_steps=100, repeats=len(samples), samples_seconds=samples,
                throughput=throughput, metadata={"benchmark_config": config},
                metrics={
                    "latency_seconds": summarize(samples),
                    "throughput": summarize(throughput),
                },
            )

        baseline = result("xpu-compiled", [10] * 5, {"batch_size": 4})
        eager = result("xpu-eager", [5] * 5, {"batch_size": 4})
        candidate = result("ht", [12] * 5, {"batch_size": 4})
        report = render([baseline, eager, candidate])
        eager_row = next(line for line in report.splitlines() if "xpu-eager" in line)
        ht_row = next(line for line in report.splitlines() if " ht " in line)
        self.assertNotIn("FAIL", eager_row)
        self.assertIn("PASS", ht_row)

        mismatched = result("ht", [12] * 5, {"batch_size": 8})
        errors = validate_results([baseline, mismatched])
        self.assertTrue(any("exactly one" in error for error in errors))

    def test_report_rejects_incomplete_samples(self):
        invalid = BenchmarkResult(
            workload="tiny", variant="cpu", device="cpu", warmup_steps=1,
            measured_steps=1, repeats=2, samples_seconds=[1.0], throughput=[1.0],
            metrics={"latency_seconds": summarize([1.0]), "throughput": summarize([1.0])},
        )
        with self.assertRaisesRegex(ValueError, "sample count"):
            render([invalid], baseline="cpu")


if __name__ == "__main__":
    unittest.main()
