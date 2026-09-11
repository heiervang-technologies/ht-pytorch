from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    baseline_median: float
    candidate_median: float
    speedup: float
    ci_low: float
    ci_high: float
    gate: float

    @property
    def passes(self) -> bool:
        return self.speedup >= self.gate and self.ci_low > 1.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("at least one sample is required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("at least one sample is required")
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def compare(
    baseline: list[float], candidate: list[float], *, gate: float = 1.10,
    bootstrap_samples: int = 10_000, seed: int = 0,
) -> Comparison:
    """Compare throughput samples using an unpaired median bootstrap."""
    if not baseline or not candidate:
        raise ValueError("baseline and candidate samples are required")
    base_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    if base_median <= 0:
        raise ValueError("baseline throughput must be positive")
    rng = random.Random(seed)
    ratios = []
    for _ in range(bootstrap_samples):
        b = statistics.median(rng.choices(baseline, k=len(baseline)))
        c = statistics.median(rng.choices(candidate, k=len(candidate)))
        ratios.append(c / b)
    return Comparison(
        base_median, candidate_median, candidate_median / base_median,
        percentile(ratios, 0.025), percentile(ratios, 0.975), gate,
    )
