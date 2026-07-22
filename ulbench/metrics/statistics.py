"""Bootstrap uncertainty helpers (spec §10.8, roadmap B-P0.6).

Percentile bootstrap with an explicit seed and replicate count so every
reported interval is reproducible from the manifest. The resampling unit is
whatever the caller passes — callers must respect the claim level (concepts
for concept-level claims, aligned pairs for paired comparisons).
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of no values")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    num_samples: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> dict:
    """Percentile bootstrap CI for the mean of *values*."""
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(num_samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "low": _percentile(means, alpha / 2),
        "high": _percentile(means, 1 - alpha / 2),
        "alpha": alpha,
        "num_samples": num_samples,
        "n": n,
        "seed": seed,
    }


def paired_bootstrap_diff_ci(
    first: Sequence[float],
    second: Sequence[float],
    *,
    num_samples: int = 2000,
    alpha: float = 0.05,
    seed: int,
) -> dict:
    """Percentile bootstrap CI for ``mean(first) - mean(second)``.

    *first* and *second* must be aligned pairs; indices resample together
    (spec §10.8: paired comparisons resample aligned IDs).
    """
    if len(first) != len(second):
        raise ValueError(
            f"paired bootstrap needs aligned sequences; got {len(first)} vs "
            f"{len(second)}"
        )
    if not first:
        raise ValueError("paired bootstrap needs at least one pair")
    diffs = [a - b for a, b in zip(first, second)]
    result = bootstrap_ci(diffs, num_samples=num_samples, alpha=alpha, seed=seed)
    result["mean_first"] = sum(first) / len(first)
    result["mean_second"] = sum(second) / len(second)
    return result


def mean_or_none(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None
