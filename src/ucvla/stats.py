# SPDX-License-Identifier: Apache-2.0
"""Small deterministic statistics helpers (no numpy). Cluster unit = the pair."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ucvla.canonical import HashRng


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proportion, lo, hi) Wilson score interval; (nan, 0, 1) when n == 0."""
    if n == 0:
        return (math.nan, 0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: Sequence[float], seed: int, n_boot: int = 2000,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap over clusters (pairs); seeded, hence reproducible."""
    n = len(values)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    rng = HashRng("bootstrap", seed)
    means = sorted(sum(values[rng.below(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (sum(values) / n, lo, hi)


def paired_diff_ci(a: Sequence[float], b: Sequence[float], seed: int, n_boot: int = 2000,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of (a - b) with a paired cluster-bootstrap interval."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_mean_ci([x - y for x, y in zip(a, b)], seed, n_boot, alpha)
