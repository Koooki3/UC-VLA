# SPDX-License-Identifier: Apache-2.0
"""Exact paired tests and multiplicity control (pure Python). Cluster unit = pair."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb


def sign_test(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float]:
    """Exact two-sided McNemar/sign test on paired binary outcomes (a vs b per cluster)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return {"n_discordant": 0, "a_only": 0, "b_only": 0, "p": 1.0}
    k = min(only_a, only_b)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return {"n_discordant": n, "a_only": only_a, "b_only": only_b, "p": min(1.0, 2 * tail)}


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out
