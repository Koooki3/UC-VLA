#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired-sample sizes (McNemar normal approximation) for M3 rollout planning."""
import json
from pathlib import Path


def min_pairs(delta: float, p_disc: float, z_a: float = 1.959964, z_b: float = 0.841621) -> int:
    """Connor (1987) McNemar sample size: n = ((z_a*sqrt(pd) + z_b*sqrt(pd - d^2)) / d)^2 (approximation)."""
    from math import ceil, sqrt
    return ceil(((z_a * sqrt(p_disc) + z_b * sqrt(p_disc - delta**2)) / delta) ** 2)


rows = []
for p_disc in (0.2, 0.4):
    for delta in (0.05, 0.10, 0.15, 0.20):
        n = min_pairs(delta, p_disc)
        rows.append({"p_discordant": p_disc, "true_difference": delta, "pairs_for_80pct_power": n,
                     "holm_m5_pairs": min_pairs(delta, p_disc, z_a=2.5758293)})
out = {"test": "McNemar, two-sided alpha=0.05, power 0.8, Connor normal approximation (the exact test needs slightly more)",
       "note": "pairs = same simulator seed/state run under both memory conditions; Holm over the family "
               "multiplies the required n (use alpha/m as a conservative bound)", "rows": rows}
Path(__file__).resolve().parents[1].joinpath("results/synthetic/m3_power.json").write_text(json.dumps(out, indent=1) + "\n")
for r in rows:
    print(r)
