# 53 | Track-collapse attribution — report (pre-registration: docs/52)

Standard G1 pipeline, GRU, non-sealed splits (exploratory). Pooled 10 seeds per data variant at lr 3e-3; test_id track PairOptimal, collapse = < 0.20.

| Cell | track per seed | collapses | events OOD-length PairOptimal (mean) |
| --- | --- | --- | --- |
| A: events 10–18 (docs/39 seeds 0–4 + R-A seeds 5–9) | 0.61 0.66 0.71 0.74 0.92 0.58 **0.00** 0.61 0.73 0.77 | **1/10** (seed 6) | 0.70 |
| B: events 10–26 (docs/51 + R-B seeds 5–9) | 0.79 **0.00** 0.68 0.79 0.71 0.73 0.81 0.87 0.67 0.87 | **1/10** (seed 1) | **0.90** |
| L: events 10–26, lr 1e-3, seeds 0–4 | 0.65 0.58 0.70 0.67 0.67 | 0/5 | 0.94 |

## Verdicts
- **P-T1 (B ≥ 2/10, A ≤ 1/10): NOT met** — B has 1/10. The pre-registered alternative reading holds: **A also collapses (1/10)**, so the track collapse is run-to-run optimisation fragility at lr 3e-3 that occurs with the original data too; no evidence that widening the events data causes it. Counts are 1 vs 1, so no difference could be detected at this n.
- **P-T2 (lr 1e-3 gives 0/5): met** (0/5; consistent with docs/49, but 5 seeds cannot show the rate is lower).
- **P-T3 (events OOD-length mean over 10 B seeds ≥ 0.80): met** (0.90; vs 0.70 for A, per-seed ranges 0.65–0.98 vs 0.52–0.82). The B minimum 0.65 is above the A mean.

## Observation (not pre-registered)
Excluding the collapsed seeds, mean track is 0.77 (B, 9 seeds) vs 0.70 (A, 9 seeds); not tested, may be chance.

## Consequence
The docs/51 P-G2 shortfall is attributed to the known lr-3e-3 fragility, not to the mix. Widening events horizon is retained as the data variant for later G1 work as an exploratory choice (events extrapolation mean 0.70 → 0.90 over 10 seeds each). Training reliability remains the open issue: candidate lr schedule/warmup, selection-rule changes, or more seeds, each needing a pre-registration. Sealed set untouched.
