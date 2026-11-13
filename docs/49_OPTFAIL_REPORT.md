# 49 | Seed-0 optimisation failure — report (pre-registration: docs/48)

Condition B data, GRU, only `lr` varied; non-sealed splits; exploratory (5 seeds per cell; counts, not rates). Training success = events `test_id` accuracy ≥ 0.95.

| lr | seeds 0–4: train success | K≥6 length-OOD accuracy per seed |
| --- | --- | --- |
| 3e-3 (seeds 0–2 from docs/47, 3–4 new) | 4/5 (seed 0 failed, chance) | 0.50 / 0.98 / 0.96 / 0.91 / 0.80 |
| 1e-3 | 5/5 | 0.93 / 0.89 / 0.89 / 0.75 / 0.93 |
| 3e-4 | 5/5 | 0.64 / 1.00 / 0.96 / 0.96 / 0.93 |

## Verdicts
- **P-O1 (failure at 3e-3 ≥ 1/5): met**, exactly at the threshold (1/5): not a robust estimate of the failure rate.
- **P-O2 (smaller lr fails less): met in counts** (0/5 and 0/5 vs 1/5); with n=5 this is weak evidence, consistent with lr fragility, not proof.
- **P-O3 (successful runs reach K≥6 ≥ 0.90 regardless of lr): NOT met.** Training success does not imply good extrapolation: 5 of 14 successful runs are below 0.90 (0.64–0.89, e.g. lr 3e-4 seed 0 at 0.64, lr 1e-3 seed 3 at 0.75). Extrapolation at K≥6 is seed-variable even where in-distribution accuracy is 1.00.

## Consequences
The failure of docs/47 is best read as an optimisation fragility at lr 3e-3 on the widened mix, not a representational limit; but robust K extrapolation is **not** established (median of the 15 K≥6 values ≈ 0.93, range 0.50–1.00). No lr is selected for later use (docs/48). Pending from docs/47: PairOptimal and track/resume side effects of the widened mix under the standard G1 pipeline (P-K3), which has not been run. Candidate follow-ups (each needs its own pre-registration): curriculum over K, lr schedule with warmup, more seeds.
