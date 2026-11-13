# 47 | K-coverage retraining — report (pre-registration: docs/46)

Exploratory; GRU, 3 training seeds per condition, non-sealed v2 `test_id` / `test_ood_length`, sealed set untouched. Metric reported is **episode accuracy** (the diagnosis script does not compute PairOptimal), so P-K2 as worded (PairOptimal) is **not evaluated**; P-K3 (track/resume) is **not evaluated** (the script measures events only). Nothing was tuned after seeing results.

## Results (events, `test_ood_length`, accuracy; per seed 0 / 1 / 2)

| Condition | train K range | all K | K≥6 | in-dist (test_id) |
| --- | --- | --- | --- | --- |
| A baseline (horizon 10–18) | 2–7 | 0.84 / 0.65 / 0.89 | 0.64 / 0.50 / 0.73 | 1.00 / 1.00 / 1.00 |
| B (horizon 10–26) | 2–10 | **0.50** / 0.99 / 0.98 | **0.50** / 0.98 / 0.96 | **0.50** / 1.00 / 1.00 |
| C (horizon 10–36) | 2–11 | **0.50** / 0.99 / 0.98 | **0.50** / 0.98 / 0.96 | **0.53–0.56** / 1.00 / 1.00 |

## Findings
1. **Coverage repairs the collapse when training succeeds.** With K up to 10–11 in training, seeds 1–2 reach 0.96–0.98 at K≥6 (A: 0.50–0.73). This supports H-K for those seeds: the failure at large K is a training-coverage effect, not an inherent counter limit of the GRU.
2. **Seed 0 failed to train at all in B and C** (chance accuracy even on in-distribution `test_id`, padded control also 0.5), so 1/3 seeds are optimisation failures. This did not occur in A. Longer sequences/larger K appear to make optimisation fragile; the cause (learning-rate/patience schedule, curriculum) is untested. These seeds are reported, not excluded.
3. **P-K1 (A < B < C, C ≥ 0.80 at K≥6): not met as worded.** B and C are indistinguishable (seed-wise identical to the reported precision on seeds 1–2, plausibly the same few hard episodes at K=6 and horizon 35; not independently verified), so there is no monotone B < C step, and the 3-seed mean in C (0.81) is carried by 2 seeds. A≈partial-coverage B already suffices when training works. The pre-registered failure criterion "C < 0.80 at K≥6 ⇒ representational limit" is **not** triggered per seed (2/3 reach ≥0.95), but the seed-0 failures mean reliability is **not** established.
4. C is not a generalisation test (length no longer OOD); B (horizon 10–26, OOD horizons 30–35 still beyond training) is the informative one: it extrapolates from K≤10 to K up to ~10 with 0.96–1.0 when it trains.

## Not shown / limits
No claim about robust length extrapolation; K and horizon remain partly confounded (docs/45); track/resume side effects of the widened mix were not measured (P-K3 open); 3 seeds, non-sealed splits previously viewed; synthetic only.

## Next
Diagnose the seed-0 optimisation failure (pre-register: lr / warmup / seed grid on non-sealed data); measure P-K3 and PairOptimal with the standard G1 pipeline before any statement about the mix.
