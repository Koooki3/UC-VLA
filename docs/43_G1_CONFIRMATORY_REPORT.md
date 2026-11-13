# 43 | G1 confirmatory report — sealed `sodh_v2c` (same generator, new random seeds), 2026-11-13

Protocol `docs/41` (frozen before the set was generated); models GRU and fixed-latent S=4, train = v2 train + balanced auxiliary mix (3,000 pairs/family), val-only early stopping, 5 training seeds each, every run evaluated **once** on the sealed set (940 pairs; `results/synthetic/g1/confirm/`, report `g1_confirm_report.json`, aggregator `scripts/aggregate_g1_confirm.py` using the corrected `apply_criteria`). Sealed registry updated: 10 model evaluations, no tuning afterwards. **Scope:** replication on unseen scene draws of the same generator — not an independent environment; resume item names are canonicalised, so counterfactual is not an identity test.

## Verdict under the pre-registered criteria (all conditions incl. order perturbation)
**GRU: PASS. Fixed-latent S4: PASS** — on 2 of 3 families (events, resume); control accuracy and swap invariance 1.00; the track order-perturbation drop is ≥ 0.15 (GRU 0.49, FL4 0.18, the latter marginal). Track fails the family criteria for both models.

| test_id PairOptimal | GRU pooled [Wilson 95 %] | GRU seeds (min–max) | FL4 pooled [Wilson] | FL4 seeds |
| --- | --- | --- | --- | --- |
| track | 0.59 [0.55, 0.63] | **0.00 – 0.77** | 0.27 [0.24, 0.30] | 0.03 – 0.45 |
| events | 0.99 [0.98, 0.99] | 0.98 – 1.00 | 0.98 [0.97, 0.99] | 0.98 |
| resume | 1.00 [0.99, 1.00] | 1.00 | 0.99 [0.98, 1.00] | 0.97 – 1.00 |
Memoryless (same model without history): 0.00 on every family. Swap-follow: events 0.99, resume 1.00 (GRU/FL4), track 0.71/0.47. Critical-minus-non-critical deletion gap: events 0.41/0.33, resume 0.72/0.73, track 0.30/0.14. Order perturbation: events 0.93/0.92 and resume 0.98 (order-insensitive tasks by construction), track 0.10/0.09.

## Negative results and failure analysis (kept)
- **Track is unreliable.** GRU has one seed that collapsed to 0.00 (early-stopped at 14 epochs on a track-less solution; reported, not dropped) and a 0.77 best seed; FL4 never exceeds 0.45. The registered seed-range rule would reject track even if the mean passed. Consistent with the data-limitation finding (docs/35, 39); not a confirmation that tracking is learned reliably.
- **Length OOD (pre-declared failure analysis).** Events collapse beyond training lengths: 0.52 (GRU) / 0.51 (FL4) vs 0.99 in-distribution; track 0.45/0.14; resume holds (0.997/0.987). The counting representation does not extrapolate in length; **cause not yet isolated** (candidate: state saturation/threshold tied to training horizons) — analysis pending, no repair claimed.
- **Structural OOD:** events 0.97–0.98, resume 0.97–0.98, track 0.58 (GRU) / 0.22 (FL4).
- Uncertainty reported at pair level (Wilson over pooled pairs), seed level (min–max, clipped t-interval) — bounded in [0,1]; scene level equals pair level (unique templates).

## Claim status
Replication of history-dependent learning for events and resume on unseen draws from the same generator: **supported (exploratory-to-confirmatory within scope)**; learned tracking: **not confirmed**; length extrapolation: **fails**; compression/G2/real VLA: **not addressed**. Still not independent-environment validation; the sealed set is now used (further tuning on it would burn it).
