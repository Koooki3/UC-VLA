# 45 | Events length-OOD collapse: diagnosis on non-sealed data (2026-11-13; exploratory, 3 seeds)

Script `scripts/diag_events_length.py`, output `results/synthetic/g1/events_length_diagnosis.json`. GRU, docs/25 hyper-parameters, train = v2 train + balanced mix 3,000/family (as in docs/43), 3 seeds; evaluated on the **non-sealed** v2 `test_id` and `test_ood_length` events pairs (the sealed set was not touched). Background: the confirmatory runs showed events 0.99 in-distribution vs 0.52 on length-OOD, resume unaffected.
**Hypotheses.** H1: failure appears when the target count K exceeds the well-covered training range (counter/threshold saturation). H2: failure comes from sequence length per se. **Controlled test for H2:** in-distribution histories padded with 0/10/20 blank ticks at the start (same evidence, longer sequence, same K).

| evidence | result |
| --- | --- |
| H2 control (padding, in-dist, 3 seeds) | accuracy 1.00 → 1.00/0.99/1.00 (+10 ticks) → 1.00/0.96/1.00 (+20): **length alone costs ≤ 0.04** |
| OOD-length accuracy by target count K (mean of 3 seeds; n pairs-episodes) | K=2: 1.00 (2), K=3: 1.00 (18), K=4: 0.92 (22), K=5: 0.81 (34), K=6: 0.69 (26), K=7: 0.53 (10), K≥8: 0.50 (8) |
| in-dist by K (K=2–5) | 1.00 throughout |
| OOD by horizon bucket | 25–29: 0.90, 30–34: 0.73, 35–39: 0.59 (horizon and K are correlated: longer histories contain more accepted triggers) |
| training coverage | K range seen in training data: 2–7 (auxiliary data included); in-dist test K only 2–5 |

**Reading.** H2 is **not supported** (length without extra evidence is nearly harmless). The degradation tracks the **target count K**: perfect up to K=3, falling steadily from K=4 and reaching chance at K≥7–8 — consistent with H1 (the learned counter/threshold generalises poorly to larger counts than those well represented in training), with the caveat that K and horizon co-vary in the OOD split, only partly disentangled by the padding control (which holds K fixed). K=5 is 1.00 in-dist (n=16) but 0.81 OOD, so longer sequences with more events/dropouts add some further harm beyond K. **Not tested:** training-set K coverage frequencies; a retraining with K-balanced data or counting curriculum; other architectures. **No repair is claimed**; a K-covering retrain would be a new pre-registered experiment. Exploratory (3 seeds).
