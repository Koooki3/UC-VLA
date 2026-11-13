# 39 | G1 mixture report (MLDA CPU, 10 runs, 2026-11-13; exploratory; protocol docs/36)

GRU, docs/25 hyper-parameters and the unchanged model-selection rule; base train split + train-only balanced auxiliary data, N pairs **per relevant family**; 5 seeds; results pulled back with an identical sha256 manifest (`results/synthetic/g1/mix/MANIFEST.sha256`); cluster temporary files cleaned. Test splits previously viewed ⇒ exploratory.

| N per family | track | events | resume | memoryless track | OOD-structure track/events/resume | OOD-length track/events/resume |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (baseline, docs/35) | 0.11 | 0.98 | 0.90 | 0.00 | 0.10 / 0.90 / 0.77 | 0.07 / 0.31 / 0.89 |
| 1000 | 0.38 [.33,.45] | 1.00 [1,1] | 0.99 [.97,1.0] | 0.00 | 0.29 / 0.98 / 0.95 | 0.17 / 0.57 / 0.97 |
| 3000 | 0.73 [.61,.92] | 1.00 [1,1] | 1.00 [.99,1.0] | 0.00 | 0.56 / 0.98 / 0.99 | 0.49 / 0.70 / 1.00 |

**Pre-registered predictions, literal outcome.** P-M1 (events/resume within 0.05 of baseline): *literally violated* at both N because resume **improved** by 0.09–0.10 (0.90 → 0.99/1.00) — the criterion was intended to detect degradation but was written two-sided; no degradation occurred. P-M2 (track ≥ 0.75 at N=3000): **narrowly not met** (0.73, range 0.61–0.92). P-M3 (no seed below 0.50 on any family): met at N=3000, not at N=1000 (track seeds 0.33–0.45).
**Interpretation.** The collapse in `docs/35` was task imbalance, not capacity or unavoidable interference: with balanced data nothing degrades, events and resume reach ≈1.0, and track still climbs (0.11 → 0.38 → 0.73; track-only data reached 0.78 at 3k and 0.96 at 9k, so balanced mixing costs some track accuracy at equal track count). Extra data also improved length extrapolation of counting (events OOD-length 0.31 → 0.57 → 0.70) and structural OOD for track (0.10 → 0.29 → 0.56). Still data-limited for track; no run used the real test labels for selection. **Not shown:** that these learners compress or select memory (G2), anything about real VLAs, or confirmatory validity. Next, if pursued: balanced N = 9000/family and a fixed per-family model-selection rule (pre-registered), plus a sealed confirmatory seed set.
