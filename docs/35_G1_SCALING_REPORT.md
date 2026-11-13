# 35 | G1 scaling report (MLDA CPU batch, 2026-11-13; exploratory, protocol docs/33)

GRU (docs/25 hyper-parameters), base train split + N train-only auxiliary **track** pairs, N ∈ {0, 1000, 3000, 9000}, 5 seeds each (20 runs, same host, nice 19, checkpointed, no restarts needed), results pulled back with an identical sha256 manifest (`results/synthetic/g1/scale/MANIFEST.sha256`), cluster temporary files removed. Test splits were previously viewed ⇒ **exploratory**.

| N extra track pairs | track PairOptimal mean [min,max] | events | resume |
| --- | --- | --- | --- |
| 0 | 0.11 [0.07, 0.13] | 0.98 | 0.90 |
| 1000 | 0.33 [0.25, 0.48] | 0.97 | 0.93 |
| 3000 | 0.78 [0.52, 0.98] | 0.95 | 0.76 [0.19, 0.93] |
| 9000 | **0.96 [0.93, 0.99]** | **0.55 [0.00, 0.99]** | **0.30 [0.00, 0.96]** |

**Pre-registered predictions.** P-S1 (track rises monotonically with N and reaches ≥0.75 at 9000): **supported** (0.11→0.33→0.78→0.96); tracking in this benchmark is data-limited, not a benchmark or representation limit. P-S2 (between-seed range shrinks with N): **not supported as stated** (ranges 0.06, 0.23, 0.46, 0.06 — wide in the middle, tight again at 9000). P-S3 (memoryless stays 0): **supported** (0.00 at every N). P-S3b (no negative transfer > 0.05 on events/resume): **violated** — at N ≥ 3000 resume drops by 0.13 and at N = 9000 events by 0.44 and resume by 0.59, with some seeds collapsing to 0.00.
**Interpretation (hypothesis, not tested here):** the auxiliary data is track-only, so at 9000 pairs track makes up ≈ 90 % of the training examples; with early stopping on a val metric averaged over families, some seeds settle on a track-only solution (task imbalance / catastrophic interference), not a loss of capacity — a balanced-mixture rerun would test this. **Consequence:** the headline "GRU solves track with enough data" holds only together with this multi-task degradation; any release/claim about a single model must use a balanced mixture and a per-family model-selection rule fixed in advance.
