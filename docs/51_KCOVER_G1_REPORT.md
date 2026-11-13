# 51 | K-coverage mix under the standard G1 pipeline — report (pre-registration: docs/50)

GRU, docs/25 hyper-parameters, 5 seeds, local CPU, non-sealed splits (exploratory), auxiliary data = docs/39 N=3000 with events horizon 10–26. PairOptimal (`pair_both_correct`), per seed 0–4:

| split / family | per seed | mean | docs/39 reference (N=3000) |
| --- | --- | --- | --- |
| test_id track | 0.79 / **0.00** / 0.68 / 0.79 / 0.71 | 0.59 | 0.73 |
| test_id events | 1.00 / 0.99 / 1.00 / 1.00 / 1.00 | 1.00 | 1.00 |
| test_id resume | 0.99 / 1.00 / 0.99 / 1.00 / 0.99 | 1.00 | 1.00 |
| OOD-length events | 0.65 / 0.85 / 0.85 / 0.93 / 0.98 | **0.85** | 0.70 |
| OOD-length resume | 1.00 / 0.97 / 1.00 / 1.00 / 1.00 | 0.99 | 1.00 |
| OOD-length track | 0.53 / 0.00 / 0.48 / 0.63 / 0.52 | 0.43 | 0.49 |
| OOD-structure events / resume | 0.95–0.98 / 0.97–1.00 | 0.98 / 0.98 | 0.98 / 0.99 |

## Verdicts
- **P-G1 (events, resume within 0.05): met** (1.00, 1.00).
- **P-G2 (track ≥ 0.68): NOT met** — mean 0.59, driven by seed 1 collapsing to 0.00 (early-stopped at epoch 25, train accuracy 0.75, val 0.70). The other four seeds are 0.68–0.79 (mean 0.74, i.e. like the reference). Seed-matched reference (docs/39 N=3000, same seed 1, events horizon 10–18): track 0.66, val 0.92, 45 epochs — no collapse. Changing the auxiliary data also changes the batch order and trajectory for the same seed, so this does not isolate the cause: it shows the collapse is **not a property of seed 1 alone** but cannot separate an effect of the widened data from run-to-run optimisation fragility (docs/49). Undetermined; resolving it needs more seeds on both data variants.
- **P-G3 (events OOD-length > 0.70): met**, 0.85 (seed range 0.65–0.98); the standard metric confirms the docs/47 diagnosis, with the large seed spread remaining.
- **P-G4 (no events seed at chance): met** (the collapse occurred in track, not events).

## Reading and limits
Widening the events horizon of the auxiliary data improves events length extrapolation (0.70 → 0.85, mean) without hurting events/resume in-distribution; the one adverse signal is a track collapse in 1/5 seeds. Not shown: reliability across seeds (events OOD-length min 0.65, track one failure), anything on sealed data, any independence from the generator. Exploratory; negative result P-G2 reported as is, mix not tuned further. Next candidate (needs its own pre-registration): matched-seed control for the seed-1 track collapse, lr 1e-3 repeat of this run.
