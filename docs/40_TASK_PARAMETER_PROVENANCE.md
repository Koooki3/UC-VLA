# 40 | Task-parameter provenance and visibility audit (SODH v2 → policies/learners), 2026-11-13

Audit of every key in `DecisionQuery.params` (what `PolicyInput.task.params` exposes to policies and learners). Classes: **P** public task instruction · **K** known environment prior (fixed mechanism constant, same for all episodes of a mechanism) · **O** estimable from observations · **H** hidden generator/simulator statistic.

| Family | Key | Source in `sodh2.py` | Class | Verdict |
| --- | --- | --- | --- | --- |
| track | `n_slots` | sampled n, observable from the first frames | P | ok |
| track | `sensor_eps` | declared sensor reliability (0.0/0.1) | K | ok (declared noise model; not an episode statistic) |
| track | `swap_rate` | **`round(#true swaps / horizon, 2)` computed from the TRUE swap sequence, including swaps hidden by occlusion** | **H** | **NOT legitimate**: per-episode hidden statistic. Used by the oracle (as the occlusion prior) and fed to learners |
| events | `target_count` | `K = (#accepted others) + 1`, derived from the hidden count | P (instruction) with a **design confound**: K is constructed from the true count, so the true count is always K−1 or K (tighter than any natural instruction) | confound documented |
| events | `jam`, `p_attempt`, `p_hold` | mechanism constants (train vs OOD mechanism) | K | ok |
| resume | `items` | task vocabulary (alphabetical) | P | ok |
| resume | `fail`, `removal` | mechanism constants | K | ok |
| resume | `p_attempt` | `round(#attempt events / horizon, 2)` over the true script (attempt events are observable; occluded results do not change the attempt count) | O (computable from observations except dropped ticks) | acceptable, flagged minor |
| control | — | none | — | ok |

**Findings.** (1) `swap_rate` is a hidden statistic given to both the posterior oracle and the learned readers (**Major**, to be treated as privileged unless replaced by a constant prior or an observation-based estimate). Effect on results is *unmeasured until the ablation below*; reported v2/G1/G2 numbers stand as run (inputs included `swap_rate`) and are not silently altered. (2) `events` K is constructed from the hidden count (design confound). (3) The oracle's "Bayes-optimal" claim for track holds only with the (hidden) per-episode `swap_rate`; with a fixed prior it is approximate.

**Ablations (GRU, balanced mixture N=3000 per family, seeds 0–2, `scripts/run_g1.py --drop-params`; compare with the 5-seed `docs/39` run restricted to the same seeds):** *full* (as reported) · *no `swap_rate`* (public + known priors) · *minimal* (only `n_slots`, `target_count`, item count, task id). *History-only vs parameter-only:* the memoryless runs of every G1 batch feed all parameters and no history and score PairOptimal 0.00 on all history-relevant families, i.e. **parameters alone carry no pair-level label information** (pairs share parameters). Results: see `docs/40` addendum when the runs finish; `results/synthetic/g1/params/`.

## Addendum — ablation results (GRU, balanced mixture N=3000/family, seeds 0–2; `results/synthetic/g1/params_ablation_report.json`; exploratory)
| inputs | track PairOptimal (3 seeds) | events | resume | memoryless track | order-perturbed track |
| --- | --- | --- | --- | --- | --- |
| full (as originally reported, incl. `swap_rate`) | 0.61 / 0.66 / 0.71 | 1.00 ×3 | 1.00 ×3 | 0.00 | 0.11–0.13 |
| without `swap_rate` | 0.79 / 0.66 / 0.61 | 1.00 ×3 | 1.00, 1.00, 0.99 | 0.00 | 0.13–0.15 |
| minimal params (no priors, no `swap_rate`) | 0.56 / 0.76 / 0.52 | 1.00 ×3 | 1.00 ×3 | 0.00 | 0.11–0.13 |
**Reading (3 seeds, wide spread — not significance-tested):** removing the hidden statistic, and even all numeric priors, does not reduce performance (track means 0.66 / 0.66 / 0.61 vs 0.66 full); history ablation still collapses to 0 and order perturbation still cuts track to ≈0.12, so the learners' history use is **not attributable to `swap_rate`**. The `events` result also does not depend on the numeric priors (K stays an instruction). The oracle's Bayes-optimality claim for track still relies on the per-episode `swap_rate` (documented limitation). The parameter-only control is the memoryless run (all parameters, no history): PairOptimal 0.00 everywhere. Reported earlier numbers stay as run; this addendum is additive. Remaining: the events-K construction confound (K ∈ {count, count+1}) is untested here.
