# 06 | Benchmark plan, leakage defense and datasets

## Evaluations (2026-11-11 existence verified)

| Suite | Function | Official source | Track |
| --- | --- | --- | --- |
| RoboMME | 16 tasks in Counting/Permanence/Reference/Imitation | https://github.com/RoboMME/robomme_benchmark | Main in-distribution |
| RoboMME-Plus | 16 tasks and more demanding count/motion/occlusion/history settings | https://robomme.github.io/plus.html | Main OOD, held out |
| RoboTwin-MeM | memory-dependent manipulation | https://arxiv.org/abs/2606.20092 | Optional after code/data review |
| RoboCasa-Mem | environment and process memory; behavior-conditioned evaluation | https://arxiv.org/abs/2611.06475 | Independent OOD if implementation accessible |
| LIBERO-Long | multitask long horizon but not always strict history-identifiability | https://libero-project.github.io/ | Regression guard, **not principal memory proof** |
| Custom toy / FR3 | causally paired histories + recovery + timing | Defined below | Novel diagnostic plus robot validation |

RoboMME project's docs report 100 train / 50 val / 50 test per task and fixed test episodes; preserve author protocol. RoboMME-Plus should be evaluated zero-shot before any few-shot adaptation; do not use its test metadata for training. EventVLA's RoboTwin-MeM release availability needs verification before integrating.

## Key diagnostic design: Same Observation / Different History (SODH)

In a simulator create matched pairs with identical final *rendered observation* (or quantified near equivalence), but distinct occluded historical state with divergent correct actions. If the physics diverges, it is allowed to have a different hidden world state by design. Key is to remove non-historical visual cues and match superficial statistics. Examples:

1. **Object correspondence:** cup swap with hidden object; choice depends on earlier relation, not final frame.
2. **Process memory:** visually identical post-press states, choose whether to press/count/release by past events.
3. **Task interleaving:** interrupt/resume, previously completed phase should be skipped even if final scene is similar.
4. **State uncertainty:** selectively remove evidence and require retrieve / reobserve / abstain before risky action.

Each pair includes `pair_id, episode_a, episode_b, o_t_reference, counterfactual_target, privileged_state_hash_train_only, accepted_sensory_tolerance, seed`. Exclude pair leakage across splits by shared scene script and object identity mapping.

## Data collection

- Simulation: 100+ matched pairs as first engineering sanity target. Final N chosen via pilot variance/power and hierarchical clustering, not arbitrary magic count.
- Real robot: collect only after human safety review; include documented action executed and verified outcomes; preserve raw timestamped evidence.
- Preprocess to agreed canonical camera/proprio/action frame contracts. Checks for dropped frames, drift, delayed feedback, unit mismatches and reset irregularity.
- Truncate by causal cutoff in each sample. Future event labels may be used as supervised *targets*, never as input.

## Splits

- `train`: can view privileged simulator state for labels.
- `val`: tune model and hyperparameters; freeze prompt templates, thresholds and budgets.
- `test_id`: identical distribution but disjoint episode groups.
- `test_ood`: changed count, occlusion length, distractor, layout, event frequency, execution delay.
- `test_counterfactual`: paired histories not visible during training.

Use consistent underlying scene/action distributions across methods; detect train-test duplicates with hash / perceptual dedup. Publish split manifest and seed lists.

## Negative controls

- Shuffle irrelevant history (should minimally affect decisions).
- Swap relevant history between paired cases (should change decisions in known direction).
- Memory-only oracle *from privileged state* (upper bound, labeled **not deployable**, not a fair baseline).
- No-history but larger backbone / more training steps (capacity control).
- Vision-only / current-image classifier (shortcut audit).

## Metric unit

Per-episode success with confidence intervals, history-dependent decision accuracy, group-matched paired success difference, cost per episode, p50/p95/p99 latency, memory storage, evidence retrieval precision/recall, state and event identity errors, calibration ECE/Brier, unsafe action frequency, recovery when original evidence is present/absent.

## Status note (2026-11-11, after adversarial review; see `docs/18`)

Implemented SODH **v1** is synthetic/symbolic and **frozen as a unit-test fixture**. In v1, `test_ood` means *longer horizon and larger object count only*, `test_counterfactual` means *unseen name combinations of seen atoms*, and `process_count` counts one `pressed` tick as one press. None of these supports a claim of structural/identity generalisation. The `test_ood` distractor/layout/delay shifts listed above are **planned for SODH v2**, together with tracking-based cups, a generated evidence-position axis and an alpha-invariant duplicate gate.
