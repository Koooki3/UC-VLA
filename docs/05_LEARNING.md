# 05 | Learning algorithm and ablation-compatible stages

## Dataset representation

`Episode = {episode_id, task, instructions, cameras[frames,timestamp], proprio[timestamp], actions_commanded, actions_executed, events, measured_outcome, optional privileged_sim_state}`. Privileged state may supervise training/evaluation in sim but must not enter online policy observation or test-time Context.

Classify every sample `source_type = observed | measured | annotation | inferred | simulated_privileged`. Mask labels by causal cutoff at sample construction. Store partition/group key by physical scene and trajectory rather than filename to avoid leakage.

## Stage 0 — Establish evaluation causality

Generate pairs `(h1,h2,o_t,g)` with identical current observation but divergent valid actions because of prior occluded event, object identity, timer or process state. Ensure physics state consistency and no deterministic shortcuts. Add non-history tasks as regression guards.

## Stage 1 — Reader/adapter viability

Freeze pretrained visual backbone and action expert. Train small context reader + contextual encoder using teacher-forcing and history-derived state labels. Optimize original FM action loss plus future event/identity probes. Control: equal parameter and equal data fixed-memory reader. **Pass condition**: deleting/swapping relevant history affects action in the expected direction, not just probe accuracy.

## Stage 2 — Decision-aware utility and compression

Use resettable sim and fixed policy to gather memory interventions: keep, remove, merge, retrieve, corrupt candidate event; controlled environment seed. Evaluate *future return*, history-dependent action correctness and cost. Train utility model `U_phi(e,z,g)` with pairwise ranking calibrated against rollout outcome. For small cases enumerate subsets, larger ones use causal item groups and conditional sampling. Preserve interactions; no claim of additivity.

Training objectives (proposed, not optimized weights):

```
L = L_flow + λ_state L_state + λ_value L_value + λ_rank L_intervention
    + λ_cal L_calibration + λ_budget C_measured.
```

`L_value`: value distribution regression or ranking vs paired rollout targets, with bootstrap interval. `L_flow`: same as baseline action expert flow matching. `L_state`: evidence-grounded task/object progress targets. `L_calibration`: proper scoring/Brier or NLL on held-out conditions. λ tuned only on validation; record sweep.

## Stage 3 — Management policy: offline first

Controller outputs candidate operations and budgets; learned policy initialized by teacher heuristics and supervised intervention labels. Enforce operation masks/invariants at API boundary. Transition to offline policy optimization / conservative methods using fully logged behavior policies; OPE limitations acknowledged.

## Stage 4 — RL in simulator

Optimize manager with fixed action policy first; compare to learned writer-only and fixed manager. Reward combines task return, storage/compute cost and real-time lateness. Safety as explicit constraint/termination, not only penalty. Train with multiple environment seeds, many task families and reward scales; re-check variance and calibration. Joint fine-tuning of action expert occurs **only after** manager-only gains are established; use trust-region/proximal updates to avoid wiping pretrained control.

## Stage 5 — Real world (human authorized)

First observation-only shadow mode for memory accuracy, provenance, latency and plan validity. Then restricted non-contact tasks, then risk-assessed contact tasks. No autonomous RL exploration on physical robot without separate authorization and safety process. Hardware tests do not use simulator privileged state.

## Preventing shortcut learning

- Evaluate true shuffled and counterfactual *valid* histories, not arbitrary invalid attention tensors.
- Include a `history irrelevant` negative control; swapping past should not affect action there.
- Train/test split by latent object associations, exact event sequences, camera setup, and horizon length where practical.
- Counterbalance location/color/target/order, randomize distractors, and test changed embodiment dynamics.
- Use memory-first two-stage training as a fair shared optimization baseline; compare against it rather than claiming it as original.

## Model-family adaptation

The first adapter should support one pretrained flow VLA (prefer available OpenPI checkpoint) and one compact toy policy. Keep `PolicyAdapter` API stable. Report checkpoint, license, action normalization, calibration, camera alignment and training data overlap. For FAST tokenized autoregressive policies a separate adapter is needed; do not assume FM and FAST action heads are interchangeable.

## Training reproducibility

Record seed for model weights, dataloader, environment, simulator physics and memory intervention; model optimizer, LR schedule, batch size, accumulation, mixed precision dtype, GPU model, package versions, and artifacts hashed. Resume from optimizer state only under matching config/data fingerprints.
