# 07 | Preregistered experimental program

## Primary hypotheses (freeze before test evaluation)

H1: With identical backbone/training inputs and memory token budget, future-decision utility memory exceeds strongest action-MI or fixed compression baseline on held-out history-sensitive success.

H2: Provenance-aware verification + recovery improves failure recovery on evidence-corruption/interruption without reducing ordinary-task success outside an agreed non-inferiority margin.

H3: Adaptive memory budget improves the success-vs-actual GPU compute / p95 control-latency Pareto frontier over fixed budget at matched training cost.

H4: Versioned memory + chunk invalidation reduces stale-action mistakes under object/goal transitions while preserving control-rate viability.

## Main matrix

| ID | Memory | Utility | Recovery | Budget | Control | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | none | — | — | 0 | frozen base | lower bound |
| B1 | uniform frame selection | — | — | fixed | same | naive control |
| B2 | fixed latent | action loss | — | fixed | same | simple learned |
| B3 | full history | — | — | matched wall-clock when feasible | same | high-cost ceiling |
| B4 | D&R reproduced | one-step action MI | — | fixed | same | primary prior art |
| B5 | Mem4Act reproduced | memory-first supervision | — | fixed | same | latent prior art |
| B6 | strongest released benchmark-specific model | original method | original | original | method-specific | external reference, **not causal comparison** |
| U1 | UC-VLA | future-decision utility | off | fixed | same | isolate new objective |
| U2 | UC-VLA | future-decision utility | on | fixed | same | recovery contribution |
| U3 | UC-VLA | future-decision utility | on | adaptive | same | dynamic budget contribution |
| U4 | UC-VLA | future-decision utility | on | adaptive | rolling | chunk coherence contribution |

Ablate: state labels, evidence provenance, future utility target, Q ensemble, uncertainty calibration, retrieval, stale-plan detector, event clock, fixed vs variable memory, task-centric vs task-agnostic memory, training sequence length.

## Fairness constraints

- Two separate comparison axes: **equal hot-memory tokens** and **equal measured end-to-end wall-clock latency / GPU FLOPs**. Never claim one implies the other.
- Same starting weights, policy parameter trainability, training steps/episodes, augmentations, test seeds, sensors and action rates for causal ablations. If a prior art method requires different adaptation recipe, show both original best setup and matched controlled version.
- Do not optimize target method on test split; fixed benchmark protocol and secret test seeds where available.
- Include external published results only in labeled non-comparable table.

## Statistical protocol

- Primary unit is task/scene *cluster*, then paired episode; avoid treating correlated frames as independent samples.
- Report per-task success and aggregate mean with paired bootstrap over task families and seed blocks; include 95% interval, raw counts and signed differences.
- At least 3 independent model-training seeds for primary comparisons as an initial target; final design based on resource and pilot variance. Report both between-seed and episode variance.
- Choose a single primary metric and key test family before training; separate H1/H2/H3 multiplicity (e.g. Holm correction) and exploratory analyses.
- Non-inferiority threshold for latency/ordinary tasks must be registered **after pilot but before test**; do not choose it from favorable outcomes.
- Keep a compute ledger: GPU SKU/count, GPU-hours, driver/CUDA/torch, BF16/FP16, step count, token count, p95 latency, service load.

## Power and resources

No fixed GPU cluster was provided. Run a pilot to estimate GPU-hours/episode and success variance, calculate a budget allocation by stages, then produce a resource reservation request for human approval. Never fabricate an 8×H100 estimate. `configs/cluster/slurm_train.sbatch` only demonstrates command structure and deliberately requires operator inputs.

## Go/no-go gates

G0 causal data suite passes anti-shortcut audits; G1 adapter evidence interventions have expected impact; G2 H1 sign + CI and matched-budget advantage hold on at least two task families; G3 learned recovery exceeds oracle-free controls without unacceptable false-stop; G4 p95 deadline and safety constraints; G5 OOD generalization; G6 paper replicability. If G2 fails: pivot to benchmark/negative result rather than piling modules.

## Repro artifacts

`runs/<run_id>/manifest.json`, `config.yaml`, `git_commit.txt`, `dataset_fingerprints.json`, `third_party_manifest.json`, `episode_results.parquet`, `latency.jsonl`, `memory_patches.jsonl`, `training_trace.jsonl`, `metrics.json`, `videos/` if approved. Enforce content hashes and read-only raw results.
