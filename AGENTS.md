# AGENTS.md — binding repository instructions

## Authority hierarchy

1. Safety and study integrity (`docs/12_SAFETY_AND_LIMITS.md`).
2. Frozen public protocols (`docs/06_BENCHMARKS.md`, `docs/07_EXPERIMENTS.md`).
3. Contracts and reproducibility (`docs/04_ARCHITECTURE.md`, `docs/10_CLUSTER.md`).
4. Implementation plan (`docs/09_IMPLEMENTATION.md`).
5. User/local operator's explicit instructions.

## Required workflow

- Read the linked documents before editing. Begin by writing a `TASK_PLAN.md` with acceptance tests, files affected, alternatives and reproducibility impact.
- Make a small independent change; run unit tests, type/lint gates and CPU determinism checks. Do not silently change or delete old tests.
- No synthetic benchmark numbers presented as real results. Store dry-run outputs separately with `mode: synthetic`.
- Dataset splits, evaluation seeds, benchmark algorithms, hardware configs and raw results are immutable once registered. New versions get new IDs.
- Verify all train-only statistics fit on the training split. Do not train on test episodes or leak future observations.
- Reference model and benchmark code pinned to exact commits and vetted licenses; write `third_party_manifest.json`.
- Preserve the separation between predicted future actions, issued commands and measured executed actions.
- Keep clock-domain, timestamps, episode_id, trajectory_id, and source hashes throughout.
- Memory changes are transactional: observation cutoff, monotone version, evidence references, patch reason and validated schema. Edit interpreted state; never rewrite source observations.
- Reject any memory update that arrives out of order or references future frames.
- Patch memory must invalidate action continuations affected by changed goals, critical entities, safety constraints, or task phase.
- Robot actuation is forbidden from tests, web services, model agents or notebooks; only an approved external controller may actuate after human authorization.
- For SLURM: use `--dry-run` first. Do not invent hardware/GPU settings. Do not submit or allocate cluster jobs without explicit operator approval.
- Every issue has DoD, review evidence, metrics and an owner; document dead ends and negative results.

## Scope of delivered scaffold

Contracts and toy baselines are executable. VLA adapters, model training, benchmark wrappers, actual GPU allocation and real-robot drivers are **not implemented**. No code should describe a nonexistent integration as functional.

## Do not overclaim

Theory may support a conditional decision-regret bound but cannot guarantee practical memory sufficiency without bounds on learned Q error, policy distribution shift, partial observability and evidence loss. A learned proxy is not an oracle. Treat novelty as an empirical question relative to D&R, Mem4Act, EventVLA, EvoMem and MOST.
