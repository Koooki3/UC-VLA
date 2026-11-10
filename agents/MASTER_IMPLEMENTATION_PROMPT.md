# Master prompt for Codex / Claude Code (paste whole document as task brief)

You are implementing **UC-VLA**, an original research project scaffold, not copying a paper implementation. Read `AGENTS.md` and all `docs/00`–`docs/15` first. Your task is to transform a research specification into a reproducible method by executing **M1 → M7** sequentially. Start from the present repository state; do not rewrite working code without demonstrating necessity.

## Science-first goals

1. Formalize and empirically test the claim: future-value-guided context management under explicit budget enables better closed-loop history-dependent robotic control.
2. Rigorously distinguish from D&R (action-MI), Mem4Act (latent memory-first), EventVLA (future evidence), EvoMem/MOST (state evolution), CLM (self-editing context).
3. Treat negative evidence as a legitimate outcome. Do NOT optimize or falsify metrics to manufacture SOTA.

## Implementation phases

### Phase M1: deterministic episode data substrate

Implement versioned multi-camera/time-aligned episode loaders, LEVI event read-only bridge, toy SODH generator with pair-level train/val/test separation and anti-shortcut probes. Add property tests for timestamps and identity corruption. Ensure no future labels enter online memory.

### Phase M2: matched baselines

Build `PolicyAdapter` with at least one pinned OpenPI pretrained flow VLA and a clean CPU fake. Implement Memoryless/Uniform/Long-window/Fixed-Latent baselines. Replicate or integrate licensed official D&R and Mem4Act if accessible, otherwise label an approximation as such. Check original checkpoint and action normalization. Freeze evaluation scripts/seeds. Run only CPU or local authorized GPU tests.

### Phase M3: paired simulator rollout and utility estimation

Enable exact state reset + controlled random seed + counterfactual memory edits. Compute paired return deltas, uncertainty, Critic calibration, and item interactions; never assume a noisy Q estimate is ground truth. Add mandatory item-keep/item-remove causality tests.

### Phase M4: learned memory policy

Implement memory writer/reader and operation controller (WRITE/UPDATE/ABSTRACT/EVICT/RETRIEVE/PIN/VERIFY) with transactional commit. Train future-value objective and calibration; ablate action-MI vs future utility under *identical* budgets and backbones.

### Phase M5: rolling action and p95 validation

Integrate existing chunking via a `PlanRef` using context_version/goal_revision/safety_revision and measured execution prefixes; invalidate incompatible cached suffix. Separate fast inference from asynchronous heavy context consolidation. Report p95 end-to-end latency on the actual target GPU.

### Phase M6: complete experiments

Pre-register all H1–H4, pin baselines/commits, evaluate RoboMME, RoboMME-Plus and causal SODH with multiple seeds, rigorous CIs and OOD. Implement reproducible reports and scripts for tables/plots (not screenshot-only metrics). Use test data only once after freeze, or strictly log any deviations.

### Phase M7: real robot and publication

**DO NOT perform or command real robot experiments yourself.** Prepare an operator-reviewed shadow-mode bridge and risk checklist. Real actuation requires a separately authorized local human and established external control authority. Then write paper and artifact documentation grounded in results.

## Non-negotiable behavior

- Start with M1 only and make an explicit task plan; no speculative code bloat.
- Separate `implemented`/`mock`/`planned`. No claims that demos are real VLA evaluation.
- Keep original LEVI/AERI untouched. Use an adapter and read-only events.
- Do not clone/reuse paper code that has incompatible license without approval.
- Never start background jobs, train on expensive GPUs, spend money, or request robot motion unless authorized by operator. Prepare exact command/config and seek approval.
- Every PR: contract tests, negative controls, exact config, commit hash, metadata and limitations.
- Stop at any gate failure; propose simpler alternatives and report disconfirming evidence.

## First concrete deliverable (M1 PR)

An `EpisodeRecord` schema; deterministic `SODH` generator; pair-level split checker; event time alignment; data serialization and reproducible tests; a manifest and no-leakage test. Do not train large models yet. Update `docs/09_IMPLEMENTATION.md` with exact progress and unresolved issues.
