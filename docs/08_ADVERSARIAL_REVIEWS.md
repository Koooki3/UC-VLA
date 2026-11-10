# 08 | Five-round adversarial self-review (not independent external peer review)

> These rounds are a structured **internal design red-team** conducted on 2026-11-11. They do not pretend that five independent experts, GPUs or paper reviewers evaluated the project. Every challenge has an explicit design correction and falsification gate.

## Round 1 — Novelty collision: Is this just D&R + EventVLA + MOST + CLM?

**Attack:** D&R already maximizes action-relevant information; EventVLA predicts future evidence value; MOST supervises world-state change and task progress; **ECoMEM already separates an evidence-grounded Writer and learned Reader**; RB-VLA already maintains a compact action-conditioned belief; CLM supports learned context control. Hierarchical memory + RL is recombination.

**Criticism:** Original “Decision-Sufficient Memory” alone is too broad to distinguish. No novelty claim based on static memory, evidence-grounded writer/reader, recurrent belief, keyframe selection, state-progress supervision, token pruning, or generic RL.

**Revision:** Make **controlled future closed-loop value distortion** the primary learning target, with paired-replay evidence, not one-step action MI. Treat state memory and retrieval as validated tools, not contributions by themselves. Match D&R/Mem4Act exactly where possible. 

**Kill test:** If value-regret training does not beat strong action-MI with matched inference cost and training budget on OOD, abandon this as main innovation; pivot to a robust intervention benchmark if findings merit publication.

## Round 2 — Theoretical: Is Q-based compression just an intractable oracle?

**Attack:** True Q* unknowable, Critic may be OOD, stochastic futures make memory-utility target noisy. `2epsilon` bound is trivial if epsilon cannot be measured.

**Revision:** Separate conditional lemma from practical learned proxy. In small controlled simulators evaluate exhaustive or high-sample actions and paired common-random-number rollouts; train cross-fitted distributional/ensemble Critic, report held-out value error and calibration. Train an intervention *ranking* objective when absolute values are unreliable. No online privileged state.

**Kill test:** If future utility predictions are not better calibrated than simple salience heuristics, and oracle/learner performance gap remains large, weaken theory claim to diagnostic insight; do not publish a fake guarantee.

## Round 3 — Empirical: Are the benchmarks genuinely history-dependent?

**Attack:** LIBERO-LONG or RoboMME may allow visual shortcuts and memorized sequence priors, with inflated gains from training data overlap, non-matched baselines or small N. This is consistent with 2026 manipulation benchmark audit literature.

**Revision:** Separate paired-history causal suite as a first-class test; counterbalance object positions and task scripts; same current image but divergent history and correct answer; history-swap and no-history classifier controls. Pre-register seeds, report clustered CIs and OOD extrapolation. Preserve author's official benchmark setting alongside strict matched comparison.

**Kill test:** If a current-image-only classifier performs close to a memory-aware policy, do not call that task evidence for memory intelligence; regenerate the task or change claims.

## Round 4 — Systems: Memory optimization kills low-latency VLA

**Attack:** Async retrieval, Q estimation, VLM writer, KV cache resets and rolling chunk invalidations can cost more than the saved context. Predicted actions are not executed facts, and stale chunks may become unsafe.

**Revision:** No heavy LLM in control tick; retain existing flow-matching/rolling expert; thin conditioning reader; event-triggered atomic versioning; shadow-mode latency test; p95/p99 and GPU contention tracked end-to-end. Any material context version change invalidates cached suffix. External safety controller has final authority.

**Kill test:** If meeting hardware deadline requires disabling key memory behavior, report Pareto instead of a real-time claim; split slow memory into asynchronous nonblocking path.

## Round 5 — Generality: Is it a fragile benchmark-specific pipeline?

**Attack:** Hand-coded identity bookkeeping and task-state labels may secretly encode the answer, with no transfer to new objects, tasks, robots or unseen history lengths. Recovery may only work when evidence was trivially recorded.

**Revision:** Keep broad generic evidence schema; learn utility using multiple task families; use new object/layout/event distributions and cross-embodiment external evaluation. Disclose oracle-labeled supervised signals and compare against simpler state-engine methods; test evidence missing irrecoverably. Rank performance across history-length × memory-budget × OOD axes.

**Kill test:** If gains vanish on new task families or require privileged labels at inference, publish as narrow method, not unified general system.

## Cross-cutting security / epistemic audit

- Evidence cannot be overwritten; derived belief may be corrected.
- No external agent obtains real robot motion authority.
- Memory poisoning / hallucination / clock drift / stale plan need negative controls.
- Check data governance and licenses; avoid embedding copied CC-BY-NC code in an unrestricted package.
- Negative results are retained, failure rates reported, no cherry-picked videos.

## Synthesis

**Surviving falsifiable claim:** Future-outcome-oriented compression with causal memory interventions plus provenance/version management can move the success–cost–latency frontier on genuinely history-dependent long-horizon tasks. This remains *untested*.

**Removed inflated claims:** “first memory-native VLA”; “proof that compression = intelligence”; “lossless infinite history”; “general safe RL”; “first future utility memory”; “universal fixed-sized belief”; “automatic SOTA”.
