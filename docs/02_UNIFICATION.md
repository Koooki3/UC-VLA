# 02 | 从同一性原则到统一可验证决策状态

## Why identity, rather than adding modules?

机器人操作的本体是一个随时间演化的物理状态。图像帧、深度、夹爪反馈、语言子任务、Action Chunk 和观测事件是这个状态的不完整投影。如果每个模块独立维护“杯子 2 里有球”“抓取已完成”等判断，系统会出现：对象身份漂移、动作意图被误写成已执行事实、模型记忆覆盖真实事件、滚动动作使用过期任务计划。

**Operational identity:** every verified fact has a unique identity (episode + actor/object + temporal interval + source evidence); every interpretation declares its provenance, confidence status and version; every action is conditioned on a *specific committed context version*. This is an engineering interpretation of unity/identity, not a philosophical proof.

## Single source of physical truth; two ledgers

- **Immutable observation/execution ledger** E_t: signed / hashed source references, timestamps, measured states, operator verified facts. Never overwritten or compacted destructively; storage retention policy may archive but record logical identity.
- **Mutable epistemic state** M_t: current best estimates, subtask progress, object associations, uncertainty, future hypotheses, and memory-budget policy. Every edit yields patch and a version; it can be revised.

Do not treat a predicted action A_pred as an executed action A_exec. Time alignment and causal cutoffs are mandatory.

## Unified transitions

Let z_t be compressed belief, e_t new evidence, u_t memory management action, a_t selected control action, y_{t+1} execution feedback. Use:

1. `propose`: \tilde z_{t+1} = F_phi(z_t, e_{t+1}, g, a_t^executed, u_t)`.
2. `validate`: check causal window, evidence refs, invariant consistency, policy safety, version monotonicity.
3. `commit`: `z_{t+1}, version_{t+1}` or reject/rollback.
4. `act`: `pi_theta(a | o_t, g, z_t, version_t)`.
5. `verify`: reconcile predicted state and measured state, potentially invalidating queued chunk.

Memory may be semantically unified while remaining physically separated into fast belief, bounded context projection and episodic external store. These are **representations of the same state** at distinct fidelity and time scales, not three independent agents inventing facts.

## Five hard invariants (enforced by code)

I1 — **Causality:** no state update may read observation with timestamp after the decision cutoff.

I2 — **Provenance:** committed fact references an existing observable or is explicitly tagged `hypothesis` / `inference`, never promoted to measured without evidence.

I3 — **Identity:** object/event IDs and frames remain stable across compaction; when uncertain, represent a distribution rather than assigning an arbitrary stable ID.

I4 — **Version / plan coherence:** every action plan declares context_version; material change to target, constraint, safety or task stage invalidates stale plans, including rolling suffix.

I5 — **Recoverability:** deleting fast-context content must not silently destroy preserved evidence that will be referenced in audit; retention/archival is separately controlled.

## Unified scientific claim

We evaluate the **closed-loop quality of a consistent, compressed belief process**, not merely image retrieval accuracy. One hypothesis → one architecture → one loss family → one matched-budget Pareto evaluation. The hypotheses may fail independently, and then the claim must be narrowed.
