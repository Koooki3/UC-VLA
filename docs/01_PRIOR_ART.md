# 01 | Prior-art / novelty collision register (2026-11-11)

> 日期界限：本清单为 2026-11-11 的快速高可信文献核对，不是系统综述；论文/代码状态随时可能更新。第一项实现任务是复查论文完整版与开源许可。下方所有差异定位均为**研究假设**，非已验证创新性。

| Work (verified paper) | What it already does | Novelty collision | Required differentiator |
| --- | --- | --- | --- |
| ECoMEM https://arxiv.org/abs/2610.00801 | evidence-grounded concept memory, Writer/Reader, RoboMME | **provenance-grounded writer/reader is prior art** | target causal future-value + uncertainty-recovery + versioned real-time plan, not evidence grounding alone |
| RB-VLA https://arxiv.org/abs/2602.20659 | action-conditioned recurrent belief for VLA | persistent compact belief is prior art | value-guided memory operations and matched-budget causal tests |
| MemoryVLA (ICLR 2026) https://arxiv.org/abs/2508.19236 | perception + cognitive bank / retrieval / diffusion actions | hierarchical memory, selective retrieval | regret-driven policy + verification + formal counterfactual tests |
| NativeMEM https://arxiv.org/abs/2607.06678 | native visual encoder compresses historic frames, real-time | cheap tokens | better causal robustness at matched actual latency |
| EventVLA https://arxiv.org/abs/2606.20092 | future utility predicted keyframe memory | future-aware storage already exists | action-value-guided *edit/retrieve/reobserve* and verified state |
| D&R https://arxiv.org/abs/2610.00982 | recursive action-relevant top-K via conditional MI | action-related memory objective already exists | *closed-loop future Q/return*, not one-step imitation sufficiency |
| Mem4Act https://arxiv.org/abs/2610.22323 | latent fixed memory, question/perception supervision, memory-first training | task-relevant latent memory | resource policy and recoverability, not representation alone |
| EvoMem-VLA https://arxiv.org/abs/2610.05418 | state-evolution/delta tokenization | transition memory | source-grounded state and counterfactual future-regret control |
| MOST https://arxiv.org/abs/2611.00734 | object state + subtask progress memory; predictive future tokens | state progress & future token supervision | provable contracts, adaptive recovery, causal matched budget |
| RoboCasa-Mem https://arxiv.org/abs/2611.06475 | environment and process memory, behavior-conditioned eval | task-progress benchmark | use as independent external evaluation and avoid duplicate tasks |
| CLM https://arxiv.org/abs/2609.37725 | self-edited context, RL and system-side cache optimization | autonomous context management | physical state estimation/action/safety constrained management |
| RoboMME-Plus https://robomme.github.io/plus.html | OOD memory generalization benchmark | OOD suite already exists | evaluate transparently; no invented benchmark claims |
| Benchmark Audit https://arxiv.org/abs/2606.04233 | shortcut & statistical critique of manipulation benchmarks | fragile improvements | enforce paired-history causal diagnostics + statistical power |

### Methodological requirement

A new technique is not new because it combines four familiar blocks. To claim a distinct advance, demonstrate at least one of:

1. New theoretical relationship *and* empirical gain not obtained by existing action-MI supervision.
2. A reliably measured recovery/control frontier unavailable to fixed-memory methods with matched parameters.
3. A clean, reusable intervention-based evaluation methodology uncovering errors in existing benchmarks.

### Minimum comparisons

D&R (exact author implementation when available), Mem4Act (when code released; otherwise clearly label approximate reproduction), NativeMEM, EventVLA, Most/EvoMem where compatible, strong fixed-history and long-window baselines. Cross-backbone comparison is secondary; primary ablations need the **same pretrained backbone**, data and adaptation compute.

### License caveat

CLM's official repository lists CC BY-NC 4.0; do not copy its source into an unrestricted commercial project. Separate paper ideas from code reuse; inspect each license, checkpoint restriction and benchmark data use before downloading.

### References

See `references.bib` and URLs in `README.md`. Rerun the literature refresh before paper submission.
