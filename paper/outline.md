# UC-VLA paper skeleton (do not fill result cells with guesses)

## Title
UC-VLA: Learning Decision-Sufficient and Recoverable Context for Long-Horizon Robot Control

## Abstract
Problem [history dependence and finite context]. Method [unified verified state and decision-aware compression]. Theory [conditional regret lemma, bounded budget]. Experiments [to be conducted; actual benchmark results only]. Efficiency [measured p95/FLOPs/GPU-hours only]. Limitations [Q model bias, OOD safety].

## 1 Introduction
- Same current observation, different history, different action counterexample.
- Why longer context/more latent tokens does not solve memory selection economically.
- Novelty against D&R/EventVLA/Mem4Act/MOST; avoid broad unsupported claims.

## 2 Related Work
- Native memory, event/change memory, planner and CLM/context programming, constrained POMDP, belief compression.

## 3 Method
- Context state, evidence/epistemic separation, future value distortion, memory manager, action control contract.

## 4 Experiments
- Dataset/benchmarks, matched baselines and budget fairness.
- Table 1 main performance; Table 2 Pareto; Table 3 detailed ablation.
- Figures as specified in docs/11_PAPER_STRATEGY.md.

## 5 Limitations and reproducibility
- Explicit failure cases, cluster and privacy caveats.

## Appendix
- Definitions/proof, implementation specifics, benchmark protocol and statistical tests.

**No results have been generated.**
