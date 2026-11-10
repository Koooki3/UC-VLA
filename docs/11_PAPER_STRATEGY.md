# 11 | Publication strategy and claims audit

## Suggested English title (working)

**UC-VLA: Learning Decision-Sufficient and Recoverable Context for Long-Horizon Robot Control**

Alternative if focus narrows: **When Should a Robot Forget? Future-Decision-Driven Compression for VLA Policies**.

## Single-sentence abstract thesis

We study whether future-value-aware context compression, verified state evolution and uncertainty-triggered evidence recovery can improve closed-loop history-dependent robotic manipulation under explicit memory and latency constraints.

## Core paper sections

1. Introduction: precise counterexamples where current observation is insufficient and long-window solutions are costly.
2. Related Work: action-relevant memory (D&R), latent memory (Mem4Act/NativeMEM), evidence forecasting (EventVLA), evidence-grounded concepts (ECoMEM), belief models (RB-VLA), state evolution (EvoMem/MOST), CLM, POMDP and rate–distortion.
3. Formulation: unified verified belief, memory operations and bounded decision-rate distortion; conditional lemma, strict caveats.
4. Method: state/evidence model, future utility estimation, controller training, policy conditioning, plan-version contract.
5. Main experiments: SODH causal benchmark, RoboMME/RoboMME-Plus, operational efficiency, independent benchmark.
6. Ablations: loss/utility/recovery/latency/rollout length / data leak defenses.
7. Limitations: Q estimation, unverifiable history, budget scaling, real-robot safety, task distribution.
8. Reproducibility: code/config/checkpoints/test policies/data manifests with permission restrictions.

## Figures that must exist

- Fig 1: paired same observation but different correct actions.
- Fig 2: unified update/act/verify loop with trusted/untrusted state distinction.
- Fig 3: decision-regret / success vs memory and actual latency Pareto.
- Fig 4: deletion/intervention heatmap plus utility calibration.
- Fig 5: OOD horizon/event-count performance with uncertainty.
- Fig 6: real robot sequence and recovery failures (only if obtained).

## Strong claims require strong evidence

| Claim | Mandatory evidence |
| --- | --- |
| `future-decision-aware` | Direct ablation vs current-action MI and state progression memory |
| `unified` | Same shared context process supervises memory/decision/verification, not independent learned modules |
| `real-time` | Target hardware, actual control rate and p95/p99 end-to-end latency incl. memory |
| `recoverable` | Evidence corruption, retrieval missing, false recovery and safety abstain cases |
| `generalizable` | New history lengths, object counts, layouts, task families and independent baseline |
| `theoretical` | Formal assumptions, proof, estimator error limitations, no unsupported policy guarantee |

## Submission target, conditional

CoRL for robotics-focused causality and real-world validation; ICLR/NeurIPS for stronger novel learning objective/theory and cross-benchmark evidence; T-RO for mature system+hardware+method analysis. Choose based on real results and current venue rules. No claim that any venue will accept.

## Publication integrity

Check conference anonymization/data handling policies near submission; never publicize private benchmarks or pre-release test tasks. Plagiarism/code attribution audited via source manifest. Report negative experiments and exact basis for selection of results.
