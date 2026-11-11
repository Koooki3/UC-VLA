# 21 | G0 weak-leak detection power (SODH v2 gate), 2026-11-11

Data: `results/synthetic/g0_power_study.json`; script `scripts/g0_power_study.py` (plants leaks in an in-memory copy of the v2 *events* family — 2×(126+60+60+180+54) = 960 episodes, binary label, chance 0.5; frozen files untouched). `leak_reader_accuracy` = accuracy of a reader that exploits the planted leak directly on non-train splits, i.e. **how much shortcut the leak hands out**. Detection = a leak-related finding on the events family. Replicates: 4 per strength, 12 for the null. D0 = g0v2/1 probes (single token per position/channel, metadata lookups, label prior). D1 = D0 + aggregate (per-token count/parity over history) + conjunctive (token pairs) probes = **g0v2/2, now the gate**.

| Leak form | strength | reader acc. | D0 detects | D1 detects |
| --- | --- | --- | --- | --- |
| null (no leak) | 0 | 0.50 | **0/12** | **0/12** |
| single token (linear) | ρ=0.2 | 0.60 | 0 % | 0 % |
|  | ρ=0.4 | 0.70 | 100 % | 100 % |
|  | ρ=0.6 / 1.0 | 0.80 / 1.00 | 100 % | 100 % |
| XOR of two tokens (nonlinear, each token alone uninformative) | ρ=0.2 | 0.60 | 0 % | 0 % |
|  | ρ=0.4 | 0.69 | **0 %** | 100 % |
|  | ρ=0.6 / 1.0 | 0.79 / 1.00 | **0 %** | 100 % |
| parity of a bit stream across all history ticks (cross-time) | ρ=0.2 | 0.60 | 0 % | 0 % |
|  | ρ=0.4 | 0.70 | **0 %** | 100 % |
|  | ρ=0.6 / 1.0 | 0.80 / 1.00 | **0 %** | 100 % |
| rate code (label shifts a token's frequency over time) | δ=0.05 | 0.67 | 0 % | 0 % |
|  | δ=0.10 | 0.80 | 25 % | 100 % |
|  | δ=0.2 / 0.3 | 0.95 / 0.99 | 100 % | 100 % |

## Findings
1. **g0v2/1 would have passed a *perfect* (100 %) XOR or parity leak.** The earlier "G0 passes" claim was blind to any non-single-token leak. Fixed by the extended probes (D1), which detect XOR/parity/rate leaks that give a reader ≥ ~0.70 accuracy.
2. **Residual blind zone (not fixed):** every leak form, including D1, goes undetected when the reader accuracy is ≤ ~0.67 (ρ ≤ 0.2, δ ≤ 0.05). Minimum detectable shortcut ≈ 0.70 accuracy on a binary label at ~1k episodes. Such weak leaks sit in *history* tokens, so a history-using agent (including a future UC-VLA policy) could exploit them without genuinely remembering. Mitigations: pair-level metric (a leak must separate both mates to inflate it), larger splits, and learner-based leak probes (e.g. train a classifier on random-history shuffles) — **planned, not implemented**.
3. **Forms not covered at all:** leaks in more than two jointly-required tokens, in token *order* or inter-token gaps, or conditioned on a non-linear function of count and position; leaks that only matter in OOD splits; leaks in non-front channels beyond those covered by the channel digests. G0 validity statement therefore stays **"validated against the current attack set and the power grid above"**, never "detects all leakage".
4. **False positives:** 0/12 on the null for both detectors; 0 findings on the clean frozen v2 data (g0v2/2).
5. Regression: `tests/test_v2.py::test_extended_probes_catch_what_single_token_probes_miss` (documents D0's blind spot and D1's detection) and the 8-channel mutation suite.
