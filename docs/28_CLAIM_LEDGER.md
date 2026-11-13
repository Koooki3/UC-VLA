# 28 | Claim Ledger (updated 2026-11-13)

Classes: **VSE** Validated Scientific Evidence · **EEE** Established Engineering Evidence · **EXP** Exploratory Observation · **NEG** Negative Result · **UH** Unverified Hypothesis. Nothing is VSE yet: no confirmatory (sealed, pre-registered, untouched) evaluation exists. Reviews are independent-context model reviews, not external human review.

| # | Claim | Class | Basis / limits |
| --- | --- | --- | --- |
| 1 | 152+ tests pass; results reproduce across fresh processes | EEE | `make smoke`; synthetic only |
| 2 | SODH v2 passes G0 v2/2 | EEE | limited to the attack set and measured power (`docs/21`): leaks giving ≤0.67 reader accuracy undetected |
| 3 | Planted XOR/parity leaks escaped g0v2/1; extended probes detect ≥~0.70 | EEE | `g0_power_study.json`, 0/12 null FP |
| 4 | Selector ranking (uniform vs last) flips across instruments in 17/50 cells (events, resume) under the registered rule | NEG | pre-registered F7; track consistent |
| 5 | Last-k beats uniform-k for tracking (oracle, decoder, learner agree; Holm-significant) | EXP | synthetic; v2 test splits viewed repeatedly |
| 6 | Bag-of-Tokens NB cannot use history | NEG | learner-specific capability, not a statement about memory learning |
| 7 | Learned recurrent readers (GRU, fixed-latent ≥4 slots) learn history-dependent decisions on events and resume; swap/deletion attribution supports history use. **The full registered G1 criteria are NOT met** (order-perturbation condition fails; track not learned at base size) | EXP | G1, 5 seeds, `docs/29` (corrected); v2 test splits already viewed → non-confirmatory |
| 8 | Track failure of learners at the base data size is data insufficiency (GRU reaches ~0.64–0.93 with extra train-only data) | EXP | 3 seeds; high seed variance |
| 9 | Causal Transformer (80k params) failed at lr 3e-3 (optimisation) and generalises worse than GRU at lr 1e-3 on this data size | NEG | 3–5 seeds, small data |
| 10 | RoboMME StopCube train sample imports bit-exactly into the Episode contract | EEE | 100/100 episodes; one task; derived timestamps |
| 11 | Exact toy: a ONE-STEP objective cannot select later-only evidence (reconstruction/random at chance; salience helps only if correlated with future relevance); a lookahead objective coincides with future-value selection; future-value selection can be worse under reachability gating or a noisy critic | EXP (theory toy, reworded after review) | `g2_toy.json`, `docs/38`; does NOT show an advantage over lookahead/whole-episode baselines; says nothing about learned systems |
| 12 | Decision-Aware Compression beats existing methods | UH | untested; novelty relative to D&R/Mem4Act/EventVLA etc. unverified |
| 13 | Autonomous memory management improves a real VLA | UH | no real VLA, no real-policy rollout |
| 14 | Unified Context–Action learning | UH | no joint model exists |
| 15 | Benchmark has memory-selection construct validity (B2b) | UH / open | F7 failures; needs learners+fair controls |
| 16 | pi0/pi0.5 weights may be used / released | UH (NO-GO) | Gemma-term ambiguity (`docs/20`) |
| 17 | Tracking accuracy of the GRU is data-limited: 0.11→0.33→0.78→0.96 for 0/1k/3k/9k extra track pairs (5 seeds) | EXP | `docs/35`; synthetic; test splits viewed |
| 18 | Track-only auxiliary data causes multi-task degradation (events 0.55, resume 0.30 at N=9000; some seeds 0.00) | NEG | `docs/35`; cause (task imbalance) is an untested hypothesis |
| 19 | LeRobot importer aligns timestamps/video exactly and keeps the commanded action invisible at its own tick | EEE | `docs/34`; one dataset; no visual-memory claim |
| 20 | The multi-task collapse of docs/35 was task imbalance: balanced auxiliary data (1k/3k per family) gives events≈1.00, resume≈0.99–1.00 and track 0.38/0.73 (5 seeds) | EXP | `docs/39`; two pre-registered predictions not met literally (P-M1 two-sided wording, P-M2 0.73<0.75) |
| 21 | M3 rollouts need ≥155–1,866 paired episodes depending on effect size/discordance; GPU-hours not estimated | EEE (planning arithmetic) | `docs/37`, McNemar approximation |
| 22 | Under evidence complementarity, per-item marginal scoring misselects (mean regret 0.44 vs exact joint; P(optimal) 0.32); group selection is exact when groups are known; Shapley needs ~2000 permutations for 0.81 | EXP (theory toy) | `g2_selection.json`; the earlier additive cost estimate (16 rollouts) is INVALID |
| 23 | Confirmatory (sealed, same generator, new seeds): GRU and fixed-latent S4 meet the full pre-registered G1 criteria on events+resume (PairOptimal 0.98–1.00, memoryless 0.00, swap/deletion/order conditions met); track NOT reliable (GRU seeds 0.00–0.77, FL4 ≤0.45); events length-OOD 0.52 | EXP (replication within scope) | `docs/43`; not an independent environment; no compression/VLA content |
| 24 | Learned toy: a current-action-only objective keeps nothing for a later decision it does not need; reconstruction suffices only when capacity covers the history; a whole-episode (lookahead) objective keeps later evidence at small capacity but with seed variance 0.49–1.00 | EXP (toy) | `docs/44`; probe changed after first results (deviation logged); no intervention-based utility, no advantage over lookahead shown |
| 25 | Events length-OOD collapse is driven by the target count K (perfect to K=3, chance at K≥7–8), not by sequence length per se (padding control costs ≤0.04) | EXP | `docs/45`; 3 seeds, non-sealed data; K and horizon partly confounded; no repair tested |
| 26 | Widening events training K (to 10–11) lifts K≥6 length-OOD accuracy from 0.50–0.73 to 0.96–0.98 in 2/3 seeds; the third seed fails to train at all (chance, also in-dist) | EXP | `docs/47`; P-K1 not met as worded (B≈C), P-K2/P-K3 not evaluated (accuracy only; events only); non-sealed, 3 seeds |
| 27 | The docs/47 seed-0 failure is lr-fragility (3e-3: 1/5 fail; 1e-3 and 3e-4: 0/5), but K≥6 extrapolation stays seed-variable even when training succeeds (5/14 successful runs <0.90) | EXP | `docs/49`; n=5 per cell, non-sealed; P-O3 not met; P-K3/PairOptimal still unmeasured |
| 28 | Widening the events horizon of the balanced mix (10–26) raises events length-OOD PairOptimal 0.70 → 0.85 (5 seeds, range 0.65–0.98) with events/resume in-dist ≈1.00, but one of five seeds collapses on track (0.00); cause undetermined | EXP | `docs/51`; P-G2 not met; non-sealed, no matched control |
| 29 | The 1/5 track collapse in docs/51 is lr-3e-3 optimisation fragility, not an effect of widening events data (collapses 1/10 with original data, 1/10 widened; 0/5 at lr 1e-3); widened events data raises events length-OOD PairOptimal mean 0.70 → 0.90 over 10 seeds each | EXP | `docs/53`; n=10 per cell, non-sealed; P-T1 not met, alternative reading accepted; no rate difference detectable |
