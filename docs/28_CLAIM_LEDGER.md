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
| 7 | Learned recurrent readers (GRU, fixed-latent ≥4 slots) learn history-dependent decisions on events and resume; swap/deletion attribution supports history use | EXP | G1, 5 seeds, `docs/29`; v2 test splits already viewed → non-confirmatory |
| 8 | Track failure of learners at the base data size is data insufficiency (GRU reaches ~0.64–0.93 with extra train-only data) | EXP | 3 seeds; high seed variance |
| 9 | Causal Transformer (80k params) failed at lr 3e-3 (optimisation) and generalises worse than GRU at lr 1e-3 on this data size | NEG | 3–5 seeds, small data |
| 10 | RoboMME StopCube train sample imports bit-exactly into the Episode contract | EEE | 100/100 episodes; one task; derived timestamps |
| 11 | Exact toy: when current action is history-independent, action-MI/reconstruction/salience select at chance while future-value selection is optimal; aligned regimes tie | EXP (theory toy) | `g2_toy.json`; says nothing about learned systems |
| 12 | Decision-Aware Compression beats existing methods | UH | untested; novelty relative to D&R/Mem4Act/EventVLA etc. unverified |
| 13 | Autonomous memory management improves a real VLA | UH | no real VLA, no real-policy rollout |
| 14 | Unified Context–Action learning | UH | no joint model exists |
| 15 | Benchmark has memory-selection construct validity (B2b) | UH / open | F7 failures; needs learners+fair controls |
| 16 | pi0/pi0.5 weights may be used / released | UH (NO-GO) | Gemma-term ambiguity (`docs/20`) |
| 17 | Tracking accuracy of the GRU is data-limited: 0.11→0.33→0.78→0.96 for 0/1k/3k/9k extra track pairs (5 seeds) | EXP | `docs/35`; synthetic; test splits viewed |
| 18 | Track-only auxiliary data causes multi-task degradation (events 0.55, resume 0.30 at N=9000; some seeds 0.00) | NEG | `docs/35`; cause (task imbalance) is an untested hypothesis |
| 19 | LeRobot importer aligns timestamps/video exactly and keeps the commanded action invisible at its own tick | EEE | `docs/34`; one dataset; no visual-memory claim |
