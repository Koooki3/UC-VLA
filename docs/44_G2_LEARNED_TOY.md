# 44 | G2 first learned experiment: which objective keeps later-decision evidence in a small memory? (exploratory, CPU, 2026-11-13)

Script `scripts/g2_learned.py`; results `results/synthetic/g2_learned.json` (MLP probe, final) and `g2_learned_linear_probe.json` (first run, kept). Setup: M=8 iid bits shown as (id, bit) events in random order, then 4 blank ticks; current decision = parity of bits in N, later decision = parity of bits in F={0,1}; memory = GRU with hidden size d ∈ {2,4,8}; stage 1 objectives: `now` (current decision only), `recon` (predict all 8 bits), `lookahead` (current + later decision labels = a whole-episode objective); stage 2: encoder frozen, probe for the later decision trained on **200** labelled sequences; held-out 2,000; 5 seeds. Regimes: A future-only (N=∅), B aligned (N=F), C current decision uses another bit (N={2}).

**Deviation (logged):** the first run used a linear probe; XOR is not linearly decodable, so "reconstruction fails" was confounded. The probe was changed to a small MLP *after seeing* the first results; both runs are kept. No other change.

| regime / d | `now` | `recon` | `lookahead` (later-decision probe acc, mean [min,max]) |
| --- | --- | --- | --- |
| A future-only, d=2 | 0.50 | 0.50 | 0.70 [0.49, 1.00] |
| A, d=4 | 0.51 | 0.52 | 0.90 [0.49, 1.00] |
| A, d=8 | 0.50 | **1.00** | 0.90 [0.48, 1.00] |
| B aligned, d=2/4/8 | 1.00 | 0.50 / 0.52 / **1.00** | 1.00 |
| C now-uses-other-bit, d=2/4/8 | 0.50 / 0.49 / 0.50 | 0.50 / 0.52 / **1.00** | 0.63 / 0.69 / 0.82 [0.50, 1.00] |
(`now` always solves the current decision, 1.00.)

**What it shows (within this toy):** (1) a **current-action-only objective keeps nothing for a later decision** that the current action does not need (0.50 in A and C), exactly the one-step blindness of docs/27 — and in aligned regime B it suffices; (2) **reconstruction is sufficient only when capacity covers the whole history** (d=8: 1.00; d=2,4: chance) — consistent with equivalence condition (b); (3) the **lookahead objective retains the later-decision evidence at small capacity** (d=4: mean 0.90) but with **very high seed variance (0.49–1.00)** — an optimisation/identifiability problem for parity, not a robust result.
**What it does not show:** that an *intervention-based future-utility* objective beats the lookahead objective — `lookahead` here is the strong whole-episode baseline the review (docs/38 B1) demanded, and it is the only objective that wins in the low-capacity regimes; no learned counterfactual-utility estimator exists yet. "Capacity d" is continuous hidden width, not a bit budget. One task family, no real data, 5 seeds, exploratory. **G2 status: unchanged — unverified hypothesis plus a theory toy and a learned toy showing one-step blindness and capacity effects.**
