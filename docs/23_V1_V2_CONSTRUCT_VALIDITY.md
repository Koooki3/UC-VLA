# 23 | SODH v1 vs v2: construct validity, v2 results, B2b status (2026-11-11)

All results are `mode: synthetic`. Oracle and heuristic rows are **measuring instruments or decoder self-checks, not methods**. Data: `results/synthetic/sodh_v2` (frozen, split `sodh-split-539d19e50cea`), run `results/synthetic/v2_eval/run-c313087eb3721c05` (code commit recorded in `run.json`, clean tree, ledger entry present). Spec: `docs/19`; v1 problems: `docs/18`.

## 1. Construct-validity differences (each row has a test)
| Question | SODH v1 (frozen fixture) | SODH v2 | Test / evidence |
| --- | --- | --- | --- |
| Does the cups task need tracking? | No: static cups gave identical outcomes in 792/792 cells (review M1) | Yes: removing swaps changes the oracle's answer in 441/720 = 61 % of track episodes (the rest: swaps not involving the ball) | `test_f6_static_cups_control_changes_track_outcomes` |
| Is success defined by the environment? | Label = generator rule shared with the decoder | Label = independent posterior oracle; success from the TRUE hidden state (`success_actions`); the gate failed a generator bug (wrong mirrored success item) before freeze | `oracle_env_findings`, `test_generator_bug_regression_*` |
| Event counting | one pressed tick = one press (ambiguous under held buttons) | validity-aware: holds count once, jams (lamp off) invalid, dropout ticks; contact-edge counting scores **0.00** pair-optimal | `test_events_edge_counting_falsifier_*` |
| Same obs / different history | per-channel checks missed proprio/command/event leaks | all channels of the decision view identical (digests), oracle actions differ, oracle margin ≥ 0.30 | `test_pairs_share_whole_decision_view_*` |
| Evidence position | one draw, confounded with horizon | generated axis (early/mid/late, balanced ±1) crossed with horizon; tag verified against the decisive tick | `test_evidence_position_axis_*`, `position_findings` |
| OOD | longer/larger only; `heldout_mapping` = renamed symbols from seen atoms | `ood_length` (longer) separated from `ood_structure` (new mechanisms never in training: swap bursts + flicker, long holds + dense taps, removals + distractor attempts); resume vocab with disjoint atoms | `test_regimes_are_structural_*`, `mechanism_findings` |
| Train/test overlap | alpha-twins 88–97 % (interleave) | alpha-twin rate 0 for train vs val/test/OOD (gate); observation-level template unique | `alpha_twin` gate, `template_findings` |
| Size | 9 pairs/family in test_id | 126 pairs/family test_id; 60 per OOD split | `size_findings` |
| History-free policies | capped by construction | trained obs-only lookup, no-history oracle, no-history NB: pair-optimal **0.00** on all 16 relevant cells | `test_f1_*`, g0 report |
| Leak detection | g0/2 (single token) | g0v2/2 (+aggregate, conjunctive); power in `docs/21` | `test_g0_mutants`, `test_v2` |
**Honest limits of v2:** noise/occlusion are symbolic, not vision; `events`/`track` structure shifts barely change a naive decoder (below), so "structural OOD bites" is demonstrated for `resume` only; the oracle is Bayes-exact for iid mechanisms and approximate for swap bursts; control family is by design obs-solvable (alpha-twin and structure overlap reported but excluded from gates).

## 2. Four memory metrics, reported separately (test_id unless noted)
**Necessity** (oracle full-history env success − best history-free agent, measured on the pair distribution, which is selected to need history, so the lower bound is partly by construction): track 0.56, events 0.50, resume 0.72 (OOD-length 0.48/0.50/0.69; OOD-structure 0.59/0.50/0.72; counterfactual resume 0.71). Oracle env success with full history: track 0.83 (noise+occlusion cap it), events 1.00, resume 1.00.

**Use** (swap-follow, critical vs matched non-critical deletion; test_id): oracle — all three families use history (swap 1.00; drop-critical 0.50/0.69/0.79 vs drop-noncritical ≤0.13). Heuristic `naive` — uses history. **Independent learner `BagOfTokensNB` — does NOT use history in any family (uses_history false; swap-follow 0.68/0.38/0.26), i.e. pre-registered F8 is triggered: no available independent learner reproduces history use.** This reflects the bag-of-tokens model class (no order, no state), not evidence about UC-VLA; a sequence-capable CPU learner is the missing piece.

**Sufficiency** S(k) = oracle env success restricted to the retained context / full (tick budget k = 1,2,4,8,16):
| family | selector | S(1) | S(2) | S(4) | S(8) | S(16) | k* (S ≥ .95) | norm. AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| track | last-k | .31 | .39 | .51 | .80 | 1.00 | 16 | .71 |
| track | uniform-k | .28 | .29 | .30 | .55 | 1.00 | 16 | .57 |
| events | last-k | .51 | .53 | .54 | .67 | .93 | none ≤16 | .68 |
| events | uniform-k | .50 | .51 | .56 | .62 | .95 | none ≤16 | .67 |
| resume | last-k | .31 | .33 | .37 | .56 | .89 | none ≤16 | .56 |
| resume | uniform-k | .31 | .31 | .36 | .46 | .86 | none ≤16 | .51 |
**Compression efficiency** is the k* / AUC column (horizons are 8–22 ticks, so k=16 is nearly "everything" for in-distribution episodes; compression claims beyond this need longer horizons).

## 3. F7 — does the selector ranking survive a change of measuring instrument? (closure test for B2b)
Registered rule (docs/19): sign of (uniform − last) in PairOptimal must agree across the oracle-restricted instrument, the naive decoder, and the independent learner. **Result: 17 of 50 cells disagree** (events 10/15, resume 6/20, track 1/15). A post-hoc, explicitly labelled significance-aware variant (only opposite *significant* signs count) finds 0 disagreements — consistent with many cells being underpowered/noisy rather than genuinely reversed; the registered rule is the one that counts.
Primary family (test_id, oracle PairOptimal, k ∈ {2,4,8} × 3 families, Holm over 9): **track k=2,4,8 significant, last-k > uniform-k** (Holm p = .006, <.001, <.001; discordant pairs 4:21, 3:37, 7:49), instruments agree; events and resume: not significant after Holm (p ≥ .36). Mechanism of the track effect: tracking needs a *contiguous* suffix of swaps after the reveal, which a last-k window supplies and uniform sampling does not — a property of the task family, shown identically by the oracle, so it is not a decoder artefact. Pilot (val) minimal detectable discordant imbalance at test n: 0.03–0.17 depending on cell (`pilot_val_mde`).
**Conclusion:** the v1 pathology (a ranking reversible by moving the reveal tick or the decoder prior) is **not** present for `track` in v2, but remains for `events`/`resume` cells. No selector-ranking claim is licensed for those cells.

## 4. B2b status: OPEN
Closure needs F7 sign-agreement **and** at least one independent learner with sequence capability showing history use (F8). v2 supplies the first; the second is missing. Remaining work: a CPU sequence learner (e.g. small GRU or hand-featurised state-tracker trained on train only); longer horizons for compression claims; learner-based leak probes (docs/21 §2).

## 5. Finding-by-finding falsifiable-test status (operator requirement)
| Finding | Executable test that fails on the defective design | Status |
| --- | --- | --- |
| B1 | `tests/test_g0_mutants.py` (8 channels) + `test_v2` mutants + power grid (`docs/21`) | closed for the attack set; validity bounded by power report |
| B2a | `test_free_clock_information_changes_the_picture`, strict/clock sweep, coverage reports | closed |
| B2b | F7 + F8 (above) | **open** |
| M1 tracking | `test_f6_static_cups_*` | closed in v2 (v1 is a defective fixture, kept) |
| M2 event semantics | `test_events_edge_counting_falsifier_*` | closed in v2 |
| M3 OOD | `test_regimes_are_structural_*`, `mechanism_findings`; naive decoder drops 0.86→0.47 on resume ood_structure | closed by design; bite demonstrated for 1 of 3 families only → **Minor open** |
| M4 twins | `alpha_twin` gate = 0 | closed in v2 |
| M5 priors | `label_prior`, `constant_answer_floor` | closed (power-limited) |
| M6 independent oracle | `oracle_recompute_findings`, posterior sanity, frame-derived oracle (v1) | closed for v2 labels; the oracle shares the environment model by definition |
Review limitation carried forward: reviews are **independent-context model reviews, not external human review**.

## 6. Deviation log (docs/19 §10)
- 2026-11-12 pre-freeze: mirrored `resume` mate had the wrong env-success item (found by the oracle-vs-env success check at oracle env success 0.5); fixed; data regenerated **before any evaluation or commit of the frozen manifests**. Template key changed from hidden-script to observation-level after a duplicate-content finding (same reason, same time). No test split was evaluated before these corrections.
- g0v2/1 → g0v2/2 (extended probes) after the power study; frozen data unchanged and re-audited; `FROZEN.json` still records the original g0v2/1 pass.
