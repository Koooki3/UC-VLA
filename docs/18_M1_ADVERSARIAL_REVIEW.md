# 18 | M1 adversarial review (2026-11-11): findings, fixes, residual risk

Reviewer: a delegated subagent in a fresh context, instructed to falsify. **This is an independent-context model review, not independent human review** (`docs/14`). Target: tag `m1-synthetic-baseline`. Raw scripts and outputs: `reviews/m1_2026-11-11/`. All numbers below are `mode: synthetic`.

## Verdict

M1 data are clean: labels re-derived by an independent oracle agree on 704/704 episodes; no pair, scene script, exact content or mapping crosses splits; no non-history metadata feature beat the label prior. But **"G0 passes" was over-trusted**, and the M1 budget curves are **not evidence about memory**. M1 stays a *synthetic infrastructure validation*. SODH v1 is **frozen and kept as a unit-test fixture only**; scientific claims need SODH v2 (spec below).

## Findings and status

| ID | Class | Finding | Status |
| --- | --- | --- | --- |
| B1 | Blocker | G0 audited only a subset of the CausalView. Leaks in proprio, commanded actions, online events, episode-id suffix, a history-tick token all **passed** `g0_report` (a no-history reader scored pair-both 1.0 on them). Mutants M4/M7/M8 survived all M1 tests. | **Fixed** (audit g0/2, commit 4b621a7): pair records recomputed from episodes (stale records = error); every channel of the decision view compared; opaque-id grammar; generic single-token label probe over all channels and positions; constant-position prior test; constant-answer floor reported. `tests/test_g0_mutants.py` plants a leak in each channel and requires failure *for the right reason* (7 mutants + option-order prior). |
| B2 | Blocker if used as a result | Budget curves, LongWindow-vs-Uniform ranking and "FullHistory = 1.0" reflect where the generator puts evidence and what the hand-written decoder assumes (reversed by moving the reveal tick or dropping the scale-up rule); free information (`history_len`, `complete`) was not charged; test_id had 9 pairs/family (3 of 45 CIs excluded 0). | **Partly fixed, rest deferred to v2.** Fixed: `history_len` removed from `Context`; `OracleDecoder` has `strict` (no side information) and `clock` (elapsed time) variants run side by side; coverage-conditioned and evidence-position reporting; pair-clustered CIs; oracle rows relabelled *decoder self-check*. M1 `symbolic_baselines.json` must **not** be read as a memory result. Deferred: evidence position as a *generated* axis and larger test_id (v2). |
| M1 | Major | `object_correspondence` is retrieval, not tracking: cups carry names through swaps, so static cups give identical outcomes in 792/792 cells. | Open → v2 (anonymous cups; answer requires following swaps after the reveal). |
| M2 | Major | `process_count` assumes one pressed tick = one press; adjacent pressed ticks are ambiguous under a rising-edge reading (34/56 in-dist pairs indistinguishable then). | Convention now **documented** (below); open → v2 (release tick between presses). |
| M3 | Major | `heldout_mapping` is renamed symbols (atoms all seen in train; a name-blind policy scores 1.0 on process_count); `test_ood` is only longer/bigger (a length-agnostic rule scores 1.0); docs/06 lists distractors/layout/delay that were not shifted. | Claims **renamed in docs** ("longer/larger", "renamed symbols"); open → v2 (disjoint atoms; separate `ood_length` and `ood_structure` regimes). |
| M4 | Major | Near-twins: after alpha-renaming names and dropping arm/noise, 88–97 % of `task_interleave` val/test episodes have a train twin with the same relative label (count: 12–17 %; cups: none). `check_splits` keys include names, so it passes. | **Measured** (`alpha_twin_rate_informational` in the g0/2 report; reproduces the reviewer). Gate deferred to v2. |
| M5 | Major | Position/label priors sit at the audit's cap; interleave "answer = order[1]" scores 0.50 at p=3; `memoryless` floor understates the no-history ceiling; mutants J/M1/M6 survived. | **Fixed/measured**: `label_prior` is now an error (pooled over #options, z > 3); `constant_answer_floor` reported next to memoryless. v1 interleave prior is *not significant after pooling* (low power) — balanced construction is a v2 requirement. |
| M6 | Major | Tests mirror the implementation: A1's `hasattr` check is tautological; swap audit and A8 share one decoder; 6/8 generator mutants survived. | **Partly fixed**: two independent label oracles (privileged state; raw frames via slot-permutation tracking, press and placed counting) are tests; mutation regression added. Remaining weakness: generator-level mutants (reveal-late, j=0, label prior) are only partially covered; see v2 acceptance. |
| m | Minor | Stored PairRecord digests were never recomputed; `obs_only_acc > 0.5` is the pair cap, not chance; Wilson CIs over 2n episodes violate the cluster unit; metadata audit has low power. | Stale records now an error; cluster CIs in harness; `summarize` no longer reports episode CIs. Power caveat stays. |

## Documented conventions (v1)
- `process_count`: one `pressed` tick = one press; adjacent pressed ticks count separately. This is an undocumented-until-now convention that v1 relied on; it is why A4's "divergent action" holds in v1.
- `test_ood` (v1) = longer horizon and larger object count only; `test_counterfactual` (v1) = unseen object-name combinations built from seen atoms. Neither supports a claim of structural or identity generalisation.
- Decoder `oracle_decoder_*` is a decoder self-check: its 1.0 on full history is agreement between generator and decoder, not a method score.

## What was not falsified
Label correctness; pair identity within clean pairs; split separation by id/script/content/mapping; planted structural-channel/wrist/length/id/final-obs leaks are caught; all 51 original M1 tests pass on the unmodified copy.

## SODH v2 specification (required before M3 or any comparative claim)
New version id `sodh/2` and new frozen manifest; v1 files never edited.
1. **Tracking**: anonymous cups; observed swap tokens; answer = slot reached by following swaps after the reveal (static-cups control must change outcomes).
2. **Release ticks** between presses; one pressed tick = one press unambiguously.
3. **Generated axes**: `evidence_position` ∈ {early, mid, late} and horizon crossed independently, recorded per pair; coverage-conditioned reporting is primary.
4. **Regimes**: `ood_length` (only longer) and `ood_structure` (same length; distractors, event-density schedule, pause/work ratio, layout permutation) are separate splits; `test_counterfactual` uses vocabularies with **disjoint atoms**.
5. **Balanced answers by construction**: choose two distinct answer positions uniformly (marginal uniform); interleave labels no longer biased to the middle.
6. **Alpha-invariant dedup gate**: `alpha_twin_rate` ≤ a registered threshold (set from a pilot, before test evaluation); larger test splits (≥ 60 pairs/family).
7. **Acceptance**: g0/2 passes; mutation suite (each generator mutant from `reviews/` plus reveal-late, j=0, label prior) killed for the right reason; two decoder variants and one knowledge-free learner reported side by side; a ranking that flips between decoder variants is reported as *not a memory result*.
Exit criterion: if v2 cannot be built so that a knowledge-free learner and a decoder-variant sweep agree on the sign of an effect, no memory-selection claim is made from synthetic data.
