# 19 | SODH v2 — pre-registered research specification (frozen before implementation)

Date: 2026-11-11 · Status: **specification frozen at the commit that introduces this file; implementation follows.** SODH v1 stays a permanent regression fixture; v2 is a new version (`sodh/2`) with its own config, manifest and freeze file. Nothing in v2 may be tuned to make an expected ranking appear. If independent learners fail to reproduce a history-dependence effect, that is reported as a failure, not repaired by editing frozen test splits.

## 1. Purpose and non-goals
Purpose: a falsifiable environment for four separate questions — *is memory necessary, is it used, is a given retained context sufficient, and how efficiently does a compressor reach sufficiency*. Non-goals: showing that any algorithm works; modelling vision (symbolic observations with an explicit sensor model; visual recognition ability is not confounded with memory because noise is a declared, ablatable parameter `ε`); any real-robot or real-benchmark claim.

## 2. Environments (hidden state, observations, success)
All families emit the same `EpisodeRecord`/`CausalView` contract. **Success is defined by the environment's hidden state, not by a decoder**: `privileged.success_actions` is the set of actions that succeed in the *true* state. The label `target` is the **posterior-optimal action** computed by an independent exact-enumeration oracle (`oracle2.py`) that sees only what a policy could see; the oracle's expected success under the true generative model is reported next to every metric.

| Family | Hidden state | Observation per tick | What must be remembered | Uncertainty |
| --- | --- | --- | --- | --- |
| `track` | slot of one ball under n **anonymous** cups | `swap=i-j` visible, `swap=none`, or `swap=?` (occluded; swap-or-not unknown); one `reveal=slot:r` tick with sensor error ε | identity through *all* swaps after the reveal | occlusion, noisy reveal |
| `events` | count of **accepted** button triggers | `contact=up/down/?`, `lamp=on/off/?` | validity-aware count: holds count once, repeated edges count separately, a rising edge with `lamp=off` is a jam (invalid), dropout ticks hide both | jams, dropouts |
| `resume` | which items are placed (set), with retries and (OOD) removals by another agent | `attempt=<item>/none/?`, `result=ok/fail/?`, `removed=<item>/none` | per-item state across a long, interleaved, distracted history | occluded results, dropped ticks |
| `control` | none (negative control) | current frame shows the answer | nothing | none |

Decision: `track` → `pick_slot_k`; `events` → `press`/`stop` for target count K; `resume` → `place_<item>` (success iff that item is truly unplaced); `control` → `pick_left/right`.

## 3. Counterfactual pairs
Mates share the **entire decision-time view** (all channels, final tick, query, lengths) and differ only in earlier history; the oracle's optimal actions must differ and each posterior must exceed a registered confidence gate (best−second ≥ 0.30). Pairs violating this are rejected, not repaired. `control` pairs share the answer. Each pair records `template_id`, `mechanisms`, `evidence_position`, `noise`, oracle margins.

## 4. Generated axes and regimes
- `evidence_position` ∈ {early, mid, late} (position of the decisive tick) crossed with `horizon`; both recorded; primary cells are family × position.
- Regimes: `in_dist` (→ train/val/test_id), `ood_length` (only longer), `ood_structure` (same length; mechanisms absent from training: `track` swap bursts + flicker distractor key; `events` holds/double-taps + new inter-event gap distribution; `resume` removals by another agent + distractor attempts), `heldout_mapping` (item vocabularies with **disjoint atoms**; only `resume`/`events` carry names).
- Noise: `ε` ∈ {0.0, 0.1} is a declared task parameter; a clean (ε=0) twin of every noisy cell allows separating perception from memory.

## 5. Leakage gates (mechanism, template, relation, event structure — not ids)
1. `template_id` = digest of the alpha-invariant canonical hidden script; globally unique; `alpha_twin_rate` between train and any other split must be 0.
2. Mechanism disjointness: every `ood_structure` pair carries a mechanism tag absent from all train/val/test_id pairs; in-dist splits only use train mechanisms.
3. Structure signature (family + event-type multiset + mechanism) overlap is reported between splits; for OOD it must be zero.
4. g0/2 full-view audit, token probe, constant-position prior and mutation suite must pass; new mutants: reveal-late, label prior, pause-only, lamp-leak, id-leak.
5. Size: ≥ 60 pairs per (family, split) for test_id; ≥ 40 for each OOD split.

## 6. Agents for cross-validation (no single decoder defines "ideal memory")
(a) `PosteriorOracle` over the *same retained context* (dropped ticks are unknown); (b) two hand-written heuristic decoders per family (naive, conservative) and a *deliberately wrong-but-natural* one (contact-edge counting, last-attempt-only) as falsifiers; (c) knowledge-free learners with identical access: `BagOfTokensNB`, `ObsOnlyLookup`; (d) providers `NoHistory`, `AllHistory`, `LastK`, `UniformK` at matched tick budgets.

## 7. Registered metrics (primary first; cluster = pair; 95 % percentile bootstrap, 2000 resamples, seed fixed)
- **Primary:** `PairOptimal` = both mates' decisions equal the oracle-optimal action. Secondary: `EnvSuccess` (true state), `Agreement` per episode.
- **Memory Necessity** N = EnvSuccess(oracle, full history) − EnvSuccess(best history-free agent, including a trained obs-only lookup). Reported per family/regime.
- **Memory Use** U(agent) = (swap-follow rate − chance) and (drop on critical-evidence deletion − drop on matched non-critical deletion). Reported separately; an agent can be necessary-and-unused.
- **Memory Sufficiency** S(ctx) = EnvSuccess(PosteriorOracle restricted to ctx) / EnvSuccess(PosteriorOracle, full history). Independent of any heuristic decoder.
- **Compression Efficiency** E = smallest tick budget k* with S ≥ 0.95 and area under S(k) over k ∈ {1,2,4,8,16,full}; compare selectors at equal budget and equal tokens.
The four are never combined or substituted for one another. Coverage-conditioned results (did the context contain the decisive ticks) accompany every budget result.

## 8. Statistics and power
Exact sign test on discordant pairs, Holm over the registered family (family × regime × budget comparison), clustered bootstrap CIs. Before evaluating any test split a pilot on the **val** split estimates variance; the minimal detectable effect at n pairs is reported (`docs/21`, `docs/23`). Non-inferiority margins, if used, are registered after the pilot and before test.

## 9. Pre-registered falsification criteria (each has a test)
F1 a history-free agent with PairOptimal > 0 on a history-relevant cell (excluding oracle-forced chance 0.25 slack); F2 swapping history does not move the oracle's action; F3 deleting critical evidence does not lower oracle success more than matched non-critical deletion; F4 any single token/channel/position probe beats the label prior; F5 a name-blind or length-agnostic policy solves `ood_structure`/`heldout_mapping` cells; F6 static-cups (no occlusion/with swaps removed) leaves `track` outcomes unchanged; F7 ranking of two selectors flips between the oracle-sufficiency measure and the learner/decoder measures — then **no memory-selection claim** is made for that comparison (this is B2b's closure test); F8 independent learners fail to show use of history — reported as is, test splits frozen.

## 10. Freeze protocol
Config file hash, `sodh_v2` manifests and `FROZEN.json` written once; evaluation ledger for every test-split touch; deviations from this document are logged in `docs/23` with reason and date, never silently applied.
