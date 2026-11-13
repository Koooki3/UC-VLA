# Project state — 2026-11-12 (after review round 2: SODH v2, real-data importer, licence reviews)

All results are `mode: synthetic` unless a record says `real_data_*` (schema checks only). **No UC-VLA hypothesis (H1–H4) has been tested; nothing here supports a claim about a VLA, a real benchmark, or memory selection.** Reviews are independent-*context* model reviews, **not external human review**.

## Done (each in its own commit; `git log`)
- Git baseline `5322034` tag `m1-synthetic-baseline`; v1 frozen as a regression fixture; **no remote, nothing pushed**.
- Review status split: **B1** code fix closed (validity = "current attack set + measured power"); **B2a** closed; **B2b open**.
- **Track A** — `docs/19` pre-registered spec → **SODH v2** (tracking cups, validity-aware events, per-item resume, env-defined success, independent posterior oracle, generated evidence-position axis, structural OOD, disjoint-atom holdout, template/mechanism/alpha-twin gates, 126 test_id pairs/family), frozen `sodh-split-539d19e50cea`; G0 v2/2; G0 power study (`docs/21`); four memory metrics + F7 (`docs/23`).
- **Track B** — RoboMME single-task official **train** sample (964,971,180 B, sha256 verified, seeds ⊂ train, ∩ val/test = ∅) → real **HDF5 importer**, 100/100 episodes round-trip exact; offline replay adapter (same PolicyInput boundary); budgets now also bound proprio; `docs/22`.
- Licence reviews (subagent, unverified quotes): benchmarks (`docs/16`), OpenPI/π0.5/Gemma + alternatives (`docs/20`). **Checkpoints remain NO-GO.**
- Open-source preparation (no remote): repo scan clean, lockfile, CI, third-party manifest, rules (`docs/24`).
- 152 tests, ruff clean, `make smoke` green, fresh-process reproducibility for M2 and v2 evaluations.

## M1 / M2 DoD (not equated; M2 progress does not close M1)
| Item | Status |
| --- | --- |
| M1: episode schema, time validation | done (HDF5 timestamps are *derived* 20 Hz, unverified) |
| M1: HDF5 importer | done for RoboMME layout, one task's train sample |
| M1: **LeRobot importer** | **not done** — designed only; waits for an official LeRobot-format sample |
| M1: SODH paired generator, split manifest, negative controls | done (v1 fixture, v2 frozen) |
| M1: **image-level shortcut audit** | **not done** — synthetic frames have no pixels; real data only hash-level |
| M1: deterministic replay, no future info; duplicate detector | done (synthetic + real sample) |
| M1: one benchmark adapter usable | **partial** — offline replay of a real train sample; no simulator rollouts, no official metric |
| M2: PolicyAdapter, CPU fakes, matched budgets, run records, repro | done |
| M2: causal evaluation (same-obs/diff-history, swap, deletion, OOD) | done on v2 |
| M2: independent learner shows history use | **not met (F8)** — BagOfTokensNB does not; no sequence learner yet |
| M2: fixed-latent baseline, D&R, Mem4Act, OpenPI adapter | **not done** — not built / not reproduced / checkpoint NO-GO |

## Blocker / Major / Minor (open)
**Blocker:** B2b — construct validity of memory selection (F7: 17/50 cells flip under the registered rule; F8: no independent history-using learner). **Blocker for G1:** no memory reader exists to evaluate.
**Major:** G0 blind zone — leaks giving ≤0.67 reader accuracy undetected, order/gap/multi-token forms untested (`docs/21`) · posterior oracle shares the environment model (approximate for swap bursts) · symbolic-to-visual gap unmeasured; RoboMME data has no hidden-history ground truth (paired real tests need simulator replay = M3, unauthorised) · horizons 8–22 ticks limit compression claims (k=16 ≈ full) · only independent-*context* model reviews · licence ambiguities (Gemma flow-down, RoboMME copyright holder).
**Minor:** structural OOD bites a naive decoder in 1 of 3 families · Necessity partly by construction (pairs selected to need history) · `requirements.lock` has no hashes · extracted-file hash unverifiable against publisher · derived 20 Hz · repo scan is regex-only, no history scan · v1 `symbolic_baselines.json` remains committed (flagged as fixture artefact).

## G1 (architecture validity) Go / No-Go recommendation
**G1 cannot be evaluated** (no Memory Reader, no real policy). **NO-GO** for: M3 rollouts, GPU, any checkpoint load, large downloads, real robot, paper claims. **GO (CPU, next phase):** (1) a sequence-capable independent learner on frozen v2 (small recurrent/state-tracking model, train split only) → F8; (2) SODH v3 with longer horizons only if compression claims are wanted (v2 stays frozen); (3) LeRobot importer once an official sample is approved; (4) learner-based leak probes; (5) designing real-data paired tests without a simulator. Exit rule: if no learner reproduces history use on v2, publish as an intervention-benchmark / negative-result study (docs/07 G2).

## Operator decisions needed
1. Project licence and git author-identity/history policy before any remote (`docs/24`). 2. Whether to approve a LeRobot-format official sample. 3. Gemma terms / whether to pursue SmolVLA, RDT-1B or Octo as an openly releasable backbone (licence checks pending). 4. Whether to authorise simulator replay (M3) — needs compute/licence plan. 5. Approval of SODH v3 (longer horizons) vs staying on v2.

Verify: `make smoke PYTHON=.venv/bin/python`.

## Release scope (2026-11-13)
All releasable artefacts, weights included, are to be published permissively (`docs/26`). Hence: pi0/pi0.5 excluded for released weights; SmolVLA-derived weights blocked pending a data-licence audit; from-scratch models on synthetic data are releasable; MLDA and ROSE are authorised for large runs (ROSE shared-account responsibility with the operator); first cluster step = resume smoke test on one MLDA GPU (not yet run).

## LeRobot importer (2026-11-13)
Importer + audit done on the official sample (`docs/34`): 50 episodes, all checks pass; M1 DoD *LeRobot importer* met for this sample; *image-level audit* partial (statistical, no learned probe). G1-scaling batch running on MLDA (CPU, `docs/33`).
