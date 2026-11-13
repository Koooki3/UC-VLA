# TASK_PLAN — M1: deterministic episode data substrate and causal SODH suite

Date: 2026-11-11 · Scope: M1 only (`docs/09_IMPLEMENTATION.md`, `agents/MASTER_IMPLEMENTATION_PROMPT.md`). CPU only, no GPU, no robot, no network at run time.

## Objective and claim affected

Provide the data substrate needed for gate **G0** (`docs/00`, `docs/07`): a versioned episode schema, a deterministic Same-Observation/Different-History (SODH) generator, pair-level split checks, anti-shortcut audits, and one usable benchmark adapter with fair symbolic reference readers. **No claim about novelty, learning, or a VLA is affected**; every number produced here is `mode: synthetic`.

## Acceptance tests (written first; all CPU)

| ID | Test | Falsifies |
| --- | --- | --- |
| A1 | Every cutoff of every episode: causal view contains no item with `known_at_ns > cutoff`, no annotation / privileged / target data (I1) | future leakage |
| A2 | Mutating any field (timestamp, id, payload, hash) of a serialized episode is detected by validation or content hash (property tests, hypothesis) | identity / timestamp corruption |
| A3 | Same seed ⇒ byte-identical JSONL and manifest; different seed ⇒ different content | non-determinism |
| A4 | For each history-relevant pair: identical final-observation digest, different correct action | non-SODH pair |
| A5 | Split checker rejects: pair across splits, shared scene script, duplicate content across splits, held-out identity mapping seen in train, OOD not longer than train | pair / group leakage |
| A6 | Shortcut audit: current-obs-only lookup ≤ 50 % on pairs; metadata features (episode length, frame/event counts, ID strings, label position) near chance; audit **does** flag a deliberately leaky generator (negative control for the auditor) | unsound audit |
| A7 | History-swap control: swapping relevant history flips oracle decision; swapping history in the `irrelevant_history` family does not | task has no diagnostic power (G0) |
| A8 | Memoryless reader has pair-both-correct = 0 on relevant families; full-history reader solves them; window/uniform readers degrade with horizon under a matched event budget | reader/budget semantics |
| A9 | Event alignment: clock-domain mapping is monotone, unknown clock domains rejected, sample-and-hold never uses future samples | clock errors |

## Files affected (new unless stated)

`src/ucvla/{canonical,episode,alignment,sodh,splits,audit,adapters,baselines,dataset_io,stats}.py`, `scripts/{generate_sodh.py,audit_sodh.py,package_manifest.py}`, `configs/sodh/sodh_v1.yaml`, `tests/test_*` for each module, `results/synthetic/sodh_v1/*` (synthetic outputs, separated from any real result). Updated: `pyproject.toml` (pyyaml, hypothesis), `.gitignore`, `Makefile`, `STATUS.md`, `docs/09_IMPLEMENTATION.md`, `docs/10`/`README` command lines, `PACKAGE_MANIFEST.json` (regenerated). Existing contracts/memory/toy and their 12 tests are not modified.

## Alternatives considered

1. Reuse `toy.py` only → no multi-camera schema, no splits, no audits; fails DoD. Rejected.
2. HDF5/LeRobot importers now → no data or license review yet, would add unverifiable code. **Deferred**: declared `not implemented` in the adapter registry.
3. Use numpy/pandas → unnecessary; stdlib + canonical JSON keeps hashes reproducible. Dev-only: hypothesis, ruff, pytest. Runtime: pyyaml (config loading only).

## Reproducibility impact

Canonical JSON (sorted keys, ASCII, no NaN) + sha256 over every artifact; per-item seeds derived by hashing `(master_seed, family, index)` so adding items never shifts others; config digest and a source-tree digest are recorded because the directory is **not a git repository** (`git_commit: null`, flagged, not faked). Split manifests are write-once: a new version needs a new ID.

## Gate behaviour

G0 is implemented as an executable report (`audit_sodh.py`) that exits non-zero if any anti-shortcut or history-swap check fails. Passing G0 on a *symbolic* generator only shows the generator is well-formed; it says nothing about learned models or real benchmarks.

## Risks / unresolved

- Symbolic observations ≠ rendered images; image-level shortcut audit (CNN/perceptual) is `planned`, not done.
- RoboMME / LeRobot / HDF5 adapters `not implemented` (no dataset, license review pending, `docs/15`).
- **LEVI bridge deliberately not implemented**: the operator advised (2026-11-11) against LEVI adaptation work; this deviates from the M1 text in `agents/MASTER_IMPLEMENTATION_PROMPT.md` and is recorded as deferred in `docs/09`/`docs/15`. `SourceType.COMMANDED` and the clock-alignment utilities remain, independent of LEVI.
- Python 3.14 (system) is outside `pyproject` range; a uv-managed 3.12 `.venv` is used.


---

# TASK_PLAN — M2 (CPU-only) and post-review work, 2026-11-11

Scope: operator decisions of 2026-11-11 (git init approved; benchmark readiness review approved; M2 CPU-only approved; adversarial M1 review; LEVI and GPU out of scope). Done in this order: git baseline → parallel (benchmark research, M1 review) → M2 infrastructure → review fixes.

## Objective and falsifiable questions
Make policy comparisons attributable to *history*, rule-identical across agents, and bit-reproducible, before any model exists. Falsifiers (each has a test): a history-free agent with pair-success > 0 on a relevant family; an agent receiving non-context information (parity digest differs); an illegal/over-budget context accepted; a run that differs across fresh processes; a planted leak in any decoded channel that G0 misses.

## Acceptance tests (all passing, 109 tests)
`tests/test_harness.py` (M1-parity of provider+decoder, illegal contexts, matched budgets measured by harness, parity digests, no label/privileged fields in PolicyInput, seed/id independence, predicted/commanded/executed logging, train-only fitting), `tests/test_causal_eval.py` (swap/deletion attribution incl. history-free negative controls, two independent label oracles), `tests/test_g0_mutants.py`, `tests/test_stattests_contract.py` (exact tests, conformance with negative controls, importers fail loudly, fresh-process reproducibility).

## Alternatives considered
Subagent-free serial work was not used: two read-only subagents ran in parallel (review, licence research) because 5-hour quota readout was unavailable and the briefs were bounded. Learned baselines on GPU, OpenPI loading and real importers were rejected as unauthorised. Merging M1's `SymbolicPolicy` into the new harness was *not* done: it is kept as a frozen reference and proven equivalent by a parity test.

## Reproducibility impact
New: run_id content address, real git SHA + dirty flag (`attribution_warning` if dirty), evaluation ledger for test-split use, `FROZEN.json` guard for SODH v1, write-once data.

## Gates and exit conditions
G0/2 passes on v1 (as a fixture). M2 DoD **not** met (no VLA adapter). Exit: if SODH v2 cannot be made so that a knowledge-free learner and the decoder-variant sweep agree on an effect's sign, stop making memory-selection claims from synthetic data and pivot to an intervention-benchmark/negative-result framing (docs/07 G2).

## Next phase (needs operator go-ahead): SODH v2 → H1-style matched-budget pilot on CPU toy learners → licence-cleared real-benchmark schema check.

---

# TASK_PLAN — review round 2 (2026-11-12): Track A / Track B, executed

Operator decisions of 2026-11-11 (2nd): SODH v2 approved after spec freeze; RoboMME single-task sample ≤1 GiB (train only, official channel, licence checked); checkpoints NO-GO with a licence/provenance review; no remote; CPU only. Two parallel read-only subagents (licence review, RoboMME metadata) while the main session built v2; both reports are unverified and labelled.

## Objective and falsifiers
Track A: a benchmark that can refute Memory Sufficiency / compression hypotheses (spec F1–F8 in `docs/19`). Track B: remove the symbolic-to-visual gap at the contract level (real HDF5 → same Episode/CausalView/PolicyInput). Falsifier outcomes: F1–F6 passed by tests; **F7 failed in 17/50 cells (events, resume), passed for track**; **F8 triggered** (independent learner shows no history use); G0 v2/1 blind to XOR/parity leaks (fixed by D1; weak leaks still undetected).

## Acceptance tests (152 tests): `tests/test_v2.py` (26), `test_frozen_v2.py`, `test_hdf5_import.py`, `test_offline_replay.py`, mutation suites, v2 fresh-process reproducibility; real-data tests skip when `datasets/` is absent.

## Alternatives rejected / deferred
Making v2 easier until the expected ranking appears (explicitly forbidden); redesigning to make learners succeed; downloading more RoboMME tasks or val/test; any checkpoint; editing frozen v1/v2 (a defect found pre-freeze was fixed and logged in `docs/23` §6). Deferred: LeRobot importer, sequence learner, SODH v3.

## Reproducibility impact
Real git SHA in run records; `FROZEN.json` for v1 and v2 with guard tests; data-use record for the real sample; ledger of test-split use (M2 and v2 evaluations); `requirements.lock`; scan + manifest in CI.

## Exit conditions
If a sequence learner cannot show history use on v2 → report as negative result, do not edit frozen splits. If F7 stays unstable for events/resume → no selector-ranking claims for them. Any move to GPU, checkpoints, simulator or remote needs a separate operator-approved review.


---

# TASK_PLAN — review round R3 execution (2026-11-13)
Order followed: P0 (G1 acceptance fix, G2 utility-model fix, public CI, parameter provenance) → P1 (sealed confirmatory G1, streaming interface, first learned G2 toy) → P2 (Observatory, governance). Falsifiers and outcomes: G1 order condition now enforced (main experiment fails; confirmatory passes events/resume, fails track reliability and events length-OOD); G2 additive-cost claim withdrawn; CI green only after fixing exec bits, manifest content (interpreter version), tracked-file manifest and dependency layering. Next: finish `g2_learned` analysis; events length-OOD diagnosis on non-sealed data; SmolVLA mock adapter (CPU, no weights); G1/G2 adversarial round 2; GPU pilot request only after Blockers B1/B2 of docs/38 are resolved. Exit rule unchanged: negative results are reported, sealed/frozen sets are not tuned.
