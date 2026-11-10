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
