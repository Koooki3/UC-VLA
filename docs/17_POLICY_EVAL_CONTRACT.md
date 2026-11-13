# 17 | M2 policy-evaluation contract (CPU, synthetic) and benchmark-source contract

Status 2026-11-11: **implemented and tested on symbolic SODH; no real VLA, no real benchmark.** All modules are in `src/ucvla/`.

## Why this exists (research necessity)
H1–H4 (`docs/07`) compare memory policies at matched budget. Such a comparison is only attributable to *history* if (i) every agent is run by one code path, (ii) the only thing that differs is the context it may use, and (iii) the evaluator can intervene on history without touching the world or label. This layer makes those three properties testable before any model exists. Failure exit: if an agent can obtain information outside `PolicyInput`, or if a result is not bit-reproducible in a fresh process, no comparison made with this harness may be reported.

## Information boundary
| Object | Contents | Policy may see? |
| --- | --- | --- |
| `PolicyInput` | current observation, `TaskSpec` (task, instruction, legal options, params), `Context` | yes — the only input |
| `Context` | validated raw past ticks, `history_len` (a clock count), cutoff | yes |
| `CausalView`, `EpisodeRecord` | frames/proprio/commands/online events; targets, privileged state, annotation events, provenance, ids, pair partner | **never** |
| `ActionChunk` | *predicted* actions only | output |
| `CommandedAction` / `ExecutedAction` | written by the harness (executed is simulated in synthetic mode, logged as such) | no |

`ContextProvider` (which past ticks) is separate from `PolicyAdapter` (how to decide). The harness validates provider output (real, causal, strictly ordered, within declared **and** matched budget) and measures cost itself (`ticks` and `tokens`); self-reported cost is ignored. Latent/learned contexts are rejected until M4.

## Policies (`readers.py`) — all CPU, none is a VLA
- `OracleDecoder`: hand-written per-task decoder. **Information-sufficiency probe, not a method.** Its 1.0 on full history is by construction.
- `ObsOnlyLookup` (trainable): current-observation memoriser; capped at pair-success 0 on history-relevant families.
- `BagOfTokensNB` (trainable): knowledge-free generic learner; does not solve the relational tasks, showing that the oracle numbers come from decoder knowledge.
Fitting is allowed on `train` only (`fit_agent` raises otherwise) and the fitted-on episode digest is recorded.

## Causal attribution (`causal_eval.py`)
Per agent and family: accuracy; **history swap** (A's current frame + B's history; should follow B for relevant, stay for control); **critical vs matched non-critical history deletion** (critical ticks come from annotation events, evaluator-side). `uses_history` is a diagnostic requiring swap-follow ≥ 0.9, critical drop ≥ 0.3, non-critical drop ≤ 0.1. It is conditional on what the provider retained (a tick never in context cannot matter).

## Reproducibility
`run_m2_eval.py` writes `run.json` (run_id = hash of config, data hashes, agent specs, seed and result hashes; real git SHA + dirty flag; env; code-tree hash) and `report.json`; `repro_check.py` re-runs in fresh processes with different `PYTHONHASHSEED` and requires identical hashes. Test splits are logged in an append-only evaluation ledger. A dirty tree is flagged as non-attributable.

## Benchmark-source contract (`benchmark_contract.py`)
`EpisodeSource` + `SourceManifest` (pinned revision, licence status, protocol ref, privileged fields) + `check_source_conformance` (schema validity, cross-split id/content duplicates, causal view prefix property, provenance origins `native|derived|absent`, no executed stream unless native). `SODHSource` is the reference implementation. **HDF5 and LeRobot importers are `not implemented`** and raise; RoboMME and others remain `not implemented` (see `docs/16 (withheld from the public release)`). Passing conformance on real files and operator-verified licences are both required before a source counts as integrated.

## Falsifiable expectations (what would refute the design)
- A history-free agent ever scoring pair-success > 0 on a relevant family ⇒ the task or harness leaks (G0 breach).
- `uses_history` true for an agent whose provider retained nothing ⇒ attribution method is broken.
- Any difference between independent-process runs ⇒ nondeterminism; results invalid.
