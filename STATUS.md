# Project state — 2026-11-11 (after M1 review and M2 CPU infrastructure)

All results are `mode: synthetic` (symbolic frames, hand-written or toy policies). **No UC-VLA hypothesis (H1–H4) has been tested; no claim about a VLA, a real benchmark, or memory selection follows from anything here.**

## Done
- M0 research scaffold; M1 episode schema, causal views, deterministic SODH v1, pair-level splits, G0 (see `docs/09`).
- Git: baseline `5322034` tag `m1-synthetic-baseline`; SODH v1 **frozen** (`results/synthetic/sodh_v1/FROZEN.json`, guard test). No remote, nothing pushed.
- Independent-context adversarial review of M1 (`docs/18`, `reviews/`): 2 Blockers, 6 Majors. G0 hardened to g0/2 (full decision view, mutation regression for 8 planted leaks); B2 partly fixed; v1 downgraded to *unit-test fixture*.
- M2 CPU infrastructure (`docs/17`): PolicyAdapter + access boundary, budget validation by the harness, train-only fitting, history swap/deletion attribution, exact paired tests + Holm, run records with real git SHA, fresh-process reproducibility check, evaluation ledger, benchmark-source conformance contract.
- Benchmark Readiness Matrix (`docs/16`): web research by a subagent, **unverified**, no downloads.
- 109 tests, ruff clean.

## Not done (and why)
| Item | Blocker |
| --- | --- |
| **M1 DoD** — HDF5/LeRobot importers, a real-benchmark adapter, image-level shortcut/dedup audits, independent *human* review | no data approved/downloaded; licences unverified (`docs/16`); symbolic frames have no pixels |
| **SODH v2** (tracking cups, release ticks, generated evidence-position axis, structural OOD, disjoint-atom holdout, alpha-dedup gate, larger test) | not built yet; spec in `docs/18`; required before M3 |
| **M2 DoD** — OpenPI adapter, D&R/Mem4Act, learned fixed-latent baseline | checkpoint licence (Gemma terms) undecided, no GPU authorisation, no code/licence review |
| LEVI bridge | deferred by operator decision |
| M3–M7 | not started |

## Answers to the M2 acceptance questions (honest status)
1. *Does a performance difference come from history rather than current obs or data regularities?* The harness can **test** it (obs-only cap, swap, critical vs matched non-critical deletion, coverage-conditioned reporting), and history-free agents are shown to score 0 pair-success while a knowledge-free learner stays low. For **v1** the attribution is only as good as the generator; v1 has known regularities (`docs/18`), so no such claim is made from it.
2. *Do baselines obey the same information and evaluation rules?* Yes by construction and by test: one evaluation path, identical non-context inputs (parity digest), harness-measured budgets, rejected illegal contexts. Caveat: the decoders carry task knowledge; strict/clock variants bracket it.
3. *Can every result be reproduced independently?* Fresh-process runs with different `PYTHONHASHSEED` give identical run_id/result hashes (`scripts/repro_check.py`, tested). Real-VLA/GPU nondeterminism is out of scope.
4. *Is there a clear interface to a real VLA / benchmark?* Interface and conformance checks exist; **no real adapter exists**, importers raise.
5. *Do tests/data support or refute core UC-VLA hypotheses?* No — they validate infrastructure. Hypothesis tests need v2 plus learned memory policies (M4).

## Go / No-Go
- **GO (CPU):** build SODH v2 and its acceptance suite; keep M2 CPU infrastructure; obtain operator decisions below.
- **NO-GO:** any M3 rollout, GPU job, OpenPI/MME-VLA checkpoint load, large downloads, RoboMME data download, real robot; any paper claim from v1/M2 numbers. G1 not evaluable yet.
- Operator decisions needed: (1) approve v2 build; (2) RoboMME: approve a single-task-archive download for schema checking? (3) Gemma-terms decision for pi0.5-derived checkpoints; (4) remote repository (none created).

Verify: `make smoke PYTHON=.venv/bin/python`.
