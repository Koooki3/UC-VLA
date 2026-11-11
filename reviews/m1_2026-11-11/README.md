# M1 adversarial review — falsification scripts and raw outputs (2026-11-11)

Produced by a delegated subagent that was told to *falsify*, not praise, `m1-synthetic-baseline`.
It is an **independent-context model review, not independent human review** (`docs/14`).
Scripts are verbatim; they reference absolute paths of the original session
(`<repo>/configs/sodh/sodh_v1.yaml`) and need `PYTHONPATH=src`. Some scripts assume
the pre-review (g0/1) `audit.py`; against the current tree a leak they planted may now be caught
(that is the intended outcome, see `tests/test_g0_mutants.py`). Findings and status: `docs/18`.
