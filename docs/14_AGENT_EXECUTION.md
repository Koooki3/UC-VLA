# 14 | Agent execution protocol

For autonomous coding agents: implement *one milestone per PR*. Read `AGENTS.md` and `agents/MASTER_IMPLEMENTATION_PROMPT.md`. An agent should not immediately write 50K lines. Prioritize falsification: data causality → reader viability → matched baselines → utility estimator → adaptive manager → RL → real robot.

## Required agent artifact per PR

- `TASK_PLAN.md`: task objective, novelty claim affected, files changed, tests, risks, alternative simpler method.
- One coherent implementation with stable interfaces.
- Unit tests including at least one adversarial negative control.
- Reproducibility: command output, source commit, exact dependencies, runtime/hardware/seed record.
- Explicit declaration `implemented`, `mocked`, or `not implemented` for all interfaces.
- Performance regression gate when touching online path.

## Default policy

Do not merge training/validation/test manifests. Never use the test split for model design or repeated tuning. No `pip install` from mutable `main` in production jobs. Never overwrite raw results. Avoid introducing changes to LEVI itself unless specifically scoped in a separate reviewed PR.

## Reviewer roles

- Theory reviewer: proof assumptions and differentiability/identifiability.
- Novelty reviewer: nearest recent work and identical contribution detection.
- Systems reviewer: scheduling, stale plan, GPU residency and asynchronous semantics.
- Evaluation reviewer: leakage, controls, power, replication.
- Safety reviewer: actuation authority, provenance, data permissions.

A coding agent may run these as a structured checklist, but must not claim them to be independent human review or externally verified until completed by genuine independent reviewers.
