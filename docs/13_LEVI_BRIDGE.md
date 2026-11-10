# 13 | LEVI / AERI / OpenPI integration design

> **Status 2026-11-11: LEVI adaptation is deferred by operator decision; nothing here is implemented.** The M1 `levi_bridge` prototype was removed.

## Repository review basis (2026-11-11)

Inspected public `Koooki3/LEVI` README and `docs/AERI_QUICKSTART.md`, `docs/AERI_EXTERNAL.md`, `docs/EVENTS.md`. LEVI is primarily a data validation/curation/evaluation workbench; its AERI pipeline separates motion authority, run journal, judges and external agents. The official docs state fake/dry-run validation exists while specific hardware acceptance is pending. Do not mistake project design claims for verified integration with UC-VLA.

## Principle: do not fork or rebuild LEVI

UC-VLA stays an independent research repository; LEVI provides read-only datasets/events and externally authorized evaluation sessions. Integration must be versioned and opt-in. Avoid duplicating LEVI's dataset conversion, event extraction, streaming UI, AERI hardware gate or journal ownership.

## Proposed read-only interfaces

- Dataset ingestion: access LeRobot versioned episode table/cameras and immutable source hash via export; adopt explicit timestamp and frame/units contract.
- Event ingestion: map `levi.events` `SignalObservation` (measured/commanded/derived, clock domain, source frame index, time window) and `EventCandidate` (candidate vs verified) to UC-VLA EvidenceRef, never infer proof from salience.
- Evaluation: read AERI state journal and blind human-approved outcome; the model never writes operator labels, controls the robot or replays a reset policy without an authorized executor.
- Memory evidence API: allow LEVI to optionally export permitted event IDs; UC-VLA only writes to its own derived state/log.

## Proposed interfaces (to implement after pinned commit review)

```
LEVIEventAdapter.read(episode_id, t_start, t_end) -> list[EvidenceEvent]
UCVLAPolicyAdapter.predict(snapshot, observation, instruction, deadline) -> PlannedChunk
AERIReportImporter.read_verified_outcomes(run_dir) -> Results
```

Validate LEVI version/commit and event schema with contract tests. No direct imports to LEVI internals in `src/ucvla/` until an integration adapter exists. Do not use AERI control APIs for autonomous experiments without an operator.

## OpenPI adapter

Read `Physical-Intelligence/openpi` repo and pinned checkpoint; expose normalized observation and action chunk interface, training checkpoint adapter and latency profiler. Start with frozen action expert, then LoRA/small context reader. Avoid assuming OpenPI weights/robots identical to real platform. Action normalization must match dataset transform.

## Validation sequence

1. Synthetic fake events + CPU contracts.
2. Offline LEVI episode export (read-only), clock/domain checks.
3. OpenPI simulation inference in isolated process.
4. LEVI/AERI fake-run shadow test; compare memories to human review.
5. Operator-approved real evaluation only after external safety gate.

External third-party repos are referenced, not vendored, cloned or changed in this scaffold.
