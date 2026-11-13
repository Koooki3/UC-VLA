# 15 | Open design decisions — do not guess

| Decision | Safe default | Evidence required | Owner |
| --- | --- | --- | --- |
| Robot embodiment and camera topology | offline data only | actual device contract | operator |
| VLA pretrained backbone | OpenPI family candidate | checkpoint rights/compute/baseline reproduction | researcher |
| GPU allocation | none | site inventory and quotas | cluster operator |
| SLURM account / partition | unset | real partition, account, wall limit | cluster operator |
| Target p95 latency | unset | robot control period and safety margin | hardware team |
| Memory token budget | sweep candidates | pilot memory/latency Pareto | researcher |
| Memory utility estimator | paired sim MC + ensemble | value error & cost on pilot | researcher |
| LEVI event bridge | not implemented | operator re-enables it (decided 2026-11-11 to skip LEVI adaptation) | operator |
| Benchmark integration | RoboMME first | official code/commit/dataset license | engineer |
| Real robot task | no actuation | hazard/risk assessment and operator approval | operator |
| Submission date/venue | undecided | achieved evidence and venue policy | authors |

These are non-blocking for CPU contracts/scaffold. They **do** block GPU experiment launch or real-robot activation when required.

## Added 2026-11-12
| Decision | Safe default | Evidence required | Owner |
| --- | --- | --- | --- |
| Project licence; git author identity/history before any remote | none; no remote | `docs/24` checklist | operator |
| Backbone for open release | none (pi0/pi0.5 NO-GO) | licence checks of SmolVLA/RDT-1B/Octo weights + encoders (`docs/20 (withheld from the public release)`) | researcher + operator |
| LeRobot-format official sample | not downloaded | licence + size + operator approval | operator |
| Simulator replay (M3) for real paired tests | not authorised | compute/licence/cost plan | operator |
| SODH v3 (longer horizons) | stay on frozen v2 | justification from compression-claim need | researcher |
| Sequence-capable CPU learner (F8) | not built | spec + train-split-only protocol | researcher |
