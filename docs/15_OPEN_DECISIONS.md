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
