# 31 | MLDA / ROSE cluster notes for UC-VLA (from the operator's briefing, 2026-11-13; no credentials, hosts or user names recorded)

Status: **nothing has been run or submitted on either cluster from this project.** Large-data / GPU-heavy experiments are intended to move there (operator, 2026-11-13) but each use needs explicit authorisation (`docs/10`, `docs/15`).

## Facts relevant to our design
- Both clusters need the campus VPN first; connection loss and host outages happen. Never copy the operator's private keys to another machine.
- **MLDA** (shared 4×24 GB workstation, no scheduler): ≤2 GPUs per account, clean GPU memory after use, admin purges idle memory (2 days) or >2-GPU use; host reboots/outages and externally killed processes (exit 137) occurred; **no auto-restart cron without admin consent**; work only under the project's own directory; pull results to the local machine, then clean.
- **ROSE** (Slurm): the account in the briefing is a **shared account belonging to another project's author, and the cluster's terms forbid account sharing** → **this project must not use it without explicit operator confirmation**; do not touch other people's jobs/dirs. Constraints: `--gres=gpu:<model>:N` with explicit model, no `--mem/-c/--exclusive`, ≤3-day batch jobs, login node light only (no recursive scans), Blackwell GPUs need CUDA ≥ 12.8, new project storage must be created interactively by the operator. Paid QoS consumes the group's SU quota (check balance first); free QoS is killable/preemptible (needs `--requeue`; preemptions are frequent).
- Shared GPUs: announce planned card counts/time windows to the other project's coordinator before use.

## Consequences for our experiments (design rules, effective now)
1. Every cluster training entry point must be **resumable** (save parameters, optimiser, RNG state, data cursor, log position; resume into the same output directory; truncate log lines beyond the saved step); resume equivalence must be tested on CPU first (kill mid-run, restart, compare with an uninterrupted run).
2. Run queue file + per-run lock + bounded retries; externally killed ≠ crashed; completion = exit code 0 **and** final checkpoint present (not wrapper exit 0).
3. Every run records host, GPU, job id, each (re)start, resume count; results are pulled back with a checksum manifest before any cleanup.
4. Pre-register criteria before reading results; paired same-host/same-seed comparisons; no borrowed numbers from other hosts; deviations logged. (Same as our Claim Ledger/G1 protocol discipline.)
5. No test-set peeking from cluster runs; sealed confirmatory sets stay sealed.
6. Cost/GPU-hours must be measured by a pilot, then requested (no invented estimates).

## Before any cluster use (checklist for the operator)
[ ] which cluster and which account (a personal ROSE account, or explicit permission for the shared one) · [ ] GPU/time window agreed with the other coordinator · [ ] storage directory created · [ ] resumable-entry CPU smoke test passed · [ ] `--dry-run` / sbatch `--test-only` reviewed · [ ] data/licence status of everything uploaded · [ ] cleanup/pull-back plan.
