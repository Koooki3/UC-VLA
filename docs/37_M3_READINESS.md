# 37 | M3 simulation & compute readiness report (2026-11-13) — preparation only, **no rollout authorised or run**

Basis: measured facts from this project, operator briefings, and subagent web research (docs 16/22/31; third-party items **UNVERIFIED**). Where a number cannot be measured without running the simulator it is **not estimated**; a pilot is specified instead.

## 1. What M3 is
Paired simulator rollouts (docs/09 M3, docs/27): exact state reset + controlled seeds + counterfactual memory edits, fixed policy, paired return differences. Contract and leakage rules: `src/ucvla/rollout_contract.py` (CPU fake env, 2 tests): edits change what the policy is shown, never the world or the label; same seeds in both arms.

## 2. Candidate environment (RoboMME, per the earlier licence/format research — unverified)
Benchmark code Apache-2.0; training data apache-2.0 per card (copyright holder unnamed); ManiSkill/SAPIEN simulator with Vulkan rendering and CUDA ≥ 12.8 for Blackwell GPUs; Docker image reported; a CPU-render fallback is mentioned but its speed is unknown. Fixed seed lists exist per split (official `env_metadata`, train/val/test pairwise disjoint — verified for StopCube). **Not verified:** exact dependency versions, official success metric, whether any replay API gives hidden-history ground truth, whether paired resets with injected memory edits are supported without simulator changes. **The sample data has no hidden-history truth; paired real tests require simulator replay.**

## 3. Resource facts
| Item | Value | Status |
| --- | --- | --- |
| RoboMME train archives | 16 tasks, 56.4 GB compressed | measured sum (HF metadata) |
| Extraction | StopCube: 0.90 GiB → 19.6 GiB (21.8×); full set if the ratio held: ~1.2 TB | **ratio measured on one task only; extrapolation unverified** |
| Peak temp disk for one task | archive + extracted ≈ 20.5 GiB | measured |
| MLDA | 4×A5000 24 GB shared, ≤2 GPUs/account, 966 GB free on a shared 87 % full disk, host reboots/kills occurred | operator briefing + checks |
| ROSE | A5000-class and Blackwell nodes; 5090/pro6000 fully allocated at check time; free queue preemptible, ≤5 submitted, paid queue consumes shared SU quota | operator briefing + read-only check |
| Vulkan/GPU rendering on MLDA/ROSE | **unknown** (needs a smoke test) | open |
| Pixel decode cost | AV1 640×480: one 200-frame episode ≈ 0.7 s CPU (measured) | measured |

## 4. Experiments and budget (structure only — GPU-hours are NOT invented)
Per experiment the budget is `episodes × (seconds/episode from the pilot) × arms ÷ concurrent GPUs`. **Pilot (needs authorisation):** run 20 episodes of one task with a scripted policy, record wall-clock per episode, VRAM peak, CPU RAM, render FPS, disk written; repeat on MLDA (1 GPU) and ROSE (free queue, 1 GPU) to get both numbers. Until then no GPU-hour or SU figure is quoted. Candidate experiment list: E-A paired memory-edit rollouts with a *scripted* oracle-memory policy (instrument validation); E-B same with the G1 learned reader controlling memory selection over a frozen scripted action policy; E-C real-VLA integration (blocked: backbone licences, docs/32).

## 5. Statistical power and seeds (`scripts/m3_power.py`, McNemar approximation, α=.05 two-sided, power .8)
| discordant fraction | true Δ | pairs needed | with Holm m=5 |
| --- | --- | --- | --- |
| 0.2 | 0.10 | 155 | 231 |
| 0.2 | 0.05 | 626 | 932 |
| 0.4 | 0.10 | 312 | 465 |
| 0.4 | 0.05 | 1,254 | 1,866 |
Pairs = same simulator seed run under both memory arms; use disjoint seed blocks per purpose (train/val/test lists from the official metadata, plus a sealed confirmatory block), ≥ 5 model-training seeds, report cluster (task/seed-block) CIs.

## 6. Reliability, contamination, reproducibility
Resumable runs (per-epoch checkpoints with RNG/optimiser state, kill-and-resume bit-identical — tested on CPU and MLDA, `docs/31`); queue file + per-run lock + bounded retries; completion = exit code 0 **and** final result file present. Test-set protection: official test seeds never used for development; evaluation ledger for every touch; sealed confirmatory seeds registered (hash) before any run; no training on test-episode simulator state. Reproducibility: config/data/seed hashes + real git SHA in each run record, host/GPU/job id per (re)start, results pulled back with checksum manifests before cleanup.

## 7. Go / No-Go
**No-Go** for any M3 rollout now. **Go (needs authorisation) for the 20-episode pilot** on one task on MLDA and ROSE (free queue), after: (a) operator confirms the pilot scope; (b) RoboMME environment install is smoke-tested on CPU/one GPU; (c) the other project's coordinator is informed of the GPU window. Blocking unknowns: Vulkan availability, paired-reset support, ROSE capacity.
