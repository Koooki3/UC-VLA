# 10 | GPU cluster, scheduling and reproducibility SOP

## Deployment status

**No GPU hardware, SLURM cluster, credentials, storage, existing VLA checkpoint or dataset has been configured or accessed.** Included scripts are safe templates only; no jobs have been submitted.

## Preflight questionnaire (must be filled by operator)

Record: scheduler (SLURM/PBS/K8s), GPU SKU/memory/count, CUDA driver, networking/proxy, scratch quota, object storage, data access restrictions, walltime limits, partition names, account, project allocation, container policies and expected robot throughput. Do not blindly adopt the example resource class.

## Recommended environments

1. Core CPU: Python >=3.11 and `<3.14`, test/schema scripts only.
2. Simulation: pinned environment per simulator (RoboMME / RoboCasa / MuJoCo / Isaac) in separate venv/container when dependencies conflict.
3. Policy: pinned OpenPI checkout and matching CUDA/PyTorch; never import policy into `ucvla` core at module-import time.
4. Training: `uv` or locked pip environment / container digest; pin CUDA wheel compatible with driver.
5. Profiling: CUDA profiler, PyTorch profiler, GPU monitoring, native latency events and power/throughput if accessible.

## Source/artifact roots

```
$UCVLA_DATA_ROOT/{raw,derived,split_manifests,oracle_labels_train_only}/
$UCVLA_RUN_ROOT/{project}/{experiment}/{run_id}/
$UCVLA_CACHE_ROOT/{hf,uv,models}/
```

`raw/` immutable and checksummed. Derived data records source hashes and generator commit. Test data is read-only and cannot be accessed by training workers. All paths are external to git; protect sensitive physical-robot video.

## Scheduling

- First run `python scripts/doctor.py` and `python scripts/dry_run.py` on login node.
- Run `scripts/cluster_preflight.sh` only with explicit site inputs.
- Review `configs/cluster/slurm_train.sbatch`; set `UCVLA_CLUSTER_PARTITION`, `UCVLA_CLUSTER_ACCOUNT`, `UCVLA_TRAIN_COMMAND`, `UCVLA_CONTAINER_DIGEST` and dataset roots; *do not launch an empty placeholder as research*.
- Validate `sbatch --test-only` if site allows, then explicitly submit approved job.
- Train jobs get unique run IDs and write atomic checkpoints. Evaluators never alter train artifacts.
- Use array jobs for independent seeds, not for implicit data leakage.

## Profiling metrics

Capture model FLOPs when measurable, GPU utilization, CPU worker time, tokens read/written, VRAM peak, per-stage p50/p95/p99 latency, action completion rate, dropped frames, cold vs warm cache, retrieval I/O. Record exact measurement hardware and whether observed or estimated. Never infer FLOPs reduction directly from token reduction.

## Budgeting: no invented GPU-hours

Pilot jobs measure wallclock/iteration, peak VRAM, inference Hz and episodes/hour. With `n_models × n_seeds × training_steps`, compute estimated GPU-hours from measured pilot; label forecasts and uncertainty. Cut failing methods early through fixed gate tests, not by silently starving baselines of training steps.

## Failure recovery

- On preemption: atomic checkpoint with optimizer/scheduler/RNG/data cursor/state, then write `interrupted` status.
- On NaN: halt and save minidump, not silently restart with altered settings.
- On OOM: record and propose new experiment ID; do not overwrite old config.
- On incomplete distributed worker: mark run invalid; no partial unqualified success reports.
- On lost robot connection: external controller safe stop; model only records diagnostic.

## Data ethics and licenses

Review benchmark dataset and pretrained checkpoint terms, model weights license, human subject consent where relevant, retention and sharing. Do not upload private lab data to public tracker by default. When redistributing outputs, only share permitted metadata/examples.
