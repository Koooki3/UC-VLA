#!/usr/bin/env bash
set -euo pipefail
printf 'UC-VLA: read-only SLURM preflight (no job submission)\n'
command -v python3 >/dev/null || { echo 'Missing Python3' >&2; exit 1; }
python3 scripts/doctor.py
if command -v sbatch >/dev/null; then echo 'sbatch detected: do not submit without operator approval'; else echo 'SLURM absent: use CPU smoke only'; fi
for var in UCVLA_DATA_ROOT UCVLA_RUN_ROOT UCVLA_CLUSTER_PARTITION UCVLA_CLUSTER_ACCOUNT; do
    if [[ -n "${!var:-}" ]]; then echo "$var=set"; else echo "$var=NOT SET"; fi
done
