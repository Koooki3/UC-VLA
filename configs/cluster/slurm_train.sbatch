#!/usr/bin/env bash
# TEMPLATE ONLY. Needs local site review and explicit sbatch by an authorized operator.
#SBATCH --job-name=ucvla-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/ucvla-%j.out
#SBATCH --error=logs/ucvla-%j.err
# Partition/account/GPU/time/memory intentionally UNSET. Set via validated site template.
set -euo pipefail
: "${UCVLA_DATA_ROOT:?Export reviewed dataset root}"
: "${UCVLA_RUN_ROOT:?Export reviewed result root}"
: "${UCVLA_TRAIN_COMMAND:?Set reviewed train command, not a placeholder}"
: "${UCVLA_CLUSTER_PARTITION:?Set real partition through sbatch CLI or site config}"
: "${UCVLA_CLUSTER_ACCOUNT:?Set real account through sbatch CLI or site config}"
mkdir -p "$UCVLA_RUN_ROOT"
python3 scripts/doctor.py
printf 'WARNING: run is experimental; approved train command required.\n'
# Site operators must review UCVLA_TRAIN_COMMAND to prevent arbitrary execution.
exec bash -lc "$UCVLA_TRAIN_COMMAND"
