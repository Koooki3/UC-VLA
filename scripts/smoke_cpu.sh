#!/usr/bin/env bash
# CPU-only offline verification; never executes a policy, simulator or robot.
# PYTHON may point at a supported interpreter (3.11-3.13), e.g. PYTHON=.venv/bin/python
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m pytest -q
"$PY" scripts/dry_run.py
"$PY" scripts/doctor.py
"$PY" scripts/generate_sodh.py
"$PY" scripts/audit_sodh.py
"$PY" scripts/generate_sodh2.py
"$PY" scripts/repo_scan.py --strict
"$PY" scripts/package_manifest.py --check
"$PY" -m compileall -q src scripts tests
bash -n scripts/cluster_preflight.sh configs/cluster/slurm_train.sbatch
