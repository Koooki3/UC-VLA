#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent-run reproducibility: N fresh processes, different PYTHONHASHSEED, same hashes."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--script", default="run_m2_eval.py")
    ap.add_argument("--run-seed", type=int, default=0)
    args = ap.parse_args()
    outs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.n):
            env = {**os.environ, "PYTHONHASHSEED": str(i * 7919 + 1)}
            p = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / args.script), "--run-seed",
                 str(args.run_seed), "--out", str(Path(tmp) / f"r{i}")],
                capture_output=True, text=True, env=env, check=False)
            if p.returncode:
                sys.exit(f"run {i} failed:\n{p.stderr}")
            outs.append(json.loads(p.stdout.strip().splitlines()[-1]))
    keys = ("run_id", "results_sha256", "metrics_sha256")
    same = all(all(o[k] == outs[0][k] for k in keys) for o in outs)
    print(json.dumps({"runs": args.n, "identical": same, **{k: outs[0][k] for k in keys}}))
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
