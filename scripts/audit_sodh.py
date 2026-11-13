#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently re-verify a written SODH dataset: hashes, split manifest, G0. Exit 1 on failure."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.audit import g0_report
from ucvla.dataset_io import read_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, nargs="?", default=ROOT / "results/synthetic/sodh_v1")
    args = ap.parse_args()
    episodes, pairs, assignment = read_dataset(args.data_dir)  # raises on any hash mismatch
    stored = json.loads((args.data_dir / "split_manifest.json").read_text(encoding="ascii"))
    members = {s: sorted(p for p, v in assignment.items() if v == s) for s in stored["pairs"]}
    if members != stored["pairs"]:
        raise SystemExit("INVALID: assignment.json disagrees with split_manifest.json")
    report = g0_report(SODHAdapter(episodes, pairs, assignment), pairs, episodes, assignment)
    print(f"G0 {'PASS' if report['passed'] else 'FAIL'}  ({len(episodes)} episodes, "
          f"{len(pairs)} pairs, {stored['split_manifest_id']})")
    for f in report["findings"]:
        print(f"  [{f['level']}] {f['code']}: {f['message']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
