#!/usr/bin/env python3
"""Generate the deterministic synthetic SODH dataset, check splits, run G0. CPU only."""

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import ADAPTER_STATUS, SODHAdapter
from ucvla.audit import g0_report
from ucvla.canonical import canonical_json
from ucvla.dataset_io import source_tree_sha256, write_dataset
from ucvla.runenv import env_info, git_info
from ucvla.sodh import SODHConfig, generate_suite
from ucvla.splits import assign_splits, build_split_manifest, check_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/sodh/sodh_v1.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/sodh_v1")
    args = ap.parse_args()

    config = SODHConfig.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    if config.leak is not None:
        raise SystemExit("refusing to write a dataset with a planted leak (tests only)")
    suite = generate_suite(config)
    assignment = assign_splits(suite.pairs, config)
    errors = [f for f in check_splits(suite.episodes, suite.pairs, assignment) if f.level == "error"]
    if errors:
        raise SystemExit(f"split check failed: {errors}")
    split_manifest = build_split_manifest(config, suite.pairs, suite.episodes, assignment)
    manifest = write_dataset(
        args.out, suite.episodes, suite.pairs, assignment, split_manifest,
        {"config_sha256": config.sha256(), "generator_version": config.generator_version,
         "split_manifest_id": split_manifest["split_manifest_id"],
         "rejected_duplicate_scripts": suite.rejected_duplicate_scripts},
    )
    # Derived, overwritable: depends on the code tree, so it is NOT part of the immutable data.
    (args.out / "provenance.json").write_text(canonical_json({
        "code_tree_sha256": source_tree_sha256(ROOT),
        **git_info(ROOT), "env": env_info(),
        "config_path": args.config.relative_to(ROOT).as_posix() if args.config.is_relative_to(ROOT)
        else str(args.config),
        "adapters": dict(ADAPTER_STATUS),
    }) + "\n", encoding="ascii")
    adapter = SODHAdapter(suite.episodes, suite.pairs, assignment)
    report = g0_report(adapter, suite.pairs, suite.episodes, assignment)
    (args.out / "g0_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}  split={split_manifest['split_manifest_id']}  "
          f"episodes={manifest['counts']['episodes']}  G0={'PASS' if report['passed'] else 'FAIL'}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
