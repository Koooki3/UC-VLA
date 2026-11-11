#!/usr/bin/env python3
"""Generate SODH v2 (docs/19), run the v2 gate, and write-once freeze it. CPU only, synthetic."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.audit2 import AUDIT_VERSION, g0_v2_report
from ucvla.canonical import canonical_json
from ucvla.dataset_io import source_tree_sha256, write_dataset
from ucvla.runenv import env_info, git_info
from ucvla.sodh2 import SODH2Config, assign_splits2, generate_suite2
from ucvla.splits import build_split_manifest, write_once


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/sodh/sodh_v2.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/sodh_v2")
    args = ap.parse_args()
    config = SODH2Config.from_dict(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    suite = generate_suite2(config)
    assignment = assign_splits2(suite.pairs, config)
    adapter = SODHAdapter(suite.episodes, suite.pairs, assignment)
    report = g0_v2_report(adapter, suite.pairs, suite.episodes, assignment)
    if not report["passed"]:
        print(json.dumps(report["findings"][:10], indent=1))
        raise SystemExit("G0-v2 FAILED: not writing a dataset (failures are kept, not repaired)")
    sm = build_split_manifest(config, suite.pairs, suite.episodes, assignment)  # type: ignore[arg-type]
    manifest = write_dataset(args.out, suite.episodes, suite.pairs, assignment, sm,
                             {"config_sha256": config.sha256(),
                              "generator_version": config.generator_version,
                              "split_manifest_id": sm["split_manifest_id"],
                              "rejected": suite.rejected})
    rep_path = args.out / f"g0_report.{AUDIT_VERSION.replace('/', '-')}.json"
    rep_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "provenance.json").write_text(canonical_json({
        "code_tree_sha256": source_tree_sha256(ROOT), **git_info(ROOT), "env": env_info()}) + "\n",
        encoding="ascii")
    frozen = {
        "frozen_on": "2026-11-12", "dataset": "sodh_v2", "mode": "synthetic",
        "rule": "Test splits, generator config and v2 data are immutable. A change needs a new "
                "id (sodh_v3); results must never influence these files.",
        "spec": "docs/19_SODH_V2_SPEC.md", "config_file": "configs/sodh/sodh_v2.yaml",
        "config_file_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "config_sha256": config.sha256(), "split_manifest_id": sm["split_manifest_id"],
        "data_files_sha256": manifest["files"], "counts": manifest["counts"],
        "g0_v2": {"audit_version": AUDIT_VERSION, "passed": True,
                  "validity": report["validity"]},
    }
    write_once(args.out / "FROZEN.json", (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode())
    print(f"wrote {args.out} split={sm['split_manifest_id']} episodes={manifest['counts']['episodes']} "
          f"G0-v2=PASS rejected={suite.rejected}")


if __name__ == "__main__":
    main()
