# SPDX-License-Identifier: Apache-2.0
"""Freeze guard for SODH v2: config, data and split manifest may never change silently."""

import hashlib
import json
from pathlib import Path

import yaml

from ucvla.canonical import canonical_json, sha256_hex
from ucvla.sodh2 import SODH2Config, assign_splits2, generate_suite2
from ucvla.splits import build_split_manifest, manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/synthetic/sodh_v2"


def test_v2_regeneration_reproduces_frozen_hashes():
    f = json.loads((D / "FROZEN.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((ROOT / f["config_file"]).read_bytes()).hexdigest() == f["config_file_sha256"]
    cfg = SODH2Config.from_dict(yaml.safe_load((ROOT / f["config_file"]).read_text(encoding="utf-8")))
    assert cfg.sha256() == f["config_sha256"]
    suite = generate_suite2(cfg)
    asg = assign_splits2(suite.pairs, cfg)
    eps = "".join(canonical_json(e.to_dict()) + "\n" for e in suite.episodes).encode("ascii")
    assert sha256_hex(eps) == f["data_files_sha256"]["episodes.jsonl"]
    pairs = "".join(canonical_json(p) + "\n" for p in suite.pairs).encode("ascii")
    assert sha256_hex(pairs) == f["data_files_sha256"]["pairs.jsonl"]
    sm = build_split_manifest(cfg, suite.pairs, suite.episodes, asg)  # type: ignore[arg-type]
    assert sm["split_manifest_id"] == f["split_manifest_id"]
    assert sha256_hex(manifest_bytes(sm)) == f["data_files_sha256"]["split_manifest.json"]


def test_v1_and_v2_are_separate_versions():
    v1 = json.loads((ROOT / "results/synthetic/sodh_v1/FROZEN.json").read_text())
    v2 = json.loads((D / "FROZEN.json").read_text())
    assert v1["dataset"] == "sodh_v1" and v2["dataset"] == "sodh_v2"
    assert v1["split_manifest_id"] != v2["split_manifest_id"]
