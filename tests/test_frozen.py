# SPDX-License-Identifier: Apache-2.0
"""Freeze guard: SODH v1 config, data and split manifest may never change silently."""

import hashlib
import json
from pathlib import Path

import yaml

from ucvla.canonical import canonical_json, sha256_hex
from ucvla.sodh import SODHConfig, generate_suite
from ucvla.splits import assign_splits, build_split_manifest, manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/synthetic/sodh_v1"


def frozen():
    return json.loads((D / "FROZEN.json").read_text(encoding="utf-8"))


def test_config_file_unchanged():
    f = frozen()
    assert hashlib.sha256((ROOT / f["config_file"]).read_bytes()).hexdigest() == f["config_file_sha256"]


def test_regeneration_reproduces_frozen_hashes():
    f = frozen()
    cfg = SODHConfig.from_dict(yaml.safe_load((ROOT / f["config_file"]).read_text(encoding="utf-8")))
    assert cfg.sha256() == f["config_sha256"]
    suite = generate_suite(cfg)
    asg = assign_splits(suite.pairs, cfg)
    episodes = "".join(canonical_json(e.to_dict()) + "\n" for e in suite.episodes).encode("ascii")
    assert sha256_hex(episodes) == f["data_files_sha256"]["episodes.jsonl"]
    pairs = "".join(canonical_json(p) + "\n" for p in suite.pairs).encode("ascii")
    assert sha256_hex(pairs) == f["data_files_sha256"]["pairs.jsonl"]
    sm = build_split_manifest(cfg, suite.pairs, suite.episodes, asg)
    assert sm["split_manifest_id"] == f["split_manifest_id"]
    assert sha256_hex(manifest_bytes(sm)) == f["data_files_sha256"]["split_manifest.json"]


def test_committed_manifest_matches_freeze():
    f = frozen()
    for name in ("split_manifest.json", "assignment.json", "pairs.jsonl"):
        assert sha256_hex((D / name).read_bytes()) == f["data_files_sha256"][name]
