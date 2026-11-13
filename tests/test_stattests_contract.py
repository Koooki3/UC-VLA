# SPDX-License-Identifier: Apache-2.0
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from ucvla.benchmark_contract import (
    HDF5_IMPORTER,
    LEROBOT_IMPORTER,
    SODHSource,
    SourceManifest,
    check_source_conformance,
)
from ucvla.stattests import holm, sign_test

ROOT = Path(__file__).resolve().parents[1]


def test_sign_test_exact_values():
    assert sign_test([True] * 10, [False] * 10)["p"] == pytest.approx(2 / 1024)
    assert sign_test([True, False] * 5, [False, True] * 5)["p"] == 1.0
    assert sign_test([True] * 4, [True] * 4)["p"] == 1.0
    with pytest.raises(ValueError):
        sign_test([True], [])


def test_holm_matches_hand_computation():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": pytest.approx(0.03), "c": pytest.approx(0.06), "b": pytest.approx(0.06)}
    assert all(0 <= v <= 1 for v in holm({"x": 0.9, "y": 0.8}).values())


def test_sodh_source_conforms_and_negative_controls_fail(adapter):
    src = SODHSource(adapter, "sodh-split-test0001")
    assert check_source_conformance(src) == []
    bad = SODHSource(adapter, "main")
    assert any("moving branch" in v for v in check_source_conformance(bad))
    unverified = SODHSource(adapter, "sodh-split-test0001")
    unverified.manifest = dataclasses.replace(unverified.manifest, license_status="unverified")
    assert any("not cleared" in v for v in check_source_conformance(unverified))


def test_conformance_catches_cross_split_duplicates_and_executed_origin(adapter):
    class Dup(SODHSource):
        def episode_ids(self, split):
            ids = super().episode_ids(split)
            return ids + super().episode_ids("train")[:1] if split == "val" else ids

    v = check_source_conformance(Dup(adapter, "sodh-split-test0001"))
    assert any("appears in splits" in x for x in v)

    class Mislabelled(SODHSource):
        def load(self, eid):
            ep = super().load(eid)
            return dataclasses.replace(ep, provenance={**ep.provenance,
                                                       "executed_origin": "bogus"})

    assert any("executed_origin" in x for x in check_source_conformance(
        Mislabelled(adapter, "sodh-split-test0001"), max_episodes=3))


def test_real_importers_fail_loudly_not_silently():
    for imp in (HDF5_IMPORTER, LEROBOT_IMPORTER):
        with pytest.raises(NotImplementedError, match="not implemented"):
            imp("/nonexistent")
    assert SourceManifest("x", "abcdef0", "hdf5", "unverified", "?", "?").license_status


def test_independent_processes_reproduce_identical_results(tmp_path):
    """Fresh interpreters, different PYTHONHASHSEED: same run_id and result hashes."""
    if not (ROOT / "results/synthetic/sodh_v1/episodes.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_sodh.py")], check=True,
                       capture_output=True)
    p = subprocess.run([sys.executable, str(ROOT / "scripts/repro_check.py"), "-n", "2"],
                       capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert '"identical": true' in p.stdout


def test_v2_eval_is_reproducible_across_fresh_processes():
    """Same property for the v2 evaluation (different hash seeds, fresh interpreters)."""
    if not (ROOT / "results/synthetic/sodh_v2/episodes.jsonl").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_sodh2.py")], check=True,
                       capture_output=True)
    p = subprocess.run([sys.executable, str(ROOT / "scripts/repro_check.py"), "-n", "2",
                        "--script", "run_v2_eval.py"], capture_output=True, text=True, check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert '"identical": true' in p.stdout
