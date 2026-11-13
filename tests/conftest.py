# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import yaml

from ucvla.adapters import SODHAdapter
from ucvla.sodh import SODHConfig, generate_suite
from ucvla.splits import assign_splits

ROOT = Path(__file__).resolve().parents[1]


def small_config(**over):
    """Small but complete suite: every family x regime non-empty, OOD strictly longer."""
    fam = {"n_objects_id": [3, 4], "n_objects_ood": [4, 5], "horizon_id": [3, 6],
           "horizon_ood": [9, 12], "pairs_in_dist": 16, "pairs_ood": 4, "pairs_heldout": 4}
    count = {**fam, "n_objects_id": [2, 2], "n_objects_ood": [2, 2], "horizon_id": [2, 4],
             "horizon_ood": [6, 8]}
    raw = {"master_seed": 7, "n_identity_mappings": 8, "heldout_mappings": [7],
           "split_fractions": [0.7, 0.15, 0.15],
           "families": {"object_correspondence": fam, "process_count": count,
                        "task_interleave": fam, "irrelevant_history": {**fam, "n_objects_id": [2, 2],
                                                                       "n_objects_ood": [2, 2]}}}
    raw.update(over)
    return SODHConfig.from_dict(raw)


@pytest.fixture(scope="session")
def cfg():
    return small_config()


@pytest.fixture(scope="session")
def suite(cfg):
    return generate_suite(cfg)


@pytest.fixture(scope="session")
def assignment(suite, cfg):
    return assign_splits(suite.pairs, cfg)


@pytest.fixture(scope="session")
def adapter(suite, assignment):
    return SODHAdapter(suite.episodes, suite.pairs, assignment)


@pytest.fixture(scope="session")
def frozen_config():
    return SODHConfig.from_dict(
        yaml.safe_load((ROOT / "configs/sodh/sodh_v1.yaml").read_text(encoding="utf-8")))


def small_config2(**over):
    from ucvla.sodh2 import SODH2Config
    fam = {"n_id": [3, 4], "horizon_id": [8, 12], "horizon_ood_length": [18, 22],
           "pairs_in_dist": 30, "pairs_ood_length": 6, "pairs_ood_structure": 6,
           "pairs_heldout": 0}
    ev = {**fam, "n_id": [2, 2], "horizon_id": [10, 14], "horizon_ood_length": [20, 24]}
    rs = {**fam, "horizon_id": [14, 18], "horizon_ood_length": [24, 28], "pairs_heldout": 6}
    raw = {"master_seed": 5, "n_identity_mappings": 8, "heldout_mappings": [7],
           "split_fractions": [0.5, 0.15, 0.35], "noise_levels": [0.0, 0.1], "margin_gate": 0.3,
           "families": {"track": fam, "events": ev, "resume": rs,
                        "control": {**fam, "pairs_in_dist": 12, "pairs_ood_length": 4,
                                    "pairs_ood_structure": 4}}}
    raw.update(over)
    return SODH2Config.from_dict(raw)


@pytest.fixture(scope="session")
def cfg2():
    return small_config2()


@pytest.fixture(scope="session")
def suite2(cfg2):
    from ucvla.sodh2 import generate_suite2
    return generate_suite2(cfg2)


@pytest.fixture(scope="session")
def assignment2(suite2, cfg2):
    from ucvla.sodh2 import assign_splits2
    return assign_splits2(suite2.pairs, cfg2)


@pytest.fixture(scope="session")
def adapter2(suite2, assignment2):
    return SODHAdapter(suite2.episodes, suite2.pairs, assignment2)
