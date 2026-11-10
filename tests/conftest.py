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
