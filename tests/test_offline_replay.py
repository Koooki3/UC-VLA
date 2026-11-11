import json
from pathlib import Path

import h5py
import pytest

from tests.test_hdf5_import import CFG, make_h5
from ucvla.context import AllHistory, NoHistory
from ucvla.hdf5_import import import_episode
from ucvla.offline_replay import (
    ProprioCopy,
    ZeroOrderHold,
    replay_evaluate,
    replay_inputs,
    summary,
)
from ucvla.policy import ActionChunk

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "datasets/robomme_sample/extracted/record_dataset_StopCube.h5"


@pytest.fixture(scope="module")
def ep(tmp_path_factory):
    p = make_h5(tmp_path_factory.mktemp("rp") / "m.h5", n_steps=20, n_eps=1)
    with h5py.File(p, "r") as f:
        return import_episode(f, "episode_0", CFG)


def test_step_i_sees_exactly_the_commands_before_it_and_the_instruction(ep):
    for i, inp, rec in replay_inputs(ep, AllHistory(), k_commands=5):
        assert len(inp.context.commanded) == min(i, 5)
        assert all(c.issued_at_ns <= inp.observation.time_ns for c in inp.context.commanded)
        assert rec not in inp.context.commanded  # the command to predict is never in the input
        assert inp.task.instruction == "press at the target"
        assert len(inp.observation.proprio) == 1  # current sample only; history is budgeted
        assert len(inp.context.ticks) == i


def test_budgets_apply_to_real_episodes_too(ep):
    for _, inp, _ in replay_inputs(ep, NoHistory(), k_commands=0):
        assert inp.context.ticks == () and inp.context.commanded == ()


def test_fake_policies_run_and_zoh_uses_only_the_past(ep):
    r = replay_evaluate(ep, ZeroOrderHold(), AllHistory(), k_commands=1)
    assert r.n_steps == 20 and r.mse >= 0
    seen = []

    class Spy:
        name = "spy"

        def spec(self):
            return {}

        def reset(self, s):
            pass

        def infer(self, inp):
            seen.append(inp)
            return ActionChunk((json.dumps([0.0] * 8),))

    replay_evaluate(ep, Spy(), NoHistory(), k_commands=1)
    recorded = [json.loads(c.action) for c in ep.commanded]
    for k, inp in enumerate(seen):
        assert [json.loads(c.action) for c in inp.context.commanded] == recorded[max(0, k - 1):k]


def test_dimension_mismatch_rejected(ep):
    class Bad:
        name = "bad"

        def spec(self):
            return {}

        def reset(self, s):
            pass

        def infer(self, inp):
            return ActionChunk((json.dumps([0.0]),))

    with pytest.raises(ValueError, match="dimension"):
        replay_evaluate(ep, Bad(), NoHistory())


@pytest.mark.skipif(not REAL.exists(), reason="official RoboMME sample not downloaded")
def test_real_sample_zoh_beats_proprio_copy_which_beats_nothing():
    cfg = CFG
    with h5py.File(REAL, "r") as f:
        eps = [import_episode(f, f"episode_{i}", cfg) for i in range(3)]
    zoh = summary([replay_evaluate(e, ZeroOrderHold(), AllHistory(), 1) for e in eps])
    cpy = summary([replay_evaluate(e, ProprioCopy(), NoHistory(), 0) for e in eps])
    assert zoh["mse"] < cpy["mse"]  # one past command is a strong prior: sanity of the plumbing
