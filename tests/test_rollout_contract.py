# SPDX-License-Identifier: Apache-2.0
from ucvla.rollout_contract import FakeKeyEnv, paired, rollout


def remember(ctx, obs):
    key = next((h["key"] for h in ctx if h["key"] != "none"), "0")
    return key


def test_same_seed_same_arm_is_deterministic():
    e = FakeKeyEnv()
    assert rollout(e, remember, 5, None) == rollout(e, remember, 5, None)


def test_edit_changes_what_the_policy_sees_not_the_world():
    e = FakeKeyEnv()
    drop_key = lambda ctx: [h for h in ctx if h["key"] == "none"]
    out = paired(e, remember, drop_key, range(40))
    assert all(o.ret_keep == 1.0 for o in out)  # world/label identical, policy with memory always right
    assert 0.2 < sum(o.ret_edit for o in out) / len(out) < 0.8  # without the key it guesses
    assert all(o.delta >= 0 for o in out)
