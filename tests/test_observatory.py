# SPDX-License-Identifier: Apache-2.0
import pytest

from ucvla.canonical import sha256_hex
from ucvla.observatory import (
    TraceError,
    TraceRecorder,
    replay,
    stale_plans,
    version_timeline,
)

H = sha256_hex(b"x")


def make(tmp_path):
    r = TraceRecorder(tmp_path / "t.jsonl")
    r.observation(10, H, 6)
    r.memory_op(20, "write", 0, (H,), 16, 6, 0.4)
    r.plan(30, "p1", 1, 8)
    r.execution(40, "p1", 0, True)
    r.memory_op(50, "update", 1, (H,), 16, 8, 0.5)
    r.execution(60, "p1", 1, True)
    r.intervention(70, "swap_history", {"pair": "a"})
    return r


def test_replay_and_timeline(tmp_path):
    make(tmp_path)
    assert [e.kind for e in replay(tmp_path / "t.jsonl")][:3] == ["observation", "memory_op", "plan"]
    assert version_timeline(tmp_path / "t.jsonl") == [(20, 1, "write"), (50, 2, "update")]


def test_stale_plan_is_flagged_in_time_order_only(tmp_path):
    make(tmp_path)
    assert stale_plans(tmp_path / "t.jsonl") == ["p1"]  # executed after version 1 was superseded


def test_tamper_and_reorder_detected(tmp_path):
    make(tmp_path)
    p = tmp_path / "t.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n")
    with pytest.raises(TraceError):
        list(replay(p))
    p.write_text(p.read_text().replace('"tokens":6', '"tokens":7'))
    with pytest.raises(TraceError):
        list(replay(p))


@pytest.mark.parametrize("bad", ["target", "privileged_state", "oracle_action", "Label"])
def test_privileged_or_label_keys_rejected(tmp_path, bad):
    r = TraceRecorder(tmp_path / "t.jsonl")
    with pytest.raises(TraceError, match="forbidden"):
        r.intervention(1, "x", {bad: 1})


def test_rules_enforced(tmp_path):
    r = TraceRecorder(tmp_path / "t.jsonl")
    r.memory_op(5, "write", 0, (H,), 4, 4, 0.1)
    with pytest.raises(TraceError, match="stale"):
        r.memory_op(6, "update", 0, (H,), 4, 4, 0.1)
    with pytest.raises(TraceError, match="over budget"):
        r.memory_op(6, "update", 1, (H,), 4, 5, 0.1)
    with pytest.raises(TraceError, match="time"):
        r.observation(1, H, 1)
    with pytest.raises(TraceError, match="sha256"):
        r.observation(7, "nothex", 1)
    with pytest.raises(TraceError, match="future"):
        r.plan(8, "p", 9, 1)
    with pytest.raises(TraceError, match="unknown plan"):
        r.execution(8, "ghost", 0, True)
