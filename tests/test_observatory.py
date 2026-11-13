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
    r.execution(60, "p1", 1, True, allow_stale=True)
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
    r.observation(4, H, 1)
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
    with pytest.raises(TraceError, match="unknown or invalidated"):
        r.execution(8, "ghost", 0, True)


def test_stale_plan_cannot_be_executed_by_default_and_invalidation_works(tmp_path):
    r = TraceRecorder(tmp_path / "s.jsonl")
    r.observation(1, H, 3)
    r.memory_op(2, "write", 0, (H,), 8, 3, 0.1)
    r.plan(3, "p", 1, 4)
    r.memory_op(4, "update", 1, (H,), 8, 4, 0.1)
    with pytest.raises(TraceError, match="stale plan"):
        r.execution(5, "p", 0, True)
    r.invalidate_plan(5, "p", "context v2")
    with pytest.raises(TraceError, match="invalidated"):
        r.execution(6, "p", 0, True)
    r.plan(7, "p2", 2, 4)
    r.execution(8, "p2", 0, True)


def test_nested_forbidden_keys_and_evidence_rules(tmp_path):
    r = TraceRecorder(tmp_path / "e.jsonl")
    with pytest.raises(TraceError, match="forbidden"):
        r.intervention(1, "x", {"detail": {"inner": [{"oracle_state": 1}]}})
    with pytest.raises(TraceError, match="never observed"):
        r.memory_op(2, "write", 0, (H,), 8, 3, 0.1)
    r.observation(5, H, 1)
    r.memory_op(6, "write", 0, (H,), 8, 3, 0.1)


def test_existing_trace_is_never_overwritten_and_resume_is_verified(tmp_path):
    p = tmp_path / "t.jsonl"
    r = TraceRecorder(p)
    r.observation(1, H, 1)
    r.memory_op(2, "write", 0, (H,), 8, 3, 0.1)
    with pytest.raises(TraceError, match="refusing to overwrite"):
        TraceRecorder(p)
    r2 = TraceRecorder(p, resume=True)
    r2.plan(3, "p", 1, 2)
    r2.execution(4, "p", 0, True)
    assert [e.kind for e in replay(p)] == ["observation", "memory_op", "plan", "execution"]
    p.write_text(p.read_text().replace('"tokens":1', '"tokens":9'))
    with pytest.raises(TraceError):
        TraceRecorder(p, resume=True)
