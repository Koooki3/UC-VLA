# SPDX-License-Identifier: Apache-2.0
import pytest

from ucvla.contracts import (
    ContextSnapshot,
    EvidenceKind,
    EvidenceRef,
    Fact,
    Patch,
    PlanGuardState,
    PlanRef,
    hash_payload,
)
from ucvla.memory import ContextStore
from ucvla.toy import paired_episode, run_memoryless, run_oracle_memory


def ref(eid="e1", t=1, episode="episode-A"):
    return EvidenceRef(episode, eid, t, EvidenceKind.OBSERVED, hash_payload(eid.encode()))


def test_no_future_leakage_and_atomicity():
    store = ContextStore("episode-A")
    with pytest.raises(ValueError, match="future"):
        store.commit(Patch("episode-A", 0, 2, "x", Fact("1", EvidenceKind.OBSERVED, ("e1",)), (ref(t=9),)))
    assert store.current.version == 0
    assert len(store.ledger) == 0


def test_no_out_of_order_commit():
    store = ContextStore("episode-A")
    store.commit(Patch("episode-A", 0, 5, "x", Fact("1", EvidenceKind.OBSERVED, ("e1",)), (ref(t=5),)))
    with pytest.raises(ValueError, match="out-of-order"):
        store.commit(Patch("episode-A", 1, 4, "y", Fact("2", EvidenceKind.INFERRED)))


def test_stale_version_rejected():
    store = ContextStore("episode-A")
    store.commit(Patch("episode-A", 0, 1, "x", Fact("1", EvidenceKind.INFERRED)))
    with pytest.raises(ValueError, match="stale"):
        store.commit(Patch("episode-A", 0, 2, "y", Fact("2", EvidenceKind.INFERRED)))


def test_evidence_kept_after_eviction():
    store = ContextStore("episode-A", max_slots=1)
    store.commit(Patch("episode-A", 0, 1, "x", Fact("1", EvidenceKind.OBSERVED, ("e1",)), (ref(),)))
    store.evict("x", 2)
    assert "x" not in store.current.facts
    assert store.ledger.exists("e1")
    assert "e1" in store.current.evidence_ids


def test_budget_and_pinning():
    store = ContextStore("episode-A", max_slots=1)
    store.commit(Patch("episode-A", 0, 1, "target", Fact("left", EvidenceKind.INFERRED), pin=True))
    with pytest.raises(ValueError, match="pinned"):
        store.evict("target", 2)
    with pytest.raises(ValueError, match="budget"):
        store.commit(Patch("episode-A", 1, 2, "other", Fact("v", EvidenceKind.INFERRED)))
    assert store.current.version == 1


def test_missing_provenance_rejected():
    with pytest.raises(ValueError, match="cite evidence"):
        Fact("true", EvidenceKind.MEASURED)
    store = ContextStore("episode-A")
    with pytest.raises(ValueError, match="missing evidence"):
        store.commit(Patch("episode-A", 0, 2, "x", Fact("a", EvidenceKind.OBSERVED, ("fake",))))


def test_cross_episode_rejected():
    store = ContextStore("episode-A")
    with pytest.raises(ValueError, match="cross-episode"):
        store.commit(Patch("episode-B", 0, 1, "x", Fact("v", EvidenceKind.INFERRED)))


def test_plan_version_guard():
    plan = PlanRef("episode-A", 2, 0, "cup-left", 0, 123)
    valid = PlanGuardState("episode-A", 2, 0, "cup-left", 0)
    assert valid.permits(plan)
    assert not PlanGuardState("episode-A", 3, 0, "cup-left", 0).permits(plan)
    assert not PlanGuardState("episode-A", 2, 0, "cup-right", 0).permits(plan)
    assert not PlanGuardState("episode-A", 2, 1, "cup-left", 0).permits(plan)
    assert not PlanGuardState("episode-A", 2, 0, "cup-left", 1).permits(plan)


def test_same_final_observation_different_history():
    _, f_left = paired_episode("a", "left")
    _, f_right = paired_episode("b", "right")
    assert f_left.visible == f_right.visible
    assert run_oracle_memory("a", "left").success
    assert run_oracle_memory("b", "right").success
    assert run_memoryless("a", "left").success
    assert not run_memoryless("b", "right").success


def test_snapshot_invalid_pin():
    with pytest.raises(ValueError, match="pinned"):
        ContextSnapshot("e", 0, 0, pinned=frozenset({"missing"}))
