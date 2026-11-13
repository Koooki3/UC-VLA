# SPDX-License-Identifier: Apache-2.0
"""Minimal deterministic, fully auditable context writer.

Not learned, not a production VLA memory. Only CPU contracts and tests.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from ucvla.contracts import ContextSnapshot, EvidenceRef, Patch


@dataclass
class EvidenceLedger:
    """Append-only in memory; production implementation needs durable WAL + ACLs."""

    episode_id: str
    _entries: dict[str, EvidenceRef] = field(default_factory=dict)

    def append(self, evidence: EvidenceRef) -> None:
        if evidence.episode_id != self.episode_id:
            raise ValueError("cross-episode evidence forbidden")
        if evidence.event_id in self._entries:
            raise ValueError("duplicate evidence id; immutable ledger")
        self._entries[evidence.event_id] = evidence

    def exists(self, event_id: str) -> bool:
        return event_id in self._entries

    def get(self, event_id: str) -> EvidenceRef:
        return self._entries[event_id]

    def __len__(self) -> int:
        return len(self._entries)


class ContextStore:
    """Deterministic, version-checked context store with explicit budget.

    The caller supplies patches. No inference occurs here. Evidence is never erased.
    """

    def __init__(self, episode_id: str, max_slots: int = 4) -> None:
        if max_slots <= 0:
            raise ValueError("max_slots must be positive")
        self.max_slots = max_slots
        self.ledger = EvidenceLedger(episode_id=episode_id)
        self.current = ContextSnapshot(episode_id=episode_id, version=0, cutoff_ns=0)
        self.audit: list[Patch] = []

    def commit(self, patch: Patch) -> ContextSnapshot:
        old = self.current
        if patch.episode_id != old.episode_id:
            raise ValueError("patch cross-episode")
        if patch.base_version != old.version:
            raise ValueError("stale version")
        if patch.cutoff_ns < old.cutoff_ns:
            raise ValueError("out-of-order cutoff")
        if not patch.key:
            raise ValueError("fact key required")
        # Validate before any mutation: transactions must be atomic on failure.
        staged = {e.event_id: e for e in patch.evidence_refs}
        if len(staged) != len(patch.evidence_refs):
            raise ValueError("duplicate evidence inside patch")
        for eid, evidence in staged.items():
            if evidence.episode_id != old.episode_id or evidence.time_ns > patch.cutoff_ns:
                raise ValueError("future or cross-episode evidence")
            if self.ledger.exists(eid):
                raise ValueError("source evidence cannot be overwritten")
        refs = set(old.evidence_ids) | set(staged)
        if patch.value is not None and not set(patch.value.refs).issubset(refs):
            raise ValueError("fact references missing evidence")
        facts = dict(old.facts)
        pinned = set(old.pinned)
        if patch.value is None:
            if patch.key in pinned:
                raise ValueError("cannot evict pinned fact")
            facts.pop(patch.key, None)
        else:
            facts[patch.key] = patch.value
            if patch.pin:
                pinned.add(patch.key)
        if len(facts) > self.max_slots:
            raise ValueError("context memory over budget; explicit eviction required")
        # Commit the validated event records then new state.
        for evidence in patch.evidence_refs:
            self.ledger.append(evidence)
        updated = ContextSnapshot(
            episode_id=old.episode_id,
            version=old.version + 1,
            cutoff_ns=patch.cutoff_ns,
            facts=facts,
            pinned=frozenset(pinned),
            evidence_ids=frozenset(refs),
        )
        self.current = updated
        self.audit.append(patch)
        return updated

    def evict(self, key: str, cutoff_ns: int) -> ContextSnapshot:
        patch = Patch(
            episode_id=self.current.episode_id,
            base_version=self.current.version,
            cutoff_ns=cutoff_ns,
            key=key,
            value=None,
        )
        return self.commit(patch)

    def validate(self, predicate: Callable[[ContextSnapshot], bool]) -> bool:
        return bool(predicate(self.current))
