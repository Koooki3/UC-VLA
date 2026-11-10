"""Pure-Python contracts; nothing in this module commands a physical robot."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256


class EvidenceKind(str, Enum):
    OBSERVED = "observed"
    MEASURED = "measured"
    COMMAND = "commanded"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True)
class EvidenceRef:
    episode_id: str
    event_id: str
    time_ns: int
    kind: EvidenceKind
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.episode_id or not self.event_id:
            raise ValueError("episode and event identifiers required")
        if self.time_ns < 0:
            raise ValueError("negative timestamp")
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.source_sha256
        ):
            raise ValueError("evidence source_sha256 must be lowercase 64-char hex")


def hash_payload(data: bytes) -> str:
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class Observation:
    episode_id: str
    time_ns: int
    visible: Mapping[str, str]
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        if self.evidence.episode_id != self.episode_id:
            raise ValueError("observation/evidence episode mismatch")
        if self.evidence.time_ns != self.time_ns:
            raise ValueError("observation/evidence time mismatch")
        if self.evidence.kind not in (EvidenceKind.OBSERVED, EvidenceKind.MEASURED):
            raise ValueError("observation requires sensor evidence")


@dataclass(frozen=True)
class ExecutedAction:
    episode_id: str
    command_id: str
    issued_at_ns: int
    executed_at_ns: int
    verified: bool

    def __post_init__(self) -> None:
        if self.executed_at_ns < self.issued_at_ns:
            raise ValueError("execution before command issue")
        if not self.verified:
            raise ValueError("only verified executed action can update factual context")


@dataclass(frozen=True)
class Fact:
    value: str
    status: EvidenceKind
    refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status in (EvidenceKind.OBSERVED, EvidenceKind.MEASURED) and not self.refs:
            raise ValueError("measured/observed fact must cite evidence")


@dataclass(frozen=True)
class ContextSnapshot:
    episode_id: str
    version: int
    cutoff_ns: int
    facts: Mapping[str, Fact] = field(default_factory=dict)
    pinned: frozenset[str] = frozenset()
    evidence_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.version < 0 or self.cutoff_ns < 0:
            raise ValueError("snapshot version/time invalid")
        if not self.pinned.issubset(self.facts):
            raise ValueError("pinned keys must exist")


@dataclass(frozen=True)
class Patch:
    episode_id: str
    base_version: int
    cutoff_ns: int
    key: str
    value: Fact | None  # None means remove from hot facts, NOT delete evidence
    evidence_refs: tuple[EvidenceRef, ...] = ()
    pin: bool = False


@dataclass(frozen=True)
class PlanRef:
    episode_id: str
    context_version: int
    goal_revision: int
    target_identity: str
    safety_revision: int
    issued_at_ns: int


@dataclass(frozen=True)
class PlanGuardState:
    episode_id: str
    context_version: int
    goal_revision: int
    target_identity: str
    safety_revision: int

    def permits(self, plan: PlanRef) -> bool:
        return (
            plan.episode_id == self.episode_id
            and plan.context_version == self.context_version
            and plan.goal_revision == self.goal_revision
            and plan.target_identity == self.target_identity
            and plan.safety_revision == self.safety_revision
        )
