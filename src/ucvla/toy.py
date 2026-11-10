"""History-dependent CPU toy used only to test causal interfaces.

Two episodes have IDENTICAL visible final observation but distinct earlier
unobstructed object association; no VLA/GPU/model training here.
"""

from dataclasses import dataclass

from ucvla.contracts import EvidenceKind, EvidenceRef, Fact, Observation, Patch, hash_payload
from ucvla.memory import ContextStore


@dataclass(frozen=True)
class ToyResult:
    episode_id: str
    selected_cup: str
    correct_cup: str
    success: bool
    memory_version: int


def _observation(episode_id: str, t: int, visible: dict[str, str]) -> Observation:
    eid = f"{episode_id}-obs-{t}"
    ref = EvidenceRef(
        episode_id=episode_id,
        event_id=eid,
        time_ns=t,
        kind=EvidenceKind.OBSERVED,
        source_sha256=hash_payload(f"{eid}:{sorted(visible.items())}".encode()),
    )
    return Observation(episode_id=episode_id, time_ns=t, visible=visible, evidence=ref)


def paired_episode(episode_id: str, hidden_target: str) -> tuple[Observation, Observation]:
    if hidden_target not in ("left", "right"):
        raise ValueError("target must be left or right")
    early = _observation(episode_id, 10, {"red_ball_under": hidden_target})
    final = _observation(episode_id, 20, {"left": "opaque", "right": "opaque"})
    return early, final


def run_oracle_memory(episode_id: str, hidden_target: str) -> ToyResult:
    early, final = paired_episode(episode_id, hidden_target)
    store = ContextStore(episode_id, max_slots=1)
    store.commit(
        Patch(
            episode_id=episode_id,
            base_version=0,
            cutoff_ns=early.time_ns,
            key="target_cup",
            value=Fact(value=early.visible["red_ball_under"], status=EvidenceKind.OBSERVED, refs=(early.evidence.event_id,)),
            evidence_refs=(early.evidence,),
        )
    )
    store.commit(
        Patch(
            episode_id=episode_id,
            base_version=1,
            cutoff_ns=final.time_ns,
            key="target_cup",
            value=store.current.facts["target_cup"],
            evidence_refs=(final.evidence,),
        )
    )
    selected = store.current.facts["target_cup"].value
    return ToyResult(episode_id, selected, hidden_target, selected == hidden_target, store.current.version)


def run_memoryless(episode_id: str, hidden_target: str) -> ToyResult:
    _, final = paired_episode(episode_id, hidden_target)
    # Deterministic tie-breaker; final image alone contains no target identity.
    selected = min(final.visible)
    return ToyResult(episode_id, selected, hidden_target, selected == hidden_target, 0)
