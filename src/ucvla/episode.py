# SPDX-License-Identifier: Apache-2.0
"""Versioned episode schema and causal (cutoff-truncated) views.

Status: ``implemented`` for symbolic/synthetic episodes. Image bytes are not stored; a frame
carries a ``visible`` token map (symbolic) and a source hash that binds it to its content.

Separation enforced by the schema:
  * ``commanded`` (issued commands)  vs  ``executed`` (measured, verified executions);
    predicted future actions live in plans (contracts.PlanRef), never in an episode.
  * online inputs (frames, proprio, commands, verified executions, observed/measured events)
    vs labels (``targets``, ANNOTATION events) vs ``privileged`` simulator state.
Only the first group can appear in a :class:`CausalView` (invariant I1).
"""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from ucvla.canonical import digest, is_sha256_hex, to_jsonable

SCHEMA_VERSION = "ucvla.episode/1"
REQUIRED_PROVENANCE = ("generator", "generator_version", "config_sha256", "seed")


class SourceType(str, enum.Enum):
    OBSERVED = "observed"
    MEASURED = "measured"
    COMMANDED = "commanded"
    ANNOTATION = "annotation"
    INFERRED = "inferred"
    SIMULATED_PRIVILEGED = "simulated_privileged"


# Sources a deployed policy may legitimately read. Annotation / inferred / privileged never.
ONLINE_SOURCES = frozenset({SourceType.OBSERVED, SourceType.MEASURED, SourceType.COMMANDED})


def _is_time(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


@dataclass(frozen=True)
class Frame:
    frame_id: str
    camera_id: str
    time_ns: int
    visible: Mapping[str, str]
    source_sha256: str
    source_type: SourceType = SourceType.OBSERVED

    @staticmethod
    def content_hash(camera_id: str, time_ns: int, visible: Mapping[str, str]) -> str:
        return digest({"camera_id": camera_id, "time_ns": time_ns, "visible": visible})

    @classmethod
    def make(cls, frame_id: str, camera_id: str, time_ns: int, visible: Mapping[str, str]) -> Frame:
        return cls(frame_id, camera_id, time_ns, dict(visible),
                   cls.content_hash(camera_id, time_ns, visible))


@dataclass(frozen=True)
class ProprioSample:
    time_ns: int
    values: tuple[float, ...]
    source_type: SourceType = SourceType.MEASURED


@dataclass(frozen=True)
class CommandedAction:
    command_id: str
    issued_at_ns: int
    action: str


@dataclass(frozen=True)
class ExecutedAction:
    command_id: str
    executed_at_ns: int
    measured_action: str
    verified: bool


@dataclass(frozen=True)
class Event:
    event_id: str
    time_ns: int
    window_end_ns: int
    kind: str
    payload: Mapping[str, str]
    source_type: SourceType
    source_sha256: str

    @staticmethod
    def content_hash(kind: str, time_ns: int, window_end_ns: int, payload: Mapping[str, str],
                     source_type: SourceType) -> str:
        return digest({"kind": kind, "time_ns": time_ns, "window_end_ns": window_end_ns,
                       "payload": payload, "source_type": source_type})

    @classmethod
    def make(cls, event_id: str, time_ns: int, kind: str, payload: Mapping[str, str],
             source_type: SourceType, window_end_ns: int | None = None) -> Event:
        end = time_ns if window_end_ns is None else window_end_ns
        return cls(event_id, time_ns, end, kind, dict(payload), source_type,
                   cls.content_hash(kind, time_ns, end, payload, source_type))


@dataclass(frozen=True)
class DecisionQuery:
    """What the policy must decide at ``time_ns``; contains no label."""

    query_id: str
    time_ns: int
    task: str
    instruction: str
    options: tuple[str, ...]
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Target:
    """Label for a query. Never an input at any cutoff."""

    query_id: str
    value: str
    source_type: SourceType = SourceType.ANNOTATION


@dataclass(frozen=True)
class PrivilegedState:
    """Simulator ground truth. Train-time supervision only; never online, never in a view."""

    time_ns: int
    state: Mapping[str, str]
    source_type: SourceType = SourceType.SIMULATED_PRIVILEGED


@dataclass(frozen=True)
class EpisodeRecord:
    schema_version: str
    episode_id: str
    task: str
    clock_domain: str
    frames: tuple[Frame, ...]
    proprio: tuple[ProprioSample, ...]
    commanded: tuple[CommandedAction, ...]
    executed: tuple[ExecutedAction, ...]
    events: tuple[Event, ...]
    queries: tuple[DecisionQuery, ...]
    targets: tuple[Target, ...]
    privileged: PrivilegedState | None
    provenance: Mapping[str, str | int | bool]

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("invalid episode: " + "; ".join(problems))

    # ---- validation -------------------------------------------------------------------
    def problems(self) -> list[str]:
        p: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            p.append(f"schema_version {self.schema_version!r} != {SCHEMA_VERSION!r}")
        if not self.episode_id or not self.task or not self.clock_domain:
            p.append("episode_id/task/clock_domain required")
        for name, ids in (
            ("frame", [f.frame_id for f in self.frames]),
            ("event", [e.event_id for e in self.events]),
            ("command", [c.command_id for c in self.commanded]),
            ("query", [q.query_id for q in self.queries]),
        ):
            if len(set(ids)) != len(ids) or not all(ids):
                p.append(f"duplicate or empty {name} id")
        for key in REQUIRED_PROVENANCE:
            if key not in self.provenance:
                p.append(f"provenance missing {key}")
        p += self._frame_problems()
        p += self._action_problems()
        p += self._event_problems()
        tick_times = {f.time_ns for f in self.frames}
        for q in self.queries:
            if not _is_time(q.time_ns) or q.time_ns not in tick_times:
                p.append(f"query {q.query_id} not at an observation tick")
            if len(set(q.options)) != len(q.options) or len(q.options) < 2:
                p.append(f"query {q.query_id} needs >=2 distinct options")
        qids = {q.query_id: q for q in self.queries}
        seen: set[str] = set()
        for t in self.targets:
            if t.query_id not in qids or t.query_id in seen:
                p.append(f"target for unknown/duplicate query {t.query_id}")
            elif t.value not in qids[t.query_id].options:
                p.append(f"target {t.query_id} not among options")
            seen.add(t.query_id)
            if t.source_type != SourceType.ANNOTATION:
                p.append("target must be ANNOTATION")
        if self.privileged is not None and (
            self.privileged.source_type != SourceType.SIMULATED_PRIVILEGED
            or not _is_time(self.privileged.time_ns)
        ):
            p.append("privileged state must be SIMULATED_PRIVILEGED with valid time")
        return p

    def _frame_problems(self) -> list[str]:
        p: list[str] = []
        if not self.frames:
            return ["no frames"]
        last: dict[str, int] = {}
        by_time: dict[int, set[str]] = {}
        for f in self.frames:
            if not _is_time(f.time_ns):
                p.append(f"frame {f.frame_id}: bad time")
                continue
            if f.source_type != SourceType.OBSERVED:
                p.append(f"frame {f.frame_id}: must be OBSERVED")
            if not is_sha256_hex(f.source_sha256) or f.source_sha256 != Frame.content_hash(
                f.camera_id, f.time_ns, f.visible
            ):
                p.append(f"frame {f.frame_id}: source hash mismatch")
            if f.camera_id in last and f.time_ns <= last[f.camera_id]:
                p.append(f"frame {f.frame_id}: camera time not strictly increasing")
            last[f.camera_id] = f.time_ns
            by_time.setdefault(f.time_ns, set()).add(f.camera_id)
        cams = set(last)
        if any(c != cams for c in by_time.values()):
            p.append("cameras not time-aligned: every tick needs every camera")
        times = [f.time_ns for f in self.frames]
        if times != sorted(times):
            p.append("frames must be ordered by time")
        prop = [s.time_ns for s in self.proprio]
        if not all(_is_time(t) for t in prop) or any(a >= b for a, b in pairwise(prop)):
            p.append("proprio times must be valid and strictly increasing")
        if any(s.source_type != SourceType.MEASURED for s in self.proprio):
            p.append("proprio must be MEASURED")
        return p

    def _action_problems(self) -> list[str]:
        p: list[str] = []
        issued = {c.command_id: c.issued_at_ns for c in self.commanded}
        if not all(_is_time(t) for t in issued.values()):
            p.append("bad command time")
        cmd_t = [c.issued_at_ns for c in self.commanded]
        if cmd_t != sorted(cmd_t):
            p.append("commands out of order")
        exec_ids: set[str] = set()
        for e in self.executed:
            if e.command_id not in issued or e.command_id in exec_ids:
                p.append(f"execution of unknown/duplicate command {e.command_id}")
            elif not _is_time(e.executed_at_ns) or e.executed_at_ns < issued[e.command_id]:
                p.append(f"execution {e.command_id} before issue")
            exec_ids.add(e.command_id)
            if not e.verified:
                p.append(f"execution {e.command_id} not verified")
        return p

    def _event_problems(self) -> list[str]:
        p: list[str] = []
        for e in self.events:
            if not (_is_time(e.time_ns) and _is_time(e.window_end_ns)
                    and e.window_end_ns >= e.time_ns):
                p.append(f"event {e.event_id}: bad time window")
            if e.source_type == SourceType.SIMULATED_PRIVILEGED:
                p.append(f"event {e.event_id}: privileged state must not be an event")
            elif e.source_sha256 != Event.content_hash(
                e.kind, e.time_ns, e.window_end_ns, e.payload, e.source_type
            ):
                p.append(f"event {e.event_id}: source hash mismatch")
        times = [e.time_ns for e in self.events]
        if times != sorted(times):
            p.append("events must be ordered by time")
        return p

    # ---- identity / serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def episode_sha256(self) -> str:
        return digest(self)

    def content_sha256(self) -> str:
        """Hash of everything except identifiers/provenance: the duplicate-detection key."""
        return digest({
            "task": self.task,
            "frames": [(f.camera_id, f.time_ns, f.visible) for f in self.frames],
            "proprio": self.proprio,
            "commanded": [(c.issued_at_ns, c.action) for c in self.commanded],
            "executed": [(e.executed_at_ns, e.measured_action) for e in self.executed],
            "events": [(e.time_ns, e.window_end_ns, e.kind, e.payload, e.source_type)
                       for e in self.events],
            "queries": [(q.time_ns, q.task, q.instruction, q.options, q.params)
                        for q in self.queries],
            "targets": sorted(t.value for t in self.targets),
        })

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> EpisodeRecord:
        priv = d["privileged"]
        return cls(
            schema_version=d["schema_version"],
            episode_id=d["episode_id"],
            task=d["task"],
            clock_domain=d["clock_domain"],
            frames=tuple(Frame(**{**f, "source_type": SourceType(f["source_type"])})
                         for f in d["frames"]),
            proprio=tuple(ProprioSample(s["time_ns"], tuple(s["values"]), SourceType(s["source_type"]))
                          for s in d["proprio"]),
            commanded=tuple(CommandedAction(**c) for c in d["commanded"]),
            executed=tuple(ExecutedAction(**e) for e in d["executed"]),
            events=tuple(Event(**{**e, "source_type": SourceType(e["source_type"])})
                         for e in d["events"]),
            queries=tuple(DecisionQuery(**{**q, "options": tuple(q["options"])})
                          for q in d["queries"]),
            targets=tuple(Target(t["query_id"], t["value"], SourceType(t["source_type"]))
                          for t in d["targets"]),
            privileged=None if priv is None else PrivilegedState(
                priv["time_ns"], priv["state"], SourceType(priv["source_type"])),
            provenance=dict(d["provenance"]),
        )

    # ---- causal access ----------------------------------------------------------------
    def tick_times(self) -> tuple[int, ...]:
        return tuple(sorted({f.time_ns for f in self.frames}))

    def query_at(self, time_ns: int) -> DecisionQuery | None:
        return next((q for q in self.queries if q.time_ns == time_ns), None)

    def target_for(self, query_id: str) -> str:
        return next(t.value for t in self.targets if t.query_id == query_id)

    def causal_view(self, cutoff_ns: int) -> CausalView:
        """Everything a deployed policy may know at ``cutoff_ns`` (inclusive), nothing else."""
        ticks: dict[int, dict[str, Mapping[str, str]]] = {}
        for f in self.frames:
            if f.time_ns <= cutoff_ns and f.source_type in ONLINE_SOURCES:
                ticks.setdefault(f.time_ns, {})[f.camera_id] = dict(f.visible)
        if not ticks:
            raise ValueError("cutoff precedes first observation")
        issued = {c.command_id: c for c in self.commanded}
        return CausalView(
            episode_id=self.episode_id,
            cutoff_ns=cutoff_ns,
            task=self.task,
            ticks=tuple(ViewTick(t, ticks[t]) for t in sorted(ticks)),
            proprio=tuple(s for s in self.proprio
                          if s.time_ns <= cutoff_ns and s.source_type in ONLINE_SOURCES),
            commanded=tuple(c for c in self.commanded if c.issued_at_ns <= cutoff_ns),
            executed=tuple(e for e in self.executed
                           if e.executed_at_ns <= cutoff_ns and e.verified
                           and e.command_id in issued),
            events=tuple(e for e in self.events
                         if e.window_end_ns <= cutoff_ns and e.source_type in ONLINE_SOURCES),
            query=self.query_at(cutoff_ns),
        )

    def replay(self) -> Iterator[CausalView]:
        """Deterministic replay: one causal view per observation tick, in time order."""
        for t in self.tick_times():
            yield self.causal_view(t)


@dataclass(frozen=True)
class ViewTick:
    time_ns: int
    cameras: Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class CausalView:
    episode_id: str
    cutoff_ns: int
    task: str
    ticks: tuple[ViewTick, ...]
    proprio: tuple[ProprioSample, ...]
    commanded: tuple[CommandedAction, ...]
    executed: tuple[ExecutedAction, ...]
    events: tuple[Event, ...]
    query: DecisionQuery | None

    @property
    def current(self) -> ViewTick:
        return self.ticks[-1]

    @property
    def history(self) -> tuple[ViewTick, ...]:
        return self.ticks[:-1]

    def digest(self, include_history: bool = True) -> str:
        """Hash of observation *content*; ids and absolute times are excluded on purpose.

        Times are excluded so that "identical observation" means identical content; episode
        length / timing are audited separately as metadata features (audit.STRUCTURAL).
        """
        ticks = self.ticks if include_history else self.ticks[-1:]
        q = self.query
        query = None if q is None else {
            "task": q.task, "instruction": q.instruction, "options": q.options,
            "params": q.params}
        return digest({"task": self.task, "ticks": [t.cameras for t in ticks], "query": query})

    def channel_digests(self, include_history: bool = False) -> dict[str, str]:
        """Per-channel content hashes of the WHOLE view (ids and absolute times excluded).

        Used by the auditor: two mates of a clean pair may differ only in ``history``.
        """
        ticks = self.ticks if include_history else self.ticks[-1:]
        q = self.query
        rel = self.cutoff_ns
        return {
            "task": digest(self.task),
            "ticks": digest([t.cameras for t in ticks]),
            "proprio": digest([s.values for s in self.proprio]),
            "commanded": digest([(c.issued_at_ns - rel, c.action) for c in self.commanded]),
            "executed": digest([(e.executed_at_ns - rel, e.measured_action) for e in self.executed]),
            "events": digest([(e.window_end_ns - rel, e.kind, e.payload, e.source_type)
                              for e in self.events]),
            "query": digest(None if q is None else
                            [q.task, q.instruction, q.options, q.params]),
        }
