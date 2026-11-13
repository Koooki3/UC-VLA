# SPDX-License-Identifier: Apache-2.0
"""Contract that every episode source (HDF5, LeRobot, real benchmarks) must satisfy.

Status: the *contract and conformance checks* are ``implemented``; the importers below are
``not implemented`` and raise. A source only counts as integrated after it passes
:func:`check_source_conformance` on real files AND its licence status is ``verified`` by the
operator (docs/16). Nothing here downloads data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ucvla.adapters import SODHAdapter
from ucvla.episode import ONLINE_SOURCES, EpisodeRecord

REQUIRED_PROVENANCE = ("source_format", "timestamps_origin", "executed_origin", "license_ref")
ORIGINS = ("native", "derived", "absent")


@dataclass(frozen=True)
class SourceManifest:
    name: str
    version: str  # pinned commit / dataset revision, never a branch name
    source_format: str  # "hdf5" | "lerobot" | "symbolic" | ...
    license_status: str  # "verified" | "unverified" (operator decision)
    license_ref: str
    official_protocol_ref: str
    privileged_fields: tuple[str, ...] = ()  # source fields that must never reach a policy


class EpisodeSource(Protocol):
    manifest: SourceManifest

    def split_names(self) -> Sequence[str]: ...

    def episode_ids(self, split: str) -> Sequence[str]: ...

    def load(self, episode_id: str) -> EpisodeRecord: ...


def check_source_conformance(source: EpisodeSource, max_episodes: int = 200) -> list[str]:
    """Return a list of violations (empty = conformant). Run on the real files of any importer."""
    bad: list[str] = []
    m = source.manifest
    if m.license_status != "verified":
        bad.append(f"licence status {m.license_status!r}: not cleared for use")
    if len(m.version) < 7 or m.version in ("main", "master", "latest"):
        bad.append("version must pin an exact revision, not a moving branch")
    seen_ids: dict[str, str] = {}
    seen_content: dict[str, str] = {}
    n = 0
    for split in source.split_names():
        for eid in source.episode_ids(split):
            if n >= max_episodes:
                break
            n += 1
            ep = source.load(eid)  # EpisodeRecord construction already validates the schema
            if ep.episode_id != eid:
                bad.append(f"{eid}: loaded id differs")
            if eid in seen_ids and seen_ids[eid] != split:
                bad.append(f"{eid}: appears in splits {seen_ids[eid]} and {split}")
            seen_ids[eid] = split
            h = ep.content_sha256()
            if h in seen_content and seen_content[h] != eid and seen_ids.get(seen_content[h]) != split:
                bad.append(f"{eid}: duplicate content of {seen_content[h]} across splits")
            seen_content.setdefault(h, eid)
            bad += _episode_violations(ep)
    return bad


def _episode_violations(ep: EpisodeRecord) -> list[str]:
    bad = []
    for key in REQUIRED_PROVENANCE:
        if key not in ep.provenance:
            bad.append(f"{ep.episode_id}: provenance missing {key}")
    for key in ("timestamps_origin", "executed_origin"):
        if ep.provenance.get(key) not in ORIGINS:
            bad.append(f"{ep.episode_id}: {key} must be one of {ORIGINS}")
    if ep.provenance.get("executed_origin") != "native" and ep.executed:
        bad.append(f"{ep.episode_id}: executed actions present but origin is not 'native'")
    prev = None
    for view in ep.replay():
        if any(t.time_ns > view.cutoff_ns for t in view.ticks):
            bad.append(f"{ep.episode_id}: future tick in view at {view.cutoff_ns}")
        if any(e.source_type not in ONLINE_SOURCES for e in view.events):
            bad.append(f"{ep.episode_id}: non-online event leaked into a view")
        if prev is not None and view.ticks[: len(prev)] != prev:
            bad.append(f"{ep.episode_id}: view is not an extension of the previous view")
        prev = view.ticks
    return bad


class SODHSource:
    """Reference implementation of the contract (symbolic, synthetic)."""

    def __init__(self, adapter: SODHAdapter, version: str) -> None:
        self._a = adapter
        self.manifest = SourceManifest("sodh_symbolic", version, "symbolic", "verified",
                                       "project-internal synthetic data", "docs/06_BENCHMARKS.md")

    def split_names(self) -> Sequence[str]:
        return ("train", "val", "test_id", "test_ood", "test_counterfactual")

    def episode_ids(self, split: str) -> list[str]:
        return [e for p in self._a.pairs(split) for e in (p.episode_a, p.episode_b)]

    def load(self, episode_id: str) -> EpisodeRecord:
        # View-level shim: stored (frozen) v1 episodes are untouched; the contract keys are added
        # here. SODH has native timestamps, no robot, hence no executed stream.
        ep = self._a.episode(episode_id)
        extra = {"source_format": "symbolic", "timestamps_origin": "native",
                 "executed_origin": "absent", "license_ref": self.manifest.license_ref}
        return replace(ep, provenance={**ep.provenance, **extra})


class NotImplementedImporter:
    """Placeholder that fails loudly; guarantees no code path claims real-data support."""

    def __init__(self, fmt: str, hint: str) -> None:
        self.fmt, self.hint = fmt, hint

    def __call__(self, *args: object, **kwargs: object) -> Iterable[EpisodeRecord]:
        raise NotImplementedError(f"{self.fmt} importer is not implemented. {self.hint}")


HDF5_IMPORTER = NotImplementedImporter(
    "hdf5", "Map frames->Frame(source_sha256 of raw bytes), state->ProprioSample, commanded "
            "actions->CommandedAction; timestamps from native clock or (derived, from Hz, "
            "labelled); privileged/oracle fields->PrivilegedState only (docs/16).")
LEROBOT_IMPORTER = NotImplementedImporter(
    "lerobot", "Use the pinned dataset revision; do not mix LeRobot/LEVI tensor conventions "
               "in one class (docs/09); episode table timestamps map through alignment.ClockMap.")
REQUIRED_FIELDS: Mapping[str, str] = {
    "frames": "multi-camera, time-aligned, per-frame source hash",
    "proprio": "measured state with units",
    "commanded": "issued commands (never mixed with executed)",
    "executed": "native only; otherwise absent and provenance executed_origin != native",
    "targets/privileged": "labels and simulator/oracle state, excluded from every CausalView",
}
