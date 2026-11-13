# SPDX-License-Identifier: Apache-2.0
"""Clock-domain mapping and causal time alignment (integer arithmetic, no float drift).

Status: ``implemented`` (pure functions). Never assumes an unknown clock domain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TypeVar

T = TypeVar("T")


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class ClockMap:
    """canonical = offset + ref + (t - ref) * (1e6 + drift_ppm) // 1e6   (all integers, ns)."""

    domain: str
    offset_ns: int = 0
    drift_ppm: int = 0
    ref_ns: int = 0

    def __post_init__(self) -> None:
        if not self.domain:
            raise AlignmentError("clock domain name required")
        if self.drift_ppm <= -1_000_000:
            raise AlignmentError("drift would reverse time")

    def to_canonical(self, t_ns: int) -> int:
        if isinstance(t_ns, bool) or not isinstance(t_ns, int):
            raise AlignmentError("timestamps must be integer nanoseconds")
        scaled = self.ref_ns + ((t_ns - self.ref_ns) * (1_000_000 + self.drift_ppm)) // 1_000_000
        out = scaled + self.offset_ns
        if out < 0:
            raise AlignmentError("canonical time negative")
        return out


def resolve_map(domain: str, maps: Mapping[str, ClockMap]) -> ClockMap:
    try:
        return maps[domain]
    except KeyError:
        raise AlignmentError(f"unknown clock domain {domain!r}; refusing to assume") from None


def causal_ticks(
    stream: Sequence[tuple[int, T]], tick_times_ns: Sequence[int], max_staleness_ns: int
) -> list[tuple[int, T | None]]:
    """Sample-and-hold ``stream`` (canonical-time, payload) onto ``tick_times_ns``.

    A tick only ever receives a sample with time <= tick (never a future sample). A sample
    older than ``max_staleness_ns`` is reported as ``None`` (dropped) rather than reused.
    """
    times = [t for t, _ in stream]
    if any(b < a for a, b in pairwise(times)):
        raise AlignmentError("stream not time-ordered in canonical clock")
    if any(b <= a for a, b in pairwise(tick_times_ns)):
        raise AlignmentError("tick grid must be strictly increasing")
    out: list[tuple[int, T | None]] = []
    i = -1
    for tick in tick_times_ns:
        while i + 1 < len(stream) and stream[i + 1][0] <= tick:
            i += 1
        if i >= 0 and tick - stream[i][0] <= max_staleness_ns:
            out.append((tick, stream[i][1]))
        else:
            out.append((tick, None))
    return out


def merge_streams(
    streams: Mapping[str, Sequence[tuple[int, str, T]]], maps: Mapping[str, ClockMap]
) -> list[tuple[int, str, T]]:
    """Merge per-domain event streams ``name -> [(raw_t, clock_domain, payload)]``.

    Output is ordered by canonical time; ties break by stream name then source order, so the
    result is independent of dict ordering. Per-stream raw order must already be monotone.
    """
    merged: list[tuple[int, str, int, T]] = []
    for name in sorted(streams):
        last = -1
        for seq, (raw_t, domain, payload) in enumerate(streams[name]):
            t = resolve_map(domain, maps).to_canonical(raw_t)
            if t < last:
                raise AlignmentError(f"stream {name!r} non-monotone after mapping")
            last = t
            merged.append((t, name, seq, payload))
    merged.sort(key=lambda x: (x[0], x[1], x[2]))
    return [(t, name, payload) for t, name, _, payload in merged]
