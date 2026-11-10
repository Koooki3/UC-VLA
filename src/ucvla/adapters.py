"""Benchmark adapter interface, the usable SODH adapter and paired evaluation.

Registry statuses are explicit so that no code path implies a nonexistent integration works.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ucvla.baselines import ContextPolicy
from ucvla.episode import CausalView, EpisodeRecord
from ucvla.sodh import PairRecord

ADAPTER_STATUS: Mapping[str, str] = {
    "sodh_symbolic": "implemented (synthetic, symbolic frames)",
    "levi_events": "not implemented (deferred: operator decided 2026-11-11 not to do LEVI adaptation)",
    "lerobot_hdf5": "not implemented",
    "robomme": "not implemented (dataset/license review pending, docs/15)",
    "robomme_plus": "not implemented",
    "robotwin_mem": "not implemented",
    "robocasa_mem": "not implemented",
}


class BenchmarkAdapter(Protocol):
    name: str

    def pairs(self, split: str) -> Sequence[PairRecord]: ...

    def view(self, episode_id: str) -> CausalView: ...

    def score(self, episode_id: str, decision: str) -> bool: ...


class SODHAdapter:
    name = "sodh_symbolic"

    def __init__(self, episodes: Sequence[EpisodeRecord], pairs: Sequence[PairRecord],
                 assignment: Mapping[str, str]) -> None:
        self._eps = {e.episode_id: e for e in episodes}
        self._pairs = list(pairs)
        self._assignment = dict(assignment)

    def pairs(self, split: str) -> list[PairRecord]:
        return [p for p in self._pairs if self._assignment.get(p.pair_id) == split]

    def episode(self, episode_id: str) -> EpisodeRecord:
        return self._eps[episode_id]

    def view(self, episode_id: str) -> CausalView:
        ep = self._eps[episode_id]
        return ep.causal_view(ep.queries[0].time_ns)

    def score(self, episode_id: str, decision: str) -> bool:
        ep = self._eps[episode_id]
        return decision == ep.target_for(ep.queries[0].query_id)


@dataclass(frozen=True)
class PairOutcome:
    pair_id: str
    family: str
    regime: str
    horizon: int
    history_relevant: bool
    correct_a: bool
    correct_b: bool
    decisions_differ: bool
    targets_differ: bool
    retained_ticks: int


def evaluate_policy(adapter: SODHAdapter, policy: ContextPolicy, split: str) -> list[PairOutcome]:
    out = []
    for p in adapter.pairs(split):
        va, vb = adapter.view(p.episode_a), adapter.view(p.episode_b)
        da, db = policy.decide(va), policy.decide(vb)
        out.append(PairOutcome(
            p.pair_id, p.family, p.regime, p.horizon, p.history_relevant,
            adapter.score(p.episode_a, da), adapter.score(p.episode_b, db),
            da != db, p.target_a != p.target_b,
            (policy.retained(va) + policy.retained(vb)) // 2,
        ))
    return out


def summarize(outcomes: Sequence[PairOutcome]) -> dict[str, dict[str, float]]:
    """Per-family metrics. Pair-both-correct is the history-sensitive metric."""
    groups: dict[str, list[PairOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[o.family].append(o)
    summary = {}
    for family, rows in sorted(groups.items()):
        n = len(rows)
        correct = sum(r.correct_a + r.correct_b for r in rows)
        acc = correct / (2 * n)  # no CI here: mates share the observation, cluster CIs live in harness
        flips = [r for r in rows if r.targets_differ]
        summary[family] = {
            "n_pairs": n,
            "episode_acc": acc,
            "pair_both_correct": sum(r.correct_a and r.correct_b for r in rows) / n,
            "decision_flip_rate": (sum(r.decisions_differ for r in flips) / len(flips)
                                   if flips else float("nan")),
            "mean_retained_ticks": sum(r.retained_ticks for r in rows) / n,
        }
    return summary
