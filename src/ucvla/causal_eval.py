"""Causal attribution of a policy's performance to *history* (not current obs or data regularity).

Interventions edit only the history shown to the policy; the world and the label stay fixed.
The evaluator may use annotation events (which ticks carried the critical evidence); a policy
never can. Interpretation is conditional on what the agent's provider retained: if the critical
tick was never in context, deleting it cannot matter, and that is reported, not hidden.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ucvla.adapters import SODHAdapter
from ucvla.canonical import HashRng, digest
from ucvla.context import ContextBudget, ViewTick
from ucvla.episode import EpisodeRecord, SourceType
from ucvla.harness import Agent, decide, make_input


def critical_times(ep: EpisodeRecord) -> frozenset[int]:
    """Observation ticks carrying the decisive evidence (evaluator-side, annotation-derived)."""
    q_time = ep.queries[0].time_ns
    return frozenset(e.time_ns for e in ep.events
                     if e.source_type == SourceType.ANNOTATION and e.time_ns < q_time)


def delete_ticks(history: tuple[ViewTick, ...], times: frozenset[int]) -> tuple[ViewTick, ...]:
    return tuple(t for t in history if t.time_ns not in times)


def matched_noncritical(history: tuple[ViewTick, ...], critical: frozenset[int], seed: int,
                        episode_id: str) -> frozenset[int]:
    """Same number of deleted ticks as the critical deletion, drawn from non-critical ticks."""
    pool = [t.time_ns for t in history if t.time_ns not in critical]
    rng = HashRng("noncritical", seed, episode_id)
    rng.shuffle(pool)
    return frozenset(pool[:len(critical)])


def causal_attribution(adapter: SODHAdapter, agent: Agent, split: str, run_seed: int,
                       max_cost: ContextBudget | None = None) -> dict[str, dict[str, Any]]:
    """Per family: original accuracy, swap behaviour, and critical vs matched non-critical deletion."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in adapter.pairs(split):
        eps = {p.episode_a: adapter.episode(p.episode_a), p.episode_b: adapter.episode(p.episode_b)}
        ids = (p.episode_a, p.episode_b)
        for me, other in (ids, ids[::-1]):
            ep, partner = eps[me], eps[other]
            view = ep.causal_view(ep.queries[0].time_ns)
            pview = partner.causal_view(partner.queries[0].time_ns)
            target = ep.target_for(ep.queries[0].query_id)
            ptarget = partner.target_for(partner.queries[0].query_id)
            d0, _, _ = decide(agent, view, run_seed, me, max_cost)
            a = acc[p.family]
            a["original"].append(float(d0 == target))
            d_swap, _, _ = decide(agent, view, run_seed, me, max_cost, history=pview.history)
            if p.history_relevant:
                a["swap_follows_partner"].append(float(d_swap == ptarget))
            else:
                a["swap_invariance"].append(float(d_swap == d0))
            crit = critical_times(ep)
            if crit:
                d_c, _, _ = decide(agent, view, run_seed, me, max_cost,
                                   history=delete_ticks(view.history, crit))
                non = matched_noncritical(view.history, crit, run_seed, me)
                d_n, _, _ = decide(agent, view, run_seed, me, max_cost,
                                   history=delete_ticks(view.history, non))
                a["original_on_crit_subset"].append(float(d0 == target))
                a["del_critical"].append(float(d_c == target))
                a["del_noncritical"].append(float(d_n == target))
    out: dict[str, dict[str, Any]] = {}
    for fam, m in sorted(acc.items()):
        row: dict[str, Any] = {k: sum(v) / len(v) for k, v in m.items()}
        row["n"] = {k: len(v) for k, v in m.items()}
        if "del_critical" in m:
            row["drop_critical"] = row["original_on_crit_subset"] - row["del_critical"]
            row["drop_noncritical"] = row["original_on_crit_subset"] - row["del_noncritical"]
        out[fam] = row
    return out


def uses_history(row: dict[str, Any], min_drop: float = 0.3, max_noncrit: float = 0.1,
                 min_swap: float = 0.9) -> bool:
    """Diagnostic (not a proof): flips with swapped history, hurt by deleting critical evidence only."""
    return (row.get("swap_follows_partner", 0.0) >= min_swap
            and row.get("drop_critical", 0.0) >= min_drop
            and row.get("drop_noncritical", 1.0) <= max_noncrit)


def access_parity_digest(adapter: SODHAdapter, agents: Sequence[Agent], split: str,
                         max_cost: ContextBudget | None = None) -> dict[str, str]:
    """Digest of every agent's (observation, task) inputs: must be identical across agents.

    Only the Context may differ between agents; this is the test that no agent is handed extra
    information through the non-context channels.
    """
    out = {}
    for agent in agents:
        parts = []
        for p in adapter.pairs(split):
            for eid in (p.episode_a, p.episode_b):
                inp, _, _ = make_input(adapter.view(eid), agent.provider, max_cost)
                parts.append([inp.observation.time_ns, inp.observation.cameras,
                              inp.task.task, inp.task.instruction, inp.task.options,
                              inp.task.params, inp.context.history_len])
        out[agent.name] = digest(parts)
    return out


