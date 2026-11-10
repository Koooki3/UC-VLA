"""Policy evaluation harness: one code path, identical rules for every agent.

The harness alone (a) builds the PolicyInput from a causal view, (b) runs the context provider
and validates/measures it, (c) calls reset+infer, (d) records predicted -> commanded -> executed
(simulated, logged as such), (e) scores against the label the policy never sees. Interventions
(history swap/deletion) edit the *history the policy is shown*, never the world or the label.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ucvla.adapters import SODHAdapter
from ucvla.canonical import digest
from ucvla.context import ContextBudget, ContextProvider, ViewTick, build_context, token_cost
from ucvla.episode import CausalView, CommandedAction, EpisodeRecord, ExecutedAction
from ucvla.policy import PolicyAdapter, PolicyInput, PolicyObservation, TaskSpec, TrainablePolicy
from ucvla.sodh import TICK_NS, PairRecord
from ucvla.stats import wilson

FIT_SPLITS = ("train",)


@dataclass(frozen=True)
class Agent:
    """A context provider (memory-selection rule) paired with a policy."""

    provider: ContextProvider
    policy: PolicyAdapter

    @property
    def name(self) -> str:
        return f"{self.provider.name}+{self.policy.name}"

    def spec(self) -> dict[str, Any]:
        return {"provider": self.provider.name, "budget": [self.provider.budget.unit,
                                                           self.provider.budget.limit],
                "policy": dict(self.policy.spec())}


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: str
    pair_id: str
    pair_role: str
    family: str
    regime: str
    horizon: int
    history_relevant: bool
    decision: str
    target: str
    correct: bool
    ticks_retained: int
    tokens_retained: int
    commanded: CommandedAction
    executed: ExecutedAction  # simulated execution: measured == commanded, mode synthetic


def make_input(view: CausalView, provider: ContextProvider, max_cost: ContextBudget | None,
               history: tuple[ViewTick, ...] | None = None) -> tuple[PolicyInput, int, int]:
    """Build what the policy may see. ``history`` overrides the raw past (interventions only)."""
    q = view.query
    if q is None:
        raise ValueError("view has no decision query")
    raw = view.history if history is None else history
    ctx = build_context(provider, raw, view.current.time_ns, max_cost)
    inp = PolicyInput(
        observation=PolicyObservation(view.current.time_ns, view.current.cameras, view.proprio),
        task=TaskSpec(q.task, q.instruction, q.options, dict(q.params)),
        context=ctx,
    )
    return inp, len(ctx.ticks), token_cost(ctx.ticks)


def episode_seed(run_seed: int, episode_id: str) -> int:
    return int(digest([run_seed, "episode", episode_id])[:12], 16)


def decide(agent: Agent, view: CausalView, run_seed: int, episode_id: str,
           max_cost: ContextBudget | None, history: tuple[ViewTick, ...] | None = None
           ) -> tuple[str, int, int]:
    inp, n_ticks, n_tok = make_input(view, agent.provider, max_cost, history)
    agent.policy.reset(episode_seed(run_seed, episode_id))
    chunk = agent.policy.infer(inp)
    if chunk.actions[0] not in inp.task.options:
        raise ValueError(f"policy returned illegal action {chunk.actions[0]!r}")
    return chunk.actions[0], n_ticks, n_tok


def run_episode(ep: EpisodeRecord, pair: PairRecord, agent: Agent, run_seed: int,
                max_cost: ContextBudget | None) -> EpisodeResult:
    q = ep.queries[0]
    view = ep.causal_view(q.time_ns)
    decision, n_ticks, n_tok = decide(agent, view, run_seed, ep.episode_id, max_cost)
    target = ep.target_for(q.query_id)
    cmd = CommandedAction(f"{ep.episode_id}-cmd0", q.time_ns, decision)
    exe = ExecutedAction(cmd.command_id, q.time_ns + TICK_NS, decision, True)
    return EpisodeResult(ep.episode_id, pair.pair_id, str(ep.provenance["pair_role"]),
                         pair.family, pair.regime, pair.horizon, pair.history_relevant,
                         decision, target, decision == target, n_ticks, n_tok, cmd, exe)


def evaluate(adapter: SODHAdapter, agent: Agent, split: str, run_seed: int,
             max_cost: ContextBudget | None = None) -> list[EpisodeResult]:
    out = []
    for p in adapter.pairs(split):
        for eid in (p.episode_a, p.episode_b):
            out.append(run_episode(adapter.episode(eid), p, agent, run_seed, max_cost))
    return out


def fit_agent(adapter: SODHAdapter, agent: Agent, run_seed: int,
              split: str = "train", max_cost: ContextBudget | None = None) -> str:
    """Fit a trainable policy on TRAIN pairs only; returns a digest of the fitted-on episode ids."""
    if split not in FIT_SPLITS:
        raise PermissionError(f"fitting is only allowed on {FIT_SPLITS}, not {split!r}")
    if not hasattr(agent.policy, "fit"):
        return "not-trainable"
    examples, ids = [], []
    for p in adapter.pairs(split):
        for eid in (p.episode_a, p.episode_b):
            ep = adapter.episode(eid)
            view = ep.causal_view(ep.queries[0].time_ns)
            inp, _, _ = make_input(view, agent.provider, max_cost)
            examples.append((inp, ep.target_for(ep.queries[0].query_id)))
            ids.append(eid)
    trainable: TrainablePolicy = agent.policy  # type: ignore[assignment]
    trainable.fit(examples, run_seed)
    return digest(sorted(ids))


def pair_metrics(results: Sequence[EpisodeResult]) -> dict[str, dict[str, float]]:
    """Per family: episode accuracy (Wilson CI) and pair-both-correct (cluster = pair)."""
    by_pair: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        by_pair[r.pair_id].append(r)
    fam: dict[str, list[list[EpisodeResult]]] = defaultdict(list)
    for rs in by_pair.values():
        fam[rs[0].family].append(rs)
    out = {}
    for family, pairs in sorted(fam.items()):
        n_ep = sum(len(p) for p in pairs)
        k = sum(r.correct for p in pairs for r in p)
        acc, lo, hi = wilson(k, n_ep)
        out[family] = {
            "n_pairs": len(pairs), "episode_acc": acc, "acc_lo": lo, "acc_hi": hi,
            "pair_both_correct": sum(all(r.correct for r in p) for p in pairs) / len(pairs),
            "mean_ticks": sum(r.ticks_retained for p in pairs for r in p) / n_ep,
            "mean_tokens": sum(r.tokens_retained for p in pairs for r in p) / n_ep,
        }
    return out


def results_digest(results: Sequence[EpisodeResult]) -> str:
    return digest(sorted((r.episode_id, r.decision, r.ticks_retained, r.tokens_retained)
                         for r in results))


def results_to_rows(results: Sequence[EpisodeResult]) -> list[Mapping[str, Any]]:
    return [{"episode_id": r.episode_id, "pair_id": r.pair_id, "family": r.family,
             "regime": r.regime, "decision": r.decision, "target": r.target,
             "correct": r.correct, "ticks": r.ticks_retained, "tokens": r.tokens_retained}
            for r in results]
