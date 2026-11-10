"""Budgeted context selection and the harness-side access validator.

A ContextProvider decides *which past observations* a policy may use; the harness, not the
provider, measures the cost and rejects anything that is not a real, causal, in-budget subset
of the observed history (invariant I1). Status: ``implemented`` for raw-tick contexts;
latent/learned contexts (M4) are ``planned`` and rejected by the validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from ucvla.episode import ViewTick


class AccessViolation(ValueError):
    """A provider/policy tried to use information outside the declared access rules."""


def token_cost(ticks: tuple[ViewTick, ...]) -> int:
    """Hot-memory tokens = number of (camera, key) symbols retained."""
    return sum(len(c) for t in ticks for c in t.cameras.values())


@dataclass(frozen=True)
class ContextBudget:
    unit: str  # "ticks" | "tokens"
    limit: int | None  # None = unbounded (reference upper bound only; never a matched baseline)

    def __post_init__(self) -> None:
        if self.unit not in ("ticks", "tokens"):
            raise ValueError("unit must be 'ticks' or 'tokens'")
        if self.limit is not None and self.limit < 0:
            raise ValueError("negative budget")

    def cost(self, ticks: tuple[ViewTick, ...]) -> int:
        return len(ticks) if self.unit == "ticks" else token_cost(ticks)


@dataclass(frozen=True)
class Context:
    """What the harness hands the policy besides the current observation."""

    ticks: tuple[ViewTick, ...]  # raw past observations, strictly before the current tick
    cutoff_ns: int
    kind: str = "raw_ticks"


class ContextProvider(Protocol):
    name: str
    budget: ContextBudget

    def select(self, history: tuple[ViewTick, ...]) -> tuple[ViewTick, ...]: ...


def build_context(provider: ContextProvider, history: tuple[ViewTick, ...], current_ns: int,
                  max_cost: ContextBudget | None = None) -> Context:
    """Run ``provider`` and validate its output against the access rules."""
    chosen = tuple(provider.select(history))
    by_time = {t.time_ns: t for t in history}
    for t in chosen:
        if t.time_ns >= current_ns:
            raise AccessViolation("context contains the current or a future tick")
        if by_time.get(t.time_ns) != t:
            raise AccessViolation("context tick is not an observed history tick (fabricated/edited)")
    if any(b.time_ns <= a.time_ns for a, b in pairwise(chosen)):
        raise AccessViolation("context ticks must be strictly time-ordered (no duplicates)")
    declared = provider.budget
    if declared.limit is not None and declared.cost(chosen) > declared.limit:
        raise AccessViolation("provider exceeded its own declared budget")
    if max_cost is not None:
        if max_cost.limit is None:
            raise ValueError("max_cost must be bounded")
        if max_cost.cost(chosen) > max_cost.limit:
            raise AccessViolation(f"context cost exceeds matched budget {max_cost}")
    return Context(chosen, current_ns)


@dataclass(frozen=True)
class NoHistory:
    name: str = "no_history"
    budget: ContextBudget = ContextBudget("ticks", 0)

    def select(self, history: tuple[ViewTick, ...]) -> tuple[ViewTick, ...]:
        return ()


@dataclass(frozen=True)
class AllHistory:
    name: str = "all_history"
    budget: ContextBudget = ContextBudget("ticks", None)

    def select(self, history: tuple[ViewTick, ...]) -> tuple[ViewTick, ...]:
        return history


@dataclass(frozen=True)
class LastK:
    k: int
    unit: str = "ticks"

    @property
    def name(self) -> str:
        return f"last_{self.k}{'' if self.unit == 'ticks' else 'tok'}"

    @property
    def budget(self) -> ContextBudget:
        return ContextBudget(self.unit, self.k)

    def select(self, history: tuple[ViewTick, ...]) -> tuple[ViewTick, ...]:
        if self.unit == "ticks":
            return history[len(history) - min(self.k, len(history)):]
        out: list[ViewTick] = []
        used = 0
        for t in reversed(history):
            c = token_cost((t,))
            if used + c > self.k:
                break
            out.append(t)
            used += c
        return tuple(reversed(out))


@dataclass(frozen=True)
class UniformK:
    k: int

    @property
    def name(self) -> str:
        return f"uniform_{self.k}"

    @property
    def budget(self) -> ContextBudget:
        return ContextBudget("ticks", self.k)

    def select(self, history: tuple[ViewTick, ...]) -> tuple[ViewTick, ...]:
        n = len(history)
        if self.k >= n:
            return history
        return tuple(history[(2 * i + 1) * n // (2 * self.k)] for i in range(self.k))
