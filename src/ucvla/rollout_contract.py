# SPDX-License-Identifier: Apache-2.0
"""M3 contract (status: CPU fake only; NO simulator, NO real rollout is implemented or authorised).

A paired rollout runs the SAME fixed policy from an identical environment state and identical
random streams under two memory conditions that differ by one counterfactual edit; the paired return
difference is the (noisy) evidence-value estimate. This module fixes the interface and the leakage
rules and provides a deterministic fake environment to test them."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ucvla.canonical import HashRng


class ResettableEnv(Protocol):
    def reset(self, seed: int) -> Mapping[str, str]: ...  # exact state + controlled RNG streams

    def step(self, action: str) -> tuple[Mapping[str, str], float, bool]: ...


@dataclass(frozen=True)
class PairedOutcome:
    seed: int
    ret_keep: float
    ret_edit: float

    @property
    def delta(self) -> float:
        return self.ret_keep - self.ret_edit


MemoryPolicy = Callable[[Sequence[Mapping[str, str]], Mapping[str, str]], str]


def rollout(env: ResettableEnv, policy: MemoryPolicy, seed: int, edit: Callable[[list], list] | None,
            max_steps: int = 50) -> float:
    obs, history, total = env.reset(seed), [], 0.0
    for _ in range(max_steps):
        ctx = list(history)
        ctx = edit(ctx) if edit else ctx  # counterfactual memory edit: edits what the policy is SHOWN only
        action = policy(ctx, obs)
        history.append(dict(obs))
        obs, r, done = env.step(action)
        total += r
        if done:
            break
    return total


def paired(env: ResettableEnv, policy: MemoryPolicy, edit: Callable[[list], list], seeds: Sequence[int]
           ) -> list[PairedOutcome]:
    """Same seeds, same policy; the world and the label never change between the two arms."""
    return [PairedOutcome(s, rollout(env, policy, s, None), rollout(env, policy, s, edit)) for s in seeds]


class FakeKeyEnv:
    """Deterministic CPU fake: a hidden key is shown at step 0; at the final step the correct action is the key."""

    def __init__(self, horizon: int = 4) -> None:
        self.horizon = horizon

    def reset(self, seed: int) -> Mapping[str, str]:
        self._key = str(HashRng("fake", seed).below(2))
        self._t = 0
        return {"phase": "show", "key": self._key}

    def step(self, action: str) -> tuple[Mapping[str, str], float, bool]:
        if self._t == self.horizon:  # the decision step: reward comes from the true hidden key
            return {"phase": "end", "key": "none"}, float(action == self._key), True
        self._t += 1
        phase = "ask" if self._t == self.horizon else "wait"
        return {"phase": phase, "key": "none"}, 0.0, False
