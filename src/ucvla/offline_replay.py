"""Minimal offline replay adapter for real episodes (status: ``implemented`` for the RoboMME HDF5
sample; CPU fakes only). Feeds each tick of a recorded episode to a policy through the SAME
PolicyInput boundary as the synthetic suites and scores the predicted action against the recorded
commanded action (imitation error). This is an interface/leakage test, NOT a VLA evaluation.
Rules: the command answering observation i is never visible at tick i; pixels are not loaded here
(the observation carries hashes/refs; a real VLA adapter would resolve refs through the importer).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ucvla.context import ContextProvider, build_context
from ucvla.episode import EpisodeRecord
from ucvla.policy import ActionChunk, PolicyAdapter, PolicyInput, PolicyObservation, TaskSpec


@dataclass(frozen=True)
class ReplayResult:
    episode_id: str
    n_steps: int
    mse: float
    mse_gripper: float


def _instruction(ep: EpisodeRecord) -> str:
    return next((e.payload["text"] for e in ep.events if e.kind == "instruction"), "")


def replay_inputs(ep: EpisodeRecord, provider: ContextProvider, k_commands: int):
    """Yield (step, PolicyInput, recorded_command) with strict causality."""
    for i, t in enumerate(ep.tick_times()):
        view = ep.causal_view(t)
        ctx = build_context(provider, view.history, t)
        ctx = type(ctx)(ctx.ticks, ctx.cutoff_ns, ctx.kind,
                        view.commanded[len(view.commanded) - min(k_commands, len(view.commanded)):])
        inp = PolicyInput(
            PolicyObservation(t, view.current.cameras, view.proprio[-1:]),
            TaskSpec(ep.task, _instruction(ep), (), {}), ctx)
        yield i, inp, ep.commanded[i]


def replay_evaluate(ep: EpisodeRecord, policy: PolicyAdapter, provider: ContextProvider,
                    k_commands: int = 1, seed: int = 0) -> ReplayResult:
    policy.reset(seed)
    se, sg, n = 0.0, 0.0, 0
    for _i, inp, rec in replay_inputs(ep, provider, k_commands):
        pred: list[float] = json.loads(policy.infer(inp).actions[0])
        target: list[float] = json.loads(rec.action)
        if len(pred) != len(target):
            raise ValueError("action dimension mismatch")
        se += sum((p - t) ** 2 for p, t in zip(pred, target)) / len(target)
        sg += (pred[-1] - target[-1]) ** 2
        n += 1
    return ReplayResult(ep.episode_id, n, se / n, sg / n)


@dataclass
class ZeroOrderHold:
    """Repeat the most recent *past* command; before any command, hold the measured joints."""

    name: str = "zero_order_hold"

    def spec(self) -> Mapping[str, Any]:
        return {"class": "ZeroOrderHold", "role": "CPU fake; uses only past commands/proprio"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        if inp.context.commanded:
            return ActionChunk((inp.context.commanded[-1].action,))
        p = inp.observation.proprio[-1].values
        return ActionChunk((json.dumps([*p[:7], 1.0]),))


@dataclass
class ProprioCopy:
    """Command = current measured joint state (+ open gripper): ignores all history."""

    name: str = "proprio_copy"

    def spec(self) -> Mapping[str, Any]:
        return {"class": "ProprioCopy", "role": "CPU fake; current observation only"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        p = inp.observation.proprio[-1].values
        grip = 1.0 if p[7] > 0.02 else -1.0  # gripper_state[0] is a finger opening (heuristic)
        return ActionChunk((json.dumps([*p[:7], grip]),))


def summary(results: list[ReplayResult]) -> dict[str, float]:
    n = sum(r.n_steps for r in results)
    return {"episodes": len(results), "steps": n,
            "mse": sum(r.mse * r.n_steps for r in results) / n,
            "rmse": math.sqrt(sum(r.mse * r.n_steps for r in results) / n)}
