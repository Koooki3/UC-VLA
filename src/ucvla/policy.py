"""PolicyAdapter contract: the ONLY interface through which a policy meets the evaluator.

Information boundary (enforced by construction; see harness.py and tests/test_harness.py):
  * A policy receives a :class:`PolicyInput` = current observation + task spec + a harness-built
    :class:`~ucvla.context.Context`. It never receives an episode, a CausalView, ids, targets,
    privileged state, annotation events, provenance, or the pair partner.
  * ``ActionChunk`` holds *predicted* actions only. Only the harness turns a prediction into a
    *commanded* action and records the *executed* (measured, verified) one; in the synthetic
    suite that step is simulated and logged as such. No code here actuates anything.
Status: ``implemented`` (CPU, symbolic). A real VLA adapter is ``not implemented``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ucvla.context import Context
from ucvla.episode import ProprioSample


@dataclass(frozen=True)
class PolicyObservation:
    time_ns: int
    cameras: Mapping[str, Mapping[str, str]]
    proprio: tuple[ProprioSample, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    """Instruction and legal actions; no query id, no label."""

    task: str
    instruction: str
    options: tuple[str, ...]
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyInput:
    observation: PolicyObservation
    task: TaskSpec
    context: Context


@dataclass(frozen=True)
class ActionChunk:
    actions: tuple[str, ...]  # predicted; len == chunk horizon (1 for discrete SODH decisions)
    kind: str = "predicted"

    def __post_init__(self) -> None:
        if self.kind != "predicted" or not self.actions:
            raise ValueError("a policy returns a non-empty chunk of *predicted* actions")


class PolicyAdapter(Protocol):
    name: str

    def spec(self) -> Mapping[str, Any]:
        """JSON-able identity (class, hyper-parameters, fitted-on digest) for run records."""

    def reset(self, episode_seed: int) -> None:
        """Clear all per-episode state. Called before every episode and every intervention."""

    def infer(self, inp: PolicyInput) -> ActionChunk: ...


class TrainablePolicy(PolicyAdapter, Protocol):
    def fit(self, examples: Sequence[tuple[PolicyInput, str]], seed: int) -> None:
        """Fit on (input, target) pairs the harness drew from the TRAIN split only."""
