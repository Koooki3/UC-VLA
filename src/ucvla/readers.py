# SPDX-License-Identifier: Apache-2.0
"""CPU reference policies behind the PolicyAdapter (status: ``mock`` — not VLAs).

oracle_decoder   hand-written per-task decoder with built-in task knowledge. It measures whether
                 the *context it is given* is information-sufficient; it is NOT a method and its
                 scores must never be reported as method performance.
obs_only_lookup  trainable memoriser of current observation -> label (a current-image-only proxy).
bag_of_tokens_nb trainable generic naive-Bayes over retained-history tokens (no task knowledge).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ucvla.canonical import digest
from ucvla.policy import ActionChunk, PolicyInput

FRONT = "cam_front"


def _guess(inp: PolicyInput, ticks: Sequence[Mapping[str, Mapping[str, str]]]) -> str:
    """Deterministic function of exactly what the policy holds (never ids, labels or seeds)."""
    t = inp.task
    h = digest({"kept": list(ticks), "now": inp.observation.cameras, "task": t.task,
                "params": t.params})
    return t.options[int(h[:8], 16) % len(t.options)]


TICK_NS = 100_000_000  # control period of the synthetic clock (public task constant)


@dataclass
class OracleDecoder:
    """variant "strict": uses only retained ticks, never infers absence of evidence.
    variant "clock":  additionally uses elapsed time from the observation clock (free side
    information) for completeness and for scaling subsampled press counts.
    The two bracket how much a result depends on decoder priors (review finding B2)."""

    variant: str = "strict"

    def __post_init__(self) -> None:
        if self.variant not in ("strict", "clock"):
            raise ValueError("variant must be 'strict' or 'clock'")

    @property
    def name(self) -> str:
        return f"oracle_decoder_{self.variant}"

    def spec(self) -> Mapping[str, Any]:
        return {"class": "OracleDecoder", "variant": self.variant,
                "role": "decoder self-check / information-sufficiency probe, not a method"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        t, ctx = inp.task, inp.context
        kept = ctx.ticks
        clock = self.variant == "clock"
        total = round(inp.observation.time_ns / TICK_NS) - 1 if clock else len(kept)
        complete = clock and len(kept) == total
        now = inp.observation.cameras[FRONT]
        front = [k.cameras[FRONT] for k in kept]
        guess = _guess(inp, [k.cameras for k in kept])
        if t.task == "cups_ball":
            cups = [f["ball"][6:] for f in front if f["ball"].startswith("under:")]
            slots = {v: k for k, v in now.items() if k.startswith("slot")}
            out = f"pick_slot_{slots[cups[0]][4:]}" if cups and cups[0] in slots else guess
        elif t.task == "button_count":
            k_target = int(t.params["target_count"])
            if not kept:
                out = "press" if (clock and total == 0) else guess
            else:
                pressed = sum(1 for f in front if f["button"] == "pressed")
                n_hat = round(pressed * total / len(kept)) if clock else pressed
                out = "press" if n_hat < k_target else "stop"
        elif t.task == "resume_sequence":
            order = t.params["phase_order"].split(",")
            done = [f["placing"] for f in front if f["placing"] != "none"]
            if done:
                i = order.index(done[-1])
                out = f"place_{order[i + 1]}" if i + 1 < len(order) else guess
            else:
                out = f"place_{order[0]}" if complete else guess
        elif t.task == "pick_flagged":
            out = f"pick_{now['flag']}"
        else:
            raise ValueError(f"oracle decoder has no rule for task {t.task!r}")
        return ActionChunk((out,))


@dataclass
class ObsOnlyLookup:
    name: str = "obs_only_lookup"
    _table: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    _fit_digest: str | None = None

    @staticmethod
    def _key(inp: PolicyInput) -> str:
        t = inp.task
        return digest({"now": inp.observation.cameras, "task": t.task, "instr": t.instruction,
                       "params": t.params})

    def spec(self) -> Mapping[str, Any]:
        return {"class": "ObsOnlyLookup", "fit_digest": self._fit_digest}

    def reset(self, episode_seed: int) -> None:
        return None

    def fit(self, examples: Sequence[tuple[PolicyInput, str]], seed: int) -> None:
        self._table = defaultdict(Counter)
        for inp, y in examples:
            self._table[self._key(inp)][y] += 1
        self._fit_digest = digest(sorted((k, sorted(c.items())) for k, c in self._table.items()))

    def infer(self, inp: PolicyInput) -> ActionChunk:
        c = self._table.get(self._key(inp))
        legal = [(n, o) for o, n in (c or {}).items() if o in inp.task.options]
        if legal:
            return ActionChunk((min(legal, key=lambda x: (-x[0], x[1]))[1],))
        return ActionChunk((_guess(inp, []),))


@dataclass
class BagOfTokensNB:
    """Naive Bayes on unordered (key=value) tokens of context + current tick; class = action."""

    name: str = "bag_of_tokens_nb"
    alpha: float = 1.0
    _prior: dict[tuple[str, str], int] = field(default_factory=dict)
    _tok: dict[tuple[str, str], Counter[str]] = field(default_factory=dict)
    _vocab: dict[str, set[str]] = field(default_factory=dict)
    _fit_digest: str | None = None

    @staticmethod
    def _tokens(inp: PolicyInput) -> list[str]:
        toks = [f"now:{c}:{k}={v}" for c, d in inp.observation.cameras.items() for k, v in d.items()]
        for tick in inp.context.ticks:
            toks += [f"hist:{c}:{k}={v}" for c, d in tick.cameras.items() for k, v in d.items()]
        return toks

    def spec(self) -> Mapping[str, Any]:
        return {"class": "BagOfTokensNB", "alpha": self.alpha, "fit_digest": self._fit_digest}

    def reset(self, episode_seed: int) -> None:
        return None

    def fit(self, examples: Sequence[tuple[PolicyInput, str]], seed: int) -> None:
        prior: Counter[tuple[str, str]] = Counter()
        tok: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        vocab: dict[str, set[str]] = defaultdict(set)
        for inp, y in examples:
            prior[(inp.task.task, y)] += 1
            for w in self._tokens(inp):
                tok[(inp.task.task, y)][w] += 1
                vocab[inp.task.task].add(w)
        self._prior, self._tok, self._vocab = dict(prior), dict(tok), dict(vocab)
        self._fit_digest = digest({"prior": sorted(map(list, prior.items())), "n": len(examples)})

    def infer(self, inp: PolicyInput) -> ActionChunk:
        task = inp.task.task
        toks = self._tokens(inp)
        v = max(1, len(self._vocab.get(task, ())))
        best, best_s = None, -math.inf
        for opt in inp.task.options:  # option order is fixed by the task spec -> stable ties
            n_y = self._prior.get((task, opt), 0)
            counts = self._tok.get((task, opt), Counter())
            total = sum(counts.values())
            s = math.log(n_y + self.alpha)
            for w in toks:
                s += math.log((counts.get(w, 0) + self.alpha) / (total + self.alpha * v))
            if s > best_s + 1e-12:
                best, best_s = opt, s
        assert best is not None
        return ActionChunk((best,))
