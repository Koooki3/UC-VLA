# SPDX-License-Identifier: Apache-2.0
"""Symbolic reference readers for the SODH suite (status: ``mock`` — hand-written, not learned).

All policies share ONE decoding rule per task and differ only in which history ticks they are
allowed to retain (the hot-memory budget, counted in ticks). That isolates the memory-selection
question and makes budget comparisons fair by construction. They are *not* the M2 baselines
(fixed-latent, D&R, Mem4Act on a VLA); they exist to validate the data (gate G0) and to give a
controlled lower/upper reference. A policy sees only a ``CausalView``; labels are unreachable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ucvla.canonical import digest
from ucvla.episode import CausalView, ViewTick

FRONT = "cam_front"


class ContextPolicy(Protocol):
    name: str

    def decide(self, view: CausalView) -> str: ...

    def retained(self, view: CausalView) -> int: ...


def select_history(history: tuple[ViewTick, ...], selector: str, k: int | None) -> tuple[ViewTick, ...]:
    n = len(history)
    if selector == "all":
        return history
    if selector == "none":
        return ()
    if k is None or k < 0:
        raise ValueError("selector needs a non-negative budget k")
    if selector == "last_k":
        return history[n - min(k, n):]
    if selector == "uniform_k":
        if k >= n:
            return history
        return tuple(history[(2 * i + 1) * n // (2 * k)] for i in range(k))
    raise ValueError(f"unknown selector {selector!r}")


def _guess(view: CausalView, retained: tuple[ViewTick, ...]) -> str:
    """Deterministic function of exactly what the policy retained (never of ids or labels)."""
    q = view.query
    assert q is not None
    h = digest({"kept": [t.cameras for t in retained], "now": view.current.cameras,
                "task": q.task, "params": q.params})
    return q.options[int(h[:8], 16) % len(q.options)]


def _read(view: CausalView, retained: tuple[ViewTick, ...]) -> str:
    q = view.query
    if q is None:
        raise ValueError("view has no decision query")
    total = len(view.history)
    complete = len(retained) == total
    now = view.current.cameras[FRONT]
    front = [t.cameras[FRONT] for t in retained]
    if q.task == "cups_ball":
        cups = [f["ball"][6:] for f in front if f["ball"].startswith("under:")]
        slots = {v: k for k, v in now.items() if k.startswith("slot")}
        if cups and cups[0] in slots:
            return f"pick_{slots[cups[0]].replace('slot', 'slot_')}"
        return _guess(view, retained)
    if q.task == "button_count":
        k = int(q.params["target_count"])
        if not retained:
            return "press" if total == 0 else _guess(view, retained)
        pressed = sum(1 for f in front if f["button"] == "pressed")
        n_hat = round(pressed * total / len(retained))  # scale-up when history was subsampled
        return "press" if n_hat < k else "stop"
    if q.task == "resume_sequence":
        order = q.params["phase_order"].split(",")
        done = [f["placing"] for f in front if f["placing"] != "none"]
        if done:
            i = order.index(done[-1])
            return f"place_{order[i + 1]}" if i + 1 < len(order) else _guess(view, retained)
        return f"place_{order[0]}" if complete else _guess(view, retained)
    if q.task == "pick_flagged":
        return f"pick_{now['flag']}"
    raise ValueError(f"no reader for task {q.task!r}")


@dataclass(frozen=True)
class SymbolicPolicy:
    name: str
    selector: str
    k: int | None = None

    def kept(self, view: CausalView) -> tuple[ViewTick, ...]:
        return select_history(view.history, self.selector, self.k)

    def retained(self, view: CausalView) -> int:
        return len(self.kept(view))

    def decide(self, view: CausalView) -> str:
        return _read(view, self.kept(view))


def memoryless() -> SymbolicPolicy:
    return SymbolicPolicy("memoryless", "none")


def full_history() -> SymbolicPolicy:
    return SymbolicPolicy("full_history", "all")


def long_window(k: int) -> SymbolicPolicy:
    return SymbolicPolicy(f"long_window_{k}", "last_k", k)


def uniform_subsample(k: int) -> SymbolicPolicy:
    return SymbolicPolicy(f"uniform_{k}", "uniform_k", k)
