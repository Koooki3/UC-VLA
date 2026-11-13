# SPDX-License-Identifier: Apache-2.0
"""Policies for SODH v2 behind the PolicyAdapter (status: ``mock``; none is a VLA).

PosteriorOraclePolicy   Bayes-optimal under the declared environment model, restricted to the
                        retained context (dropped ticks = unknown). Independent of every decoder
                        below; it is the measuring instrument for Memory Sufficiency.
HeuristicDecoder        hand-written readers; variants include deliberately natural-but-wrong
                        ones used as falsifiers ("edge_count", "last_only").
Knowledge-free learners with identical access: readers.BagOfTokensNB / ObsOnlyLookup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ucvla.oracle2 import optimal, posterior, tick_index
from ucvla.policy import ActionChunk, PolicyInput
from ucvla.readers import _guess

FRONT = "cam_front"
HEURISTIC_VARIANTS = ("naive", "conservative", "edge_count", "last_only")


def _kept(inp: PolicyInput) -> dict[int, Mapping[str, str]]:
    return {tick_index(t.time_ns): t.cameras[FRONT] for t in inp.context.ticks}


@dataclass
class PosteriorOraclePolicy:
    name: str = "posterior_oracle"

    def spec(self) -> Mapping[str, Any]:
        return {"class": "PosteriorOraclePolicy", "role": "Bayes-optimal under the env model, "
                "restricted to the retained context; instrument for Memory Sufficiency"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        t = inp.task
        post = posterior(t.task, t.params, t.options, _kept(inp), inp.observation.cameras[FRONT],
                         inp.observation.time_ns)
        return ActionChunk((optimal(post, t.options)[0],))


@dataclass
class HeuristicDecoder:
    variant: str = "naive"

    def __post_init__(self) -> None:
        if self.variant not in HEURISTIC_VARIANTS:
            raise ValueError(f"variant must be one of {HEURISTIC_VARIANTS}")

    @property
    def name(self) -> str:
        return f"heuristic_{self.variant}"

    def spec(self) -> Mapping[str, Any]:
        return {"class": "HeuristicDecoder", "variant": self.variant,
                "role": "hand-written decoder; 'edge_count'/'last_only' are natural-but-wrong falsifiers"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        t, v = inp.task, self.variant
        now = inp.observation.cameras[FRONT]
        kept = _kept(inp)
        order = sorted(kept)
        front = [kept[i] for i in order]
        guess = _guess(inp, [k.cameras for k in inp.context.ticks])
        n_hist = tick_index(inp.observation.time_ns)
        gap = len(order) < n_hist
        if t.task == "track_ball":
            n = int(t.params["n_slots"])
            rev = next((i for i in order if kept[i].get("reveal", "none").startswith("slot:")), None)
            if rev is None:
                return ActionChunk((guess,))
            pos = int(kept[rev]["reveal"][5:])
            unseen = False
            for i in range(rev, n_hist):
                tick = kept.get(i)
                sw = tick.get("swap", "none") if tick else "?"
                if sw == "?":
                    unseen = True
                elif sw != "none":
                    a, b = (int(x) for x in sw.split("-"))
                    pos = b if pos == a else a if pos == b else pos
            out = guess if (v == "conservative" and unseen) else f"pick_slot_{min(pos, n - 1)}"
            return ActionChunk((out,))
        if t.task == "button_events":
            k_target = int(t.params["target_count"])
            if v == "edge_count":  # ignores the lamp: counts contact rising edges (wrong on jams)
                prev, count = "up", 0
                for f in [*front, now]:
                    count += prev == "up" and f["contact"] == "down"
                    prev = f["contact"] if f["contact"] != "?" else prev
            else:
                count = sum(f["lamp"] == "on" for f in front)
            if v == "conservative" and (gap or any(f["lamp"] == "?" for f in front)):
                return ActionChunk((guess,))
            return ActionChunk(("press" if count < k_target else "stop",))
        if t.task == "resume_items":
            items = t.params["items"].split(",")
            placed = {a for f in front if f.get("result") == "ok" and (a := f["attempt"]) in items}
            removed = {f["removed"] for f in front if f.get("removed") in items}
            if v == "last_only":  # only the most recent successful attempt counts as placed
                oks = [f["attempt"] for f in front if f.get("result") == "ok"]
                placed = {oks[-1]} if oks else set()
            elif v != "conservative" or not gap:
                placed -= {r for r in removed if v == "naive" and r in placed}
            elif gap:
                return ActionChunk((guess,))
            free = [i for i in items if i not in placed] or items
            return ActionChunk((f"place_{free[0]}",))
        if t.task == "pick_flagged":
            return ActionChunk((f"pick_{now['flag']}",))
        raise ValueError(f"no heuristic for task {t.task!r}")
