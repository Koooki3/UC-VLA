# SPDX-License-Identifier: Apache-2.0
"""Stateful streaming interface for the recurrent readers (status: GRU and fixed-latent; the causal
Transformer is NOT streamable here). The persistent state is the ONLY memory: each update consumes one
new tick (plus the static task parameters) and never re-reads history. Exactness against full-history
recompute, segmentation and checkpoint/resume equivalence are tested (tests/test_streaming.py).

Cost accounting separates: hot-memory size (state floats), cumulative history read (ticks consumed),
per-update multiply-accumulates (MACs, analytic), and the full-recompute alternative."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ucvla.learned import FixedLatentReader, GRUReader


@dataclass
class Cost:
    ticks_consumed: int = 0
    macs: int = 0


class StreamingReader:
    def __init__(self, model: torch.nn.Module) -> None:
        if not isinstance(model, (GRUReader, FixedLatentReader)):
            raise TypeError("only GRUReader and FixedLatentReader are streamable")
        self.model = model.eval()
        self.state: Any = None
        self.par: torch.Tensor | None = None
        self.cost = Cost()

    # --- sizes -----------------------------------------------------------------------------
    def hot_memory_floats(self) -> int:
        m = self.model
        if isinstance(m, GRUReader):
            return m.rnn.hidden_size
        return m.slots * m.d_slot

    def macs_per_update(self) -> int:
        m = self.model
        if isinstance(m, GRUReader):
            d, v = m.rnn.hidden_size, m.inp.in_features
            return v * d + 3 * (d * d + d * d)  # input projection + GRU gates
        d, v = m.inp.out_features, m.inp.in_features
        return v * d + m.slots * ((d + m.d_slot) * m.d_slot * 2)

    # --- API -------------------------------------------------------------------------------
    def reset(self, par: torch.Tensor) -> None:
        self.par = par[None]
        m = self.model
        if isinstance(m, GRUReader):
            self.state = torch.zeros(1, 1, m.rnn.hidden_size)
        else:
            self.state = torch.zeros(1, m.slots, m.d_slot)
        self.cost = Cost()

    @torch.no_grad()
    def update(self, x_t: torch.Tensor) -> torch.Tensor:
        """Consume ONE tick feature vector (V,). Returns the embedding of that tick."""
        m = self.model
        emb = torch.relu(m.inp(x_t[None, None, :]) + m.par(self.par)[:, None, :])  # (1,1,d)
        if isinstance(m, GRUReader):
            _, self.state = m.rnn(emb, self.state)
        else:
            e = emb[:, 0, :][:, None, :] + m.slot_id[None]
            z = torch.cat([e, self.state], dim=-1)
            g = torch.sigmoid(m.gate(z))
            self.state = (1 - g) * self.state + g * torch.tanh(m.cand(z))
        self.cost.ticks_consumed += 1
        self.cost.macs += self.macs_per_update()
        self._last_emb = emb[:, 0, :]
        return emb

    @torch.no_grad()
    def decide(self, mask: torch.Tensor) -> torch.Tensor:
        """Logits after the CURRENT tick has been consumed (state + last tick embedding)."""
        m = self.model
        if isinstance(m, GRUReader):
            return m.head(self.state[0]).masked_fill(~mask[None], -1e9)
        return m.head(torch.cat([self.state.flatten(1), self._last_emb], dim=-1)).masked_fill(~mask[None], -1e9)

    def snapshot(self) -> dict:
        return {"state": self.state.clone(), "par": self.par.clone(), "last": self._last_emb.clone(),
                "ticks": self.cost.ticks_consumed, "macs": self.cost.macs}

    def restore(self, snap: dict) -> None:
        self.state, self.par, self._last_emb = snap["state"].clone(), snap["par"].clone(), snap["last"].clone()
        self.cost = Cost(snap["ticks"], snap["macs"])

    def full_recompute_macs(self, n_ticks: int) -> int:
        """MACs if every decision re-read all ticks (what the offline reader does)."""
        return n_ticks * self.macs_per_update()
