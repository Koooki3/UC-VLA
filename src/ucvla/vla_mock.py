# SPDX-License-Identifier: Apache-2.0
"""CPU mock of a flow-matching VLA with a context-token injection point (status: MOCK, random weights,
NO pretrained weights, NOT SmolVLA). Tensor contract follows what the public SmolVLA code is reported
to use (docs/30, unverified): prefix = [image tokens, language tokens, CONTEXT tokens, state token];
an action expert cross-attends the prefix and denoises an action chunk (B, H, A) with Euler steps.
The point of the mock is the INTERFACE: a fixed-size memory state is projected to K context tokens
whose presence/absence/version demonstrably changes the action chunk, with no backbone change."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MockConfig:
    d: int = 64
    n_img: int = 16
    n_lang: int = 8
    n_ctx: int = 4  # fixed number of context tokens K
    state_dim: int = 6
    action_dim: int = 6
    horizon: int = 10
    steps: int = 5
    ctx_state_dim: int = 64  # size of the persistent memory state fed to the projector


class MockFlowVLA(nn.Module):
    def __init__(self, cfg: MockConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or MockConfig()
        self.cfg = cfg
        d = cfg.d
        self.img = nn.Linear(32, d)
        self.lang = nn.Linear(32, d)
        self.state = nn.Linear(cfg.state_dim, d)
        self.ctx_proj = nn.Linear(cfg.ctx_state_dim, cfg.n_ctx * d)  # memory state -> K context tokens
        self.prefix_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.0, batch_first=True), 1)
        self.act_in = nn.Linear(cfg.action_dim + 1, d)
        self.cross = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Linear(d, cfg.action_dim)

    def prefix(self, img: torch.Tensor, lang: torch.Tensor, state: torch.Tensor,
               ctx_state: torch.Tensor | None) -> torch.Tensor:
        c = self.cfg
        parts = [self.img(img), self.lang(lang)]
        if ctx_state is not None:  # context tokens sit BEFORE the state token
            parts.append(self.ctx_proj(ctx_state).view(-1, c.n_ctx, c.d))
        parts.append(self.state(state)[:, None, :])
        return self.prefix_enc(torch.cat(parts, dim=1))

    @torch.no_grad()
    def act(self, img, lang, state, ctx_state=None, seed: int = 0) -> torch.Tensor:
        """Action chunk (B, H, A) by Euler integration of a velocity field from seeded noise."""
        c = self.cfg
        pre = self.prefix(img, lang, state, ctx_state)
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(img.shape[0], c.horizon, c.action_dim, generator=g)
        for k in range(c.steps):
            t = torch.full((a.shape[0], c.horizon, 1), k / c.steps)
            h = self.act_in(torch.cat([a, t], dim=-1))
            v, _ = self.cross(h, pre, pre)
            a = a + self.out(v) / c.steps
        return a
