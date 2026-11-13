# SPDX-License-Identifier: Apache-2.0
"""Learned context readers for G1 (docs/25). CPU PyTorch. Status: ``implemented``; none is a VLA.

One shared featuriser serves training and harness inference: the sequence always spans the full
elapsed clock (n_hist past ticks + current); ticks not in the retained context are an UNK flag
(same semantics as the posterior oracle's unknown ticks). Item names are canonicalised to their
index in the task's item list (in-context binding)."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ucvla.oracle2 import tick_index
from ucvla.policy import ActionChunk, PolicyInput

FRONT = "cam_front"
MAX_OPT = 5
TASKS = ("track_ball", "button_events", "resume_items", "pick_flagged")
PARAM_KEYS = ("n_slots", "sensor_eps", "swap_rate", "target_count", "jam", "p_attempt", "p_hold",
              "fail", "removal")
NAME_KEYS = ("attempt", "removed")


def canon_tick(tick: Mapping[str, str], items: Sequence[str]) -> list[str]:
    out = []
    for k, v in sorted(tick.items()):
        if k in NAME_KEYS and v in items:
            v = f"item{items.index(v)}"
        out.append(f"{k}={v}")
    return out


def param_vec(task: str, params: Mapping[str, str]) -> list[float]:
    vec = [float(params[k]) if k in params else 0.0 for k in PARAM_KEYS]
    vec.append(float(len(params.get("items", "").split(","))) if "items" in params else 0.0)
    vec += [1.0 if task == t else 0.0 for t in TASKS]
    return vec


N_PARAM = len(PARAM_KEYS) + 1 + len(TASKS)


@dataclass
class Vocab:
    tokens: dict[str, int] = field(default_factory=dict)  # 0 = UNKNOWN-TOKEN, 1 = UNKNOWN-TICK

    @classmethod
    def build(cls, token_lists: Sequence[Sequence[str]]) -> Vocab:
        v = cls({})
        for toks in token_lists:
            for t in toks:
                v.tokens.setdefault(t, len(v.tokens) + 2)
        return v

    @property
    def dim(self) -> int:
        return len(self.tokens) + 2


def encode(vocab: Vocab, task: str, params: Mapping[str, str], options: Sequence[str],
           kept: Mapping[int, Mapping[str, str]], current: Mapping[str, str], n_hist: int
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items = params.get("items", "").split(",") if "items" in params else []
    x = np.zeros((n_hist + 1, vocab.dim), dtype=np.float32)
    for i in range(n_hist + 1):
        tick = current if i == n_hist else kept.get(i)
        if tick is None:
            x[i, 1] = 1.0
            continue
        for t in canon_tick(tick, items):
            x[i, vocab.tokens.get(t, 0)] = 1.0
    mask = np.zeros(MAX_OPT, dtype=bool)
    mask[:len(options)] = True
    return x, np.asarray(param_vec(task, params), dtype=np.float32), mask


class _Base(nn.Module):
    def __init__(self, vdim: int, d: int) -> None:
        super().__init__()
        self.inp = nn.Linear(vdim, d)
        self.par = nn.Linear(N_PARAM, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, MAX_OPT))

    def _mask(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits.masked_fill(~mask, -1e9)


class GRUReader(_Base):
    def __init__(self, vdim: int, d: int = 64) -> None:
        super().__init__(vdim, d)
        self.rnn = nn.GRU(d, d, batch_first=True)

    def forward(self, x, lengths, par, mask):
        h, _ = self.rnn(torch.relu(self.inp(x) + self.par(par)[:, None, :]))
        last = h[torch.arange(x.shape[0]), lengths - 1]
        return self._mask(self.head(last), mask)


class CausalTransformerReader(_Base):
    def __init__(self, vdim: int, d: int = 64, layers: int = 2, heads: int = 4,
                 max_len: int = 64) -> None:
        super().__init__(vdim, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x, lengths, par, mask):
        t = x.shape[1]
        z = self.inp(x) + self.par(par)[:, None, :] + self.pos(torch.arange(t))[None]
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
        h = self.enc(z, mask=causal)
        return self._mask(self.head(h[torch.arange(x.shape[0]), lengths - 1]), mask)


class FixedLatentReader(nn.Module):
    """S memory slots of width d_slot; per-tick gated write; capacity = slots * d_slot."""

    def __init__(self, vdim: int, slots: int = 4, d_slot: int = 16, d: int = 64) -> None:
        super().__init__()
        self.slots, self.d_slot = slots, d_slot
        self.inp = nn.Linear(vdim, d)
        self.par = nn.Linear(N_PARAM, d)
        self.gate = nn.Linear(d + d_slot, d_slot)
        self.cand = nn.Linear(d + d_slot, d_slot)
        self.slot_id = nn.Parameter(torch.randn(slots, d) * 0.1)
        self.head = nn.Sequential(nn.Linear(slots * d_slot + d, d), nn.ReLU(), nn.Linear(d, MAX_OPT))

    def forward(self, x, lengths, par, mask):
        b, t, _ = x.shape
        emb = torch.relu(self.inp(x) + self.par(par)[:, None, :])
        m = torch.zeros(b, self.slots, self.d_slot)
        final_m = torch.zeros_like(m)
        final_e = torch.zeros(b, emb.shape[-1])
        for i in range(t):
            e = emb[:, i, :][:, None, :] + self.slot_id[None]
            z = torch.cat([e, m], dim=-1)
            g = torch.sigmoid(self.gate(z))
            m = (1 - g) * m + g * torch.tanh(self.cand(z))
            sel = (lengths - 1 == i)
            final_m = torch.where(sel[:, None, None], m, final_m)
            final_e = torch.where(sel[:, None], emb[:, i, :], final_e)
        return self.head(torch.cat([final_m.flatten(1), final_e], dim=-1)).masked_fill(~mask, -1e9)


def make_model(kind: str, vdim: int, slots: int = 4) -> nn.Module:
    if kind == "gru":
        return GRUReader(vdim)
    if kind == "ctransformer":
        return CausalTransformerReader(vdim)
    if kind == "fixedlatent":
        return FixedLatentReader(vdim, slots=slots)
    raise ValueError(kind)


def pad(batch: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]]):
    t = max(x.shape[0] for x, _, _ in batch)
    xs = np.zeros((len(batch), t, batch[0][0].shape[1]), dtype=np.float32)
    for i, (x, _, _) in enumerate(batch):
        xs[i, :x.shape[0]] = x
    lengths = torch.tensor([x.shape[0] for x, _, _ in batch])
    par = torch.tensor(np.stack([p for _, p, _ in batch]))
    mask = torch.tensor(np.stack([m for _, _, m in batch]))
    return torch.tensor(xs), lengths, par, mask


def train_model(kind: str, vocab: Vocab, train: Sequence[tuple[Any, int]], val: Sequence[tuple[Any, int]],
                val_pairs: Sequence[tuple[int, int]], seed: int, slots: int = 4, epochs: int = 60,
                patience: int = 8, lr: float = 3e-3, batch: int = 64, ckpt: Any = None,
                stop_after: int | None = None) -> tuple[nn.Module, dict]:
    """Fit on TRAIN; early-stop on VAL pair-both-correct. Returns (best model, trace)."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = make_model(kind, vocab.dim, slots)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best, best_state, bad, trace = -1.0, None, 0, []
    start = 0
    if ckpt is not None and ckpt.exists():  # RESUME: parameters, optimiser, RNG, early-stop state, trace
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["np_rng"]
        torch.set_rng_state(c["torch_rng"])
        best, best_state, bad, trace, start = c["best"], c["best_state"], c["bad"], c["trace"], c["epoch"]

    def predict(rows):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(rows), 256):
                out.append(model(*pad([r for r, _ in rows[i:i + 256]])).argmax(-1))
            return torch.cat(out).numpy()

    ytr = np.array([y for _, y in train])
    for ep in range(start, epochs):
        if stop_after is not None and ep >= stop_after:
            raise KeyboardInterrupt("simulated kill")
        model.train()
        order = rng.permutation(len(train))
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            logits = model(*pad([train[j][0] for j in idx]))
            loss = nn.functional.cross_entropy(logits, torch.tensor(ytr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        pv = predict(val)
        yv = np.array([y for _, y in val])
        ok = pv == yv
        pair = float(np.mean([ok[a] and ok[b] for a, b in val_pairs]))
        trace.append({"epoch": ep, "train_loss": total / len(order), "val_pair_optimal": pair,
                      "val_acc": float(ok.mean())})
        if pair > best:
            best, bad = pair, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ckpt is not None:  # atomic checkpoint after every epoch
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "best": best, "best_state": best_state, "bad": bad, "trace": trace,
                        "epoch": ep + 1}, tmp)
            tmp.replace(ckpt)
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    ptr = predict(list(train))
    trace.append({"final_train_acc": float((ptr == ytr).mean()), "best_val_pair_optimal": best,
                  "epochs_run": len(trace)})
    return model, {"trace": trace}


@dataclass
class LearnedPolicy:
    model: nn.Module
    vocab: Vocab
    label: str

    @property
    def name(self) -> str:
        return self.label

    def spec(self) -> Mapping[str, Any]:
        n = sum(p.numel() for p in self.model.parameters())
        return {"class": type(self.model).__name__, "params": n, "vocab": self.vocab.dim,
                "role": "learned reader trained on train split only (docs/25)"}

    def reset(self, episode_seed: int) -> None:
        return None

    def infer(self, inp: PolicyInput) -> ActionChunk:
        t = inp.task
        n_hist = tick_index(inp.observation.time_ns)
        kept = {tick_index(k.time_ns): k.cameras[FRONT] for k in inp.context.ticks}
        x, par, mask = encode(self.vocab, t.task, t.params, t.options, kept,
                              inp.observation.cameras[FRONT], n_hist)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(*pad([(x, par, mask)]))
        return ActionChunk((t.options[int(logits.argmax(-1))],))
