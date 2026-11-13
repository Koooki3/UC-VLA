#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 minimal LEARNED experiment (CPU): which training objective makes a capacity-limited recurrent
memory keep the evidence that a LATER decision needs? (docs/27; pre-declared in docs/44.)

History: M=8 events shown in random order (id one-hot + bit) then K blank ticks. Current decision y_now =
parity of bits in N; later decision y_fut = parity of bits in F. Memory = GRU with hidden size d.
Objectives for stage 1 (state encoder): `now` (CE on y_now only), `recon` (predict all M bits),
`lookahead` (CE on y_now and y_fut -- a whole-episode objective, NOT the intervention-based utility).
Stage 2: encoder frozen, a linear probe for y_fut trained on only 200 labelled sequences (limited
supervision). Reports held-out probe accuracy of y_fut and accuracy of y_now. 5 seeds."""
import json
import statistics as st
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
M, K = 8, 4
PROBE = sys.argv[1] if len(sys.argv) > 1 else "mlp"  # 'linear' = first run (XOR not linearly decodable: confound)
REGIMES = {"A_future_only": ((), (0, 1)), "B_aligned": ((0, 1), (0, 1)), "C_now_other_bit": ((2,), (0, 1))}


def make(n, seed):
    g = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n, M), generator=g)
    order = torch.stack([torch.randperm(M, generator=g) for _ in range(n)])
    x = torch.zeros(n, M + K, M + 1)
    for t in range(M):
        ids = order[:, t]
        x[torch.arange(n), t, ids] = 1.0
        x[torch.arange(n), t, M] = bits[torch.arange(n), ids].float()
    return x, bits


def labels(bits, now, fut):
    yn = bits[:, list(now)].sum(1) % 2 if now else torch.zeros(len(bits), dtype=torch.long)
    yf = bits[:, list(fut)].sum(1) % 2
    return yn, yf


class Enc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.rnn = nn.GRU(M + 1, d, batch_first=True)
        self.now, self.fut, self.rec = nn.Linear(d, 2), nn.Linear(d, 2), nn.Linear(d, M)

    def state(self, x):
        h, _ = self.rnn(x)
        return h[:, -1]


def train(obj, d, now, fut, seed, n=6000, epochs=40):
    torch.manual_seed(seed)
    x, bits = make(n, seed)
    yn, yf = labels(bits, now, fut)
    m = Enc(d)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            s = m.state(x[idx])
            if obj == "now":
                loss = nn.functional.cross_entropy(m.now(s), yn[idx])
            elif obj == "recon":
                loss = nn.functional.binary_cross_entropy_with_logits(m.rec(s), bits[idx].float())
            else:  # lookahead
                loss = nn.functional.cross_entropy(m.now(s), yn[idx]) + nn.functional.cross_entropy(m.fut(s), yf[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def evaluate(m, obj, d, now, fut, seed):
    xt, bt = make(2000, 10_000 + seed)
    ynt, yft = labels(bt, now, fut)
    xp, bp = make(200, 20_000 + seed)  # limited-supervision probe set
    _, yfp = labels(bp, now, fut)
    with torch.no_grad():
        st_t, st_p = m.state(xt), m.state(xp)
        acc_now = (m.now(st_t).argmax(1) == ynt).float().mean().item()
        if obj == "lookahead":
            acc_fut_head = (m.fut(st_t).argmax(1) == yft).float().mean().item()
        else:
            acc_fut_head = None
    probe = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2)) if PROBE == "mlp" else nn.Linear(d, 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(600 if PROBE == "mlp" else 300):
        loss = nn.functional.cross_entropy(probe(st_p.detach()), yfp)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_probe = (probe(st_t).argmax(1) == yft).float().mean().item()
    return {"acc_now": acc_now, "acc_future_probe200": acc_probe, "acc_future_trained_head": acc_fut_head}


def main():
    out = {"mode": "synthetic_learned_toy", "M": M, "K": K, "seeds": 5, "rows": []}
    for reg, (now, fut) in REGIMES.items():
        for d in (2, 4, 8):
            for obj in ("now", "recon", "lookahead"):
                rs = [evaluate(train(obj, d, now, fut, s), obj, d, now, fut, s) for s in range(5)]
                row = {"regime": reg, "capacity_d": d, "objective": obj}
                for k in ("acc_now", "acc_future_probe200"):
                    v = [r[k] for r in rs]
                    row[k] = {"mean": st.mean(v), "min": min(v), "max": max(v)}
                out["rows"].append(row)
                print(reg, d, obj, round(row["acc_now"]["mean"], 2), round(row["acc_future_probe200"]["mean"], 2), flush=True)
    (ROOT / f"results/synthetic/g2_learned{'' if PROBE == 'mlp' else '_linear_probe'}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
