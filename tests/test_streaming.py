# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from ucvla.learned import FixedLatentReader, GRUReader, Vocab, encode, pad
from ucvla.streaming import StreamingReader

V = Vocab.build([[f"k{i}=v" for i in range(12)]])


def episode(n, seed):
    rng = np.random.default_rng(seed)
    toks = lambda: [f"k{int(rng.integers(0, 12))}=v" for _ in range(3)]
    kept = {i: {t.split("=")[0]: "v" for t in toks()} for i in range(n)}
    cur = {t.split("=")[0]: "v" for t in toks()}
    return encode(V, "button_events", {"target_count": "3", "jam": "0.25"}, ("press", "stop"), kept, cur, n)


@pytest.mark.parametrize("make", [lambda v: GRUReader(v), lambda v: FixedLatentReader(v, slots=4)])
def test_streaming_equals_full_recompute(make):
    torch.manual_seed(0)
    model = make(V.dim).eval()
    for n in (1, 5, 17):
        x, par, mask = episode(n, n)
        full = model(*pad([(x, par, mask)]))
        s = StreamingReader(model)
        s.reset(torch.tensor(par))
        for t in range(x.shape[0]):
            s.update(torch.tensor(x[t]))
        assert torch.allclose(s.decide(torch.tensor(mask)), full, atol=1e-5)


@pytest.mark.parametrize("make", [lambda v: GRUReader(v), lambda v: FixedLatentReader(v, slots=2)])
def test_segmented_execution_and_checkpoint_resume_are_identical(make, tmp_path):
    torch.manual_seed(1)
    model = make(V.dim).eval()
    x, par, mask = episode(12, 3)
    a = StreamingReader(model)
    a.reset(torch.tensor(par))
    for t in range(x.shape[0]):
        a.update(torch.tensor(x[t]))
    ref = a.decide(torch.tensor(mask))
    b = StreamingReader(model)
    b.reset(torch.tensor(par))
    for t in range(6):
        b.update(torch.tensor(x[t]))
    torch.save(b.snapshot(), tmp_path / "snap.pt")  # process 'dies' here
    c = StreamingReader(model)
    c.restore(torch.load(tmp_path / "snap.pt"))
    for t in range(6, x.shape[0]):
        c.update(torch.tensor(x[t]))
    assert torch.equal(c.decide(torch.tensor(mask)), ref)
    assert c.cost.ticks_consumed == x.shape[0]


def test_cost_accounting_streaming_reads_each_tick_once():
    model = FixedLatentReader(V.dim, slots=4).eval()
    s = StreamingReader(model)
    x, par, _ = episode(30, 5)
    s.reset(torch.tensor(par))
    for t in range(x.shape[0]):
        s.update(torch.tensor(x[t]))
    # per-decision streaming cost is one update; recomputing from scratch at each of the 31 steps
    # would cost sum_{k=1..31} k updates
    recompute_total = sum(s.full_recompute_macs(k) for k in range(1, x.shape[0] + 1))
    assert s.cost.macs == x.shape[0] * s.macs_per_update() < recompute_total / 10
    assert s.hot_memory_floats() == 4 * model.d_slot


def test_transformer_is_rejected():
    from ucvla.learned import CausalTransformerReader
    with pytest.raises(TypeError):
        StreamingReader(CausalTransformerReader(V.dim))
