# SPDX-License-Identifier: Apache-2.0
"""Resume equivalence: killing a run mid-training and restarting reproduces the uninterrupted run
bit-for-bit (precondition for MLDA/ROSE use, docs/31)."""
import numpy as np
import pytest
import torch

from ucvla.learned import Vocab, encode, train_model


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    vocab = Vocab.build([["a=0", "a=1", "b=0"]])
    rows = []
    for _ in range(n):
        bit = int(rng.integers(0, 2))
        kept = {0: {"a": str(bit)}, 1: {"b": "0"}}
        x, p, m = encode(vocab, "pick_flagged", {}, ("o0", "o1"), kept, {"b": "0"}, 2)
        rows.append(((x, p, m), bit))
    return vocab, rows


@pytest.mark.parametrize("kind", ["gru", "fixedlatent"])
def test_killed_and_resumed_run_equals_uninterrupted(tmp_path, kind):
    vocab, rows = toy()
    pairs = [(2 * i, 2 * i + 1) for i in range(len(rows[:20]) // 2)]
    full, _ = train_model(kind, vocab, rows[:40], rows[40:], pairs, 3, epochs=6, patience=99)
    ck = tmp_path / "c.pt"
    with pytest.raises(KeyboardInterrupt):
        train_model(kind, vocab, rows[:40], rows[40:], pairs, 3, epochs=6, patience=99, ckpt=ck,
                    stop_after=3)
    assert ck.exists()
    resumed, info = train_model(kind, vocab, rows[:40], rows[40:], pairs, 3, epochs=6, patience=99,
                                ckpt=ck)
    for a, b in zip(full.state_dict().values(), resumed.state_dict().values()):
        assert torch.equal(a, b)
    assert len([t for t in info["trace"] if "epoch" in t and "train_loss" in t]) == 6
