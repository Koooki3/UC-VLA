# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from ucvla.vla_mock import MockConfig, MockFlowVLA


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    m = MockFlowVLA().eval()
    c = m.cfg
    b = 3
    return m, torch.randn(b, c.n_img, 32), torch.randn(b, c.n_lang, 32), torch.randn(b, c.state_dim)


def test_tensor_contract_and_determinism(setup):
    m, img, lang, st = setup
    a = m.act(img, lang, st, torch.randn(3, m.cfg.ctx_state_dim), seed=1)
    assert a.shape == (3, m.cfg.horizon, m.cfg.action_dim) and a.dtype == torch.float32
    assert not torch.equal(a, m.act(img, lang, st, None, seed=1))  # context presence changes the chunk
    ctx = torch.randn(3, m.cfg.ctx_state_dim)
    assert torch.equal(m.act(img, lang, st, ctx, seed=7), m.act(img, lang, st, ctx, seed=7))


def test_context_changes_actions_and_version_swap_changes_them_again(setup):
    m, img, lang, st = setup
    c1, c2 = torch.randn(3, m.cfg.ctx_state_dim), torch.randn(3, m.cfg.ctx_state_dim)
    base = m.act(img, lang, st, None, seed=2)
    a1, a2 = m.act(img, lang, st, c1, seed=2), m.act(img, lang, st, c2, seed=2)
    assert (a1 - base).abs().max() > 1e-4  # context tokens matter
    assert (a1 - a2).abs().max() > 1e-4  # a revised memory state yields a different chunk


def test_fixed_token_budget_independent_of_history_length(setup):
    m, img, lang, st = setup
    n_prefix = m.prefix(img, lang, st, torch.zeros(3, m.cfg.ctx_state_dim)).shape[1]
    assert n_prefix == m.cfg.n_img + m.cfg.n_lang + m.cfg.n_ctx + 1  # K context tokens, however long the history
    assert MockConfig().n_ctx == 4
