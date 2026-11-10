import dataclasses

import pytest

from ucvla.baselines import full_history, long_window, memoryless, uniform_subsample
from ucvla.causal_eval import access_parity_digest
from ucvla.context import (
    AccessViolation,
    AllHistory,
    ContextBudget,
    LastK,
    NoHistory,
    UniformK,
    ViewTick,
    build_context,
)
from ucvla.harness import (
    Agent,
    decide,
    evaluate,
    fit_agent,
    make_input,
    pair_metrics,
    run_episode,
)
from ucvla.policy import ActionChunk, PolicyInput
from ucvla.readers import BagOfTokensNB, ObsOnlyLookup, OracleDecoder

TEST_SPLITS = ("test_id", "test_ood", "test_counterfactual")


def test_harness_reproduces_m1_symbolic_policies_exactly(adapter):
    """Refactor equivalence: provider+oracle decoder == M1 SymbolicPolicy for every episode."""
    pairs = [
        (Agent(NoHistory(), OracleDecoder("clock")), memoryless()),
        (Agent(AllHistory(), OracleDecoder("clock")), full_history()),
        *[(Agent(LastK(k), OracleDecoder("clock")), long_window(k)) for k in (1, 3, 6)],
        *[(Agent(UniformK(k), OracleDecoder("clock")), uniform_subsample(k)) for k in (1, 3, 6)],
    ]
    for split in ("train", *TEST_SPLITS):
        for agent, m1 in pairs:
            for p in adapter.pairs(split):
                for eid in (p.episode_a, p.episode_b):
                    view = adapter.view(eid)
                    assert decide(agent, view, 0, eid, None)[0] == m1.decide(view)


class FabricatingProvider:
    name = "fab"
    budget = ContextBudget("ticks", None)

    def __init__(self, mode):
        self.mode = mode

    def select(self, history):
        t = history[0]
        if self.mode == "edited":
            cams = {c: {**d, "ball": "under:FORGED"} for c, d in t.cameras.items()}
            return (ViewTick(t.time_ns, cams), *history[1:])
        if self.mode == "future":
            return (*history, ViewTick(history[-1].time_ns + 10**12, t.cameras))
        if self.mode == "duplicate":
            return (t, t)
        if self.mode == "unordered":
            return tuple(reversed(history))
        if self.mode == "overbudget":
            return history
        raise AssertionError


@pytest.mark.parametrize("mode", ["edited", "future", "duplicate", "unordered"])
def test_harness_rejects_illegal_contexts(adapter, mode):
    view = adapter.view(adapter.pairs("train")[0].episode_a)
    with pytest.raises(AccessViolation):
        make_input(view, FabricatingProvider(mode), None)


def test_provider_cannot_exceed_declared_or_matched_budget(adapter):
    view = adapter.view(adapter.pairs("train")[0].episode_a)
    liar = dataclasses.replace(LastK(2))
    liar_provider = type("Liar", (), {"name": "liar", "budget": ContextBudget("ticks", 2),
                                      "select": lambda self, h: h})()
    with pytest.raises(AccessViolation, match="own declared"):
        build_context(liar_provider, view.history, view.current.time_ns)
    with pytest.raises(AccessViolation, match="matched budget"):
        build_context(LastK(8), view.history, view.current.time_ns, ContextBudget("ticks", 4))
    assert len(build_context(liar, view.history, view.current.time_ns).ticks) <= 2


def test_matched_budgets_are_measured_by_harness_not_self_reported(adapter):
    for k in (1, 2, 4):
        for provider in (LastK(k), UniformK(k)):
            res = evaluate(adapter, Agent(provider, OracleDecoder()), "test_ood", 0,
                           ContextBudget("ticks", k))
            assert max(r.ticks_retained for r in res) <= k
    tok = evaluate(adapter, Agent(LastK(6, unit="tokens"), OracleDecoder()), "test_id", 0,
                   ContextBudget("tokens", 6))
    assert max(r.tokens_retained for r in tok) <= 6


def test_every_agent_receives_identical_non_context_inputs(adapter):
    agents = [Agent(NoHistory(), OracleDecoder()), Agent(AllHistory(), OracleDecoder()),
              Agent(LastK(3), BagOfTokensNB()), Agent(UniformK(3), ObsOnlyLookup())]
    for split in ("train", *TEST_SPLITS):
        digests = access_parity_digest(adapter, agents, split)
        assert len(set(digests.values())) == 1


def test_policy_input_has_no_label_privileged_or_identity_fields(adapter):
    inp = make_input(adapter.view(adapter.pairs("train")[0].episode_a), AllHistory(), None)[0]
    names = set()

    def walk(obj):
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                names.add(f.name)
                walk(getattr(obj, f.name))

    walk(inp)
    assert not {n for n in names if any(
        w in n for w in ("target", "privileged", "provenance", "episode_id", "pair", "query_id",
                         "annotation", "label"))}
    assert isinstance(inp, PolicyInput)


def test_decisions_do_not_depend_on_episode_identity_or_seed(adapter):
    agent = Agent(UniformK(4), OracleDecoder())
    for p in adapter.pairs("test_id")[:12]:
        view = adapter.view(p.episode_a)
        d = {decide(agent, view, seed, eid, None)[0]
             for seed, eid in ((0, "x"), (123, "y"), (7, p.episode_a))}
        assert len(d) == 1


def test_illegal_or_non_predicted_actions_rejected(adapter):
    class Bad:
        name = "bad"

        def spec(self):
            return {}

        def reset(self, s):
            pass

        def infer(self, inp):
            return ActionChunk(("teleport",))

    ep = adapter.episode(adapter.pairs("train")[0].episode_a)
    p = adapter.pairs("train")[0]
    with pytest.raises(ValueError, match="illegal action"):
        run_episode(ep, p, Agent(NoHistory(), Bad()), 0, None)
    with pytest.raises(ValueError):
        ActionChunk(("x",), kind="executed")
    with pytest.raises(ValueError):
        ActionChunk(())


def test_predicted_commanded_executed_are_logged_separately(adapter):
    p = adapter.pairs("test_id")[0]
    r = run_episode(adapter.episode(p.episode_a), p, Agent(AllHistory(), OracleDecoder()), 0, None)
    assert r.commanded.action == r.decision == r.executed.measured_action
    assert r.executed.verified and r.executed.executed_at_ns > r.commanded.issued_at_ns
    assert r.commanded.command_id == r.executed.command_id


def test_fitting_only_on_train_and_fit_digest_is_train_ids(adapter):
    agent = Agent(NoHistory(), ObsOnlyLookup())
    with pytest.raises(PermissionError):
        fit_agent(adapter, agent, 0, split="test_id")
    with pytest.raises(PermissionError):
        fit_agent(adapter, agent, 0, split="val")
    d = fit_agent(adapter, agent, 0, "train")
    train_ids = sorted(e for p in adapter.pairs("train") for e in (p.episode_a, p.episode_b))
    from ucvla.canonical import digest
    assert d == digest(train_ids)
    assert agent.policy.spec()["fit_digest"]
    assert fit_agent(adapter, Agent(NoHistory(), OracleDecoder()), 0) == "not-trainable"


def test_obs_only_policies_cap_at_zero_pair_success_on_relevant_families(adapter):
    for reader in (ObsOnlyLookup(), OracleDecoder()):
        agent = Agent(NoHistory(), reader)
        fit_agent(adapter, agent, 0)
        for split in TEST_SPLITS:
            for fam, m in pair_metrics(evaluate(adapter, agent, split, 0)).items():
                if fam != "irrelevant_history":
                    assert m["pair_both_correct"] == 0.0


def test_generic_learner_does_not_solve_relational_tasks(adapter):
    """The decoder's task knowledge matters: a knowledge-free learner stays far from 1.0."""
    agent = Agent(AllHistory(), BagOfTokensNB())
    fit_agent(adapter, agent, 0)
    m = pair_metrics(evaluate(adapter, agent, "test_ood", 0))
    assert m["object_correspondence"]["pair_both_correct"] < 0.5
    assert m["task_interleave"]["pair_both_correct"] < 0.5


def test_determinism_under_reset(adapter):
    agent = Agent(UniformK(4), OracleDecoder())
    p = adapter.pairs("test_ood")[0]
    r1 = run_episode(adapter.episode(p.episode_a), p, agent, 3, None)
    r2 = run_episode(adapter.episode(p.episode_a), p, agent, 3, None)
    assert r1 == r2
