"""SODH v2 acceptance tests (docs/19 §9). Every F-criterion has a test that fails on a defective
design. Small config for speed; the full frozen data is covered by tests/test_frozen_v2.py."""

import dataclasses
from collections import Counter

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import ROOT, small_config2
from tests.test_g0_mutants import MUTANTS, mutated
from ucvla.adapters import SODHAdapter
from ucvla.audit2 import (
    front_ticks,
    g0_v2_report,
    oracle_env_findings,
    position_findings,
)
from ucvla.canonical import canonical_json
from ucvla.causal_eval import causal_attribution, uses_history
from ucvla.context import AllHistory, LastK, NoHistory
from ucvla.harness import Agent, evaluate, fit_agent, pair_metrics
from ucvla.oracle2 import posterior
from ucvla.readers import BagOfTokensNB
from ucvla.readers2 import HeuristicDecoder, PosteriorOraclePolicy
from ucvla.sodh import TICK_NS
from ucvla.sodh2 import (
    RELEVANT2,
    SODH2Config,
    assign_splits2,
    generate_suite2,
    oracle_for_ticks,
)

RELEVANT = sorted(RELEVANT2)


def test_deterministic_and_unique(suite2, cfg2):
    again = generate_suite2(cfg2)
    blob = lambda s: "".join(canonical_json(e) for e in s.episodes)
    assert blob(again) == blob(suite2)
    assert len({p.template_id for p in suite2.pairs}) == len(suite2.pairs)
    assert len({e.content_sha256() for e in suite2.episodes}) == len(suite2.episodes)
    assert blob(generate_suite2(small_config2(master_seed=6))) != blob(suite2)


def test_pairs_share_whole_decision_view_and_differ_in_oracle_action(adapter2, suite2):
    for p in suite2.pairs:
        va, vb = adapter2.view(p.episode_a), adapter2.view(p.episode_b)
        assert va.channel_digests() == vb.channel_digests()  # every channel of the decision view
        if p.history_relevant:
            assert p.target_a != p.target_b
            assert va.history != vb.history
            assert min(p.margin_a, p.margin_b) >= 0.3
        else:
            assert p.target_a == p.target_b


def test_labels_are_the_independent_oracle_and_success_is_env_defined(adapter2, suite2):
    for e in suite2.episodes:
        q = e.queries[0]
        assert e.target_for(q.query_id) == oracle_for_ticks(
            q.task, q.params, q.options, front_ticks(e))[0]
        assert e.privileged.state["success_actions"] in q.options
    assert oracle_env_findings(adapter2) == []


def test_gate_passes_clean_small_suite(adapter2, suite2, assignment2):
    r = g0_v2_report(adapter2, suite2.pairs, suite2.episodes, assignment2, enforce_size=False)
    assert r["passed"], r["findings"][:5]


def test_f1_history_free_agents_cannot_solve_relevant_cells(adapter2):
    for reader in (PosteriorOraclePolicy(), BagOfTokensNB()):
        agent = Agent(NoHistory(), reader)
        fit_agent(adapter2, agent, 0)
        for split in ("test_id", "test_ood_length", "test_ood_structure"):
            for fam, m in pair_metrics(evaluate(adapter2, agent, split, 0)).items():
                if fam in RELEVANT2:
                    assert m["pair_both_correct"] == 0.0, (reader.name, split, fam)


@pytest.mark.parametrize("split", ["test_id", "test_ood_structure"])
def test_f2_f3_oracle_follows_swapped_history_and_needs_critical_evidence(adapter2, split):
    res = causal_attribution(adapter2, Agent(AllHistory(), PosteriorOraclePolicy()), split, 0)
    for fam in RELEVANT:
        assert res[fam]["swap_follows_partner"] == 1.0
        assert res[fam]["drop_critical"] >= res[fam]["drop_noncritical"] + 0.2, (fam, res[fam])
        assert uses_history(res[fam], min_drop=0.2, max_noncrit=0.2)
    assert res["control"]["swap_invariance"] == 1.0


def test_f6_static_cups_control_changes_track_outcomes(adapter2):
    """If swaps were irrelevant (v1 cups), removing them would not change the answer."""
    changed = total = 0
    for p in [q for s in ("train", "val", "test_id") for q in adapter2.pairs(s)]:
        if p.family != "track":
            continue
        for eid in (p.episode_a, p.episode_b):
            e = adapter2.episode(eid)
            q = e.queries[0]
            ticks = [{**t, "swap": "none"} for t in front_ticks(e)]
            static, _ = oracle_for_ticks(q.task, q.params, q.options, ticks)
            changed += static != e.target_for(q.query_id)
            total += 1
    assert total >= 20 and changed / total > 0.5


def test_events_edge_counting_falsifier_fails_where_lamp_validity_matters(adapter2):
    """v1's 'count pressed ticks' analogue (contact edges) must NOT solve the event task."""
    bad = Agent(AllHistory(), HeuristicDecoder("edge_count"))
    good = Agent(AllHistory(), HeuristicDecoder("naive"))
    m_bad = pair_metrics(evaluate(adapter2, bad, "test_id", 0))["events"]
    m_good = pair_metrics(evaluate(adapter2, good, "test_id", 0))["events"]
    assert m_bad["pair_both_correct"] == 0.0 and m_good["pair_both_correct"] > 0.8


def test_evidence_position_axis_is_balanced_and_truthful(adapter2, suite2):
    assert position_findings(adapter2) == []
    for fam in RELEVANT:
        c = Counter(p.evidence_position for p in suite2.pairs
                    if p.family == fam and p.regime == "in_dist")
        assert set(c) == {"early", "mid", "late"} and max(c.values()) - min(c.values()) <= 1


def test_regimes_are_structural_not_just_longer(suite2):
    for fam in RELEVANT:
        ind = {m for p in suite2.pairs if p.family == fam and p.regime == "in_dist"
               for m in p.mechanisms.split(",")}
        ood = {m for p in suite2.pairs if p.family == fam and p.regime == "ood_structure"
               for m in p.mechanisms.split(",")}
        assert ood - ind  # at least one mechanism never seen in training
        lens_in = max(p.horizon for p in suite2.pairs if p.family == fam and p.regime == "in_dist")
        lens_ood = [p.horizon for p in suite2.pairs if p.family == fam
                    and p.regime == "ood_structure"]
        assert max(lens_ood) <= lens_in + 8  # structure regime is NOT a length regime
        assert min(p.horizon for p in suite2.pairs if p.family == fam
                   and p.regime == "ood_length") > lens_in


def test_heldout_vocabulary_has_disjoint_atoms(suite2):
    by_id = {e.episode_id: e for e in suite2.episodes}

    def names(regimes):
        got = set()
        for p in suite2.pairs:
            if p.family == "resume" and p.regime in regimes:
                for f in by_id[p.episode_a].frames:
                    got |= {w for v in f.visible.values() for part in v.split("_") for w in [part]
                            if part[:1] in "an" and part[1:].isdigit()}
        return got
    assert names(("heldout_mapping",)) and not (
        names(("heldout_mapping",)) & names(("in_dist", "ood_length", "ood_structure")))


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_planted_leak_in_every_channel_fails_v2_gate(suite2, assignment2, name):
    transform, code = MUTANTS[name]
    eps, pairs, ad = mutated(suite2, assignment2, transform)
    r = g0_v2_report(ad, pairs, eps, assignment2, enforce_size=False)
    assert not r["passed"]
    assert code in {f["code"] for f in r["findings"]}, (name, r["findings"][:4])


def test_generator_bug_regression_wrong_success_action_is_caught(suite2, assignment2):
    """The pre-freeze bug: mirrored mate got the wrong env-success item. Gate must flag it."""
    def corrupt(e):
        if e.provenance["family"] != "resume" or e.provenance["pair_role"] != "b":
            return e
        q = e.queries[0]
        other = next(o for o in q.options if o != e.privileged.state["success_actions"])
        priv = dataclasses.replace(e.privileged, state={**e.privileged.state,
                                                        "success_actions": other})
        return dataclasses.replace(e, privileged=priv)
    eps, pairs, ad = mutated(suite2, assignment2, corrupt)
    r = g0_v2_report(ad, pairs, eps, assignment2, enforce_size=False)
    assert "oracle_env_success_low" in {f["code"] for f in r["findings"]}


def test_position_tag_lie_is_caught(suite2, assignment2):
    def lie(e):
        if e.provenance["family"] == "track":
            return dataclasses.replace(e, provenance={**e.provenance, "position_tick": 0,
                                                      "horizon": 99})
        return e
    *_, ad = mutated(suite2, assignment2, lie)
    assert position_findings(ad)


def test_sufficiency_is_monotone_for_the_oracle_context(adapter2):
    """A restricted context can never beat the full one in expected env success (S <= 1)."""
    def env_rate(provider):
        res = evaluate(adapter2, Agent(provider, PosteriorOraclePolicy()), "test_id", 0)
        return sum(r.env_success for r in res if r.history_relevant) / sum(
            r.history_relevant for r in res)
    assert env_rate(LastK(2)) <= env_rate(AllHistory()) + 1e-9


@given(st.integers(0, 2), st.integers(3, 5))
@settings(max_examples=30, deadline=None)
def test_oracle_posterior_is_normalised_and_unknown_ticks_widen_it(pos, n):
    params = {"n_slots": str(n), "sensor_eps": "0.10", "swap_rate": "0.60"}
    opts = tuple(f"pick_slot_{k}" for k in range(n))
    ticks = {0: {"swap": "none", "reveal": f"slot:{pos}"}, 1: {"swap": "0-1", "reveal": "none"},
             2: {"swap": "none", "reveal": "none"}}
    cur = {"swap": "none", "reveal": "none"}
    full = posterior("track_ball", params, opts, ticks, cur, 4 * TICK_NS)
    part = posterior("track_ball", params, opts, {0: ticks[0]}, cur, 4 * TICK_NS)
    assert abs(sum(full.values()) - 1) < 1e-9 and abs(sum(part.values()) - 1) < 1e-9
    assert max(part.values()) <= max(full.values()) + 1e-9


def test_config_file_loads_and_validates():
    raw = yaml.safe_load((ROOT / "configs/sodh/sodh_v2.yaml").read_text())
    assert SODH2Config.from_dict(raw).master_seed == raw["master_seed"]
    with pytest.raises(ValueError):
        SODH2Config.from_dict({**raw, "split_fractions": [0.5, 0.2, 0.2]})


def test_splits_are_pair_level_and_complete(suite2, assignment2, cfg2):
    assert assignment2 == assign_splits2(suite2.pairs, cfg2)
    assert set(assignment2) == {p.pair_id for p in suite2.pairs}
    assert {"train", "val", "test_id", "test_ood_length", "test_ood_structure"} <= set(
        assignment2.values())
    assert isinstance(SODHAdapter, type)
