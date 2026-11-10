import pytest

from ucvla.causal_eval import causal_attribution, critical_times, delete_ticks, uses_history
from ucvla.context import AllHistory, NoHistory
from ucvla.episode import EpisodeRecord
from ucvla.harness import Agent, fit_agent
from ucvla.readers import BagOfTokensNB, ObsOnlyLookup, OracleDecoder
from ucvla.sodh import RELEVANT_FAMILIES


def attribution(adapter, agent, split="test_id"):
    fit_agent(adapter, agent, 0)
    return causal_attribution(adapter, agent, split, 0)


@pytest.mark.parametrize("split", ["test_id", "test_ood", "test_counterfactual"])
def test_history_using_agent_is_attributed_to_history(adapter, split):
    res = attribution(adapter, Agent(AllHistory(), OracleDecoder("strict")), split)
    for fam in RELEVANT_FAMILIES:
        assert uses_history(res[fam], min_swap=0.6, min_drop=0.15), (fam, res[fam])
        assert res[fam]["del_critical"] < res[fam]["original_on_crit_subset"]
        assert res[fam]["del_noncritical"] == res[fam]["original_on_crit_subset"]
    assert res["object_correspondence"]["swap_follows_partner"] == 1.0
    assert res["process_count"]["swap_follows_partner"] == 1.0
    assert res["irrelevant_history"]["swap_invariance"] == 1.0  # negative control


def test_free_clock_information_changes_the_picture(adapter):
    """Clock decoder follows swapped history perfectly (completeness is free) but its scale-up
    makes it fragile to non-critical deletion: decoder priors move the attribution numbers."""
    res = attribution(adapter, Agent(AllHistory(), OracleDecoder("clock")), "test_id")
    assert all(res[f]["swap_follows_partner"] == 1.0 for f in RELEVANT_FAMILIES)
    assert res["process_count"]["drop_noncritical"] > 0.1


@pytest.mark.parametrize("agent", [Agent(NoHistory(), OracleDecoder()),
                                   Agent(NoHistory(), ObsOnlyLookup())])
def test_history_free_agents_show_no_history_dependence(adapter, agent):
    res = attribution(adapter, agent)
    for fam in RELEVANT_FAMILIES:
        assert not uses_history(res[fam])
        assert res[fam]["drop_critical"] == 0.0 and res[fam]["drop_noncritical"] == 0.0
        assert res[fam]["swap_follows_partner"] <= 0.5 + 1e-9
    assert res["irrelevant_history"]["swap_invariance"] == 1.0


def test_generic_learner_is_not_credited_with_using_history(adapter):
    res = attribution(adapter, Agent(AllHistory(), BagOfTokensNB()))
    assert not any(uses_history(res[f]) for f in ("object_correspondence", "task_interleave"))


def test_critical_ticks_come_from_annotations_only_and_precede_decision(adapter, suite):
    for ep in suite.episodes[:80]:
        crit = critical_times(ep)
        assert all(t < ep.queries[0].time_ns for t in crit)
        if ep.provenance["family"] == "irrelevant_history":
            assert not crit


def test_delete_ticks_removes_exactly_requested(adapter):
    ep = adapter.episode(adapter.pairs("train")[0].episode_a)
    view = ep.causal_view(ep.queries[0].time_ns)
    times = frozenset(t.time_ns for t in view.history[:2])
    out = delete_ticks(view.history, times)
    assert len(out) == len(view.history) - 2 and not {t.time_ns for t in out} & times


def test_independent_oracle_recomputes_every_label_from_privileged_state(suite):
    """Labels re-derived by separate code from privileged state + task params (not the generator)."""
    for ep in suite.episodes:
        assert isinstance(ep, EpisodeRecord)
        q, priv = ep.queries[0], ep.privileged.state
        got = ep.target_for(q.query_id)
        if q.task == "cups_ball":
            exp = f"pick_slot_{priv['final_slot']}"
        elif q.task == "button_count":
            exp = "press" if int(priv["presses_done"]) < int(q.params["target_count"]) else "stop"
        elif q.task == "resume_sequence":
            exp = f"place_{q.params['phase_order'].split(',')[int(priv['phases_done'])]}"
        else:
            exp = f"pick_{priv['flag']}"
        assert got == exp
