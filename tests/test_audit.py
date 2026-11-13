# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from tests.conftest import small_config
from ucvla.adapters import SODHAdapter, evaluate_policy, summarize
from ucvla.audit import (
    g0_report,
    history_swap_audit,
    id_leak_findings,
    metadata_lookup_audit,
    obs_only_audit,
    structural_pair_diffs,
)
from ucvla.baselines import full_history, long_window, memoryless, uniform_subsample
from ucvla.sodh import RELEVANT_FAMILIES, generate_suite
from ucvla.splits import assign_splits


def test_g0_passes_on_clean_suite(adapter, suite, assignment):
    r = g0_report(adapter, suite.pairs, suite.episodes, assignment)
    assert r["passed"], r["findings"]
    assert r["mode"] == "synthetic"


def test_obs_only_is_capped_at_half_on_relevant_families(adapter):
    """A6: structural cap for any observation-only policy."""
    res = obs_only_audit(adapter)
    for fam in RELEVANT_FAMILIES:
        assert res[fam]["conflict_fraction"] == 1.0
        assert res[fam]["obs_only_acc"] <= 0.5
    assert res["irrelevant_history"]["obs_only_acc"] >= 0.9  # control is solvable from obs


def test_structural_features_identical_within_pairs(adapter, suite):
    assert set(structural_pair_diffs(adapter, suite.pairs).values()) == {0.0}


def test_history_swap_follows_history_but_not_for_control(adapter):
    """A7."""
    res = history_swap_audit(adapter)
    assert all(v == 1.0 for v in res.values())


@pytest.mark.parametrize("leak,expected", [
    ("length", {"structural_diff"}),
    ("id", {"id_leak"}),
    ("final_obs", {"obs_only_solves", "obs_differs"}),
])
def test_auditor_detects_planted_leaks(leak, expected):
    """Negative control for the auditor: planted shortcuts must make G0 fail."""
    cfg = small_config(leak=leak)
    s = generate_suite(cfg)
    asg = assign_splits(s.pairs, cfg)
    r = g0_report(SODHAdapter(s.episodes, s.pairs, asg), s.pairs, s.episodes, asg)
    assert not r["passed"]
    assert expected <= {f["code"] for f in r["findings"]}


def test_id_leak_detector_direct(suite):
    ep = suite.episodes[0]
    leaked = dataclasses.replace(ep, episode_id=ep.episode_id + ep.target_for(ep.queries[0].query_id))
    assert id_leak_findings([ep]) == []
    assert id_leak_findings([leaked])


def test_metadata_lookup_audit_quiet_on_clean_suite(adapter):
    assert metadata_lookup_audit(adapter) == []


def test_memoryless_cannot_solve_pairs_and_full_history_does(adapter):
    """A8."""
    for split in ("test_id", "test_ood", "test_counterfactual"):
        for fam, m in summarize(evaluate_policy(adapter, memoryless(), split)).items():
            if fam in RELEVANT_FAMILIES:
                assert m["pair_both_correct"] == 0.0 and m["decision_flip_rate"] == 0.0
        for fam, m in summarize(evaluate_policy(adapter, full_history(), split)).items():
            assert m["pair_both_correct"] == 1.0


def test_budget_semantics_and_window_degrades_with_horizon(adapter):
    ind = summarize(evaluate_policy(adapter, long_window(2), "train"))
    ood = summarize(evaluate_policy(adapter, long_window(2), "test_ood"))
    for fam in ("object_correspondence", "task_interleave"):
        assert ood[fam]["pair_both_correct"] <= ind[fam]["pair_both_correct"]
        assert ood[fam]["pair_both_correct"] < 1.0
    assert all(m["mean_retained_ticks"] <= 2 for m in ood.values())
    assert all(m["mean_retained_ticks"] <= 3
               for m in summarize(evaluate_policy(adapter, uniform_subsample(3), "test_ood")).values())


def test_policies_cannot_reach_labels(adapter, suite):
    """The only object a policy receives has no target/privileged/provenance attribute."""
    view = adapter.view(suite.episodes[0].episode_id)
    for name in ("targets", "privileged", "provenance", "events_annotation"):
        assert not hasattr(view, name)
    assert view.query is not None and not hasattr(view.query, "value")
