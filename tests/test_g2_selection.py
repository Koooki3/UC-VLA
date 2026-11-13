# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("g2s", Path(__file__).resolve().parents[1] / "scripts/g2_selection.py")
g2s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2s)


def test_synergy_single_item_marginals_are_zero_but_set_value_is_positive():
    groups = [(frozenset({0, 1}), 1.0)]
    v = g2s.value
    assert v(frozenset({0}), groups) - v(frozenset(), groups) == 0
    assert v(frozenset({0, 1}), groups) - v(frozenset(), groups) == 0.5


def test_marginal_greedy_misselects_and_exact_joint_is_optimal():
    groups = [(frozenset({0, 1}), 1.0), (frozenset({2, 3, 4}), 1.2)]
    best = g2s.select("joint_exact", groups, 2, 0)
    assert best == frozenset({0, 1}) and g2s.value(best, groups) == 1.6
    assert all(g2s.value(frozenset({i}), groups) == g2s.value(frozenset(), groups) for i in range(8))  # no signal: ties
    assert g2s.value(g2s.select("loo_from_full", groups, 2, 0), groups) < 1.6  # picks {2,3}
    assert g2s.value(g2s.select("group_selection", groups, 2, 0), groups) == 1.6
