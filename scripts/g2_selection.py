#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 evidence-selection under complementarity: exact-enumeration oracle vs practical selectors.

Instances: M=8 binary history events; decision groups (parity over a set G, weight w) each pay
w*(0.5 + 0.5*[G subset of kept]) -- every decisive event is individually worthless but jointly
valuable (synergy). Budget B. We distinguish three quantities (docs/27 corrigendum):
  single-item marginal   value({i}) - value({})
  conditional marginal   value(S + i) - value(S)
  set value              value(S)
Selectors: marginal_greedy, conditional_greedy, loo_from_full, group_selection (knapsack over groups),
shapley (permutation sampling; cost K*M evaluations), joint_exact (enumeration, the oracle).
"""
import itertools
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ucvla.canonical import HashRng

M = 8


def value(keep, groups):
    return sum(w * (0.5 + 0.5 * (g <= keep)) for g, w in groups)


def instance(seed):
    rng = HashRng("g2sel", seed)
    n_groups = rng.randint(2, 3)
    pool = list(range(M))
    rng.shuffle(pool)
    groups, pos = [], 0
    for _ in range(n_groups):
        size = rng.randint(1, 3)
        groups.append((frozenset(pool[pos:pos + size]), 0.5 + rng.below(1500) / 1000))
        pos += size
    return groups, rng.randint(2, 4)


def select(method, groups, b, seed):
    items = list(range(M))
    if method == "joint_exact":
        return max((frozenset(c) for c in itertools.combinations(items, b)), key=lambda s: value(s, groups))
    if method == "marginal_greedy":  # single-item marginal from the empty context
        sc = {i: value(frozenset({i}), groups) - value(frozenset(), groups) for i in items}
        return frozenset(sorted(items, key=lambda i: (-sc[i], i))[:b])
    if method == "loo_from_full":
        full = frozenset(items)
        sc = {i: value(full, groups) - value(full - {i}, groups) for i in items}
        return frozenset(sorted(items, key=lambda i: (-sc[i], i))[:b])
    if method == "conditional_greedy":  # conditional marginal given the current set
        s = frozenset()
        for _ in range(b):
            i = max((j for j in items if j not in s), key=lambda j: (value(s | {j}, groups) - value(s, groups), -j))
            s |= {i}
        return s
    if method == "group_selection":  # knapsack over decision groups (value gain per group, size <= budget)
        best, best_v = frozenset(), value(frozenset(), groups)
        for r in range(1, len(groups) + 1):
            for combo in itertools.combinations(groups, r):
                cover = frozenset().union(*(g for g, _ in combo))
                if len(cover) <= b and value(cover, groups) > best_v:
                    best, best_v = cover, value(cover, groups)
        pad = [i for i in items if i not in best][: b - len(best)]
        return best | frozenset(pad)
    if method.startswith("shapley"):
        k = int(method.split("_")[1])
        rng = HashRng("shap", seed, k)
        phi = dict.fromkeys(items, 0.0)
        for _ in range(k):
            perm = items[:]
            rng.shuffle(perm)
            s = frozenset()
            for i in perm:
                phi[i] += value(s | {i}, groups) - value(s, groups)
                s |= {i}
        return frozenset(sorted(items, key=lambda i: (-phi[i], i))[:b])
    raise ValueError(method)


def main():
    methods = ["marginal_greedy", "loo_from_full", "conditional_greedy", "group_selection",
               "shapley_20", "shapley_200", "shapley_2000", "joint_exact"]
    regret = {m: [] for m in methods}
    optimal_hit = {m: 0 for m in methods}
    n = 400
    for seed in range(n):
        groups, b = instance(seed)
        opt = value(select("joint_exact", groups, b, seed), groups)
        for m in methods:
            v = value(select(m, groups, b, seed), groups)
            regret[m].append(opt - v)
            optimal_hit[m] += v >= opt - 1e-9
    out = {"mode": "synthetic_theory_toy", "instances": n, "M": M,
           "mean_regret_vs_exact_joint": {m: st.mean(r) for m, r in regret.items()},
           "P_optimal": {m: optimal_hit[m] / n for m, _ in optimal_hit.items()},
           "cost_in_value_evaluations": {"marginal_greedy": M, "loo_from_full": M,
                                         "conditional_greedy": "B*M", "group_selection": "2^groups",
                                         "shapley_K": "K*M (approximate; error shrinks ~1/sqrt(K))",
                                         "joint_exact": "C(M,B)  (exponential)"},
           "note": "single-item marginals are 0 for every member of a synergistic group; the old "
                   "selection_accuracy (additive future bits) and the 16-rollout cost bound derived "
                   "from it are INVALID and must not be cited"}
    (ROOT / "results/synthetic/g2_selection.json").write_text(json.dumps(out, indent=1) + "\n")
    for m in methods:
        print(f"{m:20s} mean_regret={out['mean_regret_vs_exact_joint'][m]:.3f} P(optimal)={out['P_optimal'][m]:.2f}")


if __name__ == "__main__":
    main()
