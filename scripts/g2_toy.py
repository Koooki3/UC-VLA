#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 minimal computable model v2 (docs/27, corrected after the adversarial review docs/38).

M iid fair binary history events; decision groups g (needed events G_g, weight w_g, 'now' = weight 1
group N, later groups F...) each pay w_g*(0.5 + 0.5*[G_g subset of kept]) (parity needs ALL its events).
A memory keeps B events. Selectors differ only in HOW they choose. Value is exact by enumeration.
What this toy can show: a ONE-STEP objective is blind to later-only evidence. What it cannot show:
that a future-value objective beats a lookahead/whole-episode objective (they coincide below).
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ucvla.canonical import HashRng

M, B = 8, 2


def value(keep, groups):
    return sum(w * (0.5 + 0.5 * (g <= keep)) for g, w in groups)


def subsets():
    return [frozenset(c) for c in itertools.combinations(range(M), B)]


def avg_over_best(score, groups):
    top = max(score(s) for s in subsets())
    best = [s for s in subsets() if score(s) == top]
    return sum(value(s, groups) for s in best) / len(best)


def run_regime(now, fut, w, gated=False):
    groups = [(now, 1.0)] if now else []
    groups += [(fut, w)]
    def true_value(s):
        if not gated:
            return value(s, groups)
        r_now = 0.5 + 0.5 * (now <= s) if now else 1.0
        return r_now + w * r_now * (0.5 + 0.5 * (fut <= s))  # later decision reached only if now succeeds (scaled)
    ceil = max(true_value(s) for s in subsets())
    out = {"ceiling": ceil}
    out["random/reconstruction"] = sum(true_value(s) for s in subsets()) / len(subsets())
    out["onestep_overlap (|keep & N|)"] = avg_over_best_tv(lambda s: len(s & now), true_value)
    out["lookahead (now+future needed sets)"] = avg_over_best_tv(
        lambda s: (now <= s) * 1.0 + w * (fut <= s), true_value)
    out["future_value (exact, ungated model)"] = avg_over_best_tv(lambda s: value(s, groups), true_value)
    for rho in (0.25, 0.5, 0.75, 1.0):  # salience correlated with future relevance
        tot, reps = 0.0, 600
        for r in range(reps):
            rng = HashRng("sal", rho, now and sorted(now), sorted(fut), w, r)
            sc = {i: (rho if i in fut else 0.0) + (1 - rho) * rng.below(1000) / 1000 + rng.below(1000) / 1e7
                  for i in range(M)}
            tot += true_value(frozenset(sorted(sc, key=lambda i: -sc[i])[:B]))
        out[f"salience(rho={rho})"] = tot / reps
    return out


def avg_over_best_tv(score, true_value):
    top = max(score(s) for s in subsets())
    best = [s for s in subsets() if score(s) == top]
    return sum(true_value(s) for s in best) / len(best)


def noisy_critic(now, fut, w, sigma, reps=2000):
    groups = ([(now, 1.0)] if now else []) + [(fut, w)]
    tot = 0.0
    for r in range(reps):
        rng = HashRng("crit", sigma, r)
        est = {s: value(s, groups) + sigma * (sum(rng.below(1000) for _ in range(12)) / 1000 - 6)
               for s in subsets()}
        tot += value(max(est, key=est.get), groups)
    return tot / reps


def identifiability():
    """Two non-additive decision groups: per-item leave-one-out Delta from the full context ranks the
    wrong pair first; subset-level evaluation with n rollouts finds the best subset."""
    groups = [(frozenset({0, 1}), 1.0), (frozenset({2, 3, 4}), 1.2)]
    full = frozenset(range(M))
    loo = {i: value(full, groups) - value(full - {i}, groups) for i in range(M)}
    pick = frozenset(sorted(range(M), key=lambda i: (-loo[i], i))[:B])
    best = max(subsets(), key=lambda s: value(s, groups))
    rec = []
    for n in (4, 16, 64, 256):
        ok = 0
        for r in range(300):
            rng = HashRng("subset", n, r)
            est = {}
            for s in subsets():
                tot = 0
                for g, w in groups:
                    p = 1.0 if g <= s else 0.5
                    tot += w * sum(int(rng.below(1000) < p * 1000) for _ in range(n)) / n
                est[s] = tot
            ok += max(est, key=est.get) == best
        rec.append({"rollouts_per_subset": n, "P_select_best_subset": ok / 300})
    return {"loo_delta_from_full": loo, "loo_pick": sorted(pick), "loo_pick_value": value(pick, groups),
            "best_subset": sorted(best), "best_value": value(best, groups),
            "random_value": sum(value(s, groups) for s in subsets()) / len(subsets()),
            "subset_level_selection": rec,
            "delta_from_empty_context": {i: value(frozenset({i}), groups) - value(frozenset(), groups) for i in range(M)}}


def main():
    regimes = {
        "A_future_only (N empty, F={0,1})": (frozenset(), frozenset({0, 1})),
        "B_aligned (N=F={0,1})": (frozenset({0, 1}), frozenset({0, 1})),
        "C_partial (N={0,2}, F={0,1})": (frozenset({0, 2}), frozenset({0, 1})),
        "D_conflict (N={2,3}, F={0,1})": (frozenset({2, 3}), frozenset({0, 1})),
        "E_gated (N={2,3}, F={0,1}, later reached only if now succeeds)": (frozenset({2, 3}), frozenset({0, 1})),
        "F_N_larger_than_B (N={0,1,2}, F={0,1})": (frozenset({0, 1, 2}), frozenset({0, 1})),
    }
    out = {"mode": "synthetic_theory_toy", "M": M, "B": B, "rows": [], "noisy_critic_aligned_regime": [],
           "identifiability": identifiability(),
           "caveats": ["lookahead selector == future-value selector by construction: the toy does not separate them",
                       "selectors are idealised, not D&R/EventVLA reproductions"]}
    for name, (nn, ff) in regimes.items():
        for w in (0.5, 1.0, 2.0):
            out["rows"].append({"regime": name, "w_future": w, **run_regime(nn, ff, w, gated=name.startswith("E_"))})
    for sigma in (0.0, 0.25, 0.5, 1.0, 2.0):
        out["noisy_critic_aligned_regime"].append({"sigma": sigma, "future_value_with_noisy_critic":
            noisy_critic(frozenset({0, 1}), frozenset({0, 1}), 1.0, sigma), "onestep_overlap": 2.0, "ceiling": 2.0})
    (ROOT / "results/synthetic/g2_toy.json").write_text(json.dumps(out, indent=1, default=sorted) + "\n")
    for r in out["rows"]:
        if r["w_future"] == 1.0:
            print(r["regime"][:34], {k: round(v, 2) for k, v in r.items() if k not in ("regime", "w_future") and not k.startswith("salience(rho=0.")})
    print(out["noisy_critic_aligned_regime"]); print(json.dumps(out["identifiability"])[:700])


if __name__ == "__main__":
    main()
