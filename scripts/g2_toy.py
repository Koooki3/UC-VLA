#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 minimal computable theory model (docs/27): exact enumeration, no learning, no GPU.

M iid fair binary history events. The current decision needs events N (parity), a later decision
needs events F (parity); reward = r_now + w * r_future, each 1 if the parity is predicted, else a
fair guess. A memory keeps B events. Selectors differ only in HOW they pick which events to keep.
Value(S) is exact, so any difference between selectors is a property of the objective, not noise.
"""

import argparse
import itertools
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ucvla.canonical import HashRng

ROOT = Path(__file__).resolve().parents[1]


def value(keep: frozenset[int], need_now: frozenset[int], need_fut: frozenset[int], w: float) -> float:
    return (0.5 + 0.5 * (need_now <= keep)) + w * (0.5 + 0.5 * (need_fut <= keep))


def expected_value(selector: str, m: int, b: int, need_now, need_fut, w: float) -> float:
    """Exact expected value; ties among selector-equivalent subsets are broken uniformly."""
    subsets = [frozenset(c) for c in itertools.combinations(range(m), b)]

    def score(s: frozenset[int]) -> float:
        if selector == "action_mi":  # information about the CURRENT optimal action only
            return float(len(s & need_now))  # parity of N: a subset carries it only through N
        if selector == "regret_aware":  # exact future-decision value of keeping s
            return value(s, need_now, need_fut, w)
        if selector in ("reconstruction", "random", "salience"):  # iid events: all subsets tie
            return 0.0
        raise ValueError(selector)

    top = max(score(s) for s in subsets)
    best = [s for s in subsets if score(s) == top]
    return sum(value(s, need_now, need_fut, w) for s in best) / len(best)


def selection_accuracy(m: int, b: int, need_fut: frozenset[int], n_rollouts: int, seed: int,
                       reps: int = 400) -> float:
    """Counterfactual-intervention estimator of evidence value: delta_i = mean over rollouts of
    reward(keep i) - reward(drop i); keep the top-B estimated. P(selects exactly F)."""
    ok = 0
    for r in range(reps):
        rng = HashRng("g2", seed, m, b, n_rollouts, r)
        est = []
        for i in range(m):
            tot = 0
            for _ in range(n_rollouts):
                keep_hit = 1 if i in need_fut else int(rng.below(2))
                drop_hit = int(rng.below(2))
                tot += keep_hit - drop_hit
            est.append((tot / n_rollouts, rng.below(10**6), i))
        chosen = frozenset(i for _, _, i in sorted(est, reverse=True)[:b])
        ok += chosen == need_fut
    return ok / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/g2_toy.json")
    args = ap.parse_args()
    m = 8
    regimes = {
        "A_future_only (N empty, F={0,1}); current obs identical, current action identical":
            (frozenset(), frozenset({0, 1})),
        "B_aligned (F subset of N)": (frozenset({0, 1}), frozenset({0, 1})),
        "C_partial_overlap (N={0,2}, F={0,1})": (frozenset({0, 2}), frozenset({0, 1})),
        "D_conflict (N={2,3}, F={0,1}) budget 2": (frozenset({2, 3}), frozenset({0, 1})),
    }
    out = {"mode": "synthetic_theory_toy", "M": m, "budget_B": 2, "rows": [], "recovery": []}
    for name, (nn, ff) in regimes.items():
        for w in (0.5, 1.0, 2.0):
            row = {"regime": name, "w_future": w}
            for sel in ("action_mi", "reconstruction", "salience", "regret_aware"):
                row[sel] = expected_value(sel, m, 2, nn, ff, w)
            row["ceiling"] = value(ff | nn if len(ff | nn) <= 2 else ff, nn, ff, w)
            out["rows"].append(row)
    for n in (1, 2, 4, 8, 16, 32, 64):
        out["recovery"].append({"rollouts_per_event": n,
                                "P_select_exactly_F": selection_accuracy(m, 2, frozenset({0, 1}), n, 0),
                                "chance": 1 / comb(m, 2)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    for r in out["rows"]:
        if r["w_future"] == 1.0:
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(out["recovery"])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
