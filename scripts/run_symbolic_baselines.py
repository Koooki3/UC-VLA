#!/usr/bin/env python3
"""Matched-budget sweep of the symbolic readers on SODH. mode: synthetic, NOT a model result.

Budget unit = retained history ticks (hot memory). Every policy uses the same decoding rule, so
differences come only from which history was retained. CIs: cluster bootstrap over pairs.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter, evaluate_policy
from ucvla.baselines import full_history, long_window, memoryless, uniform_subsample
from ucvla.dataset_io import read_dataset
from ucvla.sodh import RELEVANT_FAMILIES
from ucvla.stats import bootstrap_mean_ci, paired_diff_ci

BUDGETS = (1, 2, 4, 8, 16)
SPLITS = ("test_id", "test_ood", "test_counterfactual")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, nargs="?", default=ROOT / "results/synthetic/sodh_v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    episodes, pairs, assignment = read_dataset(args.data_dir)
    adapter = SODHAdapter(episodes, pairs, assignment)
    policies = [memoryless(), full_history()]
    for k in BUDGETS:
        policies += [long_window(k), uniform_subsample(k)]
    rows, per_pair = [], {}
    for split in SPLITS:
        for pol in policies:
            outs = evaluate_policy(adapter, pol, split)
            for fam in sorted({o.family for o in outs}):
                sel = [o for o in outs if o.family == fam]
                both = [float(o.correct_a and o.correct_b) for o in sel]
                per_pair[(split, fam, pol.name)] = both
                m, lo, hi = bootstrap_mean_ci(both, args.seed)
                rows.append({"split": split, "family": fam, "policy": pol.name,
                             "history_relevant": fam in RELEVANT_FAMILIES, "n_pairs": len(sel),
                             "pair_both_correct": m, "ci95": [lo, hi],
                             "mean_retained_ticks": sum(o.retained_ticks for o in sel) / len(sel)})
    diffs = []
    for split in SPLITS:
        for fam in sorted(RELEVANT_FAMILIES):
            for k in BUDGETS:
                a = per_pair[(split, fam, f"uniform_{k}")]
                b = per_pair[(split, fam, f"long_window_{k}")]
                d, lo, hi = paired_diff_ci(a, b, args.seed)
                diffs.append({"split": split, "family": fam, "budget_ticks": k,
                              "uniform_minus_window": d, "ci95": [lo, hi]})
    out = {"mode": "synthetic", "kind": "symbolic_reader_budget_sweep",
           "warning": "hand-written readers on symbolic frames; not a learned model or benchmark",
           "metric": "pair_both_correct (cluster = pair); chance for obs-only readers is 0.0",
           "rows": rows, "paired_uniform_vs_window": diffs}
    path = args.data_dir / "symbolic_baselines.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
