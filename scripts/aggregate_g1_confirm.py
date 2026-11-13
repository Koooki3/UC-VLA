#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Apply docs/41 to the sealed confirmatory runs; bounded intervals; per-seed values; length-OOD."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.aggregate_g1 import REL, summarise

D = ROOT / "results/synthetic/g1/confirm"


def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


runs = {}
for p in sorted(D.glob("*_s*.json")):
    r = json.loads(p.read_text())
    runs.setdefault(r["arch"], []).append(r)
rep = {"protocol": "docs/41", "scope": "sealed sodh_v2c: same generator, new random seeds", "archs": {}}
for arch, rs in runs.items():
    a = summarise(rs)
    npairs = {s: {f: rs[0]["metrics"][s][f]["n_pairs"] for f in REL if f in rs[0]["metrics"][s]} for s in rs[0]["metrics"]}
    a["n_pairs"] = npairs
    a["per_seed_pair_optimal_test_id"] = {f: [r["metrics"]["test_id"][f]["pair_both_correct"] for r in rs] for f in REL}
    a["bounded_intervals_test_id"] = {}
    for f in REL:
        v = a["per_seed_pair_optimal_test_id"][f]
        n = npairs["test_id"][f]
        pooled = sum(v) / len(v)
        a["bounded_intervals_test_id"][f] = {"pooled": pooled, "pair_level_wilson": wilson(pooled, n * len(v)),
                                              "seed_min_max": [min(v), max(v)],
                                              "seed_t_interval_clipped": [max(0.0, a["fam"][f]["pair_optimal"]["test_id"]["lo"]),
                                                                          min(1.0, a["fam"][f]["pair_optimal"]["test_id"]["hi"])]}
    a["length_ood"] = {s: {f: a["fam"][f]["pair_optimal"][s]["mean"] for f in REL if s in a["fam"][f]["pair_optimal"]}
                       for s in ("test_ood_length", "test_ood_structure", "test_counterfactual")}
    rep["archs"][arch] = a
(ROOT / "results/synthetic/g1/g1_confirm_report.json").write_text(json.dumps(rep, indent=1) + "\n")
for arch, a in rep["archs"].items():
    print(arch, "seeds", a["seeds"], "G1 verdict (all conditions):", a["g1_pass"], "| legacy:", a["audit"]["g1_pass_legacy_without_order"])
    print("  conditions", a["audit"]["conditions"], "order_drop_track", round(a["audit"]["order_drop_track"], 2))
    for f in REL:
        b = a["bounded_intervals_test_id"][f]
        print(f"  {f}: pooled {b['pooled']:.2f} wilson[{b['pair_level_wilson'][0]:.2f},{b['pair_level_wilson'][1]:.2f}] seeds {b['seed_min_max']} swap {a['fam'][f]['swap_follow']['mean']:.2f} dropgap {a['fam'][f]['drop_critical_minus_noncrit']['mean']:.2f} mem0 {a['fam'][f]['memoryless_test_id']['mean']:.2f} order {a['fam'][f]['order_perturbed_pair_optimal']['mean']:.2f} | OOD-struct {a['length_ood']['test_ood_structure'].get(f)} OOD-len {a['length_ood']['test_ood_length'].get(f)}")
    print("  control", a["control_acc"]["mean"], a["control_swap_invariance"]["mean"], "per-family audit", {f: a["audit"]["per_family"][f]["all"] for f in REL})
