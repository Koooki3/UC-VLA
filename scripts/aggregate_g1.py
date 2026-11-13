#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate G1 runs and apply the pre-registered criteria of docs/25. Writes g1_report.json and
appends the test-access ledger. Between-seed t-interval (n=5) stands in for the per-pair CI."""
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/synthetic/g1"
REL = ("track", "events", "resume")
T975 = {4: 2.776}


def ms(v):
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    h = T975.get(len(v) - 1, 2.776) * sd / math.sqrt(len(v))
    return {"mean": m, "lo": m - h, "hi": m + h, "min": min(v), "max": max(v)}


runs = {}
for p in sorted((D / "runs").glob("*.json")):
    r = json.loads(p.read_text())
    runs.setdefault(r["arch"], []).append(r)
rep = {"mode": "synthetic", "protocol": "docs/25", "status": "EXPLORATORY: v2 test splits were viewed before",
       "archs": {}}
for arch, rs in runs.items():
    a = {"seeds": len(rs), "params": rs[0]["spec"]["params"],
         "train_acc": ms([r["train"]["final_train_acc"] for r in rs]),
         "epochs": [r["epochs"] for r in rs], "fam": {}}
    for f in REL:
        g = lambda s, k, f=f, rs=rs: [r["metrics"][s][f][k] for r in rs]
        att = lambda s, k, f=f, rs=rs: [r["attribution"][s][f][k] for r in rs if k in r["attribution"][s][f]]
        a["fam"][f] = {
            "pair_optimal": {s: ms(g(s, "pair_both_correct")) for s in
                             ("test_id", "test_ood_length", "test_ood_structure", "test_counterfactual") if f in rs[0]["metrics"][s]},
            "memoryless_test_id": ms([r["memoryless"]["test_id"][f]["pair_both_correct"] for r in rs]),
            "swap_follow": ms(att("test_id", "swap_follows_partner")),
            "drop_critical_minus_noncrit": ms([x - y for x, y in zip(att("test_id", "drop_critical"), att("test_id", "drop_noncritical"))]),
            "order_perturbed_pair_optimal": ms([r["order_perturbation"]["test_id"][f] for r in rs])}
    a["control_acc"] = ms([r["metrics"]["test_id"]["control"]["episode_acc"] for r in rs])
    a["control_swap_invariance"] = ms([r["attribution"]["test_id"]["control"]["swap_invariance"] for r in rs])
    ok_f = []
    for f in REL:
        x = a["fam"][f]
        c = (x["pair_optimal"]["test_id"]["mean"] >= 0.60 and x["pair_optimal"]["test_id"]["lo"] > 0.20
             and x["swap_follow"]["mean"] >= 0.70 and x["drop_critical_minus_noncrit"]["mean"] >= 0.20
             and x["memoryless_test_id"]["mean"] <= 0.05)
        x["meets_criteria"] = bool(c)
        x["seed_range"] = x["pair_optimal"]["test_id"]["max"] - x["pair_optimal"]["test_id"]["min"]
        ok_f.append(c and x["seed_range"] <= 0.25)
    a["control_ok"] = a["control_acc"]["mean"] >= 0.95 and a["control_swap_invariance"]["mean"] >= 0.99
    a["order_ok_track"] = (a["fam"]["track"]["pair_optimal"]["test_id"]["mean"]
                           - a["fam"]["track"]["order_perturbed_pair_optimal"]["mean"]) >= 0.15
    a["g1_pass"] = sum(ok_f) >= 2 and a["control_ok"]
    rep["archs"][arch] = a
rep["G1_pass_any_architecture"] = any(a["g1_pass"] for a in rep["archs"].values())
(D / "g1_report.json").write_text(json.dumps(rep, indent=1) + "\n")
led = D / "evaluation_ledger.jsonl"
if not led.exists():
    led.write_text(json.dumps({"purpose": "G1 official runs", "splits": ["test_id", "test_ood_length", "test_ood_structure", "test_counterfactual"],
                               "note": "one scratch smoke run (gru seed0, 25 epochs) saw test numbers earlier; hyper-parameters unchanged; v2 test splits previously viewed => exploratory"}) + "\n")
for arch, a in rep["archs"].items():
    f = a["fam"]
    print(f"{arch:13s} params={a['params']:6d} train_acc={a['train_acc']['mean']:.2f} PASS={a['g1_pass']} ctrl={a['control_ok']} | " + " ".join(
        f"{k[:3]}: po={f[k]['pair_optimal']['test_id']['mean']:.2f}[{f[k]['pair_optimal']['test_id']['min']:.2f},{f[k]['pair_optimal']['test_id']['max']:.2f}] swap={f[k]['swap_follow']['mean']:.2f} dd={f[k]['drop_critical_minus_noncrit']['mean']:.2f} mem0={f[k]['memoryless_test_id']['mean']:.2f} ord={f[k]['order_perturbed_pair_optimal']['mean']:.2f}" for k in REL))
print("G1 pass any:", rep["G1_pass_any_architecture"])
