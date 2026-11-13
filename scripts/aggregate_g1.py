#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate G1 runs and apply EVERY pre-registered criterion of docs/25, each recorded explicitly.

Corrected 2026-11-13 after external review: the order-perturbation requirement was computed but
not part of the verdict. `g1_pass_legacy_without_order` keeps the earlier (insufficient) verdict
for the record; `g1_pass` requires all conditions."""
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/synthetic/g1"
REL = ("track", "events", "resume")
T975 = {4: 2.776}
THRESH = {"po_mean": 0.60, "po_lo": 0.20, "swap": 0.70, "drop_gap": 0.20, "memoryless_max": 0.05,
          "control_acc": 0.95, "control_inv": 0.99, "seed_range": 0.25, "order_drop_track": 0.15,
          "families_required": 2}


def ms(v):
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    h = T975.get(len(v) - 1, 2.776) * sd / math.sqrt(len(v))
    return {"mean": m, "lo": m - h, "hi": m + h, "min": min(v), "max": max(v)}


def apply_criteria(a: dict) -> dict:
    """Explicit audit of docs/25 for one architecture summary `a` (see main for its construction)."""
    fam_ok = {}
    for f in REL:
        x = a["fam"][f]
        po = x["pair_optimal"]["test_id"]
        fam_ok[f] = {
            "pair_optimal_mean": po["mean"] >= THRESH["po_mean"],
            "pair_optimal_lower_ci": po["lo"] > THRESH["po_lo"],
            "swap_follow": x["swap_follow"]["mean"] >= THRESH["swap"],
            "drop_gap": x["drop_critical_minus_noncrit"]["mean"] >= THRESH["drop_gap"],
            "memoryless": x["memoryless_test_id"]["mean"] <= THRESH["memoryless_max"],
            "seed_range": (po["max"] - po["min"]) <= THRESH["seed_range"]}
        fam_ok[f]["all"] = all(fam_ok[f].values())
    track_po = a["fam"]["track"]["pair_optimal"]["test_id"]["mean"]
    order_drop = track_po - a["fam"]["track"]["order_perturbed_pair_optimal"]["mean"]
    cond = {
        "families_meeting_criteria": sum(v["all"] for v in fam_ok.values()) >= THRESH["families_required"],
        "control_accuracy": a["control_acc"]["mean"] >= THRESH["control_acc"],
        "control_swap_invariance": a["control_swap_invariance"]["mean"] >= THRESH["control_inv"],
        "order_perturbation_cuts_track": order_drop >= THRESH["order_drop_track"]}
    legacy = cond["families_meeting_criteria"] and cond["control_accuracy"] and cond["control_swap_invariance"]
    return {"per_family": fam_ok, "conditions": cond, "order_drop_track": order_drop,
            "g1_pass": all(cond.values()), "g1_pass_legacy_without_order": legacy}


def summarise(rs: list) -> dict:
    a = {"seeds": len(rs), "params": rs[0]["spec"]["params"],
         "train_acc": ms([r["train"]["final_train_acc"] for r in rs]),
         "epochs": [r["epochs"] for r in rs], "fam": {}}
    for f in REL:
        def g(s, k, f=f, rs=rs):
            return [r["metrics"][s][f][k] for r in rs]

        def att(s, k, f=f, rs=rs):
            return [r["attribution"][s][f][k] for r in rs if k in r["attribution"][s][f]]
        a["fam"][f] = {
            "pair_optimal": {s: ms(g(s, "pair_both_correct")) for s in
                             ("test_id", "test_ood_length", "test_ood_structure", "test_counterfactual")
                             if f in rs[0]["metrics"][s]},
            "memoryless_test_id": ms([r["memoryless"]["test_id"][f]["pair_both_correct"] for r in rs]),
            "swap_follow": ms(att("test_id", "swap_follows_partner")),
            "drop_critical_minus_noncrit": ms([x - y for x, y in zip(att("test_id", "drop_critical"),
                                                                      att("test_id", "drop_noncritical"))]),
            "order_perturbed_pair_optimal": ms([r["order_perturbation"]["test_id"][f] for r in rs])}
    a["control_acc"] = ms([r["metrics"]["test_id"]["control"]["episode_acc"] for r in rs])
    a["control_swap_invariance"] = ms([r["attribution"]["test_id"]["control"]["swap_invariance"] for r in rs])
    a["audit"] = apply_criteria(a)
    a["g1_pass"] = a["audit"]["g1_pass"]
    return a


def main():
    runs = {}
    for p in sorted((D / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["arch"], []).append(r)
    rep = {"mode": "synthetic", "protocol": "docs/25", "thresholds": THRESH,
           "status": "EXPLORATORY: v2 test splits were viewed before; main experiment = 5 seeds at base data size",
           "archs": {k: summarise(v) for k, v in runs.items()}}
    rep["G1_pass_any_architecture"] = any(a["g1_pass"] for a in rep["archs"].values())
    rep["G1_pass_any_architecture_legacy_without_order"] = any(
        a["audit"]["g1_pass_legacy_without_order"] for a in rep["archs"].values())
    (D / "g1_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    for arch, a in rep["archs"].items():
        c = a["audit"]["conditions"]
        print(f"{arch:13s} PASS={a['g1_pass']} legacy={a['audit']['g1_pass_legacy_without_order']} "
              f"order_drop_track={a['audit']['order_drop_track']:.2f} conditions={c}")
    print("G1 pass any (all conditions):", rep["G1_pass_any_architecture"],
          "| legacy:", rep["G1_pass_any_architecture_legacy_without_order"])


if __name__ == "__main__":
    main()
