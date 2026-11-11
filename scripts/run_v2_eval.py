#!/usr/bin/env python3
"""SODH v2 evaluation (docs/19 §6-§9): Memory Necessity / Use / Sufficiency / Compression
Efficiency, decoder-and-learner cross-validation, F7 ranking-agreement test. CPU, synthetic.
Test splits are used for evaluation only and logged in the ledger; the val split gives the pilot
variance. Nothing here tunes anything. Oracle rows are measuring instruments, not methods."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.canonical import canonical_json, digest, sha256_hex
from ucvla.causal_eval import causal_attribution, coverage_report, uses_history
from ucvla.context import AllHistory, ContextBudget, LastK, NoHistory, UniformK
from ucvla.dataset_io import read_dataset
from ucvla.harness import Agent, evaluate, fit_agent, results_digest
from ucvla.readers import BagOfTokensNB, ObsOnlyLookup
from ucvla.readers2 import HEURISTIC_VARIANTS, HeuristicDecoder, PosteriorOraclePolicy
from ucvla.runrecord import append_ledger, build_run_record, write_run
from ucvla.sodh2 import RELEVANT2, PairRecord2
from ucvla.stats import bootstrap_mean_ci
from ucvla.stattests import holm, sign_test

DATA = ROOT / "results/synthetic/sodh_v2"
TEST = ("test_id", "test_ood_length", "test_ood_structure", "test_counterfactual")
BUDGETS = (1, 2, 4, 8, 16)
REG = sorted(RELEVANT2)


def per_pair(res):
    out = {}
    for r in res:
        d = out.setdefault(r.pair_id, {"fam": r.family, "opt": [], "env": []})
        d["opt"].append(r.correct)
        d["env"].append(r.env_success)
    return out


def cell(res, fam):
    pp = [d for d in per_pair(res).values() if d["fam"] == fam]
    if not pp:
        return None
    both = [float(all(d["opt"])) for d in pp]
    env = [sum(d["env"]) / len(d["env"]) for d in pp]
    m, lo, hi = bootstrap_mean_ci(both, 0, 1000)
    return {"n_pairs": len(pp), "pair_optimal": m, "ci": [lo, hi],
            "env_success": sum(env) / len(env)}


def mde(p_disc, n):
    """Approx. minimal detectable discordant imbalance (alpha .05 two-sided, power .8)."""
    return 2.8 * math.sqrt(max(p_disc, 1e-9) / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/v2_eval")
    ap.add_argument("--ledger", action="store_true")
    args = ap.parse_args()
    eps, pairs, asg = read_dataset(DATA, PairRecord2)
    ad = SODHAdapter(eps, pairs, asg)
    oracle = PosteriorOraclePolicy()
    base = [("oracle_full", Agent(AllHistory(), oracle), None),
            ("oracle_none", Agent(NoHistory(), oracle), ContextBudget("ticks", 0)),
            ("obs_only_lookup", Agent(NoHistory(), ObsOnlyLookup()), ContextBudget("ticks", 0)),
            ("nb_full", Agent(AllHistory(), BagOfTokensNB()), None)]
    base += [(f"heur_{v}_full", Agent(AllHistory(), HeuristicDecoder(v)), None)
             for v in HEURISTIC_VARIANTS]
    sweep = []
    for k in BUDGETS:
        b = ContextBudget("ticks", k)
        for sel, prov in (("last", LastK(k)), ("uniform", UniformK(k))):
            for inst, pol in (("oracle", oracle), ("heur_naive", HeuristicDecoder("naive")),
                              ("nb", BagOfTokensNB())):
                sweep.append((f"{inst}|{sel}|{k}", Agent(prov, pol), b))
    results, digests, fits = {}, [], {}
    for name, agent, budget in base + sweep:
        fits[name] = fit_agent(ad, agent, args.run_seed, "train", budget)
        for split in ("val", *TEST):
            res = evaluate(ad, agent, split, args.run_seed, budget)
            results[(name, split)] = res
            digests.append(results_digest(res))
    report = {"mode": "synthetic",
              "warning": "oracle/heuristic rows are measuring instruments or decoder self-checks, "
                         "not methods; no UC-VLA claim follows (docs/19 F7, docs/23)",
              "fits": fits}
    # --- per-cell references -------------------------------------------------------------
    ref = {}
    for split in TEST:
        for fam in REG:
            row = {n: cell(results[(n, split)], fam) for n, _, _ in base}
            if row["oracle_full"] is None:
                continue
            best_free = max(row["oracle_none"]["env_success"], row["obs_only_lookup"]["env_success"])
            row["memory_necessity"] = row["oracle_full"]["env_success"] - best_free
            ref[f"{split}/{fam}"] = row
    report["reference_cells"] = ref
    # --- sufficiency & compression efficiency (oracle with restricted context) -------------
    suff = {}
    for split in TEST:
        for fam in REG:
            full = ref.get(f"{split}/{fam}")
            if not full:
                continue
            denom = full["oracle_full"]["env_success"]
            for sel in ("last", "uniform"):
                curve = {}
                for k in BUDGETS:
                    c = cell(results[(f"oracle|{sel}|{k}", split)], fam)
                    curve[str(k)] = {"S": c["env_success"] / denom, "pair_optimal": c["pair_optimal"]}
                ks = [k for k in BUDGETS if curve[str(k)]["S"] >= 0.95]
                grid = [0, *BUDGETS]
                svals = [full["oracle_none"]["env_success"] / denom, *[curve[str(k)]["S"] for k in BUDGETS]]
                auc = sum((grid[i + 1] - grid[i]) * (svals[i] + svals[i + 1]) / 2
                          for i in range(len(grid) - 1)) / (grid[-1] - grid[0])
                suff[f"{split}/{fam}/{sel}"] = {"curve": curve, "k_star_S95": min(ks) if ks else None,
                                                "normalised_auc": auc}
    report["sufficiency_compression"] = suff
    # --- memory use (swap/deletion attribution) --------------------------------------------
    use = {}
    for name, agent, budget in base:
        if name in ("oracle_full", "nb_full", "heur_naive_full", "heur_conservative_full"):
            use[name] = {s: {f: {**row, "uses_history": uses_history(row, 0.2, 0.2, 0.6)
                                 if "drop_critical" in row else None}
                             for f, row in causal_attribution(ad, agent, s, args.run_seed,
                                                              budget).items()}
                         for s in ("test_id", "test_ood_structure")}
    report["memory_use"] = use
    # --- F7: ranking agreement between instruments (uniform minus last) --------------------
    f7, raw_p = {}, {}
    for split in TEST:
        for fam in REG:
            for k in BUDGETS:
                signs, sig_signs, row = {}, {}, {}
                for inst in ("oracle", "heur_naive", "nb"):
                    u = per_pair(results[(f"{inst}|uniform|{k}", split)])
                    w = per_pair(results[(f"{inst}|last|{k}", split)])
                    ids = sorted(i for i in u if u[i]["fam"] == fam)
                    if not ids:
                        continue
                    st = sign_test([all(u[i]["opt"]) for i in ids], [all(w[i]["opt"]) for i in ids])
                    diff = st["a_only"] - st["b_only"]
                    row[inst] = {"uniform_only": st["a_only"], "last_only": st["b_only"],
                                 "p": st["p"], "n_pairs": len(ids)}
                    signs[inst] = (diff > 0) - (diff < 0)
                    sig_signs[inst] = signs[inst] if st["p"] < 0.05 else 0
                if not row:
                    continue
                key = f"{split}/{fam}/k={k}"
                nz = {s for s in signs.values() if s != 0}
                row["agree"] = len(nz) <= 1  # PRE-REGISTERED criterion (docs/19 F7): sign only
                # secondary, POST-HOC and labelled as such: opposite *significant* signs only
                sig = {x for x in sig_signs.values() if x != 0}
                row["agree_significance_aware_posthoc"] = len(sig) <= 1
                row["status"] = ("not_a_memory_result: instruments disagree on the sign"
                                 if len(nz) > 1 else "sign_consistent_across_instruments")
                f7[key] = row
                if "oracle" in row:
                    raw_p[key] = row["oracle"]["p"]
    prim = {k: v for k, v in raw_p.items() if k.startswith("test_id/") and k.split("=")[1] in ("2", "4", "8")}
    holmp = holm(prim)
    for k, p in holmp.items():
        f7[k]["oracle_p_holm_primary_family"] = p
    report["f7_ranking_agreement"] = f7
    report["f7_summary"] = {"cells": len(f7), "disagree": sum(not v["agree"] for v in f7.values()),
                            "disagree_significance_aware_posthoc": sum(
                                not v["agree_significance_aware_posthoc"] for v in f7.values()),
                            "primary_family_size": len(prim),
                            "primary_holm_significant": sum(p < 0.05 for p in holmp.values())}
    # --- pilot variance on VAL and MDE at test size -----------------------------------------
    pilot = {}
    for fam in REG:
        for k in (2, 4, 8):
            u = per_pair(results[(f"oracle|uniform|{k}", "val")])
            w = per_pair(results[(f"oracle|last|{k}", "val")])
            ids = sorted(i for i in u if u[i]["fam"] == fam)
            st = sign_test([all(u[i]["opt"]) for i in ids], [all(w[i]["opt"]) for i in ids])
            n_test = sum(p.family == fam for p in ad.pairs("test_id"))
            pdisc = st["n_discordant"] / max(len(ids), 1)
            pilot[f"{fam}/k={k}"] = {"val_pairs": len(ids), "p_discordant": pdisc, "n_test_pairs": n_test,
                                     "mde_discordant_imbalance_at_test_n": mde(pdisc, n_test)}
    report["pilot_val_mde"] = pilot
    # --- coverage-conditioned view for the oracle sufficiency instrument ---------------------
    report["coverage_oracle_last4_uniform4"] = {
        n: {s: coverage_report(ad, agent, s, args.run_seed, budget) for s in ("test_id",)}
        for n, agent, budget in sweep if n in ("oracle|last|4", "oracle|uniform|4")}
    rep_sha = sha256_hex(canonical_json(report).encode("ascii"))
    record = build_run_record(
        kind="v2_eval", root=ROOT, data_dir=DATA,
        eval_config={"splits": list(TEST), "budgets": list(BUDGETS), "registered_primary_family":
                     "test_id x {track,events,resume} x k in {2,4,8}: uniform vs last, oracle PairOptimal, Holm"},
        agents=[a.spec() for _, a, _ in base + sweep], run_seed=args.run_seed,
        results_sha256=digest(digests), metrics_sha256=rep_sha)
    path = write_run(args.out, record, report)
    if args.ledger:
        append_ledger(args.out / "evaluation_ledger.jsonl", {
            "run_id": record["run_id"], "splits": ["val", *TEST], "purpose": "evaluation_only",
            "git_commit": record["git_commit"], "git_dirty": record["git_dirty"]})
    print(json.dumps({"run_id": record["run_id"], "results_sha256": record["results_sha256"],
                      "metrics_sha256": record["metrics_sha256"], "f7": report["f7_summary"],
                      "dir": str(path)}))


if __name__ == "__main__":
    main()
