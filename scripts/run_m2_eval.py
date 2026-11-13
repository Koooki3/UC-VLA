#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M2 CPU evaluation: matched-budget agents, causal attribution, paired tests, run record.

mode: synthetic. The oracle decoder is an information-sufficiency probe, NOT a method; nothing
here is a UC-VLA result. Test splits are used for evaluation only (see the ledger).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.canonical import canonical_json, digest, sha256_hex
from ucvla.causal_eval import (
    access_parity_digest,
    causal_attribution,
    coverage_report,
    uses_history,
)
from ucvla.context import AllHistory, ContextBudget, LastK, NoHistory, UniformK
from ucvla.dataset_io import read_dataset
from ucvla.harness import Agent, evaluate, fit_agent, pair_metrics, results_digest
from ucvla.readers import BagOfTokensNB, ObsOnlyLookup, OracleDecoder
from ucvla.runrecord import append_ledger, build_run_record, write_run
from ucvla.sodh import RELEVANT_FAMILIES
from ucvla.stattests import holm, sign_test

DATA = ROOT / "results/synthetic/sodh_v1"
TEST_SPLITS = ("test_id", "test_ood", "test_counterfactual")
BUDGETS = (1, 2, 4, 8, 16)


def agent_suite() -> list[tuple[Agent, ContextBudget | None]]:
    suite: list[tuple[Agent, ContextBudget | None]] = [
        (Agent(AllHistory(), OracleDecoder("strict")), None),  # decoder self-check, unbounded
        (Agent(AllHistory(), OracleDecoder("clock")), None),
        (Agent(NoHistory(), OracleDecoder("strict")), ContextBudget("ticks", 0)),
        (Agent(NoHistory(), ObsOnlyLookup()), ContextBudget("ticks", 0)),
        (Agent(AllHistory(), BagOfTokensNB()), None),
    ]
    for k in BUDGETS:
        b = ContextBudget("ticks", k)
        for v in ("strict", "clock"):
            suite += [(Agent(LastK(k), OracleDecoder(v)), b),
                      (Agent(UniformK(k), OracleDecoder(v)), b)]
    return suite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/m2")
    ap.add_argument("--ledger", action="store_true", help="append to the evaluation ledger")
    args = ap.parse_args()

    eps, pairs, asg = read_dataset(DATA)
    adapter = SODHAdapter(eps, pairs, asg)
    rows, per_pair, attribution, coverage, fits, all_results = [], {}, {}, {}, {}, []
    suite = agent_suite()
    for agent, budget in suite:
        fits[agent.name] = fit_agent(adapter, agent, args.run_seed, "train", budget)
        for split in TEST_SPLITS:
            res = evaluate(adapter, agent, split, args.run_seed, budget)
            all_results.append(results_digest(res))
            rows.append({"agent": agent.name, "split": split, "metrics": pair_metrics(res)})
            for r in res:
                per_pair.setdefault((agent.name, split, r.family), {}).setdefault(
                    r.pair_id, []).append(r.correct)
        attribution[agent.name] = {
            split: {fam: {**row, "uses_history": uses_history(row) if "drop_critical" in row else None}
                    for fam, row in causal_attribution(adapter, agent, split, args.run_seed,
                                                       budget).items()}
            for split in ("test_id", "test_ood")}
        if budget is not None and budget.limit:
            coverage[agent.name] = {split: coverage_report(adapter, agent, split, args.run_seed,
                                                           budget) for split in TEST_SPLITS}
    # paired exact tests: uniform_k vs last_k at the same tick budget, pair-both-correct, Holm
    tests, raw_p = {}, {}
    for split in TEST_SPLITS:
        for fam in sorted(RELEVANT_FAMILIES):
            for k in BUDGETS:
                for v in ("strict", "clock"):
                    u = per_pair[(f"uniform_{k}+oracle_decoder_{v}", split, fam)]
                    w = per_pair[(f"last_{k}+oracle_decoder_{v}", split, fam)]
                    ids = sorted(u)
                    key = f"{split}/{fam}/k={k}/{v}"
                    tests[key] = sign_test([all(u[i]) for i in ids], [all(w[i]) for i in ids])
                    raw_p[key] = tests[key]["p"]
    for key, adj in holm(raw_p).items():
        tests[key]["p_holm"] = adj
    parity = {s: access_parity_digest(adapter, [a for a, _ in suite[:3]], s) for s in TEST_SPLITS}
    report = {
        "mode": "synthetic",
        "warning": ("oracle_decoder rows are DECODER SELF-CHECKS / information-sufficiency probes; "
                    "they are not method performance. No UC-VLA claim follows from this report."),
        "metric_note": "pair_both_correct is the history-sensitive metric; chance for any "
                       "observation-only policy is 0 on history-relevant families",
        "fits": fits, "results": rows, "attribution": attribution, "coverage": coverage,
        "decoder_variants_note": "strict vs clock decoders bracket how much a ranking depends on "
                                 "decoder priors; a ranking that flips between them is not a "
                                 "memory result",
        "uniform_vs_last_sign_tests": tests, "access_parity": parity,
    }
    report_sha = sha256_hex(canonical_json(report).encode("ascii"))
    record = build_run_record(
        kind="m2_symbolic_eval", root=ROOT, data_dir=DATA,
        eval_config={"splits": list(TEST_SPLITS), "budgets_ticks": list(BUDGETS),
                     "budget_unit": "ticks"},
        agents=[a.spec() for a, _ in suite], run_seed=args.run_seed,
        results_sha256=digest(all_results), metrics_sha256=report_sha)
    path = write_run(args.out, record, report)
    if args.ledger:
        append_ledger(args.out / "evaluation_ledger.jsonl", {
            "run_id": record["run_id"], "splits": list(TEST_SPLITS), "purpose": "evaluation_only",
            "git_commit": record["git_commit"], "git_dirty": record["git_dirty"]})
    print(json.dumps({"run_id": record["run_id"], "results_sha256": record["results_sha256"],
                      "metrics_sha256": report_sha, "dir": str(path)}))


if __name__ == "__main__":
    main()
