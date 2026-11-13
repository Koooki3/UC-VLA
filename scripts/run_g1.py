#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1 (docs/25): train one learned-reader configuration on the frozen SODH v2 TRAIN split for
several seeds, early-stop on VAL, then evaluate once on the test splits with the harness and the
causal interventions. Writes results/synthetic/g1/runs/<tag>_s<seed>.json. CPU, synthetic."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.audit2 import front_ticks
from ucvla.canonical import HashRng
from ucvla.causal_eval import causal_attribution
from ucvla.context import AllHistory, NoHistory, ViewTick
from ucvla.dataset_io import read_dataset
from ucvla.harness import Agent, decide, evaluate, pair_metrics
from ucvla.learned import LearnedPolicy, Vocab, canon_tick, encode, train_model
from ucvla.sodh2 import PairRecord2

DATA = ROOT / "results/synthetic/sodh_v2"
TEST = ("test_id", "test_ood_length", "test_ood_structure", "test_counterfactual")
REL = ("track", "events", "resume")


def example(ad, eid, vocab):
    e = ad.episode(eid)
    q = e.queries[0]
    ticks = front_ticks(e)
    n = len(ticks) - 1
    x = encode(vocab, q.task, q.params, q.options, dict(enumerate(ticks[:-1])), ticks[-1], n)
    return x, q.options.index(e.target_for(q.query_id))


def order_perturbation(ad, agent, split, seed):
    """PairOptimal with history tick CONTENTS permuted (timestamps kept)."""
    both = []
    for p in ad.pairs(split):
        if not p.history_relevant:
            continue
        ok = []
        for eid in (p.episode_a, p.episode_b):
            e = ad.episode(eid)
            view = ad.view(eid)
            contents = [t.cameras for t in view.history]
            HashRng("order", seed, eid).shuffle(contents)
            hist = tuple(ViewTick(t.time_ns, c) for t, c in zip(view.history, contents))
            d, _, _ = decide(agent, view, seed, eid, None, history=hist)
            ok.append(d == e.target_for(e.queries[0].query_id))
        both.append((p.family, all(ok)))
    return {f: float(np.mean([b for fam, b in both if fam == f])) for f in REL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["gru", "ctransformer", "fixedlatent"])
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/g1/runs")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    eps, pairs, asg = read_dataset(DATA, PairRecord2)
    ad = SODHAdapter(eps, pairs, asg)
    tr_ids = [e for p in ad.pairs("train") for e in (p.episode_a, p.episode_b)]
    va_pairs_rec = ad.pairs("val")
    va_ids = [e for p in va_pairs_rec for e in (p.episode_a, p.episode_b)]
    toklists = []
    for eid in tr_ids:
        e = ad.episode(eid)
        items = e.queries[0].params.get("items", "").split(",") if "items" in e.queries[0].params else []
        for t in front_ticks(e):
            toklists.append(canon_tick(t, items))
    vocab = Vocab.build(toklists)
    train = [example(ad, i, vocab) for i in tr_ids]
    val = [example(ad, i, vocab) for i in va_ids]
    val_pairs = [(2 * i, 2 * i + 1) for i in range(len(va_pairs_rec))]
    args.out.mkdir(parents=True, exist_ok=True)
    tag = args.arch + (f"{args.slots}" if args.arch == "fixedlatent" else "")
    for seed in args.seeds:
        model, info = train_model(args.arch, vocab, train, val, val_pairs, seed, args.slots, args.epochs)
        pol = LearnedPolicy(model, vocab, tag)
        full, none = Agent(AllHistory(), pol), Agent(NoHistory(), pol)
        res = {"arch": tag, "seed": seed, "spec": pol.spec(), "train": info["trace"][-1],
               "epochs": len(info["trace"]) - 1, "splits_evaluated": list(TEST), "metrics": {},
               "memoryless": {}, "attribution": {}, "order_perturbation": {}}
        for s in TEST:
            r = evaluate(ad, full, s, seed)
            res["metrics"][s] = pair_metrics(r)
            res["memoryless"][s] = pair_metrics(evaluate(ad, none, s, seed))
        for s in ("test_id", "test_ood_structure"):
            res["attribution"][s] = causal_attribution(ad, full, s, seed)
            res["order_perturbation"][s] = order_perturbation(ad, full, s, seed)
        (args.out / f"{tag}_s{seed}.json").write_text(json.dumps(res, indent=1, default=float))
        m = res["metrics"]["test_id"]
        print(tag, seed, "epochs", res["epochs"], "train_acc", round(res["train"]["final_train_acc"], 2),
              "val_pair", round(res["train"]["best_val_pair_optimal"], 2),
              {f[:3]: round(m[f]["pair_both_correct"], 2) for f in REL}, flush=True)


if __name__ == "__main__":
    main()
