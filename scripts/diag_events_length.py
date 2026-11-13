#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose the events length-OOD collapse on NON-sealed v2 data (docs/45).
H1: failure when the target count K exceeds the training range (counter/threshold saturation).
H2: failure from sequence length per se even when K is within the training range.
Trains GRU (docs/25 hyper-parameters, mix3000 auxiliary data) for 3 seeds, then stratifies events
accuracy on test_id and test_ood_length by K and horizon; plus a controlled probe that pads
in-distribution histories with blank ticks at the START (length without extra evidence)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.run_g1 import example
from ucvla.adapters import SODHAdapter
from ucvla.audit2 import front_ticks
from ucvla.context import AllHistory, ViewTick
from ucvla.dataset_io import read_dataset
from ucvla.harness import Agent, decide
from ucvla.learned import LearnedPolicy, Vocab, canon_tick, train_model
from ucvla.sodh2 import PairRecord2, SODH2Config, generate_suite2

DATA = ROOT / "results/synthetic/sodh_v2"
BLANK = {"phase": "work", "contact": "up", "lamp": "off"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="configs/sodh/g1_trainaug_mix_3000.yaml")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    torch.set_num_threads(4)
    eps, pairs, asg = read_dataset(DATA, PairRecord2)
    ad = SODHAdapter(eps, pairs, asg)
    tr = [e for p in ad.pairs("train") for e in (p.episode_a, p.episode_b)]
    vocab = Vocab.build([canon_tick(t, []) for e in tr for t in front_ticks(ad.episode(e))])
    train = [example(ad, i, vocab) for i in tr]
    va_p = ad.pairs("val")
    val = [example(ad, e, vocab) for p in va_p for e in (p.episode_a, p.episode_b)]
    xs = generate_suite2(SODH2Config.from_dict(yaml.safe_load((ROOT / args.extra).read_text())))
    xad = SODHAdapter(xs.episodes, xs.pairs, {p.pair_id: "train" for p in xs.pairs})
    train += [example(xad, e.episode_id, vocab) for e in xs.episodes]
    train_k = sorted({int(xad.episode(e.episode_id).queries[0].params["target_count"]) for e in xs.episodes if e.provenance["family"] == "events"})
    out = {"train_K_range": [train_k[0], train_k[-1]], "seeds": []}
    for seed in args.seeds:
        model, tr_trace = train_model("gru", vocab, train, val, [(2 * i, 2 * i + 1) for i in range(len(va_p))], seed, lr=args.lr)
        agent = Agent(AllHistory(), LearnedPolicy(model, vocab, "gru"))
        res = {"seed": seed, "lr": args.lr, "n_epochs": len(tr_trace), "by_K": {}, "by_horizon": {}, "padded_indist": {}}
        for split in ("test_id", "test_ood_length"):
            byk, byh = defaultdict(list), defaultdict(list)
            for p in ad.pairs(split):
                if p.family != "events":
                    continue
                for eid in (p.episode_a, p.episode_b):
                    e = ad.episode(eid)
                    d, _, _ = decide(agent, ad.view(eid), seed, eid, None)
                    ok = d == e.target_for(e.queries[0].query_id)
                    byk[int(e.queries[0].params["target_count"])].append(ok)
                    byh[int(e.provenance["horizon"]) // 5 * 5].append(ok)
            res["by_K"][split] = {k: [float(np.mean(v)), len(v)] for k, v in sorted(byk.items())}
            res["by_horizon"][split] = {k: [float(np.mean(v)), len(v)] for k, v in sorted(byh.items())}
        # controlled: pad in-dist events histories with blank ticks at the start (same evidence, longer)
        for pad in (0, 10, 20):
            oks = []
            for p in ad.pairs("test_id"):
                if p.family != "events":
                    continue
                for eid in (p.episode_a, p.episode_b):
                    e = ad.episode(eid)
                    view = ad.view(eid)
                    hist = view.history
                    step = hist[1].time_ns - hist[0].time_ns
                    t0 = hist[0].time_ns
                    pre = tuple(ViewTick(t0 - (pad - i) * step + pad * step - pad * step, {"cam_front": dict(BLANK), "cam_wrist": {"gripper": "open"}}) for i in range(pad))
                    # shift all times so ticks stay on the grid: prepend by relabelling times sequentially
                    allt = list(pre) + list(hist)
                    shifted = tuple(ViewTick(t0 + i * step, t.cameras) for i, t in enumerate(allt))
                    cur = view.current
                    # current tick time must follow the last history tick: build a view with shifted current time
                    object.__setattr__(view, "ticks", (*shifted, ViewTick(t0 + len(shifted) * step, cur.cameras)))
                    d, _, _ = decide(agent, view, seed, eid, None, history=shifted)
                    oks.append(d == e.target_for(e.queries[0].query_id))
            res["padded_indist"][pad] = float(np.mean(oks))
        out["seeds"].append(res)
        print("seed", seed, json.dumps(res["padded_indist"]), flush=True)
    (ROOT / f"results/synthetic/g1/events_length_diagnosis_{args.tag}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
