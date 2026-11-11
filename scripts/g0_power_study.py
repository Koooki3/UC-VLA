#!/usr/bin/env python3
"""Detection-power study of the SODH v2 leak probes (docs/21). Plants graded leaks in a COPY of
the events family (binary labels) held in memory, runs the current probes (D0) and the extended
aggregate/conjunctive probes (D1), and reports detection rate vs the accuracy a leak-exploiting
reader would reach. Frozen data on disk is never modified."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.adapters import SODHAdapter
from ucvla.audit import label_prior_findings, metadata_lookup_audit, token_label_probe
from ucvla.audit2 import extended_probe_findings
from ucvla.canonical import HashRng
from ucvla.dataset_io import read_dataset
from ucvla.episode import EpisodeRecord, Frame
from ucvla.sodh2 import RELEVANT2, PairRecord2
from ucvla.splits import SPLITS

DATA = ROOT / "results/synthetic/sodh_v2"
LEAK_CODES = {"token_shortcut", "metadata_shortcut", "agg_shortcut", "pair_shortcut"}


def lab_bit(e: EpisodeRecord) -> int:
    return int(e.target_for(e.queries[0].query_id) == "stop")


def rebuild(e: EpisodeRecord, fn) -> EpisodeRecord:
    ts = e.tick_times()
    frames = tuple(Frame.make(f.frame_id, f.camera_id, f.time_ns, fn(ts.index(f.time_ns), len(ts), dict(f.visible))
                              if f.camera_id == "cam_front" else dict(f.visible)) for f in e.frames)
    return dataclasses.replace(e, frames=frames)


def plant(kind: str, strength: float, e: EpisodeRecord, seed: int) -> EpisodeRecord:
    rng = HashRng("leak", kind, strength, seed, e.episode_id)
    lb = lab_bit(e)
    flip = lambda p: rng.below(1000) < int(p * 1000)
    if kind == "single_token":  # linear: one token copies the label with probability rho
        bit = lb if flip(strength) else rng.below(2)
        return rebuild(e, lambda i, n, v: {**v, "aux": str(bit)} if i == n - 2 else v)
    if kind == "xor_pair":  # nonlinear: label = b1 xor b2, each token alone is uninformative
        b1 = rng.below(2)
        b2 = (b1 ^ lb) if flip(strength) else rng.below(2)
        return rebuild(e, lambda i, n, v: {**v, "aux1": str(b1)} if i == n - 2 else
                       ({**v, "aux2": str(b2)} if i == n - 3 else v))
    if kind == "parity_across_ticks":  # cross-time: parity of a bit stream spread over the history
        n_h = len(e.tick_times()) - 1
        bits = [rng.below(2) for _ in range(n_h)]
        want = lb if flip(strength) else rng.below(2)
        bits[-1] = (want ^ (sum(bits[:-1]) % 2)) & 1
        return rebuild(e, lambda i, n, v: {**v, "aux": str(bits[i])} if i < n - 1 else v)
    if kind == "rate_code":  # cross-time: label shifts the rate of a token (strength = delta)
        p = 0.5 + strength * (2 * lb - 1)
        bits = [int(rng.below(1000) < int(p * 1000)) for _ in range(len(e.tick_times()) - 1)]
        return rebuild(e, lambda i, n, v: {**v, "aux": str(bits[i])} if i < n - 1 else v)
    raise ValueError(kind)


def reader_acc(kind: str, eps: list[EpisodeRecord]) -> float:
    hit = 0
    for e in eps:
        ticks = [dict(f.visible) for f in e.frames if f.camera_id == "cam_front"]
        hist = ticks[:-1]
        if kind == "single_token":
            pred = int(hist[-1]["aux"])
        elif kind == "xor_pair":
            pred = int(hist[-1]["aux1"]) ^ int(hist[-2]["aux2"])
        elif kind == "parity_across_ticks":
            pred = sum(int(t["aux"]) for t in hist) % 2
        else:
            pred = int(sum(int(t["aux"]) for t in hist) * 2 > len(hist))
        hit += pred == lab_bit(e)
    return hit / len(eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--null-reps", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "results/synthetic/g0_power_study.json")
    args = ap.parse_args()
    eps, pairs, asg = read_dataset(DATA, PairRecord2)
    keep = {p.pair_id for p in pairs if p.family == "events"}
    pairs = [p for p in pairs if p.pair_id in keep]
    ids = {x for p in pairs for x in (p.episode_a, p.episode_b)}
    base = [e for e in eps if e.episode_id in ids]
    grid = {"single_token": (0.0, 0.2, 0.4, 0.6, 1.0), "xor_pair": (0.0, 0.2, 0.4, 0.6, 1.0),
            "parity_across_ticks": (0.0, 0.2, 0.4, 0.6, 1.0), "rate_code": (0.0, 0.05, 0.1, 0.2, 0.3)}
    rows = []
    for kind, strengths in grid.items():
        for st in strengths:
            reps = args.null_reps if st == 0.0 else args.reps
            d0 = d1 = 0
            accs = []
            for seed in range(reps):
                leaked = [plant(kind, st, e, seed) for e in base]
                ad = SODHAdapter(leaked, pairs, asg)
                f0 = [*token_label_probe(ad, RELEVANT2), *metadata_lookup_audit(ad),
                      *label_prior_findings(ad)]
                f1 = [*f0, *extended_probe_findings(ad)]
                hit = lambda fs: any(f.code in LEAK_CODES and f.message.startswith("events") for f in fs)
                d0 += hit(f0)
                d1 += hit(f1)
                test = [e for e in leaked for s in SPLITS if s != "train"
                        for p in ad.pairs(s) if e.episode_id in (p.episode_a, p.episode_b)]
                accs.append(reader_acc(kind, test))
            rows.append({"leak": kind, "strength": st, "replicates": reps,
                         "leak_reader_accuracy": sum(accs) / len(accs),
                         "detect_rate_current_probes": d0 / reps,
                         "detect_rate_extended_probes": d1 / reps})
            print(rows[-1], flush=True)
    out = {"mode": "synthetic", "family": "events (binary label, chance 0.5)",
           "n_episodes": len(base), "detectors": {
               "D0": "g0v2/1 probes: single token per (position, channel), metadata lookups, label prior",
               "D1": "D0 + aggregate (count/parity over history) + conjunctive (token pairs) probes"},
           "note": "strength 0.0 rows are the null (false-positive) rate", "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
