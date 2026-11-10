"""Anti-shortcut audits and the executable G0 gate (docs/00, docs/06 negative controls).

G0 asks: does the task have diagnostic power for history? Passing it on a *symbolic* generator
shows only that the generator is well formed. It says nothing about learned models, images or
real benchmarks (image-level shortcut audits are ``planned``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from ucvla.adapters import SODHAdapter, evaluate_policy, summarize
from ucvla.baselines import full_history, memoryless
from ucvla.canonical import digest
from ucvla.episode import EpisodeRecord
from ucvla.sodh import RELEVANT_FAMILIES, PairRecord
from ucvla.splits import SPLITS, Finding, check_splits

LEAK_MARGIN = 0.15
MIN_EVAL = 20
CONTROL_MIN_ACC = 0.9


def _front_wrist(ep: EpisodeRecord) -> list[Any]:
    return [(f.camera_id, f.time_ns, dict(f.visible)) for f in ep.frames if f.camera_id != "cam_front"]


# Features that carry no history semantics: they must not differ inside a relevant pair.
STRUCTURAL: Mapping[str, Callable[[EpisodeRecord], Any]] = {
    "n_ticks": lambda e: len(e.tick_times()),
    "n_frames": lambda e: len(e.frames),
    "last_time_ns": lambda e: e.tick_times()[-1],
    "instruction": lambda e: e.queries[0].instruction,
    "options": lambda e: e.queries[0].options,
    "params": lambda e: dict(e.queries[0].params),
    "secondary_cameras": _front_wrist,
}
# Features whose lookup-table accuracy is compared to the label prior.
LOOKUP: Mapping[str, Callable[[EpisodeRecord], Any]] = {
    **STRUCTURAL,
    "first_tick": lambda e: digest([dict(f.visible) for f in e.frames[:2]]),
}


def check_pairs(adapter: SODHAdapter, pairs: Sequence[PairRecord]) -> list[Finding]:
    """A4: relevant pairs share the decision observation but not the answer; controls share both."""
    out = []
    for p in pairs:
        same_obs = p.o_t_digest_a == p.o_t_digest_b
        a, b = adapter.episode(p.episode_a), adapter.episode(p.episode_b)
        if p.history_relevant:
            if not same_obs:
                out.append(Finding("error", "obs_differs", f"{p.pair_id}: decision obs differ"))
            if p.target_a == p.target_b:
                out.append(Finding("error", "same_target", f"{p.pair_id}: no divergent action"))
            if a.frames[:-2] == b.frames[:-2]:
                out.append(Finding("error", "no_history_diff", f"{p.pair_id}: identical history"))
        elif not same_obs or p.target_a != p.target_b:
            out.append(Finding("error", "control_pair_broken", p.pair_id))
    return out


def id_leak_findings(episodes: Sequence[EpisodeRecord]) -> list[Finding]:
    out = []
    for e in episodes:
        target = e.target_for(e.queries[0].query_id)
        ids = [e.episode_id, e.queries[0].query_id, *(f.frame_id for f in e.frames),
               *(ev.event_id for ev in e.events)]
        if any(target in i for i in ids):
            out.append(Finding("error", "id_leak", f"{e.episode_id}: identifier contains label"))
    return out


def structural_pair_diffs(adapter: SODHAdapter, pairs: Sequence[PairRecord]) -> dict[str, float]:
    rel = [p for p in pairs if p.history_relevant]
    res = {}
    for name, fn in STRUCTURAL.items():
        res[name] = (sum(fn(adapter.episode(p.episode_a)) != fn(adapter.episode(p.episode_b))
                         for p in rel) / len(rel)) if rel else 0.0
    return res


def obs_only_audit(adapter: SODHAdapter) -> dict[str, dict[str, float]]:
    """Memorising lookup from the decision observation to the label (a current-image-only proxy).

    ``conflict_fraction`` is structural: pairs whose two members share an observation but not an
    answer, which caps ANY observation-only policy at 50 % on those pairs.
    """
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for p in adapter.pairs("train"):
        for eid in (p.episode_a, p.episode_b):
            ep = adapter.episode(eid)
            table[adapter.view(eid).digest(False)][ep.target_for(ep.queries[0].query_id)] += 1
    out: dict[str, dict[str, float]] = {}
    for family in sorted({p.family for s in SPLITS for p in adapter.pairs(s)}):
        hit = n = conflict = n_pairs = 0
        for split in SPLITS:
            if split == "train":
                continue
            for p in adapter.pairs(split):
                if p.family != family:
                    continue
                n_pairs += 1
                conflict += (p.o_t_digest_a == p.o_t_digest_b) and (p.target_a != p.target_b)
                for eid in (p.episode_a, p.episode_b):
                    ep = adapter.episode(eid)
                    d = adapter.view(eid).digest(False)
                    opts = ep.queries[0].options
                    pred = (table[d].most_common(1)[0][0] if d in table
                            else opts[int(d[:8], 16) % len(opts)])
                    hit += pred == ep.target_for(ep.queries[0].query_id)
                    n += 1
        out[family] = {"obs_only_acc": hit / n if n else float("nan"),
                       "conflict_fraction": conflict / n_pairs if n_pairs else float("nan"),
                       "n_episodes": n}
    return out


def metadata_lookup_audit(adapter: SODHAdapter, margin: float = LEAK_MARGIN) -> list[Finding]:
    """Flag a non-history feature whose lookup beats the train label prior by > ``margin``."""
    out = []
    families = sorted({p.family for p in adapter.pairs("train")})
    for family in families:
        train = [adapter.episode(e) for p in adapter.pairs("train") if p.family == family
                 for e in (p.episode_a, p.episode_b)]
        test = [adapter.episode(e) for s in SPLITS if s != "train" for p in adapter.pairs(s)
                if p.family == family for e in (p.episode_a, p.episode_b)]
        if len(test) < MIN_EVAL or not train:
            continue
        lab = lambda e: e.target_for(e.queries[0].query_id)
        prior = Counter(lab(e) for e in train).most_common(1)[0][0]
        base = sum(lab(e) == prior for e in test) / len(test)
        for name, fn in LOOKUP.items():
            tab: dict[str, Counter[str]] = defaultdict(Counter)
            for e in train:
                tab[digest(fn(e))][lab(e)] += 1
            acc = sum((tab[digest(fn(e))].most_common(1)[0][0] if digest(fn(e)) in tab
                       else prior) == lab(e) for e in test) / len(test)
            if acc - base > margin:
                out.append(Finding("error", "metadata_shortcut",
                                   f"{family}: feature {name} acc {acc:.2f} vs prior {base:.2f}"))
    return out


def history_swap_audit(adapter: SODHAdapter) -> dict[str, float]:
    """Swap the history between the members of each pair, keeping the (identical) current frame.

    Oracle (full-history) decisions must follow the swapped history for relevant families and be
    invariant for the control; the memoryless reader must be invariant everywhere.
    """
    full, none = full_history(), memoryless()
    follow: dict[str, list[bool]] = defaultdict(list)
    inv_none: list[bool] = []
    for s in SPLITS:
        for p in adapter.pairs(s):
            va, vb = adapter.view(p.episode_a), adapter.view(p.episode_b)
            spliced = replace(va, ticks=vb.ticks[:-1] + va.ticks[-1:])
            expected = p.target_b if p.history_relevant else p.target_a
            follow[p.family].append(full.decide(spliced) == expected)
            inv_none.append(none.decide(spliced) == none.decide(va))
    res = {f"{fam}.swap_follows_history": sum(v) / len(v) for fam, v in sorted(follow.items())}
    res["memoryless.swap_invariance"] = sum(inv_none) / len(inv_none)
    return res


def g0_report(adapter: SODHAdapter, pairs: Sequence[PairRecord],
              episodes: Sequence[EpisodeRecord], assignment: Mapping[str, str]) -> dict[str, Any]:
    """Executable G0. ``passed`` is False if ANY check fails; failures are listed, not hidden."""
    findings = [*check_splits(episodes, pairs, assignment), *check_pairs(adapter, pairs),
                *id_leak_findings(episodes), *metadata_lookup_audit(adapter)]
    diffs = structural_pair_diffs(adapter, pairs)
    findings += [Finding("error", "structural_diff", f"{k} differs inside {v:.0%} of pairs")
                 for k, v in diffs.items() if v > 0]
    obs = obs_only_audit(adapter)
    for fam, r in obs.items():
        if fam in RELEVANT_FAMILIES and (r["obs_only_acc"] > 0.5 or r["conflict_fraction"] < 1.0):
            findings.append(Finding("error", "obs_only_solves", f"{fam}: {r}"))
        if fam not in RELEVANT_FAMILIES and r["obs_only_acc"] < CONTROL_MIN_ACC:
            findings.append(Finding("error", "control_not_obs_solvable", f"{fam}: {r}"))
    swap = history_swap_audit(adapter)
    for k, v in swap.items():
        if v < 1.0:
            findings.append(Finding("error", "history_swap", f"{k} = {v:.3f}"))
    ref: dict[str, dict[str, float]] = {}
    for name, pol in (("memoryless", memoryless()), ("full_history", full_history())):
        ref[name] = {}
        for s in SPLITS:
            for fam, m in summarize(evaluate_policy(adapter, pol, s)).items():
                ref[name][f"{s}.{fam}.pair_both_correct"] = m["pair_both_correct"]
                if fam in RELEVANT_FAMILIES and name == "memoryless" and m["pair_both_correct"] > 0:
                    findings.append(Finding("error", "memoryless_solves", f"{s}/{fam}"))
                if fam in RELEVANT_FAMILIES and name == "full_history" and m["pair_both_correct"] < 1:
                    findings.append(Finding("error", "reader_unsolvable", f"{s}/{fam}"))
    errors = [f for f in findings if f.level == "error"]
    return {
        "gate": "G0", "mode": "synthetic", "passed": not errors,
        "scope": "symbolic generator well-formedness only; not a model or benchmark result",
        "obs_only": obs, "structural_pair_diffs": diffs, "history_swap": swap,
        "reference_policies": ref,
        "findings": [{"level": f.level, "code": f.code, "message": f.message} for f in findings],
    }
