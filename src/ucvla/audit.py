"""Anti-shortcut audits and the executable G0 gate (docs/00, docs/06 negative controls).

G0 asks: does the task have diagnostic power for history? Passing it on a *symbolic* generator
shows only that the generator is well formed. It says nothing about learned models, images or
real benchmarks (image-level shortcut audits are ``planned``).
"""

from __future__ import annotations

import math
import re
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

AUDIT_VERSION = "g0/2"
LEAK_MARGIN = 0.15
PRIOR_MARGIN = 0.12
PROBE_Z = 3.3
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


def _full_view(ep: EpisodeRecord) -> Any:
    return ep.causal_view(ep.queries[0].time_ns)


def check_pairs(adapter: SODHAdapter, pairs: Sequence[PairRecord]) -> list[Finding]:
    """Pair validity recomputed from the episodes (stored digests are never trusted).

    Relevant pairs: every channel of the decision view (all cameras, proprio, commands,
    executions, online events, query) is identical, only the history and the answer differ.
    Controls: identical decision view AND identical answer.
    """
    out = []
    for p in pairs:
        a, b = adapter.episode(p.episode_a), adapter.episode(p.episode_b)
        va, vb = _full_view(a), _full_view(b)
        if (va.digest(False), vb.digest(False)) != (p.o_t_digest_a, p.o_t_digest_b):
            out.append(Finding("error", "stale_pair_record",
                               f"{p.pair_id}: stored digests differ from recomputed"))
        da, db = va.channel_digests(), vb.channel_digests()
        differing = sorted(k for k in da if da[k] != db[k])
        if differing:
            out.append(Finding("error", "obs_differs",
                               f"{p.pair_id}: decision view differs in {differing}"))
        ta = a.target_for(a.queries[0].query_id)
        tb = b.target_for(b.queries[0].query_id)
        if (ta, tb) != (p.target_a, p.target_b):
            out.append(Finding("error", "stale_pair_record", f"{p.pair_id}: targets differ"))
        if p.history_relevant:
            if ta == tb:
                out.append(Finding("error", "same_target", f"{p.pair_id}: no divergent action"))
            if a.frames[:-2] == b.frames[:-2]:
                out.append(Finding("error", "no_history_diff", f"{p.pair_id}: identical history"))
        elif ta != tb:
            out.append(Finding("error", "control_pair_broken", p.pair_id))
    return out


_ID_GRAMMAR = {
    "episode": re.compile(r"^ep-[0-9a-f]{16}$"),
}


def id_leak_findings(episodes: Sequence[EpisodeRecord]) -> list[Finding]:
    """Identifiers must follow the opaque SODH grammar (no label-bearing suffixes)."""
    out = []
    for e in episodes:
        eid = e.episode_id
        target = e.target_for(e.queries[0].query_id)
        ok = bool(_ID_GRAMMAR["episode"].match(eid))
        ok &= e.queries[0].query_id == f"{eid}-q0"
        ok &= all(re.fullmatch(re.escape(eid) + r"-f\d+-(front|wrist)", f.frame_id)
                  for f in e.frames)
        ok &= all(re.fullmatch(re.escape(eid) + r"-a\d+", ev.event_id) for ev in e.events)
        if not ok or target in eid:
            out.append(Finding("error", "id_leak", f"{eid}: identifier outside opaque grammar"))
    return out


def _label(e: EpisodeRecord) -> str:
    return e.target_for(e.queries[0].query_id)


def _position(e: EpisodeRecord) -> int:
    """Index of the answer in the task's *natural* order (phase order if given, else options)."""
    q = e.queries[0]
    lab = _label(e)
    order = q.params.get("phase_order")
    if order:
        return order.split(",").index(lab.removeprefix("place_"))
    return q.options.index(lab)


def _probe_tokens(ep: EpisodeRecord) -> dict[str, str]:
    """Single-feature views of the decision-time information, one token type per channel/slot."""
    v = _full_view(ep)
    toks: dict[str, str] = {}
    n = len(v.ticks)
    for pos in range(min(n, 14)):  # position counted from the decision tick backwards
        for cam, d in v.ticks[n - 1 - pos].cameras.items():
            for k, val in d.items():
                toks[f"end-{pos}:{cam}:{k}"] = val
    for pos in range(min(n, 4)):  # and from the start
        for cam, d in v.ticks[pos].cameras.items():
            for k, val in d.items():
                toks[f"start+{pos}:{cam}:{k}"] = val
    for chan, dig in v.channel_digests(include_history=False).items():
        if chan not in ("task", "ticks", "query"):
            toks[f"chan:{chan}"] = dig
    for k, val in (v.query.params if v.query else {}).items():
        toks[f"param:{k}"] = val
    toks["chan:n_options"] = str(len(v.query.options) if v.query else 0)
    return toks


def token_label_probe(adapter: SODHAdapter,
                      relevant: Sequence[str] = RELEVANT_FAMILIES) -> list[Finding]:
    """Generic single-token shortcut probe over EVERY channel, not a hand-picked feature list.

    For each (family, token type): lookup value -> majority train label, evaluated on all
    non-train splits; flagged when it beats the label-prior baseline by margin and z > PROBE_Z.
    Low power on weak leaks (see docs/18); it complements, not replaces, the structural checks.
    """
    out = []
    for family in sorted({p.family for p in adapter.pairs("train")} & set(relevant)):
        # controls (history-irrelevant by design) are meant to be solvable from the observation
        train = [adapter.episode(e) for p in adapter.pairs("train") if p.family == family
                 for e in (p.episode_a, p.episode_b)]
        test = [adapter.episode(e) for s in SPLITS if s != "train" for p in adapter.pairs(s)
                if p.family == family for e in (p.episode_a, p.episode_b)]
        if len(test) < MIN_EVAL or not train:
            continue
        prior = Counter(_label(e) for e in train).most_common(1)[0][0]
        chance = sum(1 / len(e.queries[0].options) for e in test) / len(test)
        base = max(sum(_label(e) == prior for e in test) / len(test), chance)
        tr = [(_probe_tokens(e), _label(e)) for e in train]
        te = [(_probe_tokens(e), _label(e)) for e in test]
        types = sorted({k for t, _ in tr for k in t})
        for ty in types:
            tab: dict[str, Counter[str]] = defaultdict(Counter)
            for t, y in tr:
                if ty in t:
                    tab[t[ty]][y] += 1
            hit = sum((tab[t[ty]].most_common(1)[0][0] if ty in t and t[ty] in tab else prior) == y
                      for t, y in te)
            acc = hit / len(te)
            sd = math.sqrt(max(base * (1 - base), 1e-9) / len(te))
            if acc - base > LEAK_MARGIN and (acc - base) / sd > PROBE_Z:
                out.append(Finding("error", "token_shortcut",
                                   f"{family}: {ty} acc {acc:.2f} vs prior {base:.2f}"))
    return out


def label_prior_findings(adapter: SODHAdapter, margin: float = PRIOR_MARGIN) -> list[Finding]:
    """Constant-position shortcuts: "always answer position c" vs chance, pooled over #options.

    Position = index of the answer in the task's natural order. Per (family, split) and pooled
    over all non-train splits; flagged when the hit rate beats mean(1/k) by ``margin`` and z > 3.
    """
    out = []
    rows: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for s in SPLITS:
        for p in adapter.pairs(s):
            for eid in (p.episode_a, p.episode_b):
                e = adapter.episode(eid)
                row = (_position(e), len(e.queries[0].options))
                rows[(p.family, s)].append(row)
                if s != "train":
                    rows[(p.family, "all_non_train")].append(row)
    for (fam, split), rs in sorted(rows.items()):
        for c in range(max(k for _, k in rs)):
            sel = [(pos, k) for pos, k in rs if k > c]
            n = len(sel)
            if n < MIN_EVAL:
                continue
            chance = sum(1 / k for _, k in sel) / n
            rate = sum(pos == c for pos, _ in sel) / n
            z = (rate - chance) / math.sqrt(chance * (1 - chance) / n)
            if rate - chance > margin and z > 3.0:
                out.append(Finding("error", "label_prior",
                                   f"{fam}/{split}: answer position {c} rate {rate:.2f} vs "
                                   f"chance {chance:.2f} (n={n})"))
    return out


def constant_floor(adapter: SODHAdapter) -> dict[str, float]:
    """Best fixed option-index predictor per family on non-train splits (floor beside memoryless)."""
    res: dict[str, float] = {}
    by_fam: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for s in SPLITS:
        if s == "train":
            continue
        for p in adapter.pairs(s):
            for eid in (p.episode_a, p.episode_b):
                e = adapter.episode(eid)
                by_fam[p.family].append((_position(e), len(e.queries[0].options)))
    for fam, rows in sorted(by_fam.items()):
        best = max(sum(1 for i, _ in rows if i == c) for c in range(max(n for _, n in rows)))
        res[fam] = best / len(rows)
    return res


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
        chance = sum(1 / len(e.queries[0].options) for e in test) / len(test)
        base = max(sum(lab(e) == prior for e in test) / len(test), chance)
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


_NOISE_KEYS = {"arm", "noise"}


def _alpha_key(e: EpisodeRecord) -> str:
    """Name-invariant canonical form of the history plus the answer's relative position."""
    names: dict[str, str] = {}
    canon = []
    for f in e.frames:
        if f.camera_id != "cam_front":
            continue
        row = []
        for k, v in sorted(f.visible.items()):
            if k in _NOISE_KEYS:
                continue
            parts = [names.setdefault(w, f"o{len(names)}") if "_" in w and w[:1].isalpha()
                     and w not in ("pose_final",) and not w.startswith("pose_") else w
                     for w in v.replace("under:", "under: ").split(" ")]
            row.append((k, " ".join(parts)))
        canon.append(row)
    return digest([e.task, canon, _position(e), len(e.queries[0].options)])


def alpha_twin_rates(adapter: SODHAdapter) -> dict[str, dict[str, float]]:
    """Share of non-train episodes whose alpha-renamed history (arm/noise dropped) and relative
    answer also occur in TRAIN. High values mean a name-invariant learner can memorise the test
    set (review finding M4). Informational for v1 (frozen), a gate for later versions."""
    train: dict[str, set[str]] = defaultdict(set)
    for p in adapter.pairs("train"):
        for eid in (p.episode_a, p.episode_b):
            train[p.family].add(_alpha_key(adapter.episode(eid)))
    out: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        if split == "train":
            continue
        hit: Counter[str] = Counter()
        tot: Counter[str] = Counter()
        for p in adapter.pairs(split):
            for eid in (p.episode_a, p.episode_b):
                tot[p.family] += 1
                hit[p.family] += _alpha_key(adapter.episode(eid)) in train[p.family]
        out[split] = {fam: hit[fam] / tot[fam] for fam in sorted(tot)}
    return out


def g0_report(adapter: SODHAdapter, pairs: Sequence[PairRecord],
              episodes: Sequence[EpisodeRecord], assignment: Mapping[str, str]) -> dict[str, Any]:
    """Executable G0. ``passed`` is False if ANY check fails; failures are listed, not hidden."""
    findings = [*check_splits(episodes, pairs, assignment), *check_pairs(adapter, pairs),
                *id_leak_findings(episodes), *metadata_lookup_audit(adapter),
                *token_label_probe(adapter), *label_prior_findings(adapter)]
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
        "gate": "G0", "audit_version": AUDIT_VERSION, "mode": "synthetic", "passed": not errors,
        "constant_answer_floor": constant_floor(adapter),
        "alpha_twin_rate_informational": alpha_twin_rates(adapter),
        "scope": "symbolic generator well-formedness only; not a model or benchmark result",
        "obs_only": obs, "structural_pair_diffs": diffs, "history_swap": swap,
        "reference_policies": ref,
        "findings": [{"level": f.level, "code": f.code, "message": f.message} for f in findings],
    }
