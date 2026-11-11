"""SODH v2 gate (docs/19 §5, §9). Reuses the g0/2 full-view audits and adds mechanism/template/
oracle/size gates and the history-free falsifier. Validity statement: "validated against the
current attack set" (tests/test_g0_mutants.py, tests/test_v2.py); NOT a proof of leak-freeness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ucvla.adapters import SODHAdapter
from ucvla.audit import (
    alpha_twin_rates,
    check_pairs,
    constant_floor,
    id_leak_findings,
    label_prior_findings,
    metadata_lookup_audit,
    token_label_probe,
)
from ucvla.canonical import digest
from ucvla.episode import EpisodeRecord
from ucvla.oracle2 import posterior
from ucvla.sodh import TICK_NS
from ucvla.sodh2 import (
    POSITIONS,
    RELEVANT2,
    TRAIN_MECHANISMS,
    PairRecord2,
    _pos_range,
    oracle_for_ticks,
)
from ucvla.splits import SPLITS, Finding, check_splits

AUDIT_VERSION = "g0v2/2"  # /1 = single-token probes only; /2 adds aggregate + conjunctive probes
MIN_TEST_ID_PAIRS = 60
MIN_OOD_PAIRS = 40
IN_DIST_SPLITS = ("train", "val", "test_id")
OOD_SPLITS = ("test_ood_length", "test_ood_structure", "test_counterfactual")


def front_ticks(ep: EpisodeRecord) -> list[Mapping[str, str]]:
    return [dict(f.visible) for f in ep.frames if f.camera_id == "cam_front"]


def oracle_recompute_findings(episodes: Sequence[EpisodeRecord]) -> list[Finding]:
    """Every stored label must equal the oracle's action recomputed from the episode's own frames."""
    out = []
    for e in episodes:
        q = e.queries[0]
        got, _ = oracle_for_ticks(q.task, q.params, q.options, front_ticks(e))
        if got != e.target_for(q.query_id):
            out.append(Finding("error", "label_not_oracle", e.episode_id))
    return out


def mechanism_findings(pairs: Sequence[PairRecord2], assignment: Mapping[str, str]) -> list[Finding]:
    out = []
    train_mech = {m for p in pairs if assignment.get(p.pair_id) in IN_DIST_SPLITS
                  for m in p.mechanisms.split(",")}
    allowed = {m for ms in TRAIN_MECHANISMS.values() for m in ms}
    if not train_mech <= allowed:
        out.append(Finding("error", "train_has_ood_mechanism", str(sorted(train_mech - allowed))))
    for fam in sorted({p.family for p in pairs if p.family in RELEVANT2}):
        struct = [p for p in pairs if p.family == fam
                  and assignment.get(p.pair_id) == "test_ood_structure"]
        if not struct:
            continue
        fam_train = {m for p in pairs if p.family == fam
                     and assignment.get(p.pair_id) in IN_DIST_SPLITS
                     for m in p.mechanisms.split(",")}
        if any(set(p.mechanisms.split(",")) <= fam_train for p in struct):
            out.append(Finding("error", "ood_structure_not_structural",
                               f"{fam}: ood_structure pair reuses only train mechanisms"))
    return out


def template_findings(pairs: Sequence[PairRecord2]) -> list[Finding]:
    ids = [p.template_id for p in pairs]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    return [Finding("error", "template_not_unique", f"{len(dup)} duplicated template ids")] if dup \
        else []


def structure_signature(ep: EpisodeRecord) -> str:
    """Coarse event-structure signature (family, mechanism, multiset of token kinds) — informational."""
    kinds: Counter[str] = Counter()
    for t in front_ticks(ep):
        for k, v in t.items():
            if k in ("phase", "noise", "flag"):
                continue
            kinds[f"{k}={'name' if '_' in v and v[:1].isalpha() else v}"] += 1
    return digest([ep.provenance["family"], ep.provenance["mechanisms"], sorted(kinds.items())])


def structure_overlap(adapter: SODHAdapter) -> dict[str, dict[str, float]]:
    train = defaultdict(set)
    for p in adapter.pairs("train"):
        for e in (p.episode_a, p.episode_b):
            train[p.family].add(structure_signature(adapter.episode(e)))
    out: dict[str, dict[str, float]] = {}
    for s in SPLITS:
        if s == "train":
            continue
        hit, tot = Counter(), Counter()
        for p in adapter.pairs(s):
            for e in (p.episode_a, p.episode_b):
                tot[p.family] += 1
                hit[p.family] += structure_signature(adapter.episode(e)) in train[p.family]
        if tot:
            out[s] = {f: hit[f] / tot[f] for f in sorted(tot)}
    return out


def size_findings(adapter: SODHAdapter) -> list[Finding]:
    out = []
    for fam in RELEVANT2:
        n_id = sum(p.family == fam for p in adapter.pairs("test_id"))
        if n_id < MIN_TEST_ID_PAIRS:
            out.append(Finding("error", "test_id_too_small", f"{fam}: {n_id} < {MIN_TEST_ID_PAIRS}"))
        for s in OOD_SPLITS:
            n = sum(p.family == fam for p in adapter.pairs(s))
            if 0 < n < MIN_OOD_PAIRS:
                out.append(Finding("error", "ood_too_small", f"{fam}/{s}: {n} < {MIN_OOD_PAIRS}"))
    return out


def oracle_value(adapter: SODHAdapter) -> dict[str, dict[str, float]]:
    """Expected env success of the posterior oracle with FULL history (informational ceiling)."""
    out: dict[str, dict[str, float]] = {}
    for s in SPLITS:
        hit, tot = Counter(), Counter()
        for p in adapter.pairs(s):
            for eid in (p.episode_a, p.episode_b):
                e = adapter.episode(eid)
                tot[p.family] += 1
                hit[p.family] += e.target_for(e.queries[0].query_id) in str(
                    e.privileged.state["success_actions"]).split("|")
        if tot:
            out[s] = {f: hit[f] / tot[f] for f in sorted(tot)}
    return out


MIN_ORACLE_ENV_SUCCESS = 0.70


def oracle_env_findings(adapter: SODHAdapter) -> list[Finding]:
    """Env-defined success must agree with the oracle (catches generator/oracle disagreement)."""
    out = []
    for split, fams in oracle_value(adapter).items():
        if split not in IN_DIST_SPLITS:
            continue
        for fam, v in fams.items():
            if fam in RELEVANT2 and v < MIN_ORACLE_ENV_SUCCESS:
                out.append(Finding("error", "oracle_env_success_low",
                                   f"{split}/{fam}: {v:.2f} < {MIN_ORACLE_ENV_SUCCESS}"))
    return out


def position_findings(adapter: SODHAdapter) -> list[Finding]:
    """The recorded evidence_position tag must match the decisive tick it claims."""
    out = []
    for s in SPLITS:
        for p in adapter.pairs(s):
            if not p.history_relevant:
                continue
            e = adapter.episode(p.episode_a)
            tick, horizon = int(e.provenance["position_tick"]), int(e.provenance["horizon"])
            lo, hi = _pos_range(p.evidence_position, horizon)
            if p.evidence_position not in POSITIONS or not lo <= tick <= hi:
                out.append(Finding("error", "position_tag_mismatch",
                                   f"{p.pair_id}: tick {tick}/{horizon} tagged {p.evidence_position}"))
    return out


def history_free_falsifier(adapter: SODHAdapter) -> dict[str, float]:
    """F1: a trained current-observation lookup must score pair-both-correct 0 on relevant cells."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for p in adapter.pairs("train"):
        for eid in (p.episode_a, p.episode_b):
            e = adapter.episode(eid)
            v = adapter.view(eid)
            table[digest([v.current.cameras, e.queries[0].params])][e.target_for(
                e.queries[0].query_id)] += 1
    res: dict[str, float] = {}
    for s in SPLITS:
        both: dict[str, list[bool]] = defaultdict(list)
        for p in adapter.pairs(s):
            if not p.history_relevant:
                continue
            oks = []
            for eid in (p.episode_a, p.episode_b):
                e = adapter.episode(eid)
                v = adapter.view(eid)
                key = digest([v.current.cameras, e.queries[0].params])
                opts = e.queries[0].options
                pred = table[key].most_common(1)[0][0] if key in table else opts[
                    int(key[:8], 16) % len(opts)]
                oks.append(pred == e.target_for(e.queries[0].query_id))
            both[p.family].append(all(oks))
        for fam, v in both.items():
            res[f"{s}.{fam}"] = sum(v) / len(v)
    return res


def posterior_sanity_findings(adapter: SODHAdapter, n: int = 200) -> list[Finding]:
    """Posterior for the true pair partner history must disagree (swap-sensitivity of the oracle)."""
    out = []
    for p in list(adapter.pairs("test_id"))[:n]:
        if not p.history_relevant:
            continue
        a, b = adapter.episode(p.episode_a), adapter.episode(p.episode_b)
        q = a.queries[0]
        ta, tb = front_ticks(a), front_ticks(b)
        swapped = [*tb[:-1], ta[-1]]  # B's history with A's (identical) current tick
        got = posterior(q.task, q.params, q.options,
                        dict(enumerate(swapped[:-1])), swapped[-1], len(swapped) * TICK_NS)
        best = max(got, key=lambda k: got[k])
        if best != p.target_b:
            out.append(Finding("error", "oracle_swap_insensitive", p.pair_id))
    return out


def g0_v2_report(adapter: SODHAdapter, pairs: Sequence[PairRecord2],
                 episodes: Sequence[EpisodeRecord], assignment: Mapping[str, str],
                 enforce_size: bool = True, extended: bool = True) -> dict[str, Any]:
    findings = [*check_splits(episodes, pairs, assignment), *check_pairs(adapter, pairs),
                *id_leak_findings(episodes), *metadata_lookup_audit(adapter),
                *token_label_probe(adapter, RELEVANT2), *label_prior_findings(adapter),
                *oracle_recompute_findings(episodes), *mechanism_findings(pairs, assignment),
                *template_findings(pairs), *(size_findings(adapter) if enforce_size else []),
                *posterior_sanity_findings(adapter), *oracle_env_findings(adapter),
                *position_findings(adapter),
                *(extended_probe_findings(adapter) if extended else [])]
    twin = alpha_twin_rates(adapter)
    for s, rates in twin.items():
        for fam, r in rates.items():
            if r > 0 and fam in RELEVANT2 and s in OOD_SPLITS + ("val", "test_id"):
                findings.append(Finding("error", "alpha_twin", f"{s}/{fam}: {r:.3f} > 0"))
    falsifier = history_free_falsifier(adapter)
    for key, v in falsifier.items():
        if v > 0:
            findings.append(Finding("error", "history_free_solves", f"{key}: pair-both {v:.3f}"))
    errors = [f for f in findings if f.level == "error"]
    return {
        "gate": "G0-v2", "audit_version": AUDIT_VERSION, "mode": "synthetic",
        "passed": not errors,
        "validity": "validated against the current attack set only; not proof of leak-freeness. "
                    "Measured power (docs/21): leaks that let a reader exceed ~0.70 accuracy are "
                    "detected; weaker ones (<=0.67) are NOT, and the probes cannot see leaks "
                    "outside single-token, token-pair and aggregate (count/parity) forms.",
        "alpha_twin_rate": twin, "structure_overlap_informational": structure_overlap(adapter),
        "oracle_full_history_env_success": oracle_value(adapter),
        "constant_answer_floor": constant_floor(adapter),
        "history_free_pair_both_correct": falsifier,
        "findings": [{"level": f.level, "code": f.code, "message": f.message} for f in findings],
    }


# ---- extended probes (aggregate and conjunctive leaks; power study docs/21) ------------------

def _z_crit(m: int) -> float:
    """Bonferroni-style critical z for m simultaneous probes at ~1% family-wise level."""
    import math
    return math.sqrt(2 * math.log(max(m, 2) / 0.01))


def _history_counts(ep: EpisodeRecord) -> dict[str, str]:
    """Per (key=value) token: how many history ticks carry it (rate/aggregate features), + parity."""
    counts: Counter[str] = Counter()
    for t in front_ticks(ep)[:-1]:
        for k, v in t.items():
            counts[f"{k}={v}"] += 1
    out: dict[str, str] = {}
    for tok, c in counts.items():
        out[f"count:{tok}"] = str(c)
        out[f"parity:{tok}"] = str(c % 2)
    return out


def _lookup_findings(train: Sequence[EpisodeRecord], test: Sequence[EpisodeRecord],
                     feats: Mapping[str, Mapping[str, str]], fam: str, code: str) -> list[Finding]:
    import math

    def lab(e: EpisodeRecord) -> str:
        return e.target_for(e.queries[0].query_id)
    chance = sum(1 / len(e.queries[0].options) for e in test) / len(test)
    prior = Counter(lab(e) for e in train).most_common(1)[0][0]
    base = max(sum(lab(e) == prior for e in test) / len(test), chance)
    types = sorted({k for e in train for k in feats[e.episode_id]})
    z_crit = max(3.3, _z_crit(len(types)))
    out = []
    for ty in types:
        tab: dict[str, Counter[str]] = defaultdict(Counter)
        for e in train:
            v = feats[e.episode_id].get(ty, "∅")
            tab[v][lab(e)] += 1
        hit = sum((tab[feats[e.episode_id].get(ty, "∅")].most_common(1)[0][0]
                   if feats[e.episode_id].get(ty, "∅") in tab else prior) == lab(e) for e in test)
        acc = hit / len(test)
        sd = math.sqrt(max(base * (1 - base), 1e-9) / len(test))
        if acc - base > 0.15 and (acc - base) / sd > z_crit:
            out.append(Finding("error", code, f"{fam}: {ty} acc {acc:.2f} vs {base:.2f}"))
    return out


def extended_probe_findings(adapter: SODHAdapter,
                            relevant: Sequence[str] = RELEVANT2) -> list[Finding]:
    """Aggregate (count/parity over history) and conjunctive (token pairs) shortcut probes."""
    out: list[Finding] = []
    from ucvla.audit import _probe_tokens
    for fam in sorted({p.family for p in adapter.pairs("train")} & set(relevant)):
        train = [adapter.episode(e) for p in adapter.pairs("train") if p.family == fam
                 for e in (p.episode_a, p.episode_b)]
        test = [adapter.episode(e) for s in SPLITS if s != "train" for p in adapter.pairs(s)
                if p.family == fam for e in (p.episode_a, p.episode_b)]
        if len(test) < 20 or not train:
            continue
        agg = {e.episode_id: _history_counts(e) for e in (*train, *test)}
        out += _lookup_findings(train, test, agg, fam, "agg_shortcut")
        tok = {e.episode_id: _probe_tokens(e) for e in (*train, *test)}
        cands = sorted(ty for ty in {k for e in train for k in tok[e.episode_id]}
                       if ty.startswith(("end-0:", "end-1:", "end-2:", "end-3:"))
                       and len({tok[e.episode_id].get(ty) for e in train}) <= 6)[:10]
        pair_feats = {eid: {f"{a}&{b}": f"{t.get(a)}|{t.get(b)}"
                            for i, a in enumerate(cands) for b in cands[i + 1:]}
                      for eid, t in tok.items()}
        out += _lookup_findings(train, test, pair_feats, fam, "pair_shortcut")
    return out
