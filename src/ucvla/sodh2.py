# SPDX-License-Identifier: Apache-2.0
"""SODH v2 generator (spec: docs/19). Symbolic, ``mode: synthetic``; new version, v1 untouched.

Differences from v1 that matter scientifically: anonymous cups that must be tracked through
visible and occluded swaps with a noisy reveal; validity-aware event counting (holds, repeated
edges, jams, dropout); per-item state across interleaved histories with retries and (OOD)
removals; labels are *posterior-optimal actions from an independent oracle*, success is judged
from the true hidden state; evidence position is a generated axis; OOD changes mechanisms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ucvla.canonical import HashRng, digest
from ucvla.episode import (
    SCHEMA_VERSION,
    DecisionQuery,
    EpisodeRecord,
    Event,
    Frame,
    PrivilegedState,
    SourceType,
    Target,
)
from ucvla.oracle2 import optimal, posterior
from ucvla.sodh import CAMERAS, CLOCK_DOMAIN, TICK_NS, PairRecord

GENERATOR = "ucvla.sodh2"
GENERATOR_VERSION = "sodh/2"
FAMILIES = ("track", "events", "resume", "control")
RELEVANT2 = ("track", "events", "resume")
REGIMES2 = ("in_dist", "ood_length", "ood_structure", "heldout_mapping")
POSITIONS = ("early", "mid", "late")
SPLIT_OF_REGIME = {"ood_length": "test_ood_length", "ood_structure": "test_ood_structure",
                   "heldout_mapping": "test_counterfactual"}
TRAIN_MECHANISMS = {
    "track": ("iid_swaps",), "events": ("taps",), "resume": ("retries",), "control": ("noise",)}
OOD_MECHANISMS = {
    "track": ("swap_bursts", "flicker_distractor"), "events": ("long_holds", "dense_taps"),
    "resume": ("removals", "distractor_attempts"), "control": ("noise",)}
ADJ = tuple(f"a{i:02d}" for i in range(40))
NOUN = tuple(f"n{i:02d}" for i in range(40))
GROUP = 5  # identity mapping m owns names ADJ[5m+i]_NOUN[5m+i]: disjoint atoms, no shared parts
TASK = {"track": "track_ball", "events": "button_events", "resume": "resume_items",
        "control": "pick_flagged"}


def vocab(mapping: int) -> tuple[str, ...]:
    return tuple(f"{ADJ[mapping * GROUP + i]}_{NOUN[mapping * GROUP + i]}" for i in range(GROUP))


@dataclass(frozen=True)
class FamilyConfig2:
    n_id: tuple[int, int]
    horizon_id: tuple[int, int]
    horizon_ood_length: tuple[int, int]
    pairs_in_dist: int
    pairs_ood_length: int
    pairs_ood_structure: int
    pairs_heldout: int


@dataclass(frozen=True)
class SODH2Config:
    master_seed: int
    n_identity_mappings: int
    heldout_mappings: tuple[int, ...]
    split_fractions: tuple[float, float, float]
    noise_levels: tuple[float, ...]
    margin_gate: float
    families: Mapping[str, FamilyConfig2]
    generator_version: str = GENERATOR_VERSION

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SODH2Config:
        fams = {
            k: FamilyConfig2(tuple(v["n_id"]), tuple(v["horizon_id"]),
                             tuple(v["horizon_ood_length"]), int(v["pairs_in_dist"]),
                             int(v["pairs_ood_length"]), int(v["pairs_ood_structure"]),
                             int(v["pairs_heldout"]))
            for k, v in d["families"].items()}
        unknown = set(fams) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}")
        heldout = tuple(d["heldout_mappings"])
        if abs(sum(d["split_fractions"]) - 1.0) > 1e-9 or not heldout:
            raise ValueError("bad split fractions or no heldout mapping")
        if int(d["n_identity_mappings"]) * GROUP > len(ADJ):
            raise ValueError("vocabulary too small")
        return cls(int(d["master_seed"]), int(d["n_identity_mappings"]), heldout,
                   tuple(d["split_fractions"]), tuple(d["noise_levels"]),
                   float(d["margin_gate"]), fams, d.get("generator_version", GENERATOR_VERSION))

    def sha256(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class PairRecord2(PairRecord):
    template_id: str = ""
    mechanisms: str = ""
    evidence_position: str = ""
    noise: float = 0.0
    margin_a: float = 0.0
    margin_b: float = 0.0


@dataclass(frozen=True)
class SODH2Suite:
    config_sha256: str
    episodes: tuple[EpisodeRecord, ...]
    pairs: tuple[PairRecord2, ...]
    rejected: Mapping[str, int]


@dataclass
class _Mate:
    ticks: list[dict[str, str]]
    success: str
    annotations: list[tuple[int, str, dict[str, str]]]
    hidden: dict[str, str]


@dataclass
class _Spec:
    options: tuple[str, ...]
    params: dict[str, str]
    mates: tuple[_Mate, _Mate]
    script: Any
    n: int
    position_tick: int
    horizon: int
    mechanisms: tuple[str, ...]


def _pos_range(pos: str, horizon: int) -> tuple[int, int]:
    k = POSITIONS.index(pos)
    lo = k * horizon // 3
    return lo, max(lo, (k + 1) * horizon // 3 - 1)


def _pair(rng: HashRng, n: int) -> tuple[int, int]:
    i = rng.below(n)
    return i, (i + 1 + rng.below(n - 1)) % n


# ---- families --------------------------------------------------------------------------------


def _build_track(rng: HashRng, fc: FamilyConfig2, regime: str, pos: str, eps: float) -> _Spec:
    n = rng.randint(*fc.n_id)
    lo, hi = fc.horizon_ood_length if regime == "ood_length" else fc.horizon_id
    horizon = rng.randint(lo, hi)
    burst = regime == "ood_structure"
    swaps: list[tuple[int, int] | None] = []
    while len(swaps) < horizon:
        if burst and rng.below(100) < 25:
            swaps += [_pair(rng, n) for _ in range(min(3, horizon - len(swaps)))]
        else:
            swaps.append(_pair(rng, n) if rng.below(100) < 60 else None)
    occluded = [rng.below(100) < 15 for _ in range(horizon)]
    aux = [("flick" if rng.below(100) < 30 else "off") if burst else None for _ in range(horizon)]
    plo, phi = _pos_range(pos, horizon)
    reveal = min(rng.randint(plo, phi), horizon - 1)
    x, y = _pair(rng, n)
    rate = round(sum(s is not None for s in swaps) / horizon, 2)
    mates = []
    readings = []
    for start in (x, y):
        reading = start if rng.below(1000) >= int(eps * 1000) else (
            (start + 1 + rng.below(n - 1)) % n)
        readings.append(reading)
        cur, decisive = start, []
        for t in range(reveal, horizon):
            if t == reveal:
                decisive.append(t)
            sw = swaps[t]
            if sw and cur in sw:
                cur = sw[1] if cur == sw[0] else sw[0]
                decisive.append(t)
        ticks = []
        for t in range(horizon):
            tok = "?" if occluded[t] else ("none" if swaps[t] is None
                                          else f"{swaps[t][0]}-{swaps[t][1]}")  # type: ignore[index]
            d = {"phase": "shuffle", "swap": tok,
                 "reveal": f"slot:{reading}" if t == reveal else "none"}
            if aux[t] is not None:
                d["aux"] = aux[t]  # type: ignore[assignment]
            ticks.append(d)
        last = {"phase": "choose", "swap": "none", "reveal": "none"}
        if burst:
            last["aux"] = "off"
        ticks.append(last)
        mates.append(_Mate(ticks, f"pick_slot_{cur}",
                           [(t, "decisive", {}) for t in sorted(set(decisive))],
                           {"start_slot": str(start), "final_slot": str(cur)}))
    script = {"n": n, "swaps": swaps, "occ": occluded, "aux": aux, "reveal": reveal,
              "starts": sorted((x, y)), "eps": eps}
    mech = OOD_MECHANISMS["track"] if burst else TRAIN_MECHANISMS["track"]
    return _Spec(tuple(f"pick_slot_{k}" for k in range(n)),
                 {"n_slots": str(n), "sensor_eps": f"{eps:.2f}", "swap_rate": f"{rate:.2f}"},
                 (mates[0], mates[1]), script, n, reveal, horizon, mech)


def _build_events(rng: HashRng, fc: FamilyConfig2, regime: str, pos: str) -> _Spec:
    lo, hi = fc.horizon_ood_length if regime == "ood_length" else fc.horizon_id
    horizon = rng.randint(lo, hi)
    struct = regime == "ood_structure"
    p_up, p_hold, jam = (50, 70, 25) if struct else (30, 30, 25)
    cs, edges, accepted, cp = [], [], [], 0
    for _ in range(horizon):
        c = int(rng.below(100) < (p_up if cp == 0 else p_hold))
        edge = cp == 0 and c == 1
        cs.append(c)
        edges.append(edge)
        accepted.append(edge and rng.below(100) >= jam)
        cp = c
    plo, phi = _pos_range(pos, horizon)
    cand = [t for t in range(plo, phi + 1) if edges[t] and not accepted[t] or edges[t]]
    if not cand:
        raise _Reject("no edge in evidence-position window")
    diff = rng.choice(cand)
    others = [t for t in range(horizon) if accepted[t] and t != diff]
    if not others:
        raise _Reject("need at least one other accepted trigger")
    drop = [rng.below(100) < 8 and t != diff for t in range(horizon)]
    k_target = len(others) + 1
    mates = []
    for lamp_on in (False, True):
        ticks = []
        for t in range(horizon):
            if drop[t]:
                ticks.append({"phase": "work", "contact": "?", "lamp": "?"})
                continue
            lamp = (t == diff and lamp_on) or (t in others)
            ticks.append({"phase": "work", "contact": "down" if cs[t] else "up",
                          "lamp": "on" if lamp else "off"})
        ticks.append({"phase": "decide", "contact": "up", "lamp": "off"})
        count = len(others) + int(lamp_on)
        crit = sorted(set(others) | {diff})
        mates.append(_Mate(ticks, "press" if count < k_target else "stop",
                           [(t, "decisive", {}) for t in crit if not drop[t]],
                           {"count": str(count)}))
    script = {"cs": cs, "acc_others": others, "diff": diff, "drop": drop, "k": k_target}
    mech = OOD_MECHANISMS["events"] if struct else TRAIN_MECHANISMS["events"]
    return _Spec(("press", "stop"),
                 {"target_count": str(k_target), "jam": f"{jam / 100:.2f}",
                  "p_attempt": f"{p_up / 100:.2f}", "p_hold": f"{p_hold / 100:.2f}"},
                 (mates[0], mates[1]), script, k_target, diff, horizon, mech)


class _Reject(Exception):
    pass


def _build_resume(rng: HashRng, fc: FamilyConfig2, regime: str, pos: str,
                  names: tuple[str, ...]) -> _Spec:
    m = rng.randint(*fc.n_id)
    items = list(names[:m])
    lo, hi = fc.horizon_ood_length if regime == "ood_length" else fc.horizon_id
    horizon = rng.randint(lo, hi)
    struct = regime == "ood_structure"
    x, y = _pair(rng, m)
    others = [r for r in range(m) if r not in (x, y)]
    removal_roles = [r for r in others if struct and rng.below(100) < 50][:2]
    # Per-role causal sequences (fail* ... ok ... [removed ok]); y's final ok is the decisive tick.
    seqs: dict[int, list[str]] = {x: ["fail"] * rng.below(3)}
    for r in others:
        seqs[r] = ["fail"] * rng.below(3) + (["ok", "removed", "ok"] if r in removal_roles
                                             else ["ok"])
    plo, phi = _pos_range(pos, horizon)
    ty = rng.randint(plo, phi)
    y_fails = min(rng.below(3), ty)
    fixed: dict[int, tuple[str, int]] = {ty: ("ok", y)}
    if y_fails:
        for t in _ticks(rng, ty, y_fails):
            fixed[t] = ("fail", y)
    extra = [rng.below(m) for _ in range(rng.randint(1, 3))] if struct else []
    merged = [r for r, sq in seqs.items() for _ in sq] + extra
    rng.shuffle(merged)
    free = [t for t in range(horizon) if t not in fixed]
    if len(merged) > len(free) - 1:
        raise _Reject("too many events for horizon")
    rng.shuffle(free)
    ticks_for = sorted(free[:len(merged)])
    cursor = dict.fromkeys(range(m), 0)
    chosen: dict[int, tuple[str, int]] = dict(fixed)
    for t, r in zip(ticks_for, merged):
        if r in extra and cursor[r] >= len(seqs.get(r, [])):
            chosen[t] = ("fail", r)  # distractor attempt, state-neutral
            continue
        chosen[t] = (seqs[r][cursor[r]], r)
        cursor[r] += 1
    occ = {t: rng.below(100) < 15 and t != ty for t in chosen}
    drop = {t: rng.below(100) < 8 for t in range(horizon) if t not in chosen}
    n_att = sum(1 for k, _ in chosen.values() if k != "removed")
    p_att = round(n_att / horizon, 2)
    mates = []
    for swap in (False, True):
        def nm(r: int, swap: bool = swap) -> str:
            return items[y if r == x else x if r == y else r] if swap else items[r]
        ticks: list[dict[str, str]] = []
        ann = []
        for t in range(horizon):
            if t in chosen:
                k, r = chosen[t]
                if k == "removed":
                    ticks.append({"phase": "work", "attempt": "none", "result": "none",
                                  "removed": nm(r)})
                else:
                    ticks.append({"phase": "work", "attempt": nm(r),
                                  "result": "?" if occ[t] else k, "removed": "none"})
                    if k == "ok":
                        ann.append((t, "decisive", {}))
            elif drop.get(t):
                ticks.append({"phase": "work", "attempt": "?", "result": "?", "removed": "?"})
            else:
                ticks.append({"phase": "work", "attempt": "none", "result": "none",
                              "removed": "none"})
        ticks.append({"phase": "resume", "attempt": "none", "result": "none", "removed": "none"})
        unplaced = nm(x)  # role x is the unplaced one in BOTH mates (mate B just renames it)
        mates.append(_Mate(ticks, f"place_{unplaced}", ann, {"unplaced": unplaced}))
    script = {"m": m, "x": x, "y": y, "chosen": sorted(chosen.items()),
              "occ": sorted(occ.items()), "drop": sorted(drop.items()), "horizon": horizon}
    mech = OOD_MECHANISMS["resume"] if struct else TRAIN_MECHANISMS["resume"]
    srt = sorted(items)  # params/options in alphabetical item order: no role-order leak
    return _Spec(tuple(f"place_{i}" for i in srt),
                 {"items": ",".join(srt), "fail": "0.25", "p_attempt": f"{p_att:.2f}",
                  "removal": "0.05" if struct else "0.00"},
                 (mates[0], mates[1]), script, m, ty, horizon, mech)


def _ticks(rng: HashRng, upto: int, k: int) -> list[int]:
    pool = list(range(upto))
    rng.shuffle(pool)
    return pool[:k]


def _build_control(rng: HashRng, fc: FamilyConfig2, regime: str, index: int) -> _Spec:
    lo, hi = fc.horizon_ood_length if regime == "ood_length" else fc.horizon_id
    horizon = rng.randint(lo, hi)
    flag = ("left", "right")[index % 2]
    hist: list[list[dict[str, str]]] = []
    while len(hist) < 2 or hist[0] == hist[1]:
        hist = [[{"phase": "idle", "noise": f"n{rng.below(8)}"} for _ in range(horizon)]
                for _ in range(2)]
    mates = []
    for h in hist:
        ticks = [dict(t) for t in h] + [{"phase": "choose", "noise": "none", "flag": flag}]
        for t in ticks[:-1]:
            t["flag"] = "none"
        mates.append(_Mate(ticks, f"pick_{flag}", [], {"flag": flag}))
    return _Spec(("pick_left", "pick_right"), {}, (mates[0], mates[1]),
                 {"flag": flag, "h": hist}, 2, horizon - 1, horizon,
                 TRAIN_MECHANISMS["control"])


# ---- assembly --------------------------------------------------------------------------------


def oracle_for_ticks(spec_task: str, params: Mapping[str, str], options: Sequence[str],
                     ticks: Sequence[Mapping[str, str]]) -> tuple[str, float]:
    kept = dict(enumerate(ticks[:-1]))
    post = posterior(spec_task, params, options, kept, ticks[-1], (len(ticks)) * TICK_NS)
    return optimal(post, options)


def _assemble(family: str, spec: _Spec, mate: _Mate, target: str, episode_id: str,
              prov: dict[str, Any]) -> EpisodeRecord:
    frames = []
    for i, vis in enumerate(mate.ticks):
        t = (i + 1) * TICK_NS
        frames.append(Frame.make(f"{episode_id}-f{i}-front", CAMERAS[0], t, vis))
        frames.append(Frame.make(f"{episode_id}-f{i}-wrist", CAMERAS[1], t, {"gripper": "open"}))
    q_time = len(mate.ticks) * TICK_NS
    query = DecisionQuery(f"{episode_id}-q0", q_time, TASK[family],
                          "decide the next action from the history you observed", spec.options,
                          spec.params)
    events = [Event.make(f"{episode_id}-a{n}", (i + 1) * TICK_NS, kind, payload,
                         SourceType.ANNOTATION) for n, (i, kind, payload) in enumerate(mate.annotations)]
    priv = {**mate.hidden, "success_actions": mate.success, "oracle_target": target}
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION, episode_id=episode_id, task=TASK[family],
        clock_domain=CLOCK_DOMAIN, frames=tuple(frames), proprio=(), commanded=(), executed=(),
        events=tuple(events), queries=(query,), targets=(Target(query.query_id, target),),
        privileged=PrivilegedState(q_time, priv), provenance=prov)


def generate_suite2(config: SODH2Config) -> SODH2Suite:
    cfg_hash = config.sha256()
    train_maps = [m for m in range(config.n_identity_mappings) if m not in config.heldout_mappings]
    episodes: list[EpisodeRecord] = []
    pairs: list[PairRecord2] = []
    seen: set[str] = set()
    seen_eps: set[str] = set()  # observable episode content: no episode may appear twice
    rejected: dict[str, int] = defaultdict(int)
    for family, fc in config.families.items():
        counts = {"in_dist": fc.pairs_in_dist, "ood_length": fc.pairs_ood_length,
                  "ood_structure": fc.pairs_ood_structure, "heldout_mapping": fc.pairs_heldout}
        for regime in REGIMES2:
            for index in range(counts[regime]):
                for attempt in range(400):
                    key = (config.master_seed, family, regime, index, attempt)
                    rng = HashRng(*key)
                    pos = POSITIONS[index % 3]
                    eps = config.noise_levels[(index // 3) % len(config.noise_levels)] \
                        if family == "track" else 0.0
                    pool = config.heldout_mappings if regime == "heldout_mapping" else train_maps
                    mapping = rng.choice(list(pool))
                    gen_regime = "in_dist" if regime == "heldout_mapping" else regime
                    try:
                        if family == "track":
                            spec = _build_track(rng, fc, gen_regime, pos, eps)
                        elif family == "events":
                            spec = _build_events(rng, fc, gen_regime, pos)
                        elif family == "resume":
                            spec = _build_resume(rng, fc, gen_regime, pos, vocab(mapping))
                        else:
                            spec = _build_control(rng, fc, gen_regime, index)
                    except _Reject as e:
                        rejected[str(e)] += 1
                        continue
                    task = TASK[family]
                    ta, ma = oracle_for_ticks(task, spec.params, spec.options, spec.mates[0].ticks)
                    tb, mb = oracle_for_ticks(task, spec.params, spec.options, spec.mates[1].ticks)
                    relevant = family in RELEVANT2
                    ok = (ma >= config.margin_gate and mb >= config.margin_gate)
                    ok &= (ta != tb) if relevant else (ta == tb)
                    # observation-level canonical key: hidden details that no tick reveals must not
                    # make two identical observation streams look distinct
                    template = digest({"f": family, "mech": spec.mechanisms,
                                       "params": spec.params, "options": spec.options,
                                       "obs": [m.ticks for m in spec.mates],
                                       "y": [m.success for m in spec.mates]})
                    mate_keys = [digest([m.ticks, spec.params, spec.options, m.success])
                                 for m in spec.mates]
                    if not ok or template in seen or any(k in seen_eps for k in mate_keys):
                        rejected["gate" if not ok else "duplicate"] += 1
                        continue
                    seen.add(template)
                    seen_eps.update(mate_keys)
                    break
                else:
                    raise RuntimeError(f"cannot build {family}/{regime}/{index}")
                pair_id = f"pair-{digest(key)[:16]}"
                order = (0, 1) if HashRng(*key, "role").coin() else (1, 0)
                eps_rec = []
                for role, which in zip("ab", order):
                    prov = {"generator": GENERATOR, "generator_version": config.generator_version,
                            "config_sha256": cfg_hash, "seed": config.master_seed,
                            "family": family, "pair_id": pair_id, "pair_role": role,
                            "template_id": template, "regime": regime,
                            "mechanisms": ",".join(spec.mechanisms), "evidence_position": pos,
                            "noise": f"{eps:.2f}", "identity_mapping_id": f"idm-{mapping}",
                            "horizon": spec.horizon, "history_relevant": relevant,
                            "position_tick": spec.position_tick}
                    eid = "ep-" + digest(key + (role,))[:16]
                    tgt = (ta, tb)[which]
                    eps_rec.append(_assemble(family, spec, spec.mates[which], tgt, eid, prov))
                a, b = eps_rec
                va, vb = (e.causal_view(e.queries[0].time_ns) for e in (a, b))
                pairs.append(PairRecord2(
                    pair_id=pair_id, family=family, episode_a=a.episode_id, episode_b=b.episode_id,
                    query_time_ns=a.queries[0].time_ns, o_t_digest_a=va.digest(False),
                    o_t_digest_b=vb.digest(False), target_a=a.target_for(a.queries[0].query_id),
                    target_b=b.target_for(b.queries[0].query_id),
                    privileged_state_hash_train_only=digest([a.privileged, b.privileged]),
                    accepted_sensory_tolerance=0, seed=config.master_seed, scene_script_id=template,
                    identity_mapping_id=f"idm-{mapping}", regime=regime, horizon=spec.horizon,
                    n_objects=spec.n, history_relevant=relevant, template_id=template,
                    mechanisms=",".join(spec.mechanisms), evidence_position=pos, noise=eps,
                    margin_a=round((ma, mb)[order[0]], 4), margin_b=round((ma, mb)[order[1]], 4)))
                episodes += [a, b]
    return SODH2Suite(cfg_hash, tuple(episodes), tuple(pairs), dict(rejected))


def assign_splits2(pairs: Sequence[PairRecord2], config: SODH2Config) -> dict[str, str]:
    """In-dist pairs are cut per (family, evidence position) after a keyed shuffle; OOD regimes
    map to their own splits (no pair is ever split)."""
    out: dict[str, str] = {}
    groups: dict[tuple[str, str], list[PairRecord2]] = defaultdict(list)
    for p in pairs:
        if p.regime in SPLIT_OF_REGIME:
            out[p.pair_id] = SPLIT_OF_REGIME[p.regime]
        else:
            groups[(p.family, p.evidence_position)].append(p)
    f_train, f_val, _ = config.split_fractions
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda p: digest([config.master_seed, "split2", p.pair_id]))
        n = len(ordered)
        n_train, n_val = round(f_train * n), round(f_val * n)
        for i, p in enumerate(ordered):
            out[p.pair_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test_id"
    return out
