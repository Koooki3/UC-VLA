# SPDX-License-Identifier: Apache-2.0
"""Deterministic Same-Observation / Different-History (SODH) generator.

Status: ``implemented`` as a *symbolic* simulator (frames are token maps, not pixels).
All output is ``mode: synthetic``; nothing here is a benchmark result or a VLA evaluation.

Every history-relevant pair shares one scene script: the two episodes have an identical
decision-time observation and differ only in an earlier, later-occluded event, which flips the
correct action. Families:

  object_correspondence  needle: a ball is revealed once under one of several cups, cups then swap
  process_count          aggregation: press a button exactly K times (n = K-1 vs K done)
  task_interleave        resume: which sub-task finished before a pause (no trace remains)
  irrelevant_history     negative control: label depends only on the current frame

``leak`` modes exist only so tests can prove the auditors detect planted shortcuts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ucvla.canonical import HashRng, digest, to_jsonable
from ucvla.episode import (
    SCHEMA_VERSION,
    CausalView,
    DecisionQuery,
    EpisodeRecord,
    Event,
    Frame,
    PrivilegedState,
    SourceType,
    Target,
)

GENERATOR = "ucvla.sodh"
GENERATOR_VERSION = "sodh/1"
TICK_NS = 100_000_000
CAMERAS = ("cam_front", "cam_wrist")
CLOCK_DOMAIN = "sim_ns"
REGIMES = ("in_dist", "ood", "heldout_mapping")
RELEVANT_FAMILIES = ("object_correspondence", "process_count", "task_interleave")
CONTROL_FAMILIES = ("irrelevant_history",)
LEAK_MODES = (None, "length", "id", "final_obs")
VOCAB = tuple(
    f"{adj}_{noun}"
    for adj, noun in zip(
        ("red", "blue", "green", "gold", "teal", "pink", "gray", "lime") * 5,
        ("cup", "box", "jar", "mug", "tin") * 8,
    )
)
GROUP_SIZE = 5  # identity mapping m owns VOCAB[5m:5m+5]  -> disjoint object vocabularies


@dataclass(frozen=True)
class FamilyConfig:
    n_objects_id: tuple[int, int]
    n_objects_ood: tuple[int, int]
    horizon_id: tuple[int, int]
    horizon_ood: tuple[int, int]
    pairs_in_dist: int
    pairs_ood: int
    pairs_heldout: int


@dataclass(frozen=True)
class SODHConfig:
    master_seed: int
    n_identity_mappings: int
    heldout_mappings: tuple[int, ...]
    split_fractions: tuple[float, float, float]
    families: Mapping[str, FamilyConfig]
    generator_version: str = GENERATOR_VERSION
    leak: str | None = None

    def __post_init__(self) -> None:
        if self.leak not in LEAK_MODES:
            raise ValueError(f"unknown leak mode {self.leak!r}")
        if self.n_identity_mappings * GROUP_SIZE > len(VOCAB):
            raise ValueError("vocabulary too small for requested identity mappings")
        if not all(0 <= m < self.n_identity_mappings for m in self.heldout_mappings):
            raise ValueError("heldout mapping out of range")
        if len(self.heldout_mappings) >= self.n_identity_mappings:
            raise ValueError("no training mappings left")
        if abs(sum(self.split_fractions) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        unknown = set(self.families) - set(RELEVANT_FAMILIES + CONTROL_FAMILIES)
        if unknown:
            raise ValueError(f"unknown families {sorted(unknown)}")
        for name, fc in self.families.items():
            if max(fc.n_objects_id[1], fc.n_objects_ood[1]) > GROUP_SIZE:
                raise ValueError(f"{name}: n_objects exceeds mapping vocabulary")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SODHConfig:
        fams = {
            name: FamilyConfig(
                n_objects_id=tuple(f["n_objects_id"]), n_objects_ood=tuple(f["n_objects_ood"]),
                horizon_id=tuple(f["horizon_id"]), horizon_ood=tuple(f["horizon_ood"]),
                pairs_in_dist=int(f["pairs_in_dist"]), pairs_ood=int(f["pairs_ood"]),
                pairs_heldout=int(f["pairs_heldout"]),
            )
            for name, f in d["families"].items()
        }
        return cls(
            master_seed=int(d["master_seed"]),
            n_identity_mappings=int(d["n_identity_mappings"]),
            heldout_mappings=tuple(d["heldout_mappings"]),
            split_fractions=tuple(d["split_fractions"]),
            families=fams,
            generator_version=d.get("generator_version", GENERATOR_VERSION),
            leak=d.get("leak"),
        )

    def sha256(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    family: str
    episode_a: str
    episode_b: str
    query_time_ns: int
    o_t_digest_a: str
    o_t_digest_b: str
    target_a: str
    target_b: str
    privileged_state_hash_train_only: str
    accepted_sensory_tolerance: int  # 0 = exact symbolic equality of the decision observation
    seed: int
    scene_script_id: str
    identity_mapping_id: str
    regime: str
    horizon: int
    n_objects: int
    history_relevant: bool


@dataclass(frozen=True)
class SODHSuite:
    config_sha256: str
    episodes: tuple[EpisodeRecord, ...]
    pairs: tuple[PairRecord, ...]
    rejected_duplicate_scripts: int


@dataclass
class _Mate:
    ticks: list[dict[str, str]]
    task: str
    instruction: str
    options: tuple[str, ...]
    params: dict[str, str]
    target: str
    annotations: list[tuple[int, str, dict[str, str]]]
    privileged: dict[str, str]


@dataclass
class _PairSpec:
    script: dict[str, Any]
    mates: tuple[_Mate, _Mate]


# ---- family builders ---------------------------------------------------------------------


def _build_cups(rng: HashRng, names: tuple[str, ...], n: int, horizon: int) -> _PairSpec:
    layout = list(names[:n])
    rng.shuffle(layout)
    swaps = []
    for _ in range(horizon):
        i = rng.below(n)
        j = (i + 1 + rng.below(n - 1)) % n
        swaps.append((i, j))
    layouts = [list(layout)]
    for i, j in swaps:
        cur = list(layouts[-1])
        cur[i], cur[j] = cur[j], cur[i]
        layouts.append(cur)
    reveal_tick = rng.randint(0, max(0, horizon // 2))
    x, y = _two_distinct(rng, names[:n])
    final = layouts[-1]
    options = tuple(f"pick_slot_{k}" for k in range(n))
    mates = []
    for cup in (x, y):
        ticks = []
        for t, lay in enumerate(layouts):
            vis = {f"slot{k}": lay[k] for k in range(n)}
            vis["ball"] = f"under:{cup}" if t == reveal_tick else "hidden"
            vis["phase"] = "shuffle"
            ticks.append(vis)
        last = {f"slot{k}": final[k] for k in range(n)}
        last["ball"] = "hidden"
        last["phase"] = "choose"
        ticks.append(last)
        target = f"pick_slot_{final.index(cup)}"
        mates.append(_Mate(
            ticks, "cups_ball", "pick the cup that hid the ball", options,
            {"n_slots": str(n)}, target,
            [(reveal_tick, "ball_under", {"cup": cup})],
            {"ball_cup": cup, "final_slot": str(final.index(cup))},
        ))
    script = {"layout": layouts[0], "swaps": swaps, "reveal_tick": reveal_tick, "n": n}
    return _PairSpec(script, (mates[0], mates[1]))


def _two_distinct(rng: HashRng, pool: tuple[str, ...]) -> tuple[str, str]:
    i = rng.below(len(pool))
    j = (i + 1 + rng.below(len(pool) - 1)) % len(pool)
    return pool[i], pool[j]


def _build_count(rng: HashRng, names: tuple[str, ...], k: int) -> _PairSpec:
    button = names[0]
    total = 2 * k + rng.randint(0, k)
    poses = [f"pose_{rng.below(6)}" for _ in range(total)]
    idx = list(range(total))
    rng.shuffle(idx)
    base_press = set(idx[: k - 1])
    extra = idx[k - 1]  # the one tick that is a press only in the "done" episode
    mates = []
    for presses, n_done in ((base_press, k - 1), (base_press | {extra}, k)):
        ticks = [{"button": "pressed" if t in presses else "up", "arm": poses[t],
                  "phase": "work"} for t in range(total)]
        ticks.append({"button": "up", "arm": "pose_final", "phase": "decide"})
        mates.append(_Mate(
            ticks, "button_count", f"press the {button} exactly {k} times, then stop",
            ("press", "stop"), {"target_count": str(k), "button": button},
            "press" if n_done < k else "stop",
            [(t, "button_press", {"n_so_far": str(sum(1 for u in sorted(presses) if u <= t))})
             for t in sorted(presses)],
            {"presses_done": str(n_done)},
        ))
    script = {"k": k, "total": total, "poses": poses, "base_press": sorted(base_press),
              "button": button}
    return _PairSpec(script, (mates[0], mates[1]))


def _build_interleave(rng: HashRng, names: tuple[str, ...], p: int, pause: int) -> _PairSpec:
    order = list(names[:p])
    rng.shuffle(order)
    j = rng.below(p - 1)  # mates have completed j and j+1 phases
    options = tuple(sorted(f"place_{n}" for n in order))
    mates = []
    for done in (j, j + 1):
        ticks: list[dict[str, str]] = [{"phase": "plan", "placing": "none"}]
        for s in range(p - 1):
            ticks.append({"phase": "work", "placing": order[s] if s < done else "none"})
        ticks += [{"phase": "paused", "placing": "none"} for _ in range(pause)]
        ticks.append({"phase": "resume", "placing": "none"})
        mates.append(_Mate(
            ticks, "resume_sequence",
            "after the interruption continue the placement sequence", options,
            {"phase_order": ",".join(order)}, f"place_{order[done]}",
            [(1 + s, "placed", {"item": order[s]}) for s in range(done)],
            {"phases_done": str(done)},
        ))
    script = {"order": order, "pause": pause, "p": p}
    return _PairSpec(script, (mates[0], mates[1]))


def _build_irrelevant(rng: HashRng, names: tuple[str, ...], horizon: int,
                      index: int) -> _PairSpec:
    flag = ("left", "right")[index % 2]  # exact label balance within each regime
    mates = []
    histories = []
    while len(histories) < 2 or histories[0] == histories[1]:
        histories = [
            [{"noise": f"n{rng.below(8)}", "ball": f"under:{rng.choice(names)}", "phase": "idle"}
             for _ in range(horizon)] for _ in range(2)
        ]
    for hist in histories:
        ticks = [dict(h) for h in hist]
        ticks.append({"flag": flag, "phase": "choose", "noise": "none", "ball": "hidden"})
        mates.append(_Mate(
            ticks, "pick_flagged", "pick the side whose flag is raised",
            ("pick_left", "pick_right"), {}, f"pick_{flag}", [], {"flag": flag},
        ))
    return _PairSpec({"flag": flag, "histories": histories}, (mates[0], mates[1]))


# ---- assembly ---------------------------------------------------------------------------


def _assemble(spec_mate: _Mate, episode_id: str, provenance: dict[str, Any],
              leak: str | None) -> EpisodeRecord:
    ticks = [dict(t) for t in spec_mate.ticks]
    annotations = list(spec_mate.annotations)
    smallest = min(spec_mate.options)
    if leak == "length" and spec_mate.target == smallest:
        ticks.insert(0, dict(ticks[0]))
        annotations = [(i + 1, k, p) for i, k, p in annotations]
    if leak == "final_obs":
        ticks[-1]["jitter"] = digest(spec_mate.target)[:6]
    if leak == "id":
        episode_id = f"{episode_id}-{spec_mate.target}"
    frames = []
    for i, vis in enumerate(ticks):
        t = (i + 1) * TICK_NS
        frames.append(Frame.make(f"{episode_id}-f{i}-front", CAMERAS[0], t, vis))
        frames.append(Frame.make(f"{episode_id}-f{i}-wrist", CAMERAS[1], t, {"gripper": "open"}))
    q_time = len(ticks) * TICK_NS
    query = DecisionQuery(f"{episode_id}-q0", q_time, spec_mate.task, spec_mate.instruction,
                          spec_mate.options, spec_mate.params)
    events = [Event.make(f"{episode_id}-a{n}", (i + 1) * TICK_NS, kind, payload,
                         SourceType.ANNOTATION)
              for n, (i, kind, payload) in enumerate(annotations)]
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION, episode_id=episode_id, task=spec_mate.task,
        clock_domain=CLOCK_DOMAIN, frames=tuple(frames), proprio=(), commanded=(),
        executed=(), events=tuple(events), queries=(query,),
        targets=(Target(query.query_id, spec_mate.target),),
        privileged=PrivilegedState(q_time, spec_mate.privileged), provenance=provenance,
    )


def _ranges(fc: FamilyConfig, regime: str) -> tuple[tuple[int, int], tuple[int, int]]:
    return (fc.n_objects_ood, fc.horizon_ood) if regime == "ood" else (fc.n_objects_id,
                                                                       fc.horizon_id)


def _build_family(family: str, rng: HashRng, names: tuple[str, ...], n: int,
                  horizon: int, index: int) -> _PairSpec:
    if family == "object_correspondence":
        return _build_cups(rng, names, n, horizon)
    if family == "process_count":
        return _build_count(rng, names, horizon)
    if family == "task_interleave":
        return _build_interleave(rng, names, n, horizon)
    return _build_irrelevant(rng, names, horizon, index)


def generate_suite(config: SODHConfig) -> SODHSuite:
    """Pure function of ``config``: same config => byte-identical suite."""
    cfg_hash = config.sha256()
    train_maps = [m for m in range(config.n_identity_mappings) if m not in config.heldout_mappings]
    episodes: list[EpisodeRecord] = []
    pairs: list[PairRecord] = []
    seen_scripts: set[str] = set()
    rejected = 0
    for family, fc in config.families.items():
        counts = {"in_dist": fc.pairs_in_dist, "ood": fc.pairs_ood,
                  "heldout_mapping": fc.pairs_heldout}
        for regime in REGIMES:
            (n_lo, n_hi), (h_lo, h_hi) = _ranges(fc, regime)
            for index in range(counts[regime]):
                for attempt in range(200):
                    key = (config.master_seed, family, regime, index, attempt)
                    rng = HashRng(*key)
                    pool = config.heldout_mappings if regime == "heldout_mapping" else train_maps
                    mapping = rng.choice(list(pool))
                    names = VOCAB[mapping * GROUP_SIZE:(mapping + 1) * GROUP_SIZE]
                    n = rng.randint(n_lo, n_hi)
                    horizon = rng.randint(h_lo, h_hi)
                    spec = _build_family(family, rng, names, n, horizon, index)
                    script_id = digest({"family": family, "mapping": mapping,
                                        "script": spec.script})
                    if script_id in seen_scripts:
                        rejected += 1
                        continue
                    seen_scripts.add(script_id)
                    break
                else:
                    raise RuntimeError(f"scene space exhausted for {family}/{regime}")
                pair_id = f"pair-{digest(key)[:16]}"
                order = (0, 1) if HashRng(*key, "role").coin() else (1, 0)
                eps = []
                for role, which in zip("ab", order):
                    prov = {
                        "generator": GENERATOR, "generator_version": config.generator_version,
                        "config_sha256": cfg_hash, "seed": config.master_seed,
                        "family": family, "pair_id": pair_id, "pair_role": role,
                        "scene_script_id": script_id, "identity_mapping_id": f"idm-{mapping}",
                        "regime": regime, "horizon": horizon, "n_objects": n,
                        "history_relevant": family in RELEVANT_FAMILIES,
                    }
                    eid = "ep-" + digest(key + (role,))[:16]
                    eps.append(_assemble(spec.mates[which], eid, prov, config.leak))
                a, b = eps
                oa = a.causal_view(a.queries[0].time_ns).digest(include_history=False)
                ob = b.causal_view(b.queries[0].time_ns).digest(include_history=False)
                pairs.append(PairRecord(
                    pair_id=pair_id, family=family, episode_a=a.episode_id, episode_b=b.episode_id,
                    query_time_ns=a.queries[0].time_ns, o_t_digest_a=oa, o_t_digest_b=ob,
                    target_a=a.target_for(a.queries[0].query_id),
                    target_b=b.target_for(b.queries[0].query_id),
                    privileged_state_hash_train_only=digest([a.privileged, b.privileged]),
                    accepted_sensory_tolerance=0, seed=config.master_seed,
                    scene_script_id=script_id, identity_mapping_id=f"idm-{mapping}",
                    regime=regime, horizon=horizon, n_objects=n,
                    history_relevant=family in RELEVANT_FAMILIES,
                ))
                episodes += [a, b]
    return SODHSuite(cfg_hash, tuple(episodes), tuple(pairs), rejected)


def decision_view(ep: EpisodeRecord) -> CausalView:
    return ep.causal_view(ep.queries[0].time_ns)


def suite_to_jsonable(suite: SODHSuite) -> Mapping[str, Any]:
    return to_jsonable({"config_sha256": suite.config_sha256, "pairs": suite.pairs})
