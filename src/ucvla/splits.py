"""Pair-level split assignment, leakage checker and write-once split manifest.

Rules (docs/06): splits are assigned per *pair* (never per episode), the shared scene script
and the object-identity mapping are group keys, ``test_counterfactual`` uses identity mappings
never seen elsewhere, and ``test_ood`` must be strictly longer-horizon than every in-distribution
split. A registered manifest is immutable: a change needs a new id.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ucvla.canonical import canonical_json, digest
from ucvla.episode import EpisodeRecord
from ucvla.sodh import PairRecord, SODHConfig

V1_SPLITS = ("train", "val", "test_id", "test_ood", "test_counterfactual")
SPLITS = (*V1_SPLITS, "test_ood_length", "test_ood_structure")  # v2 adds the last two
IN_DIST_SPLITS = ("train", "val", "test_id")
BALANCE_MARGIN = 0.20


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warn"
    code: str
    message: str


def assign_splits(pairs: Sequence[PairRecord], config: SODHConfig) -> dict[str, str]:
    """Deterministic pair_id -> split. In-dist pairs are cut per family after a keyed shuffle."""
    out: dict[str, str] = {}
    by_family: dict[str, list[PairRecord]] = defaultdict(list)
    for p in pairs:
        if p.regime == "ood":
            out[p.pair_id] = "test_ood"
        elif p.regime == "heldout_mapping":
            out[p.pair_id] = "test_counterfactual"
        else:
            by_family[p.family].append(p)
    f_train, f_val, _ = config.split_fractions
    for family in sorted(by_family):
        ordered = sorted(by_family[family],
                         key=lambda p: digest([config.master_seed, "split", p.pair_id]))
        n = len(ordered)
        n_train = round(f_train * n)
        n_val = round(f_val * n)
        for i, p in enumerate(ordered):
            out[p.pair_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test_id"
    return out


def check_splits(
    episodes: Sequence[EpisodeRecord], pairs: Sequence[PairRecord], assignment: Mapping[str, str]
) -> list[Finding]:
    f: list[Finding] = []
    by_id = {e.episode_id: e for e in episodes}
    if len(by_id) != len(episodes):
        f.append(Finding("error", "dup_episode_id", "episode ids not unique"))
    ep_split: dict[str, str] = {}
    for p in pairs:
        split = assignment.get(p.pair_id)
        if split not in SPLITS:
            f.append(Finding("error", "unassigned_pair", f"{p.pair_id} -> {split!r}"))
            continue
        for eid in (p.episode_a, p.episode_b):
            if eid not in by_id:
                f.append(Finding("error", "missing_episode", eid))
            elif ep_split.setdefault(eid, split) != split:
                f.append(Finding("error", "pair_split_leak", f"episode {eid} in two splits"))
    if set(assignment) - {p.pair_id for p in pairs}:
        f.append(Finding("error", "unknown_pair", "assignment names pairs not in suite"))
    # group keys: shared scene script / content must not straddle splits
    script_splits: dict[str, set[str]] = defaultdict(set)
    maps_by_split: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        s = assignment.get(p.pair_id)
        script_splits[p.scene_script_id].add(s)
        maps_by_split[s].add(p.identity_mapping_id)
    for script, splits in script_splits.items():
        if len(splits) > 1:
            f.append(Finding("error", "script_group_leak", f"script {script[:12]} in {sorted(splits)}"))
    content_splits: dict[str, list[str]] = defaultdict(list)
    for eid, split in ep_split.items():
        content_splits[by_id[eid].content_sha256()].append(split)
    for h, splits in content_splits.items():
        if len(splits) > 1:
            f.append(Finding("error", "duplicate_content",
                             f"content {h[:12]} appears {len(splits)}x in {sorted(set(splits))}"))
    seen_elsewhere = set().union(*(maps_by_split[s] for s in SPLITS if s != "test_counterfactual"))
    if maps_by_split["test_counterfactual"] & seen_elsewhere:
        f.append(Finding("error", "mapping_leak",
                         "test_counterfactual shares identity mappings with other splits"))
    f += _ood_findings(pairs, assignment)
    f += _balance_findings(by_id, pairs, assignment)
    return f


def _ood_findings(pairs: Sequence[PairRecord], assignment: Mapping[str, str]) -> list[Finding]:
    out = []
    for family in sorted({p.family for p in pairs}):
        ind = [p.horizon for p in pairs
               if p.family == family and assignment.get(p.pair_id) in IN_DIST_SPLITS]
        ood = [p.horizon for p in pairs if p.family == family
               and assignment.get(p.pair_id) in ("test_ood", "test_ood_length")]
        if ind and ood and min(ood) <= max(ind):
            out.append(Finding("error", "ood_not_longer",
                               f"{family}: ood min {min(ood)} <= in-dist max {max(ind)}"))
    return out


def _balance_findings(by_id: Mapping[str, EpisodeRecord], pairs: Sequence[PairRecord],
                      assignment: Mapping[str, str]) -> list[Finding]:
    out = []
    groups: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    n_opts: dict[tuple[str, str], int] = {}
    for p in pairs:
        key = (p.family, assignment.get(p.pair_id, "?"))
        groups[key].update([p.target_a, p.target_b])
        n_opts[key] = len(by_id[p.episode_a].queries[0].options) if p.episode_a in by_id else 2
    for key, counts in sorted(groups.items()):
        total = sum(counts.values())
        top = max(counts.values()) / total
        if total >= 20 and top > 1 / n_opts[key] + BALANCE_MARGIN:
            out.append(Finding("warn", "label_imbalance", f"{key}: majority label {top:.2f}"))
    return out


def build_split_manifest(config: SODHConfig, pairs: Sequence[PairRecord],
                         episodes: Sequence[EpisodeRecord],
                         assignment: Mapping[str, str]) -> dict[str, Any]:
    by_id = {e.episode_id: e for e in episodes}
    present = tuple(s for s in SPLITS if s in set(assignment.values()))
    members = {s: sorted(p for p, v in assignment.items() if v == s) for s in present}
    pair_by_id = {p.pair_id: p for p in pairs}
    ep_hashes = {
        s: sorted(by_id[e].episode_sha256()
                  for pid in ids for e in (pair_by_id[pid].episode_a, pair_by_id[pid].episode_b))
        for s, ids in members.items()
    }
    body = {
        "mode": "synthetic",
        "config_sha256": config.sha256(),
        "generator_version": config.generator_version,
        "master_seed": config.master_seed,
        "heldout_mappings": list(config.heldout_mappings),
        "split_fractions": list(config.split_fractions),
        "pairs": members,
        "episode_sha256": ep_hashes,
        "counts": {s: {"pairs": len(members[s]), "episodes": 2 * len(members[s])} for s in present},
    }
    return {"split_manifest_id": "sodh-split-" + digest(body)[:12], **body}


def write_once(path: Path, data: bytes) -> None:
    """Create ``path`` or confirm it already holds identical bytes; never overwrite."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"{path} is immutable; register a new version id instead")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (canonical_json(manifest) + "\n").encode("ascii")
