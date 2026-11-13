# SPDX-License-Identifier: Apache-2.0
import dataclasses

import pytest

from ucvla.splits import (
    V1_SPLITS,
    assign_splits,
    build_split_manifest,
    check_splits,
    manifest_bytes,
    write_once,
)


def codes(findings):
    return {f.code for f in findings if f.level == "error"}


def test_assignment_is_deterministic_and_complete(suite, cfg, assignment):
    assert assignment == assign_splits(suite.pairs, cfg)
    assert set(assignment) == {p.pair_id for p in suite.pairs}
    assert set(assignment.values()) == set(V1_SPLITS)


def test_clean_suite_passes(suite, assignment):
    assert codes(check_splits(suite.episodes, suite.pairs, assignment)) == set()


def test_pairs_never_straddle_splits(suite, assignment):
    by_id = {p.pair_id: p for p in suite.pairs}
    split_of = {}
    for pid, s in assignment.items():
        for e in (by_id[pid].episode_a, by_id[pid].episode_b):
            assert split_of.setdefault(e, s) == s


def test_checker_rejects_split_between_pair_members(suite, assignment):
    """A5: negative control for the checker itself."""
    bad = dict(assignment)
    p = next(p for p in suite.pairs if assignment[p.pair_id] == "train")
    # Move another pair that shares nothing: still valid. Now force a shared script across splits.
    q = next(q for q in suite.pairs if assignment[q.pair_id] == "val")
    pairs = [dataclasses.replace(q, scene_script_id=p.scene_script_id) if x is q else x
             for x in suite.pairs]
    assert "script_group_leak" in codes(check_splits(suite.episodes, pairs, bad))


def test_checker_rejects_unassigned_and_unknown(suite, assignment):
    bad = dict(assignment)
    bad.pop(suite.pairs[0].pair_id)
    bad["pair-ghost"] = "train"
    c = codes(check_splits(suite.episodes, suite.pairs, bad))
    assert {"unassigned_pair", "unknown_pair"} <= c


def test_checker_rejects_duplicate_content_across_splits(suite, assignment):
    train = next(p for p in suite.pairs if assignment[p.pair_id] == "train")
    test = next(p for p in suite.pairs if assignment[p.pair_id] == "test_id"
                and p.family == train.family)
    by_id = {e.episode_id: e for e in suite.episodes}
    clone = dataclasses.replace(by_id[train.episode_a], episode_id="ep-clone")
    # frame ids embed the original id; content hash ignores ids, which is the point
    eps = [clone if e.episode_id == test.episode_a else e for e in suite.episodes]
    pairs = [dataclasses.replace(p, episode_a="ep-clone") if p is test else p for p in suite.pairs]
    assert "duplicate_content" in codes(check_splits(eps, pairs, assignment))


def test_checker_rejects_seen_mapping_in_counterfactual(suite, assignment):
    held = next(p for p in suite.pairs if assignment[p.pair_id] == "test_counterfactual")
    bad = dict(assignment)
    train = next(p for p in suite.pairs if assignment[p.pair_id] == "train")
    pairs = [dataclasses.replace(held, identity_mapping_id=train.identity_mapping_id)
             if p is held else p for p in suite.pairs]
    assert "mapping_leak" in codes(check_splits(suite.episodes, pairs, bad))


def test_checker_rejects_non_longer_ood(suite, assignment):
    ood = next(p for p in suite.pairs if assignment[p.pair_id] == "test_ood")
    pairs = [dataclasses.replace(ood, horizon=1) if p is ood else p for p in suite.pairs]
    assert "ood_not_longer" in codes(check_splits(suite.episodes, pairs, assignment))


def test_manifest_is_content_addressed_and_write_once(tmp_path, suite, cfg, assignment):
    m = build_split_manifest(cfg, suite.pairs, suite.episodes, assignment)
    assert m == build_split_manifest(cfg, suite.pairs, suite.episodes, assignment)
    assert m["mode"] == "synthetic" and m["split_manifest_id"].startswith("sodh-split-")
    path = tmp_path / "split.json"
    write_once(path, manifest_bytes(m))
    write_once(path, manifest_bytes(m))  # identical re-write is a no-op
    with pytest.raises(FileExistsError, match="immutable"):
        write_once(path, manifest_bytes({**m, "master_seed": 1}))
