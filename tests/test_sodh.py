# SPDX-License-Identifier: Apache-2.0
import pytest

from tests.conftest import small_config
from ucvla.canonical import HashRng, canonical_json
from ucvla.sodh import RELEVANT_FAMILIES, VOCAB, SODHConfig, generate_suite


def blob(suite):
    return "".join(canonical_json(e) for e in suite.episodes) + canonical_json(suite.pairs)


def test_same_config_gives_byte_identical_suite(suite, cfg):
    """A3."""
    assert blob(generate_suite(cfg)) == blob(suite)


def test_different_seed_changes_content(suite):
    other = generate_suite(small_config(master_seed=8))
    assert blob(other) != blob(suite)
    assert {e.content_sha256() for e in other.episodes}.isdisjoint(
        {e.content_sha256() for e in suite.episodes})


def test_items_do_not_shift_when_suite_grows(cfg):
    """Per-item keyed randomness: adding pairs never perturbs earlier ones."""
    bigger = small_config()
    raw = {f: v for f, v in cfg.families.items()}
    raw["object_correspondence"] = type(raw["object_correspondence"])(
        **{**raw["object_correspondence"].__dict__, "pairs_in_dist": 20})
    grown = generate_suite(SODHConfig(**{**bigger.__dict__, "families": raw}))
    base = {e.episode_id: e.content_sha256() for e in generate_suite(cfg).episodes}
    new = {e.episode_id: e.content_sha256() for e in grown.episodes}
    assert all(new.get(k) == v for k, v in base.items() if k in new)


def test_pairs_are_same_observation_different_history(suite):
    """A4."""
    by_id = {e.episode_id: e for e in suite.episodes}
    for p in suite.pairs:
        a, b = by_id[p.episode_a], by_id[p.episode_b]
        va, vb = a.causal_view(p.query_time_ns), b.causal_view(p.query_time_ns)
        assert va.digest(False) == vb.digest(False) == p.o_t_digest_a == p.o_t_digest_b
        if p.history_relevant:
            assert p.target_a != p.target_b
            assert va.history != vb.history
        else:
            assert p.target_a == p.target_b
        assert len(a.frames) == len(b.frames)  # no episode-length shortcut


def test_all_families_and_regimes_present(suite):
    assert {p.family for p in suite.pairs} >= set(RELEVANT_FAMILIES) | {"irrelevant_history"}
    assert {p.regime for p in suite.pairs} == {"in_dist", "ood", "heldout_mapping"}
    assert len(suite.pairs) >= 4 * 24


def test_scene_scripts_unique(suite):
    ids = [p.scene_script_id for p in suite.pairs]
    assert len(ids) == len(set(ids))


def test_heldout_mapping_uses_unseen_vocabulary(suite):
    by_id = {e.episode_id: e for e in suite.episodes}

    def vocab(pair):
        return {v for e in (by_id[pair.episode_a], by_id[pair.episode_b]) for f in e.frames
                for v in f.visible.values() if v in VOCAB}

    held = [p for p in suite.pairs if p.regime == "heldout_mapping" and p.history_relevant]
    seen = [p for p in suite.pairs if p.regime != "heldout_mapping" and p.history_relevant]
    held_v = set().union(*(vocab(p) for p in held))
    seen_v = set().union(*(vocab(p) for p in seen))
    assert held_v and not (held_v & seen_v)


def test_ood_is_strictly_longer_horizon(suite):
    for fam in RELEVANT_FAMILIES:
        ind = max(p.horizon for p in suite.pairs if p.family == fam and p.regime == "in_dist")
        ood = min(p.horizon for p in suite.pairs if p.family == fam and p.regime == "ood")
        assert ood > ind


def test_config_validation():
    with pytest.raises(ValueError, match="leak"):
        small_config(leak="bogus")
    with pytest.raises(ValueError, match="heldout"):
        small_config(heldout_mappings=[9])
    with pytest.raises(ValueError, match="sum to 1"):
        small_config(split_fractions=[0.5, 0.2, 0.2])


def test_hash_rng_is_stable_and_unbiased():
    r = HashRng("pin", 1)
    # Pinned stream: any change breaks cross-version reproducibility of every artifact.
    assert [r.below(10**6) for _ in range(3)] == [268238, 30812, 601768]
    counts = [0] * 5
    r = HashRng("u")
    for _ in range(5000):
        counts[r.below(5)] += 1
    assert min(counts) > 850
