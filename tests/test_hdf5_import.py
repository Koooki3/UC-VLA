"""HDF5 importer: schema round-trip on a synthetic file with the official RoboMME layout (always
runs) and on the real downloaded sample (skipped when the data is absent)."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ucvla.benchmark_contract import HDF5_IMPORTER, check_source_conformance
from ucvla.canonical import canonical_json
from ucvla.episode import EpisodeRecord, SourceType
from ucvla.hdf5_import import (
    LOSSY_FIELDS,
    PRIVILEGED_FIELDS,
    HDF5ImportConfig,
    RoboMMEHDF5Source,
    import_episode,
    read_frame_pixels,
    timestep_ids,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "datasets/robomme_sample/extracted/record_dataset_StopCube.h5"
CFG = HDF5ImportConfig("StopCube", "a5e4e25ffe8af34f64944f9533d06455ce5f8337", "0" * 64)


def make_h5(path: Path, n_steps: int = 12, n_eps: int = 3, boundaries=(0, 5)) -> Path:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for e in range(n_eps):
            ep = f.create_group(f"episode_{e}")
            s = ep.create_group("setup")
            s["seed"] = 2000 + e
            s["difficulty"] = "easy"
            s["task_goal"] = np.array(["press at the target", "stop the cube"], dtype=object)
            s["available_multi_choices"] = "[]"
            s["front_camera_intrinsic"] = np.eye(3, dtype=np.float32)
            for i in range(n_steps):
                g = ep.create_group(f"timestep_{i}")
                for cam in ("front", "wrist"):
                    g[f"obs/{cam}_rgb"] = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                    g[f"obs/{cam}_depth"] = rng.integers(0, 3000, (8, 8, 1)).astype(np.int16)
                g["obs/joint_state"] = rng.normal(size=7).astype(np.float32)
                g["obs/gripper_state"] = rng.normal(size=2).astype(np.float32)
                g["obs/eef_state"] = rng.normal(size=6).astype(np.float32)
                g["action/joint_action"] = rng.normal(size=8)
                g["info/is_subgoal_boundary"] = i in boundaries
                g["info/is_completed"] = i == n_steps - 1
                g["info/is_video_demo"] = False
                g["info/simple_subgoal"] = "reach"
                g["info/grounded_subgoal"] = "reach button"
                g["info/simple_subgoal_online"] = f"online-{i // 4}"
                g["info/grounded_subgoal_online"] = f"g-online-{i // 4}"
    return path


@pytest.fixture(scope="module")
def h5_path(tmp_path_factory):
    return make_h5(tmp_path_factory.mktemp("h5") / "mini.h5")


def test_timestep_keys_sorted_numerically_not_lexicographically(h5_path):
    with h5py.File(h5_path, "r") as f:
        keys = [k for k in f["episode_0"] if k.startswith("timestep_")]
        assert keys != [f"timestep_{i}" for i in range(12)]  # storage order is lexicographic
        assert timestep_ids(f["episode_0"]) == list(range(12))


def test_roundtrip_is_exact_for_mapped_fields(h5_path):
    with h5py.File(h5_path, "r") as f:
        ep = import_episode(f, "episode_1", CFG)
        assert EpisodeRecord.from_dict(json.loads(canonical_json(ep.to_dict()))) == ep
        for i in range(12):
            g = f[f"episode_1/timestep_{i}"]
            exp = [*g["obs/joint_state"][()], *g["obs/gripper_state"][()], *g["obs/eef_state"][()]]
            assert list(ep.proprio[i].values) == [float(v) for v in exp]  # exact float32 values
            assert json.loads(ep.commanded[i].action) == [float(v) for v in g["action/joint_action"][()]]
        fr = ep.frames[2]
        pix = read_frame_pixels(f, fr.visible["ref"])
        from hashlib import sha256
        assert sha256(np.ascontiguousarray(pix).tobytes()).hexdigest() == fr.visible["rgb_sha256"]
        assert fr.source_sha256 == type(fr).content_hash(fr.camera_id, fr.time_ns, fr.visible)


def test_a_policy_never_sees_the_action_it_must_predict(h5_path):
    with h5py.File(h5_path, "r") as f:
        ep = import_episode(f, "episode_0", CFG)
        ticks = ep.tick_times()
        for i in range(1, 12):
            view = ep.causal_view(ticks[i])
            assert len(view.commanded) == i  # commands 0..i-1 only; command i answers obs i
            assert max(c.issued_at_ns for c in view.commanded) <= ticks[i]
        assert len(ep.causal_view(ticks[0]).commanded) == 0


def test_oracle_fields_and_annotations_never_reach_a_view(h5_path):
    with h5py.File(h5_path, "r") as f:
        ep = import_episode(f, "episode_0", CFG)
        assert {e.source_type for e in ep.events} == {SourceType.ANNOTATION}
        assert "oracle_subgoal_online" in ep.privileged.state and "seed" in ep.privileged.state
        for view in ep.replay():
            assert view.events == ()
            blob = canonical_json([view.ticks, view.proprio, view.commanded])
            for secret in ("online-", "g-online", "easy", "stop the cube"):
                assert secret not in blob
            for fr in view.ticks:
                for cam in fr.cameras.values():
                    assert set(cam) == {"rgb_sha256", "depth_sha256", "shape", "ref"}


def test_provenance_labels_derived_and_absent_honestly(h5_path):
    with h5py.File(h5_path, "r") as f:
        ep = import_episode(f, "episode_0", CFG)
    assert ep.provenance["timestamps_origin"] == "derived" and ep.provenance["assumed_hz"] == "20.0"
    assert ep.provenance["executed_origin"] == "absent" and ep.executed == ()
    assert ep.provenance["source_revision"] == CFG.source_revision
    assert ep.tick_times()[1] - ep.tick_times()[0] == 50_000_000


def test_source_conformance_flags_only_the_uncleared_licence(h5_path):
    src = RoboMMEHDF5Source(str(h5_path), CFG)
    try:
        assert check_source_conformance(src, max_episodes=3) == [
            "licence status 'unverified': not cleared for use"]
    finally:
        src.close()


def test_lossy_fields_are_declared_and_importers_for_other_formats_still_fail():
    assert {"timestamps", "executed actions"} <= set(LOSSY_FIELDS)
    assert "seed" in PRIVILEGED_FIELDS and "simple_subgoal_online" in PRIVILEGED_FIELDS
    with pytest.raises(NotImplementedError):
        HDF5_IMPORTER("x")  # the generic placeholder remains a placeholder


def test_non_contiguous_timesteps_rejected(tmp_path):
    p = make_h5(tmp_path / "gap.h5", n_eps=1)
    with h5py.File(p, "a") as f:
        del f["episode_0/timestep_5"]
    with h5py.File(p, "r") as f, pytest.raises(ValueError, match="contiguous"):
        import_episode(f, "episode_0", CFG)


@pytest.mark.skipif(not REAL.exists(), reason="official RoboMME sample not downloaded")
def test_real_sample_first_episode_imports_and_is_causal():
    with h5py.File(REAL, "r") as f:
        ep = import_episode(f, "episode_0", HDF5ImportConfig(
            "StopCube", "a5e4e25ffe8af34f64944f9533d06455ce5f8337", "b936e52b" + "0" * 56))
        assert len(ep.tick_times()) == len(timestep_ids(f["episode_0"])) == 459
        assert int(ep.provenance["seed"]) == int(f["episode_0/setup/seed"][()])
        v = ep.causal_view(ep.tick_times()[100])
        assert len(v.commanded) == 100 and len(v.ticks) == 101
