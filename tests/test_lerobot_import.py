# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ucvla.canonical import canonical_json
from ucvla.episode import EpisodeRecord, SourceType
from ucvla.lerobot_import import LeRobotImportConfig, LeRobotSource

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "datasets/lerobot_sample"
CFG = LeRobotImportConfig("toy", "f641879e22172be7e8161d5e6c1503c2d2feb657", cameras=("observation.images.up",))


def fake_decoder(path, start, n):
    return [(start + i / 30, bytes([i % 251]) * 12, (2, 2, 3)) for i in range(n)]


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "meta/info.json").write_text(json.dumps({"fps": 30}))
    lens = [5, 4]
    eps, rows = [], {"action": [], "observation.state": [], "timestamp": [], "frame_index": [], "episode_index": [], "index": [], "task_index": []}
    g = 0
    for e, n in enumerate(lens):
        eps.append({"episode_index": e, "dataset_from_index": g, "dataset_to_index": g + n, "length": n,
                    "tasks": ["put the block in the box"],
                    "videos/observation.images.up/chunk_index": 0, "videos/observation.images.up/file_index": 0,
                    "videos/observation.images.up/from_timestamp": 10.0 * e})
        for i in range(n):
            rows["action"].append([float(g + i), 1.0]); rows["observation.state"].append([float(g + i) - 0.5, 0.0])
            rows["timestamp"].append(i / 30); rows["frame_index"].append(i); rows["episode_index"].append(e)
            rows["index"].append(g + i); rows["task_index"].append(0)
        g += n
    pq.write_table(pa.Table.from_pylist(eps), tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.table(rows), tmp_path / "data/chunk-000/file-000.parquet")
    return tmp_path


def test_mapping_roundtrip_and_causality(root):
    src = LeRobotSource(root, CFG, fake_decoder)
    ep = src.load(src.episode_ids("train")[1])
    assert EpisodeRecord.from_dict(json.loads(canonical_json(ep.to_dict()))) == ep
    assert len(ep.tick_times()) == 4 and ep.executed == () and ep.provenance["executed_origin"] == "absent"
    assert ep.provenance["timestamps_origin"] == "native"
    assert json.loads(ep.commanded[0].action) == [5.0, 1.0] and ep.proprio[0].values == (4.5, 0.0)
    ticks = ep.tick_times()
    for i in range(4):
        assert len(ep.causal_view(ticks[i]).commanded) == i  # the command answering obs i is invisible
    assert [e.source_type for e in ep.events] == [SourceType.OBSERVED]
    v = ep.causal_view(ticks[2])
    assert "episode_index" not in canonical_json([v.ticks, v.proprio, v.commanded])


def test_boundary_and_frame_count_errors(root):
    src = LeRobotSource(root, CFG, lambda p, s, n: fake_decoder(p, s, n)[:-1])
    with pytest.raises(ValueError, match="decoded"):
        src.load(src.episode_ids("train")[0])


def test_non_contiguous_frames_rejected(root):
    t = pq.read_table(root / "data/chunk-000/file-000.parquet").to_pydict()
    t["frame_index"][2] = 9
    pq.write_table(pa.table(t), root / "data/chunk-000/file-000.parquet")
    src = LeRobotSource(root, CFG, fake_decoder)
    with pytest.raises(ValueError, match="contiguous"):
        src.load(src.episode_ids("train")[0])


@pytest.mark.skipif(not (REAL / "meta/info.json").exists(), reason="official LeRobot sample not downloaded")
def test_real_sample_episode_decodes_lazily_and_aligns():
    cfg = LeRobotImportConfig("svla_so101_pickplace", "f641879e22172be7e8161d5e6c1503c2d2feb657")
    src = LeRobotSource(REAL, cfg)
    ep = src.load(src.episode_ids("train")[3])
    assert float(ep.provenance["max_video_pts_dev_s"]) < 1e-6 and float(ep.provenance["max_parquet_ts_dev_s"]) < 1e-3
    assert {f.camera_id for f in ep.frames} == {"cam_up", "cam_side"} and ep.executed == ()
