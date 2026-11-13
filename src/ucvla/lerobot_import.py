# SPDX-License-Identifier: Apache-2.0
"""LeRobotDataset v3.0 -> EpisodeRecord importer (status: ``implemented``; validated on ONE official
sample, ``lerobot/svla_so101_pickplace`` at a pinned revision).

Contract decisions: pixels are never copied (sha256 of decoded rgb24 bytes + a video ref); video is
decoded LAZILY per episode (seek to the episode's start timestamp); time is the canonical
``frame_index / fps`` and the parquet timestamp and video pts are CHECKED against it (max deviations
reported); ``action`` is a COMMANDED action issued 1 ns after its observation (visible from the next
tick only); ``observation.state`` is MEASURED proprio; there is NO executed-action stream in this
format, so ``executed`` stays empty (``executed_origin='absent'``) and is never inferred; the task
text is an OBSERVED t=0 instruction event; episode/frame/global indices and episode end times are
bookkeeping and never enter a view."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq

from ucvla.benchmark_contract import SourceManifest
from ucvla.canonical import canonical_json, digest
from ucvla.episode import (
    SCHEMA_VERSION,
    CommandedAction,
    EpisodeRecord,
    Event,
    Frame,
    ProprioSample,
    SourceType,
)

IMPORTER = "ucvla.lerobot_import"
IMPORTER_VERSION = "lerobot-v3/1"
LOSSY_FIELDS = {
    "executed actions": "not present in the format; executed_origin=absent, never inferred from state",
    "pixels": "only sha256 of decoded rgb24 + ref; re-decode on demand",
    "timestamp (float32)": "canonical time = frame_index/fps; deviation recorded in provenance",
    "index, episode_index, task_index": "bookkeeping, excluded from views",
    "meta/stats.json": "normalisation statistics are NOT used by the importer (train-only rule)",
    "dataset task text": "mapped to an OBSERVED instruction event",
}
Decoder = Callable[[Path, float, int], list[tuple[float, bytes, tuple[int, int, int]]]]


def av_decode(path: Path, start_s: float, n_frames: int) -> list[tuple[float, bytes, tuple[int, int, int]]]:
    """Seek to the keyframe before ``start_s`` and decode forward; returns (pts_s, rgb24 bytes, shape)."""
    import av
    out: list[tuple[float, bytes, tuple[int, int, int]]] = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate)
        if start_s > 0:
            c.seek(int(max(0.0, start_s - 1.0) / s.time_base), stream=s, backward=True)
        for f in c.decode(video=0):
            pts = float(f.pts * s.time_base)
            if pts < start_s - 0.5 / fps:
                continue
            a = f.to_ndarray(format="rgb24")
            out.append((pts, a.tobytes(), tuple(a.shape)))
            if len(out) >= n_frames:
                break
    return out


@dataclass(frozen=True)
class LeRobotImportConfig:
    task: str
    revision: str
    cameras: tuple[str, ...] = ("observation.images.up", "observation.images.side")

    def sha256(self) -> str:
        return digest(self)


class LeRobotSource:
    def __init__(self, root: Path, cfg: LeRobotImportConfig, decoder: Decoder = av_decode) -> None:
        self.root, self.cfg, self.decoder = Path(root), cfg, decoder
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = float(self.info["fps"])
        self._eps = pq.read_table(self.root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
        self._data = pq.read_table(self.root / "data/chunk-000/file-000.parquet").to_pydict()
        self.manifest = SourceManifest(
            name=f"lerobot_{cfg.task}", version=cfg.revision, source_format="lerobot",
            license_status="unverified", license_ref="apache-2.0 per dataset card only; no LICENSE file",
            official_protocol_ref="https://huggingface.co/docs/lerobot/lerobot-dataset-v3",
            privileged_fields=())

    def split_names(self) -> Sequence[str]:
        return ("train",)

    def episode_ids(self, split: str) -> list[str]:
        return [f"lerobot-{self.cfg.task}-ep{r['episode_index']}" for r in self._eps]

    def load(self, episode_id: str) -> EpisodeRecord:
        idx = int(episode_id.rsplit("-ep", 1)[1])
        row = self._eps[idx]
        lo, hi = row["dataset_from_index"], row["dataset_to_index"]
        n = hi - lo
        d = self._data
        assert all(d["episode_index"][i] == idx for i in range(lo, hi)), "episode boundary mismatch"
        fidx = d["frame_index"][lo:hi]
        if fidx != list(range(n)):
            raise ValueError("frame_index not contiguous within the episode")
        ts_dev = max(abs(d["timestamp"][lo + i] - i / self.fps) for i in range(n))
        step = round(1e9 / self.fps)
        times = [i * step for i in range(n)]
        eid = episode_id
        frames: list[Frame] = []
        pts_dev = 0.0
        for cam in self.cfg.cameras:
            vp = self.root / f"videos/{cam}/chunk-{row[f'videos/{cam}/chunk_index']:03d}/file-{row[f'videos/{cam}/file_index']:03d}.mp4"
            start = float(row[f"videos/{cam}/from_timestamp"])
            dec = self.decoder(vp, start, n)
            if len(dec) != n:
                raise ValueError(f"{cam}: decoded {len(dec)} frames, expected {n}")
            for i, (pts, raw, shape) in enumerate(dec):
                pts_dev = max(pts_dev, abs((pts - start) - i / self.fps))
                short = "up" if cam.endswith("up") else "side" if cam.endswith("side") else cam
                vis = {"rgb_sha256": sha256(raw).hexdigest(), "shape": "x".join(map(str, shape)),
                       "ref": f"{cam}@{start + i / self.fps:.6f}"}
                frames.append(Frame.make(f"{eid}-f{i}-{short}", f"cam_{short}", times[i], vis))
        frames.sort(key=lambda f: (f.time_ns, f.camera_id))
        state = d["observation.state"][lo:hi]
        action = d["action"][lo:hi]
        proprio = tuple(ProprioSample(t, tuple(float(x) for x in s)) for t, s in zip(times, state))
        commanded = tuple(CommandedAction(f"{eid}-c{i}", times[i] + 1,
                                          canonical_json([float(x) for x in a]))
                          for i, a in enumerate(action))
        task_text = (row.get("tasks") or [self.cfg.task])[0]
        events = (Event.make(f"{eid}-instr", 0, "instruction", {"text": str(task_text)},
                             SourceType.OBSERVED),)
        prov = {"generator": IMPORTER, "generator_version": IMPORTER_VERSION,
                "config_sha256": self.cfg.sha256(), "seed": idx, "source_format": "lerobot",
                "timestamps_origin": "native", "executed_origin": "absent",
                "source_revision": self.cfg.revision, "source_episode": idx,
                "fps": f"{self.fps}", "max_parquet_ts_dev_s": f"{ts_dev:.3e}",
                "max_video_pts_dev_s": f"{pts_dev:.3e}", "n_frames": n,
                "license_ref": "apache-2.0 per dataset card only (operator clearance pending)"}
        return EpisodeRecord(
            schema_version=SCHEMA_VERSION, episode_id=eid, task=self.cfg.task,
            clock_domain=f"lerobot_{self.fps:g}fps", frames=tuple(frames), proprio=proprio,
            commanded=commanded, executed=(), events=events, queries=(), targets=(),
            privileged=None, provenance=prov)
