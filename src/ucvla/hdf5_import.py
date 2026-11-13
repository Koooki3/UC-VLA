# SPDX-License-Identifier: Apache-2.0
"""RoboMME HDF5 -> EpisodeRecord importer (status: ``implemented`` for the documented layout,
validated on ONE official 100-episode training sample (StopCube); other tasks untested).

Layout (doc/h5_data_format.md of the benchmark repo, verified on the sample):
  episode_N/{setup/{seed,difficulty,task_goal,available_multi_choices,*_intrinsic},
             timestep_K/{obs/*, action/*, info/*}}   (timestep keys are NOT in numeric order)

Contract decisions (what cannot be mapped losslessly is listed in LOSSY_FIELDS / docs/22 (withheld from the public release)):
  * pixels are never copied: a Frame carries sha256 of the raw rgb/depth bytes plus an HDF5 ref;
  * time is DERIVED from an operator-set rate (``assumed_hz``): the file has no timestamps;
  * ``joint_action`` is a *commanded* action. It is issued 1 ns after its observation, so a causal
    view at tick i contains commands 0..i-1 only (a policy never sees the action it must predict);
  * there is no executed-action stream: executed is empty and ``executed_origin='absent'``;
  * oracle fields (``*_online`` subgoals, seed, difficulty, goal text, choices) go to PrivilegedState,
    planner-view subgoals/flags to ANNOTATION events — neither can enter a CausalView.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import h5py
import numpy as np

from ucvla.benchmark_contract import SourceManifest
from ucvla.canonical import canonical_json, digest
from ucvla.episode import (
    SCHEMA_VERSION,
    CommandedAction,
    EpisodeRecord,
    Event,
    Frame,
    PrivilegedState,
    ProprioSample,
    SourceType,
)

IMPORTER = "ucvla.hdf5_import"
IMPORTER_VERSION = "robomme-h5/1"
LOSSY_FIELDS: dict[str, str] = {
    "obs/*_rgb, obs/*_depth": "pixels/depth not stored; only sha256 + HDF5 ref (re-read on demand)",
    "obs/*_camera_extrinsic, setup/*_camera_intrinsic": "calibration not represented in EpisodeRecord",
    "obs/eef_state_raw/*, obs/is_gripper_close": "redundant with eef_state/gripper_state; dropped",
    "action/eef_action, action/eef_action_raw/*, action/waypoint_action, action/choice_action":
        "only joint_action is mapped to CommandedAction; others need their own typed channel",
    "timestamps": "absent in file; derived from assumed_hz (provenance timestamps_origin=derived)",
    "executed actions": "absent in file (commands only); executed_origin=absent",
    "info/is_video_demo": "mapped to annotation events only when true",
    "setup/available_multi_choices": "privileged state string (not policy input)",
    "setup/task_goal": "instruction is an OBSERVED t=0 event; which paraphrase is used is unverified",
}
PRIVILEGED_FIELDS = (
    "simple_subgoal_online", "grounded_subgoal_online", "seed", "difficulty",
    "available_multi_choices", "simple_subgoal", "grounded_subgoal", "is_subgoal_boundary",
    "is_completed", "is_video_demo", "waypoint_action", "choice_action")
CAMERAS = {"front": "cam_front", "wrist": "cam_wrist"}


@dataclass(frozen=True)
class HDF5ImportConfig:
    task: str
    source_revision: str  # exact HF dataset commit
    source_file_sha256: str  # hash of the extracted .h5 (or of the archive; recorded either way)
    assumed_hz: float = 20.0  # UNVERIFIED for MME; operator-set, flagged derived

    def sha256(self) -> str:
        return digest(self)


def _s(x: Any) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def _raw_hash(arr: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def timestep_ids(ep: h5py.Group) -> list[int]:
    ids = sorted(int(k.split("_")[1]) for k in ep if k.startswith("timestep_"))
    if ids != list(range(len(ids))):
        raise ValueError("timestep keys are not contiguous 0..N-1")
    return ids


def import_episode(f: h5py.File, key: str, cfg: HDF5ImportConfig) -> EpisodeRecord:
    ep = f[key]
    ids = timestep_ids(ep)
    dt = round(1e9 / cfg.assumed_hz)
    eid = f"robomme-{cfg.task}-{key}"
    frames, proprio, commanded, events = [], [], [], []
    oracle: list[list[str]] = []
    last_online: tuple[str, str] | None = None
    for i in ids:
        t = i * dt
        g = ep[f"timestep_{i}"]
        for cam, cam_id in CAMERAS.items():
            rgb, depth = g[f"obs/{cam}_rgb"][()], g[f"obs/{cam}_depth"][()]
            vis = {"rgb_sha256": _raw_hash(rgb), "depth_sha256": _raw_hash(depth),
                   "shape": "x".join(map(str, rgb.shape)),
                   "ref": f"{key}/timestep_{i}/obs/{cam}_rgb"}
            frames.append(Frame.make(f"{eid}-f{i}-{cam}", cam_id, t, vis))
        values = [*g["obs/joint_state"][()], *g["obs/gripper_state"][()], *g["obs/eef_state"][()]]
        proprio.append(ProprioSample(t, tuple(float(v) for v in values)))
        act = [float(v) for v in g["action/joint_action"][()]]
        commanded.append(CommandedAction(f"{eid}-c{i}", t + 1, canonical_json(act)))
        info = g["info"]
        if bool(info["is_subgoal_boundary"][()]):
            events.append(Event.make(f"{eid}-e{i}b", t, "subgoal_boundary",
                                     {"simple_subgoal": _s(info["simple_subgoal"][()])},
                                     SourceType.ANNOTATION))
        if bool(info["is_completed"][()]):
            events.append(Event.make(f"{eid}-e{i}c", t, "completed", {}, SourceType.ANNOTATION))
        if bool(info["is_video_demo"][()]):
            events.append(Event.make(f"{eid}-e{i}v", t, "video_demo", {}, SourceType.ANNOTATION))
        online = (_s(info["simple_subgoal_online"][()]), _s(info["grounded_subgoal_online"][()]))
        if online != last_online:
            oracle.append([str(i), *online])
            last_online = online
    setup = ep["setup"]
    goals = [_s(x) for x in setup["task_goal"][()]]
    # the language instruction is a legitimate policy input: an OBSERVED event at t=0. Which of the
    # paraphrases a policy receives is undocumented (UNVERIFIED), so both are carried.
    events.insert(0, Event.make(f"{eid}-instr", 0, "instruction",
                                {"text": goals[0], "alt": goals[-1]}, SourceType.OBSERVED))
    seed = int(setup["seed"][()])
    priv = {"seed": str(seed), "difficulty": _s(setup["difficulty"][()]),
            "task_goal": json.dumps(goals),
            "available_multi_choices": _s(setup["available_multi_choices"][()]),
            "oracle_subgoal_online": json.dumps(oracle)}
    prov = {
        "generator": IMPORTER, "generator_version": IMPORTER_VERSION,
        "config_sha256": cfg.sha256(), "seed": seed, "source_format": "hdf5",
        "timestamps_origin": "derived", "assumed_hz": f"{cfg.assumed_hz}",
        "executed_origin": "absent", "source_revision": cfg.source_revision,
        "source_file_sha256": cfg.source_file_sha256, "source_episode": key,
        "license_ref": "dataset card: apache-2.0; LICENSE has unfilled copyright holder "
                       "(operator clearance pending, docs/22 (withheld from the public release))",
        "n_timesteps": len(ids),
    }
    return EpisodeRecord(
        schema_version=SCHEMA_VERSION, episode_id=eid, task=cfg.task,
        clock_domain=f"derived_{cfg.assumed_hz:g}hz", frames=tuple(frames),
        proprio=tuple(proprio), commanded=tuple(commanded), executed=(), events=tuple(events),
        queries=(), targets=(), privileged=PrivilegedState((len(ids) - 1) * dt, priv),
        provenance=prov)


def read_frame_pixels(f: h5py.File, ref: str) -> np.ndarray:
    """Re-read raw pixels by the HDF5 ref stored in a Frame (the hash binds them)."""
    return np.asarray(f[ref][()])


class RoboMMEHDF5Source:
    """EpisodeSource over one downloaded task file. Only the official TRAIN sample is supported."""

    def __init__(self, path: str, cfg: HDF5ImportConfig, split: str = "train") -> None:
        self._f = h5py.File(path, "r")
        self._cfg = cfg
        self._split = split
        self.manifest = SourceManifest(
            name=f"robomme_{cfg.task}_{split}_sample", version=cfg.source_revision,
            source_format="hdf5", license_status="unverified",
            license_ref="apache-2.0 per card; copyright holder unspecified (docs/22 (withheld from the public release))",
            official_protocol_ref="https://github.com/RoboMME/robomme_benchmark",
            privileged_fields=PRIVILEGED_FIELDS)

    def split_names(self) -> Sequence[str]:
        return (self._split,)

    def episode_ids(self, split: str) -> list[str]:
        keys = sorted((k for k in self._f if k.startswith("episode_")),
                      key=lambda s: int(s.split("_")[1]))
        return [f"robomme-{self._cfg.task}-{k}" for k in keys]

    def load(self, episode_id: str) -> EpisodeRecord:
        key = episode_id.removeprefix(f"robomme-{self._cfg.task}-")
        return import_episode(self._f, key, self._cfg)

    def raw_file(self) -> h5py.File:
        return self._f

    def close(self) -> None:
        self._f.close()
