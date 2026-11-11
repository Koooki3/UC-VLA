#!/usr/bin/env python3
"""Schema round-trip validation of the official RoboMME StopCube TRAIN sample (read-only).

Checks every episode: import, serialisation round-trip, exact proprio/command equality with the
raw HDF5, pixel-hash re-verification on sampled frames, causal-view rules, seed provenance
against the official public metadata (train/val/test). Writes a JSON report; no data is copied.
"""

import argparse
import json
import random
import sys
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ucvla.canonical import canonical_json
from ucvla.episode import EpisodeRecord
from ucvla.hdf5_import import (
    LOSSY_FIELDS,
    HDF5ImportConfig,
    import_episode,
    read_frame_pixels,
    timestep_ids,
)

D = ROOT / "datasets/robomme_sample"
REVISION = "a5e4e25ffe8af34f64944f9533d06455ce5f8337"
ARCHIVE_SHA = "b936e52b6f63dccadba998e83f3e480e5cb74130185c3b022826416469bf0565"


def seeds(path: Path) -> set[int]:
    out: set[int] = set()

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if "seed" in o:
                out.add(int(o["seed"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(json.loads(path.read_text()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", type=Path, default=ROOT / "results/real/robomme_sample/schema_validation.json")
    args = ap.parse_args()
    path = D / "extracted/record_dataset_StopCube.h5"
    cfg = HDF5ImportConfig("StopCube", REVISION, ARCHIVE_SHA)  # archive hash; file hash in report
    rng = random.Random(0)
    meta = {s: seeds(D / f"metadata/StopCube_{s}.json") for s in ("train", "val", "test")}
    per_ep, problems = [], []
    with h5py.File(path, "r") as f:
        keys = sorted((k for k in f if k.startswith("episode_")), key=lambda s: int(s.split("_")[1]))
        for key in keys[:args.episodes]:
            ep = import_episode(f, key, cfg)
            ok = {"roundtrip": EpisodeRecord.from_dict(json.loads(canonical_json(ep.to_dict()))) == ep}
            ids = timestep_ids(f[key])
            exact = True
            for i in rng.sample(ids, 6):
                g = f[f"{key}/timestep_{i}"]
                exp = [*g["obs/joint_state"][()], *g["obs/gripper_state"][()], *g["obs/eef_state"][()]]
                exact &= list(ep.proprio[i].values) == [float(v) for v in exp]
                exact &= json.loads(ep.commanded[i].action) == [float(v) for v in g["action/joint_action"][()]]
                for fr in (ep.frames[2 * i], ep.frames[2 * i + 1]):
                    pix = read_frame_pixels(f, fr.visible["ref"])
                    exact &= sha256(np.ascontiguousarray(pix).tobytes()).hexdigest() == fr.visible["rgb_sha256"]
            ok["mapped_fields_exact_and_pixel_hashes"] = bool(exact)
            ticks = ep.tick_times()
            ok["no_current_action_in_view"] = all(
                len(ep.causal_view(ticks[i]).commanded) == i for i in (0, 1, len(ticks) // 2, len(ticks) - 1))
            ok["no_annotation_or_oracle_in_views"] = all(
                [e.kind for e in v.events] == ["instruction"] for v in (ep.causal_view(t) for t in ticks[:: max(1, len(ticks) // 8)]))
            seed = int(ep.provenance["seed"])
            ok["seed_in_official_train"] = seed in meta["train"]
            ok["seed_not_in_val_or_test"] = seed not in meta["val"] | meta["test"]
            per_ep.append({"episode": key, "n_timesteps": len(ids), "seed": seed,
                           "n_events": len(ep.events), **ok})
            problems += [f"{key}:{k}" for k, v in ok.items() if not v]
    n = len(per_ep)
    report = {
        "mode": "real_data_schema_check", "task": "StopCube", "split": "train (official sample)",
        "source": {"hf_repo": "Yinpei/robomme_data_h5", "revision": REVISION,
                   "archive_sha256": ARCHIVE_SHA, "archive_bytes": 964971180,
                   "extracted_h5_bytes": path.stat().st_size},
        "episodes_checked": n, "all_checks_passed": not problems, "problems": problems,
        "timesteps": {"min": min(e["n_timesteps"] for e in per_ep),
                      "max": max(e["n_timesteps"] for e in per_ep),
                      "total": sum(e["n_timesteps"] for e in per_ep)},
        "seed_provenance": {s: len(v) for s, v in meta.items()},
        "lossy_or_unmapped": LOSSY_FIELDS,
        "unverified": ["control rate (assumed_hz=20 is an operator assumption)",
                       "extracted-file hash against the publisher (only the archive LFS sha256 is published)"],
        "per_episode": per_ep,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("episodes_checked", "all_checks_passed", "problems", "timesteps")}))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
