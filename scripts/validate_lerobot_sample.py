#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Schema round-trip + visual/metadata/time shortcut audit of the official LeRobot sample
(read-only, lazy decoding). Writes results/real/lerobot_sample/schema_audit.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ucvla.canonical import canonical_json
from ucvla.episode import EpisodeRecord
from ucvla.lerobot_import import LOSSY_FIELDS, LeRobotImportConfig, LeRobotSource

REV = "f641879e22172be7e8161d5e6c1503c2d2feb657"


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    src = LeRobotSource(ROOT / "datasets/lerobot_sample", LeRobotImportConfig("svla_so101_pickplace", REV))
    first_mean, per_ep, all_hashes, problems = [], [], {}, []
    time_bins, lag_next, lag_same = [], [], []
    for k, eid in enumerate(src.episode_ids("train")):
        ep = src.load(eid)
        rt = EpisodeRecord.from_dict(json.loads(canonical_json(ep.to_dict()))) == ep
        ticks = ep.tick_times()
        n = len(ticks)
        causal = all(len(ep.causal_view(ticks[i]).commanded) == i for i in (0, 1, n // 2, n - 1))
        blob = canonical_json([ep.causal_view(ticks[n // 2]).ticks, ep.causal_view(ticks[n // 2]).proprio])
        meta_clean = not any(w in blob for w in ("episode_index", "frame_index", "timestamp", "dataset_to"))
        dup = 0
        for f in ep.frames:
            h = f.visible["rgb_sha256"]
            dup += h in all_hashes and all_hashes[h] != eid
            all_hashes.setdefault(h, eid)
        raw0 = src.decoder(src.root / "videos/observation.images.up/chunk-000/file-000.mp4",
                           float(src._eps[k]["videos/observation.images.up/from_timestamp"]), 1)[0][1]
        first_mean.append(float(np.frombuffer(raw0, dtype=np.uint8).mean()))
        acts = np.array([json.loads(c.action) for c in ep.commanded])
        st = np.array([p.values for p in ep.proprio])
        lag_next.append(float(np.abs(acts[:-1] - st[1:]).mean()))
        lag_same.append(float(np.abs(acts - st).mean()))
        time_bins.append(np.stack([acts[min(n - 1, int(b * n / 20))] for b in range(20)]))
        ok = {"roundtrip": rt, "no_current_action_in_view": causal, "no_bookkeeping_in_views": meta_clean,
              "frame_count_matches_meta": n == src._eps[k]["length"]}
        per_ep.append({"episode": eid, "n": n, "dup_frames_across_episodes": int(dup),
                       "ts_dev": ep.provenance["max_parquet_ts_dev_s"], "pts_dev": ep.provenance["max_video_pts_dev_s"], **ok})
        problems += [f"{eid}:{x}" for x, v in ok.items() if not v]
    tb = np.stack(time_bins)  # episodes x 20 x dims
    train, test = tb[:35], tb[35:]
    pred = train.mean(0)  # time-only predictor (normalised time -> mean action)
    r2 = float(1 - ((test - pred) ** 2).sum() / ((test - test.mean((0, 1))) ** 2).sum())
    rep = {
        "mode": "real_data_schema_and_shortcut_audit", "dataset": "lerobot/svla_so101_pickplace", "revision": REV,
        "episodes": len(per_ep), "frames": sum(e["n"] for e in per_ep), "all_checks_passed": not problems,
        "problems": problems,
        "alignment": {"max_parquet_ts_dev_s": max(float(e["ts_dev"]) for e in per_ep),
                      "max_video_pts_dev_s": max(float(e["pts_dev"]) for e in per_ep)},
        "visual_audit": {"exact_duplicate_frames_across_episodes": sum(e["dup_frames_across_episodes"] for e in per_ep),
                         "first_frame_mean_intensity_vs_episode_index_spearman": spearman(first_mean, list(range(len(first_mean)))),
                         "note": "|rho| > 0.5 would mean sequential/ordered splits leak recording drift"},
        "time_shortcut": {"time_only_action_predictor_R2_on_held_out_episodes": r2,
                          "note": "share of action variance explained by normalised time alone; a policy should beat it"},
        "commanded_vs_measured": {"mean_abs_action_minus_next_state": float(np.mean(lag_next)),
                                  "mean_abs_action_minus_same_state": float(np.mean(lag_same)),
                                  "note": "action leads state; NO executed stream exists and none is inferred"},
        "lossy_or_unmapped": LOSSY_FIELDS,
        "unverified": ["robot_type so100_follower vs repo name so101", "authorship/provenance of teleoperation data",
                       "licence only in card metadata"],
        "per_episode": per_ep}
    out = ROOT / "results/real/lerobot_sample/schema_audit.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("episodes", "frames", "all_checks_passed", "problems", "alignment", "visual_audit", "time_shortcut", "commanded_vs_measured")}, indent=1))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
