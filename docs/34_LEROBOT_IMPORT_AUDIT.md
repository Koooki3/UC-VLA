# 34 | LeRobot importer and visual/metadata/time audit (2026-11-13)

Data: official `lerobot/svla_so101_pickplace`, revision `f641879e…`, 86.1 MB, 5 LFS hashes verified, record `results/real/lerobot_sample/data_use_record.json`. Importer `src/ucvla/lerobot_import.py`, audit `scripts/validate_lerobot_sample.py`, report `results/real/lerobot_sample/schema_audit.json`. Real teleoperation data: never mixed with SODH; used only to validate the data path. **A successful import says nothing about visual-memory ability.**

## Mapping (what is and is not lossless)
Mapped: both AV1 cameras (lazy per-episode decode; sha256 of decoded rgb24 + video ref), measured `observation.state` (proprio), `action` as **commanded** (issued 1 ns after its observation: invisible at tick *i*, visible from *i+1*), canonical time `frame_index/fps`, task text as an OBSERVED t=0 instruction. **Not mapped / not inferred:** executed actions (none exist in this format; `executed_origin=absent`, never derived from state), pixels (hash only), `meta/stats.json` (not used: train-only normalisation rule), bookkeeping indices (kept out of views).

## Results (50 episodes, 11,939 frames, all checks passed)
- Alignment: max |parquet timestamp − frame_index/30| = 4.5e-7 s; max |video pts − expected| = 9e-14 s; decoded frame counts equal episode lengths for both cameras.
- Round-trip serialisation equal; `causal_view` at tick *i* holds exactly commands 0..*i*−1; no bookkeeping field (`episode_index`, `frame_index`, timestamps, episode end) appears in any view.
- **Visual audit:** 0 exact duplicate decoded frames across episodes; first-frame mean intensity vs episode index Spearman ρ = 0.27 (below the 0.5 flag: no strong recording-drift leak for ordered splits, weak evidence at n=50).
- **Time shortcut:** normalised time alone explains R² = 0.27 of held-out action variance — a real policy must beat this to show anything beyond a phase prior.
- **Commanded vs measured:** mean |action_t − state_{t+1}| = 2.40 < |action_t − state_t| = 2.98: commands lead the measured state, as expected; this is *not* an executed-action stream.

## Limits (UNVERIFIED / not done)
robot_type says `so100_follower` while the repo is named so101; data authorship and provenance unknown; licence only in the card metadata (no LICENSE file); the visual audit is hash/statistics-level — **no learned image-shortcut probe** (e.g. a CNN predicting episode id or phase from a single frame) and no perceptual near-duplicate detection beyond exact hashes; one dataset only. M1 DoD item "LeRobot importer": **met for this one official sample**; "image-level shortcut audit": **partially met** (statistical, not learned).
