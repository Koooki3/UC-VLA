"""Independent oracle: recompute targets from privileged state / annotations / raw frames using separate code
(slot-permutation tracking, not name lookup; tick-level press counting; placed-event counting)."""
import json
from common import *
cfg, suite, asg = load()
bad = defaultdict(list); n = Counter()
for e in suite.episodes:
    fam = e.provenance["family"]; q = e.queries[0]; fr = [dict(f.visible) for f in e.frames if f.camera_id == "cam_front"]
    T = e.tick_times(); priv = dict(e.privileged.state); target = lab(e); n[fam] += 1
    ev = [x for x in e.events]
    def frame_idx(t): return T.index(t)
    if fam == "object_correspondence":
        nslot = int(q.params["n_slots"])
        lay = [[f[f"slot{k}"] for k in range(nslot)] for f in fr[:-1]]
        # (1) frame/event consistency: annotation 'ball_under' tick == tick where the frame shows 'under:'
        rev = [i for i, f in enumerate(fr) if f["ball"].startswith("under:")]
        if len(rev) != 1 or len(ev) != 1 or frame_idx(ev[0].time_ns) != rev[0]: bad[fam].append(("reveal_inconsistent", e.episode_id))
        cup = fr[rev[0]]["ball"][6:]
        if ev[0].payload["cup"] != cup or priv["ball_cup"] != cup: bad[fam].append(("cup_mismatch", e.episode_id))
        # (2) track by SLOT PERMUTATION (swap transitions), not by name
        slot = lay[rev[0]].index(cup)
        for a, b in zip(lay[rev[0]:], lay[rev[0] + 1:]):
            diff = [k for k in range(nslot) if a[k] != b[k]]
            if len(diff) not in (0, 2) or (diff and not (a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])): bad[fam].append(("non_swap_transition", e.episode_id))
            if len(diff) == 2 and slot in diff: slot = diff[1] if slot == diff[0] else diff[0]
        if f"pick_slot_{slot}" != target or priv["final_slot"] != str(slot): bad[fam].append(("target_mismatch", e.episode_id))
        # decision frame layout equals last shuffle layout
        if [fr[-1][f"slot{k}"] for k in range(nslot)] != lay[-1]: bad[fam].append(("final_layout_changed", e.episode_id))
    if fam == "process_count":
        K = int(q.params["target_count"]); pressed = sum(f["button"] == "pressed" for f in fr[:-1])
        if len(ev) != pressed or int(priv["presses_done"]) != pressed: bad[fam].append(("press_count_mismatch", e.episode_id))
        if any(frame_idx(x.time_ns) >= len(fr) - 1 or fr[frame_idx(x.time_ns)]["button"] != "pressed" for x in ev): bad[fam].append(("event_not_on_pressed_tick", e.episode_id))
        if ("press" if pressed < K else "stop") != target: bad[fam].append(("target_mismatch", e.episode_id))
        if pressed not in (K - 1, K): bad[fam].append(("press_outside_K-1/K", e.episode_id))
    if fam == "task_interleave":
        order = q.params["phase_order"].split(","); placed = [f["placing"] for f in fr if f["placing"] != "none"]
        if placed != order[:len(placed)] or len(ev) != len(placed) or int(priv["phases_done"]) != len(placed): bad[fam].append(("placed_mismatch", e.episode_id))
        if f"place_{order[len(placed)]}" != target: bad[fam].append(("target_mismatch", e.episode_id))
    if fam == "irrelevant_history":
        if f"pick_{fr[-1]['flag']}" != target or priv["flag"] != fr[-1]["flag"]: bad[fam].append(("flag_mismatch", e.episode_id))
    # leak scan: the label token must appear in the CausalView only inside `options`
    v = e.causal_view(q.time_ns)
    blob = json.dumps({"ticks": [t.cameras for t in v.ticks], "params": dict(v.query.params), "instr": v.query.instruction, "task": v.task,
                       "eid": v.episode_id, "qid": v.query.query_id}, sort_keys=True)
    if fam != "process_count" and target in blob: bad["view_contains_label_string"].append(e.episode_id)
print("episodes checked:", dict(n))
print("oracle disagreements:", {k: Counter(x[0] if isinstance(x, tuple) else "ep" for x in v) for k, v in bad.items()} or "none")
