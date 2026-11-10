from common import *
cfg, suite, asg = load()
eps = {e.episode_id: e for e in suite.episodes}
c = Counter((p.family, asg[p.pair_id]) for p in suite.pairs)
for k in sorted(c): print(k, c[k])
print("rejected dup scripts", suite.rejected_duplicate_scripts)
for fam in ("object_correspondence","process_count","task_interleave","irrelevant_history"):
    p = next(p for p in suite.pairs if p.family==fam and p.regime=="in_dist")
    e = eps[p.episode_a]
    print("\n==", fam, p.horizon, p.n_objects, "targets", p.target_a, p.target_b)
    for f in e.frames:
        if f.camera_id=="cam_front": print(f.time_ns//10**8, dict(f.visible))
    print(e.queries[0]); print(e.events)
