import dataclasses
from common import *
from ucvla.audit import g0_report, check_pairs, structural_pair_diffs, obs_only_audit
import ucvla.audit as A
from ucvla.episode import Frame
from ucvla.canonical import digest
from ucvla import baselines
cfg, suite, asg = load()
eps_by = {e.episode_id: e for e in suite.episodes}
# F: print what caught it
new = {}
for p in suite.pairs:
    if p.family not in ("object_correspondence", "task_interleave"): continue
    a, b = eps_by[p.episode_a], eps_by[p.episode_b]
    ta = lab(a); opts = list(a.queries[0].options)
    no = tuple([ta] + [o for o in opts if o != ta])
    for e in (a, b): new[e.episode_id] = dataclasses.replace(e, queries=(dataclasses.replace(e.queries[0], options=no),))
eps2 = [new.get(e.episode_id, e) for e in suite.episodes]
ad = SODHAdapter(eps2, suite.pairs, asg); r = g0_report(ad, suite.pairs, eps2, asg)
print("F findings:", [(f["code"], f["message"]) for f in r["findings"]])
# F2: same leak but only for cups (n options 3-5), interleave untouched -> does the lookup catch it?
new2 = {k: v for k, v in new.items() if e2 if False} if False else {}
for p in suite.pairs:
    if p.family != "object_correspondence": continue
    a, b = eps_by[p.episode_a], eps_by[p.episode_b]; ta = lab(a); opts = list(a.queries[0].options)
    no = tuple([ta] + [o for o in opts if o != ta])
    for e in (a, b): new2[e.episode_id] = dataclasses.replace(e, queries=(dataclasses.replace(e.queries[0], options=no),))
eps3 = [new2.get(e.episode_id, e) for e in suite.episodes]
ad = SODHAdapter(eps3, suite.pairs, asg); r = g0_report(ad, suite.pairs, eps3, asg)
print("F2 (cups only) passed:", r["passed"], [(f["code"], f["message"][:100]) for f in r["findings"]])

# G: final-frame leak, stale pair records: which individual checks see it?
def rebuild(e):
    ts = e.tick_times(); out = []
    for f in e.frames:
        v = dict(f.visible)
        if f.camera_id == "cam_front" and f.time_ns == ts[-1]: v["jitter"] = digest(lab(e))[:6]
        out.append(Frame.make(f.frame_id, f.camera_id, f.time_ns, v))
    return dataclasses.replace(e, frames=tuple(out))
epsG = [rebuild(e) for e in suite.episodes]
adG = SODHAdapter(epsG, suite.pairs, asg)
print("G: check_pairs findings:", len(check_pairs(adG, suite.pairs)), "| structural diffs:", set(structural_pair_diffs(adG, suite.pairs).values()))
print("G: recomputed decision-obs digests equal within pair?", sum(adG.view(p.episode_a).digest(False) == adG.view(p.episode_b).digest(False) for p in suite.pairs), "/", len(suite.pairs), "(stored digests claim all equal)")
print("G: full g0 passed:", g0_report(adG, suite.pairs, epsG, asg)["passed"])
# G with the memoryless reader's hash-guess replaced by a constant (hypothetical: any non-hash fallback)
orig = A.memoryless
class Const:
    name = "const"
    def decide(self, v): return v.query.options[0]
    def retained(self, v): return 0
A.memoryless = lambda: Const()
rr = g0_report(adG, suite.pairs, epsG, asg)
A.memoryless = orig
print("G with a constant-guess memoryless reader: passed =", rr["passed"], [f["code"] for f in rr["findings"]][:5])
