import dataclasses, time
from common import *
import ucvla.audit as A
from ucvla.audit import g0_report
from ucvla.episode import Frame
from ucvla.canonical import digest
cfg, suite, asg = load()
class Const:
    name = "const"
    def decide(self, v): return v.query.options[0]
    def retained(self, v): return 0
def make(salt):
    out = []
    for e in suite.episodes:
        ts = e.tick_times(); fr = []
        for f in e.frames:
            v = dict(f.visible)
            if f.camera_id == "cam_front" and f.time_ns == ts[-1]: v["jitter"] = digest([salt, lab(e)])[:6]
            fr.append(Frame.make(f.frame_id, f.camera_id, f.time_ns, v))
        out.append(dataclasses.replace(e, frames=tuple(fr)))
    return out
orig = A.memoryless; t = time.time()
res = {"repo(hash-guess)": [], "const-guess": []}; obs = []
for salt in range(20):
    eps = make(salt); ad = SODHAdapter(eps, suite.pairs, asg)
    r = g0_report(ad, suite.pairs, eps, asg); res["repo(hash-guess)"].append(r["passed"]); obs.append(round(r["obs_only"]["process_count"]["obs_only_acc"], 2))
    A.memoryless = lambda: Const()
    r2 = g0_report(ad, suite.pairs, eps, asg); A.memoryless = orig; res["const-guess"].append(r2["passed"])
for k, v in res.items(): print(f"stale-pair-record final-frame leak, G0 pass rate [{k}]: {sum(v)}/{len(v)}")
print("process_count obs_only_acc across salts:", obs, f"({time.time()-t:.0f}s)")
