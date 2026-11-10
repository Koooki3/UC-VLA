"""Step 3: is test_ood only longer? does heldout_mapping test identity?"""
from common import *
import re, random
from dataclasses import replace
from ucvla.sodh import VOCAB
from ucvla.baselines import full_history, memoryless, long_window
from ucvla.episode import CausalView, ViewTick
cfg, suite, asg = load()
eps = {e.episode_id: e for e in suite.episodes}
e2p = {}
for p in suite.pairs: e2p[p.episode_a] = p; e2p[p.episode_b] = p
VS = set(VOCAB)
ad = SODHAdapter(suite.episodes, suite.pairs, asg)

def names_in(e):
    out = set()
    for f in e.frames:
        for v in f.visible.values():
            vv = v[6:] if v.startswith("under:") else v
            if vv in VS: out.add(vv)
    return out
print("== vocabulary / key novelty per family: tokens appearing in ood or cf but not in train")
for fam in ("object_correspondence", "process_count", "task_interleave"):
    byS = defaultdict(set); keys = defaultdict(set); opts = defaultdict(set); phases = defaultdict(set)
    for e in suite.episodes:
        p = e2p[e.episode_id]
        if p.family != fam: continue
        s = asg[p.pair_id]
        byS[s] |= names_in(e)
        for f in e.frames:
            keys[s] |= set(f.visible)
            if "phase" in f.visible: phases[s].add(f.visible["phase"])
        opts[s] |= set(e.queries[0].options)
    tr = byS["train"] | byS["val"] | byS["test_id"]
    print(f" {fam}: new object names in test_ood vs train/val/test_id: {len(byS['test_ood'] - tr)} of {len(byS['test_ood'])}; "
          f"new frame keys: {sorted(keys['test_ood'] - keys['train'])}; new phases {sorted(phases['test_ood']-phases['train'])}; "
          f"new option strings (non-name part): {sorted(o for o in opts['test_ood'] if o.startswith('pick_slot') and o not in opts['train'])}")
    # which noun/mapping position are the new names
    print("     OOD-only names:", sorted(byS['test_ood'] - tr)[:8])

print("\n== structural stats ID vs OOD (what actually changed)")
def stats(fam, regime):
    rows = []
    for p in suite.pairs:
        if p.family != fam or p.regime != regime: continue
        e = eps[p.episode_a]; fr = [f.visible for f in e.frames if f.camera_id == "cam_front"]
        r = {"len": len(fr), "h": p.horizon, "n": p.n_objects}
        if fam == "process_count":
            r["press_density"] = sum(f["button"] == "pressed" for f in fr) / (len(fr) - 1)
            r["K"] = p.horizon
            pr = [f["button"] == "pressed" for f in fr[:-1]]
            r["adjacent_press"] = any(a and b for a, b in zip(pr, pr[1:]))
        if fam == "object_correspondence":
            r["reveal_rel"] = next(i for i, f in enumerate(fr) if f["ball"].startswith("under:")) / (len(fr) - 1)
        if fam == "task_interleave":
            r["pause_frac"] = p.horizon / len(fr)
        rows.append(r)
    return rows
for fam in ("object_correspondence", "process_count", "task_interleave"):
    for reg in ("in_dist", "ood", "heldout_mapping"):
        rows = stats(fam, reg); m = lambda k: sum(r[k] for r in rows) / len(rows)
        extra = {k: round(m(k), 3) for k in rows[0] if k not in ("len", "h", "n")}
        print(f" {fam:22s} {reg:16s} len={m('len'):5.1f} horizon={m('h'):5.1f} n_obj={m('n'):.2f} {extra}")

print("\n== length-extrapolating rule: hand-written FullHistory decoder, zero adaptation, pair-both-correct per split")
for s in ("train", "test_id", "test_ood", "test_counterfactual"):
    r = defaultdict(list)
    for p in ad.pairs(s):
        va, vb = ad.view(p.episode_a), ad.view(p.episode_b)
        r[p.family].append(ad.score(p.episode_a, full_history().decide(va)) and ad.score(p.episode_b, full_history().decide(vb)))
    print(f" {s:20s}", {f: round(sum(v)/len(v), 2) for f, v in r.items()})

print("\n== independent length-agnostic rules (own code, not baselines.py), all splits")
def rule(e):
    q = e.queries[0]; fr = [f.visible for f in e.frames if f.camera_id == "cam_front"]
    if q.task == "cups_ball":
        who = next(v[6:] for f in fr for v in f.values() if v.startswith("under:"))
        for k, v in fr[-1].items():
            if v == who: return "pick_slot_" + k[4:]
    if q.task == "button_count":
        return "press" if sum(f["button"] == "pressed" for f in fr) < int(q.params["target_count"]) else "stop"
    if q.task == "resume_sequence":
        order = q.params["phase_order"].split(","); n_done = sum(f["placing"] != "none" for f in fr)
        return "place_" + order[n_done]
    if q.task == "pick_flagged": return "pick_" + fr[-1]["flag"]
for s in SPLITS:
    r = defaultdict(list)
    for p in ad.pairs(s):
        for eid in (p.episode_a, p.episode_b): r[p.family].append(rule(eps[eid]) == lab(eps[eid]))
    print(f" {s:20s}", {f: round(sum(v)/len(v), 3) for f, v in r.items()})

print("\n== name-blind test: random per-episode bijection of every object name; repo FullHistory decoder")
def rename_view(v, e, rng):
    toks = sorted(VS); perm = toks[:]; rng.shuffle(perm); mp = dict(zip(toks, [f"zz{i}" for i in range(len(perm))])) if False else {t: f"zq{rng.randrange(10**6)}x{i}" for i, t in enumerate(toks)}
    pat = re.compile("|".join(sorted(map(re.escape, toks), key=len, reverse=True)))
    sub = lambda s: pat.sub(lambda m: mp[m.group(0)], s)
    q = v.query
    q2 = replace(q, options=tuple(sub(o) for o in q.options), params={k: sub(x) for k, x in q.params.items()}, instruction=sub(q.instruction))
    ticks = tuple(ViewTick(t.time_ns, {c: {k: sub(x) for k, x in d.items()} for c, d in t.cameras.items()}) for t in v.ticks)
    return replace(v, ticks=ticks, query=q2), sub(lab(e))
rng = random.Random(0); res = defaultdict(list)
for s in SPLITS:
    for p in ad.pairs(s):
        for eid in (p.episode_a, p.episode_b):
            v2, l2 = rename_view(ad.view(eid), eps[eid], rng)
            res[(s, p.family)].append(full_history().decide(v2) == l2)
for s in SPLITS: print(f" {s:20s}", {f: round(sum(v)/len(v), 3) for (ss, f), v in res.items() if ss == s})
print("   (1.0 everywhere => the decoder treats names as opaque symbols; test_counterfactual probes nothing about identity beyond string inequality)")

print("\n== process_count: policy that never reads any object name (strip all names incl. params.button / instruction)")
ok = []
for p in ad.pairs("test_counterfactual"):
    if p.family != "process_count": continue
    for eid in (p.episode_a, p.episode_b):
        e = eps[eid]; fr = [f.visible for f in e.frames if f.camera_id == "cam_front"]
        pred = "press" if sum(f["button"] == "pressed" for f in fr) < int(e.queries[0].params["target_count"]) else "stop"
        ok.append(pred == lab(e))
print("  accuracy on heldout-mapping process_count:", sum(ok)/len(ok), "(n=%d)" % len(ok), "-> 'new object identity' is vacuous for this family")
