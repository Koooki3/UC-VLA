"""Step 2: information overlap across splits beyond ids/exact content."""
from common import *
from ucvla.sodh import VOCAB, GROUP_SIZE
from ucvla.canonical import digest
cfg, suite, asg = load()
eps = {e.episode_id: e for e in suite.episodes}
P = {p.pair_id: p for p in suite.pairs}
e2p = {}
for p in suite.pairs: e2p[p.episode_a] = p; e2p[p.episode_b] = p
VS = set(VOCAB)

def canon(e, drop_irrelevant):
    """alpha-rename object names by first appearance; optionally drop arm pose/noise tokens."""
    mp = {}
    def r(v):
        if v in VS: return mp.setdefault(v, f"N{len(mp)}")
        if v.startswith("under:") and v[6:] in VS: return "under:" + mp.setdefault(v[6:], f"N{len(mp)}")
        return v
    q = e.queries[0]
    # order params first so names appear in a deterministic order
    par = {}
    for k in sorted(q.params):
        v = q.params[k]
        par[k] = ",".join(r(x) for x in v.split(",")) if k == "phase_order" else (r(v) if v in VS else v)
    ticks = []
    for f in e.frames:
        if f.camera_id != "cam_front": continue
        d = {k: r(v) for k, v in f.visible.items()}
        if drop_irrelevant: d = {k: v for k, v in d.items() if k not in ("arm", "noise")}
        ticks.append(d)
    # relative label
    t = lab(e)
    if q.task == "cups_ball": cl = t
    elif q.task == "resume_sequence": cl = par["phase_order"].split(",").index(mp[t[6:]])
    else: cl = t
    instr = q.instruction if q.task != "button_count" else "press exactly %s" % q.params["target_count"]
    return digest({"t": ticks, "par": {k: v for k, v in par.items() if k != "button"}, "i": instr}), cl

for fam in ("object_correspondence", "process_count", "task_interleave", "irrelevant_history"):
    print(f"\n=== {fam} ===")
    for level, drop in (("L1 alpha-renamed full content", False), ("L2 alpha-renamed, arm/noise dropped", True)):
        tab = defaultdict(lambda: defaultdict(set))  # key -> split -> labels
        for e in suite.episodes:
            p = e2p[e.episode_id]
            if p.family != fam: continue
            k, cl = canon(e, drop)
            tab[k][asg[p.pair_id]].add(cl)
        # fraction of eval episodes whose canonical twin exists in train, and label agrees
        for s in SPLITS[1:]:
            n = twin = agree = 0
            for e in suite.episodes:
                p = e2p[e.episode_id]
                if p.family != fam or asg[p.pair_id] != s: continue
                k, cl = canon(e, drop); n += 1
                if "train" in tab[k]:
                    twin += 1; agree += cl in tab[k]["train"]
            print(f"  {level:38s} {s:20s} n={n:3d} train-twin={twin/n:5.2f} twin-with-same-label={agree/n:5.2f}")
    # exact decision-observation (L0) repeated across splits with the same label
    obs = defaultdict(lambda: defaultdict(set))
    for e in suite.episodes:
        p = e2p[e.episode_id]
        if p.family != fam: continue
        v = e.causal_view(e.queries[0].time_ns)
        obs[v.digest(False)][asg[p.pair_id]].add(lab(e))
    cross = sum(1 for k, d in obs.items() if len(d) > 1)
    print(f"  exact decision-obs digests shared by >1 split: {cross}/{len(obs)}")

# abstract swap script (cups): permutation sequence relative to tick-0 layout + reveal tick + n
print("\n=== cups abstract scene script (layout names removed) overlap across splits ===")
def cups_abs(e):
    fr = [f.visible for f in e.frames if f.camera_id == "cam_front"]
    n = sum(1 for k in fr[0] if k.startswith("slot"))
    base = [fr[0][f"slot{k}"] for k in range(n)]
    seq = []
    for f in fr[:-1]:
        seq.append(tuple(base.index(f[f"slot{k}"]) for k in range(n)))
    rev = next(i for i, f in enumerate(fr) if f["ball"].startswith("under:"))
    return digest([n, seq, rev])
tab = defaultdict(set)
for e in suite.episodes:
    p = e2p[e.episode_id]
    if p.family == "object_correspondence": tab[cups_abs(e)].add(asg[p.pair_id])
multi = sum(len(v) > 1 for v in tab.values())
print("  abstract scripts:", len(tab), "shared across >1 split:", multi)
# horizon-only coarse: (n, horizon) cell overlap
cells = defaultdict(set)
for p in suite.pairs:
    if p.family == "object_correspondence": cells[(p.n_objects, p.horizon)].add(asg[p.pair_id])
print("  (n,horizon) cells and splits:", {k: sorted(v) for k, v in sorted(cells.items())})

# interleave: abstract scene space size vs sample count
print("\n=== interleave abstract-space size ===")
import math
for p_ in (3, 4, 5):
    print(f"  p={p_}: abstract scenes (j,pause) per pair = {(p_-1)} x pause options; name-orderings are renaming-equivalent")
ab = defaultdict(set)
for pr in suite.pairs:
    if pr.family == "task_interleave":
        ab[(pr.n_objects, pr.horizon, pr.regime)].add(asg[pr.pair_id])
print("  (p,pause,regime) -> splits:", {k: sorted(v) for k, v in sorted(ab.items())})

# vocabulary overlap
print("\n=== vocabulary overlap: heldout mapping vs seen mappings ===")
seen_names = set(); held_names = set()
for p in suite.pairs:
    for e in (eps[p.episode_a],):
        for f in e.frames:
            for v in f.visible.values():
                vv = v[6:] if v.startswith("under:") else v
                if vv in VS: (held_names if p.regime == "heldout_mapping" else seen_names).add(vv)
adj = lambda s: {x.split("_")[0] for x in s}; noun = lambda s: {x.split("_")[1] for x in s}
print("  heldout names", sorted(held_names)); print("  full-name overlap:", held_names & seen_names)
print("  heldout adjectives", sorted(adj(held_names)), "seen-adj overlap", sorted(adj(held_names) & adj(seen_names)))
print("  heldout nouns", sorted(noun(held_names)), "seen-noun overlap", sorted(noun(held_names) & noun(seen_names)))
print("  fraction of heldout name atoms (adj+noun) seen in train:", len((adj(held_names)|noun(held_names)) & (adj(seen_names)|noun(seen_names))), "/", len(adj(held_names)|noun(held_names)))
# split vs mapping coverage
cov = defaultdict(Counter)
for p in suite.pairs: cov[asg[p.pair_id]][p.identity_mapping_id] += 1
for s in SPLITS: print("  ", s, dict(sorted(cov[s].items())))
