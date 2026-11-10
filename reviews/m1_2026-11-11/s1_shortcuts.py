"""Step 1: non-history-semantics predictors. Pure python decision tree + prior predictors."""
from common import *
import math, random
cfg, suite, asg = load()
eps = {e.episode_id: e for e in suite.episodes}
pairs = {p.pair_id: p for p in suite.pairs}
ep2pair = {}
for p in suite.pairs:
    ep2pair[p.episode_a] = (p, "a"); ep2pair[p.episode_b] = (p, "b")
file_idx = {e.episode_id: i for i, e in enumerate(suite.episodes)}

def rel_label(e, fam):
    t = lab(e); q = e.queries[0]
    if fam == "object_correspondence": return int(t.split("_")[-1])
    if fam == "task_interleave": return q.params["phase_order"].split(",").index(t[len("place_"):])
    return t

def feats(e):
    """Only metadata + decision-obs *position/structure* features; no history token counting."""
    q = e.queries[0]; p, role = ep2pair[e.episode_id]
    front = [dict(f.visible) for f in e.frames if f.camera_id == "cam_front"]
    cur = front[-1]
    F = {}
    F["n_ticks"] = len(front); F["n_opts"] = len(q.options); F["n_obj"] = p.n_objects
    F["horizon"] = p.horizon; F["regime"] = p.regime; F["mapping"] = p.identity_mapping_id
    F["role"] = role; F["file_parity"] = file_idx[e.episode_id] % 2
    F["eid_last"] = e.episode_id[-1]; F["eid_first"] = e.episode_id[3]
    F["eid_sorted_first"] = int(p.episode_a < p.episode_b) if role == "a" else int(p.episode_a > p.episode_b)
    F["n_param_keys"] = len(q.params); F["n_keys_cur"] = len(cur)
    pass
    F["total_tick_mod2"] = len(front) % 2; F["total_tick_mod3"] = len(front) % 3
    if q.task == "cups_ball":
        names = [v for k, v in cur.items() if k.startswith("slot")]
        F["alpha_rank_slot0"] = sorted(names).index(names[0])
        for k in range(len(names)): F[f"alpha_rank_slot{k}"] = sorted(names).index(names[k])
        first = front[0]; 
        F["reveal_tick_pos"] = next(i for i, f in enumerate(front) if f["ball"].startswith("under:"))  # position of the event, not its content
        F["reveal_rel"] = F["reveal_tick_pos"] / len(front)
        F["first_layout_eq_final"] = int([v for k, v in first.items() if k.startswith("slot")] == names)
    if q.task == "button_count":
        F["K"] = int(q.params["target_count"]); F["last_pose_x"] = 0
        F["total_minus_2K"] = len(front) - 1 - 2 * F["K"]
        F["pose_sum_mod"] = sum(int(f["arm"].split("_")[1]) for f in front[:-1]) % 6
        F["pose_first"] = front[0]["arm"]; F["pose_last_work"] = front[-2]["arm"]
        F["last_work_pressed"] = int(front[-2]["button"] == "pressed")
        F["first_tick_pressed"] = int(front[0]["button"] == "pressed")
    if q.task == "resume_sequence":
        order = q.params["phase_order"].split(","); F["p"] = len(order)
        F["pause"] = len(front) - 1 - len(order)  # = pause ticks (+1 plan + p-1 work + resume)
        F["opt_sorted_rank_of_order0"] = sorted(order).index(order[0])
        F["alpha_first_is_order0"] = int(min(order) == order[0])
        F["alpha_first_is_order1"] = int(min(order) == order[1])
        F["alpha_last_is_orderlast"] = int(max(order) == order[-1])
    return F

fams = ["object_correspondence", "process_count", "task_interleave"]
def gini(ys):
    n = len(ys); c = Counter(ys); return 1 - sum((v / n) ** 2 for v in c.values())
def fit_tree(X, y, depth, min_leaf=6):
    maj = Counter(y).most_common(1)[0][0]
    if depth == 0 or len(set(y)) == 1 or len(y) < 2 * min_leaf: return ("leaf", maj)
    best = None; g0 = gini(y)
    for f in X[0]:
        vals = sorted({x[f] for x in X if f in x}, key=str)
        cands = []
        for v in vals:
            cands.append((f, "eq", v))
            if isinstance(v, (int, float)): cands.append((f, "le", v))
        for (f_, op, v) in cands:
            L = [i for i, x in enumerate(X) if (x.get(f_) == v if op == "eq" else (x.get(f_) is not None and x[f_] <= v))]
            R = [i for i in range(len(X)) if i not in set(L)] if False else None
            Ls = set(L)
            if len(L) < min_leaf or len(X) - len(L) < min_leaf: continue
            yl = [y[i] for i in L]; yr = [y[i] for i in range(len(y)) if i not in Ls]
            g = (len(yl) * gini(yl) + len(yr) * gini(yr)) / len(y)
            if best is None or g < best[0]: best = (g, f_, op, v, Ls)
    if best is None or best[0] >= g0 - 1e-9: return ("leaf", maj)
    _, f_, op, v, Ls = best
    XL = [x for i, x in enumerate(X) if i in Ls]; yL = [y[i] for i in Ls]
    XR = [x for i, x in enumerate(X) if i not in Ls]; yR = [y[i] for i in range(len(y)) if i not in Ls]
    return ("node", f_, op, v, fit_tree(XL, yL, depth - 1, min_leaf), fit_tree(XR, yR, depth - 1, min_leaf))
def predict(t, x):
    while t[0] == "node":
        _, f_, op, v, L, R = t
        ok = (x.get(f_) == v) if op == "eq" else (x.get(f_) is not None and x[f_] <= v)
        t = L if ok else R
    return t[1]

res = {}
for fam in fams:
    data = {s: [] for s in SPLITS}
    for e in suite.episodes:
        p, _ = ep2pair[e.episode_id]
        if p.family != fam: continue
        data[asg[p.pair_id]].append((feats(e), rel_label(e, fam), p))
    Xtr = [d[0] for d in data["train"]]; ytr = [d[1] for d in data["train"]]
    prior = Counter(ytr).most_common(1)[0][0]
    chance = 1 / {"object_correspondence": 3.5, "process_count": 2, "task_interleave": 3.5}[fam]
    print(f"\n=== {fam}: train n={len(ytr)} prior class={prior} ===")
    for depth in (2, 3, 5):
        tr = fit_tree(Xtr, ytr, depth)
        row = []
        for s in SPLITS[1:]:
            ys = [d[1] for d in data[s]]
            acc = sum(predict(tr, d[0]) == d[1] for d in data[s]) / len(ys)
            pri = sum(prior == y for y in ys) / len(ys)
            row.append(f"{s}: tree{depth}={acc:.2f} prior={pri:.2f} n={len(ys)}")
        print("  ", " | ".join(row))
    # pair-both-correct of tree: always 0 if features identical in pair; check how often tree distinguishes mates
    tr = fit_tree(Xtr, ytr, 5)
    ndiff = both = npairs = 0
    seen = defaultdict(list)
    for s in SPLITS[1:]:
        for d in data[s]: seen[d[2].pair_id].append((predict(tr, d[0]), d[1]))
    for pid, v in seen.items():
        npairs += 1; ndiff += v[0][0] != v[1][0]; both += all(a == b for a, b in v)
    print(f"   tree5 pair-both-correct {both}/{npairs}; mates given different predictions {ndiff}/{npairs}")

# Interleave positional prior (relative to the phase_order param, which IS visible at decision time)
print("\n=== task_interleave relative-position prior (no history used) ===")
for split in SPLITS:
    by_p = defaultdict(Counter)
    for e in suite.episodes:
        p, _ = ep2pair[e.episode_id]
        if p.family != "task_interleave" or asg[p.pair_id] != split: continue
        by_p[p.n_objects][rel_label(e, "task_interleave")] += 1
    for pp, c in sorted(by_p.items()):
        tot = sum(c.values()); k, v = c.most_common(1)[0]
        print(f" {split:20s} p={pp}: n={tot} dist={dict(sorted(c.items()))} best-fixed-position acc={v/tot:.2f} chance={1/pp:.2f}")
# analytic prior
for p in (3,4,5):
    n = p - 1
    probs = [0]*p
    for j in range(n): probs[j] += 1/(2*n); probs[j+1] += 1/(2*n)
    print(f" analytic p={p}: P(done=i)={[round(x,3) for x in probs]}  max={max(probs):.3f} vs 1/p={1/p:.3f}")
# same for cups slot prior / count label balance
for fam in ("object_correspondence","process_count"):
    c = Counter(); 
    for e in suite.episodes:
        p,_ = ep2pair[e.episode_id]
        if p.family == fam: c[rel_label(e, fam)] += 1
    print(fam, "label dist", dict(sorted(c.items(), key=str)))
