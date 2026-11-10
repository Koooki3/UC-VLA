"""Step 4: baseline fairness. Decompose budget curves; alternative decoders; evidence placement."""
from common import *
import sys
from ucvla.baselines import select_history, _read, _guess, full_history, long_window, uniform_subsample, memoryless
from ucvla import sodh as sodh_mod
from ucvla.canonical import HashRng

BUD = (1, 2, 4, 8, 16)
FAMS = ("object_correspondence", "process_count", "task_interleave")

def build(cfg_over=None, patch=None):
    cfg, suite, asg = load(cfg_over)
    return cfg, suite, asg, SODHAdapter(suite.episodes, suite.pairs, asg)

def run(adapter, decide_fn, selector, k, splits, fams=FAMS):
    """pair-both-correct per (split,family) for decide_fn(view, kept, total)."""
    out = {}
    for s in splits:
        r = defaultdict(list)
        for p in adapter.pairs(s):
            if p.family not in fams: continue
            ok = []
            for eid in (p.episode_a, p.episode_b):
                v = adapter.view(eid)
                kept = select_history(v.history, selector, k) if selector != "all" else v.history
                ok.append(adapter.score(eid, decide_fn(v, kept, len(v.history))))
            r[p.family].append(all(ok))
        out.update({(s, f): sum(x)/len(x) for f, x in r.items()})
    return out

def repo_decide(v, kept, total):
    return _read(v, kept)

# ---- alternative decoders ----
def mk_decoder(count="repo", fallback="hash", interleave_prior=False):
    def dec(view, kept, total):
        q = view.query; FR = "cam_front"; now = view.current.cameras[FR]
        front = [t.cameras[FR] for t in kept]; complete = len(kept) == total
        def fb():
            if fallback == "hash": return _guess(view, kept)
            if fallback == "first": return q.options[0]
            raise ValueError
        if q.task == "cups_ball":
            cups = [f["ball"][6:] for f in front if f["ball"].startswith("under:")]
            slots = {v: k for k, v in now.items() if k.startswith("slot")}
            if cups and cups[0] in slots: return "pick_" + slots[cups[0]].replace("slot", "slot_")
            return fb()
        if q.task == "button_count":
            K = int(q.params["target_count"]); pressed = sum(f["button"] == "pressed" for f in front)
            if count == "repo": 
                if not kept: return "press" if total == 0 else fb()
                return "press" if round(pressed * total / len(kept)) < K else "stop"
            if count == "unscaled":  # no knowledge of how many ticks were dropped
                return "press" if pressed < K else "stop"
            if count == "edge":      # rising edges (held button = 1 press), scaled like repo
                prev = False; e = 0
                for f in front:
                    cur = f["button"] == "pressed"; e += cur and not prev; prev = cur
                if not kept: return "press" if total == 0 else fb()
                return "press" if round(e * total / len(kept)) < K else "stop"
            if count == "edge_full":
                prev = False; e = 0
                for f in front:
                    cur = f["button"] == "pressed"; e += cur and not prev; prev = cur
                return "press" if e < K else "stop"
        if q.task == "resume_sequence":
            order = q.params["phase_order"].split(","); done = [f["placing"] for f in front if f["placing"] != "none"]
            if done:
                i = order.index(done[-1])
                return f"place_{order[i+1]}" if i + 1 < len(order) else fb()
            if complete: return f"place_{order[0]}"
            if interleave_prior: return f"place_{order[1]}"   # no-history positional prior
            return fb()
        if q.task == "pick_flagged": return "pick_" + now["flag"]
    return dec

if __name__ == "__main__":
    cfg, suite, asg, ad = build()
    SP = ("test_id", "test_ood", "test_counterfactual")
    # 4.0 sanity: my reimplementation of repo decoder == repo
    d0 = mk_decoder()
    mism = 0; tot = 0
    for p in suite.pairs:
        for eid in (p.episode_a, p.episode_b):
            v = ad.view(eid)
            for sel, k in (("all", None), ("last_k", 2), ("uniform_k", 3), ("none", None)):
                kept = select_history(v.history, sel, k); tot += 1
                mism += d0(v, kept, len(v.history)) != _read(v, kept)
    print(f"[4.0] reimplemented decoder vs repo _read mismatches: {mism}/{tot}")

    # 4.1 curves are history-length CDF at ID
    print("\n[4.1] fraction of episodes whose full history already fits in the budget (curve point trivially == FullHistory)")
    for s in SP:
        row = {}
        for f in FAMS:
            hl = [len(ad.view(e).history) for p in ad.pairs(s) if p.family == f for e in (p.episode_a,)]
            row[f] = {k: round(sum(h <= k for h in hl) / len(hl), 2) for k in BUD}
        print(" ", s, row)

    # 4.2 repo curves (reproduce), and coverage decomposition
    print("\n[4.2] repo decoder: pair-both-correct (window / uniform) per budget")
    base = {}
    for sel, nm in (("last_k", "window"), ("uniform_k", "uniform")):
        for k in BUD: base[(nm, k)] = run(ad, repo_decide, sel, k, SP)
    for s in SP:
        for f in FAMS:
            print(f"  {s:20s} {f:22s} " + "  ".join(f"k={k}: W={base[('window',k)][(s,f)]:.2f}/U={base[('uniform',k)][(s,f)]:.2f}" for k in BUD))

    print("\n[4.3] coverage decomposition: is differing-tick retained? P(pair_both | covered), P(pair_both | not covered), repo decoder, k in {2,4,8}")
    for fam in FAMS:
        for sel, nm in (("last_k", "window"), ("uniform_k", "uniform")):
            c = defaultdict(list)
            for s in SP:
                for p in ad.pairs(s):
                    if p.family != fam: continue
                    va, vb = ad.view(p.episode_a), ad.view(p.episode_b)
                    for k in (2, 4, 8):
                        n = len(va.history)
                        idx = list(range(n)); kept_idx = select_history(tuple(idx), sel, k)
                        diff = [i for i in range(n) if va.history[i] != vb.history[i]]
                        covered = all(i in set(kept_idx) for i in diff)
                        ka = select_history(va.history, sel, k); kb = select_history(vb.history, sel, k)
                        ok = ad.score(p.episode_a, _read(va, ka)) and ad.score(p.episode_b, _read(vb, kb))
                        c[(k, covered)].append(ok)
            print(f"  {fam:22s} {nm:8s} " + "  ".join(f"k={k} cov: n={len(c[(k,True)])} ok={sum(c[(k,True)])/max(1,len(c[(k,True)])):.2f} | uncov: n={len(c[(k,False)])} ok={sum(c[(k,False)])/max(1,len(c[(k,False)])):.2f}" for k in (2,4,8)))

    # 4.4 decoder variants for process_count and interleave
    print("\n[4.4] process_count: decoder variants. pair-both (window/uniform); test_id+test_ood+cf pooled")
    for nm, dec in (("repo(scale-up)", mk_decoder("repo")), ("unscaled", mk_decoder("unscaled")),
                    ("edge+scale", mk_decoder("edge")), ("edge_full(all history)", mk_decoder("edge_full"))):
        row = []
        for k in BUD:
            w = run(ad, dec, "last_k", k, SP, ("process_count",)); u = run(ad, dec, "uniform_k", k, SP, ("process_count",))
            row.append(f"k={k}: W={sum(w.values())/3:.2f} U={sum(u.values())/3:.2f}")
        full = run(ad, dec, "all", None, SP, ("process_count",))
        print(f"  {nm:24s} " + " | ".join(row) + f" || FULL id/ood/cf = {[round(full[(s,'process_count')],2) for s in SP]}")
    print("\n[4.5] interleave: fallback variants (hash vs positional-prior) -> episode accuracy, k=2,4")
    for nm, dec in (("hash", mk_decoder()), ("prior-order[1]", mk_decoder(interleave_prior=True))):
        for sel in ("last_k", "uniform_k"):
            accs = []
            for k in (1, 2, 4, 8):
                hit = n = 0
                for s in SP:
                    for p in ad.pairs(s):
                        if p.family != "task_interleave": continue
                        for eid in (p.episode_a, p.episode_b):
                            v = ad.view(eid); kept = select_history(v.history, sel, k)
                            hit += ad.score(eid, dec(v, kept, len(v.history))); n += 1
                accs.append(f"k={k}:{hit/n:.2f}")
            print(f"  {nm:15s} {sel:10s} episode_acc " + " ".join(accs))

    # 4.6 rising-edge identifiability of process_count
    print("\n[4.6] process_count: under a rising-edge reading of 'pressed', are mates still distinguishable?")
    for reg in ("in_dist", "ood", "heldout_mapping"):
        n = same = 0
        for p in suite.pairs:
            if p.family != "process_count" or p.regime != reg: continue
            es = []
            for eid in (p.episode_a, p.episode_b):
                fr = [t.cameras["cam_front"] for t in ad.view(eid).history]
                prev = False; e = 0
                for f in fr:
                    cur = f["button"] == "pressed"; e += cur and not prev; prev = cur
                es.append(e)
            n += 1; same += es[0] == es[1]
        print(f"  {reg:16s} pairs with equal rising-edge count (history indistinguishable under edge semantics): {same}/{n} = {same/n:.2f}")
