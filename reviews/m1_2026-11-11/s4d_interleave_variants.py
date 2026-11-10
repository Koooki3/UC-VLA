from common import *
from s4_baselines import *
cfg, suite, asg, ad = build()
SP = ("test_id", "test_ood", "test_counterfactual")
def dec_variant(variant):
    base = mk_decoder()
    def d(view, kept, total):
        q = view.query
        if q.task != "resume_sequence": return base(view, kept, total)
        order = q.params["phase_order"].split(","); done = [t.cameras["cam_front"]["placing"] for t in kept if t.cameras["cam_front"]["placing"] != "none"]
        if variant == "assume0_if_unseen" and not done: return f"place_{order[0]}"
        if variant == "no_total_knowledge" and not done: return f"place_{order[0]}"  # same; total/complete flag never consulted
        return base(view, kept, total)
    return d
print("interleave pair-both-correct / episode-acc, pooled over test_id+test_ood+test_cf")
for nm in ("repo(hash, uses complete flag)", "assume0_if_unseen"):
    dec = mk_decoder() if nm.startswith("repo") else dec_variant("assume0_if_unseen")
    for sel in ("last_k", "uniform_k"):
        row = []
        for k in (1, 2, 4, 8, 16):
            pb = []; ac = []
            for s in SP:
                for p in ad.pairs(s):
                    if p.family != "task_interleave": continue
                    oks = []
                    for eid in (p.episode_a, p.episode_b):
                        v = ad.view(eid); kept = select_history(v.history, sel, k); oks.append(ad.score(eid, dec(v, kept, len(v.history))))
                    pb.append(all(oks)); ac += oks
            row.append(f"k={k}: pb={sum(pb)/len(pb):.2f} acc={sum(ac)/len(ac):.2f}")
        print(f"  {nm:32s} {sel:10s} " + " | ".join(row))
# memoryless floor: repo vs prior-aware
print("memoryless episode acc, interleave: repo hash =", end=" ")
for nm, dec in (("hash", mk_decoder()), ("order[0]", dec_variant("assume0_if_unseen")), ("order[1]", mk_decoder(interleave_prior=True))):
    hit = n = 0
    for s in SP:
        for p in ad.pairs(s):
            if p.family != "task_interleave": continue
            for eid in (p.episode_a, p.episode_b):
                v = ad.view(eid); 
                if nm == "order[1]": order = v.query.params["phase_order"].split(","); pred = f"place_{order[1]}"
                else: pred = dec(v, (), len(v.history))
                hit += ad.score(eid, pred); n += 1
    print(f"{nm}={hit/n:.2f}", end="  ")
print()
