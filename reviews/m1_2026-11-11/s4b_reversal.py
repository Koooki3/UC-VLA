"""Step 4b: evidence-placement sensitivity of LongWindow-vs-Uniform (patched generators, in-process only)."""
import inspect, textwrap
from common import *
from s4_baselines import *
from ucvla import sodh as S

src_c = inspect.getsource(S._build_cups)
assert "reveal_tick = rng.randint(0, max(0, horizon // 2))" in src_c
late = src_c.replace("reveal_tick = rng.randint(0, max(0, horizon // 2))",
                     "reveal_tick = horizon - rng.randint(0, max(0, horizon // 2))")
src_k = inspect.getsource(S._build_count)
assert "idx = list(range(total))\n    rng.shuffle(idx)" in src_k
late_k = src_k.replace("idx = list(range(total))\n    rng.shuffle(idx)",
                       "idx = list(range(total - (3 * total) // 5, total))\n    rng.shuffle(idx)")  # presses only in last 60% of ticks
orig_c, orig_k = S._build_cups, S._build_count

def curves(label, decs, fams, splits=("test_id", "test_ood", "test_counterfactual")):
    cfg, suite, asg, ad = build()
    print(f"\n--- {label}")
    for nm, dec in decs:
        for fam in fams:
            row = []
            for k in (2, 4, 8, 16):
                w = run(ad, dec, "last_k", k, splits, (fam,)); u = run(ad, dec, "uniform_k", k, splits, (fam,))
                n = sum(len(ad.pairs(s)) for s in splits) / 4  # rough per-family n
                row.append(f"k={k}: W={sum(w.values())/len(w):.2f} U={sum(u.values())/len(u):.2f}")
            print(f"  [{nm}] {fam:22s} " + " | ".join(row))

curves("baseline frozen generator (reveal in first half)", [("repo", repo_decide)], ["object_correspondence"])
ns = {}; exec(textwrap.dedent(late), S.__dict__, ns); S._build_cups = ns["_build_cups"]
curves("PATCHED: reveal tick in LAST half of the shuffle (everything else identical)", [("repo", repo_decide)], ["object_correspondence"])
S._build_cups = orig_c
curves("count, frozen generator (presses uniform over ticks)", [("repo scale-up", mk_decoder("repo")), ("unscaled", mk_decoder("unscaled"))], ["process_count"])
ns = {}; exec(textwrap.dedent(late_k), S.__dict__, ns); S._build_count = ns["_build_count"]
curves("count, PATCHED: presses only in last 60% of ticks (K,total unchanged)", [("repo scale-up", mk_decoder("repo")), ("unscaled", mk_decoder("unscaled"))], ["process_count"])
S._build_count = orig_k

print("\n--- budget unit: tokens per retained tick (front+wrist) per family")
cfg, suite, asg = load()
ad = SODHAdapter(suite.episodes, suite.pairs, asg)
for fam in FAMS + ("irrelevant_history",):
    t = []
    for p in suite.pairs:
        if p.family == fam:
            e = ad.episode(p.episode_a); t.append(sum(len(f.visible) for f in e.frames) / len(e.tick_times()))
    print(f"  {fam:22s} mean tokens per tick (both cams) = {sum(t)/len(t):.2f}  -> 'k ticks' budget costs {sum(t)/len(t):.1f}*k tokens; wrist camera share {1/(sum(t)/len(t)):.0%} of tokens, constant 'gripper:open'")
