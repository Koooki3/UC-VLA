"""Step 5: flawed variants that PASS g0_report. Transform frozen-config episodes in-process."""
import inspect, textwrap, dataclasses
from common import *
from ucvla.audit import g0_report, obs_only_audit
from ucvla.episode import EpisodeRecord, Frame, ProprioSample, CommandedAction, Event, SourceType, Target, DecisionQuery
from ucvla.canonical import digest
from ucvla import sodh as S

cfg, suite, asg = load()
REL = ("object_correspondence", "process_count", "task_interleave")
e2p = {}
for p in suite.pairs: e2p[p.episode_a] = p; e2p[p.episode_b] = p

def rebuild_frames(e, fn):
    """fn(i_tick, camera, visible)->visible'"""
    ts = e.tick_times(); out = []
    for f in e.frames:
        i = ts.index(f.time_ns); v = fn(i, len(ts), f.camera_id, dict(f.visible))
        out.append(Frame.make(f.frame_id, f.camera_id, f.time_ns, v))
    return dataclasses.replace(e, frames=tuple(out))

def opt_idx(e): return e.queries[0].options.index(lab(e))

def g0(episodes, pairs, name):
    ad = SODHAdapter(episodes, pairs, asg)
    r = g0_report(ad, pairs, episodes, asg)
    codes = Counter(f["code"] + ":" + f["level"] for f in r["findings"])
    return ad, r, f"G0 passed={r['passed']} findings={dict(codes)}"

def evaluate(ad, policy, name):
    """policy(view)->decision. Reports episode acc, pair-both per split group over relevant families."""
    res = {}
    for grp, sp in (("id", ("test_id",)), ("ood", ("test_ood",)), ("cf", ("test_counterfactual",))):
        acc = []; both = []
        for s in sp:
            for p in ad.pairs(s):
                if p.family not in REL: continue
                oks = [ad.score(e, policy(ad.view(e))) for e in (p.episode_a, p.episode_b)]
                acc += oks; both.append(all(oks))
        res[grp] = (sum(acc) / len(acc), sum(both) / len(both))
    print(f"   policy '{name}': " + "  ".join(f"{g}: ep_acc={a:.2f} pair_both={b:.2f}" for g, (a, b) in res.items()))

print("== control: unmodified frozen suite")
ad0, r0, msg = g0(suite.episodes, suite.pairs, "clean"); print("  ", msg)
print("   obs_only (clean):", {f: round(v["obs_only_acc"], 2) for f, v in r0["obs_only"].items()})

variants = {}
# A proprio
variants["A_proprio_at_decision_tick"] = lambda e: dataclasses.replace(
    e, proprio=(ProprioSample(e.queries[0].time_ns, (float(opt_idx(e)),)),))
# B commanded action
variants["B_commanded_action_before_cutoff"] = lambda e: dataclasses.replace(
    e, commanded=(CommandedAction(e.episode_id + "-c0", e.queries[0].time_ns, lab(e)),))
# C online (OBSERVED) event
variants["C_observed_event_at_cutoff"] = lambda e: dataclasses.replace(
    e, events=e.events + (Event.make(e.episode_id + "-n0", e.queries[0].time_ns, "note",
                                     {"v": lab(e)}, SourceType.OBSERVED),))
# D episode id encodes option index (not the label string)
variants["D_episode_id_suffix_encodes_option_index"] = lambda e: dataclasses.replace(e, episode_id=f"{e.episode_id}-i{opt_idx(e)}")
# E history-copy: extra key in front tick history[-1] copies the label index
variants["E_history_copy_hint_in_tick_t-1"] = lambda e: rebuild_frames(
    e, lambda i, n, cam, v: {**v, "aux": str(opt_idx(e))} if (cam == "cam_front" and i == n - 2) else v)
# G final-frame label leak (the planted 'final_obs' mode) but pair records are STALE (not regenerated)
variants["G_final_frame_jitter_stale_pair_records"] = lambda e: rebuild_frames(
    e, lambda i, n, cam, v: {**v, "jitter": digest(lab(e))[:6]} if (cam == "cam_front" and i == n - 1) else v)
# I wrist-camera leak (expect: caught)
variants["I_wrist_camera_gripper_encodes_label"] = lambda e: rebuild_frames(
    e, lambda i, n, cam, v: {"gripper": f"g{opt_idx(e)}"} if (cam == "cam_wrist" and i == n - 1) else v)

def pol_proprio(v): return v.query.options[int(v.proprio[-1].values[0])]
def pol_cmd(v): return v.commanded[-1].action
def pol_event(v): return next(e.payload["v"] for e in v.events if e.kind == "note")
def pol_id(v): return v.query.options[int(v.episode_id.rsplit("-i", 1)[1])]
def pol_hint(v): return v.query.options[int(v.history[-1].cameras["cam_front"]["aux"])]
def pol_jit(v):
    j = v.current.cameras["cam_front"]["jitter"]
    return next(o for o in v.query.options if digest(o)[:6] == j)
def pol_wrist(v): return v.query.options[int(v.current.cameras["cam_wrist"]["gripper"][1:])]
policies = {"A": pol_proprio, "B": pol_cmd, "C": pol_event, "D": pol_id, "E": pol_hint, "G": pol_jit, "I": pol_wrist}

for name, tf in variants.items():
    print(f"\n== {name}")
    eps2 = []; pairs2 = []
    remap = {}
    for e in suite.episodes:
        e2 = tf(e); remap[e.episode_id] = e2.episode_id; eps2.append(e2)
    pairs2 = [dataclasses.replace(p, episode_a=remap[p.episode_a], episode_b=remap[p.episode_b]) for p in suite.pairs]
    ad, r, msg = g0(eps2, pairs2, name); print("  ", msg)
    # ad uses asg keyed by pair_id (unchanged)
    evaluate(ad, policies[name[0]], "leak reader")
    if name[0] == "E":
        # LongWindow(1) = only history[-1]; compare horizon independence
        evaluate(ad, pol_hint, "reads aux of history[-1] only (== LongWindow(1))")
    if name[0] == "G":
        print("   obs_only audit on leaked data:", {f: round(v["obs_only_acc"], 2) for f, v in r["obs_only"].items()})
    if not r["passed"]: print("   caught by:", [f["code"] + ": " + f["message"][:90] for f in r["findings"][:3]])

# F options ordering prior (pair-constant): mate-a's target first. Needs both mates, so do pairwise.
print("\n== F_options_permuted_so_mate_a_target_is_first (identical in both mates)")
eps_by = {e.episode_id: e for e in suite.episodes}
new = {}
for p in suite.pairs:
    if p.family not in ("object_correspondence", "task_interleave"):
        continue
    a, b = eps_by[p.episode_a], eps_by[p.episode_b]
    opts = list(a.queries[0].options); ta = lab(a)
    new_opts = tuple([ta] + [o for o in opts if o != ta])
    for e in (a, b):
        q = dataclasses.replace(e.queries[0], options=new_opts)
        new[e.episode_id] = dataclasses.replace(e, queries=(q,))
eps2 = [new.get(e.episode_id, e) for e in suite.episodes]
ad, r, msg = g0(eps2, suite.pairs, "F"); print("  ", msg)
print("   obs_only:", {f: round(v["obs_only_acc"], 2) for f, v in r["obs_only"].items()})
evaluate(ad, lambda v: v.query.options[0], "always options[0]")
print("   chance (1/n_options): cups ~0.28, interleave ~0.30; theoretical cap for any obs-only policy = 0.50")

# J generator-level label prior (patched generator), full pipeline
print("\n== J_label_prior: cups ball-cup always lands in slot_0 for one mate (patched generator, regenerated + re-split)")
src = inspect.getsource(S._build_cups)
tgt = "x, y = _two_distinct(rng, names[:n])"
assert tgt in src
src = src.replace(tgt, "x = layouts[-1][0]; y = rng.choice([c for c in names[:n] if c != x])")
ns = {}; exec(textwrap.dedent(src), S.__dict__, ns); orig = S._build_cups; S._build_cups = ns["_build_cups"]
cfgJ, suiteJ, asgJ = load()
S._build_cups = orig
adJ = SODHAdapter(suiteJ.episodes, suiteJ.pairs, asgJ)
rJ = g0_report(adJ, suiteJ.pairs, suiteJ.episodes, asgJ)
print("   G0 passed =", rJ["passed"], " findings:", Counter(f["code"] + ":" + f["level"] for f in rJ["findings"]))
for f in rJ["findings"][:6]: print("     ", f["level"], f["code"], f["message"][:100])
for grp, sp in (("id", "test_id"), ("ood", "test_ood"), ("cf", "test_counterfactual")):
    oks = [adJ.score(e, "pick_slot_0") for p in adJ.pairs(sp) if p.family == "object_correspondence" for e in (p.episode_a, p.episode_b)]
    print(f"   'always pick_slot_0' on {sp}: acc={sum(oks)/len(oks):.2f} (n={len(oks)}); chance=1/n_slots ~0.28")
