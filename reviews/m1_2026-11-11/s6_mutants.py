"""Step 6: mutation test of the M1 test-suite + G0 (copies live in the scratchpad; repo untouched)."""
import shutil, subprocess, sys, os
from pathlib import Path
R = Path("/tmp/claude-1000/-home-kooki3-UC-VLA/9124c0fa-b067-4e71-a68e-aa4a5bd30524/scratchpad/review")
BASE = R / "mut_base"
PY = "/home/kooki3/UC-VLA/.venv/bin/python"
TESTS = ["tests/test_audit.py", "tests/test_sodh.py", "tests/test_splits.py", "tests/test_episode.py",
         "tests/test_dataset_io.py", "tests/test_alignment.py"]
MUT = {
 "M1_label_prior(ball cup always lands in slot_0 for one mate)": ("sodh.py",
   "x, y = _two_distinct(rng, names[:n])", "x = layouts[-1][0]; y = rng.choice([c for c in names[:n] if c != x])"),
 "M2_reveal_in_last_half": ("sodh.py",
   "reveal_tick = rng.randint(0, max(0, horizon // 2))", "reveal_tick = horizon - rng.randint(0, max(0, horizon // 2))"),
 "M3_cups_never_swap(static layout)": ("sodh.py",
   "cur[i], cur[j] = cur[j], cur[i]", "pass"),
 "M4_proprio_leak": ("sodh.py",
   "proprio=(), commanded=()",
   "proprio=(__import__('ucvla.episode', fromlist=['x']).ProprioSample(q_time, (float(spec_mate.options.index(spec_mate.target)),)),), commanded=()"),
 "M5_hint_copy_in_history_tick(t-1)": ("sodh.py",
   "    annotations = list(spec_mate.annotations)\n",
   "    annotations = list(spec_mate.annotations)\n    ticks[-2]['aux'] = str(spec_mate.options.index(spec_mate.target))\n"),
 "M6_interleave_always_j0(label position prior)": ("sodh.py",
   "j = rng.below(p - 1)  # mates have completed j and j+1 phases", "j = 0"),
 "M7_episode_id_encodes_option_index": ("sodh.py",
   "eps.append(_assemble(spec.mates[which], eid, prov, config.leak))",
   "eps.append(_assemble(spec.mates[which], eid + '-i' + str(spec.mates[which].options.index(spec.mates[which].target)), prov, config.leak))"),
 "M8_commanded_action_leak": ("sodh.py",
   "executed=(), events=tuple(events)",
   "executed=(), events=tuple(events)") ,  # placeholder replaced below
}
# M8 needs commanded; implement via replacing the commanded=() text separately
MUT["M8_commanded_action_leak"] = ("sodh.py", "proprio=(), commanded=()",
   "proprio=(), commanded=(__import__('ucvla.episode', fromlist=['x']).CommandedAction(episode_id + '-c0', q_time, spec_mate.target),)")
out = []
def run(name, patch):
    d = R / "mut_run"
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(BASE, d)
    if patch:
        f, old, new = patch
        p = d / "src/ucvla" / f; s = p.read_text()
        assert s.count(old) == 1, (name, s.count(old))
        p.write_text(s.replace(old, new))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=f"{d}/src:{d}")
    r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], cwd=d, env=env,
                       capture_output=True, text=True, timeout=250)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
    failed = [l.split("::")[-1].split(" ")[0] for l in r.stdout.splitlines() if l.startswith("FAILED")]
    print(f"{name:62s} -> {'SURVIVES (all tests pass)' if r.returncode == 0 else 'killed by ' + ', '.join(failed[:4])}   [{last}]", flush=True)
run("M0_unmutated (control)", None)
for k, v in MUT.items(): run(k, v)
shutil.rmtree(R / "mut_run", ignore_errors=True)
