import json
d = json.load(open("$UCVLA_ROOT/results/synthetic/sodh_v1/symbolic_baselines.json"))
rows = d["paired_uniform_vs_window"]
from collections import Counter
c = Counter(); tot = Counter()
for r in rows:
    tot[r["split"]] += 1
    if r["ci95"][0] > 0 or r["ci95"][1] < 0: c[r["split"]] += 1
print("paired uniform-minus-window comparisons whose 95% cluster-bootstrap CI excludes 0:", dict(c), "of", dict(tot))
print("sign pattern per family/split (uniform-window by k=1,2,4,8,16):")
for s in ("test_id", "test_ood", "test_counterfactual"):
    for f in ("object_correspondence", "process_count", "task_interleave"):
        print(" ", s, f, [round(r["uniform_minus_window"], 2) for r in rows if r["split"] == s and r["family"] == f])
n = Counter()
for r in d["rows"]:
    n[(r["split"], r["family"])] = r["n_pairs"]
print("pairs per cell:", {k: v for k, v in n.items() if k[0] == "test_id"})
