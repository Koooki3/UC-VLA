#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate the G1 scaling batch (docs/33) and check the pre-registered predictions."""
import itertools
import json
import statistics as st
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "results/synthetic/g1/scale"
rows = {}
for p in sorted(D.glob("gru*_s*.json")):
    r = json.loads(p.read_text())
    n = 0 if r["arch"] == "gru" else int(r["arch"].split("aug")[1])
    rows.setdefault(n, []).append(r)
rep = {"protocol": "docs/33", "status": "EXPLORATORY", "N": {}}
for n, rs in sorted(rows.items()):
    f = {}
    for fam in ("track", "events", "resume"):
        v = [r["metrics"]["test_id"][fam]["pair_both_correct"] for r in rs]
        f[fam] = {"mean": st.mean(v), "min": min(v), "max": max(v), "values": v}
    f["memoryless_track"] = st.mean(r["memoryless"]["test_id"]["track"]["pair_both_correct"] for r in rs)
    rep["N"][n] = {"seeds": len(rs), **f}
ns = sorted(rep["N"])
tm = [rep["N"][n]["track"]["mean"] for n in ns]
rep["P_S1_monotone_and_N9000_ge_0.75"] = all(a <= b + 1e-9 for a, b in itertools.pairwise(tm)) and rep["N"].get(9000, {}).get("track", {}).get("mean", 0) >= 0.75
rep["P_S2_range_shrinks"] = [rep["N"][n]["track"]["max"] - rep["N"][n]["track"]["min"] for n in ns]
rep["P_S3_memoryless_zero"] = all(rep["N"][n]["memoryless_track"] == 0 for n in ns)
base = rep["N"][0]
rep["P_S3_negative_transfer_gt_0.05"] = {n: {fam: base[fam]["mean"] - rep["N"][n][fam]["mean"] for fam in ("events", "resume")} for n in ns if n}
(D.parent / "g1_scaling_report.json").write_text(json.dumps(rep, indent=1) + "\n")
for n in ns:
    x = rep["N"][n]
    print(n, x["seeds"], {k: [round(v, 2) for v in (x[k]["mean"], x[k]["min"], x[k]["max"])] for k in ("track", "events", "resume")})
print({k: v for k, v in rep.items() if k.startswith("P_")})
