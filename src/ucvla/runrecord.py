"""Run records: everything needed to attribute and independently reproduce a result."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ucvla.canonical import canonical_json, digest, sha256_hex
from ucvla.dataset_io import source_tree_sha256
from ucvla.runenv import env_info, git_info

RUN_SCHEMA = "ucvla.run/1"


def build_run_record(*, kind: str, root: Path, data_dir: Path, eval_config: Mapping[str, Any],
                     agents: Sequence[Mapping[str, Any]], run_seed: int, results_sha256: str,
                     metrics_sha256: str) -> dict[str, Any]:
    """Deterministic fields feed ``run_id``; git/env/time-like fields are recorded beside it."""
    frozen = json.loads((data_dir / "FROZEN.json").read_text(encoding="utf-8"))
    deterministic = {
        "schema": RUN_SCHEMA, "kind": kind, "mode": "synthetic", "eval_config": dict(eval_config),
        "agents": [dict(a) for a in agents], "run_seed": run_seed,
        "data": {"dataset": frozen["dataset"],
                 "manifest_sha256": sha256_hex((data_dir / "manifest.json").read_bytes()),
                 "split_manifest_id": frozen["split_manifest_id"],
                 "config_sha256": frozen["config_sha256"]},
        "results_sha256": results_sha256, "metrics_sha256": metrics_sha256,
    }
    git = git_info(root)
    record = {
        **deterministic,
        "run_id": "run-" + digest(deterministic)[:16],
        "code_tree_sha256": source_tree_sha256(root),
        **git,
        "env": env_info(),
    }
    if git.get("git_dirty") or git.get("git_commit") is None:
        record["attribution_warning"] = ("working tree differs from the recorded commit (or no "
                                         "commit): result is NOT attributable to a commit")
    return record


def write_run(out_dir: Path, record: Mapping[str, Any], report: Mapping[str, Any]) -> Path:
    path = out_dir / record["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
    (path / "report.json").write_text(canonical_json(report) + "\n", encoding="ascii")
    return path


def append_ledger(ledger: Path, entry: Mapping[str, Any]) -> bool:
    """Append-only evaluation ledger (one line per distinct run_id). Returns True if appended."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if ledger.exists():
        seen = {json.loads(line)["run_id"] for line in ledger.read_text().splitlines() if line}
    if entry["run_id"] in seen:
        return False
    with ledger.open("a", encoding="utf-8") as f:
        f.write(canonical_json(entry) + "\n")
    return True
