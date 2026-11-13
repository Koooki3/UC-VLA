# SPDX-License-Identifier: Apache-2.0
"""Context Observatory: minimal machine-readable trace of context evolution (status: schema +
recorder + replay reader implemented; no UI). Ideas of composition/evolution/replay/cost tracking
are inspired by public context-observability tools; no third-party code is used.

Rules: append-only, hash-chained; memory versions strictly monotone; every evidence reference is a
sha256; predicted ('plan') and executed ('execution') are different event kinds; privileged or
label-like payload keys are rejected, so a trace can never be a leakage channel. A trace records
TIME ORDER only: nothing in it is a causal claim (causal attribution comes from interventions)."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ucvla.canonical import canonical_json, digest, is_sha256_hex

KINDS = ("observation", "memory_op", "plan", "execution", "intervention")
MEMORY_OPS = ("write", "update", "compress", "evict", "retrieve", "verify")
FORBIDDEN = ("target", "label", "privileged", "oracle", "success_actions", "ground_truth")


class TraceError(ValueError):
    pass


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    time_ns: int
    kind: str
    payload: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    prev_hash: str
    hash: str


def _body(seq, time_ns, kind, payload, refs, prev) -> dict[str, Any]:
    return {"seq": seq, "time_ns": time_ns, "kind": kind, "payload": payload,
            "evidence_refs": list(refs), "prev_hash": prev}


def _scan_keys(obj: Any, path: str = "") -> list[str]:
    """Forbidden label/privileged key names anywhere in a nested payload."""
    bad = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if any(f in str(k).lower() for f in FORBIDDEN):
                bad.append(f"{path}/{k}")
            bad += _scan_keys(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _scan_keys(v, f"{path}[{i}]")
    return bad


class TraceRecorder:
    def __init__(self, path: Path, resume: bool = False) -> None:
        self.path = path
        self._seq, self._prev, self._t, self._version = 0, "0" * 64, -1, 0
        self._plans: dict[str, int] = {}
        self._refs: dict[str, int] = {}  # evidence sha256 -> time it became available
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            if not resume:
                raise TraceError("trace exists: refusing to overwrite (use resume=True)")
            for e in replay(path):  # verified, non-destructive restore of the recorder state
                self._seq, self._prev, self._t = e.seq + 1, e.hash, e.time_ns
                if e.kind == "observation":
                    self._refs.setdefault(e.evidence_refs[0], e.time_ns)
                elif e.kind == "memory_op":
                    self._version = e.payload["new_version"]
                elif e.kind == "plan":
                    self._plans[e.payload["plan_id"]] = e.payload["context_version"]
                elif e.kind == "intervention" and e.payload.get("name") == "invalidate_plan":
                    self._plans.pop(e.payload["plan_id"], None)
        else:
            path.write_text("")

    def _emit(self, kind: str, time_ns: int, payload: Mapping[str, Any],
              refs: tuple[str, ...] = ()) -> TraceEvent:
        if kind not in KINDS:
            raise TraceError(f"unknown kind {kind}")
        if time_ns < self._t:
            raise TraceError("time must be non-decreasing")
        bad = _scan_keys(payload)
        if bad:
            raise TraceError(f"payload keys {bad} are forbidden (privileged/label-like)")
        if not all(is_sha256_hex(r) for r in refs):
            raise TraceError("evidence refs must be sha256 hex")
        if kind == "memory_op":
            for r in refs:
                if r not in self._refs:
                    raise TraceError("evidence ref was never observed")
                if self._refs[r] > time_ns:
                    raise TraceError("evidence ref not yet available at this time")
        h = digest(_body(self._seq, time_ns, kind, dict(payload), refs, self._prev))
        ev = TraceEvent(self._seq, time_ns, kind, dict(payload), refs, self._prev, h)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(canonical_json({**_body(self._seq, time_ns, kind, dict(payload), refs, self._prev),
                                    "hash": h}) + "\n")
        self._seq, self._prev, self._t = self._seq + 1, h, time_ns
        return ev

    def observation(self, time_ns: int, obs_sha256: str, tokens: int) -> TraceEvent:
        ev = self._emit("observation", time_ns, {"tokens": tokens}, (obs_sha256,))
        self._refs.setdefault(obs_sha256, time_ns)
        return ev

    def memory_op(self, time_ns: int, op: str, base_version: int, refs: tuple[str, ...],
                  budget_tokens: int, used_tokens: int, cost_ms: float, reason: str = "") -> TraceEvent:
        if op not in MEMORY_OPS:
            raise TraceError(f"unknown memory op {op}")
        if base_version != self._version:
            raise TraceError(f"stale base version {base_version} != {self._version}")
        if used_tokens > budget_tokens and op != "evict":
            raise TraceError("memory over budget")
        new = self._version + (0 if op in ("retrieve", "verify") else 1)
        ev = self._emit("memory_op", time_ns, {"op": op, "base_version": base_version,
                                               "new_version": new, "budget_tokens": budget_tokens,
                                               "used_tokens": used_tokens, "cost_ms": cost_ms,
                                               "reason": reason}, refs)
        self._version = new
        return ev

    def plan(self, time_ns: int, plan_id: str, context_version: int, chunk_len: int) -> TraceEvent:
        if context_version > self._version:
            raise TraceError("plan references a future context version")
        self._plans[plan_id] = context_version
        return self._emit("plan", time_ns, {"plan_id": plan_id, "context_version": context_version,
                                            "chunk_len": chunk_len, "status": "predicted"})

    def invalidate_plan(self, time_ns: int, plan_id: str, reason: str) -> TraceEvent:
        ev = self._emit("intervention", time_ns, {"name": "invalidate_plan", "plan_id": plan_id,
                                                  "reason": reason})
        self._plans.pop(plan_id, None)
        return ev

    def execution(self, time_ns: int, plan_id: str, step: int, verified: bool,
                  allow_stale: bool = False) -> TraceEvent:
        """Contract BEFORE actuation: a plan built on an older context version must be invalidated
        (re-planned) first; executing it raises unless explicitly allowed (diagnostic traces only)."""
        if plan_id not in self._plans:
            raise TraceError("execution of an unknown or invalidated plan")
        if self._plans[plan_id] != self._version and not allow_stale:
            raise TraceError("stale plan: context version changed; invalidate and re-plan first")
        return self._emit("execution", time_ns, {"plan_id": plan_id, "step": step,
                                                 "verified": verified, "status": "executed"})

    def intervention(self, time_ns: int, name: str, params: Mapping[str, Any]) -> TraceEvent:
        return self._emit("intervention", time_ns, {"name": name, **params})


def replay(path: Path) -> Iterator[TraceEvent]:
    """Read and verify the hash chain; raises TraceError on any tampering or reordering."""
    prev, seq, t = "0" * 64, 0, -1
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["seq"] != seq or d["prev_hash"] != prev or d["time_ns"] < t:
            raise TraceError(f"chain broken at seq {d['seq']}")
        body = {k: d[k] for k in ("seq", "time_ns", "kind", "payload", "evidence_refs", "prev_hash")}
        if digest(body) != d["hash"]:
            raise TraceError(f"hash mismatch at seq {seq}")
        yield TraceEvent(d["seq"], d["time_ns"], d["kind"], d["payload"],
                         tuple(d["evidence_refs"]), d["prev_hash"], d["hash"])
        prev, seq, t = d["hash"], seq + 1, d["time_ns"]


def version_timeline(path: Path) -> list[tuple[int, int, str]]:
    return [(e.time_ns, e.payload["new_version"], e.payload["op"])
            for e in replay(path) if e.kind == "memory_op"]


def stale_plans(path: Path) -> list[str]:
    """Plans whose context version was superseded before their last executed step (time order only)."""
    ver, plans, last_exec, bumps = 0, {}, {}, []
    for e in replay(path):
        if e.kind == "memory_op" and e.payload["new_version"] != ver:
            ver = e.payload["new_version"]
            bumps.append((e.time_ns, ver))
        elif e.kind == "plan":
            plans[e.payload["plan_id"]] = e.payload["context_version"]
        elif e.kind == "execution":
            last_exec[e.payload["plan_id"]] = e.time_ns
    return sorted(p for p, v in plans.items()
                  if any(t < last_exec.get(p, -1) and nv > v for t, nv in bumps))
