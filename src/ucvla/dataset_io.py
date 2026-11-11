"""Canonical JSONL serialisation, hash-verified reading and the data manifest."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ucvla.canonical import canonical_json, digest, sha256_hex
from ucvla.episode import EpisodeRecord
from ucvla.sodh import PairRecord
from ucvla.splits import manifest_bytes, write_once

DATA_FILES = ("episodes.jsonl", "pairs.jsonl", "assignment.json")


def source_tree_sha256(root: Path) -> str:
    """Content hash of the code+config that produced an artifact (the tree is not a git repo)."""
    entries = []
    for pattern in ("src/ucvla/*.py", "scripts/*.py", "configs/sodh/*.yaml"):
        for path in sorted(root.glob(pattern)):
            entries.append((path.relative_to(root).as_posix(), sha256_hex(path.read_bytes())))
    return digest(entries)


def _lines(items: Sequence[Any]) -> bytes:
    return "".join(canonical_json(i) + "\n" for i in items).encode("ascii")


def write_dataset(out_dir: Path, episodes: Sequence[EpisodeRecord], pairs: Sequence[PairRecord],
                  assignment: Mapping[str, str], split_manifest: Mapping[str, Any],
                  extra: Mapping[str, Any]) -> dict[str, Any]:
    """Write the dataset write-once (identical re-runs are no-ops; changes raise)."""
    blobs = {
        "episodes.jsonl": _lines([e.to_dict() for e in episodes]),
        "pairs.jsonl": _lines(list(pairs)),
        "assignment.json": manifest_bytes(dict(sorted(assignment.items()))),
        "split_manifest.json": manifest_bytes(split_manifest),
    }
    for name, data in blobs.items():
        write_once(out_dir / name, data)
    manifest = {
        "mode": "synthetic",
        "schema_version": episodes[0].schema_version if episodes else None,
        "files": {n: sha256_hex(d) for n, d in sorted(blobs.items())},
        "counts": {"episodes": len(episodes), "pairs": len(pairs)},
        "python": sys.version.split()[0],
        **extra,
    }
    write_once(out_dir / "manifest.json", manifest_bytes(manifest))
    return manifest


def read_dataset(in_dir: Path, pair_cls: type[PairRecord] = PairRecord
                 ) -> tuple[list[EpisodeRecord], list[PairRecord], dict[str, str]]:
    """Read and verify: file hashes against manifest.json, and every episode's own hashes."""
    import json

    manifest = json.loads((in_dir / "manifest.json").read_text(encoding="ascii"))
    for name, expected in manifest["files"].items():
        if sha256_hex((in_dir / name).read_bytes()) != expected:
            raise ValueError(f"{name}: hash mismatch (artifact modified)")
    episodes = [EpisodeRecord.from_dict(json.loads(line))
                for line in (in_dir / "episodes.jsonl").read_text(encoding="ascii").splitlines()]
    pairs = [pair_cls(**json.loads(line))
             for line in (in_dir / "pairs.jsonl").read_text(encoding="ascii").splitlines()]
    assignment = json.loads((in_dir / "assignment.json").read_text(encoding="ascii"))
    return episodes, pairs, assignment
