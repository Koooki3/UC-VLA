#!/usr/bin/env python3
"""(Re)generate or verify PACKAGE_MANIFEST.json: sha256 of every tracked project file."""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".hypothesis", ".git"}
SKIP_FILES = {"PACKAGE_MANIFEST.json"}
SKIP_PREFIX = ("results/synthetic/",)  # generated data has its own hash-verified manifest


def collect() -> list[dict]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if (not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts)
                or rel in SKIP_FILES or rel.startswith(SKIP_PREFIX) or path.suffix == ".pyc"):
            continue
        data = path.read_bytes()
        files.append({"path": rel, "bytes": len(data), "sha256": sha256(data).hexdigest()})
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    target = ROOT / "PACKAGE_MANIFEST.json"
    old = json.loads(target.read_text(encoding="utf-8"))
    manifest = {"created": old["created"], "status": "research_scaffold_m1_cpu_only",
                "files": collect()}
    text = json.dumps(manifest, indent=2) + "\n"
    if args.check:
        if json.loads(text)["files"] != old["files"]:
            sys.exit("PACKAGE_MANIFEST.json is stale: run scripts/package_manifest.py")
        print("PACKAGE_MANIFEST.json current")
    else:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target} ({len(manifest['files'])} files)")


if __name__ == "__main__":
    main()
