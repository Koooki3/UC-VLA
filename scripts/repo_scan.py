#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Open-source readiness scan of TRACKED files: secrets (blocking), personal paths/identifiers
and large files (warnings unless --strict). Read-only; never prints a matched secret value."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRETS = {
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "hf_token": r"\bhf_[A-Za-z0-9]{30,}\b",
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",
    "openai_style": r"\bsk-[A-Za-z0-9]{20,}\b",
    "generic_assignment": r"(?i)\b(api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['\"][A-Za-z0-9/+_\-]{16,}['\"]",
}
PERSONAL = {
    "abs_home_path": r"/home/[a-z0-9_.-]+/",
    "abs_users_path": r"(/Users/[A-Za-z0-9_.-]+/|[A-Za-z]:\\Users\\)",
    "wsl_unc": r"\\\\wsl",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "handle": r"(?i)kooki3|zihao|\b14983\b",
}
ALLOWED_EMAIL = re.compile(r"noreply@(anthropic\.com|github\.com)|users\.noreply\.github\.com|@types/")
MAX_BYTES = 2_000_000


def tracked() -> list[Path]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True)
    return [ROOT / p for p in out.stdout.decode().split("\0") if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="treat personal-info/large files as errors")
    ap.add_argument("--allow", nargs="*", default=[], help="path prefixes exempt from personal checks")
    args = ap.parse_args()
    sec, per, big = [], [], []
    for p in tracked():
        rel = p.relative_to(ROOT).as_posix()
        if p.stat().st_size > MAX_BYTES:
            big.append((rel, p.stat().st_size))
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, rx in SECRETS.items():
            if re.search(rx, text):
                sec.append((rel, name))
        if rel.startswith(tuple(args.allow)) or rel == "scripts/repo_scan.py":
            continue
        for name, rx in PERSONAL.items():
            for m in re.finditer(rx, text):
                if name == "email" and ALLOWED_EMAIL.search(m.group(0)):
                    continue
                per.append((rel, name))
    def tally(rows):
        d: dict = {}
        for rel, name in rows:
            d.setdefault(rel, {}).setdefault(name, 0)
            d[rel][name] += 1
        return d
    print(f"secrets: {len(sec)} | personal-info hits in {len(tally(per))} files | large files: {len(big)}")
    for rel, kinds in sorted(tally(per).items()):
        print(f"  [personal] {rel}: {kinds}")
    for rel, size in big:
        print(f"  [large] {rel}: {size} B")
    for rel, name in sec:
        print(f"  [SECRET] {rel}: {name}")
    bad = bool(sec) or (args.strict and (per or big))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
