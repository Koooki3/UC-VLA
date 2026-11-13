# SPDX-License-Identifier: Apache-2.0
"""Read-only capture of code version and runtime environment for run records."""

from __future__ import annotations

import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = ("pyyaml", "pytest", "hypothesis")


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                             timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(root: Path) -> dict[str, Any]:
    """Real commit SHA + whether tracked/untracked files differ from it. Never mutates the repo."""
    sha = _git(root, "rev-parse", "HEAD")
    if sha is None:
        return {"git_commit": None, "git_dirty": None, "git_branch": None,
                "git_note": "not a git repository or git unavailable"}
    status = _git(root, "status", "--porcelain", "--untracked-files=normal")
    return {"git_commit": sha, "git_dirty": bool(status), "git_branch": _git(
        root, "rev-parse", "--abbrev-ref", "HEAD")}


def env_info() -> dict[str, Any]:
    pkgs = {}
    for name in TRACKED_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": pkgs,
            "gpu": None, "robot": None}
