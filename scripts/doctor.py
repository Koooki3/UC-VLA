#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Non-invasive environment doctor. Never requests GPU or connects to robot."""

import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {
        "project_root": str(root),
        "python": platform.python_version(),
        "supported_python": (3, 11) <= sys.version_info[:2] < (3, 14),
        "pytest_available": importlib.util.find_spec("pytest") is not None,
        "git_available": shutil.which("git") is not None,
        "sbatch_available": shutil.which("sbatch") is not None,
        "gpu_allocated": False,
        "robot_connected": False,
        "cluster_configured": bool(os.environ.get("UCVLA_CLUSTER_PARTITION")),
        "data_root_configured": bool(os.environ.get("UCVLA_DATA_ROOT")),
        "readme_exists": (root / "README.md").exists(),
        "warning": "Availability check only; no GPU/robot/cluster action performed.",
    }
    print(json.dumps(report, indent=2))
    if not report["supported_python"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
