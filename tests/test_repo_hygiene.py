# SPDX-License-Identifier: Apache-2.0
"""Repository hygiene that CI enforces (a shebang file must be executable: ruff EXE001)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_shebang_file_is_executable():
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s"], capture_output=True, text=True,
                           check=True).stdout.splitlines()
    bad = []
    for line in files:
        mode, _, _, path = line.split(None, 3)
        p = ROOT / path
        if p.suffix in (".py", ".sh") and p.exists():
            first = p.read_text(encoding="utf-8", errors="ignore").split("\n", 1)[0]
            if first.startswith("#!") and mode != "100755":
                bad.append(path)
    assert not bad, f"shebang without exec bit in git index: {bad}"
