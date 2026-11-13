#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only symbolic diagnostic; intentionally not a learned VLA experiment."""

import json

from ucvla.toy import run_memoryless, run_oracle_memory


def main() -> None:
    episodes = [(f"toy-{i}", "left" if i % 2 == 0 else "right") for i in range(10)]
    no_memory = [run_memoryless(e, x) for e, x in episodes]
    oracle_memory = [run_oracle_memory(e, x) for e, x in episodes]
    report = {
        "mode": "synthetic_symbolic_contract_test",
        "is_real_benchmark": False,
        "is_trained_model": False,
        "episodes": len(episodes),
        "memoryless_success": sum(r.success for r in no_memory) / len(no_memory),
        "oracle_memory_success": sum(r.success for r in oracle_memory) / len(oracle_memory),
        "warning": "Sanity check only; not a scientific or policy performance measurement.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
