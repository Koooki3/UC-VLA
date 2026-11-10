#!/usr/bin/env python3
"""Fail closed on obvious missing prerequisites before a real experiment.

We deliberately do not load YAML to keep Day-0 free of third-party dependencies.
This checks the run manifest JSON produced by agent implementation.
"""

import argparse
import json
from pathlib import Path

REQUIRED = {
    "run_id",
    "experiment_id",
    "mode",
    "git_commit",
    "policy_checkpoint_sha256",
    "dataset_manifest_sha256",
    "train_seed",
    "test_seed_manifest_sha256",
    "evaluation_protocol_revision",
    "hardware",
    "code_dependencies_sha256",
}


def validate(manifest: dict) -> list[str]:
    missing = sorted(k for k in REQUIRED if not manifest.get(k))
    errors = [f"missing: {key}" for key in missing]
    if manifest.get("mode") == "real_robot" and not manifest.get("operator_authorization_ref"):
        errors.append("real_robot requires operator_authorization_ref")
    if manifest.get("online_privileged_state"):
        errors.append("online privileged simulator state forbidden")
    if manifest.get("test_tuning"):
        errors.append("test_tuning forbidden")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate(json.loads(args.manifest.read_text(encoding="utf-8")))
    print("VALID" if not errors else "INVALID: " + "; ".join(errors))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
