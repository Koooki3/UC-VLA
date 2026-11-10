from scripts.validate_protocol import validate


def test_missing_manifest_fields_fail_closed():
    assert any("missing" in e for e in validate({}))


def test_unsafe_modes_fail_closed():
    manifest = {k: "known" for k in (
        "run_id", "experiment_id", "mode", "git_commit", "policy_checkpoint_sha256",
        "dataset_manifest_sha256", "train_seed", "test_seed_manifest_sha256",
        "evaluation_protocol_revision", "hardware", "code_dependencies_sha256",
    )}
    manifest.update(mode="real_robot", online_privileged_state=True, test_tuning=True)
    errors = validate(manifest)
    assert len(errors) == 3
