import pytest

from ucvla.dataset_io import read_dataset, write_dataset
from ucvla.splits import build_split_manifest


@pytest.fixture
def written(tmp_path, suite, cfg, assignment):
    sm = build_split_manifest(cfg, suite.pairs, suite.episodes, assignment)
    write_dataset(tmp_path, suite.episodes, suite.pairs, assignment, sm, {"config_sha256": "x"})
    return tmp_path, sm


def test_roundtrip_preserves_everything(written, suite, assignment):
    d, _ = written
    eps, pairs, asg = read_dataset(d)
    assert eps == list(suite.episodes) and pairs == list(suite.pairs) and asg == assignment


def test_rewrite_identical_is_noop_but_change_is_refused(written, suite, assignment, cfg):
    d, sm = written
    write_dataset(d, suite.episodes, suite.pairs, assignment, sm, {"config_sha256": "x"})
    with pytest.raises(FileExistsError):
        write_dataset(d, suite.episodes[:-2], suite.pairs[:-1], assignment, sm, {"config_sha256": "x"})


def test_tampered_artifact_detected_on_read(written):
    d, _ = written
    path = d / "episodes.jsonl"
    path.write_text(path.read_text().replace("pick_slot_0", "pick_slot_9", 1), encoding="ascii")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_dataset(d)
