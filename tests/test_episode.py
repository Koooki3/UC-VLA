import copy

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ucvla.episode import (
    SCHEMA_VERSION,
    CommandedAction,
    EpisodeRecord,
    ExecutedAction,
    Frame,
    SourceType,
)


def episode_dict(ep):
    return copy.deepcopy(ep.to_dict())


def test_roundtrip_is_lossless(suite):
    for ep in suite.episodes[:40]:
        again = EpisodeRecord.from_dict(ep.to_dict())
        assert again == ep
        assert again.episode_sha256() == ep.episode_sha256()


def test_causal_view_never_contains_future_or_labels(suite):
    """A1: at every tick of every episode, nothing newer than the cutoff, no labels/privileged."""
    for ep in suite.episodes:
        for view in ep.replay():
            assert all(t.time_ns <= view.cutoff_ns for t in view.ticks)
            assert all(e.window_end_ns <= view.cutoff_ns for e in view.events)
            assert all(e.source_type in (SourceType.OBSERVED, SourceType.MEASURED,
                                         SourceType.COMMANDED) for e in view.events)
            assert not hasattr(view, "targets") and not hasattr(view, "privileged")
            if view.query:
                assert view.query.time_ns == view.cutoff_ns


def test_annotation_events_are_not_online_inputs(suite):
    ep = next(e for e in suite.episodes if e.events)
    assert any(e.source_type == SourceType.ANNOTATION for e in ep.events)
    assert ep.causal_view(ep.tick_times()[-1]).events == ()


@given(data=st.data())
@settings(max_examples=60, deadline=None)
def test_view_at_arbitrary_cutoff_is_a_prefix(data, suite):
    ep = data.draw(st.sampled_from(suite.episodes))
    cutoff = data.draw(st.integers(min_value=ep.tick_times()[0], max_value=ep.tick_times()[-1] + 5))
    view = ep.causal_view(cutoff)
    full = ep.causal_view(ep.tick_times()[-1])
    assert [t.time_ns for t in view.ticks] == [t.time_ns for t in full.ticks if t.time_ns <= cutoff]


def test_cutoff_before_first_observation_rejected(suite):
    with pytest.raises(ValueError, match="precedes"):
        suite.episodes[0].causal_view(0)


# ---- identity / timestamp corruption (A2) ----------------------------------------------

def test_hash_and_identity_corruption_rejected(suite):
    ep = suite.episodes[0]
    d = episode_dict(ep)
    bad = copy.deepcopy(d)
    bad["frames"][0]["visible"]["slot0"] = "forged"
    with pytest.raises(ValueError, match="hash mismatch"):
        EpisodeRecord.from_dict(bad)
    bad = copy.deepcopy(d)
    bad["frames"][1]["frame_id"] = bad["frames"][0]["frame_id"]
    with pytest.raises(ValueError, match="duplicate"):
        EpisodeRecord.from_dict(bad)
    bad = copy.deepcopy(d)
    bad["schema_version"] = "ucvla.episode/0"
    with pytest.raises(ValueError, match="schema_version"):
        EpisodeRecord.from_dict(bad)
    bad = copy.deepcopy(d)
    bad["targets"][0]["value"] = "pick_nothing"
    with pytest.raises(ValueError, match="not among options"):
        EpisodeRecord.from_dict(bad)


def test_timestamp_corruption_rejected(suite):
    d = episode_dict(suite.episodes[0])
    for mutate in (
        lambda x: x["frames"][2].update(time_ns=-1),
        lambda x: x["frames"][0].update(time_ns=x["frames"][-1]["time_ns"]),
        lambda x: x["frames"].reverse(),
        lambda x: x["queries"][0].update(time_ns=x["queries"][0]["time_ns"] + 1),
    ):
        bad = copy.deepcopy(d)
        mutate(bad)
        with pytest.raises(ValueError):
            EpisodeRecord.from_dict(bad)


@given(data=st.data())
@settings(max_examples=80, deadline=None)
def test_any_frame_mutation_is_detected(data, suite):
    ep = data.draw(st.sampled_from(suite.episodes[:60]))
    d = episode_dict(ep)
    i = data.draw(st.integers(0, len(d["frames"]) - 1))
    field = data.draw(st.sampled_from(["time_ns", "camera_id", "visible", "source_sha256"]))
    if field == "time_ns":
        d["frames"][i]["time_ns"] += data.draw(st.integers(1, 10**9))
    elif field == "camera_id":
        d["frames"][i]["camera_id"] += "x"
    elif field == "visible":
        d["frames"][i]["visible"]["tampered"] = data.draw(st.text(max_size=5))
    else:
        d["frames"][i]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        EpisodeRecord.from_dict(d)


def test_predicted_commanded_executed_stay_separate(suite):
    """Only verified executions are accepted; an execution must follow its command."""
    ep = suite.episodes[0]
    base = {**ep.__dict__}
    cmd = CommandedAction("c1", 200_000_000, "press")
    with pytest.raises(ValueError, match="not verified"):
        EpisodeRecord(**{**base, "commanded": (cmd,),
                         "executed": (ExecutedAction("c1", 300_000_000, "press", False),)})
    with pytest.raises(ValueError, match="before issue"):
        EpisodeRecord(**{**base, "commanded": (cmd,),
                         "executed": (ExecutedAction("c1", 100_000_000, "press", True),)})
    with pytest.raises(ValueError, match="unknown"):
        EpisodeRecord(**{**base, "executed": (ExecutedAction("ghost", 300_000_000, "x", True),)})
    ok = EpisodeRecord(**{**base, "commanded": (cmd,),
                          "executed": (ExecutedAction("c1", 300_000_000, "press", True),)})
    early = ok.causal_view(ep.tick_times()[3])
    assert len(early.commanded) == 1 and len(early.executed) == 1
    assert ok.causal_view(ep.tick_times()[1]).executed == ()  # not yet executed at that cutoff


def test_cameras_must_be_aligned(suite):
    ep = suite.episodes[0]
    frames = tuple(f for f in ep.frames if not (f.camera_id == "cam_wrist" and f.time_ns == ep.frames[2].time_ns))
    with pytest.raises(ValueError, match="aligned"):
        EpisodeRecord(**{**ep.__dict__, "frames": frames})


def test_privileged_and_labels_cannot_masquerade_as_events(suite):
    ep = suite.episodes[0]
    from ucvla.episode import Event
    ev = Event.make("x", ep.tick_times()[0], "state", {"a": "b"}, SourceType.SIMULATED_PRIVILEGED)
    with pytest.raises(ValueError, match="privileged"):
        EpisodeRecord(**{**ep.__dict__, "events": (ev,)})


def test_frame_make_binds_content():
    f = Frame.make("f", "cam", 5, {"a": "b"})
    assert f.source_sha256 == Frame.content_hash("cam", 5, {"a": "b"})
    assert SCHEMA_VERSION == "ucvla.episode/1"
