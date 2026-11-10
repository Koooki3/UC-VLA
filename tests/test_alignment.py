import pytest
from hypothesis import given
from hypothesis import strategies as st

from ucvla.alignment import AlignmentError, ClockMap, causal_ticks, merge_streams

times = st.lists(st.integers(0, 10**12), min_size=2, max_size=30).map(sorted)


@given(times, st.integers(-5000, 5000), st.integers(0, 10**6))
def test_clock_map_is_monotone(ts, drift, offset):
    cm = ClockMap("d", offset_ns=offset, drift_ppm=drift, ref_ns=ts[0])
    out = [cm.to_canonical(t) for t in ts]
    assert out == sorted(out)


def test_clock_map_rejects_bad_inputs():
    with pytest.raises(AlignmentError):
        ClockMap("d", drift_ppm=-1_000_000)
    with pytest.raises(AlignmentError):
        ClockMap("d").to_canonical(1.5)
    with pytest.raises(AlignmentError):
        ClockMap("d", offset_ns=-10).to_canonical(5)


def test_unknown_clock_domain_never_assumed():
    with pytest.raises(AlignmentError, match="unknown clock domain"):
        merge_streams({"cam": [(1, "mystery", "x")]}, {"sim": ClockMap("sim")})


def test_merge_is_order_independent_and_deterministic():
    maps = {"a": ClockMap("a"), "b": ClockMap("b", offset_ns=5)}
    s1 = {"x": [(10, "a", 1), (20, "a", 2)], "y": [(5, "b", 3), (15, "b", 4)]}
    s2 = {"y": s1["y"], "x": s1["x"]}
    assert merge_streams(s1, maps) == merge_streams(s2, maps)
    assert [t for t, _, _ in merge_streams(s1, maps)] == [10, 10, 20, 20]


def test_causal_ticks_never_use_future_samples_and_drop_stale():
    stream = [(10, "a"), (30, "b")]
    out = causal_ticks(stream, [5, 10, 20, 29, 30, 100], max_staleness_ns=15)
    assert out == [(5, None), (10, "a"), (20, "a"), (29, None), (30, "b"), (100, None)]
    with pytest.raises(AlignmentError):
        causal_ticks([(5, "x"), (1, "y")], [10], 100)
