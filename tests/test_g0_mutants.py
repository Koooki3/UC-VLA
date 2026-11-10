"""Mutation regression: every channel of the decision view gets a planted leak; G0 must fail
for the RIGHT reason. (Independent review 2026-11-11 showed v1 G0 passed 5 of these.)"""

import dataclasses

import pytest

from ucvla.adapters import SODHAdapter
from ucvla.audit import g0_report
from ucvla.canonical import digest
from ucvla.episode import CommandedAction, Event, Frame, ProprioSample, SourceType
from ucvla.sodh import RELEVANT_FAMILIES, generate_suite
from ucvla.splits import assign_splits


def label(e):
    return e.target_for(e.queries[0].query_id)


def opt_idx(e):
    return e.queries[0].options.index(label(e))


def rebuild(e, fn):
    ts = e.tick_times()
    frames = tuple(Frame.make(f.frame_id, f.camera_id, f.time_ns,
                              fn(ts.index(f.time_ns), len(ts), f.camera_id, dict(f.visible)))
                   for f in e.frames)
    return dataclasses.replace(e, frames=frames)


MUTANTS = {
    "A_proprio": (lambda e: dataclasses.replace(
        e, proprio=(ProprioSample(e.queries[0].time_ns, (float(opt_idx(e)),)),)), "obs_differs"),
    "B_commanded": (lambda e: dataclasses.replace(
        e, commanded=(CommandedAction(e.episode_id + "-c0", e.queries[0].time_ns, label(e)),)),
        "obs_differs"),
    "C_online_event": (lambda e: dataclasses.replace(
        e, events=(*e.events, Event.make(e.episode_id + "-n0", e.queries[0].time_ns, "note",
                                          {"v": label(e)}, SourceType.OBSERVED))), "obs_differs"),
    "D_id_suffix": (lambda e: dataclasses.replace(e, episode_id=f"{e.episode_id}-i{opt_idx(e)}"),
                    "id_leak"),
    "E_history_hint": (lambda e: rebuild(
        e, lambda i, n, cam, v: {**v, "aux": str(opt_idx(e))}
        if cam == "cam_front" and i == n - 2 else v), "token_shortcut"),
    "G_stale_pair_records": (lambda e: rebuild(
        e, lambda i, n, cam, v: {**v, "jitter": digest(label(e))[:6]}
        if cam == "cam_front" and i == n - 1 else v), "stale_pair_record"),
    "I_wrist_camera": (lambda e: rebuild(
        e, lambda i, n, cam, v: {"gripper": f"g{opt_idx(e)}"}
        if cam == "cam_wrist" and i == n - 1 else v), "obs_differs"),
}


def mutated(suite, assignment, transform):
    eps, remap = [], {}
    for e in suite.episodes:
        e2 = transform(e)
        remap[e.episode_id] = e2.episode_id
        eps.append(e2)
    pairs = [dataclasses.replace(p, episode_a=remap[p.episode_a], episode_b=remap[p.episode_b])
             for p in suite.pairs]
    return eps, pairs, SODHAdapter(eps, pairs, assignment)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_planted_leak_in_every_channel_fails_g0_for_the_right_reason(suite, assignment, name):
    transform, code = MUTANTS[name]
    eps, pairs, ad = mutated(suite, assignment, transform)
    r = g0_report(ad, pairs, eps, assignment)
    assert not r["passed"]
    assert code in {f["code"] for f in r["findings"]}, r["findings"][:5]


def test_clean_suite_still_passes(suite, assignment):
    eps, pairs, ad = mutated(suite, assignment, lambda e: e)
    assert g0_report(ad, pairs, eps, assignment)["passed"]


def test_option_order_prior_is_flagged(frozen_config):
    """Options permuted so mate-a's answer is first: an 'always options[0]' shortcut."""
    suite = generate_suite(frozen_config)
    asg = assign_splits(suite.pairs, frozen_config)
    by_id = {e.episode_id: e for e in suite.episodes}
    new = {}
    for p in suite.pairs:
        if p.family not in RELEVANT_FAMILIES or p.family == "process_count":
            continue
        a, b = by_id[p.episode_a], by_id[p.episode_b]
        opts = list(a.queries[0].options)
        reordered = (label(a), *[o for o in opts if o != label(a)])
        for e in (a, b):
            new[e.episode_id] = dataclasses.replace(
                e, queries=(dataclasses.replace(e.queries[0], options=reordered),))
    eps = [new.get(e.episode_id, e) for e in suite.episodes]
    ad = SODHAdapter(eps, suite.pairs, asg)
    r = g0_report(ad, suite.pairs, eps, asg)
    assert not r["passed"]
    assert {"label_prior", "stale_pair_record", "token_shortcut", "metadata_shortcut"} & {
        f["code"] for f in r["findings"]}


def test_constant_answer_floor_reported(adapter, suite, assignment):
    floor = g0_report(adapter, suite.pairs, suite.episodes, assignment)["constant_answer_floor"]
    assert set(floor) == {"object_correspondence", "process_count", "task_interleave",
                          "irrelevant_history"}
    assert all(0 < v <= 1 for v in floor.values())
