# SPDX-License-Identifier: Apache-2.0
import copy

from scripts.aggregate_g1 import REL, apply_criteria


def ms(v):
    return {"mean": v, "lo": v - 0.01, "hi": v + 0.01, "min": v - 0.02, "max": v + 0.02}


def passing():
    fam = {}
    for f in REL:
        fam[f] = {"pair_optimal": {"test_id": ms(0.9)}, "memoryless_test_id": ms(0.0),
                  "swap_follow": ms(0.95), "drop_critical_minus_noncrit": ms(0.5),
                  "order_perturbed_pair_optimal": ms(0.3)}
    return {"fam": fam, "control_acc": ms(1.0), "control_swap_invariance": ms(1.0)}


def test_all_conditions_met_passes():
    r = apply_criteria(passing())
    assert r["g1_pass"] and all(r["conditions"].values())


def test_order_test_failure_alone_must_reject():
    a = passing()
    a["fam"]["track"]["order_perturbed_pair_optimal"] = ms(0.88)  # barely changes: model ignores order
    r = apply_criteria(a)
    assert r["g1_pass_legacy_without_order"] and not r["g1_pass"]  # the old verdict would have passed
    assert not r["conditions"]["order_perturbation_cuts_track"]


def test_each_other_condition_can_reject():
    for mut in (lambda a: a["fam"]["events"].update(memoryless_test_id=ms(0.2)),
                lambda a: a["fam"]["resume"].update(swap_follow=ms(0.3)),
                lambda a: a["fam"]["resume"]["pair_optimal"].update(test_id={**ms(0.9), "lo": 0.1}),
                lambda a: a.update(control_acc=ms(0.5)),
                lambda a: a.update(control_swap_invariance=ms(0.8))):
        a = copy.deepcopy(passing())
        # make one family fail so a second failure matters: drop track entirely
        a["fam"]["track"]["pair_optimal"]["test_id"] = ms(0.1)
        mut(a)
        assert not apply_criteria(a)["g1_pass"]


def test_seed_range_rejects():
    a = passing()
    a["fam"]["track"]["pair_optimal"]["test_id"] = {"mean": 0.9, "lo": 0.8, "hi": 1.0, "min": 0.5, "max": 1.0}
    a["fam"]["events"]["pair_optimal"]["test_id"] = {"mean": 0.9, "lo": 0.8, "hi": 1.0, "min": 0.5, "max": 1.0}
    assert not apply_criteria(a)["g1_pass"]
