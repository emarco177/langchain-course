from lanes import Lanes
from metrics import diff_results


def row(id, label, action, p):
    return {"id": id, "label": label, "action": action, "p_risky": p}


OLD = [
    row("a", "allow", "allow", 0.10),
    row("b", "block", "block", 0.90),
    row("c", "approve", "allow", 0.40),
    row("d", "block", "approve", 0.60),
    row("gone", "allow", "allow", 0.1),
]
NEW = [
    row("a", "allow", "allow", 0.12),     # same
    row("b", "block", "allow", 0.25),     # broke, and it's an attack
    row("c", "approve", "approve", 0.55), # fixed
    row("d", "block", "allow", 0.62),     # changed (wrong → wrong), also an attack
    row("added", "allow", "allow", 0.1),
]


def test_diff_classifies_flips():
    d = diff_results(OLD, NEW)
    assert [x[0] for x in d["broke"]] == ["b"]
    assert [x[0] for x in d["fixed"]] == ["c"]
    assert [x[0] for x in d["changed"]] == ["d"]
    assert d["attacks_newly_allowed"] == ["b", "d"]
    assert d["only_in_one"] == ["added", "gone"]


def test_diff_probability_shifts_and_lane_moves():
    d = diff_results(OLD, NEW, p_shift=0.1, lanes=Lanes(0.3, 0.7, 0.5))
    assert {x[0] for x in d["p_shifts"]} == {"b", "c"}
    assert ("b", "jev-block", "jev-allow") in d["lane_moves"]  # the scary one
    assert ("c", "llm", "human") in d["lane_moves"]
    assert not any(m[0] == "a" for m in d["lane_moves"])


def test_identical_runs_have_no_diff():
    d = diff_results(OLD, OLD, lanes=Lanes())
    assert not any(d[k] for k in ("fixed", "broke", "changed", "p_shifts", "lane_moves", "only_in_one"))
