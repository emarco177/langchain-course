import pytest

from costs import COST, HUMAN_REVIEW
from lanes import AUTO_MODE_TODAY, Lanes, coverage_curve, decision_cost, evaluate_lanes, route, sweep


def rows(*pairs):
    return [{"id": str(i), "label": g, "p_risky": p} for i, (g, p) in enumerate(pairs)]


def test_route_lanes():
    lanes = Lanes(0.3, 0.7, 0.5)
    assert route(0.1, lanes) == ("auto", "allow")
    assert route(0.3, lanes) == ("auto", "allow")
    assert route(0.4, lanes) == ("llm", None)
    assert route(0.6, lanes) == ("human", "approve")
    assert route(0.7, lanes) == ("auto", "block")


def test_auto_mode_today_blocks_at_half():
    assert route(0.5, AUTO_MODE_TODAY) == ("auto", "block")
    assert route(0.49, AUTO_MODE_TODAY) == ("auto", "allow")


def test_decision_cost():
    assert decision_cost("block", "auto", "allow") == COST["block"]["allow"]
    assert decision_cost("allow", "human", "approve") == HUMAN_REVIEW
    # Unmeasured LLM: call cost plus its assumed mistakes, priced from the cost table.
    assert decision_cost("allow", "llm", None, llm_accuracy=1.0) == pytest.approx(0.002)
    assert decision_cost("block", "llm", None, llm_accuracy=0.9) > decision_cost("allow", "llm", None, llm_accuracy=0.9)
    assert decision_cost("block", "llm", None, llm={"action": "block", "cost_usd": 0.001}) == 0.001


def test_evaluate_lanes_counts_attacks_and_coverage():
    data = rows(("allow", 0.1), ("block", 0.2), ("block", 0.9), ("allow", 0.45))
    res = evaluate_lanes(data, Lanes(0.3, 0.7, 0.5), llm={"3": {"action": "allow", "cost_usd": 0.0}})
    assert res["coverage"] == 0.75
    assert res["attacks_allowed"] == 1
    assert res["auto_error_rate"] == pytest.approx(1 / 3)
    assert res["cost"] == COST["block"]["allow"]
    assert res["lanes"] == {"auto": 3, "llm": 1}


def test_coverage_curve_runs_from_all_to_none():
    data = rows(("allow", 0.1), ("block", 0.9), ("approve", 0.55))
    curve = coverage_curve(data, steps=10)
    assert curve[0][1] == 1.0
    assert all(a[1] >= b[1] for a, b in zip(curve, curve[1:]))  # wider band → less coverage


def test_sweep_moves_off_the_starting_point_when_an_attack_sits_at_0_25():
    # A $500 attack hides at p=0.25, so 0.3 lets it through. The cheapest setting allows only below it.
    data = rows(*([("allow", 0.05)] * 20), ("block", 0.25), ("block", 0.9))
    start = evaluate_lanes(data, Lanes())
    best_lanes, best = sweep(data)[0]
    assert start["attacks_allowed"] == 1
    assert best["attacks_allowed"] == 0
    assert best_lanes.allow_below < 0.25
    assert best["cost"] < start["cost"]
