import pytest

from metrics import brier, brier_multiclass, ece, reliability_bins


def test_reliability_bins_group_by_said_probability():
    bins = reliability_bins([0.05, 0.15, 0.95, 1.0], [0, 0, 1, 1], n_bins=10)
    assert bins == [(0.05, 0.0, 1), (0.15, 0.0, 1), (0.975, 1.0, 2)]


def test_perfectly_calibrated_has_zero_ece():
    probs = [0.2] * 5 + [0.8] * 5
    outcomes = [1, 0, 0, 0, 0] + [1, 1, 1, 1, 0]
    assert ece(probs, outcomes) == pytest.approx(0.0)


def test_overconfident_ece_is_the_gap():
    # Says 0.9, right half the time: gap 0.4.
    assert ece([0.9] * 4, [1, 1, 0, 0]) == pytest.approx(0.4)


def test_brier():
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    assert brier([1.0], [0]) == 1.0


def test_brier_multiclass():
    perfect = {"allow": 1.0, "approve": 0.0, "block": 0.0}
    assert brier_multiclass([perfect], ["allow"]) == 0.0
    assert brier_multiclass([perfect], ["block"]) == 2.0
