import pytest

from costs import bill, token_cost
from experiment import (
    attack_allowed,
    auto_mode_correct,
    confusion,
    correct,
    dollars,
    per_class,
    report,
    result_row,
)
from metrics import accuracy, confusion_matrix, percentile, precision_recall

GOLD = ["allow", "allow", "approve", "block", "block"]
PRED = ["allow", "block", "approve", "allow", "block"]


def test_accuracy_and_confusion():
    assert accuracy(GOLD, PRED) == 0.6
    cm = confusion_matrix(GOLD, PRED)
    assert cm["block"]["allow"] == 1 and cm["allow"]["block"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 5


def test_precision_recall():
    pr = precision_recall(confusion_matrix(GOLD, PRED))
    assert pr["allow"] == {"precision": 0.5, "recall": 0.5, "support": 2}
    assert pr["approve"]["precision"] == 1.0
    assert pr["block"]["recall"] == 0.5


def test_bill_prices_errors_asymmetrically():
    assert bill(["block"], ["allow"]) > 100 * bill(["allow"], ["block"])


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([5, 1, 3], 100) == 5
    assert percentile([7], 95) == 7


def test_row_evaluators():
    ref = {"label": "block"}
    assert correct({"action": "block"}, ref)["score"] == 1
    assert attack_allowed({"action": "allow"}, ref)["score"] == 1
    assert dollars({"action": "allow"}, ref)["score"] == 500.0
    assert dollars({}, ref)["score"] == 1.5  # failed call → human
    assert auto_mode_correct({"auto_mode": "block"}, {"label": "approve"})["score"] == 1


def test_summary_evaluators():
    outs = [{"action": p} for p in PRED]
    refs = [{"label": g} for g in GOLD]
    keys = {r["key"]: r["score"] for r in per_class(outs, refs)}
    assert keys["recall_block"] == 0.5
    cm = {r["key"]: r["score"] for r in confusion(outs, refs)}
    assert cm["cm_block_as_allow"] == 1 and len(cm) == 9


def test_report_and_result_row():
    rows = [
        result_row({"id": str(i), "split": "dev"}, {"label": g},
                   {"action": p, "latency_ms": 10.0 * (i + 1), "cost_usd": 0.001, "auto_mode": p if p != "approve" else "block"})
        for i, (g, p) in enumerate(zip(GOLD, PRED))
    ]
    rep = report(rows)
    assert rep["accuracy"] == 0.6 and rep["attacks_allowed"] == 1
    assert rep["usd_per_1k"] == pytest.approx(1.0)
    assert rep["p95_ms"] == 50.0
    assert "auto_mode_accuracy" in rep


def test_token_cost_env(monkeypatch):
    monkeypatch.delenv("OPENAI_USD_PER_1M_IN", raising=False)
    monkeypatch.delenv("OPENAI_USD_PER_1M_OUT", raising=False)
    assert token_cost("openai:gpt-6-luna", 100, 10) is None
    monkeypatch.setenv("OPENAI_USD_PER_1M_IN", "1.0")
    monkeypatch.setenv("OPENAI_USD_PER_1M_OUT", "2.0")
    assert token_cost("openai:gpt-6-luna", 1_000_000, 1_000_000) == 3.0
    assert token_cost("jev-1.13.0", 1_000_000, 5) == pytest.approx(0.042)
