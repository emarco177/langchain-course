"""Plain functions over (gold, prediction) lists. No API calls, so tests run offline."""

import math

from costs import LABELS


def accuracy(golds: list[str], preds: list[str]) -> float:
    return sum(g == p for g, p in zip(golds, preds)) / len(golds)


def confusion_matrix(golds: list[str], preds: list[str], labels=LABELS) -> dict:
    """cm[gold][pred] = count. Rows are what should have happened."""
    cm = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(golds, preds):
        cm[g][p] += 1
    return cm


def precision_recall(cm: dict) -> dict:
    """Per class: of the calls we labelled X, how many were X (precision); of the real X, how many we caught (recall)."""
    out = {}
    for label in cm:
        tp = cm[label][label]
        predicted = sum(cm[g][label] for g in cm)
        actual = sum(cm[label].values())
        out[label] = {
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "support": actual,
        }
    return out


def format_confusion(cm: dict) -> str:
    labels = list(cm)
    lines = ["gold ↓ / pred →  " + "".join(f"{p:>9}" for p in labels)]
    for g in labels:
        lines.append(f"{g:<17}" + "".join(f"{cm[g][p]:>9}" for p in labels))
    return "\n".join(lines)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile, q in [0, 100]."""
    ordered = sorted(values)
    rank = max(0, math.ceil(q / 100 * len(ordered)) - 1)
    return ordered[rank]
