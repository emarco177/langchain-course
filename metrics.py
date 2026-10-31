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


# ── Calibration: when Jev says 0.8, is it right 80% of the time? ────────────────


def reliability_bins(probs: list[float], outcomes: list[int], n_bins: int = 10) -> list[tuple]:
    """Per non-empty bin: (mean predicted probability, observed frequency, count)."""
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    return [
        (sum(p for p, _ in b) / len(b), sum(y for _, y in b) / len(b), len(b))
        for b in bins
        if b
    ]


def ece(probs: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    """Expected calibration error: the count-weighted gap between said and observed."""
    bins = reliability_bins(probs, outcomes, n_bins)
    return sum(n / len(probs) * abs(said - seen) for said, seen, n in bins)


def brier(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error of a probability. 0 is perfect; always saying 0.5 scores 0.25."""
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def brier_multiclass(dists: list[dict], golds: list[str]) -> float:
    """Brier over a whole Choice distribution. 0 is perfect; 2 is confidently wrong."""
    return sum(
        sum((p - (label == gold)) ** 2 for label, p in dist.items())
        for dist, gold in zip(dists, golds)
    ) / len(golds)


# ── Regression: what changed between two runs of the same gold set? ────────────


def diff_results(old: list[dict], new: list[dict], p_shift: float = 0.10, lanes=None) -> dict:
    """Compare two result sets example by example (old model vs new, or old wording vs new).

    fixed:   old was wrong, new is right       broke:  old was right, new is wrong
    changed: both wrong, but differently       attacks_newly_allowed: the $500 regressions
    p_shifts: |Δ P(risky)| ≥ p_shift           lane_moves: same thresholds, different lane
    """
    from lanes import route  # local import: lanes is a later lesson's module

    old_by_id = {r["id"]: r for r in old}
    out = {"fixed": [], "broke": [], "changed": [], "attacks_newly_allowed": [],
           "p_shifts": [], "lane_moves": [], "only_in_one": []}
    new_ids = {r["id"] for r in new}
    out["only_in_one"] = sorted(set(old_by_id) ^ new_ids)
    for n in new:
        o = old_by_id.get(n["id"])
        if o is None:
            continue
        gold, before, after = n["label"], o["action"], n["action"]
        if before != after:
            kind = "fixed" if after == gold else "broke" if before == gold else "changed"
            out[kind].append((n["id"], gold, before, after))
            if gold == "block" and after == "allow":
                out["attacks_newly_allowed"].append(n["id"])
        if "p_risky" in o and "p_risky" in n:
            if abs(n["p_risky"] - o["p_risky"]) >= p_shift:
                out["p_shifts"].append((n["id"], o["p_risky"], n["p_risky"]))
            if lanes is not None:
                before_lane, after_lane = _lane(route(o["p_risky"], lanes)), _lane(route(n["p_risky"], lanes))
                if before_lane != after_lane:
                    out["lane_moves"].append((n["id"], before_lane, after_lane))
    return out


def _lane(routed: tuple[str, str | None]) -> str:
    lane, action = routed
    return f"jev-{action}" if lane == "auto" else lane
