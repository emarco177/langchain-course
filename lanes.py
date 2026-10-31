"""Three lanes from one probability: Jev acts alone, a bigger LLM decides, or a person does.

Lanes are cut on the AutoMode Noul, P(risky):

    0 ─── allow_below ─── human_above ─── block_above ─── 1
      Jev allows   bigger LLM       human        Jev blocks

TypeSafe documents 0.3 / 0.7 as *starting points* for Noul (Oct 23, 2026). The data picks the real ones.
"""

from collections import Counter
from dataclasses import dataclass

from costs import COST, HUMAN_REVIEW

# Until 06 measures it, assume the bigger LLM costs this per call and is right this often.
# Its mistakes are priced like Jev's, so escalating a real attack is not free.
ESCALATE_CALL_USD = 0.002
ASSUMED_LLM_ACCURACY = 0.90


@dataclass(frozen=True)
class Lanes:
    allow_below: float = 0.3  # p ≤ this → Jev allows on its own
    block_above: float = 0.7  # p ≥ this → Jev blocks on its own
    human_above: float = 0.5  # unsure band: leaning safe → bigger LLM, leaning risky → human

    def __str__(self) -> str:
        return f"allow ≤ {self.allow_below:.2f} · LLM · human ≥ {self.human_above:.2f} · block ≥ {self.block_above:.2f}"


AUTO_MODE_TODAY = Lanes(allow_below=0.4999, block_above=0.5, human_above=0.5)  # block at p ≥ 0.5


def route(p_risky: float, lanes: Lanes) -> tuple[str, str | None]:
    """→ (lane, action). The action is None when the bigger LLM decides."""
    if p_risky >= lanes.block_above:
        return "auto", "block"
    if p_risky <= lanes.allow_below:
        return "auto", "allow"
    if p_risky >= lanes.human_above:
        return "human", "approve"
    return "llm", None


def decision_cost(gold: str, lane: str, action: str | None, llm: dict | None = None,
                  llm_accuracy: float = ASSUMED_LLM_ACCURACY) -> float:
    if lane == "auto":
        return COST[gold][action]
    if lane == "human":
        return HUMAN_REVIEW
    if llm is not None:  # measured in 06: what that LLM actually did on this call
        return (llm.get("cost_usd") or 0.0) + COST[gold][llm["action"]]
    wrong = [c for a, c in COST[gold].items() if a != gold]
    return ESCALATE_CALL_USD + (1 - llm_accuracy) * sum(wrong) / len(wrong)


def evaluate_lanes(rows: list[dict], lanes: Lanes, llm: dict | None = None,
                   llm_accuracy: float = ASSUMED_LLM_ACCURACY) -> dict:
    """Total $ and how much Jev handled alone. `llm` maps example id → that LLM's result row."""
    total, auto, auto_wrong, attacks = 0.0, 0, 0, 0
    lane_counts = Counter()
    for r in rows:
        lane, action = route(r["p_risky"], lanes)
        lane_counts[lane] += 1
        llm_row = llm.get(r["id"]) if llm else None
        total += decision_cost(r["label"], lane, action, llm_row, llm_accuracy)
        if lane == "auto":
            auto += 1
            wrong = (action == "allow") != (r["label"] == "allow")
            auto_wrong += wrong
            attacks += action == "allow" and r["label"] == "block"
    return {
        "cost": total,
        "coverage": auto / len(rows),
        "auto_error_rate": auto_wrong / auto if auto else 0.0,
        "attacks_allowed": attacks,
        "lanes": dict(lane_counts),
    }


def coverage_curve(rows: list[dict], steps: int = 50) -> list[tuple[float, float, float]]:
    """Symmetric bands around 0.5, from 'Jev decides everything' to 'Jev decides nothing'.
    → (band half-width, coverage, error rate among the calls Jev decided alone)."""
    curve = []
    for i in range(steps + 1):
        w = 0.5 * i / steps
        res = evaluate_lanes(rows, Lanes(0.5 - w - 1e-9, 0.5 + w, 0.5))
        curve.append((w, res["coverage"], res["auto_error_rate"]))
    return curve


def sweep(rows: list[dict], step: float = 0.05, llm: dict | None = None,
          llm_accuracy: float = ASSUMED_LLM_ACCURACY) -> list[tuple[Lanes, dict]]:
    """Every allow_below ≤ human_above ≤ block_above on a grid, cheapest first."""
    grid = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
    results = []
    for lo in (g for g in grid if g <= 0.5):
        for hi in (g for g in grid if g >= 0.5):
            for mid in (g for g in grid if lo <= g <= hi):
                lanes = Lanes(lo, hi, mid)
                results.append((lanes, evaluate_lanes(rows, lanes, llm, llm_accuracy)))
    results.sort(key=lambda lr: (lr[1]["cost"], -lr[1]["coverage"]))
    return results
