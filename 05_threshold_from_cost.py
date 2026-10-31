"""05 · Thresholds from error cost: auto-act, escalate to a bigger LLM, or send to a human.

Start at TypeSafe's Noul starting points (0.3 / 0.7), price every error, sweep, and let the data move
you. Tune on dev, then check the chosen thresholds once on heldout.

Runs offline on the results 03 saved (run 03 with --split dev and --split heldout).
Pass --llm-results <name> after 06 to price the LLM lane with that model's measured decisions.
Writes results/coverage_vs_error.png.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from costs import COST, HUMAN_REVIEW
from experiment import load_results, results_path
from gate import JEV_MODEL
from gold import RESULTS_DIR
from lanes import (
    ASSUMED_LLM_ACCURACY,
    AUTO_MODE_TODAY,
    ESCALATE_CALL_USD,
    Lanes,
    coverage_curve,
    evaluate_lanes,
    sweep,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=JEV_MODEL)
parser.add_argument("--llm-results", help="e.g. anthropic_claude-haiku-5-5 (from 06)")
args = parser.parse_args()

dev = load_results(args.model, "dev")
llm = None
if args.llm_results:
    llm = {r["id"]: r for r in load_results(args.llm_results, "dev", hint="06_jev_vs_llms.py")}

print("what an outcome costs (gold → action):")
for gold, row in COST.items():
    print(f"  {gold:<8} " + "  ".join(f"{a} ${c:g}" for a, c in row.items()))
print(f"  human review ${HUMAN_REVIEW:g} · LLM lane "
      + (f"measured ({args.llm_results})" if llm else
         f"assumed ${ESCALATE_CALL_USD:g}/call, {ASSUMED_LLM_ACCURACY:.0%} right (measure it in 06)") + "\n")


def show(name: str, lanes: Lanes, rows: list[dict]) -> dict:
    res = evaluate_lanes(rows, lanes, llm)
    print(f"{name:<22} ${res['cost']:>8.2f}  Jev alone {res['coverage']:>4.0%}  "
          f"its errors {res['auto_error_rate']:>4.0%}  attacks allowed {res['attacks_allowed']}  "
          f"lanes {res['lanes']}")
    print(f"{'':<22} {lanes}")
    return res


print(f"dev ({len(dev)} calls)")
show("AutoMode today", AUTO_MODE_TODAY, dev)
start = show("starting points", Lanes(), dev)
best_lanes, _ = sweep(dev, llm=llm)[0]
best = show("cheapest on dev", best_lanes, dev)

if results_path(args.model, "heldout").exists():
    heldout = load_results(args.model, "heldout")
    print(f"\nheldout ({len(heldout)} calls): same thresholds, no re-tuning")
    show("starting points", Lanes(), heldout)
    show("cheapest on dev", best_lanes, heldout)
else:
    print("\n(no heldout results yet: run 03 with --split heldout to check for overfitting)")

# Coverage vs error: how much can Jev decide alone, and how often is it wrong when it does?
INK, MUTED, SURFACE = "#2b2b29", "#8a8a85", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"

curve = coverage_curve(dev)
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)

left.plot([c for _, c, _ in curve], [e for _, _, e in curve], color=BLUE, lw=2, label="symmetric band around 0.5")
for name, res, color in [("0.3 / 0.7", start, MUTED), ("cheapest", best, ORANGE)]:
    left.plot(res["coverage"], res["auto_error_rate"], "o", ms=9, color=color, mec=SURFACE, mew=2)
    left.annotate(name, (res["coverage"], res["auto_error_rate"]), xytext=(8, 6),
                  textcoords="offset points", fontsize=9, color=INK)
left.set(xlabel="coverage (share Jev decides alone)", ylabel="error rate when it does", xlim=(0, 1.02))
left.set_title("Coverage vs error", loc="left", color=INK, fontsize=11)

# Cost as allow_below moves, the other cuts at their cheapest setting.
xs = [round(i * 0.025, 3) for i in range(21)]  # 0 … 0.5
ys = [
    evaluate_lanes(dev, Lanes(x, best_lanes.block_above, max(x, best_lanes.human_above)), llm)["cost"]
    for x in xs
]
right.plot(xs, ys, color=BLUE, lw=2)
right.axvline(0.3, color=MUTED, ls="--", lw=1)
right.annotate("starting point 0.3", (0.3, max(ys)), xytext=(4, -12), textcoords="offset points", fontsize=9, color=MUTED)
right.plot(best_lanes.allow_below, best["cost"], "o", ms=9, color=ORANGE, mec=SURFACE, mew=2)
right.set(xlabel="allow_below (auto-allow if P(risky) ≤ this)", ylabel="total $ on dev")
right.set_title("One wrong allow outweighs hundreds of reviews", loc="left", color=INK, fontsize=11)

for ax in (left, right):
    ax.set_facecolor(SURFACE)
    ax.grid(color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
left.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = RESULTS_DIR / "coverage_vs_error.png"
fig.savefig(out, dpi=150)
print(f"\nsaved {out.relative_to(RESULTS_DIR.parent)}")
