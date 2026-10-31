"""04 · Calibration: reliability diagram, ECE and Brier score for Jev's numbers.

Two numbers to check:
- Noul `is_risky`: when Jev says p = 0.8, are 80% of those calls actually risky?
- Choice `confidence`: does higher confidence mean the chosen action is right more often?

Runs offline on the results 03 saved. Writes results/reliability.png.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import load_results
from gate import JEV_MODEL
from gold import RESULTS_DIR
from metrics import brier, brier_multiclass, ece, reliability_bins

parser = argparse.ArgumentParser()
parser.add_argument("--model", default=JEV_MODEL)
parser.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
args = parser.parse_args()

rows = load_results(args.model, args.split)

# Noul: probability vs "should not run unasked" (approve or block).
p_risky = [r["p_risky"] for r in rows]
is_risky = [int(r["label"] != "allow") for r in rows]

# Choice: confidence vs "the chosen action was right".
confidence = [r["confidence"] for r in rows]
was_right = [int(r["action"] == r["label"]) for r in rows]

print(f"{args.model} on {args.split} ({len(rows)} calls)\n")
print(f"Noul is_risky       ECE {ece(p_risky, is_risky):.3f}   Brier {brier(p_risky, is_risky):.3f}")
print(f"Choice confidence   ECE {ece(confidence, was_right):.3f}   Brier {brier(confidence, was_right):.3f}")
print(f"Choice distribution Brier (3-class) "
      f"{brier_multiclass([r['probabilities'] for r in rows], [r['label'] for r in rows]):.3f}")
print("\nconfidence is not the top label's probability, so read its curve as")
print("'does accuracy rise with confidence?', not 'is it on the diagonal?'.\n")

INK, MUTED, SURFACE = "#2b2b29", "#8a8a85", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
panels = [
    ("Noul: P(risky) vs actually risky", p_risky, is_risky, "said P(risky)", "fraction risky"),
    ("Choice: confidence vs correct", confidence, was_right, "confidence", "fraction correct"),
]
for ax, color, (title, probs, outcomes, xlabel, ylabel) in zip(axes, SERIES, panels):
    bins = reliability_bins(probs, outcomes)
    print(title)
    for said, seen, n in bins:
        print(f"  said {said:.2f}  saw {seen:.2f}  n={n}")
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfectly calibrated")
    ax.plot([b[0] for b in bins], [b[1] for b in bins], color=color, lw=2, marker="o",
            ms=8, mec=SURFACE, mew=2, label=f"{args.model} (ECE {ece(probs, outcomes):.3f})")
    for said, seen, n in bins:
        ax.annotate(f"n={n}", (said, seen), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=MUTED)
    ax.set(xlim=(0, 1), ylim=(0, 1.1), xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(color="#e6e6e3", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.tight_layout()
out = RESULTS_DIR / "reliability.png"
fig.savefig(out, dpi=150)
print(f"\nsaved {out.relative_to(RESULTS_DIR.parent)}")
