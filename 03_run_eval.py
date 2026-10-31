"""03 · Run the eval: a LangSmith experiment over the Jev-backed gate.

Accuracy, per-class precision/recall, confusion matrix, latency and cost per call.
Needs TYPESAFE_API_KEY and LANGSMITH_API_KEY. Run 02 first.

    uv run python 03_run_eval.py              # dev split (tune on this)
    uv run python 03_run_eval.py --split heldout
"""

import argparse

from dotenv import load_dotenv

from experiment import JEV_EVALUATORS, format_report, report, run_experiment
from gate import JEV_MODEL, jev_gate
from gold import require_keys
from metrics import confusion_matrix, format_confusion, precision_recall

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
split = parser.parse_args().split

require_keys("TYPESAFE_API_KEY", "LANGSMITH_API_KEY")

rows = run_experiment(
    jev_gate(JEV_MODEL),
    name=JEV_MODEL,
    split=split,
    evaluators=JEV_EVALUATORS,
    metadata={"model": JEV_MODEL},
)

print()
print(format_report(JEV_MODEL, report(rows)))

cm = confusion_matrix([r["label"] for r in rows], [r["action"] for r in rows])
print("\n" + format_confusion(cm) + "\n")
for label, s in precision_recall(cm).items():
    print(f"{label:<8} precision {s['precision']:.2f}  recall {s['recall']:.2f}  (n={s['support']})")

# Where it goes wrong matters more than how often.
for r in rows:
    if r["label"] == "block" and r["action"] == "allow":
        print(f"  ATTACK ALLOWED  {r['id']}  conf={r['confidence']:.2f}  p_risky={r['p_risky']:.2f}")
