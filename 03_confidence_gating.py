"""03 · Probability ≠ confidence. Gate each action on confidence.

Needs `ollama pull tev1`. Local tev1 gives the same routing every run.
"""

from dotenv import load_dotenv
from langchain_typesafe import Choice, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier(
    base_url="http://localhost:11434", model="tev1", api_key="ollama"
)

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, invoices, refunds, or subscription changes.",
            "technical": "Bugs, errors, outages, or integration failures.",
            "sales": "Pricing, plans, or purchasing questions.",
            "other": "None of the above.",
        },
    ),
}

# Thresholds belong to the action, and are tuned per model.
AUTO_ROUTE = 0.75
ASK_USER = 0.40

TICKETS = [
    "Every API call has returned 502 since this morning. Our checkout is down.",
    "Can I switch to annual billing?",
    "My invoice looks wrong and the export button errors out too. "
    "Also, do you offer annual pricing?",
]

# One call per ticket, run in parallel.
results = classifier.batch([{"state": t, "questions": QUESTIONS} for t in TICKETS])

for ticket, result in zip(TICKETS, results):
    answer = result.choices["department"]
    if answer.confidence >= AUTO_ROUTE:
        action = f"auto-route to {answer.choice}"
    elif answer.confidence >= ASK_USER:
        action = f"ask: is this {answer.choice}?"
    else:
        action = "escalate to a human"

    ranked = sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
    print(ticket)
    print("  " + "  ".join(f"{label} {p:.2f}" for label, p in ranked))
    print(f"  confidence {answer.confidence:.2f} → {action}\n")
