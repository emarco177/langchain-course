"""03 · Probability ≠ confidence. Gate each action on confidence.

Needs TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_typesafe import Choice, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier()

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

# Thresholds belong to the action, not the model.
AUTO_ROUTE = 0.75
ASK_USER = 0.45


def route(ticket: str) -> None:
    response = classifier.invoke({"state": ticket, "questions": QUESTIONS})
    answer = response.choices["department"]
    top_label, top_p = max(answer.probabilities.items(), key=lambda kv: kv[1])

    print(f"ticket:      {ticket!r}")
    print(f"choice:      {answer.choice}")
    print(f"top prob:    {top_p:.3f}  ({top_label})")
    print(f"confidence:  {answer.confidence:.3f}")
    print("distribution:", {k: round(v, 3) for k, v in answer.probabilities.items()})

    if answer.confidence >= AUTO_ROUTE:
        print(f"→ auto-routed to {answer.choice}")
    elif answer.confidence >= ASK_USER:
        print(f"→ tentatively {answer.choice}; ask the customer to confirm")
    else:
        print("→ escalate to a human triager")
    print()


# Clear → auto-route
route("Every API call has returned 502 since this morning. Our checkout is down.")

# Leaning → ask the customer
route(
    "Your integration keeps failing and I'm thinking about switching plans — "
    "what are my options?"
)

# Mixed → escalate
route(
    "My invoice looks wrong and the export button errors out too. "
    "Also, do you offer annual pricing?"
)
