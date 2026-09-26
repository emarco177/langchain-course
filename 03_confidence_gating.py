"""03 · Probability vs. confidence, and gating on it.

Choice and Score answers carry two different numbers, and conflating them is the
most common mistake with Jev:

  * `probabilities` — the full distribution over labels. "How likely is each?"
  * `confidence`    — ONE number in [0, 1] describing the SHAPE of that
                      distribution: concentrated on a single label → high,
                      spread across several → low.

Confidence is NOT the probability of the winning label. Two answers can pick the
same label with the same top probability and still differ in confidence, because
confidence also cares about how the remaining mass is spread.

Jev is trained (TypeSafe calls it RLCD — reinforcement learning for calibrated
decisions) so that confidence tracks accuracy: when it says 0.95 it is right
about 95% of the time. That is what makes the number *actionable*. The pattern
TypeSafe recommends is confidence-gated routing:

    high   → act automatically
    medium → act, but verify / confirm with the user
    low    → hand off to a human or a stronger system

and — important — the thresholds are not one global number. Gate each action by
the cost of being wrong about it.

Here the same classifier sees a clear ticket and a genuinely ambiguous one.

Needs only TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_typesafe import Choice, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier(
    questions={
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
)

# Thresholds belong to the ACTION, not the model. Auto-routing a ticket is cheap
# to get wrong, so we can act fairly early; escalation is reserved for real doubt.
AUTO_ROUTE = 0.75
ASK_USER = 0.45


def route(ticket: str) -> None:
    answer = classifier.invoke(ticket).choices["department"]
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


# 1. Unambiguous: an outage. Expect high confidence.
route("Every API call has returned 502 since this morning. Our checkout is down.")

# 2. Ambiguous: is this billing (a charge) or technical (a bug in the charge)?
route(
    "I was charged twice after your app crashed mid-checkout. "
    "Which is it — your bug or my bank?"
)
