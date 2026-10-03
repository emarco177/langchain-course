"""02 · Noul, Choice, Score: three typed questions, answered in one parallel call.

Needs TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_typesafe import Choice, Noul, NoulCriteria, Score, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier()

QUESTIONS = {
    # Choice: unordered labels. "other" is the escape hatch.
    "department": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, invoices, refunds, or subscription changes.",
            "technical": "Bugs, errors, outages, or integration failures.",
            "sales": "Pricing, plans, or purchasing questions.",
            "other": "None of the above.",
        },
    ),
    # Noul: yes/no. `criteria` sharpens what yes and no mean.
    "urgent": Noul(
        instructions="Does this ticket require a response within the hour?",
        criteria=NoulCriteria(
            true="The customer is blocked or losing money right now.",
            false="The issue is annoying but the customer can keep working.",
        ),
    ),
    # Score: ordered levels, numbered from 0.
    "frustration": Score(
        instructions="How frustrated does the customer appear?",
        criteria=[
            "Calm and neutral.",
            "Concerned but civil.",
            "Angry or using strong language.",
        ],
    ),
}

TICKET = (
    "Stripe has failed to connect for three days and I can't take payments. "
    "I've emailed twice with no answer. Fix it now or I'm cancelling."
)

response = classifier.invoke({"state": TICKET, "questions": QUESTIONS})

department = response.choices["department"]
urgent = response.nouls["urgent"]
frustration = response.scores["frustration"]


def dist(probs: dict) -> str:
    return "  ".join(f"{k}={v:.2f}" for k, v in probs.items())


print(f"ticket: {TICKET!r}\n")
print(f"Choice  department   {department.choice}  conf={department.confidence:.2f}")
print(f"        {dist(department.probabilities)}")
print(f"Noul    urgent       {urgent.noul:.2f}  (P(yes))")
print(f"Score   frustration  {frustration.score:.2f}  conf={frustration.confidence:.2f}")
print(f"        {dist(frustration.probabilities)}  (levels 0-2)")
