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

print(f"ticket: {TICKET!r}\n")

print("Choice  · department")
print(f"  choice:        {department.choice}")
print(f"  confidence:    {department.confidence:.3f}")
print("  probabilities:")
for label, p in sorted(department.probabilities.items(), key=lambda kv: -kv[1]):
    print(f"    {label:<10} {p:.3f}")

print("\nNoul    · urgent")
print(f"  noul:          {urgent.noul:.3f}  (P(yes); Noul has no separate confidence)")

print("\nScore   · frustration")
print(f"  score:         {frustration.score:.2f}  (expected level, may be fractional)")
print(f"  confidence:    {frustration.confidence:.3f}")
print("  legend / probabilities:")
for level, label in frustration.legend.items():
    print(f"    {level}: {frustration.probabilities[level]:.3f}  {label}")

print(f"\nusage: {response.usage.input_tokens} in / {response.usage.output_tokens} out")
