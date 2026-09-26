"""02 · The three primitives: Noul, Choice, Score.

Jev answers exactly three kinds of question, and everything you build with it is
a combination of them:

  * `Noul`   — a yes/no judgment. Returns the probability that the answer is yes.
  * `Choice` — pick one label from a fixed, unordered set. Returns the chosen
               label, a probability for EVERY label, and a confidence score.
  * `Score`  — place the input on an ordered rubric (low → high). Returns the
               expected level as a float (it may fall between levels), the full
               distribution over levels, and a confidence score.

The second thing to know: every question in a request is evaluated in parallel.
Three questions cost about the same latency as one. So the idiom is to bundle
all the judgments you need about a piece of state into a single classifier and
make one call — not one call per question.

Here one classifier triages a support ticket on three axes at once.

Needs only TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_typesafe import Choice, Noul, NoulCriteria, Score, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier(
    questions={
        # Choice: unordered labels. Always include an escape hatch like "other"
        # so the model is never forced to pick a wrong label.
        "department": Choice(
            instructions="Which team should handle this ticket?",
            criteria={
                "billing": "Payments, invoices, refunds, or subscription changes.",
                "technical": "Bugs, errors, outages, or integration failures.",
                "sales": "Pricing, plans, or purchasing questions.",
                "other": "None of the above.",
            },
        ),
        # Noul: yes/no. `criteria` is optional but sharpens what yes and no mean.
        "urgent": Noul(
            instructions="Does this ticket require a response within the hour?",
            criteria=NoulCriteria(
                true="The customer is blocked or losing money right now.",
                false="The issue is annoying but the customer can keep working.",
            ),
        ),
        # Score: ORDERED levels, numbered from 0 in the order you list them.
        "frustration": Score(
            instructions="How frustrated does the customer appear?",
            criteria=[
                "Calm and neutral.",
                "Concerned but civil.",
                "Angry or using strong language.",
            ],
        ),
    }
)

TICKET = (
    "Stripe has failed to connect for three days and I can't take payments. "
    "I've emailed twice with no answer. Fix it now or I'm cancelling."
)

response = classifier.invoke(TICKET)

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
