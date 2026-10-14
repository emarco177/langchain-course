"""03 · State can be JSON: LangChain messages and a policy, referenced by key.

Needs TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_typesafe import Choice, Noul, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier()

STATE = {
    "refund_policy": (
        "Full refund within 14 days of purchase. After 14 days, refunds only "
        "for verified outages longer than 24 hours."
    ),
    "conversation": [
        HumanMessage("I bought the Pro plan 3 weeks ago."),
        AIMessage("Thanks for being a customer! How can I help?"),
        HumanMessage("The service was down for two full days last week. I want a refund."),
    ],
}

# Reference state keys by name.
QUESTIONS = {
    "eligible": Noul(
        instructions="Is the customer eligible under `refund_policy`, per `conversation`?"
    ),
    "next_step": Choice(
        instructions="What should the agent do next, given `conversation`?",
        criteria={
            "issue_refund": "Clearly qualifies; process it.",
            "verify_outage": "Check outage records before deciding.",
            "decline": "Clearly does not qualify.",
            "ask_for_details": "Key facts (dates, order id) are missing.",
        },
    ),
}

response = classifier.invoke({"state": STATE, "questions": QUESTIONS})

# Likely eligible, but the policy needs a *verified* outage → verify first.
step = response.choices["next_step"]
print(f"eligible   {response.nouls['eligible'].noul:.2f}")
print(f"next_step  {step.choice}  (confidence {step.confidence:.2f})")
