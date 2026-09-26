"""04 · State can be JSON with LangChain messages inside, and .batch() works.

Needs TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice, Noul, TypeSafeClassifier

load_dotenv()

classifier = TypeSafeClassifier()

# --- 1. Messages + policy as one state -------------------------------------

REFUND_POLICY = (
    "Full refund within 14 days of purchase for any reason. After 14 days, "
    "refunds only for verified service outages longer than 24 hours."
)

conversation = [
    SystemMessage(content="You are a support agent for a SaaS product."),
    HumanMessage(content="Hi, I bought the Pro plan 3 weeks ago."),
    AIMessage(content="Thanks for being a customer! How can I help?"),
    HumanMessage(
        content=(
            "The service was completely down for two full days last week and "
            "I couldn't work. I'd like a refund."
        )
    ),
]

REFUND_QUESTIONS = {
    "eligible": Noul(
        # Reference state keys by name.
        instructions=(
            "Based on `conversation`, is the customer eligible for a refund "
            "under `refund_policy`?"
        )
    ),
    "next_step": Choice(
        instructions="What should the agent do next, given `conversation`?",
        criteria={
            "issue_refund": "The request clearly qualifies; process it.",
            "verify_outage": "Check outage records before deciding.",
            "decline": "The request clearly does not qualify.",
            "ask_for_details": "Key facts (dates, order id) are missing.",
        },
    ),
}

response = classifier.invoke(
    {
        "state": {"conversation": conversation, "refund_policy": REFUND_POLICY},
        "questions": REFUND_QUESTIONS,
    }
)

print("Structured state (messages + policy)")
print(f"  eligible:   {response.nouls['eligible'].noul:.3f}")
step = response.choices["next_step"]
print(f"  next_step:  {step.choice}  (confidence {step.confidence:.3f})")
print()

# --- 2. Batch a backlog ---------------------------------------------------

CHURN_QUESTIONS = {
    "churn_risk": Noul(
        instructions="Does the customer sound likely to cancel their subscription?"
    ),
}

backlog = [
    "Love the new dashboard, great work!",
    "Third outage this month. Evaluating alternatives.",
    "How do I export my data to CSV?",
    "Cancel my account immediately and stop charging me.",
]

results = classifier.batch(
    [{"state": m, "questions": CHURN_QUESTIONS} for m in backlog]
)

print("Batch: churn risk per message")
for message, result in zip(backlog, results):
    print(f"  {result.nouls['churn_risk'].noul:.3f}  {message}")
