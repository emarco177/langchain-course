"""04 · State is more than a string: messages, objects, and batches.

So far the state we classified was a plain string. Jev's state can also be a JSON
object or array, and — because this is the LangChain integration — LangChain
`BaseMessage` objects can appear anywhere inside it. The classifier serializes
messages to role/content JSON before sending, preserving the surrounding
structure.

Why this matters: an agent's state IS a list of messages. Being able to point a
classifier at a conversation (plus any supporting facts, like a policy document)
is the bridge between "Jev classifies text" and "Jev makes decisions inside an
agent loop", which is where examples 05 and 06 go.

Two guidelines from TypeSafe's docs on structuring state:
  * Use an OBJECT for most requests so each part has a descriptive key and the
    relationships are clear (`{"conversation": ..., "refund_policy": ...}`).
  * Keep facts in state and judgments in questions. Reference state keys in
    your instructions by name.

Finally, `TypeSafeClassifier` is a Runnable, so `.batch()` (and `.ainvoke()`,
`.abatch()`) work out of the box — handy for scoring a backlog of tickets.

Needs only TYPESAFE_API_KEY.
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_typesafe import Choice, Noul, TypeSafeClassifier

load_dotenv()

# --- 1. A conversation plus a policy, as one structured state -----------------

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

refund_classifier = TypeSafeClassifier(
    questions={
        "eligible": Noul(
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
)

response = refund_classifier.invoke(
    {"conversation": conversation, "refund_policy": REFUND_POLICY}
)

print("Structured state (messages + policy)")
print(f"  eligible:   {response.nouls['eligible'].noul:.3f}")
step = response.choices["next_step"]
print(f"  next_step:  {step.choice}  (confidence {step.confidence:.3f})")
print()

# --- 2. Batch: score a backlog in one go --------------------------------------

tone_classifier = TypeSafeClassifier(
    questions={
        "churn_risk": Noul(
            instructions="Does the customer sound likely to cancel their subscription?"
        ),
    }
)

backlog = [
    "Love the new dashboard, great work!",
    "Third outage this month. Evaluating alternatives.",
    "How do I export my data to CSV?",
    "Cancel my account immediately and stop charging me.",
]

results = tone_classifier.batch(backlog)

print("Batch: churn risk per message")
for message, result in zip(backlog, results):
    print(f"  {result.nouls['churn_risk'].noul:.3f}  {message}")
